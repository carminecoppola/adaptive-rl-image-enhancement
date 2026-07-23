import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces

# Allow direct execution (python src/training/train.py) by adding
# project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions import get_stop_action_id
from src.agents import DQNAgent, ReplayBuffer
from src.data import get_dataset_name, get_effective_image_size, load_train_dataset
from src.training.dqn_artifacts import (
    build_checkpoint_payload,
    write_final_artifacts,
)
from src.training.dqn_run_setup import create_debug_writer, create_run_paths
from src.training.dqn_tracking import (
    BestRunState,
    maybe_update_best_checkpoint,
    print_episode_log,
    print_eval_log,
)
from src.training.dqn_training_helpers import (
    build_env_for_image,
    choose_degradation_type,
    compute_action_entropy,
    compute_action_repeat_ratio,
    evaluate_on_indices,
    extract_clean_and_degraded_images,
    set_global_seed,
)
from src.training.dqn_types import EpisodeSummaryRow, EvalHistoryRow, ResolvedConfig, RunMeta
from src.utils import apply_subset_limits, build_train_eval_indices, load_config, sample_indices


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override dict into base dict.
    Override values take precedence for leaf values.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def train() -> None:
    """Run the canonical underwater training and periodic evaluation workflow.

    The function intentionally keeps configuration resolution, deterministic
    splitting, interaction, optimization, checkpoint selection, and artifact
    writing in one visible top-level flow. Lower-level operations live in the
    ``dqn_*`` helper modules so each phase can be tested independently.
    """
    supported_experiments = {
        "underwater_dqn_v1",
        "ablation_A_max_steps5",
        "ablation_B_extended_actions",
        "ablation_C_lab_stats",
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train the canonical DDQN agent for underwater image enhancement."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="underwater_dqn_v1",
        help="Canonical experiment name or approved ablation config.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="full_training",
        help="Optional phase override inside the experiment config. Only 'full_training' is supported.",
    )
    args = parser.parse_args()

    if args.experiment not in supported_experiments:
        allowed = ", ".join(sorted(supported_experiments))
        raise ValueError(f"Experiment must be one of: {allowed}")

    experiment_path = Path("configs/experiments") / f"{args.experiment}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    experiment_config = load_config(str(experiment_path))
    dataset_config = {
        "dataset": experiment_config.get("dataset", {}),
        "degradation": experiment_config.get("degradation", {}),
    }
    env_config_all = experiment_config.copy()
    training_config_all = experiment_config.copy()

    print(f"[EXPERIMENT] Loaded experiment config: {args.experiment}")
    print(f"[EXPERIMENT] Config file: {experiment_path}")
    if "metadata" in experiment_config:
        metadata = experiment_config["metadata"]
        print(f"[EXPERIMENT] Name: {metadata.get('name', 'N/A')}")
        print(f"[EXPERIMENT] Description: {metadata.get('description', 'N/A')}")

    if args.phase:
        if args.phase != "full_training":
            raise ValueError(
                "This repository now supports only the 'full_training' phase for the "
                "canonical underwater workflow."
            )
        phase_override = experiment_config.get(args.phase)
        if not isinstance(phase_override, dict):
            raise KeyError(f"Phase override '{args.phase}' not found in {experiment_path}.")

        dataset_config = deep_merge_dicts(
            dataset_config,
            {
                "dataset": phase_override.get("dataset", {}),
                "degradation": phase_override.get("degradation", {}),
            },
        )
        env_config_all = deep_merge_dicts(env_config_all, phase_override)
        training_config_all = deep_merge_dicts(training_config_all, phase_override)
        print(f"[EXPERIMENT] Applied phase override: {args.phase}")

    # Resolve configuration once and persist the same effective values with the
    # run. This prevents evaluation from silently using newer YAML defaults.
    env_config = env_config_all.get("environment", {})
    reward_config = env_config_all.get("reward", {})
    training_config = training_config_all.get("training", {})
    evaluation_config = env_config_all.get("evaluation", {})
    dataset_core_cfg = dataset_config.get("dataset", {})
    degradation_config = dataset_config.get("degradation", {})

    seed = int(training_config.get("seed", 42))
    set_global_seed(seed)

    max_steps = int(env_config.get("max_steps", 10))
    dataset_image_size = get_effective_image_size(dataset_core_cfg)
    image_size = (dataset_image_size, dataset_image_size)
    include_step_channel = bool(env_config.get("include_step_channel", True))
    include_lab_stats = bool(env_config.get("include_lab_stats", False))
    action_set_name = str(env_config.get("action_set", "general"))
    use_psnr = bool(reward_config.get("use_psnr", True))
    use_ssim = bool(reward_config.get("use_ssim", False))
    if use_psnr and use_ssim:
        reward_metric = "combined"
    elif use_psnr:
        reward_metric = "psnr"
    elif use_ssim:
        reward_metric = "ssim"
    else:
        raise ValueError("At least one of reward.use_psnr or reward.use_ssim must be enabled.")

    step_penalty = float(reward_config.get("step_penalty", 0.01))
    repeated_action_penalty = float(reward_config.get("repeated_action_penalty", 0.0))
    no_improvement_penalty = float(reward_config.get("no_improvement_penalty", 0.0))
    psnr_weight = float(reward_config.get("psnr_weight", 1.0))
    ssim_weight = float(reward_config.get("ssim_weight", 10.0))
    stop_bonus_scale = float(reward_config.get("stop_bonus_scale", 0.0))
    stop_no_improvement_penalty = float(reward_config.get("stop_no_improvement_penalty", 0.0))
    early_stop_min_improvement = float(reward_config.get("early_stop_min_improvement", 0.0))
    truncate_without_stop_penalty = float(reward_config.get("truncate_without_stop_penalty", 0.0))
    stop_action_bonus = float(reward_config.get("stop_action_bonus", 0.0))
    terminal_reward_psnr_scale = float(reward_config.get("terminal_reward_psnr_scale", 0.0))
    terminal_reward_ssim_scale = float(reward_config.get("terminal_reward_ssim_scale", 0.0))

    default_degradation_type = degradation_config.get("type", "gaussian_noise")
    candidate_degradation_types = degradation_config.get("candidate_types", [])
    if not isinstance(candidate_degradation_types, list):
        candidate_degradation_types = []
    noise_std = float(degradation_config.get("noise_std", 0.1))
    train_subset_size = int(dataset_core_cfg.get("train_subset_size", 0) or 0)
    eval_subset_size = int(dataset_core_cfg.get("eval_subset_size", 0) or 0)

    num_episodes = int(training_config.get("num_episodes", 120))
    batch_size = int(training_config.get("batch_size", 64))
    gamma = float(training_config.get("gamma", 0.99))
    use_double_dqn = bool(training_config.get("use_double_dqn", True))
    use_dueling_dqn = bool(training_config.get("use_dueling_dqn", False))
    lr = float(training_config.get("learning_rate", 1e-4))
    buffer_size = int(
        training_config.get("buffer_size", training_config.get("replay_buffer_size", 50_000))
    )
    target_update_every = int(
        training_config.get(
            "target_update_every", training_config.get("target_update_frequency", 5)
        )
    )
    eval_every = int(
        training_config.get("eval_every", training_config.get("checkpoint_frequency", 10))
    )
    num_eval_episodes = int(
        training_config.get("num_eval_episodes", evaluation_config.get("subset_size", 20))
    )
    epsilon_start = float(training_config.get("epsilon_start", 1.0))
    epsilon_end = float(training_config.get("epsilon_end", 0.05))
    epsilon_decay = float(training_config.get("epsilon_decay", 0.995))
    eval_pool_size = int(training_config.get("eval_pool_size", 500))

    dataset_root = os.getenv("DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("DATASET_ROOT is not defined in .env")

    dataset_name = get_dataset_name(dataset_core_cfg)
    train_dataset = load_train_dataset(dataset_core_cfg, dataset_root=dataset_root)
    # The evaluation pool is deterministic and disjoint from the training pool.
    # Checkpoint comparisons therefore see the same images throughout a run.
    train_indices, eval_indices = build_train_eval_indices(
        dataset_size=len(train_dataset),
        eval_pool_size=eval_pool_size,
        seed=seed,
    )
    train_indices, eval_indices = apply_subset_limits(
        train_indices=train_indices,
        eval_indices=eval_indices,
        train_subset_size=train_subset_size,
        eval_subset_size=eval_subset_size,
        seed=seed,
    )
    eval_tracking_subset = sample_indices(
        eval_indices,
        k=min(num_eval_episodes, len(eval_indices)),
        seed=seed + 20_240,
    )
    if not eval_tracking_subset:
        raise RuntimeError(
            "Empty eval_tracking_subset: check eval indices and evaluation subset size."
        )

    sample_degradation_type = choose_degradation_type(
        default_type=default_degradation_type,
        candidate_types=candidate_degradation_types,
        key=seed,
    )
    if not train_indices:
        raise RuntimeError("Empty train_indices: check eval_pool_size/dataset split configuration.")
    sample_clean_image, sample_degraded_image = extract_clean_and_degraded_images(
        train_dataset[train_indices[0]]
    )
    sample_env = build_env_for_image(
        clean_image=sample_clean_image,
        max_steps=max_steps,
        image_size=image_size,
        reward_metric=reward_metric,
        step_penalty=step_penalty,
        repeated_action_penalty=repeated_action_penalty,
        no_improvement_penalty=no_improvement_penalty,
        stop_bonus_scale=stop_bonus_scale,
        stop_no_improvement_penalty=stop_no_improvement_penalty,
        early_stop_min_improvement=early_stop_min_improvement,
        truncate_without_stop_penalty=truncate_without_stop_penalty,
        stop_action_bonus=stop_action_bonus,
        terminal_reward_psnr_scale=terminal_reward_psnr_scale,
        terminal_reward_ssim_scale=terminal_reward_ssim_scale,
        psnr_weight=psnr_weight,
        ssim_weight=ssim_weight,
        include_step_channel=include_step_channel,
        include_lab_stats=include_lab_stats,
        action_set_name=action_set_name,
        degradation_type=sample_degradation_type,
        noise_std=noise_std,
        degraded_image=sample_degraded_image,
    )
    action_space = cast(spaces.Discrete, sample_env.action_space)
    obs_shape = sample_env.observation_space.shape
    if obs_shape is None:
        raise RuntimeError("Observation space shape is None; cannot infer in_channels.")
    num_actions = int(action_space.n)

    logs_root = Path(os.getenv("LOGS_ROOT", "logs"))
    checkpoint_root = Path(os.getenv("CHECKPOINT_ROOT", "checkpoints"))

    run_paths = create_run_paths(logs_root=logs_root, checkpoint_root=checkpoint_root)
    run_id = run_paths.run_id
    run_log_dir = run_paths.run_log_dir
    run_ckpt_dir = run_paths.run_ckpt_dir

    debug_writer, debug_file_handle, debug_episodes = create_debug_writer(run_log_dir)
    stop_action_id = get_stop_action_id(action_set_name)

    agent = DQNAgent(
        num_actions=num_actions,
        in_channels=int(obs_shape[-1]),
        epsilon=epsilon_start,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        use_double_dqn=use_double_dqn,
        use_dueling_dqn=use_dueling_dqn,
    )

    print(
        f"[DEVICE] Agent device: {agent.device} | "
        f"torch.cuda.is_available={torch.cuda.is_available()} | "
        f"torch.cuda.device_count={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        print(f"[DEVICE] CUDA device name: {torch.cuda.get_device_name(0)}")

    print(f"[RUN] run_id={run_id} | logs={run_log_dir} | checkpoints={run_ckpt_dir}")
    print(
        f"[DATASET] {dataset_name} train_size={len(train_dataset)} "
        f"train_pool={len(train_indices)} eval_pool={len(eval_indices)} "
        f"(subset_cfg train={train_subset_size if train_subset_size > 0 else 'full'}, "
        f"eval={eval_subset_size if eval_subset_size > 0 else 'full'}) "
        f"data_root={dataset_root}"
    )
    print(f"[EVAL] fixed_tracking_subset={len(eval_tracking_subset)} | eval_every={eval_every}")
    print(f"[ACTION_SET] {action_set_name} | num_actions={num_actions}")
    print(
        f"[DEGRADATION] default_type={default_degradation_type} "
        f"candidate_types={candidate_degradation_types if candidate_degradation_types else [default_degradation_type]}"
    )

    replay_buffer = ReplayBuffer(capacity=buffer_size)

    rewards_history: list[float] = []
    losses_history: list[float | None] = []
    eval_history: list[EvalHistoryRow] = []
    episode_summary_rows: list[EpisodeSummaryRow] = []

    best_state = BestRunState(best_eval_subset=[])

    # Each episode samples one paired image and creates a fresh environment.
    # The image-level rollout is short (five steps in v4.0), but replay updates
    # reuse transitions across many later episodes.
    for episode in range(1, num_episodes + 1):
        episode_seed = seed + episode
        episode_rng = random.Random(episode_seed)
        image_idx = episode_rng.choice(train_indices)
        episode_degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=image_idx + episode_seed,
        )

        clean_image, degraded_image = extract_clean_and_degraded_images(train_dataset[image_idx])

        env = build_env_for_image(
            clean_image=clean_image,
            max_steps=max_steps,
            image_size=image_size,
            reward_metric=reward_metric,
            step_penalty=step_penalty,
            repeated_action_penalty=repeated_action_penalty,
            no_improvement_penalty=no_improvement_penalty,
            stop_bonus_scale=stop_bonus_scale,
            stop_no_improvement_penalty=stop_no_improvement_penalty,
            early_stop_min_improvement=early_stop_min_improvement,
            truncate_without_stop_penalty=truncate_without_stop_penalty,
            stop_action_bonus=stop_action_bonus,
            terminal_reward_psnr_scale=terminal_reward_psnr_scale,
            terminal_reward_ssim_scale=terminal_reward_ssim_scale,
            psnr_weight=psnr_weight,
            ssim_weight=ssim_weight,
            include_step_channel=include_step_channel,
            include_lab_stats=include_lab_stats,
            action_set_name=action_set_name,
            degradation_type=episode_degradation_type,
            noise_std=noise_std,
            degraded_image=degraded_image,
        )

        state, _ = env.reset(seed=episode_seed)
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_actions: list[int] = []
        stop_used = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            episode_actions.append(action)
            if action == stop_action_id:
                stop_used = 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if debug_writer is not None and episode <= debug_episodes:
                debug_writer.writerow(
                    {
                        "episode": episode,
                        "step": step + 1,
                        "action": action,
                        "action_name": info.get("action_name"),
                        "reward": reward,
                        "delta_quality": info.get("delta_quality"),
                        "step_penalty_applied": info.get("step_penalty_applied"),
                        "repeated_penalty_applied": info.get("repeated_penalty_applied"),
                        "no_improvement_penalty_applied": info.get(
                            "no_improvement_penalty_applied"
                        ),
                        "stop_bonus_applied": info.get("stop_bonus_applied"),
                        "stop_no_improvement_penalty_applied": info.get(
                            "stop_no_improvement_penalty_applied"
                        ),
                        "previous_quality": info.get("previous_quality"),
                        "quality": info.get("quality"),
                        "psnr": info.get("psnr"),
                        "ssim": info.get("ssim"),
                        "terminated": terminated,
                        "truncated": truncated,
                        "epsilon": agent.epsilon,
                    }
                )

            replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            loss = agent.optimize_model(replay_buffer)
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Exponential epsilon decay: explore heavily in early episodes, then
        # gradually shift toward the greedy policy, floored at `epsilon_end`
        # so a small amount of exploration always remains.
        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        # Sync the target network every few episodes rather than every step.
        # A target that moves too quickly would chase its own predictions
        # (the same instability Double DQN's selection/evaluation split
        # helps mitigate); a fixed target for a while keeps the Bellman
        # targets stable long enough to actually converge toward them.
        if episode % target_update_every == 0:
            agent.update_target_network()

        avg_loss = float(np.mean(episode_losses)) if episode_losses else None
        action_entropy = compute_action_entropy(episode_actions, num_actions)
        action_repeat_ratio = compute_action_repeat_ratio(episode_actions)

        rewards_history.append(episode_reward)
        losses_history.append(avg_loss)

        episode_summary_rows.append(
            EpisodeSummaryRow(
                episode=float(episode),
                image_idx=float(image_idx),
                reward=float(episode_reward),
                avg_loss=float(avg_loss) if avg_loss is not None else np.nan,
                epsilon=float(agent.epsilon),
                steps=float(len(episode_actions)),
                action_entropy=float(action_entropy),
                action_repeat_ratio=float(action_repeat_ratio),
                stop_used=float(stop_used),
            )
        )

        print_episode_log(
            episode=episode,
            num_episodes=num_episodes,
            episode_reward=episode_reward,
            avg_loss=avg_loss,
            epsilon=agent.epsilon,
            steps=len(episode_actions),
            action_entropy=action_entropy,
            action_repeat_ratio=action_repeat_ratio,
            stop_used=stop_used,
        )

        if episode % eval_every == 0:
            eval_stats = evaluate_on_indices(
                agent=agent,
                dataset=train_dataset,
                eval_indices=eval_tracking_subset,
                eval_step_seed=seed + 10_000 + episode,
                max_steps=max_steps,
                image_size=image_size,
                reward_metric=reward_metric,
                step_penalty=step_penalty,
                repeated_action_penalty=repeated_action_penalty,
                no_improvement_penalty=no_improvement_penalty,
                stop_bonus_scale=stop_bonus_scale,
                stop_no_improvement_penalty=stop_no_improvement_penalty,
                early_stop_min_improvement=early_stop_min_improvement,
                truncate_without_stop_penalty=truncate_without_stop_penalty,
                stop_action_bonus=stop_action_bonus,
                terminal_reward_psnr_scale=terminal_reward_psnr_scale,
                terminal_reward_ssim_scale=terminal_reward_ssim_scale,
                psnr_weight=psnr_weight,
                ssim_weight=ssim_weight,
                include_step_channel=include_step_channel,
                include_lab_stats=include_lab_stats,
                action_set_name=action_set_name,
                default_degradation_type=default_degradation_type,
                candidate_degradation_types=candidate_degradation_types,
                noise_std=noise_std,
            )

            eval_history.append(
                EvalHistoryRow(
                    **eval_stats,
                    episode=float(episode),
                    eval_subset=list(eval_tracking_subset),
                )
            )

            print_eval_log(episode, eval_stats)
            # Checkpoint selection uses this periodic evaluation on a fixed,
            # held-out subset — not the training reward and not simply the
            # last episode — because RL training is not monotonic (see
            # docs/CURRENT_STATE.md: the final checkpoint of run 1494 is
            # measurably worse than its best one).
            best_state = maybe_update_best_checkpoint(
                run_state=best_state,
                eval_stats=eval_stats,
                eval_subset=eval_tracking_subset,
                episode=episode,
                run_ckpt_dir=run_ckpt_dir,
                agent=agent,
                num_actions=num_actions,
                use_double_dqn=use_double_dqn,
                use_dueling_dqn=use_dueling_dqn,
                seed=seed,
                train_indices=train_indices,
                eval_indices=eval_indices,
                run_id=run_id,
                image_size=image_size,
            )

    best_eval_reward, best_delta_psnr, best_eval_episode, best_eval_subset = best_state.as_tuple()

    final_checkpoint_payload = build_checkpoint_payload(
        agent=agent,
        num_actions=num_actions,
        best_eval_reward=best_eval_reward,
        best_delta_psnr=best_delta_psnr,
        best_eval_episode=best_eval_episode,
        best_eval_subset=best_eval_subset,
        use_double_dqn=use_double_dqn,
        use_dueling_dqn=use_dueling_dqn,
        seed=seed,
        train_indices=train_indices,
        eval_indices=eval_indices,
        run_id=run_id,
        image_size=image_size,
    )
    resolved: ResolvedConfig = {
        "run_id": run_id,
        "seed": seed,
        "dataset_root": dataset_root,
        "reward_metric": reward_metric,
        "default_degradation_type": default_degradation_type,
        "candidate_degradation_types": candidate_degradation_types,
        "noise_std": noise_std,
    }
    run_meta: RunMeta = {
        "run_id": run_id,
        "best_eval_reward": best_eval_reward,
        "best_delta_psnr": best_delta_psnr,
        "best_by_metric": "mean_delta_psnr",
        "best_eval_episode": best_eval_episode,
        "best_eval_subset_size": len(best_eval_subset),
        "checkpoint_dir": str(run_ckpt_dir),
        "log_dir": str(run_log_dir),
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "dataset_root": dataset_root,
        "seed": seed,
        "use_double_dqn": use_double_dqn,
        "use_dueling_dqn": use_dueling_dqn,
    }
    final_checkpoint_path = write_final_artifacts(
        run_log_dir=run_log_dir,
        run_ckpt_dir=run_ckpt_dir,
        final_checkpoint_payload=final_checkpoint_payload,
        episode_summary_rows=episode_summary_rows,
        eval_history=eval_history,
        seed=seed,
        train_indices=train_indices,
        eval_indices=eval_indices,
        dataset_config=dataset_config,
        env_config_all=env_config_all,
        training_config_all=training_config_all,
        resolved=resolved,
        run_meta=run_meta,
    )

    print("Training completed.")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    print(f"Best eval reward: {best_eval_reward:.4f}")
    print(f"Best delta PSNR: {best_delta_psnr:+.4f}")
    print(f"Episode summary saved to: {run_log_dir / 'episode_summary.csv'}")
    print(f"Eval summary saved to: {run_log_dir / 'eval_summary.json'}")

    if debug_file_handle is not None:
        debug_file_handle.close()
        print("[DEBUG] Reward diagnostics file closed.")


if __name__ == "__main__":
    train()
