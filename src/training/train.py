import os
import random
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from gymnasium import spaces

# Allow direct execution (python src/training/train_dqn.py) by adding
# project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import DQNAgent, ReplayBuffer
from src.actions.filters import ImageAction
from src.data import get_dataset_name, get_effective_image_size, load_train_dataset
from src.training.dqn_artifacts import (
    build_checkpoint_payload,
    write_final_artifacts,
)
from src.training.dqn_training_helpers import (
    build_env_for_image,
    choose_degradation_type,
    compute_action_entropy,
    compute_action_repeat_ratio,
    evaluate_on_indices,
    set_global_seed,
)
from src.training.dqn_run_setup import create_debug_writer, create_run_paths
from src.training.dqn_tracking import (
    BestRunState,
    maybe_update_best_checkpoint,
    print_episode_log,
    print_eval_log,
)
from src.training.dqn_types import EpisodeSummaryRow, EvalHistoryRow, ResolvedConfig, RunMeta
from src.utils import load_config, build_train_eval_indices, sample_indices, apply_subset_limits


def train() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    env_config_all = load_config("configs/environment.yaml")
    training_config_all = load_config("configs/training.yaml")

    env_config = env_config_all.get("environment", {})
    reward_config = env_config_all.get("reward", {})
    training_config = training_config_all.get("training", {})
    dataset_core_cfg = dataset_config.get("dataset", {})
    degradation_config = dataset_config.get("degradation", {})

    seed = int(training_config.get("seed", 42))
    set_global_seed(seed)

    max_steps = int(env_config.get("max_steps", 10))
    dataset_image_size = get_effective_image_size(dataset_core_cfg)
    image_size = (dataset_image_size, dataset_image_size)
    include_step_channel = bool(env_config.get("include_step_channel", True))
    reward_metric = "psnr" if bool(reward_config.get("use_psnr", True)) else "ssim"

    step_penalty = float(reward_config.get("step_penalty", 0.01))
    repeated_action_penalty = float(reward_config.get("repeated_action_penalty", 0.0))
    no_improvement_penalty = float(reward_config.get("no_improvement_penalty", 0.0))
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
    buffer_size = int(training_config.get("buffer_size", 50_000))
    target_update_every = int(training_config.get("target_update_every", 5))
    eval_every = int(training_config.get("eval_every", 10))
    num_eval_episodes = int(training_config.get("num_eval_episodes", 20))
    epsilon_start = float(training_config.get("epsilon_start", 1.0))
    epsilon_end = float(training_config.get("epsilon_end", 0.05))
    epsilon_decay = float(training_config.get("epsilon_decay", 0.995))
    eval_pool_size = int(training_config.get("eval_pool_size", 500))

    dataset_root = os.getenv("DATASET_ROOT")
    if dataset_root is None:
        raise ValueError("DATASET_ROOT is not defined in .env")

    dataset_name = get_dataset_name(dataset_core_cfg)
    train_dataset = load_train_dataset(dataset_core_cfg, dataset_root=dataset_root)
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

    sample_degradation_type = choose_degradation_type(
        default_type=default_degradation_type,
        candidate_types=candidate_degradation_types,
        key=seed,
    )
    if not train_indices:
        raise RuntimeError("Empty train_indices: check eval_pool_size/dataset split configuration.")
    sample_image, _ = train_dataset[train_indices[0]]
    sample_env = build_env_for_image(
        clean_image=sample_image.convert("RGB"),
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
        include_step_channel=include_step_channel,
        degradation_type=sample_degradation_type,
        noise_std=noise_std,
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

    print(
        f"[RUN] run_id={run_id} | logs={run_log_dir} | checkpoints={run_ckpt_dir}"
    )
    print(
        f"[DATASET] {dataset_name} train_size={len(train_dataset)} "
        f"train_pool={len(train_indices)} eval_pool={len(eval_indices)} "
        f"(subset_cfg train={train_subset_size if train_subset_size > 0 else 'full'}, "
        f"eval={eval_subset_size if eval_subset_size > 0 else 'full'}) "
        f"data_root={dataset_root}"
    )
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

    for episode in range(1, num_episodes + 1):
        episode_seed = seed + episode
        episode_rng = random.Random(episode_seed)
        image_idx = episode_rng.choice(train_indices)
        episode_degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=image_idx + episode_seed,
        )

        # Load image(s) based on dataset type
        if dataset_name == "UIEB":
            # UIEB returns (raw_degraded, reference_clean) tuple
            degraded_image, clean_image = train_dataset[image_idx]
        else:
            # CIFAR10/STL10 return (image, label) tuple
            clean_image, _ = train_dataset[image_idx]
            degraded_image = None  # Will be generated by degrade_image in build_env_for_image
        
        env = build_env_for_image(
            clean_image=clean_image.convert("RGB"),
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
            include_step_channel=include_step_channel,
            degradation_type=episode_degradation_type,
            noise_std=noise_std,
            degraded_image=degraded_image.convert("RGB") if degraded_image else None,
        )

        state, _ = env.reset(seed=episode_seed)
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_actions: list[int] = []
        stop_used = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            episode_actions.append(action)
            if action == int(ImageAction.STOP):
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
                        "no_improvement_penalty_applied": info.get("no_improvement_penalty_applied"),
                        "stop_bonus_applied": info.get("stop_bonus_applied"),
                        "stop_no_improvement_penalty_applied": info.get("stop_no_improvement_penalty_applied"),
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

        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

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
            eval_subset = sample_indices(
                eval_indices,
                k=num_eval_episodes,
                seed=seed + episode,
            )
            eval_stats = evaluate_on_indices(
                agent=agent,
                dataset=train_dataset,
                eval_indices=eval_subset,
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
                include_step_channel=include_step_channel,
                default_degradation_type=default_degradation_type,
                candidate_degradation_types=candidate_degradation_types,
                noise_std=noise_std,
            )

            eval_history.append(
                EvalHistoryRow(
                    **eval_stats,
                    episode=float(episode),
                    eval_subset=list(eval_subset),
                )
            )

            print_eval_log(episode, eval_stats)
            best_state = maybe_update_best_checkpoint(
                run_state=best_state,
                eval_stats=eval_stats,
                eval_subset=eval_subset,
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
