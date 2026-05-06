import csv
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10

# Allow direct execution (python src/training/train_dqn.py) by adding
# project root to sys.path for absolute imports from `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import DQNAgent, ReplayBuffer
from src.data.degradation import degrade_image
from src.envs.image_enhancement_env import ImageEnhancementEnv
from src.actions.filters import get_action_name, ImageAction
from src.metrics import compute_psnr, compute_ssim
from src.utils import load_config, build_train_eval_indices, sample_indices


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_action_entropy(actions: list[int], num_actions: int) -> float:
    if not actions:
        return 0.0
    counts = np.zeros(num_actions, dtype=np.float64)
    for action in actions:
        counts[action] += 1.0
    probs = counts / counts.sum()
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum())


def compute_action_repeat_ratio(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return float(repeats / (len(actions) - 1))


def choose_degradation_type(
    default_type: str,
    candidate_types: list[str],
    key: int,
) -> str:
    if default_type != "mixed":
        return default_type
    if not candidate_types:
        return "gaussian_noise"
    return candidate_types[key % len(candidate_types)]


def build_env_for_image(
    clean_image: Image.Image,
    max_steps: int,
    reward_metric: str,
    step_penalty: float,
    repeated_action_penalty: float,
    no_improvement_penalty: float,
    stop_bonus_scale: float,
    stop_no_improvement_penalty: float,
    early_stop_min_improvement: float,
    truncate_without_stop_penalty: float,
    stop_action_bonus: float,
    terminal_reward_psnr_scale: float,
    terminal_reward_ssim_scale: float,
    include_step_channel: bool,
    degradation_type: str,
    noise_std: float,
) -> ImageEnhancementEnv:
    degraded_image = degrade_image(
        clean_image,
        degradation_type=degradation_type,
        noise_std=noise_std,
    )

    return ImageEnhancementEnv(
        clean_image=clean_image,
        degraded_image=degraded_image,
        max_steps=max_steps,
        image_size=(128, 128),
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
    )


def evaluate_on_indices(
    agent: DQNAgent,
    dataset: CIFAR10,
    eval_indices: list[int],
    eval_step_seed: int,
    max_steps: int,
    reward_metric: str,
    step_penalty: float,
    repeated_action_penalty: float,
    no_improvement_penalty: float,
    stop_bonus_scale: float,
    stop_no_improvement_penalty: float,
    early_stop_min_improvement: float,
    truncate_without_stop_penalty: float,
    stop_action_bonus: float,
    terminal_reward_psnr_scale: float,
    terminal_reward_ssim_scale: float,
    include_step_channel: bool,
    default_degradation_type: str,
    candidate_degradation_types: list[str],
    noise_std: float,
) -> dict[str, float]:
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards: list[float] = []
    action_counter: Counter = Counter()
    episode_lengths: list[int] = []
    stop_count = 0
    delta_psnr_values: list[float] = []
    delta_ssim_values: list[float] = []

    for offset, idx in enumerate(eval_indices):
        degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=idx + eval_step_seed,
        )
        clean_image, _ = dataset[idx]
        env = build_env_for_image(
            clean_image=clean_image.convert("RGB"),
            max_steps=max_steps,
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
            degradation_type=degradation_type,
            noise_std=noise_std,
        )
        state, _ = env.reset(seed=eval_step_seed + offset)
        clean_eval = env.clean_image
        degraded_eval = env.initial_degraded_image.copy()

        episode_reward = 0.0
        step_actions: list[int] = []

        for _ in range(max_steps):
            action = agent.select_action(state)
            step_actions.append(action)
            action_counter[get_action_name(action)] += 1
            if action == int(ImageAction.STOP):
                stop_count += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        episode_lengths.append(len(step_actions))
        final_eval = env.current_image.copy()
        delta_psnr_values.append(compute_psnr(final_eval, clean_eval) - compute_psnr(degraded_eval, clean_eval))
        delta_ssim_values.append(compute_ssim(final_eval, clean_eval) - compute_ssim(degraded_eval, clean_eval))

    agent.epsilon = old_epsilon

    total_actions = sum(action_counter.values())
    dominant_action_share = 0.0
    if total_actions > 0:
        dominant_action_share = max(action_counter.values()) / total_actions

    return {
        "mean_eval_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_eval_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_delta_psnr": float(np.mean(delta_psnr_values)) if delta_psnr_values else 0.0,
        "std_delta_psnr": float(np.std(delta_psnr_values)) if delta_psnr_values else 0.0,
        "mean_delta_ssim": float(np.mean(delta_ssim_values)) if delta_ssim_values else 0.0,
        "std_delta_ssim": float(np.std(delta_ssim_values)) if delta_ssim_values else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "stop_rate": float(stop_count / total_actions) if total_actions else 0.0,
        "dominant_action_share": float(dominant_action_share),
    }


def train() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    env_config_all = load_config("configs/environment.yaml")
    training_config_all = load_config("configs/training.yaml")

    env_config = env_config_all.get("environment", {})
    reward_config = env_config_all.get("reward", {})
    training_config = training_config_all.get("training", {})
    degradation_config = dataset_config.get("degradation", {})

    seed = int(training_config.get("seed", 42))
    set_global_seed(seed)

    max_steps = int(env_config.get("max_steps", 10))
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

    num_episodes = int(training_config.get("num_episodes", 120))
    batch_size = int(training_config.get("batch_size", 64))
    gamma = float(training_config.get("gamma", 0.99))
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

    train_dataset = CIFAR10(root=dataset_root, train=True, download=False)
    train_indices, eval_indices = build_train_eval_indices(
        dataset_size=len(train_dataset),
        eval_pool_size=eval_pool_size,
        seed=seed,
    )

    sample_degradation_type = choose_degradation_type(
        default_type=default_degradation_type,
        candidate_types=candidate_degradation_types,
        key=seed,
    )
    sample_image, _ = train_dataset[train_indices[0]]
    sample_env = build_env_for_image(
        clean_image=sample_image.convert("RGB"),
        max_steps=max_steps,
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
    num_actions = sample_env.action_space.n

    logs_root = Path(os.getenv("LOGS_ROOT", "logs"))
    checkpoint_root = Path(os.getenv("CHECKPOINT_ROOT", "checkpoints"))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local")
    run_id = os.getenv("RUN_ID", f"dqn_{timestamp}_{slurm_job_id}")

    run_log_dir = logs_root / "dqn" / run_id
    run_ckpt_dir = checkpoint_root / "dqn" / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    debug_reward = os.getenv("DQN_DEBUG_REWARD", "0").strip() == "1"
    debug_episodes = int(os.getenv("DQN_DEBUG_EPISODES", "5"))

    debug_writer = None
    debug_file_handle = None
    if debug_reward:
        debug_path = run_log_dir / "reward_debug.csv"
        debug_file_handle = open(debug_path, "w", newline="")
        debug_writer = csv.DictWriter(
            debug_file_handle,
            fieldnames=[
                "episode",
                "step",
                "action",
                "action_name",
                "reward",
                "delta_quality",
                "step_penalty_applied",
                "repeated_penalty_applied",
                "no_improvement_penalty_applied",
                "stop_bonus_applied",
                "stop_no_improvement_penalty_applied",
                "previous_quality",
                "quality",
                "psnr",
                "ssim",
                "terminated",
                "truncated",
                "epsilon",
            ],
        )
        debug_writer.writeheader()
        print(f"[DEBUG] Reward diagnostics enabled -> {debug_path}")

    agent = DQNAgent(
        num_actions=num_actions,
        in_channels=int(sample_env.observation_space.shape[-1]),
        epsilon=epsilon_start,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
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
        f"[DATASET] CIFAR10 train_size={len(train_dataset)} "
        f"train_pool={len(train_indices)} eval_pool={len(eval_indices)} "
        f"data_root={dataset_root}"
    )
    print(
        f"[DEGRADATION] default_type={default_degradation_type} "
        f"candidate_types={candidate_degradation_types if candidate_degradation_types else [default_degradation_type]}"
    )

    replay_buffer = ReplayBuffer(capacity=buffer_size)

    rewards_history: list[float] = []
    losses_history: list[float | None] = []
    eval_history: list[dict[str, float]] = []
    episode_summary_rows: list[dict[str, float]] = []

    best_eval_reward = -float("inf")
    best_delta_psnr = -float("inf")
    best_eval_episode = 0
    best_eval_subset: list[int] = []

    for episode in range(1, num_episodes + 1):
        episode_seed = seed + episode
        episode_rng = random.Random(episode_seed)
        image_idx = episode_rng.choice(train_indices)
        episode_degradation_type = choose_degradation_type(
            default_type=default_degradation_type,
            candidate_types=candidate_degradation_types,
            key=image_idx + episode_seed,
        )

        clean_image, _ = train_dataset[image_idx]
        env = build_env_for_image(
            clean_image=clean_image.convert("RGB"),
            max_steps=max_steps,
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
            {
                "episode": float(episode),
                "image_idx": float(image_idx),
                "reward": float(episode_reward),
                "avg_loss": float(avg_loss) if avg_loss is not None else np.nan,
                "epsilon": float(agent.epsilon),
                "steps": float(len(episode_actions)),
                "action_entropy": float(action_entropy),
                "action_repeat_ratio": float(action_repeat_ratio),
                "stop_used": float(stop_used),
            }
        )

        print(
            f"Episode {episode:03d}/{num_episodes} | "
            f"Train reward: {episode_reward:.4f} | "
            f"Avg loss: {avg_loss} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Steps: {len(episode_actions)} | "
            f"Entropy: {action_entropy:.3f} | "
            f"Repeat: {action_repeat_ratio:.3f} | "
            f"Stop: {stop_used}"
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

            eval_stats["episode"] = float(episode)
            eval_stats["eval_subset"] = eval_subset
            eval_history.append(eval_stats)

            print(
                f"[EVAL] Episode {episode:03d} | "
                f"Mean reward: {eval_stats['mean_eval_reward']:.4f} | "
                f"Delta PSNR: {eval_stats['mean_delta_psnr']:+.4f} | "
                f"Stop rate: {eval_stats['stop_rate']:.3f} | "
                f"Dominant action share: {eval_stats['dominant_action_share']:.3f}"
            )

            current_delta_psnr = float(eval_stats["mean_delta_psnr"])
            current_eval_reward = float(eval_stats["mean_eval_reward"])
            is_better_psnr = current_delta_psnr > best_delta_psnr
            is_psnr_tie = abs(current_delta_psnr - best_delta_psnr) <= 1e-12
            is_better_tie_break = is_psnr_tie and current_eval_reward > best_eval_reward

            if is_better_psnr or is_better_tie_break:
                best_delta_psnr = current_delta_psnr
                best_eval_reward = current_eval_reward
                best_eval_episode = episode
                best_eval_subset = list(eval_subset)

                best_checkpoint_path = run_ckpt_dir / "dqn_best_policy_net.pt"
                torch.save(
                    {
                        "policy_net_state_dict": agent.policy_net.state_dict(),
                        "target_net_state_dict": agent.target_net.state_dict(),
                        "epsilon": agent.epsilon,
                        "num_actions": num_actions,
                        "episode": episode,
                        "best_eval_reward": best_eval_reward,
                        "best_delta_psnr": best_delta_psnr,
                        "best_by_metric": "mean_delta_psnr",
                        "best_eval_episode": best_eval_episode,
                        "best_eval_subset": best_eval_subset,
                        "seed": seed,
                        "train_indices": train_indices,
                        "eval_indices": eval_indices,
                        "run_id": run_id,
                    },
                    best_checkpoint_path,
                )
                print(
                    "[BEST] New best checkpoint saved | "
                    f"Delta PSNR: {best_delta_psnr:+.4f} | "
                    f"Eval reward (tie-break): {best_eval_reward:.4f}"
                )

    final_checkpoint_path = run_ckpt_dir / "dqn_final_policy_net.pt"
    torch.save(
        {
            "policy_net_state_dict": agent.policy_net.state_dict(),
            "target_net_state_dict": agent.target_net.state_dict(),
            "epsilon": agent.epsilon,
            "num_actions": num_actions,
            "best_eval_reward": best_eval_reward,
            "best_delta_psnr": best_delta_psnr,
            "best_by_metric": "mean_delta_psnr",
            "best_eval_episode": best_eval_episode,
            "best_eval_subset": best_eval_subset,
            "seed": seed,
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "run_id": run_id,
        },
        final_checkpoint_path,
    )

    episode_csv = run_log_dir / "episode_summary.csv"
    with open(episode_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "image_idx",
                "reward",
                "avg_loss",
                "epsilon",
                "steps",
                "action_entropy",
                "action_repeat_ratio",
                "stop_used",
            ],
        )
        writer.writeheader()
        writer.writerows(episode_summary_rows)

    eval_json = run_log_dir / "eval_summary.json"
    with open(eval_json, "w") as f:
        json.dump(eval_history, f, indent=2)

    split_json = run_log_dir / "dataset_split.json"
    with open(split_json, "w") as f:
        json.dump(
            {
                "seed": seed,
                "train_indices": train_indices,
                "eval_indices": eval_indices,
            },
            f,
        )

    effective_config_json = run_log_dir / "effective_config.json"
    with open(effective_config_json, "w") as f:
        json.dump(
            {
                "dataset": dataset_config,
                "environment": env_config_all,
                "training": training_config_all,
                "resolved": {
                    "run_id": run_id,
                    "seed": seed,
                    "dataset_root": dataset_root,
                    "reward_metric": reward_metric,
                    "default_degradation_type": default_degradation_type,
                    "candidate_degradation_types": candidate_degradation_types,
                    "noise_std": noise_std,
                },
            },
            f,
            indent=2,
        )

    run_meta_json = run_log_dir / "run_meta.json"
    with open(run_meta_json, "w") as f:
        json.dump(
            {
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
            },
            f,
            indent=2,
        )

    print("Training completed.")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    print(f"Best eval reward: {best_eval_reward:.4f}")
    print(f"Best delta PSNR: {best_delta_psnr:+.4f}")
    print(f"Episode summary saved to: {episode_csv}")
    print(f"Eval summary saved to: {eval_json}")

    if debug_file_handle is not None:
        debug_file_handle.close()
        print("[DEBUG] Reward diagnostics file closed.")


if __name__ == "__main__":
    train()
