from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.training.dqn_artifacts import build_checkpoint_payload, write_best_checkpoint
from src.training.dqn_types import EvalStats


@dataclass
class BestRunState:
    best_eval_reward: float = -float("inf")
    best_delta_psnr: float = -float("inf")
    best_eval_episode: int = 0
    best_eval_subset: list[int] | None = None

    def as_tuple(self) -> tuple[float, float, int, list[int]]:
        return (
            self.best_eval_reward,
            self.best_delta_psnr,
            self.best_eval_episode,
            self.best_eval_subset or [],
        )


def print_episode_log(
    episode: int,
    num_episodes: int,
    episode_reward: float,
    avg_loss: float | None,
    epsilon: float,
    steps: int,
    action_entropy: float,
    action_repeat_ratio: float,
    stop_used: int,
) -> None:
    print(
        f"Episode {episode:03d}/{num_episodes} | "
        f"Train reward: {episode_reward:.4f} | "
        f"Avg loss: {avg_loss} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Steps: {steps} | "
        f"Entropy: {action_entropy:.3f} | "
        f"Repeat: {action_repeat_ratio:.3f} | "
        f"Stop: {stop_used}"
    )


def print_eval_log(episode: int, eval_stats: EvalStats) -> None:
    print(
        f"[EVAL] Episode {episode:03d} | "
        f"Mean reward: {eval_stats['mean_eval_reward']:.4f} | "
        f"Delta PSNR: {eval_stats['mean_delta_psnr']:+.4f} | "
        f"Stop rate: {eval_stats['stop_rate']:.3f} | "
        f"Dominant action share: {eval_stats['dominant_action_share']:.3f}"
    )


def maybe_update_best_checkpoint(
    *,
    run_state: BestRunState,
    eval_stats: EvalStats,
    eval_subset: list[int],
    episode: int,
    run_ckpt_dir: Path,
    agent,
    num_actions: int,
    use_double_dqn: bool,
    use_dueling_dqn: bool,
    seed: int,
    train_indices: list[int],
    eval_indices: list[int],
    run_id: str,
    image_size: tuple[int, int] | None = None,
) -> BestRunState:
    current_delta_psnr = float(eval_stats["mean_delta_psnr"])
    current_eval_reward = float(eval_stats["mean_eval_reward"])
    is_better_psnr = current_delta_psnr > run_state.best_delta_psnr
    is_psnr_tie = abs(current_delta_psnr - run_state.best_delta_psnr) <= 1e-12
    is_better_tie_break = is_psnr_tie and current_eval_reward > run_state.best_eval_reward

    if not (is_better_psnr or is_better_tie_break):
        return run_state

    run_state.best_delta_psnr = current_delta_psnr
    run_state.best_eval_reward = current_eval_reward
    run_state.best_eval_episode = episode
    run_state.best_eval_subset = list(eval_subset)

    best_checkpoint_path = run_ckpt_dir / "dqn_best_policy_net.pt"
    best_payload = build_checkpoint_payload(
        agent=agent,
        num_actions=num_actions,
        best_eval_reward=run_state.best_eval_reward,
        best_delta_psnr=run_state.best_delta_psnr,
        best_eval_episode=run_state.best_eval_episode,
        best_eval_subset=run_state.best_eval_subset,
        use_double_dqn=use_double_dqn,
        use_dueling_dqn=use_dueling_dqn,
        seed=seed,
        train_indices=train_indices,
        eval_indices=eval_indices,
        run_id=run_id,
        image_size=image_size,
        episode=episode,
    )
    write_best_checkpoint(best_checkpoint_path, best_payload)
    print(
        "[BEST] New best checkpoint saved | "
        f"Delta PSNR: {run_state.best_delta_psnr:+.4f} | "
        f"Eval reward (tie-break): {run_state.best_eval_reward:.4f}"
    )
    return run_state
