from __future__ import annotations

from typing import TypedDict


class EvalStats(TypedDict):
    mean_eval_reward: float
    std_eval_reward: float
    mean_delta_psnr: float
    std_delta_psnr: float
    mean_delta_ssim: float
    std_delta_ssim: float
    mean_episode_length: float
    stop_rate: float
    dominant_action_share: float


class EvalHistoryRow(EvalStats):
    episode: float
    eval_subset: list[int]


class EpisodeSummaryRow(TypedDict):
    episode: float
    image_idx: float
    reward: float
    avg_loss: float
    epsilon: float
    steps: float
    action_entropy: float
    action_repeat_ratio: float
    stop_used: float


class ResolvedConfig(TypedDict):
    run_id: str
    seed: int
    dataset_root: str
    reward_metric: str
    default_degradation_type: str
    candidate_degradation_types: list[str]
    noise_std: float


class RunMeta(TypedDict):
    run_id: str
    best_eval_reward: float
    best_delta_psnr: float
    best_by_metric: str
    best_eval_episode: int
    best_eval_subset_size: int
    checkpoint_dir: str
    log_dir: str
    num_episodes: int
    max_steps: int
    dataset_root: str
    seed: int
    use_double_dqn: bool
    use_dueling_dqn: bool
