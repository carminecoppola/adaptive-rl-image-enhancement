from __future__ import annotations

from typing import TypedDict


class PolicyMetrics(TypedDict):
    mean_psnr_enhanced: float
    std_psnr_enhanced: float
    mean_ssim_enhanced: float
    std_ssim_enhanced: float
    mean_delta_psnr: float
    std_delta_psnr: float
    mean_delta_ssim: float
    std_delta_ssim: float


class PolicyRow(TypedDict):
    psnr_enhanced: float
    ssim_enhanced: float
    delta_psnr: float
    delta_ssim: float


class AcceptanceChecks(TypedDict):
    baseline_report_generated: bool
    mean_delta_psnr_positive: bool
    output_psnr_ge_input_psnr: bool
    dominant_action_share_ok: bool
    stop_rate_ok: bool
    action_analysis_available: bool


class SampleActionRecord(TypedDict):
    sample_index: int
    degradation_type: str
    sequence: list[str]
    episode_length: int
    input_psnr: float
    output_psnr: float
    delta_psnr: float
    input_ssim: float
    output_ssim: float
    delta_ssim: float
