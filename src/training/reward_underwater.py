"""
Underwater Image Enhancement Reward Function

Reference-based reward for paired underwater dataset.
Measures improvement in PSNR and SSIM compared to reference image.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


def compute_psnr(img1: Tensor, img2: Tensor) -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        img1, img2: Tensors with values in [0, 1]
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0  # Perfect match
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1: Tensor, img2: Tensor, window_size: int = 11) -> float:
    """
    Compute SSIM (Structural Similarity Index).
    
    Simplified version (using Gaussian window is more accurate but slower).
    
    Args:
        img1, img2: Tensors with values in [0, 1]
        window_size: Size of Gaussian window
    
    Returns:
        SSIM value in [-1, 1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


class UnderwaterReward:
    """
    Reward function for underwater image enhancement.
    
    Formula:
        R_t = alpha * delta_PSNR_t + beta * delta_SSIM_t
              - lambda * step_penalty
              + gamma * terminal_bonus
    
    Where:
        delta_PSNR_t = PSNR(output_t, reference) - PSNR(output_{t-1}, reference)
        delta_SSIM_t = SSIM(output_t, reference) - SSIM(output_{t-1}, reference)
        step_penalty = 1 if t < T else 0  (cost per step)
        terminal_bonus = 1 if is_terminal else 0  (bonus for STOP)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        step_penalty: float = 0.01,
        terminal_bonus: float = 0.2,
        use_perceptual_loss: bool = False,
        perceptual_weight: float = 0.05,
    ):
        """
        Initialize reward function.
        
        Args:
            alpha: Weight for PSNR improvement (primary metric)
            beta: Weight for SSIM improvement (secondary metric)
            step_penalty: Cost per step (encourages early stopping)
            terminal_bonus: Bonus for choosing STOP action
            use_perceptual_loss: If True, add perceptual loss component (future)
            perceptual_weight: Weight for perceptual loss if enabled
        """
        self.alpha = alpha
        self.beta = beta
        self.step_penalty = step_penalty
        self.terminal_bonus = terminal_bonus
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
    
    def __call__(
        self,
        image_prev: Tensor,
        image_curr: Tensor,
        image_reference: Tensor,
        is_terminal: bool = False,
        is_final_step: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for a single step.
        
        Args:
            image_prev: Previous image (output from previous step)
            image_curr: Current image (output after current action)
            image_reference: Reference/ground-truth image
            is_terminal: Whether this step chose STOP action
            is_final_step: Whether this is the final step (max_steps reached)
        
        Returns:
            (reward_value, components_dict)
            where components_dict contains:
                - delta_psnr: PSNR improvement
                - delta_ssim: SSIM improvement
                - psnr_reward: alpha * delta_psnr
                - ssim_reward: beta * delta_ssim
                - step_penalty: penalty applied
                - terminal_bonus: bonus if is_terminal
                - total_reward: sum of all components
        """
        # Ensure tensors are on same device
        if image_prev.device != image_reference.device:
            image_reference = image_reference.to(image_prev.device)
        if image_curr.device != image_reference.device:
            image_curr = image_curr.to(image_reference.device)
        
        # Add batch dimension if needed
        if image_prev.dim() == 3:
            image_prev = image_prev.unsqueeze(0)
        if image_curr.dim() == 3:
            image_curr = image_curr.unsqueeze(0)
        if image_reference.dim() == 3:
            image_reference = image_reference.unsqueeze(0)
        
        # Compute metrics
        psnr_prev = compute_psnr(image_prev, image_reference)
        psnr_curr = compute_psnr(image_curr, image_reference)
        delta_psnr = psnr_curr - psnr_prev
        
        ssim_prev = compute_ssim(image_prev, image_reference)
        ssim_curr = compute_ssim(image_curr, image_reference)
        delta_ssim = ssim_curr - ssim_prev
        
        # Compute reward components
        psnr_reward = self.alpha * delta_psnr
        ssim_reward = self.beta * delta_ssim
        
        # Step penalty (cost per step, encourages efficiency)
        penalty = self.step_penalty  # Always apply penalty
        
        # Terminal bonus (reward for choosing STOP)
        bonus = self.terminal_bonus if is_terminal else 0.0
        
        # Total reward
        total_reward = psnr_reward + ssim_reward - penalty + bonus
        
        # Perceptual loss (future)
        perceptual_reward = 0.0
        if self.use_perceptual_loss:
            # Placeholder for Phase 7 v1.1
            # Would use VGG-19 features here
            perceptual_reward = 0.0
        
        total_reward += perceptual_reward
        
        # Return components for logging/analysis
        components = {
            "psnr_prev": psnr_prev,
            "psnr_curr": psnr_curr,
            "delta_psnr": delta_psnr,
            "ssim_prev": ssim_prev,
            "ssim_curr": ssim_curr,
            "delta_ssim": delta_ssim,
            "psnr_reward": psnr_reward,
            "ssim_reward": ssim_reward,
            "step_penalty": penalty,
            "terminal_bonus": bonus,
            "perceptual_reward": perceptual_reward,
            "total_reward": total_reward,
        }
        
        return float(total_reward), components


def create_reward_function(config: dict) -> UnderwaterReward:
    """
    Create reward function from config dict.
    
    Args:
        config: Reward configuration from YAML (reward section)
    
    Returns:
        UnderwaterReward instance
    """
    return UnderwaterReward(
        alpha=config.get("psnr_weight", 1.0),
        beta=config.get("ssim_weight", 0.5),
        step_penalty=config.get("step_penalty", 0.01),
        terminal_bonus=config.get("terminal_bonus", 0.2),
        use_perceptual_loss=config.get("use_perceptual_loss", False),
        perceptual_weight=config.get("perceptual_weight", 0.05),
    )


if __name__ == "__main__":
    # Quick test
    print("Testing underwater reward function...")
    
    reward_fn = UnderwaterReward()
    
    # Create dummy images
    image_ref = torch.rand(3, 128, 128)
    image_prev = image_ref.clone() + 0.05 * torch.randn_like(image_ref)  # Degraded
    image_curr = image_ref.clone() + 0.02 * torch.randn_like(image_ref)  # Improved
    
    # Compute reward
    reward, components = reward_fn(image_prev, image_curr, image_ref, is_terminal=False)
    
    print(f"Reward: {reward:.4f}")
    print(f"Components:")
    for key, val in components.items():
        print(f"  {key}: {val:.4f}")
    
    # Check that improved image gives positive reward
    assert reward > 0, "Improved image should give positive reward"
    assert components["delta_psnr"] > 0, "Improved image should have positive delta_psnr"
    
    print("\n✓ Reward function test passed!")
