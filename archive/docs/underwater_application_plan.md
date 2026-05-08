# Underwater Image Enhancement: Application Plan

**Date**: 2026-05-08  
**Problem Domain**: Underwater imaging enhancement via RL  
**Inspiration**: University of Bologna 2022 + broader underwater imaging research  
**Status**: Phase 7 Planning

---

## Executive Summary

Underwater image enhancement is a **real-world problem** with practical value. RL is particularly suited because different underwater environments require different enhancement strategies.

This document justifies the move from Phase 0-3 (CIFAR-10 toy problem) to Phase 7 (realistic application) and outlines the implementation roadmap.

---

## The Problem: Underwater Image Degradation

### **Why Underwater Imaging is Difficult**

When a camera is submerged, light travels through water and is affected by:

#### **1. Wavelength-Dependent Absorption**

Water absorbs different wavelengths at different rates:
- **Red**: Absorbed within ~1 meter depth
- **Green**: Absorbed within ~5 meters
- **Blue**: Penetrates furthest (~50+ meters)

**Result**: Color shift toward blue/cyan (color cast)

#### **2. Backscatter**

Particles in water scatter light:
- **Effect**: Reduced contrast, haze appearance (like fog)
- **Varying**: Highly dependent on water clarity (turbidity)

#### **3. Blur & Noise**

- **Blur**: From turbidity, limited visibility, diffusion
- **Noise**: Sensor noise amplified when boosting contrast

#### **4. Low Dynamic Range**

- Water absorbs intensity across all depths
- Effective dynamic range is limited (less info available)

### **Visual Manifestation**

```
Underwater image characteristics:
├─ Blue/green color cast (red channel very weak)
├─ Low contrast (most pixels in mid-tone range)
├─ Haze/blur (backscatter dominates)
├─ Visible particles (noise-like artifacts)
└─ Limited texture detail
```

---

## Fixed Enhancement Pipeline vs Adaptive RL

### **Classical Approach (Fixed Pipeline)**

```
Input underwater image
    ↓
1. White balance (restore color cast)
    ↓
2. Contrast stretch (recover dynamic range)
    ↓
3. Sharpening (compensate for blur)
    ↓
Output enhanced image
```

**Problem**: One-size-fits-all fails on diverse images.

**Example failure cases**:
- **Very dark image**: Skip sharpening (will amplify noise)
- **Clear water**: Skip aggressive contrast (will over-saturate)
- **High turbidity**: May need aggressive CLAHE, skip sharpening

### **Adaptive RL Approach**

```
Input underwater image
    ↓
Agent decides: Which action next?
    ├─ (1) White balance?
    ├─ (2) Contrast enhancement?
    ├─ (3) Sharpening?
    ├─ (4) Denoise?
    ├─ (5) Gamma correction?
    ├─ ...
    └─ (19) STOP?
    ↓
Execute action, observe new image
    ↓
Agent gets reward (image quality improved?)
    ↓
Repeat until STOP chosen or max steps reached
    ↓
Output enhanced image
```

**Advantage**: Different images → different action sequences

**Example benefits**:
- Clear water image → fewer, gentler actions (preserve detail)
- Turbid image → aggressive CLAHE, strong white balance
- Noisy image → emphasize denoise over sharpening
- Dark image → gamma up before other enhancements

---

## Why RL is Suited for Underwater Enhancement

### **Criterion 1: Diversity of Optimal Strategies**

✅ **Underwater problem**: Different water conditions (turbidity, depth, camera) require different enhancement sequences
- Clear shallow water: light white balance + mild contrast boost
- Deep dark water: aggressive color correction + strong contrast
- Turbid water: CLAHE + denoise + careful sharpening

✅ **RL strength**: Learns per-image adaptation

### **Criterion 2: Interpretable Actions**

✅ **Underwater domain**: Enhancement operators are concrete image transformations
- White balance (color correction)
- Contrast boost (dynamic range expansion)
- Gamma correction (brightness curve)
- Denoising (noise removal)
- Sharpening (edge enhancement)
- CLAHE (adaptive histogram)
- DCP (dehazing prior)

✅ **RL advantage**: Black-box neural networks (CNN) don't reveal WHY image improved; RL action sequences are human-readable

### **Criterion 3: Avoiding Over-processing**

✅ **Underwater challenge**: Over-enhancement degrades quality (noise amplification, color distortion, loss of detail)

✅ **RL strength**: Learns STOP action (early termination) → avoids applying too many operations

**Example**:
- Fixed pipeline: Always applies 3-5 operations, even if 1-2 sufficient
- RL: Learns to STOP early if reward plateaus

### **Criterion 4: Real-World Application Value**

✅ **Underwater imaging use cases**:
- Marine research (coral surveys, species identification)
- Archaeology (submerged artifacts)
- Subsea inspection (pipelines, cables, structures)
- Autonomous underwater vehicles (AUV) vision
- Sports diving (GoPro enhancement)

✅ **Practical need**: Better image quality → better data collection → better science

---

## MDP Formulation: Underwater Enhancement

### **State Space**

Following Bologna 2022 approach (optimized for underwater domain):

$$S_t = [\text{CIELAB histogram}_{(8000)} \oplus \text{VGG-19 features}_{(4096)}]$$

**Why this representation?**

- **CIELAB**: Designed for perceptual color uniformity
  - Captures color imbalance (red-deficient underwater images)
  - Histogram bins bin color distribution without spatial information
  
- **VGG-19 features** (from fc1 layer):
  - Pre-trained on ImageNet (understands "natural" images)
  - High-level semantic features (textures, objects)
  - Captures perceptual quality beyond pixel-level metrics

**Why not raw image?**
- Direct CNN input would require end-to-end learning (large state space, slower training)
- Hand-crafted features reduce model capacity needed, faster convergence

### **Action Space**

**20 discrete underwater-specific operators** (replica Bologna):

| Group | Actions | Purpose |
|-------|---------|---------|
| **Color** | White balance (Grayworld) | Correct blue/green cast |
| **Color** | Red channel boost | Restore lost red wavelength |
| **Brightness** | Increase/decrease (×2) | Adjust overall intensity |
| **Contrast** | Increase/decrease (×2) | Expand dynamic range |
| **Gamma** | Gamma up/down (×2) | Non-linear brightness curve |
| **Denoise** | Gaussian denoise | Remove sensor noise |
| **Sharpen** | Sharpen filter | Restore edge details blurred by water |
| **Equalize** | Histogram EQ + CLAHE (×2) | Adaptive contrast enhancement |
| **Advanced** | DCP (Dark Channel Prior) | Physics-based dehazing |
| **Advanced** | Emboss | Texture emphasis |
| **Control** | STOP | End episode (avoid over-processing) |
| **Reserved** | 5 actions | Future expansion (hierarchical, multi-scale, etc.) |

**Total: 20 discrete actions**

### **Reward Function**

**Primary reward** (reference-based, requires paired data):

$$R_t = \alpha \cdot \Delta\text{PSNR}_t + \beta \cdot \Delta\text{SSIM}_t - \lambda \cdot \text{step\_penalty} + \gamma \cdot \text{terminal\_bonus}$$

Where:
- $\Delta\text{PSNR}_t = \text{PSNR}(I_t, I_{\text{ref}}) - \text{PSNR}(I_{t-1}, I_{\text{ref}})$ (PSNR improvement)
- $\Delta\text{SSIM}_t = \text{SSIM}(I_t, I_{\text{ref}}) - \text{SSIM}(I_{t-1}, I_{\text{ref}})$ (SSIM improvement)
- $\text{step\_penalty} = \lambda$ per step (encourages efficiency)
- $\text{terminal\_bonus} = \gamma$ if action is STOP (encourages early stopping)
- $\alpha = 1.0, \beta = 0.5, \lambda = 0.01, \gamma = 0.2$ (tunable)

**Interpretation**:
- Maximize PSNR improvement (primary pixel quality)
- Secondary SSIM improvement (perceptual quality)
- Penalize every step taken (encourage few operations)
- Bonus for choosing STOP (avoid over-processing)

**Future (Phase 7 v1.1)**: Add perceptual loss

$$R_t^{\text{v1.1}} = R_t + \delta \cdot \Delta\text{VGG\_MSE}_t \quad (\delta = 0.05)$$

### **Episode Horizon**

- **Max steps**: 5 enhancement operations per image
  - Empirically: underwater optimal enhancement rarely needs >5 operations
  - Limits computation (5 forward passes per image, feasible in practice)
  
- **Termination condition**: Episode ends when:
  - Agent chooses STOP action (early termination), OR
  - Max steps reached (hard cap)

- **Adaptive benefit**: Different images choose different numbers of operations
  - Example: Clear image stops after 2 steps
  - Example: Turbid image uses all 5 steps

---

## Dataset: UIEB (Underwater Image Enhancement Benchmark)

### **Overview**

- **Size**: ~20,000 paired underwater images
- **Format**: Paired data (degraded input, enhanced reference)
- **Resolution**: Variable, typically 400-600 pixels
- **Source**: Standardized benchmark from underwater imaging research
- **Availability**: Public, reproducible

### **Data Split**

```
Total: ~20,000 images
├─ Train: 14,000 images (70%) → RL training
├─ Validation: 3,000 images (15%) → Hyperparameter tuning
└─ Test: 3,000 images (15%) → Final evaluation (no data leakage)
```

**Deterministic seed-based split** (same split every run for reproducibility)

### **Preprocessing**

```
Raw image (variable resolution)
    ↓
Resize to 128×128 (for network uniformity)
    ↓
Normalize to [0, 1] float
    ↓
Extract CIELAB histogram (8000-dim)
    ↓
Extract VGG-19 features (4096-dim)
    ↓
Concatenate → 12,096-dim state
```

**Why 128×128?**
- Compromise between detail (>128 too slow) and efficiency (<128 loses info)
- Standard in underwater enhancement benchmarks
- Matches Phase 0-3 STL-10 resolution

---

## Baseline Methods

For every trained policy, we compare against:

### **Classical Baselines**

| Method | Implementation | Expected performance |
|--------|-----------------|-------------------|
| **Input-only** | No enhancement | Baseline (delta_psnr = 0) |
| **White balance only** | Grayworld in CIELAB | Removes color cast |
| **CLAHE only** | OpenCV CLAHE | Local contrast boost |
| **WB + CLAHE** | Sequential application | Combines color + contrast |
| **DCP baseline** | Dark Channel Prior + CLAHE | Physics-based dehazing |
| **Aggressive blend** | WB + CLAHE + sharpen | Multi-step classical |

### **Supervised Baselines (If Available)**

- **WaterNet** (if pre-trained model available): CNN trained for underwater enhancement
  - Expected to beat RL (supervised has more data advantage)
  - But less interpretable (black-box)

---

## Expected RL Advantages

**RL should beat classical baselines because**:

1. **Adaptive action selection**: Chooses appropriate operations per image
2. **Early stopping**: Avoids over-enhancement
3. **Joint optimization**: Actions chosen to maximize compound reward, not independently
4. **Learning from examples**: Sees thousands of (state, action, reward) tuples

**RL may lose to supervised (WaterNet) because**:

1. WaterNet trained on more labeled data
2. Supervised has explicit ground truth guidance per pixel
3. RL has exploration overhead (samples many suboptimal actions)

**Our value proposition**: Not pure PSNR, but **interpretability** + **adaptivity** + **sample efficiency**

---

## Action Specifications (20 Underwater Operators)

### **Detailed Action List**

```
ID 0: WHITE_BALANCE
  Type: Color correction
  Technique: Grayworld assumption in CIELAB space
  Effect: Restore color balance, remove blue/green cast
  Parameters: Automatic (assumes gray world)
  
ID 1-2: BRIGHTNESS_UP / BRIGHTNESS_DOWN
  Type: Luminance
  Parameters: factor = 1.15 (up) / 0.85 (down)
  Effect: Brighten or darken globally
  
ID 3-4: CONTRAST_UP / CONTRAST_DOWN
  Type: Contrast
  Parameters: factor = 1.15 (up) / 0.85 (down)
  Effect: Expand or compress dynamic range
  
ID 5: RED_CHANNEL_BOOST
  Type: Color emphasis
  Parameters: Multiply R channel by 1.5
  Effect: Restore red wavelength lost underwater
  
ID 6-7: GAMMA_UP / GAMMA_DOWN
  Type: Non-linear brightness
  Parameters: gamma = 0.85 (up, brighten) / 1.15 (down, darken)
  Effect: Curve-based brightness adjustment
  
ID 8: GAUSSIAN_DENOISE
  Type: Noise reduction
  Parameters: radius = 0.7
  Effect: Smooth noise while preserving edges
  
ID 9: SHARPEN
  Type: Edge enhancement
  Parameters: Standard kernel
  Effect: Enhance edges blurred by water scattering
  
ID 10: EMBOSS
  Type: Texture
  Parameters: alpha = 0.05
  Effect: Subtle texture emphasis
  
ID 11: HISTOGRAM_EQ
  Type: Equalization
  Parameters: None (global)
  Effect: Global histogram equalization (may over-contrast)
  
ID 12: CLAHE
  Type: Adaptive equalization
  Parameters: clip_limit = 2.0, grid = 8×8
  Effect: Local adaptive contrast (better than global HE)
  
ID 13: DCP (Dark Channel Prior)
  Type: Dehazing
  Parameters: w = 0.95 (weighting)
  Effect: Physics-based dehazing for haze reduction
  
ID 14-18: RESERVED
  Type: Future
  Parameters: (e.g., bilateral filtering, morphological ops, etc.)
  Effect: (TBD)
  
ID 19: STOP
  Type: Control
  Parameters: None
  Effect: Terminate episode (use no more operations)
```

---

## Training Strategy

### **Phase 7 v1.0 (Smoke Test)**

```
Episodes: 1,000
Train subset: 500 images
Eval subset: 100 images
Seed: 42
Duration: ~30 min
Goal: Verify pipeline works, basic learning signal exists
```

**Success criteria**:
- ✅ Training runs without errors
- ✅ mean_delta_psnr > 0
- ✅ stop_rate > 1%
- ✅ No single-action collapse

### **Phase 7 v1.0 (Full)**

```
Episodes: 5,000
Train subset: 5,000 images (full training set)
Eval subset: 500 images
Val subset: 500 images
Seed: 42
Duration: ~2-3 hours
Goal: Train policy with acceptance gate criteria
```

**Success criteria** (strict gate):
- ✅ stop_rate >= 5%
- ✅ dominant_action_share <= 50%
- ✅ mean_delta_psnr > 0
- ✅ output_psnr >= input_psnr
- ✅ Beats at least 2 baseline methods

### **Phase 7 v1.1 (Scaling)**

```
Episodes: 20,000+
Full UIEB dataset
Reward: Add perceptual loss (VGG-19 MSE)
Duration: ~1-2 days
Goal: Production-quality policy
```

---

## Roadmap: Implementation Phases

### **Week 1: Setup & Baseline**
- Days 1-2: Dataset loader (UIEB), config, basic setup
- Days 2-3: Implement 20 actions, unit test each
- Days 3-4: Implement reward function, baselines

### **Week 2: Training & Analysis**
- Days 1-2: Small training (1k episodes)
- Days 2-3: Notebook + visual analysis
- Days 3-4: Baseline comparison, action analysis

### **Week 3: Production Run**
- Days 1-2: Full training (5k episodes)
- Days 2-3: OOD validation
- Days 3-4: Documentation, artifacts, publication ready

### **Week 4: Extensions (Optional)**
- Perceptual loss (v1.1)
- Hierarchical RL (v2.0)
- Transfer learning (v2.1)

---

## How This Connects to Prior Work

### **From Phase 0-3 (CIFAR-10)**
We proved:
- ✅ DQN/DDQN is stable
- ✅ Evaluation gates work
- ✅ Action analysis is interpretable
- ✅ YAML config modularity is effective

### **Replicating for Underwater**
We will:
- ✅ Use same DQN core (no changes needed)
- ✅ Reuse eval gate framework
- ✅ Reuse action analysis code
- ✅ Extend YAML config system (add dataset_uieb.yaml, environment_underwater.yaml)

### **Improvements in Phase 7**
We will:
- ✅ Use real-world dataset (UIEB vs CIFAR synthetic)
- ✅ Use specialized actions (20 underwater-specific, not 8 generic)
- ✅ Use reference-based reward (paired data available)
- ✅ Compare to domain-specific baselines (WaterNet, CLAHE, DCP)
- ✅ Test OOD robustness (different underwater environments)

---

## Success Metrics

### **Quantitative**
- mean_delta_psnr > 0.5 dB (meaningful improvement)
- stop_rate > 10% (substantial early termination)
- Beats CLAHE + white balance baseline (PSNR)
- Reproducible: same seed → same results ±0.01

### **Qualitative**
- Learned action sequences make intuitive sense
- Visual rollout shows sensible progression (e.g., color first, then contrast)
- No catastrophic failures (image not darkened or over-saturated)
- Generalizes to unseen underwater environments (basic OOD test)

### **Technical**
- All tests pass
- No code regressions (CIFAR-10 baseline still works)
- Artifacts stored per run (checkpoint, config, analysis)
- Documentation complete

---

## Next Steps

1. ✅ **This document approved** → Proceed to implementation
2. ⏳ **Phases 2-5** (docs) → Complete first
3. ⏳ **Phase 6** (config) → YAML files
4. ⏳ **Phases 7-11** (code) → Dataset, actions, reward, baselines, notebook
5. ⏳ **Phase 12-14** (training) → Smoke test → Full training
6. ⏳ **Phase 15-16** (validation) → OOD + publishing

---

**Status**: Application plan complete. Ready for implementation.
