# Bologna 2022 vs Our Approach: Underwater RL Analysis

**Date**: 2026-05-08  
**Research Focus**: University of Bologna 2022 project "Underwater Image Enhancement with Reinforcement Learning"  
**Reference**: GitHub [sissaNassir/Underwater-Image_Enhancement](https://github.com/sissaNassir/Underwater-Image_Enhancement)

---

## Executive Summary

Bologna 2022 demonstrated that **DDQN can learn underwater image enhancement policies**. This document compares their approach with:
1. Our Phase 0-3 baseline (CIFAR-10/STL-10)
2. Our planned Phase 7 underwater enhancement

**Key finding**: Bologna's concept is sound, but their engineering is monolithic. Our framework replicates their insight with better modularity, evaluation, and reproducibility.

---

## Comparison Table: Bologna vs Our Phases

| Aspect | Bologna 2022 | Our Phase 0-3 (CIFAR) | Our Phase 7 (Underwater, Planned) |
|--------|-------------|----------------------|----------------------------------|
| **Dataset** | Li Chongyi (890 paired) | CIFAR-10 + STL-10 (60k) | UIEB (~20k paired) |
| **Dataset type** | Real underwater images | Synthetic degradation | Real underwater images |
| **Image resolution** | Full resolution (variable) | 32×32 (CIFAR), 96×96 (STL) | 128×128 (normalized) |
| **State space** | 12,096-dim (CIELAB + VGG) | 128×128×3 image + step channel | 12,096-dim (CIELAB + VGG) |
| **State extraction** | Manual: CIELAB histogram + fc1 layer | Raw image + step embedding | Manual: same as Bologna |
| **Action space** | 20 discrete operators | 8 discrete filters | 20 discrete operators (as Bologna) |
| **Action types** | Enhance, color correction, denoise | General filters | Underwater-specific (replica Bologna) |
| **Algorithm** | DDQN | DDQN | DDQN |
| **Reward function** | Reference-based hybrid | Reference-based (PSNR + SSIM) | Reference-based hybrid (PSNR + SSIM + perceptual) |
| **Reward weights** | α=1 (color), β=0.05 (perceptual) | α=1, β=0.5, λ=0.01 (step) | α=1, β=0.5, λ=0.01, γ=0.2 |
| **Training episodes** | 20,000 (fragmented) | 10,000 (per phase) | 5,000 (Phase 7 v1) → 20,000 (if stable) |
| **Batch size** | 32 | 32 | 32 |
| **Learning rate** | 1e-5 to 1e-4 | 1e-4 | 1e-4 |
| **Replay buffer** | 50,000 | 50,000 | 50,000 |
| **Evaluation metrics** | MSE, PSNR, SSIM | PSNR, SSIM, delta_psnr | PSNR, SSIM, delta_psnr, (later: UIQM, UCIQE) |
| **Baseline comparison** | WaterNet, Water-CycleGAN, DenseGAN | Simple filters | WaterNet, CLAHE, DCP, white balance |
| **Acceptance gate** | None (loose criteria) | Strict (5 criteria) | Strict (6 criteria, enhanced) |
| **Action analysis** | Minimal (results only) | Comprehensive (frequency, rollout, best/worst) | Comprehensive (same + heatmaps future) |
| **Visual analysis** | Static images | Interactive notebook + rollouts | Interactive notebook + rollout video |
| **Reproducibility** | Low (notebook-based) | High (YAML config, fixed seed, effective_config.json) | High (same + per-run artifacts) |
| **Configuration** | Hardcoded in notebook | YAML modularity | YAML modularity (same pattern) |
| **Code modularity** | Monolithic notebook | Modular (actions, datasets, reward separate) | Modular (same architecture) |
| **OOD testing** | No | No | Yes (Phase 15) |
| **Documentation** | Sparse | Complete | Complete + decision log |

---

## Detailed Technical Comparison

### **1. Dataset Choice**

**Bologna: Li Chongyi Benchmark**
- 890 paired underwater images
- Input: degraded underwater photo
- Reference: enhanced version (ground truth)
- Pros: Directly matched their formulation
- Cons: Limited size (890 → easily overfits)

**Our Phase 0-3: CIFAR-10/STL-10**
- Synthetic degradation (not real underwater)
- But allowed rapid prototyping
- Proved framework works

**Our Phase 7: UIEB**
- ~20,000 paired underwater images
- Input: degraded underwater photo
- Reference: enhanced version
- Pros: Larger, more diverse, standard benchmark
- Cons: Different distribution than Bologna (but OK for generalization test)

**Decision**: UIEB is better for production (size + diversity) than Bologna's small dataset.

---

### **2. State Representation**

**Bologna**:
```
State = [CIELAB histogram (8000) || VGG-19 fc1 features (4096)]
       = 12,096-dimensional vector
```

**Our Phase 0-3**:
```
State = [Resized image (128×128×3) || Step embedding]
      = 49,152 + 1 = 49,153-dimensional
```
(Learned directly from image without hand-crafted features)

**Our Phase 7 (Planned)**:
```
State = [CIELAB histogram (8000) || VGG-19 fc1 features (4096)]
      = 12,096-dimensional vector
```
(Replica Bologna for comparison, but with option to optimize later via PCA)

**Comparison**:
- Bologna: **Interpretable features**, smaller state space, but requires feature engineering
- Our CIFAR: **End-to-end learning**, larger state space, data-efficient
- Our underwater: **Same as Bologna** (fair comparison), with path to optimize via PCA (v1.1)

---

### **3. Action Space**

**Bologna (20 discrete actions)**:

| ID | Action | Type | Parameters |
|----|--------|------|-----------|
| 0-1 | Brightness ± | PIL Enhancer | factor=1.15/0.85 |
| 2-3 | Contrast ± | PIL Enhancer | factor=1.15/0.85 |
| 4 | Gaussian denoise | Filter | radius=0.7 |
| 5 | Sharpen | Filter | kernel |
| 6-7 | Gamma ± | LUT | gamma=0.85/1.15 |
| 8 | Histogram EQ | OpenCV | — |
| 9 | CLAHE | OpenCV | clip_limit=2.0 |
| 10 | White balance | CIELAB | Grayworld |
| 11 | Emboss | ImgAug | alpha=0.05 |
| 12 | DCP (Dark Channel Prior) | Dehazing | w=0.95 |
| 13-19 | Reserved | — | — |
| 19 | STOP | Control | — |

**Our Phase 0-3 (8 actions)**:
- White balance, brightness ±, contrast ±, sharpen, denoise, STOP
- (Simpler, general-purpose)

**Our Phase 7 (20 actions, Planned)**:
- Exact replica of Bologna to allow direct comparison
- Implementation modular: each action in separate function
- Tested individually before integration

---

### **4. Reward Function**

**Bologna**:
$$R(t) = \alpha \cdot R_c(t) + \beta \cdot R_p(t)$$

Where:
- $R_c(t) = -[\text{MAE}(\text{CIELAB}_{\text{current}}, \text{ref}) - \text{MAE}(\text{CIELAB}_{\text{prev}}, \text{ref})]$ (color)
- $R_p(t) = -[\text{MSE}(\Phi_{\text{VGG}}(\text{current}), \Phi_{\text{VGG}}(\text{ref})) - \text{MSE}(\Phi_{\text{VGG}}(\text{prev}), \text{ref})]$ (perceptual)
- $\alpha = 1.0, \beta = 0.05$

**Our Phase 0-3**:
$$R(t) = \text{PSNR}(t) - \text{PSNR}(t-1) + 0.5 \cdot (\text{SSIM}(t) - \text{SSIM}(t-1)) - 0.01 \cdot \text{step\_penalty}$$

(Simpler, no perceptual loss, but effective for CIFAR)

**Our Phase 7 (Planned, Phase 7 v1)**:
$$R(t) = \alpha \cdot \Delta\text{PSNR}(t) + \beta \cdot \Delta\text{SSIM}(t) - \lambda \cdot \text{step\_penalty} + \gamma \cdot \text{terminal\_bonus}$$

Where:
- $\alpha = 1.0, \beta = 0.5$ (initially, no perceptual loss in v1)
- $\lambda = 0.01, \gamma = 0.2$

**Our Phase 7 v1.1 (Future)**:
- Add perceptual loss: $+ 0.05 \cdot \Delta\text{MSE\_VGG}(t)$
- Matches Bologna more closely

---

### **5. Algorithm: DDQN Details**

**Bologna implementation**:
```
Network: 12096 → 4096 (ReLU) → 4096 (ReLU) → 512 (ReLU) → 20 (Q-values)
Target update: Every episode
Optimizer: ADAM (lr=1e-5 to 1e-4)
Loss: MSE (Bellman residual)
```

**Our Phase 0-3**:
```
Network: ~49k → 2048 → 512 → |A| (Q-values)
Target update: Every episode
Optimizer: ADAM (lr=1e-4)
Loss: MSE (same)
```

**Our Phase 7**:
```
Network: 12096 → 4096 → 4096 → 512 → 20 (Q-values)
Target update: Every episode (as Bologna)
Optimizer: ADAM (lr=1e-4, slightly higher than Bologna)
Loss: MSE (same)
```

**Difference**: Our architecture exactly mirrors Bologna for fair comparison.

---

### **6. Evaluation & Results**

**Bologna Results** (after 20k episodes):
- Best MSE: 1.22 (×10³)
- Best PSNR: 17.26 dB
- Best SSIM: 0.722

**Baseline comparison** (Bologna):
| Method | MSE | PSNR | SSIM |
|--------|-----|------|------|
| WaterNet | 0.80 | 19.11 | 0.797 |
| Water-CycleGAN | 1.73 | 15.75 | 0.521 |
| DenseGAN | 1.22 | 17.28 | 0.443 |
| **DDQN** | **2.12** | **15.47** | **0.628** |

→ **Bologna's DDQN loses to WaterNet** (supervised learning wins)

**Our Phase 3a Results** (10k episodes, CIFAR):
- mean_delta_psnr: +0.7033
- stop_rate: 0.0546
- Acceptance gate: Partial pass

**Future Phase 7 expectations**:
- On UIEB underwater data
- Expect to beat simple baselines (CLAHE, white balance)
- May not beat WaterNet (supervised), but goal is **interpretability + adaptivity**, not pure PSNR

---

### **7. Known Limitations in Bologna**

| Limitation | Impact | Our Mitigation |
|-----------|--------|--------------|
| Small dataset (890 images) | Overfitting, poor generalization | Use UIEB (~20k images) |
| Memory constraints (OOM every 3-4h) | Training fragmented, progress lost | Larger batch, better HPC management |
| Mode collapse (single action after 5k eps) | Policy learns only one trick | Stricter acceptance gates, action diversity monitoring |
| Manual reward tuning | Tedious, no systematic search | Ablation study planned, YAML config-driven |
| No acceptance gate | Hard to tell if result is good | Strict gate: stop_rate, action_collapse, delta_psnr, etc. |
| Limited baseline comparison | No context for DDQN performance | Compare vs WaterNet, CLAHE, DCP, white balance |
| No visual analysis | Hard to debug failures | Interactive notebook with step-by-step rollouts |
| Monolithic notebook | Hard to extend or reproduce | Modular code: src/actions/, src/data/, src/evaluation/ |

---

## What We Will Reuse from Bologna

✅ **Concept**:
- DDQN is effective for underwater RL enhancement
- 20-action set is reasonably complete
- Reference-based reward (PSNR + SSIM + perceptual) is sound

✅ **Technical Details**:
- State representation: CIELAB histogram + VGG-19 features
- Action parameterization (white balance, contrast, gamma, denoise, CLAHE, DCP, etc.)
- Network architecture (12096 → 4096 → 4096 → 512 → 20)
- Hyperparameters (lr=1e-4, gamma=0.99, batch=32, replay=50k)

✅ **Evaluation**:
- Metrics: PSNR, SSIM as primary evaluation (add UIQM/UCIQE later)
- Baseline methods: WaterNet, CLAHE, DCP, white balance comparisons

---

## What We Will Improve vs Bologna

### **1. Engineering & Reproducibility**
| Aspect | Bologna | Ours |
|--------|---------|------|
| Config | Hardcoded in notebook cells | YAML files (dataset, environment, training, reward) |
| Reproducibility | Notebook, manual setup | Deterministic: seed, effective_config.json, split saving |
| Code structure | Monolithic | Modular (actions/, data/, evaluation/ packages) |
| Testing | None visible | Unit tests per action, reward, baseline |

### **2. Evaluation Rigor**
| Aspect | Bologna | Ours |
|--------|---------|------|
| Acceptance criteria | None (post-hoc results) | Gate: stop_rate, action_collapse, delta_psnr, output_psnr, artifacts |
| Baseline comparison | 3 methods | 6+ methods (input_only, white_balance, CLAHE, DCP, WaterNet if available) |
| Visual analysis | Static images | Interactive notebook: step-by-step, rollout, frequency histograms, best/worst cases |
| Metrics detail | PSNR/SSIM only | PSNR/SSIM + delta metrics + stop rate + action frequency + component rewards |

### **3. Interpretability**
| Aspect | Bologna | Ours |
|--------|---------|------|
| Action analysis | Implicit (from results) | Explicit: action_analysis.json, histograms, stop_rate, dominant_action_share |
| Debugging | Hard (notebook implicit) | Clear: logs, effective_config, action sequence traces |
| Visual rollout | No | Yes: step-by-step image + reward + action sequence |

### **4. Scalability & Extensibility**
| Aspect | Bologna | Ours |
|--------|---------|------|
| New action? | Modify monolithic code | Add to src/actions/underwater_v1.py, register, test independently |
| New dataset? | Rewrite data loading | Add loader to src/data/datasets.py, register in YAML |
| New reward? | Modify training loop | Pluggable: src/training/reward_underwater.py |
| New baseline? | Hardcode comparison | Add to src/evaluation/baselines_underwater.py |

### **5. Publication Readiness**
| Aspect | Bologna | Ours |
|--------|---------|------|
| Artifact storage | Results only | Full run: checkpoints, logs, effective_config, splits, artifacts |
| Reproducibility checklist | None | Generated per run |
| Documentation | README in repo | Multiple docs: MDP formulation, action spec, reward formula, baseline comparison |
| Decision log | None | AGENT_CONTEXT.md + phase documentation |

---

## What We Will Build New

🔧 **New Components**:

1. **Dataset modularity**: Load UIEB, CIFAR-10, future datasets via YAML
2. **Action registry**: Underwater_v1 as modular plugin (not monolithic)
3. **Reward abstraction**: Pluggable reward classes (reference-based, perceptual, no-reference)
4. **Baseline suite**: Underwater-specific classical methods
5. **Acceptance gate**: Strict criteria prevent publishing suboptimal results
6. **Interactive notebook**: Visual policy analysis (CIFAR → Underwater version)
7. **OOD validation**: Test robustness to out-of-distribution degradations

---

## Mapping: How We Address Bologna Limitations

### **Scenario A: Training Unstable, Mode Collapse**

**Bologna's observation**: Policy collapses to single action after 5k episodes  
**Root cause**: Insufficient diversity incentive, possibly reward imbalance

**Our approach**:
1. Strict gate: `dominant_action_share <= 0.5` (fail if one action > 50%)
2. Monitor per-episode: if collapse detected, adjust reward weights
3. Action analysis: frequency histogram visible in notebook
4. Stop rate monitoring: if low → increase terminal bonus

### **Scenario B: OOD Generalization Fails**

**Bologna**: No OOD testing reported  
**Our approach (Phase 15)**:
1. Test on synthetic underwater variations (stronger noise, different blue cast)
2. Test on different camera models (if available in UIEB)
3. Test on different environments (shallow vs deep, varying turbidity)
4. Compare: in-distribution delta_psnr vs OOD delta_psnr
5. Document degradation gracefully

### **Scenario C: Stop Rate Too Low**

**Bologna**: No explicit stop rate target  
**Our Phase 3a**: stop_rate = 0.0546 (target: 0.10)

**Our tuning strategy**:
1. Increase γ (terminal bonus) in reward
2. Decrease step penalty λ to make steps cheaper (shouldn't stop too early)
3. Monitor trade-off: stop_rate vs delta_psnr
4. Accept lower stop_rate if delta_psnr strong enough

---

## Decision Rationale: UIEB vs Bologna Dataset

**Why UIEB over Bologna's Li Chongyi?**

| Criterion | UIEB | Li Chongyi |
|-----------|------|-----------|
| **Size** | ~20,000 images | 890 images |
| **Diversity** | High (different cameras, underwater conditions) | Limited |
| **Availability** | Public, standard benchmark | Less actively maintained |
| **Generalization** | Larger → better OOD | Smaller → potential overfitting |
| **Reproducibility** | Published splits available | Manual splits needed |

**Trade-off**: Not directly comparable to Bologna (different dataset), but **better for production and publication**.

---

## Conclusions: Bologna as Inspiration, Not Replication

**Bologna 2022 demonstrated**:
- ✅ DDQN can enhance underwater images
- ✅ 20-action set is effective
- ✅ Reference-based reward works

**We will**:
- ✅ Replicate their MDP formulation (for comparison)
- ✅ Use same action set (20 operators)
- ✅ Use similar network architecture
- ✗ NOT replicate their monolithic notebook approach
- ✗ NOT use their small dataset (upgrade to UIEB)

**Our improvement**:
- Better engineering (modular, configurable)
- Stronger evaluation (strict gates, baseline comparison, visual analysis)
- Higher reproducibility (YAML, seeds, artifact storage)
- Path to extensions (hierarchical RL, transfer learning, new domains)

**Expected outcome**:
- Solid Phase 7 underwater enhancement baseline
- Foundation for future RL image enhancement research
- Publication-grade code, docs, and results

---

**Next steps**: Proceed to Phase 7 implementation with UIEB dataset, 20 underwater actions, and our modular framework.
