# Underwater Implementation Phases: Technical Architecture

**Date**: 2026-05-08  
**Purpose**: Define exactly what to build, modify, preserve, and test  
**Scope**: Phase 7 implementation roadmap

---

## Overview: Zero Ambiguity

This document removes ambiguity about:
- ✅ What code to keep (DQN core, eval gate, action analysis)
- ✅ What code to build new (dataset loader, actions, reward, baselines, notebook)
- ✅ What NOT to touch (CIFAR configs, old runs, core tests)
- ✅ Integration patterns (YAML config routing, modular registry)

---

## Part 1: KEEP (No Modifications, No Touch)

### **Core RL Engine**

**File**: `src/agents/dqn.py`

**What it does**: DQN algorithm (Q-learning with neural network function approximation)

**Why keep**: Proven stable on CIFAR-10, DDQN variant works well

**Constraints**: Do NOT modify, only use via config parameter `use_double_dqn: true`

---

**File**: `src/agents/dqn_agent.py`

**What it does**: DQN agent wrapper (training loop integrations, checkpoint save/load)

**Why keep**: Working integration with training pipeline

**Constraints**: No modification

---

**File**: `src/agents/replay_buffer.py`

**What it does**: Experience replay buffer (stores and samples (s, a, r, s') tuples)

**Why keep**: Standard implementation, no bugs reported

**Constraints**: No modification

---

### **Evaluation Gate & Analysis**

**File**: `src/evaluation/evaluation_dqn_baselines.py`

**What it does**: 
- Load trained checkpoint
- Evaluate on test set
- Apply acceptance gate (stop_rate, action_collapse, delta_psnr checks)
- Compare against baselines
- Generate action_analysis.json

**Why keep**: Gate logic is sound, reusable for underwater

**Modification needed**: Will call `baselines_underwater.py` for underwater baselines (but evaluation logic unchanged)

**Constraints**: Core gate logic sacred; only add calls to new baseline methods

---

**File**: `src/evaluation/analyze_dqn_actions.py`

**What it does**: Generate action frequency histograms, rollout visualizations, best/worst cases

**Why keep**: Action analysis pipeline is generic (works for any action set)

**Constraints**: No modification (plugs into evaluation_dqn_baselines.py)

---

### **Test Suite**

**File**: `tests/test_actions.py`

**What it does**: Tests generic action module (white_balance, brightness, contrast, etc.)

**Why keep**: Validates general-purpose actions used in CIFAR-10

**Constraints**: DO NOT TOUCH (ensure it still passes after adding underwater_v1)

**Note**: New file `tests/test_underwater_actions.py` will test underwater-specific actions separately

---

**File**: `tests/test_env.py`

**What it does**: Tests Gymnasium environment wrapper

**Why keep**: Environment is generic (works for any action set)

**Constraints**: No modification (should pass for underwater too)

---

**File**: `tests/test_degradation.py`, `tests/test_baselines.py`, `tests/test_ddqn.py`

**What it does**: Various sanity checks on degradation pipeline, DDQN basics

**Why keep**: Prevent regressions

**Constraints**: No modification

---

### **Existing Configs**

**File**: `configs/dataset_stl10_safe.yaml`

**File**: `configs/dataset.yaml`

**File**: `configs/experiments/phase_a_sanity.yaml`, `phase_b_stability.yaml`, `phase_c_final.yaml`

**Why keep**: CIFAR-10/STL-10 baseline remains available for smoke tests and comparison

**Constraints**: 
- DO NOT MODIFY (backward compatibility)
- Old runs in `logs/dqn/` remain untouched

---

### **Training Scripts**

**File**: `scripts/train.sh`

**File**: `scripts/train.sbatch`

**What it does**: Launcher scripts for training

**Why keep**: Same launchers work for both CIFAR and underwater (config-driven)

**Constraints**: No modification (config handles routing)

---

### **Storage**

**File**: Old checkpoints in `checkpoints/dqn/` and runs in `logs/dqn/`

**Why keep**: Historical record, reproducibility reference

**Constraints**: 
- New underwater runs will have distinct run_id prefixes (`dqn_underwater_*`)
- Old runs never overwritten

---

## Part 2: BUILD NEW (New Files, No Existing Counterparts)

### **Dataset Loading**

**File**: `src/data/load_uieb.py` (NEW)

**Purpose**: Load UIEB paired underwater dataset

**API**:
```python
def load_uieb_dataset(
    image_size: int,
    subset_size: Optional[int] = None,
    split: str = "train",
    seed: int = 42
) -> Tuple[Tensor, Tensor, List[str]]:
    """
    Load UIEB dataset.
    
    Returns:
      - images_degraded: (N, C, H, W) tensor [0, 1]
      - images_reference: (N, C, H, W) tensor [0, 1]
      - image_ids: List of image identifiers for tracking
    """
    pass
```

**Responsibilities**:
- ✅ Download/locate UIEB dataset from path in .env or YAML
- ✅ Split into train/val/test deterministically (seed-based)
- ✅ Resize to `image_size` (128×128 default)
- ✅ Normalize to [0, 1]
- ✅ Return paired (degraded, reference) tuples
- ✅ Raise clear error if dataset not found

**Testing**: `tests/test_uieb_loader.py` (NEW)
```python
def test_load_uieb_basic():
    images_deg, images_ref, ids = load_uieb_dataset(image_size=128, subset_size=10)
    assert images_deg.shape == (10, 3, 128, 128)
    assert images_ref.shape == (10, 3, 128, 128)
    assert len(ids) == 10

def test_deterministic_split():
    # Same seed → same images
    set1 = load_uieb_dataset(subset_size=100, split="train", seed=42)
    set2 = load_uieb_dataset(subset_size=100, split="train", seed=42)
    torch.testing.assert_close(set1[0], set2[0])
```

---

### **Underwater Actions**

**File**: `src/actions/underwater_v1.py` (NEW)

**Purpose**: 20 underwater-specific enhancement operators

**Structure**:
```python
# Each action is a function: Tensor → Tensor

def white_balance_grayworld(image: Tensor) -> Tensor:
    """White balance using grayworld assumption in CIELAB."""
    pass

def brightness_up(image: Tensor, factor: float = 1.15) -> Tensor:
    """Increase brightness."""
    pass

def brightness_down(image: Tensor, factor: float = 0.85) -> Tensor:
    """Decrease brightness."""
    pass

# ... 17 more actions

def clahe(image: Tensor, clip_limit: float = 2.0, grid: int = 8) -> Tensor:
    """Contrast Limited Adaptive Histogram Equalization."""
    pass

def stop(image: Tensor) -> Tensor:
    """No-op action for termination."""
    return image

# Registry for routing
UNDERWATER_V1_ACTIONS = {
    0: white_balance_grayworld,
    1: brightness_up,
    2: brightness_down,
    # ... 19 total
}
```

**Constraints**:
- Each action is **pure function** (no state)
- Input: normalized tensor [0, 1]
- Output: normalized tensor [0, 1]
- **Deterministic** (same input → same output)
- **Non-destructive** (original image unchanged)

**Testing**: `tests/test_underwater_actions.py` (NEW)
```python
def test_white_balance():
    blue_cast_image = create_blue_cast_image()
    output = white_balance_grayworld(blue_cast_image)
    assert output.shape == blue_cast_image.shape
    assert output.min() >= 0 and output.max() <= 1

def test_all_actions_deterministic():
    image = torch.rand(3, 128, 128)
    for action_fn in UNDERWATER_V1_ACTIONS.values():
        out1 = action_fn(image)
        out2 = action_fn(image)
        torch.testing.assert_close(out1, out2)
```

**Documentation**: `docs/underwater_action_spec.md` (NEW)
- Detailed spec for each of 20 actions
- Parameters, expected effects, limitations

---

### **Underwater Reward**

**File**: `src/training/reward_underwater.py` (NEW)

**Purpose**: Reference-based reward for underwater enhancement

**Implementation**:
```python
class UnderwaterReward:
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 step_penalty: float = 0.01,
                 terminal_bonus: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.step_penalty = step_penalty
        self.terminal_bonus = terminal_bonus
    
    def __call__(self, 
                 image_prev: Tensor,
                 image_curr: Tensor,
                 image_ref: Tensor,
                 is_terminal: bool) -> Tuple[float, Dict]:
        """
        Compute reward.
        
        Returns:
          - reward_value: float, the reward
          - components: dict with PSNR, SSIM, penalties for logging
        """
        # Compute PSNR delta
        psnr_prev = compute_psnr(image_prev, image_ref)
        psnr_curr = compute_psnr(image_curr, image_ref)
        delta_psnr = psnr_curr - psnr_prev
        
        # Compute SSIM delta
        ssim_prev = compute_ssim(image_prev, image_ref)
        ssim_curr = compute_ssim(image_curr, image_ref)
        delta_ssim = ssim_curr - ssim_prev
        
        # Compute reward
        r = self.alpha * delta_psnr + self.beta * delta_ssim
        r -= self.step_penalty
        if is_terminal:
            r += self.terminal_bonus
        
        return r, {
            "delta_psnr": delta_psnr,
            "delta_ssim": delta_ssim,
            "step_penalty": self.step_penalty,
            "terminal_bonus": self.terminal_bonus if is_terminal else 0,
        }
```

**Testing**: `tests/test_underwater_reward.py` (NEW)
```python
def test_reward_positive_on_improvement():
    reward_fn = UnderwaterReward()
    image_ref = create_reference_image()
    image_prev = create_degraded_image(reference=image_ref)
    image_curr = apply_enhancement(image_prev)  # Should improve
    
    reward, _ = reward_fn(image_prev, image_curr, image_ref, is_terminal=False)
    assert reward > 0  # Improvement should yield positive reward

def test_step_penalty_applied():
    reward_fn = UnderwaterReward(step_penalty=0.01, terminal_bonus=0)
    image_ref = create_reference_image()
    # Equal input/output (no improvement, but has step cost)
    reward, _ = reward_fn(image_ref, image_ref, image_ref, is_terminal=False)
    assert reward < 0  # Just step penalty
```

---

### **Underwater Baselines**

**File**: `src/evaluation/baselines_underwater.py` (NEW)

**Purpose**: Classical baseline methods for comparison

**Baselines**:
```python
class BaselineComparison:
    def input_only(degraded: Tensor) -> Tensor:
        """Return input unchanged (sanity check)."""
        return degraded
    
    def white_balance_only(degraded: Tensor) -> Tensor:
        """Apply white balance, stop."""
        return white_balance_grayworld(degraded)
    
    def clahe_only(degraded: Tensor) -> Tensor:
        """Apply CLAHE, stop."""
        return clahe(degraded)
    
    def white_balance_and_clahe(degraded: Tensor) -> Tensor:
        """Chain: white balance → CLAHE."""
        return clahe(white_balance_grayworld(degraded))
    
    def dcp_baseline(degraded: Tensor) -> Tensor:
        """Dark Channel Prior dehazing + CLAHE."""
        return clahe(dcp_dehaze(degraded))
    
    def aggressive_blend(degraded: Tensor) -> Tensor:
        """White balance + CLAHE + sharpen."""
        x = white_balance_grayworld(degraded)
        x = clahe(x)
        x = sharpen(x)
        return x
```

**Output format** (per baseline):
```json
{
  "method": "white_balance_and_clahe",
  "metrics": {
    "psnr_input": 15.2,
    "psnr_output": 18.7,
    "delta_psnr": 3.5,
    "ssim_input": 0.52,
    "ssim_output": 0.68,
    "delta_ssim": 0.16
  },
  "runtime_seconds": 0.045
}
```

---

### **Baseline Benchmark Script**

**File**: `scripts/benchmark_baselines.py` (NEW)

**Purpose**: Run all baselines on test set, generate comparison table

**Usage**:
```bash
python scripts/benchmark_baselines.py \
  --dataset uieb \
  --subset_size 100 \
  --output results/baselines_underwater_benchmark.json
```

**Output**: JSON file with all baseline metrics

---

### **Underwater Notebook**

**File**: `notebooks/underwater_policy_analysis.ipynb` (NEW)

**Purpose**: Interactive visual analysis for underwater enhancement

**Cells**:

1. **Setup & Load Data**
   - Load UIEB dataset
   - Visualize 5 samples (degraded + reference side-by-side)

2. **Individual Action Effects**
   - Apply each of 20 actions to first sample
   - Show PSNR/SSIM delta per action
   - Table: action → PSNR delta → SSIM delta

3. **Baseline Methods**
   - Run all 6 baselines on first 10 samples
   - Table: baseline → avg PSNR → avg SSIM

4. **Post-Training Analysis** (when checkpoint available)
   - Load trained policy
   - Select 5 test images
   - Rollout DQN step-by-step:
     * Step 0: degraded input
     * Step 1: apply action[0], show output + reward
     * Step 2: apply action[1], show output + reward
     * ...
     * Final: show total reward, final PSNR/SSIM

5. **Comparison: DQN vs Baselines**
   - Side-by-side images (input, DQN output, best baseline output)
   - Table: input PSNR / output PSNR / delta / winner

6. **Action Sequence Analysis**
   - Frequency histogram: which actions chosen most?
   - Stop rate: % episodes choosing STOP before max_steps
   - Best cases: top 5 improved, show action sequence + metrics
   - Worst cases: top 5 degraded, show action sequence + metrics

---

### **Configuration Files**

**File**: `configs/dataset_uieb.yaml` (NEW)

**Contents**:
```yaml
dataset:
  name: "uieb"
  description: "Underwater Image Enhancement Benchmark"
  
  # Data source
  path: "${UIEB_ROOT}"  # from .env
  
  # Preprocessing
  image_size: 128
  paired: true  # Has reference ground truth
  
  # Splits (deterministic)
  train_subset_size: 5000
  eval_subset_size: 500
  val_subset_size: 500
  split_seed: 42
  
  # Augmentation
  train_transforms:
    - name: "random_crop"
      size: 128
    - name: "random_flip_h"
      prob: 0.5
  
  eval_transforms:
    - name: "center_crop"
      size: 128
```

**File**: `configs/environment_underwater.yaml` (NEW)

**Contents**:
```yaml
environment:
  name: "underwater_image_enhancement"
  description: "Underwater image enhancement via RL"
  
  # MDP horizon
  max_steps: 5
  include_step_channel: true
  
  # Action set
  action_set: "underwater_v1"  # References src/actions/underwater_v1.py
  
  # Observation
  observation_type: "features"  # CIELAB + VGG features
  observation_dim: 12096
  
  # Degradation (none; dataset already degraded)
  degradation:
    type: "none"
```

**File**: `configs/experiments/underwater_dqn_v1.yaml` (NEW)

**Contents**:
```yaml
# Top-level config combining all above

metadata:
  name: "underwater_dqn_v1"
  version: "1.0"

dataset:
  <<: *include configs/dataset_uieb.yaml

environment:
  <<: *include configs/environment_underwater.yaml

reward:
  alpha: 1.0
  beta: 0.5
  step_penalty: 0.01
  terminal_bonus: 0.2
  use_perceptual_loss: false

training:
  agent: "dqn"
  use_double_dqn: true
  num_episodes: 5000
  batch_size: 32
  learning_rate: 1e-4
  seed: 42

evaluation:
  gate:
    enable: true
    require_min_stop_rate: true
    min_stop_rate: 0.05
    # ... other criteria
```

---

## Part 3: MODIFY (Extend, Backward-Compatible)

### **Registry Systems**

**File**: `src/data/datasets.py` (MODIFY - ADD, don't remove)

**Current**:
```python
DATASET_LOADERS = {
    "cifar10": load_cifar10_dataset,
    "stl10": load_stl10_dataset,
}
```

**After modification**:
```python
from src.data.load_uieb import load_uieb_dataset

DATASET_LOADERS = {
    "cifar10": load_cifar10_dataset,
    "stl10": load_stl10_dataset,
    "uieb": load_uieb_dataset,  # NEW
}
```

**Constraint**: CIFAR/STL loaders unchanged, UIEB added

---

**File**: `src/actions/__init__.py` (MODIFY - ADD, don't remove)

**Current**:
```python
from src.actions import general
ACTION_SETS = {
    "general": general.GENERAL_ACTIONS,
}
```

**After modification**:
```python
from src.actions import general
from src.actions import underwater_v1

ACTION_SETS = {
    "general": general.GENERAL_ACTIONS,
    "underwater_v1": underwater_v1.UNDERWATER_V1_ACTIONS,  # NEW
}
```

**Constraint**: General action set unchanged, underwater added

---

**File**: `src/training/train.py` (MODIFY - NO LOGIC CHANGES)

**Current**: Loads action set from config
```python
action_set_name = config["environment"]["action_set"]
actions = src.actions.ACTION_SETS[action_set_name]
```

**After modification**: No change needed! (Already config-driven)

---

**File**: `src/evaluation/baselines.py` (MODIFY - ADD underwater methods)

**Current**:
```python
def compare_baselines(output_psnr, input_psnr):
    classical_baselines = {
        "clahe": ...,
        "bilateral": ...,
    }
```

**After modification**:
```python
from src.evaluation import baselines_underwater

def compare_baselines(output_psnr, input_psnr, domain="general"):
    if domain == "underwater":
        classical_baselines = baselines_underwater.get_baselines()
    else:
        classical_baselines = {...}  # existing
```

**Constraint**: Existing logic preserved, underwater branching added

---

### **Integration Points**

**File**: `docs/CURRENT_STATE.md` (MODIFY - ADD Phase 7 section)

After Phase 7 completion, add:
```markdown
## Phase 7: Underwater Enhancement (Completed 2026-MM-DD)

Latest checkpoint: dqn_underwater_v1_20260510_...
Delta PSNR: +1.2 dB
Stop rate: 0.08
Status: Gate passing
```

---

**File**: `README.md` (MODIFY - ADD underwater instructions)

Add section:
```markdown
## Phase 7: Underwater Image Enhancement

To train underwater enhancement:
```bash
bash scripts/train.sh underwater_dqn_v1
```

To evaluate:
```bash
python scripts/compare_experiment_results.py --dataset uieb --num-samples 50
```
```

---

**File**: `.env.example` (MODIFY - ADD UIEB path)

```bash
# Existing entries
...

# New for Phase 7
UIEB_ROOT="/path/to/uieb/dataset"
```

---

## Part 4: DO NOT TOUCH (Sacred, No Modifications)

### **DQN Core Logic**

- `src/agents/dqn.py` → Neural network Q-learning
- `src/agents/dqn_agent.py` → Agent wrapper
- `src/agents/replay_buffer.py` → Experience replay

### **Evaluation Gate Logic**

- `src/evaluation/evaluation_dqn_baselines.py` → Core gate logic (only extend, never change logic)

### **Environment**

- `src/envs/env.py` → Gymnasium wrapper (action_set-agnostic)

### **Tests**

- `tests/test_actions.py` → General action tests
- `tests/test_env.py` → Environment tests
- `tests/test_ddqn.py` → DDQN algorithm tests

### **Old Configs**

- `configs/dataset.yaml`, `dataset_stl10_safe.yaml`
- `configs/experiments/phase_*.yaml`

### **Old Runs**

- `logs/dqn/dqn_phase*_*` → Historical checkpoints (read-only)
- `checkpoints/dqn/` → Old models

---

## Part 5: Integration Checklist

### **After Building New Files**

- [ ] `src/data/load_uieb.py` created, tested
- [ ] `src/actions/underwater_v1.py` created, all 20 actions implemented
- [ ] `src/training/reward_underwater.py` created, formula implemented
- [ ] `src/evaluation/baselines_underwater.py` created, 6 baselines implemented
- [ ] `tests/test_uieb_loader.py` created, passing
- [ ] `tests/test_underwater_actions.py` created, passing
- [ ] `tests/test_underwater_reward.py` created, passing
- [ ] `configs/dataset_uieb.yaml` created, valid YAML
- [ ] `configs/environment_underwater.yaml` created, valid YAML
- [ ] `configs/experiments/underwater_dqn_v1.yaml` created, valid YAML
- [ ] `notebooks/underwater_policy_analysis.ipynb` created, executable
- [ ] `docs/underwater_action_spec.md` created, complete

### **After Modifying Registries**

- [ ] `src/data/datasets.py` extended with UIEB loader
- [ ] `src/actions/__init__.py` extended with underwater_v1 action set
- [ ] `.env.example` extended with UIEB_ROOT path
- [ ] All old tests still pass (no regressions)

### **Before Training**

- [ ] UIEB dataset downloaded and path in `.env`
- [ ] `pytest tests/` all passing
- [ ] `python scripts/train.sh phase_a_sanity` still works (CIFAR baseline unchanged)
- [ ] `python src/training/train.py -c configs/experiments/underwater_dqn_v1.yaml` runs without errors (1 episode test)

---

## Data Flow Diagram

```
training config (underwater_dqn_v1.yaml)
    ├─ dataset: uieb
    ├─ action_set: underwater_v1
    ├─ reward: underwater formula
    └─ training: ddqn + seed
    
        ↓
        
train.py loads config
    ├─ DATASET_LOADERS["uieb"] → load_uieb_dataset()
    ├─ ACTION_SETS["underwater_v1"] → UNDERWATER_V1_ACTIONS (20 actions)
    ├─ UnderwaterReward() → reward computation
    └─ DDQN core (unchanged)
    
        ↓
        
Training loop (unchanged)
    ├─ Sample (state, action, reward, state') from replay buffer
    ├─ Compute Q-loss (MSE)
    ├─ Update network weights
    └─ Save checkpoint
    
        ↓
        
Evaluation (modified to add underwater baselines)
    ├─ Load checkpoint
    ├─ Run on test set
    ├─ Compute metrics (PSNR, SSIM)
    ├─ Compare vs baselines_underwater methods
    ├─ Apply acceptance gate (stop_rate, action_collapse, etc.)
    └─ Generate action_analysis.json, evaluation_baselines.json
    
        ↓
        
Notebook visualizes results
    ├─ Load test images
    ├─ Apply policy step-by-step
    ├─ Show rollout + action sequence + rewards
    └─ Compare vs baselines visually
```

---

## Backward Compatibility Guarantee

✅ After all Phase 7 changes:

```bash
# Old CIFAR training still works
bash scripts/train.sh phase_a_sanity

# Old evaluation still works
bash scripts/evaluate.sh /path/to/old/checkpoint.pt

# Old tests still pass
pytest tests/test_actions.py -v

# New underwater training works alongside
bash scripts/train.sh underwater_dqn_v1

# New notebooks work
jupyter notebook notebooks/underwater_policy_analysis.ipynb
```

**Zero breaking changes to Phase 0-3 CIFAR-10 baseline.**

---

## Next Steps

1. ✅ This document approved
2. ⏳ Build files in Part 2 (Dataset, Actions, Reward, Baselines)
3. ⏳ Modify registries in Part 3 (datasets.py, actions/__init__.py)
4. ⏳ Verify backward compatibility (old tests pass)
5. ⏳ Create configs in configs/ directory
6. ⏳ Create notebook
7. ⏳ Run smoke tests
8. ⏳ Full training

---

**Status**: Architecture locked. Ready to build.
