# AGENT_CONTEXT.md

## Project Identity
- **Title:** Adaptive Image Enhancement via Deep Reinforcement Learning — Underwater Domain Adaptation
- **Domain:** Reinforcement Learning + Image Processing (Underwater Imaging)
- **Primary Goal:** Learn an interpretable sequence of classical image-processing actions that improves degraded underwater images using DDQN.

## Scientific Positioning
- This project adapts existing CIFAR-10 baseline RL approach to **underwater image enhancement** domain.
- Combines synthetic (CIFAR-10 + simulated underwater degradation) and real (UIEB dataset) data.
- The target contribution is: interpretable sequential decision process with DDQN choosing underwater-specific operators (red channel boost, LAB color balance, CLAHE, saturation boost) + terminal reward for final image quality.
- Goal: demonstrate that learned policies outperform fixed classical baselines (DCP, single-filter) on both synthetic and real underwater images.

## Scientific Positioning
- This project adapts existing CIFAR-10 baseline RL approach to **underwater image enhancement** domain.
- Combines synthetic (CIFAR-10 + simulated underwater degradation) and real (UIEB dataset) data.
- The target contribution is: interpretable sequential decision process with DDQN choosing underwater-specific operators (red channel boost, LAB color balance, CLAHE, saturation boost) + terminal reward for final image quality.
- Goal: demonstrate that learned policies outperform fixed classical baselines (DCP, single-filter) on both synthetic and real underwater images.

---

## TARGET PROJECT STRUCTURE (Post-Restructuring)

```
adaptive-rl-image-enhancement/
├── configs/
│   ├── dataset.yaml          # dataset, split, degradation
│   ├── environment.yaml      # action space, reward shaping
│   ├── training.yaml         # iperparametri DDQN
│   └── experiments/          # config per-esperimento
│       ├── exp_A_cifar_baseline.yaml
│       ├── exp_B_cifar_underwater.yaml
│       └── exp_C_uieb_full.yaml
│
├── src/
│   ├── data/
│   │   ├── degradation.py    # degrade_image() + degrade_underwater()
│   │   └── dataset.py        # UIEBDataset + loader factory
│   ├── actions/
│   │   └── filters.py        # ImageAction enum + funzioni filtro (incluse azioni underwater)
│   ├── agents/
│   │   ├── dqn.py            # architettura CNN (DQN network)
│   │   ├── dqn_agent.py      # DQNAgent (DDQN: select_action, optimize_model)
│   │   └── replay_buffer.py
│   ├── envs/
│   │   └── env.py            # ImageEnhancementEnv
│   ├── metrics/
│   │   ├── psnr.py
│   │   ├── ssim.py
│   │   └── color_cast.py     # compute_color_cast_score() (nuovo)
│   ├── training/
│   │   └── train.py          # training loop
│   ├── evaluation/
│   │   ├── baselines.py      # tutte le baseline + dcp_only
│   │   └── evaluate.py       # confronto DQN vs baselines + report JSON
│   └── utils/
│       ├── config.py         # load_config(), merge con experiment override
│       └── splits.py         # build_train_eval_indices()
│
├── tests/
│   ├── test_actions.py
│   ├── test_dataset.py
│   ├── test_ddqn.py
│   ├── test_degradation.py
│   ├── test_env.py
│   ├── test_metrics.py
│   └── test_color_cast.py
│
├── scripts/
│   ├── setup_env.sh
│   ├── download_uieb.sh      # istruzioni download UIEB
│   ├── train.sh              # lancia train.py
│   ├── evaluate.sh           # lancia evaluate.py
│   └── train.sbatch          # job Slurm HPC (adattato per risorse reali)
│
├── notebooks/
│   └── progress.ipynb        # visualizzazione progressi training (read-only)
│
├── docs/
│   ├── HPC_PROFILE.md        # profilo macchina HPC (GPU, RAM, CUDA, storage)
│   ├── CURRENT_STATE.md      # stato del progetto
│   └── RESULTS.md            # template risultati esperimenti
│
├── main.py                   # entrypoint
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── AGENT_CONTEXT.md
```

---

## HPC INFRASTRUCTURE AND STORAGE POLICY

### Storage Locations (Critical)
**All datasets and training artifacts MUST be stored in `/storage/internal_02/ccoppola/`, NOT in user home or repo.**

```bash
# Directory structure on HPC storage (do NOT commit these to git)
/storage/internal_02/ccoppola/
├── datasets/
│   ├── CIFAR10/              # automatically downloaded
│   └── UIEB/                 # manually downloaded, structure: raw-890/, reference-890/
├── adaptive_rl_image_enhancement/
│   ├── checkpoints/          # saved model weights per run
│   │   └── dqn/<RUN_ID>/...
│   ├── results/              # final eval reports
│   ├── logs/
│   │   ├── slurm/           # Slurm job output logs
│   │   └── dqn/             # training run logs
│   │       └── <RUN_ID>/
│   ├── processed/            # preprocessed data (optional)
│   └── hf_cache/            # HuggingFace model cache
├── torch_cache/             # PyTorch pretrained weights cache
└── hf_cache/                # HuggingFace hub cache
```

### .env File (Template)
Create `.env` in project root (do NOT commit):
```bash
# Storage roots
HPC_STORAGE_ROOT=/storage/internal_02/ccoppola
DATA_ROOT=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement
DATASET_ROOT=/storage/internal_02/ccoppola/datasets
PROCESSED_DATA_ROOT=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/processed
CHECKPOINT_ROOT=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/checkpoints
RESULTS_ROOT=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/results
LOGS_ROOT=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/logs

# Cache
HF_HOME=/storage/internal_02/ccoppola/hf_cache
TORCH_HOME=/storage/internal_02/ccoppola/torch_cache
```

Setup before first training:
```bash
set -a && source .env && set +a
mkdir -p "$DATASET_ROOT" "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT/slurm" "$LOGS_ROOT/dqn"
echo "All directories created and ready."
```

### HPC Profiling (One-Time Setup)
Before running any training, profile the HPC machine and document in `docs/HPC_PROFILE.md`:

```bash
# Run these commands once and save output
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nproc
free -h
df -h /storage/internal_02
nvcc --version 2>/dev/null || nvidia-smi | grep "CUDA Version"
source venv/bin/activate && python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Save results in `docs/HPC_PROFILE.md`:
```markdown
# HPC Profile
- **GPU:** [MODEL] — [VRAM] GB
- **CPU cores:** [N]
- **RAM:** [N] GB
- **CUDA:** [VERSION]
- **PyTorch:** [VERSION]
- **Storage /storage/internal_02 available:** [GB]
```

### Slurm Job Submission Template (scripts/train.sbatch)
Adapt based on HPC_PROFILE results. Example:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=rl_underwater
#SBATCH --output=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/logs/slurm/%x_%j.out
#SBATCH --error=/storage/internal_02/ccoppola/adaptive_rl_image_enhancement/logs/slurm/%x_%j.err
#SBATCH --time=01:00:00          # Adjust: 01h for exp_A, 04h for exp_B, 12h for exp_C
#SBATCH --cpus-per-task=4        # Adjust: ~half of available cores
#SBATCH --mem=16G                # Adjust: ~half of available RAM
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --partition=gpu          # Adjust: partition name on your HPC

set -euo pipefail
PROJECT_ROOT="/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement"
cd "$PROJECT_ROOT"
source venv/bin/activate
set -a && source .env && set +a
mkdir -p "${LOGS_ROOT}/slurm" "${LOGS_ROOT}/dqn" "${CHECKPOINT_ROOT}"

echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "[TORCH] $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"

export RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
export DQN_DEBUG_REWARD=1
export DQN_DEBUG_EPISODES=10

EXPERIMENT="${EXPERIMENT:-configs/experiments/exp_A_cifar_baseline.yaml}"
echo "[RUN] RUN_ID=$RUN_ID | EXPERIMENT=$EXPERIMENT"
python3 -m src.training.train --experiment "$EXPERIMENT"
```

### Launching Training on HPC

```bash
# Experiment A (Baseline CIFAR-10, ~1h)
sbatch --export=ALL,EXPERIMENT=configs/experiments/exp_A_cifar_baseline.yaml scripts/train.sbatch

# Experiment B (CIFAR-10 + underwater degradation, ~4h)
sbatch --export=ALL,EXPERIMENT=configs/experiments/exp_B_cifar_underwater.yaml scripts/train.sbatch

# Experiment C (Real UIEB dataset, ~12h)
sbatch --export=ALL,EXPERIMENT=configs/experiments/exp_C_uieb_full.yaml scripts/train.sbatch

# Monitor jobs
squeue -u ccoppola
tail -f logs/slurm/<job_name>_<id>.out
```

---

## JUPYTER PROGRESS NOTEBOOK

**File:** `notebooks/progress.ipynb` (read-only, visualizes completed training artifacts)

### Purpose
View training progress, eval metrics, baseline comparisons, and visual results **without rerunning code**. 
The notebook reads JSON/CSV artifacts produced during training.

### Sections (8 cells)

**Cell 1 — Setup & Load Env**
```python
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("../.env")
LOGS_ROOT = Path(os.environ["LOGS_ROOT"])
CHECKPOINT_ROOT = Path(os.environ["CHECKPOINT_ROOT"])

# List all available runs
runs = sorted([d.name for d in (LOGS_ROOT / "dqn").iterdir() if d.is_dir()])
print(f"Available runs ({len(runs)}):")
for r in runs[-5:]:  # show last 5
    print(f"  {r}")
```

**Cell 2 — Select Run**
```python
# Change this value to analyze different runs
RUN_ID = runs[-1]  # default: latest run
run_dir = LOGS_ROOT / "dqn" / RUN_ID
print(f"Analyzing: {RUN_ID}")

with open(run_dir / "effective_config.json") as f:
    config = json.load(f)
print(f"Dataset: {config['resolved']['dataset']['name']}")
print(f"Episodes: {config['resolved']['training']['num_episodes']}")
```

**Cell 3 — Training Curves (Reward + Loss + Epsilon)**
```python
df = pd.read_csv(run_dir / "episode_summary.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df["reward"].rolling(20).mean().plot(ax=axes[0], title="Reward (moving avg 20ep)")
df["avg_loss"].rolling(20).mean().plot(ax=axes[1], title="Loss (moving avg 20ep)")
df["epsilon"].plot(ax=axes[2], title="Epsilon Decay")
for ax in axes:
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Total episodes: {len(df)}")
print(f"Final reward (last 20): {df['reward'].tail(20).mean():.4f}")
```

**Cell 4 — Eval Set Metrics (Delta PSNR / SSIM)**
```python
with open(run_dir / "eval_summary.json") as f:
    eval_data = json.load(f)

eval_df = pd.DataFrame(eval_data)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
eval_df.plot(x="episode", y="mean_delta_psnr", ax=axes[0],
             title="Delta PSNR on eval set", color="steelblue")
eval_df.plot(x="episode", y="mean_delta_ssim", ax=axes[1],
             title="Delta SSIM on eval set", color="coral")
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
for ax in axes:
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Cell 5 — Agent Action Distribution**
```python
action_file = run_dir / "action_analysis.json"
if action_file.exists():
    with open(action_file) as f:
        action_data = json.load(f)
    action_counts = action_data.get("action_counts", {})
    pd.Series(action_counts).sort_values().plot(
        kind="barh", figsize=(8, 4), title="Action distribution (eval)"
    )
    plt.tight_layout()
    plt.show()
else:
    print("action_analysis.json not found — run evaluate.py first")
```

**Cell 6 — DQN vs Baseline Comparison**
```python
baseline_file = run_dir / "evaluation_baselines.json"
if baseline_file.exists():
    with open(baseline_file) as f:
        baseline_data = json.load(f)
    results = baseline_data.get("results", {})
    df_bl = pd.DataFrame(results).T[["mean_psnr", "mean_ssim"]].astype(float)
    df_bl.plot(kind="bar", figsize=(10, 5), title="PSNR and SSIM: DQN vs Baselines")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    
    passed = baseline_data.get("acceptance_passed", False)
    print(f"\n✅ PASS" if passed else "❌ FAIL")
else:
    print("evaluation_baselines.json not found — run evaluate.py first")
```

**Cell 7 — Visual Comparison (Input → Output → Reference)**
```python
sample_dir = run_dir / "samples"
if sample_dir.exists():
    samples = sorted(sample_dir.glob("*_degraded.png"))[:4]
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3 * len(samples)))
    for i, s in enumerate(samples):
        idx = s.stem.split("_")[0]
        for j, kind in enumerate(["degraded", "enhanced", "reference"]):
            p = sample_dir / f"{idx}_{kind}.png"
            if p.exists():
                axes[i][j].imshow(mpimg.imread(p))
            axes[i][j].set_title(kind if i == 0 else "")
            axes[i][j].axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("samples/ directory not found")
```

**Cell 8 — Multi-Run Summary Table**
```python
summary_rows = []
for run_name in runs:
    meta_path = LOGS_ROOT / "dqn" / run_name / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        summary_rows.append({
            "run_id": run_name,
            "dataset": meta.get("dataset_name", "?"),
            "best_delta_psnr": meta.get("best_delta_psnr", None),
            "best_eval_episode": meta.get("best_eval_episode", None),
        })

if summary_rows:
    pd.DataFrame(summary_rows).set_index("run_id").sort_values(
        "best_delta_psnr", ascending=False
    )
```

---

## ROADMAP: 10-Task Restructuring & Domain Adaptation

**Workflow:** One task at a time, in order. Commit after each task.
```bash
git checkout -b feature/underwater-domain
# After each task:
git add -A
git commit -m "taskN: <description>"
```

### Task 0 — Project Restructuring ✅ FIRST PRIORITY
**Goal:** Align current repo to target structure. No logic changes, only moves/renames.

**Actions:**
- Rename files: `image_enhancement_env.py` → `env.py`, `train_dqn.py` → `train.py`, `train_dqn.sbatch` → `train.sbatch`
- Unify evaluation modules into `baselines.py` and `evaluate.py`
- Delete unused configs: `evaluation.yaml`, `paths.yaml`
- Create `configs/experiments/` with placeholder `.gitkeep`
- Update all imports accordingly
- Create `.env.example` template

### Task 1 — Double DQN (DDQN) Implementation
**File:** `src/agents/dqn_agent.py`
**Change:** Replace standard DQN Q-value update with DDQN (use policy_net for action selection, target_net for evaluation)
**Test:** `tests/test_ddqn.py` verifies DDQN logic

### Task 2 — Terminal Reward Activation
**File:** `configs/environment.yaml`
**Change:** Add `terminal_reward_psnr_scale: 1.5` to reward config
**Benefit:** Agente ottimizza direttamente la qualità finale, non solo delta locali

### Task 3 — UIEB Dataset Support
**Files:** Create `src/data/uieb_dataset.py`, create `scripts/download_uieb.sh`
**Change:** UIEBDataset class + loader factory in training
**Structure:** Dataset expected at `/storage/internal_02/ccoppola/datasets/UIEB/{raw-890, reference-890}/`

### Task 4 — Underwater-Specific Actions
**File:** `src/actions/filters.py`
**Add:** 4 new ImageAction enums + implementations
- `RED_CHANNEL_BOOST`: compensa assorbimento rosso
- `LAB_COLOR_BALANCE`: white balance in LAB
- `CLAHE`: contrast enhancement locale
- `SATURATION_BOOST`: aumenta saturazione
**Dependency:** Add `opencv-python` to requirements.txt

### Task 5 — Color Cast Reward Component
**Files:** Create `src/metrics/color_cast.py`, modify `src/envs/env.py`
**Add:** `compute_color_cast_score()` metrica + integrazione in reward computation
**Config:** `environment.yaml` con `reward_metric: combined_underwater` e `color_cast_weight`

### Task 6 — Underwater Physics Degradation
**File:** `src/data/degradation.py`
**Add:** `degrade_underwater()` function simulating absorption, backscatter, contrast loss
**Config:** New degradation type `"underwater"` con `depth_range`, `turbidity_range`

### Task 7 — Experiment Configurations (A/B/C)
**Files:** Create `configs/experiments/{exp_A, exp_B, exp_C}.yaml`
- **exp_A:** CIFAR-10 baseline (sanity check, ~1h)
- **exp_B:** CIFAR-10 + underwater degradation (~4h)
- **exp_C:** Real UIEB dataset (~12h)

### Task 8 — Baseline Integration & Final Evaluation
**Files:** `src/evaluation/baselines.py`, `src/evaluation/evaluate.py`
**Add:** `dcp_only` baseline, `color_cast_score` to evaluation report
**Output:** JSON con PSNR, SSIM, color_cast_score per method (DQN, DCP, single-filter, input_only)

### Task 9 — Documentation & README
**Files:** Update `README.md`, `docs/HPC_PROFILE.md`, `docs/CURRENT_STATE.md`
**Add:** Underwater domain context, setup instructions, expected results table

---


- **Dataset (default debug):** CIFAR-10 train split as clean source images (native `32x32`).
- **Dataset (visual-safe alternative):** STL-10 train split (native `96x96`) with controlled subset.
- **Degradations:** synthetic (`gaussian_noise`, `low_brightness`, `low_contrast`, `blur`, `combined`, and `mixed` policy).
- **Environment:** Gymnasium (`ImageEnhancementEnv`) with `max_steps`, `stop` action, optional step-channel.
- **Action space:**
  - `increase_brightness`
  - `decrease_brightness`
  - `increase_contrast`
  - `decrease_contrast`
  - `gaussian_denoise`
  - `sharpen`
  - `gamma_up`
  - `gamma_down`
  - `stop`
- **Agent:** DQN with target network and replay buffer.
- **Checkpoint selection policy:** PSNR-first (`mean_delta_psnr`), tie-break on `mean_eval_reward`.

## Infrastructure and Storage Policy
- Heavy artifacts/datasets stay on HPC storage (`/storage/internal_02/...`), not in git.
- Run artifacts are scoped by run id:
  - `${CHECKPOINT_ROOT}/dqn/<RUN_ID>/...`
  - `${LOGS_ROOT}/dqn/<RUN_ID>/...`

## Phase Status (Requested Roadmap)

### Phase 1 - Robust Evaluation Criteria
**Status:** Implemented and validated.

Implemented items:
- Acceptance gate in `evaluation_dqn_baselines.py` with hard-fail checks:
  - `baseline_report_generated`
  - `mean_delta_psnr_positive` (`mean_delta_psnr > 0`)
  - `output_psnr_ge_input_psnr`
  - `stop_rate_ok` (>= `training.min_stop_rate`)
  - `dominant_action_share_ok` (<= `training.action_collapse_threshold`)
  - `action_analysis_available` (missing analysis file => fail)
- `evaluation_baselines.json` now exports explicit gate metrics:
  - `input_psnr`, `output_psnr`, `mean_delta_psnr`
  - `input_ssim`, `output_ssim`, `mean_delta_ssim`
  - `stop_rate`, `dominant_action_share`, `avg_episode_length`
  - plus grouped `gate_metrics` block.
- `compare_dqn_runs.py` updated to show gate-oriented metrics directly in run comparison.

Validation evidence:
- Historical bad runs are rejected when quality checks fail.
- `run_comparison.json` consolidates multi-run gate outcomes.

### Phase 2 - DQN Stability Upgrades
**Status:** Partially implemented.

Implemented:
- **Double DQN:** available and active in controlled experiments (`use_double_dqn=true`).

Implemented but not yet fully validated as final setting:
- **Dueling DQN:** code path exists and has been tested in A/B runs.

Not implemented yet:
- **Prioritized Experience Replay (PER)**.

Notes:
- Significant run-to-run variance still observed.
- Some runs remain quality-negative even with behavior checks passing.

### Phase 3 - Reward Refinement
**Status:** In progress (Phase 3A completed as controlled tuning).

What was done:
- Controlled reward tuning experiments with same training budget and architecture.
- Three key runs:
  - `dqn_phase3a_control_20260506_122118`
  - `dqn_phase3a_treatment_20260506_123200`
  - `dqn_phase3a_treatment2_20260506_124100`

Observed trend:
- `stop_rate`: `0.024 -> 0.041 -> 0.055` (improving, still below threshold `0.10`).
- `mean_delta_psnr`: `-1.512 -> -0.990 -> +0.703` (alignment improved in treatment2).

Current best trade-off run:
- `dqn_phase3a_treatment2_20260506_124100`
- `mean_delta_psnr = +0.7033`
- `output_psnr_ge_input_psnr = true`
- `stop_rate = 0.0546` (gate still fails only on stop-rate)

### Phase 4 - Policy Behavior Analysis
**Status:** Implemented baseline tooling; expanded with visual diagnostics and notebook validation.

Implemented:
- `analyze_dqn_actions.py` logs:
  - action frequencies
  - per-step action frequencies
  - most common sequences
  - stop-rate
  - episode length stats
  - best/worst samples by delta PSNR
- Visual inspection artifacts generated:
  - original CIFAR-10 sample grid
  - native-vs-zoom comparison
  - step-by-step agent rollout with per-step PSNR/SSIM

Latest visual artifacts:
- `${LOGS_ROOT}/dqn/visual_inspection_20260506/`
- `${LOGS_ROOT}/dqn/stl10_safe_validation_20260507/stl10_clean_degraded_grid.png`

### Phase 5 - Generalization / OOD
**Status:** Partially implemented.

Implemented:
- OOD stress evaluation protocol with stronger degradations (e.g., noise std 0.2).

Evidence:
- ID (mixed, noise 0.1): positive (`+0.7033` on best run)
- OOD (gaussian noise 0.2): negative (`-1.9320`)

Interpretation:
- In-distribution improvement is possible.
- Generalization remains weak under stronger/no-shifted degradations.

### Phase 6 - Baseline Comparison
**Status:** Implemented for classical baselines, not closed for final scientific conclusion.

Implemented:
- `evaluation_dqn_baselines.py` compares DQN against:
  - `input_only`
  - fixed classical heuristic pipelines
- Unified eval subset and run-scoped JSON reports.

Still missing for final closure:
- Stable, repeated wins under gate constraints.
- Optional supervised lightweight baseline (if decided later).

## Important Recent Engineering Change (Image Sharpness)
Problem reported:
- "Clean" and output images looked blurry.

Root cause found:
- CIFAR-10 is native `32x32`; previous `32 -> 128` resizing introduced visible smoothing.

Applied fix:
- Training/eval/action-analysis now use YAML-driven image size aligned to dataset (`32x32`).
- Environment now avoids unnecessary resize when source image already matches target size.

Additional dataset-mode fix:
- Added STL-10 safe config with small subset for visual/eval-first workflow:
  - `configs/dataset_stl10_safe.yaml`
  - `train_subset_size=5000`
  - `eval_subset_size=500`
  - `image_size=96`

Compatibility note:
- Existing DQN conv stack was built around `128x128` feature extraction.
- For backward compatibility with legacy checkpoints, network forward now upsamples tensor inputs internally when needed.
- This keeps old checkpoints runnable while preserving dataset-faithful env images.

## What Has Been Tried So Far
- PSNR-first checkpoint selection (instead of reward-only selection).
- Strict acceptance gating and mandatory action-analysis availability.
- Double DQN activation and A/B runs.
- Dueling variant exploratory run(s).
- Reward shaping iterations (control/treatment protocol).
- ID vs OOD evaluation runs.
- Visual inspection of dataset quality and policy trajectories.
- End-to-end notebook execution validation via `nbconvert --execute` with zero runtime cell errors.
- Jupyter environment repair in `venv` (`notebook`, `nbformat`, and dependencies installed).

## Open Problems (As of 2026-05-07)
1. **Stop-rate remains below threshold** in otherwise promising runs.
2. **Run instability**: outcomes vary substantially between runs.
3. **Reward-quality mismatch risk** still exists in some settings.
4. **OOD robustness is insufficient** (quality drops under stronger degradations).
5. **Dataset ceiling**: CIFAR-10 low native resolution limits visual fidelity and realism of enhancement claims.
6. **Notebook portability**: notebook file is currently ignored by `.gitignore`, so notebook edits are local unless ignore policy changes.

## Recommended Immediate Next Steps
1. Keep CIFAR-10 as default for fast RL debug, and use STL-10 safe config for visual/eval checks.
2. Consolidate Phase 3 reward tuning to raise stop-rate without sacrificing delta PSNR.
3. Run a controlled Double-vs-Double+Dueling matrix with fixed seeds and equal budgets.
4. If stability remains poor, implement PER as next incremental Phase 2 extension.

## How to Run Core Checks
From project root (with `venv`):
- `./venv/bin/python src/evaluation/analyze_dqn_actions.py --checkpoint <ckpt>`
- `./venv/bin/python src/evaluation/evaluation_dqn_baselines.py --checkpoint <ckpt>`
- `./venv/bin/python src/evaluation/compare_dqn_runs.py`

STL-10 safe validation (no training):
- `./venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/visual_policy_analysis.ipynb --output visual_policy_analysis.executed.ipynb --output-dir notebooks`
- Visual grid output: `${LOGS_ROOT}/dqn/stl10_safe_validation_20260507/stl10_clean_degraded_grid.png`

## Key Artifacts
Per run:
- `episode_summary.csv`
- `eval_summary.json`
- `action_analysis.json`
- `evaluation_baselines.json`
- `effective_config.json`
- `run_meta.json`

Global:
- `${LOGS_ROOT}/dqn/run_comparison.json`
- `${LOGS_ROOT}/dqn/visual_inspection_20260506/`

## Conventions
- Keep modules explicit and testable.
- Prefer config-driven behavior over hardcoded constants.
- Avoid storing heavy artifacts in repository.
- Update this file whenever workflow logic, acceptance policy, or experimental conclusions change.
