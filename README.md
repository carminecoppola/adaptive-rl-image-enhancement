# Adaptive RL Image Enhancement

Il branch è ora `underwater-first`: la pipeline canonica è quella su UIEB con action set `underwater_curated_v1`, mantenendo compatibile il framework generale CIFAR/STL per confronto e regressioni.

Il riferimento canonico sullo stato reale è [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md). Il workflow operativo ufficiale è descritto in [scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md).

## Workflow canonico

- training: `src/training/train.py`
- evaluation ID paired: `src/evaluation/evaluation_dqn_baselines.py`
- analisi azioni: `src/evaluation/analyze_dqn_actions.py`
- evaluation OOD challenging-60: `src/evaluation/evaluate_underwater_ood.py`
- report finale di run: `src/evaluation/generate_underwater_report.py`
- launcher SLURM ufficiale: `scripts/train_underwater.sbatch`
- notebook canonico: `notebooks/underwater_policy_analysis.ipynb`

## Setup rapido

```bash
bash scripts/setup_env.sh
cp .env.example .env
source venv/bin/activate
```

Configura poi `.env` con i path HPC/locali, inclusi `DATASET_ROOT`, `UIEB_ROOT`, `LOGS_ROOT` e `CHECKPOINT_ROOT`.

## Training underwater

Smoke test locale:

```bash
python src/training/train.py --experiment underwater_dqn_v1 --phase smoke_test
```

Training completo locale:

```bash
python src/training/train.py --experiment underwater_dqn_v1 --phase full_training
```

Training via Slurm:

```bash
sbatch scripts/train_underwater.sbatch
```

Lo script SBATCH esegue automaticamente:

1. smoke training ridotto
2. evaluation minima per sbloccare il full training
3. full training
4. evaluation ID best/final
5. evaluation OOD challenging-60
6. report canonico della run

## Evaluation manuale

Checkpoint best:

```bash
python src/evaluation/analyze_dqn_actions.py --checkpoint /path/to/dqn_best_policy_net.pt --num-images 50
python src/evaluation/evaluation_dqn_baselines.py --checkpoint /path/to/dqn_best_policy_net.pt --num-images 50
python src/evaluation/evaluate_underwater_ood.py --checkpoint /path/to/dqn_best_policy_net.pt
```

## Framework generale

I launcher `scripts/train.sh`, `scripts/train.sbatch` e `scripts/evaluate.sh` restano disponibili per gli esperimenti generali (`phase_a_sanity`, `phase_b_stability`, `phase_c_final`), ma non sono più il percorso principale del branch.

## Test

```bash
pytest tests/ -v --tb=short
```
