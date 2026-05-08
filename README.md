# Adaptive RL Image Enhancement

Questo branch va considerato come una base di lavoro per i prossimi esperimenti RL, non come progetto “chiuso”. Il riferimento canonico sullo stato reale è [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md).

## Punto di ripartenza

- La pipeline end-to-end esiste: training, analisi azioni, evaluation contro baseline.
- Il gate di accettazione è già stato irrigidito e va mantenuto.
- L’ultimo punto verificato è una run quality-positive ma non ancora pienamente accettata per `stop_rate` insufficiente.
- La prossima iterazione deve concentrarsi su stabilità del training, reward tuning e validazione OOD.

## File da usare

- `src/training/train.py`: entrypoint reale del training
- `src/evaluation/evaluation_dqn_baselines.py`: evaluation di una run/checkpoint
- `scripts/train.sh`: launcher locale minimale
- `scripts/train.sbatch`: launcher Slurm minimale
- `docs/CURRENT_STATE.md`: stato corrente e prossime fasi

## Setup rapido

```bash
bash scripts/setup_env.sh
cp .env.example .env
source venv/bin/activate
```

Configura poi `.env` con i path HPC/locali che stai usando.

## Esecuzione

Training locale:

```bash
bash scripts/train.sh phase_a_sanity
```

Training via Slurm:

```bash
sbatch scripts/train.sbatch phase_a_sanity
```

Evaluation dell’ultimo checkpoint disponibile:

```bash
bash scripts/evaluate.sh
```

Evaluation di un checkpoint specifico:

```bash
bash scripts/evaluate.sh /path/to/dqn_best_policy_net.pt
```

Confronto visuale / baseline:

```bash
python scripts/compare_experiment_results.py --dataset CIFAR10 --num-samples 50
```

## Esperimenti disponibili

- `phase_a_sanity`
- `phase_b_stability`
- `phase_c_final`

Le config sono in `configs/experiments/`.

## Test

```bash
pytest tests/ -v --tb=short
```
