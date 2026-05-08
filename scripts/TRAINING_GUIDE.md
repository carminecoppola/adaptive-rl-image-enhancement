# Underwater RL Training Guide

## Utilizzo

```bash
cd /home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement
source venv/bin/activate
sbatch scripts/train_underwater.sbatch
```

Monitoraggio:

```bash
squeue -u ccoppola
tail -f logs/sbatch_training_*.log
```

## Cosa fa il launcher ufficiale

`scripts/train_underwater.sbatch` esegue automaticamente:

1. smoke training con `--phase smoke_test`
2. smoke evaluation minima con `action_analysis.json` e `evaluation_baselines.json`
3. gate minimo smoke:
   - `mean_delta_psnr_positive`
   - artifact di action analysis presenti
4. full training con `--phase full_training`
5. evaluation ID su best checkpoint
6. evaluation ID su final checkpoint
7. evaluation OOD su `UIEB/challenging-60`
8. report canonico della run

## Artifact attesi per la run full

Nella cartella `${LOGS_ROOT}/dqn/<RUN_ID>/` devono comparire:

- `effective_config.json`
- `dataset_split.json`
- `episode_summary.csv`
- `eval_summary.json`
- `run_meta.json`
- `action_analysis_best.json`
- `action_analysis_final.json`
- `evaluation_baselines_best.json`
- `evaluation_baselines_final.json`
- `evaluation_ood_challenging60.json`
- `underwater_results.md`
- `underwater_results_summary.json`

Checkpoint attesi in `${CHECKPOINT_ROOT}/dqn/<RUN_ID>/`:

- `dqn_best_policy_net.pt`
- `dqn_final_policy_net.pt`

## Esecuzione manuale

Smoke locale:

```bash
python src/training/train.py --experiment underwater_dqn_v1 --phase smoke_test
```

Full locale:

```bash
python src/training/train.py --experiment underwater_dqn_v1 --phase full_training
```

Evaluation manuale di un checkpoint:

```bash
python src/evaluation/analyze_dqn_actions.py --checkpoint /path/to/dqn_best_policy_net.pt --num-images 50
python src/evaluation/evaluation_dqn_baselines.py --checkpoint /path/to/dqn_best_policy_net.pt --num-images 50
python src/evaluation/evaluate_underwater_ood.py --checkpoint /path/to/dqn_best_policy_net.pt
```

## Configurazione canonica

La sola config underwater ufficiale è:

- `configs/experiments/underwater_dqn_v1.yaml`

Le config e i launcher storici ridondanti sono stati spostati in `archive/`.

## Notebook finale

Il notebook canonico da usare per l’analisi della run è:

- `notebooks/underwater_policy_analysis.ipynb`

Il notebook deve leggere gli artifact della run, non ricostruire risultati con logiche parallele.
