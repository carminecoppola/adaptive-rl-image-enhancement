# Current State

Data di riferimento: `2026-05-08`

Questo file è il riferimento canonico del branch per capire da dove ripartire.

## Dove siamo

- Il branch è `underwater-first`.
- La pipeline canonica è ora basata su:
  - `src/training/train.py`
  - `src/evaluation/analyze_dqn_actions.py`
  - `src/evaluation/evaluation_dqn_baselines.py`
  - `src/evaluation/evaluate_underwater_ood.py`
  - `src/evaluation/generate_underwater_report.py`
- Gli artifact di run vengono prodotti sotto `${LOGS_ROOT}/dqn/<RUN_ID>/`.
- La selezione checkpoint è PSNR-first su subset fisso:
  - metrica primaria: `mean_delta_psnr`
  - tie-break: `mean_eval_reward`
- Le run paired UIEB usano ora davvero:
  - dataset paired degradato + reference
  - action set `underwater_curated_v1`
  - override di fase `smoke_test` / `full_training`

## Consolidamenti già applicati

- Cleanup soft del repository con archivio dei documenti/script/config legacy.
- Unificazione del naming storage su `CHECKPOINT_ROOT`.
- Evaluation e action analysis allineate alla `effective_config.json` della run.
- Tracking subset fisso durante il training per la scelta del best checkpoint.
- Supporto reale agli action set `underwater_v1` e `underwater_curated_v1` nell’environment.
- Riduzione dell’action space canonico a una variante curated che mantiene le trasformazioni più utili su UIEB.
- Gestione paired UIEB corretta: input degradato e target di riferimento non vengono più confusi.

## Gate di accettazione attuale

Una run underwater viene considerata valida solo se produce:

- `effective_config.json`
- `dataset_split.json`
- `eval_summary.json`
- `action_analysis*.json`
- `evaluation_baselines*.json`
- checkpoint `best` e `final`

e se passa i controlli di evaluation:

- `mean_delta_psnr > 0`
- `output_psnr >= input_psnr`
- `dominant_action_share <= threshold`
- `stop_rate >= min_stop_rate`
- `action_analysis` disponibile

## Stato risultati

- Le vecchie run underwater esistono come riferimento storico, ma non vanno considerate consolidate per il nuovo workflow.
- La ragione è metodologica:
  - subset di selezione best non fisso nelle run precedenti
  - phase override non applicate correttamente
  - evaluation underwater non allineata alla config effettiva della run
  - integrazione paired UIEB e action set underwater corretta solo dopo questo consolidamento
  - action space precedente troppo ampio e popolato da operatori sistematicamente peggiorativi per il target PSNR-first

## Prossima verifica ufficiale

La prossima run da considerare ufficiale dovrà essere prodotta da:

```bash
sbatch scripts/train_underwater.sbatch
```

e dovrà completare:

1. smoke training
2. smoke evaluation minima
3. full training
4. baseline evaluation su best checkpoint
5. baseline evaluation su final checkpoint
6. OOD evaluation su `challenging-60`
7. report canonico della run

## File da considerare ufficiali

- `README.md`
- `docs/CURRENT_STATE.md`
- `scripts/TRAINING_GUIDE.md`
- `scripts/train_underwater.sbatch`
- `configs/experiments/underwater_dqn_v1.yaml`
- `notebooks/underwater_policy_analysis.ipynb`
