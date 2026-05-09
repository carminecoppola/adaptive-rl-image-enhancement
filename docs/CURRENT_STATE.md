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
  - override di fase `full_training`

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

## Workflow ufficiale

La run ufficiale va prodotta da:

```bash
sbatch scripts/train_underwater.sbatch
```

e deve completare:

1. full training
2. baseline evaluation su best checkpoint
3. baseline evaluation su final checkpoint
4. OOD evaluation su `challenging-60`
5. report canonico della run

## File da considerare ufficiali

- `README.md`
- `docs/CURRENT_STATE.md`
- `scripts/TRAINING_GUIDE.md`
- `scripts/train_underwater.sbatch`
- `configs/experiments/underwater_dqn_v1.yaml`
- `underwater_policy_analysis.ipynb`

## OOD improvements applied

Data: `2026-05-09`
Run ID: `<da compilare dopo la nuova run>`

Modifiche applicate:
- Fix 1: feature cromatiche LAB aggiunte all'osservazione come canale globale extra
- Fix 2: `psnr_weight` e `ssim_weight` ora sono letti dalla config YAML e passati davvero all'environment
- Fix 3: action set canonico passato da `underwater_curated_v1` (4 azioni) a `underwater_extended_v1` (8 azioni) con `max_steps=5`

Risultati ID: `<compilare da evaluation_baselines_best.json>`
Risultati OOD: `<compilare da evaluation_ood_challenging60.json>`
Confronto con run precedente: `<delta da compilare dopo la nuova run>`
