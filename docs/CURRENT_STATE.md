# Current State

Data di riferimento: `2026-05-10`

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

Data: `2026-05-10`
Run ID: `dqn_underwater_full_20260510_095530_1489`

Modifiche applicate:
- Fix 1: feature cromatiche LAB aggiunte all'osservazione come canale globale extra
- Fix 2: `psnr_weight` e `ssim_weight` ora sono letti dalla config YAML e passati davvero all'environment
- Fix 3: action set canonico passato da `underwater_curated_v1` (4 azioni) a `underwater_extended_v1` (8 azioni) con `max_steps=5`

Risultati ID:
- best checkpoint (`ep 380`): `mean_delta_psnr = +0.6365`, `output_psnr = 17.80`, `output_ssim = 0.803`, `acceptance = false`
- final checkpoint: `mean_delta_psnr = +0.4076`, `output_psnr = 17.57`, `output_ssim = 0.791`, `acceptance = true`

Risultati OOD:
- `mean_delta_uciqe = -0.4531`
- `mean_delta_uiqm_proxy = -0.0178`

Confronto con run precedente:
- rispetto alla baseline `dqn_underwater_full_20260508_184539_1488`, la run 1489 peggiora chiaramente su ID (`+0.6365` vs `+1.0910`) e su OOD (`-0.4531` vs `-0.2440`)
- il recovery del post-processing e' stato completato senza rifare il training, correggendo il wiring degli script di evaluation verso la firma corrente di `build_env_for_image(...)`
- questa run resta utile come evidenza di regressione e per confronto ablation, ma non va promossa a baseline ufficiale

## Reward consistency

- Il reward combinato usa ora in modo coerente `psnr_weight` e `ssim_weight` in training, helper di costruzione env ed evaluation.
- Il fallback di retrocompatibilita' resta `psnr_weight=1.0`, `ssim_weight=10.0`, che replica il comportamento storico della baseline buona.
- Le metriche e i gate delle evaluation usano ora gli stessi pesi letti dal config effettivo della run, evitando divergenze tra training e analisi offline.

## Baseline reset v3.0

- `configs/experiments/underwater_dqn_v1.yaml` e' stato riallineato alla baseline di riferimento `dqn_underwater_full_20260508_184539_1488`.
- Il workflow canonico torna a usare:
  - `max_steps=3`
  - `underwater_curated_v1`
  - `psnr_weight=1.0`
  - `ssim_weight=10.0`
  - `dominant_action_threshold=0.50`
- Il supporto a `include_lab_stats` e `underwater_extended_v1` resta nel codice, ma non fa piu' parte della configurazione ufficiale finche' non vince un'ablation controllata.

## Bologna reporting

- Il report canonico underwater ora deve esporre sia le metriche delta del framework sia le metriche assolute `output_psnr` e `output_ssim`.
- La sezione Bologna del report viene usata per confrontare:
  - PSNR assoluto in output
  - SSIM assoluto in output
  - numero di episodi
  - dimensione dell'action space
  - dataset di riferimento
- La nota metodologica resta obbligatoria: il confronto con Bologna e' indicativo e non perfettamente apples-to-apples.
