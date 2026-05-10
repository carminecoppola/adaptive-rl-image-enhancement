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

## Baseline run v3.0 confirmed

- Run ID: `dqn_underwater_full_20260510_104821_1490`
- Best checkpoint:
  - episode `430`
  - `mean_delta_psnr = +1.5089`
  - `output_psnr = 18.6754`
  - `output_ssim = 0.8318`
  - `acceptance = true`
- Final checkpoint:
  - `mean_delta_psnr = +0.7461`
  - `output_psnr = 17.9126`
  - `output_ssim = 0.8032`
  - `acceptance = true`
- OOD challenging-60:
  - `mean_delta_uciqe = -0.2504`
  - `mean_delta_uiqm_proxy = -0.0161`
- Stato:
  - la baseline v3.0 e' confermata
  - la run supera Bologna sul best checkpoint sia in PSNR assoluto sia in SSIM assoluto
  - le future ablation OOD vanno confrontate contro `1490`, usando il best checkpoint come metrica primaria

## Ablation A status

- Config sperimentale preparata: `configs/experiments/ablation_A_max_steps5.yaml`
- Modifica isolata rispetto alla baseline `1490`:
  - `max_steps: 3 -> 5`
- Supporto multi-experiment abilitato in `train.py`
- `scripts/train_underwater.sbatch` ora accetta `EXPERIMENT` come override
- Run lanciata:
  - job Slurm: `1491`
  - experiment: `ablation_A_max_steps5`
- Risultati di A:
  - run id: `dqn_underwater_full_20260510_111515_1491`
  - best checkpoint (`ep 1730`):
    - `mean_delta_psnr = +1.5622`
    - `output_psnr = 18.7287`
    - `output_ssim = 0.8330`
    - `acceptance = true`
  - OOD:
    - `mean_delta_uciqe = -0.2600`
    - `mean_delta_uiqm_proxy = -0.0144`
- Confronto con baseline `1490`:
  - ID best migliora (`+1.5622` vs `+1.5089`)
  - `output_psnr` e `output_ssim` migliorano leggermente
  - `mean_delta_uciqe` peggiora leggermente (`-0.2600` vs `-0.2504`)
  - `mean_delta_uiqm_proxy` migliora (`-0.0144` vs `-0.0161`)
- Decisione:
  - Ablation A passa il gate del piano
  - `max_steps=5` diventa il nuovo riferimento per l'Ablation B

## Ablation B status

- Config sperimentale preparata: `configs/experiments/ablation_B_extended_actions.yaml`
- Modifica isolata rispetto al riferimento A:
  - `action_set: underwater_curated_v1 -> underwater_extended_v1`
  - `max_steps=5` mantenuto
- Risultati di B:
  - run id: `dqn_underwater_full_20260510_155811_1492`
  - best checkpoint (`ep 1070`):
    - `mean_delta_psnr = +0.8762`
    - `output_psnr = 18.0427`
    - `output_ssim = 0.8145`
    - `acceptance = true`
  - OOD:
    - `mean_delta_uciqe = -0.1677`
    - `mean_delta_uiqm_proxy = -0.0095`
- Confronto con riferimento A (`1491`):
  - OOD migliora nettamente
  - ID peggiora materialmente (`+0.8762` vs `+1.5622`)
  - il best checkpoint scende sotto il guardrail `+1.20`
- Decisione:
  - Ablation B non viene promossa a configurazione finale
  - l'action set esteso resta utile come evidenza che migliora OOD, ma non passa il tradeoff ID/OOD richiesto
  - per Ablation C il riferimento resta la configurazione di A: `max_steps=5`, `underwater_curated_v1`

## Ablation C status

- Config sperimentale preparata: `configs/experiments/ablation_C_lab_stats.yaml`
- Modifica isolata rispetto al riferimento A:
  - `include_lab_stats: false -> true`
  - `max_steps=5`, `underwater_curated_v1`, reward weights e seed invariati
  - `num_episodes=7000` per accomodare il cambio di observation space
- Verifica preliminare:
  - env con LAB: `channels = 5`
  - action space invariato: `num_actions = 4`
- Run lanciata:
  - job Slurm: `1493`
  - experiment: `ablation_C_lab_stats`
- Risultati di C:
  - run id: `dqn_underwater_full_20260510_162433_1493`
  - best checkpoint (`ep 1650`):
    - `mean_delta_psnr = +1.5573`
    - `output_psnr = 18.7238`
    - `output_ssim = 0.8365`
    - `acceptance = true`
  - final checkpoint:
    - `mean_delta_psnr = +1.3036`
    - `output_psnr = 18.4700`
    - `output_ssim = 0.8149`
    - `acceptance = true`
  - OOD:
    - `mean_delta_uciqe = -0.3724`
    - `mean_delta_uiqm_proxy = -0.0175`
- Confronto con riferimento A (`1491`):
  - ID best resta sostanzialmente invariato (`+1.5573` vs `+1.5622`)
  - `output_ssim` migliora leggermente
  - OOD peggiora in modo netto su entrambe le metriche (`-0.3724` vs `-0.2600`, `-0.0175` vs `-0.0144`)
- Decisione:
  - Ablation C non viene promossa
  - il canale LAB globale non migliora la generalizzazione OOD in questa forma

## Configurazione finale selezionata

- Config vincente: Ablation A
  - `max_steps = 5`
  - `action_set = underwater_curated_v1`
  - `include_lab_stats = false`
  - `psnr_weight = 1.0`
  - `ssim_weight = 10.0`
- Motivazione:
  - migliora il best checkpoint ID rispetto alla baseline `1490`
  - mantiene il workflow semplice e interpretabile
  - non introduce la regressione ID vista con l'action set esteso
  - evita il peggioramento OOD marcato visto con il canale LAB
- Stato:
  - `configs/experiments/underwater_dqn_v1.yaml` promosso a `v4.0`
  - la run ufficiale finale va lanciata con questa configurazione
