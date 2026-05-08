# Current State

Data di riferimento: `2026-05-07`

Questo file è il riferimento canonico del branch per capire da dove ripartire.

## Dove siamo

- La pipeline RL end-to-end è disponibile:
  - training: `src/training/train.py`
  - analisi azioni: `src/evaluation/analyze_dqn_actions.py`
  - evaluation baseline/gate: `src/evaluation/evaluation_dqn_baselines.py`
- Gli artifact di run vengono prodotti sotto `${LOGS_ROOT}/dqn/<RUN_ID>/`.
- La selezione checkpoint è PSNR-first:
  - metrica primaria: `mean_delta_psnr`
  - tie-break: `mean_eval_reward`
- La riproducibilità è stata migliorata con split deterministico, eval subset salvato e `effective_config.json`.

## Fix già consolidati

- Gate di accettazione rigido in evaluation.
- Allineamento fra training ed evaluation su degradazione e observation channels.
- Gestione image-size guidata dalla config dataset.
- Config `configs/dataset_stl10_safe.yaml` aggiunta e validata come smoke test.

## Gate di accettazione attuale

Una run è considerata valida solo se passa tutti questi controlli:

- `dominant_action_share <= threshold`
- `stop_rate >= min_stop_rate`
- `mean_delta_psnr > 0`
- `output_psnr >= input_psnr`
- `action_analysis.json` presente

L’esito finale viene scritto in `evaluation_baselines.json` con `acceptance_checks` e `acceptance_passed`.

## Ultimo punto verificato

Batch Phase 3A con tuning reward e `use_double_dqn=true`:

- `dqn_phase3a_control_20260506_122118`
- `dqn_phase3a_treatment_20260506_123200`
- `dqn_phase3a_treatment2_20260506_124100`

Trend osservato:

- `stop_rate`: `0.024 -> 0.041 -> 0.055`
- `mean_delta_psnr`: `-1.512 -> -0.990 -> +0.703`

Interpretazione:

- Il tuning reward sta andando nella direzione giusta.
- L’ultima run è quality-positive e leggibile.
- La run non passa ancora il gate completo per `stop_rate` troppo basso rispetto alla soglia `0.10`.

## Evidenza OOD

Sulla run `dqn_phase3a_treatment2_20260506_124100`:

- ID: `mean_delta_psnr = +0.7033`
- ID: `stop_rate = 0.0546`
- OOD con rumore più forte: `mean_delta_psnr = -1.9320`

Conclusione:

- Il comportamento in-distribution è promettente.
- La generalizzazione fuori distribuzione non è ancora robusta.

## Prossime fasi

1. Phase A: short sanity runs con la configurazione aggiornata.
2. Phase B: medium stability runs per verificare robustezza del comportamento.
3. Phase C: long final runs.
4. Selezione del modello finale solo tra le run che passano l’intero gate.

## Priorità tecnica

- Migliorare il comportamento di `STOP`.
- Continuare il reward tuning senza allentare il gate.
- Misurare meglio stabilità tra run.
- Eseguire validazione OOD controllata.

## File da considerare ufficiali

- `README.md`
- `docs/CURRENT_STATE.md`
- `scripts/train.sh`
- `scripts/train.sbatch`
- `scripts/evaluate.sh`
- `configs/experiments/phase_a_sanity.yaml`
- `configs/experiments/phase_b_stability.yaml`
- `configs/experiments/phase_c_final.yaml`
