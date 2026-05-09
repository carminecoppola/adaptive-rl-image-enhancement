# Underwater RL Results

Questo file è il riferimento canonico per i risultati underwater consolidati.

## Stato

- Il workflow ufficiale è stato consolidato.
- Le run storiche underwater restano utili come riferimento esplorativo, ma non sono considerate definitive.
- La run di riferimento va letta dagli artifact prodotti in `${LOGS_ROOT}/dqn/<RUN_ID>/`.

## Protocollo canonico

Una run consolidata deve includere:

1. full training
2. baseline evaluation su best checkpoint
3. baseline evaluation su final checkpoint
4. OOD evaluation su `challenging-60`
5. report canonico della run

## Dove leggere i risultati della run

Per ogni run ufficiale:

- report markdown: `${LOGS_ROOT}/dqn/<RUN_ID>/underwater_results.md`
- summary JSON: `${LOGS_ROOT}/dqn/<RUN_ID>/underwater_results_summary.json`
- notebook di analisi: `underwater_policy_analysis.ipynb`

## Nota metodologica

- I risultati ID paired sono riportati come delta PSNR / delta SSIM rispetto all’input degradato.
- I risultati OOD `challenging-60` non hanno reference e sono quindi riportati con metriche no-reference.
- Il confronto con Bologna va letto con cautela perché Bologna riporta metriche assolute, mentre il nostro workflow usa soprattutto metriche differenziali.

## Template nuova run post-fix OOD

- Data: `2026-05-09`
- Run ID: `<da compilare>`
- Config: `configs/experiments/underwater_dqn_v1.yaml`
- Modifiche attive:
  - Fix 1: canale LAB globale nell'osservazione
  - Fix 2: reward combinato con `psnr_weight=1.0` e `ssim_weight=5.0`
  - Fix 3: action set `underwater_extended_v1`, `max_steps=5`
- Risultati ID attesi da compilare:
  - `mean_delta_psnr`: `<...>`
  - `output_psnr`: `<...>`
  - `output_ssim`: `<...>`
  - `stop_rate`: `<...>`
  - `dominant_action_share`: `<...>`
- Risultati OOD attesi da compilare:
  - `mean_delta_uciqe`: `<...>`
  - `mean_delta_uiqm_proxy`: `<...>`
- Confronto con run precedente:
  - `delta mean_delta_psnr`: `<...>`
  - `delta mean_delta_uciqe`: `<...>`
  - `delta mean_delta_uiqm_proxy`: `<...>`
