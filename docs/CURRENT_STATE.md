# Current State — Versione finale

Data di riferimento: `2026-05-10`
Branch: `feature/underwater-domain`

## Configurazione finale ufficiale

- Experiment: `underwater_dqn_v1 v4.0`
- Agente: DDQN (`use_double_dqn: true`)
- Dataset: UIEB 890 paired (raw-890 + reference-890)
- Action set: `underwater_curated_v1` (4 azioni: white_balance, contrast_up, sharpen, stop)
- Max steps per episodio: 5
- Reward: combined PSNR + SSIM (`psnr_weight=1.0`, `ssim_weight=10.0`)
- Episodi training: 5000
- Best checkpoint: episodio 1540

## Run ufficiale finale

Run ID: `dqn_underwater_full_20260510_165955_1494`

| Metrica | Valore |
|---------|--------|
| mean_delta_psnr (ID) | +1.5492 dB |
| output_psnr (ID) | 18.7157 |
| output_ssim (ID) | 0.8275 |
| acceptance_passed | true |
| mean_delta_uciqe (OOD) | -0.1707 |
| mean_delta_uiqm_proxy (OOD) | -0.0119 |

## Confronto con Bologna 2022

| | Bologna 2022 | Questo progetto |
|---|---|---|
| PSNR | 15.47 dB | **18.72 dB** (+3.25 dB) |
| SSIM | 0.628 | **0.827** (+0.199) |
| Episodi | 20.000 | **5.000** |
| Azioni | 20 | **4** |
| OOD testing | No | Sì |

## Pipeline canonica

1. `src/training/train.py --experiment underwater_dqn_v1 --phase full_training`
2. `src/evaluation/analyze_dqn_actions.py`
3. `src/evaluation/evaluation_dqn_baselines.py`
4. `src/evaluation/evaluate_underwater_ood.py`
5. `src/evaluation/generate_underwater_report.py`

Tutti e 5 i passi sono eseguiti automaticamente da `scripts/train_underwater.sbatch`.

## Limiti documentati

- OOD (challenging-60): mean_delta_uciqe leggermente negativo (-0.17).
  Causa: distribuzione di degrado delle challenging-60 diversa da raw-890.
  Ablation B (8 azioni) e C (LAB stats) non hanno migliorato questo limite
  senza degradare i risultati ID.
