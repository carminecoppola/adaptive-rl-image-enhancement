# Underwater RL Results

Questo file è il riferimento canonico per i risultati underwater consolidati.

## Stato

- Il workflow ufficiale è stato consolidato.
- Le run storiche underwater restano utili come riferimento esplorativo, ma non sono considerate definitive.
- La prossima run eseguita con `scripts/train_underwater.sbatch` sarà la prima run da considerare ufficiale nel nuovo protocollo.

## Protocollo canonico

Una run consolidata deve includere:

1. smoke training
2. smoke evaluation minima
3. full training
4. baseline evaluation su best checkpoint
5. baseline evaluation su final checkpoint
6. OOD evaluation su `challenging-60`
7. report canonico della run

## Dove leggere i risultati della run

Per ogni run ufficiale:

- report markdown: `${LOGS_ROOT}/dqn/<RUN_ID>/underwater_results.md`
- summary JSON: `${LOGS_ROOT}/dqn/<RUN_ID>/underwater_results_summary.json`
- notebook di analisi: `notebooks/underwater_policy_analysis.ipynb`

## Nota metodologica

- I risultati ID paired sono riportati come delta PSNR / delta SSIM rispetto all’input degradato.
- I risultati OOD `challenging-60` non hanno reference e sono quindi riportati con metriche no-reference.
- Il confronto con Bologna va letto con cautela perché Bologna riporta metriche assolute, mentre il nostro workflow usa soprattutto metriche differenziali.
