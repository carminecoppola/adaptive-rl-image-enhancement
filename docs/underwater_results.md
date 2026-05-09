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
