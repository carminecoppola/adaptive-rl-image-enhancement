# Underwater RL Results Audit

Data di audit: `2026-05-08`

## Sintesi

La run underwater ha mostrato un segnale promettente, ma il confronto con Bologna non e ancora metodologicamente chiuso.
Il picco `+5.60 dB` esiste nei log, pero e stato selezionato su un subset di sole `20` immagini che cambiava ad ogni evaluation checkpoint.
Questo rende il numero utile come indicazione esplorativa, non ancora come risultato finale forte.

## Cosa emerge dai log

- Run finale: `dqn_20260508_155835_1483`
- Best checkpoint: episodio `3780`
- Best `mean_delta_psnr`: `+5.5988 dB`
- Ultima evaluation disponibile: episodio `5000`
- `mean_delta_psnr` finale: `+3.0784 dB`
- `best_eval_subset_size`: `20`
- Eval history totale: `500` snapshot

## Punti forti

- Il training ha imparato davvero qualcosa: il delta PSNR medio finale e positivo.
- La policy usa `STOP` in modo non banale nella parte finale del training.
- La dominante di una singola azione non e estrema nei checkpoint migliori.
- Anche la performance finale stabile (`+3.08 dB`) rimane interessante per un agente RL interpretabile.

## Debolezze metodologiche rispetto a Bologna

- Il best checkpoint non e selezionato su un validation set fisso.
- Il subset di evaluation cambia ad ogni evaluation (`20` immagini campionate dalla pool).
- La documentazione interna parla di smoke test da `1000` episodi, ma gli artifact mostrano `5000` episodi anche per quella run.
- Lo script `train_underwater.sbatch` lanciava due volte lo stesso comando senza applicare davvero `smoke_test` e `full_training`.
- Gli artifact finali underwater non includono ancora `evaluation_baselines.json`, quindi il claim "beat all baselines" non e ancora dimostrato per questa run.

## Implicazioni sul confronto con Bologna

Il progetto e probabilmente gia migliore di Bologna sul piano ingegneristico e, verosimilmente, competitivo anche sul piano qualitativo.
Pero oggi il confronto corretto da pubblicare e:

- `+5.60 dB`: picco esplorativo, non ancora validato in modo robusto
- `+3.08 dB`: performance finale piu credibile della run corrente
- `beat all baselines`: non ancora verificato con artifact underwater finali

## Miglioramenti prioritari

1. Usare un eval subset fisso per tutta la run.
2. Applicare davvero le phase override `smoke_test` e `full_training`.
3. Rieseguire evaluation baseline sul best checkpoint e sul final checkpoint.
4. Eseguire la validazione OOD sulle `challenging-60`.
5. Distinguere nei report tra `best-on-tracking-set` e `final-stable`.

## Stato dopo le correzioni in codice

Questo branch ora include:

- supporto a `--phase` in `src/training/train.py`
- phase override applicata davvero alla config sperimentale
- eval tracking subset fisso durante il training
- aggiornamento dello script SLURM underwater per lanciare smoke e full con la fase corretta

## Prossimo esperimento consigliato

1. `--phase smoke_test` con tracking subset fisso
2. `--phase full_training` con artifact completi
3. `evaluation_dqn_baselines.py` sul best checkpoint
4. evaluation OOD sulle `challenging-60`

Solo dopo questi quattro step il confronto con Bologna diventa forte anche sul piano sperimentale.
