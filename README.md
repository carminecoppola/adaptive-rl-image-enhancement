# Adaptive RL Image Enhancement

Progetto di miglioramento di immagini underwater con reinforcement learning su UIEB.

Il progetto attuale è una **reimplementazione da zero**, ispirata al lavoro dell'Università di Bologna del 2022, ma con pipeline, training, evaluation e report riscritti in modo modulare e riproducibile. Il riferimento concettuale è Bologna; il codice **non** è un fork del loro notebook.

## In breve

- task: enhancement di immagini underwater degradate
- dataset canonico: `UIEB`
- agente usato: **DDQN**
- architettura corrente: `DQNAgent` con `use_double_dqn=true` e `use_dueling_dqn=false`
- action set canonico: `underwater_curated_v1`
- notebook finale: `underwater_policy_analysis.ipynb`

## Cosa fa il sistema

Il flusso è questo:

1. prende un'immagine underwater degradata in input
2. applica una piccola azione discreta
3. misura se l'immagine è migliorata rispetto alla reference
4. ripete per pochi step
5. si ferma con l'azione `stop`

L'action set canonico corrente è volutamente piccolo e leggibile:

- `white_balance`
- `contrast_up`
- `sharpen`
- `stop`

## DDQN o DQN?

Usiamo **Double DQN (DDQN)**.

- la classe si chiama `DQNAgent`, ma l'algoritmo attivo è DDQN
- conferma nel codice: [src/agents/dqn_agent.py](/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement/src/agents/dqn_agent.py)
- conferma nella config ufficiale: [configs/experiments/underwater_dqn_v1.yaml](/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement/configs/experiments/underwater_dqn_v1.yaml)

Non stiamo usando la variante dueling in questo setup ufficiale.

## Bologna, fisica dell'acqua e cosa abbiamo riusato

Il progetto parte dal lavoro di Bologna come ispirazione metodologica:

- stessa idea generale: RL per migliorare immagini underwater
- stesso obiettivo: imparare una sequenza di azioni migliorative
- confronto esplicito nei report finali

Non stiamo usando il loro codice. La codebase qui è stata riscritta con:

- environment dedicato
- training pipeline separata
- evaluation ID e OOD separate
- report automatici
- notebook finale coerente con gli artifact

Sulla “fisica dell'acqua”:

- **non** stiamo usando un modello fisico completo della propagazione della luce in acqua nel reward o nello stato
- esistono però operatori e baseline **physics-inspired**, per esempio `DCP` (Dark Channel Prior), che è un metodo classico di dehazing
- nel workflow canonico attuale il policy ufficiale usa il set curato a 4 azioni, quindi DCP non è nel loop principale della policy finale

## Setup

### 1. Crea l'ambiente Python

```bash
bash scripts/setup_env.sh
source venv/bin/activate
```

### 2. Configura i path locali

Il file `.env.example` **serve** e va tenuto: è il template minimo per configurare il progetto.

```bash
cp .env.example .env
```

Poi aggiorna almeno queste variabili:

- `DATASET_ROOT`
- `UIEB_ROOT`
- `LOGS_ROOT`
- `CHECKPOINT_ROOT`

La struttura attesa di UIEB è:

```text
UIEB/
  raw/
  reference/
  challenging-60/
```

## Esecuzione canonica

### Training completo locale

```bash
python src/training/train.py --experiment underwater_dqn_v1 --phase full_training
```

### Training completo via Slurm

```bash
sbatch scripts/train_underwater.sbatch
```

## Evaluation manuale

Con un checkpoint best:

```bash
python src/evaluation/analyze_dqn_actions.py --checkpoint /path/to/dqn_best_policy_net.pt --num-images 50 --output-name action_analysis_best.json
python src/evaluation/evaluation_dqn_baselines.py --checkpoint /path/to/dqn_best_policy_net.pt --num-images 50 --output-name evaluation_baselines_best.json --action-analysis-file action_analysis_best.json
python src/evaluation/evaluate_underwater_ood.py --checkpoint /path/to/dqn_best_policy_net.pt --output-name evaluation_ood_challenging60.json
```

Per generare il report finale di una run:

```bash
python src/evaluation/generate_underwater_report.py --run-dir /path/to/logs/dqn/<RUN_ID>
```

## Dove sono i pesi del modello

Percorso canonico:

- `${CHECKPOINT_ROOT}/dqn/<RUN_ID>/dqn_best_policy_net.pt`
- `${CHECKPOINT_ROOT}/dqn/<RUN_ID>/dqn_final_policy_net.pt`

Nel repository locale esistono anche esempi sotto:

- `checkpoints/dqn/dqn_best_policy_net.pt`
- `checkpoints/dqn/dqn_final_policy_net.pt`

## Dove leggere i risultati

- stato del progetto: [docs/CURRENT_STATE.md](/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement/docs/CURRENT_STATE.md)
- guida operativa: [scripts/TRAINING_GUIDE.md](/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement/scripts/TRAINING_GUIDE.md)
- risultati underwater: [docs/underwater_results.md](/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement/docs/underwater_results.md)
- notebook finale: [underwater_policy_analysis.ipynb](/home/ccoppola/projects/ml2-project/adaptive-rl-image-enhancement/underwater_policy_analysis.ipynb)

## Test

```bash
pytest tests/ -v --tb=short
```
