# MKAN — Mobile Money KAN Fraud Detector

Implémentation PyTorch du modèle **M-KAN** pour la détection de fraude Mobile Money en environnement CEMAC/COBAC, développé dans le cadre d'un mémoire de Master.

## Architecture

```
MKANScorer
└── TKANCell  (cellule récurrente LSTM-KAN, eq. 4.7–4.12)
    ├── forget_gate    ─┐
    ├── input_gate     ─┤  HybridKANLayer (eq. 4.13–4.15)
    ├── candidate_gate ─┤  = Gaussiennes FastKAN + séries de Fourier KAN-AD
    └── output_gate    ─┘
```

La fonction d'arête hybride (eq. 4.13) combine :
- **Composante Gaussienne** (FastKAN) : `Σ wₘ · exp(-(x-μₘ)²/2h²)`
- **Composante Fourier** (KAN-AD) : `Σ [aₖ cos(kx) + bₖ sin(kx)]`

## Modules

| Fichier | Rôle |
|---|---|
| `hybrid_edge.py` | Fonction d'arête hybride (eq. 4.13) |
| `hybrid_layer.py` | Couche KAN vectorisée + nœuds MultKAN + extension grille |
| `cell.py` | Cellule T-KAN + MKANScorer |
| `loss.py` | Perte totale : BCE pondérée + L1 + entropie (eq. 4.27) |
| `drift.py` | Détection de dérive JS + localisation région (eq. 4.19–4.20) |
| `symbolic.py` | Régression symbolique sur les courbes d'arêtes (eq. 2.23) |
| `audit.py` | Élagage + rapport d'audit COBAC (eq. 4.28, section 4.4.7) |
| `visualizer.py` | 7 visualisations Plotly : loss, métriques, heatmap L1, courbes φ_ij, élagage |

## Installation

```bash
pip install torch numpy pandas pyarrow plotly
```

## Usage minimal

```python
from MKAN import MKANScorer, mkan_total_loss, MKANVisualizer

model = MKANScorer(input_size=12, hidden_size=32, M=8, K=2)
score = model(x_window)          # (batch, W, 12) → (batch,) ∈ (0,1)

# Entraînement
loss, pred_loss, reg_l1, reg_entropy = mkan_total_loss(model, x_window, targets)
loss.backward()

# Visualisation
viz = MKANVisualizer(model, history, feature_names=FEATURE_COLS)
viz.plot_training_dashboard().show()
viz.plot_edge_heatmap("input", x_pool_audit).show()
viz.plot_edge_functions("forget", x_pool_audit).show()
```

## Données

Les données proviennent du simulateur **MoMTSim** (Mobile Money Transaction Simulator).  
Le fichier `featuresLog.parquet` (généré par `MOMTSIM/main.py`) doit être placé dans `MOMTSIM/config/`.

## Pipeline d'entraînement

Voir [`train.ipynb`](train.ipynb) pour le pipeline complet :
1. Chargement `featuresLog.parquet` → normalisation → fenêtres glissantes W=10
2. Entraînement `MKANScorer` (40 epochs, Adam, lr=1e-3)
3. Évaluation : MCC, AUC-ROC, PR-AUC, Score de Brier
4. Détection de dérive JS + extension de grille adaptative
5. Élagage + régression symbolique → rapport d'audit COBAC
6. Visualisations Plotly interactives

## Métriques (section 3.4 du mémoire)

- **MCC** — Coefficient de Corrélation de Matthews (métrique principale, eq. 3.1)
- **AUC-ROC** — Aire sous la courbe ROC trapézoïdale (eq. 3.2)
- **PR-AUC** — Aire sous la courbe Précision-Rappel (eq. 3.6)
- **Score de Brier** — Calibration probabiliste (eq. 3.7)

Baseline XGBoost (Azamuke et al. 2025) : MCC = 0,82 · AUC = 0,97
