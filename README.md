# MKAN — Mobile Money KAN Fraud Detector

Implémentation PyTorch du modèle **M-KAN** pour la détection de fraude Mobile Money en environnement CEMAC/COBAC, développé dans le cadre d'un mémoire de Master.

## Architecture

```
MKANScorer
└── TKANCell  (cellule récurrente LSTM-KAN, eq. 4.7–4.12)
    ├── forget_gate    ─┐
    ├── input_gate     ─┤  HybridKANLayer (eq. 4.13–4.15)
    ├── candidate_gate ─┤  chaque porte : nœuds additifs (eq. 4.14)
    └── output_gate    ─┘              ou multiplicatifs MultKAN (eq. 4.15)
```

Chaque porte est une `HybridKANLayer` qui prend en entrée le vecteur concaténé `[h_{t-1}, x_t]` de taille `hidden_size + input_size`.

## Fonctions d'arêtes hybrides (eq. 4.13)

La fonction d'arête φᵢⱼ combine deux composantes :

- **Gaussienne** (FastKAN) : `Σₘ wₘ · exp(-(x-μₘ)²/2h²)`
- **Fourier** (KAN-AD) : `Σₖ [aₖ cos(kx) + bₖ sin(kx)]`

## Nœuds additifs et multiplicatifs

Chaque nœud de sortie j d'une `HybridKANLayer` agrège les arêtes entrantes de deux façons possibles, contrôlées par `node_types` :

| `node_types[j]`    | Équation                              | Description                                |
|--------------------|---------------------------------------|--------------------------------------------|
| `'add'` (défaut)   | `Σᵢ φᵢⱼ(xᵢ)` (eq. 4.14)             | Somme sur toutes les features entrantes    |
| `'mult'`           | `Πᵢ φᵢⱼ(xᵢ)` (eq. 4.15)             | Produit sur toutes les features entrantes  |
| `[i1, i2, ...]`    | `φᵢ₁ⱼ(xᵢ₁) · φᵢ₂ⱼ(xᵢ₂) · ...`      | Produit sur un sous-ensemble d'indices     |

Le cas `[i1, i2]` — **interaction ciblée entre deux features** — est l'usage principal de MultKAN dans ce modèle (section 4.2.3).

### Pourquoi MultKAN ?

Un nœud additif détecte la présence de chaque signal séparément.  
Un nœud multiplicatif détecte la **co-occurrence** de plusieurs signaux : φ(r1) × φ(flag_nuit) vaut fort uniquement si *les deux* sont activés simultanément.  
Cela permet d'encoder des interactions contextuelles sans augmenter la dimensionnalité ni ajouter de features croisées.

## Configuration via `mult_pairs`

`TKANCell` et `MKANScorer` acceptent un argument `mult_pairs` de la forme :

```python
mult_pairs = {
    "nom_porte": {
        j: (i1, i2)   # nœud j de la porte effectue φᵢ₁(xᵢ₁) × φᵢ₂(xᵢ₂)
    }
}
```

- `nom_porte` ∈ `{"forget", "input", "candidate", "output"}`
- `j` : indice du nœud de sortie dans la porte (0 à `hidden_size - 1`)
- `i1, i2` : **indices dans le vecteur concaténé** `[h_{t-1}, x_t]` de taille `hidden_size + input_size`

> Les `hidden_size` premiers indices correspondent à `h_{t-1}`.  
> Les features originales `x_t` commencent à l'indice `hidden_size`.

### Exemple : porte d'entrée, section 4.2.3

Avec `input_size=12`, `hidden_size=32`, les 12 features occupent les indices 32–43 dans le vecteur concaténé :

| Feature         | Indice dans `x_t` | Indice dans `[h, x]` |
|-----------------|-------------------|----------------------|
| `montant`       | 0                 | 32                   |
| `solde_avant`   | 1                 | 33                   |
| `r1`            | 2                 | 34                   |
| ...             | ...               | ...                  |
| `flag_nuit`     | 10                | 42                   |
| `flag_weekend`  | 11                | 43                   |

Interaction **"grand retrait nocturne"** (section 4.2.3) — nœud 0 de la porte `input` :

```python
model = MKANScorer(
    input_size=12,
    hidden_size=32,
    M=8,
    K=2,
    mult_pairs={"input": {0: (34, 42)}}  # φ(r1) × φ(flag_nuit)
)
```

## Modules

| Fichier | Rôle |
|---|---|
| `hybrid_edge.py` | Fonction d'arête hybride (eq. 4.13) |
| `hybrid_layer.py` | Couche KAN vectorisée : nœuds additifs (eq. 4.14) + nœuds MultKAN (eq. 4.15) + extension de grille (eq. 4.17–4.18) |
| `cell.py` | Cellule T-KAN (4 portes KAN, eq. 4.7–4.12) + MKANScorer (eq. 4.16) |
| `loss.py` | Perte totale : BCE pondérée + L1 + entropie (eq. 4.27) |
| `drift.py` | Détection de dérive JS + localisation région (eq. 4.19–4.20) |
| `symbolic.py` | Régression symbolique sur les courbes d'arêtes (eq. 2.23) |
| `audit.py` | Élagage + rapport d'audit COBAC (eq. 4.28, section 4.4.7) |
| `visualizer.py` | 7 visualisations Plotly : loss, métriques, heatmap L1, courbes φᵢⱼ, élagage |
| `heuristic_search.py` | Recherche heuristique (algorithme génétique) des hyperparamètres : mutation adaptative, contrôle de diversité, modèle EDA |

## Installation

```bash
pip install torch numpy pandas pyarrow plotly
```

## Usage minimal

```python
from MKAN import MKANScorer, mkan_total_loss, MKANVisualizer

# Modèle purement additif
model = MKANScorer(input_size=12, hidden_size=32, M=8, K=2)

# Avec nœuds MultKAN : φ(r1) × φ(flag_nuit) dans la porte d'entrée
model = MKANScorer(
    input_size=12, hidden_size=32, M=8, K=2,
    mult_pairs={"input": {0: (34, 42)}}  # indices dans [h_{t-1}, x_t]
)

score = model(x_window)   # (batch, W, 12) → (batch,) ∈ (0, 1)

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
2. Split train/val/test + construction des DataLoaders
3. Évaluation des features (SHAP/importance)
4. Pré-traitement avancé (SMOTE, normalisation)
5. **Recherche heuristique des hyperparamètres** (`heuristic_search.py`) — algorithme génétique sur `ESPACE_MKAN_DEFAULT` ou `ESPACE_MKAN_ETENDU` (9 HP dont W et batch_size)
6. Entraînement `MKANScorer` avec warm start depuis les meilleurs poids trouvés (Adam, lr optimisé)
7. Évaluation : MCC, AUC-ROC, PR-AUC, Score de Brier
8. Détection de dérive JS + extension de grille adaptative
9. Élagage + régression symbolique → rapport d'audit COBAC
10. Visualisations Plotly interactives

## Métriques (section 3.4 du mémoire)

- **MCC** — Coefficient de Corrélation de Matthews (métrique principale, eq. 3.1)
- **AUC-ROC** — Aire sous la courbe ROC trapézoïdale (eq. 3.2)
- **PR-AUC** — Aire sous la courbe Précision-Rappel (eq. 3.6)
- **Score de Brier** — Calibration probabiliste (eq. 3.7)

Baseline XGBoost (Azamuke et al. 2025) : MCC = 0,82 · AUC = 0,97
