# Résultats de l'Étape 2 — Recherche Heuristique Étendue et Analyse de Sensibilité

## 1. Configuration optimale retenue

- **Cellule** : GRUKANCell
- **Régime d'entrée** : raw
- hidden_size=128, lr=0.0001, lam=0.0001, mu1=0.1, mu2=0.6989017708691363, batch_size=32, W=5

| Porte | Base | Hyperparamètres |
|---|---|---|
| Forget | relukan | {"Grid_G": 8} |
| Input | efficientkan | {"Grid_G": 3, "Spline_Degree_k": 4} |
| Candidate | wavkan | {"Mother_Wavelet": "dog", "M_wavelets": 3} |
| Output | N/A (absente — GRUKANCell) | |

MCC_raw=0.9834 | MCC_clipped=0.9834 | PR_AUC=0.9994 | Brier=0.0048 | R2_symbolic=0.5250 | latency_ms=853.5887 | fitness_total=0.6950

## 2. Justification par l'analyse de sensibilité

sensitivity_rankings.json ou la configuration optimale sont indisponibles.

## 3. Hyperparamètres fixés (redondants)

Aucun hyperparamètre redondant identifié (ou sensitivity_rankings.json absent).

## 4. Interactions critiques

Aucun hyperparamètre interactif identifié (ou sensitivity_rankings.json absent).

## 5. Comparaison Brut vs Traité

| Régime | n évaluations | fitness (best) | MCC (best) | PR_AUC (best) | Brier (best) |
|---|---|---|---|---|---|
| Brut | 64 | 0.6950 | 0.9834 | 0.9994 | 0.0048 |
| Traité | 34 | 0.6765 | 0.9559 | 0.9963 | 0.0160 |

Δ MCC (Traité - Brut) = -0.0275
Δ fitness (Traité - Brut) = -0.0185

## 6. TKANCell vs GRUKANCell

| Cellule | hidden_size | hidden_size_gru_adjusted | fitness (best) | MCC (best) |
|---|---|---|---|---|
| TKANCell | 32 | — | 0.6878 | 0.9740 |
| GRUKANCell | 128 | 170 | 0.6950 | 0.9834 |

Δ fitness (GRUKANCell - TKANCell) = 0.0072