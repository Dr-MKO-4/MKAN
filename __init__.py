"""
MKAN  package Python du modèle M-KAN pour la détection de fraude Mobile Money.

Architecture (section 4.2) :
  HybridEdgeFunction   arête hybride Gaussienne + Fourier (eq. 4.13)
  HybridKANLayer       couche vectorisée + nœuds MultKAN + extension grille
  TKANCell             cellule LSTM-KAN (eq. 4.7–4.12)
  MKANScorer           scoreur de risque sur fenêtre glissante (eq. 4.16)

Analyse post-entraînement :
  js_divergence            détection de dérive de distribution (eq. 4.19)
  detect_drift_region      localisation de la dérive pour extension de grille
  SYMBOLIC_LIBRARY         familles de fonctions candidates
  fit_symbolic_best        régression symbolique sur une arête (eq. 2.23)
  format_formula           formatage COBAC
  edge_l1_matrix           matrice d'importance L1 des arêtes
  prune_mask               masque d'élagage (eq. 4.28)
  evaluate_edge_curve      courbe marginale d'une arête
  prune_and_extract        élagage + symbolification d'une porte
  extract_full_model_report  rapport d'audit COBAC complet (section 4.4.7)

Visualisation (section 4.4.6) :
  MKANVisualizer           7 méthodes Plotly : loss, régularisation, métriques,
                            heatmap L1, courbes d'arêtes (Gaussien + Fourier), élagage

Usage minimal :
    from mkan import MKANScorer, extract_full_model_report

    model  = MKANScorer(input_size=12, hidden_size=32)
    score  = model(x_window)            # (batch, W, 12) → (batch,)
    report = extract_full_model_report(model, x_pool, feature_names)
"""

from .hybrid_edge  import HybridEdgeFunction
from .hybrid_layer import HybridKANLayer
from .cell         import TKANCell, MKANScorer
from .loss         import weighted_bce, mkan_total_loss
from .symbolic     import (SYMBOLIC_LIBRARY, fit_symbolic_candidate,
                            fit_symbolic_best, format_formula)
from .drift        import js_divergence, detect_drift_region
from .audit        import (edge_l1_matrix, prune_mask, evaluate_edge_curve,
                            prune_and_extract, extract_full_model_report)
from .visualizer   import MKANVisualizer

__all__ = [
    # Architecture
    "HybridEdgeFunction",
    "HybridKANLayer",
    "TKANCell",
    "MKANScorer",
    # Entraînement
    "weighted_bce",
    "mkan_total_loss",
    # Dérive
    "js_divergence",
    "detect_drift_region",
    # Symbolification
    "SYMBOLIC_LIBRARY",
    "fit_symbolic_candidate",
    "fit_symbolic_best",
    "format_formula",
    # Audit
    "edge_l1_matrix",
    "prune_mask",
    "evaluate_edge_curve",
    "prune_and_extract",
    "extract_full_model_report",
    # Visualisation
    "MKANVisualizer",
]
