"""
MKAN_cuda — variante GPU-CUDA du package MKAN.

Remplacement drop-in de MKAN pour une execution sur GPU CUDA (Colab T4, A100, etc.).
Les modules inchanges sont re-exportes directement depuis MKAN.
Deux overrides GPU-specifiques :
  loss.py             : poids BCE calcules entierement on-device (suppression du
                        contournement DirectML .item())
  heuristic_search.py : num_workers / pin_memory / use_amp dans build_mkan_fitness

Usage dans train_colab.ipynb :
    import MKAN_cuda as MKAN
    from MKAN_cuda import MKANScorer, mkan_total_loss, build_mkan_fitness
"""

# ── Modules inchanges — re-exportes depuis MKAN ───────────────────────────────
from MKAN.hybrid_edge  import HybridEdgeFunction
from MKAN.hybrid_layer import HybridKANLayer
from MKAN.cell         import TKANCell, MKANScorer
from MKAN.symbolic     import (SYMBOLIC_LIBRARY, fit_symbolic_candidate,
                                fit_symbolic_best, format_formula)
from MKAN.drift        import js_divergence, detect_drift_region
from MKAN.audit        import (edge_l1_matrix, prune_mask, evaluate_edge_curve,
                                prune_and_extract, extract_full_model_report)
from MKAN.visualizer   import MKANVisualizer

# ── Overrides GPU-optimises ───────────────────────────────────────────────────
from .loss             import weighted_bce, mkan_total_loss
from .heuristic_search import (build_mkan_fitness, RechercheHeuristique,
                                ESPACE_MKAN_DEFAULT, ESPACE_MKAN_ETENDU)

__all__ = [
    # Architecture (inchangee)
    "HybridEdgeFunction",
    "HybridKANLayer",
    "TKANCell",
    "MKANScorer",
    # Entainement (GPU-optimise)
    "weighted_bce",
    "mkan_total_loss",
    # Derive (inchangee)
    "js_divergence",
    "detect_drift_region",
    # Symbolification (inchangee)
    "SYMBOLIC_LIBRARY",
    "fit_symbolic_candidate",
    "fit_symbolic_best",
    "format_formula",
    # Audit (inchange)
    "edge_l1_matrix",
    "prune_mask",
    "evaluate_edge_curve",
    "prune_and_extract",
    "extract_full_model_report",
    # Visualisation (inchangee)
    "MKANVisualizer",
    # Recherche heuristique (GPU-optimisee)
    "build_mkan_fitness",
    "RechercheHeuristique",
    "ESPACE_MKAN_DEFAULT",
    "ESPACE_MKAN_ETENDU",
]
