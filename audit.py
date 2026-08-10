"""
audit.py — Élagage + extraction symbolique pour rapport d'audit COBAC
           (eq. 4.28, section 4.4.5–4.4.7).

Pipeline d'audit complet :
  1. edge_l1_matrix      — importance L1 de chaque arête sur un pool de données
  2. prune_mask          — masque booléen des arêtes survivantes (seuil theta)
  3. evaluate_edge_curve — courbe 1D d'une arête isolée sur grille
  4. prune_and_extract   — élagage + régression symbolique sur une porte
  5. extract_full_model_report — rapport complet sur les 4 portes T-KAN

Chaque arête survivante porte soit une formule symbolique auditables
(R² ≥ r2_threshold) soit la mention explicite « non-symbolifiable (numerique) »
— cohérent avec la politique de transparence section 2.5.5.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from .hybrid_layer import HybridKANLayer
from .symbolic import fit_symbolic_best, format_formula


# ── Helpers bas niveau ────────────────────────────────────────────────────────

def edge_l1_matrix(layer: HybridKANLayer, x_pool: torch.Tensor) -> np.ndarray:
    """
    Calcule la matrice d'importance L1 des arêtes (eq. 2.19, version batch).

    L1_{i,j} = mean_batch |phi_{i,j}(x_i)|

    Args:
        layer  : HybridKANLayer à analyser
        x_pool : pool de données représentatif (batch, in_features)

    Returns:
        l1_mat : (in_features, out_features) float64 numpy
    """
    with torch.inference_mode():
        edges = layer.edge_activations(x_pool)   # (batch, in, out)
    return edges.abs().mean(dim=0).cpu().numpy()  # (in, out)


def prune_mask(l1_mat: np.ndarray, theta: float = 1e-2) -> np.ndarray:
    """
    Masque booléen des arêtes survivantes à l'élagage (eq. 4.28).

    Une arête (i, j) est supprimée si L1_{i,j} ≤ theta.

    Args:
        l1_mat : matrice d'importance (in_features, out_features)
        theta  : seuil d'élagage (défaut 1e-2)

    Returns:
        mask : (in_features, out_features) bool
    """
    return l1_mat > theta


def evaluate_edge_curve(
    layer:    HybridKANLayer,
    i:        int,
    j:        int,
    x_grid_t: torch.Tensor,
) -> np.ndarray:
    """
    Évalue la courbe de l'arête (i → j) sur une grille 1D.

    Toutes les autres features sont fixées à 0 (comportement marginal de l'arête).

    Args:
        layer    : HybridKANLayer source
        i        : indice de la feature d'entrée
        j        : indice du nœud de sortie
        x_grid_t : grille 1D (N,) — doit être dans [-domain, domain]

    Returns:
        curve : (N,) numpy float32
    """
    N = x_grid_t.shape[0]
    x = torch.zeros(N, layer.in_features, device=x_grid_t.device)
    x[:, i] = x_grid_t
    with torch.inference_mode():
        edges = layer.edge_activations(x)   # (N, in, out)
    return edges[:, i, j].cpu().numpy()


# ── Élagage + régression symbolique sur une porte ───────────────────────────

def prune_and_extract(
    layer:         HybridKANLayer,
    x_pool:        torch.Tensor,
    feature_names: list[str],
    hidden_size:   int,
    theta:         float = 1e-2,
    r2_threshold:  float = 0.99,
    domain:        float = 2.0,
    top_k:         int   = 8,
) -> list[dict]:
    """
    Élagage (eq. 4.28) + régression symbolique (eq. 2.23) sur une porte T-KAN.

    Le tri par importance L1 est appliqué AVANT le fit symbolique (coûteux) :
    seules les top_k arêtes les plus importantes sont analysées symboliquement
    (décision de tractabilité computationnelle, documentée section 4.4.5).

    Args:
        layer         : porte KAN à analyser (ex : model.cell.forget_gate)
        x_pool        : pool de données représentatif (batch, concat_size)
        feature_names : noms des features originales (len = input_size)
        hidden_size   : taille de l'état caché (pour labels des entrées)
        theta         : seuil d'élagage (défaut 1e-2)
        r2_threshold  : seuil de symbolifiabilité (défaut 0.99)
        domain        : demi-domaine de la grille d'évaluation (défaut 2.0)
        top_k         : max d'arêtes analysées symboliquement par porte (défaut 8)

    Returns:
        liste de dict triée par importance décroissante :
          {input, output, l1_importance, formula, r2, symbolifiable}
    """
    # Labels du vecteur concaténé [h_{t-1}, x_t]
    concat_labels = ([f"h[{k}]" for k in range(hidden_size)]
                     + list(feature_names))

    l1_mat  = edge_l1_matrix(layer, x_pool)
    mask    = prune_mask(l1_mat, theta=theta)
    surviving = np.argwhere(mask)

    if len(surviving) == 0:
        return []

    # Tri par importance et restriction au top_k
    importances = l1_mat[mask]
    order       = np.argsort(-importances)
    if top_k:
        order = order[:top_k]
    surviving = surviving[order]

    x_grid_t  = torch.linspace(-domain, domain, 100, device=x_pool.device)
    x_grid_np = x_grid_t.cpu().numpy()

    results = []
    for i, j in surviving:
        curve = evaluate_edge_curve(layer, int(i), int(j), x_grid_t)
        best  = fit_symbolic_best(curve, x_grid_np, r2_threshold=r2_threshold)
        results.append({
            "input":        concat_labels[i],
            "output":       f"h_out[{j}]",
            "l1_importance": float(l1_mat[i, j]),
            "formula":      (format_formula(best)
                             if best["symbolifiable"]
                             else "non-symbolifiable (numerique)"),
            "r2":           best["r2"],
            "symbolifiable": best["symbolifiable"],
        })

    results.sort(key=lambda r: -r["l1_importance"])
    return results


# ── Rapport d'audit complet ──────────────────────────────────────────────────

def extract_full_model_report(
    model,
    x_pool:        torch.Tensor,
    feature_names: list[str],
    theta:         float = 1e-2,
    r2_threshold:  float = 0.99,
) -> dict[str, list[dict]]:
    """
    Rapport d'audit complet (section 4.4.7) sur les 4 portes T-KAN.

    Pour chaque porte, liste les arêtes survivantes à l'élagage avec leur
    formule symbolique extraite (ou mention de non-symbolifiabilité).
    Ce rapport constitue la justification mathématique opposable lors d'un
    contrôle COBAC (section 2.5.5).

    Args:
        model         : MKANScorer entraîné
        x_pool        : pool de données concaténées [h, x] (batch, hidden+input)
        feature_names : noms des features originales (input_size)
        theta         : seuil d'élagage
        r2_threshold  : seuil de symbolifiabilité

    Returns:
        dict {gate_name: [arêtes auditables]} pour les 4 portes :
          "forget", "input", "candidate", "output"
    """
    hidden_size = model.hidden_size
    gates = {
        "forget":    model.cell.forget_gate,
        "input":     model.cell.input_gate,
        "candidate": model.cell.candidate_gate,
        "output":    model.cell.output_gate,
    }
    report = {}
    for gate_name, layer in gates.items():
        report[gate_name] = prune_and_extract(
            layer, x_pool, feature_names, hidden_size,
            theta=theta, r2_threshold=r2_threshold,
        )
    return report
