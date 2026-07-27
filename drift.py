"""
drift.py — Détection de dérive de distribution et déclenchement d'extension de grille
           (eq. 4.19–4.20, section 4.3).

  js_divergence        — divergence Jensen-Shannon entre deux histogrammes
  detect_drift_region  — localise la région de dérive pour extension de grille
"""

import numpy as np
from typing import Optional, Tuple


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Divergence Jensen-Shannon symétrisée entre deux distributions discrètes (eq. 4.19).

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)    où m = 0.5*(p + q)

    Args:
        p, q : histogrammes (non nécessairement normalisés)
        eps  : stabilité numérique pour les bins vides

    Returns:
        JS ∈ [0, log(2)] — 0 si les distributions sont identiques
    """
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def detect_drift_region(
    x_ref:       np.ndarray,
    x_cur:       np.ndarray,
    domain:      float = 2.0,
    bins:        int   = 20,
    theta_drift: float = 0.05,
    kappa_ext:   float = 2.0,
) -> Tuple[Optional[Tuple[float, float]], float]:
    """
    Détecte une dérive de distribution et localise la région affectée (eq. 4.19–4.20).

    Étapes :
      1. Calcul de JS(p_ref, p_cur) — si JS ≤ theta_drift : pas de dérive
      2. Identification des bins où la densité courante dépasse kappa_ext × la densité
         de référence → région candidate pour extension de grille (eq. 4.20)

    Args:
        x_ref       : échantillon de référence (distribution historique)
        x_cur       : échantillon courant
        domain      : demi-domaine normalisé ([-domain, domain])
        bins        : résolution de l'histogramme comparatif
        theta_drift : seuil JS au-dessus duquel on déclare une dérive (défaut 0.05)
        kappa_ext   : ratio min p_cur/p_ref pour qu'un bin soit « en dérive » (défaut 2.0)

    Returns:
        (region, js)
          region : tuple (xl, xr) si dérive détectée, None sinon
          js     : valeur JS calculée (utile pour logging)
    """
    edges    = np.linspace(-domain, domain, bins + 1)
    hist_ref, _ = np.histogram(x_ref, bins=edges)
    hist_cur, _ = np.histogram(x_cur, bins=edges)

    js = js_divergence(hist_ref.astype(float), hist_cur.astype(float))

    if js <= theta_drift:
        return None, js

    p_ref = hist_ref / max(hist_ref.sum(), 1)
    p_cur = hist_cur / max(hist_cur.sum(), 1)
    exceed = (p_cur / (p_ref + 1e-12)) > kappa_ext

    if not exceed.any():
        return None, js

    idx = np.where(exceed)[0]
    region = (float(edges[idx.min()]), float(edges[idx.max() + 1]))
    return region, js
