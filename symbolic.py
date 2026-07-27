"""
symbolic.py — Régression symbolique sur les courbes d'arêtes KAN (section 2.5, eq. 2.23).

Pipeline :
  1. fit_symbolic_candidate  — ajuste c*f(a*x+b)+d pour une famille f donnée
  2. fit_symbolic_best       — cherche la meilleure famille dans SYMBOLIC_LIBRARY
  3. format_formula          — formate le résultat en chaîne lisible COBAC

Décision de conception (section 2.5.5) :
  Une arête dont le meilleur R² reste < r2_threshold est déclarée
  « non-symbolifiable » plutôt que forcée dans une famille inexacte.
  Ce choix garantit l'honnêteté du rapport d'audit : un auditeur COBAC
  ne se voit pas présenter une formule trompeuse à 91% de fidélité.

Bibliothèque de fonctions (section 2.5.3) :
  Noyau formel de la thèse : {x², √|x|, sin, eˣ, log, 1/x, |x|}
  Extensions nécessaires (illustrées en eq. 4.20, 4.22 de la thèse) :
    x        — identité/linéarité (cas dégénéré fréquent après élagage)
    cos(x)   — complément de sin, composantes Fourier paires
    tanh(x)  — saturation hyperbolique des features de vélocité (eq. 4.22)
    x^3      — terme cubique pour asymétries plus marquées que x^2
    sigmoid  — saturation logistique (score de risque, eq. 4.22)
    softplus — transition douce log(1+eˣ) (eq. 4.20 : log(1+e^{β(r-r*)}))
"""

import torch
import numpy as np
from typing import Optional


# ── Bibliothèque de fonctions candidates (section 2.5.3 + extensions) ────────

SYMBOLIC_LIBRARY: dict = {
    # ── Noyau formel de la thèse (section 4.3.3, eq. 4.20) ──────────────────
    "x^2":          lambda x: x ** 2,
    "sqrt(|x|)":    lambda x: torch.sqrt(torch.abs(x) + 1e-6),
    "sin(x)":       torch.sin,
    "exp(x)":       lambda x: torch.exp(torch.clamp(x, max=10.0)),
    "log(|x|+eps)": lambda x: torch.log(torch.abs(x) + 1e-3),
    "1/(x+eps)":    lambda x: 1.0 / (x + torch.sign(x) * 1e-2
                                       + (x == 0).float() * 1e-2),
    "|x|":          torch.abs,
    # ── Extensions (présentes dans les formules illustratives de la thèse) ───
    "x":            lambda x: x,                                         # identité
    "cos(x)":       torch.cos,
    "tanh(x)":      torch.tanh,                                          # eq. 4.22
    "x^3":          lambda x: x ** 3,
    "sigmoid(x)":   lambda x: torch.sigmoid(x),
    "softplus(x)":  lambda x: torch.log(1.0 + torch.exp(               # eq. 4.20
                                  torch.clamp(x, max=20.0))),
}


# ── Ajustement d'un candidat unique ─────────────────────────────────────────

def fit_symbolic_candidate(
    curve_y:      np.ndarray,
    x_grid:       np.ndarray,
    fn,
    n_steps:      int   = 150,
    lr:           float = 0.05,
    seed:         int   = 0,
) -> dict:
    """
    Ajuste phi(x) ≈ c * f(a*x + b) + d par descente de gradient Adam (eq. 2.23).

    Args:
        curve_y : valeurs observées de l'arête sur la grille (N,)
        x_grid  : grille d'évaluation (N,)
        fn      : fonction symbolique candidate (callable torch)
        n_steps : itérations d'optimisation
        lr      : taux d'apprentissage Adam
        seed    : graine pour reproductibilité

    Returns:
        dict avec clés a, b, c, d, r2
        r2 = -inf si la fonction produit NaN/Inf
    """
    torch.manual_seed(seed)
    x_t = torch.tensor(x_grid,  dtype=torch.float32)
    y_t = torch.tensor(curve_y, dtype=torch.float32)

    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(0.0))
    c = torch.nn.Parameter(torch.tensor(1.0))
    d = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([a, b, c, d], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = c * fn(a * x_t + b) + d
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return {"a": float("nan"), "b": float("nan"),
                    "c": float("nan"), "d": float("nan"),
                    "r2": -float("inf")}
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred    = c * fn(a * x_t + b) + d
        ss_res  = ((y_t - pred) ** 2).sum()
        ss_tot  = ((y_t - y_t.mean()) ** 2).sum()
        r2      = (1 - ss_res / (ss_tot + 1e-12)).item()

    return {
        "a": a.item(), "b": b.item(),
        "c": c.item(), "d": d.item(),
        "r2": r2,
    }


# ── Sélection de la meilleure famille ────────────────────────────────────────

def fit_symbolic_best(
    curve_y:      np.ndarray,
    x_grid:       np.ndarray,
    library:      Optional[dict] = None,
    r2_threshold: float          = 0.99,
) -> dict:
    """
    Cherche la meilleure famille symbolique dans la bibliothèque (section 2.5.3).

    Si le meilleur R² < r2_threshold → arête déclarée « non-symbolifiable »
    (section 2.5.5) : l'arête reste représentée numériquement dans le rapport
    d'audit plutôt que forcée dans une formule inexacte.

    Args:
        curve_y       : courbe de l'arête (N,)
        x_grid        : grille d'évaluation numpy (N,)
        library       : dict {nom: fn_torch} — SYMBOLIC_LIBRARY par défaut
        r2_threshold  : seuil de symbolifiabilité (défaut 0.99)

    Returns:
        dict avec clés : name, a, b, c, d, r2, symbolifiable (bool)
    """
    library = library or SYMBOLIC_LIBRARY
    best: Optional[dict] = None

    for name, fn in library.items():
        result       = fit_symbolic_candidate(curve_y, x_grid, fn)
        result["name"] = name
        if best is None or result["r2"] > best["r2"]:
            best = result

    best["symbolifiable"] = best["r2"] >= r2_threshold
    return best


# ── Formatage COBAC ───────────────────────────────────────────────────────────

def format_formula(candidate: dict) -> str:
    """
    Formate un résultat de fit en expression lisible pour rapport d'audit COBAC.

    Exemple : '0.872 * tanh(1.103*x + -0.012) + 0.001'
    """
    a, b  = candidate["a"], candidate["b"]
    c, d  = candidate["c"], candidate["d"]
    name  = candidate["name"]
    sign_d = "+" if d >= 0 else "-"
    return f"{c:.3f} * {name}({a:.3f}*x + {b:.3f}) {sign_d} {abs(d):.3f}"
