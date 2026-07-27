"""
loss.py — Fonction de perte ℓ_total de MKAN (section 4.3.4).

ℓ_total = ℓ_pred + λ(μ₁·Σ_l |Φ_l|₁ + μ₂·Σ_l S(Φ_l))

  ℓ_pred    — entropie croisée binaire pondérée (section 4.3.1, eq. ℓ_pred)
  |Φ_l|₁   — norme L1 exacte de la couche l (eq. 2.20), calculée sur le batch
  S(Φ_l)   — entropie des contributions relatives des arêtes (section 4.3.3)

Valeurs initiales recommandées (section 4.3.4) : λ=1e-2, μ₁=1.0, μ₂=0.5
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple


def weighted_bce(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Entropie croisée binaire pondérée (section 4.3.1).

    w_fraude   = N_total / (2 · N_fraude)
    w_légitime = N_total / (2 · N_légitime)

    ℓ_pred = -1/N · Σ_i [w_{y_i} · y_i · log(ŷ_i)
                        + w_{1-y_i} · (1-y_i) · log(1-ŷ_i)]

    Les poids sont calculés sur le batch courant ; l'entropie croisée pondérée
    garantit que les erreurs sur la classe minoritaire frauduleuse contribuent
    autant au gradient que celles sur la classe majoritaire légitime.

    Args:
        scores  : (batch,) ∈ (0,1) — sorties de MKANScorer
        targets : (batch,) ∈ {0,1} — étiquettes réelles

    Returns:
        ℓ_pred : scalaire
    """
    targets = targets.float()
    n_fraud = targets.sum().clamp(min=1.0)
    n_legit = (1.0 - targets).sum().clamp(min=1.0)
    N       = float(targets.shape[0])

    w_fraud = N / (2.0 * n_fraud)
    w_legit = N / (2.0 * n_legit)

    weights = torch.where(targets == 1.0, w_fraud, w_legit)
    # Clamp numérique : sigmoid peut saturer à 0/1 exact en float32 sur grandes entrées
    scores = scores.clamp(min=1e-7, max=1.0 - 1e-7)
    return F.binary_cross_entropy(scores, targets, weight=weights)


def mkan_total_loss(
    model,
    x_window:  torch.Tensor,
    targets:   torch.Tensor,
    lam:  float = 1e-2,
    mu1:  float = 1.0,
    mu2:  float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fonction de perte totale MKAN (section 4.3.4) :

        ℓ_total = ℓ_pred + λ(μ₁·Σ_l |Φ_l|₁ + μ₂·Σ_l S(Φ_l))

    Les termes de régularisation Σ_l |Φ_l|₁ et Σ_l S(Φ_l) sont accumulés
    sur les W pas de temps et les 4 portes T-KAN via model.forward(return_reg=True).
    Les normes L1 utilisées sont les normes exactes (eq. 2.19 : mean_batch |φ(x)|),
    pas les proxys par poids.

    Args:
        model    : MKANScorer entraînable
        x_window : (batch, W, input_size) — fenêtre glissante de transactions
        targets  : (batch,) ∈ {0,1} — étiquettes
        lam      : force globale de régularisation (défaut 1e-2, section 4.3.4)
        mu1      : poids norme L1     (défaut 1.0, section 4.3.4)
        mu2      : poids entropie     (défaut 0.5, section 4.3.4)

    Returns:
        loss_total  : ℓ_total — à appeler .backward() dessus
        loss_pred   : ℓ_pred seul (pour logging)
        reg_l1      : Σ_l |Φ_l|₁ accumulé (pour logging, detaché)
        reg_entropy : Σ_l S(Φ_l) accumulé (pour logging, detaché)
    """
    scores, reg_l1, reg_entropy = model(x_window, return_reg=True)

    loss_pred  = weighted_bce(scores, targets)
    loss_reg   = lam * (mu1 * reg_l1 + mu2 * reg_entropy)
    loss_total = loss_pred + loss_reg

    return loss_total, loss_pred, reg_l1.detach(), reg_entropy.detach()
