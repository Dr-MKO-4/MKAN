"""
loss.py (MKAN_cuda)  version GPU-CUDA de MKAN/loss.py.

Fonctions de perte identiques a MKAN/loss.py (section 4.3.4), avec une seule
modification dans weighted_bce :

  Original  : les poids de classe sont calcules via .item() (extraction sur CPU)
              pour contourner un bug DirectML  rsub/rdiv non supportes en float64
              sur Intel Arc / Iris Xe.
  CUDA      : .item() est inutile sur CUDA (toutes les ops scalaires float32 sont
              supportees) et introduit un aller-retour CPU/GPU inutile.
              Les poids sont calcules directement comme tenseurs on-device.

Changements vs MKAN/loss.py :
  - weighted_bce : suppression de .item(), calcul des poids entierement on-device.
  - weighted_bce : ajout de scores.float() pour robustesse sous torch.amp.autocast
    (le modele peut sortir du float16 ; BCE necessite du float32).
  - mkan_total_loss : identique, pointe vers la nouvelle weighted_bce.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple


def weighted_bce(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Entropie croisee binaire ponderee (section 4.3.1).

    w_fraude   = N_total / (2 * N_fraude)
    w_legitime = N_total / (2 * N_legitime)

    Version GPU : les poids sont calcules comme tenseurs float32 sur le device
    cible  pas de round-trip CPU via .item().
    scores.float() garantit fp32 meme sous torch.amp.autocast (ou le modele
    produit du float16), car F.binary_cross_entropy necessite fp32.

    Args:
        scores  : (batch,) dans (0,1)  sorties de MKANScorer (fp16 ou fp32)
        targets : (batch,) dans {0,1}  etiquettes reelles

    Returns:
        l_pred : scalaire float32 sur le meme device que targets
    """
    scores  = scores.float()
    targets = targets.float()

    N       = targets.shape[0]                         # int Python  shape est CPU
    n_fraud = targets.sum().clamp(min=1.0)             # tenseur on-device
    n_legit = (N - n_fraud).clamp(min=1.0)            # tenseur on-device
    w_fraud = N / (2.0 * n_fraud)                     # scalaire tenseur on-device
    w_legit = N / (2.0 * n_legit)                     # scalaire tenseur on-device

    # torch.where diffuse w_fraud / w_legit (0-dim) vers la forme de targets
    weights = torch.where(targets == 1, w_fraud, w_legit)
    scores  = scores.clamp(min=1e-7, max=1.0 - 1e-7)
    # F.binary_cross_entropy est blacklistee par PyTorch dans les contextes autocast.
    # Formule inline equivalente  ops de base non blacklistees, toujours en fp32
    # car scores et targets ont deja ete castes en float() ci-dessus.
    return -(weights * (targets * scores.log() + (1.0 - targets) * (1.0 - scores).log())).mean()


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

        l_total = l_pred + lam*(mu1*sum_l|Phi_l|_1 + mu2*sum_l S(Phi_l))

    Identique a MKAN/loss.py sauf que weighted_bce est la version on-device.

    Args:
        model    : MKANScorer entrainable
        x_window : (batch, W, input_size)
        targets  : (batch,) dans {0,1}
        lam, mu1, mu2 : coefficients de regularisation (section 4.3.4)

    Returns:
        loss_total, loss_pred, reg_l1 (detache), reg_entropy (detache)
    """
    scores, reg_l1, reg_entropy = model(x_window, return_reg=True)

    loss_pred  = weighted_bce(scores, targets)
    loss_reg   = lam * (mu1 * reg_l1 + mu2 * reg_entropy)
    loss_total = loss_pred + loss_reg

    return loss_total, loss_pred, reg_l1.detach(), reg_entropy.detach()
