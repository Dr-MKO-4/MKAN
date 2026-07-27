"""
hybrid_edge.py — Fonction d'arête hybride Gaussienne + Fourier (eq. 4.13, section 4.2.2).

Module autonome : une seule entrée scalaire → une sortie scalaire.
Sert de brique de validation unitaire et de référence documentaire.
La version vectorisée in_features × out_features est dans hybrid_layer.py.
"""

import math
import torch
import torch.nn as nn


class HybridEdgeFunction(nn.Module):
    """
    Fonction d'activation d'arête hybride Gaussienne + Fourier (eq. 4.13, section 4.2.2).

    phi(x) = sum_m w_m^G * exp(-(x - mu_m)^2 / (2h^2))   [composante FastKAN]
           + sum_k [a_k cos(kx) + b_k sin(kx)]             [composante KAN-AD]

    Les centres mu_m sont fixes (buffer, non appris), uniformément répartis sur le
    domaine normalisé [-domain, domain] — cohérent avec FastKAN (Li, 2024) où les
    centres RBF sont « fixés a priori sur l'espace d'entrée » (section 2.4.2).
    Seuls les poids w_gauss, a_fourier, b_fourier sont appris par rétropropagation.

    Args:
        M       : nombre de centres gaussiens (défaut 8, section 4.2.2)
        K       : ordre de Fourier (défaut 2, section 4.2.2)
        domain  : demi-domaine d'entrée — x doit être normalisé dans [-domain, domain]
        h       : largeur de bande RBF ; par défaut espacement inter-centres 2*domain/(M-1)
    """

    def __init__(self, M: int = 8, K: int = 2, domain: float = 1.0, h: float = None):
        super().__init__()
        self.M = M
        self.K = K

        centers = torch.linspace(-domain, domain, M)
        self.register_buffer("centers", centers)

        if h is None:
            h = (2 * domain) / (M - 1)
        self.register_buffer("h", torch.tensor(float(h)))

        self.w_gauss    = nn.Parameter(torch.randn(M) * 0.1)
        self.a_fourier  = nn.Parameter(torch.randn(K) * 0.1)
        self.b_fourier  = nn.Parameter(torch.randn(K) * 0.1)

        k_idx = torch.arange(1, K + 1, dtype=torch.float32)
        self.register_buffer("k_idx", k_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch,) → sortie (batch,)"""
        x_exp = x.unsqueeze(-1)                              # (batch, 1)

        # Composante gaussienne
        diff      = x_exp - self.centers                     # (batch, M)
        gauss     = torch.exp(-(diff ** 2) / (2 * self.h ** 2))
        gauss_out = gauss @ self.w_gauss                     # (batch,)

        # Composante Fourier
        kx         = x_exp * self.k_idx                     # (batch, K)
        fourier_out = (torch.cos(kx) @ self.a_fourier) + (torch.sin(kx) @ self.b_fourier)

        return gauss_out + fourier_out

    def l1_norm(self, x: torch.Tensor) -> torch.Tensor:
        """|phi|_1 (eq. 2.19) — moyenne des valeurs absolues sur le batch."""
        return self.forward(x).abs().mean()
