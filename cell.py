"""
cell.py  Cellule récurrente T-KAN et scoreur MKAN (section 4.2.1, 4.2.4).

  TKANCell     LSTM dont les 4 portes sont remplacées par des HybridKANLayer
                (eq. 4.7–4.12)
  MKANScorer   Déroulé temporel + projection vers score de risque (eq. 4.16)
"""

import torch
import torch.nn as nn

from .hybrid_layer import HybridKANLayer


class TKANCell(nn.Module):
    """
    Cellule récurrente T-KAN (section 4.2.1, eq. 4.7–4.12).

    Les 4 portes LSTM (oubli, entrée, candidat, sortie) sont chacune calculées
    par un opérateur KAN_smart  une HybridKANLayer  au lieu d'une
    transformation affine nn.Linear standard.

    Args:
        input_size  : nombre de features en entrée (12 features MKAN + éventuels extras)
        hidden_size : dimension de l'état caché h_t / c_t
        M           : centres gaussiens par arête (défaut 8)
        K           : ordre de Fourier (défaut 2)
        domain      : demi-domaine normalisé (défaut 1.0)
        mult_pairs  : dict {nom_porte: {j: (i1, i2)}} pour activer des nœuds
                      MultKAN dans une porte spécifique. Exemple pour la porte
                      d'entrée sur le ratio r1 (section 4.2.3) :
                          {"input": {0: (r1_idx, flag_nuit_idx)}}
                      Par défaut toutes les portes sont additives.

    État interne :
        h_t : état caché  (batch, hidden_size)
        c_t : état cellule (batch, hidden_size)

    Note : les indices i1, i2 dans mult_pairs font référence au vecteur
    concaténé [h_{t-1}, x_t] de taille (hidden_size + input_size). Les
    indices des features originales commencent donc à hidden_size.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 M: int = 8, K: int = 2, domain: float = 1.0,
                 mult_pairs: dict = None):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        concat_size      = input_size + hidden_size

        def make_gate(pairs=None):
            node_types = ["add"] * hidden_size
            if pairs:
                for j, (i1, i2) in pairs.items():
                    node_types[j] = (i1, i2)
            return HybridKANLayer(concat_size, hidden_size,
                                   node_types=node_types,
                                   M=M, K=K, domain=domain)

        mult_pairs = mult_pairs or {}
        self.forget_gate    = make_gate(mult_pairs.get("forget"))    # eq. 4.7
        self.input_gate     = make_gate(mult_pairs.get("input"))     # eq. 4.8
        self.candidate_gate = make_gate(mult_pairs.get("candidate")) # eq. 4.9
        self.output_gate    = make_gate(mult_pairs.get("output"))    # eq. 4.10

    def forward(self, x_t: torch.Tensor,
                h_prev: torch.Tensor, c_prev: torch.Tensor):
        """
        Un pas de temps.

        Args:
            x_t    : (batch, input_size)
            h_prev : (batch, hidden_size)
            c_prev : (batch, hidden_size)

        Returns:
            h_t : (batch, hidden_size)
            c_t : (batch, hidden_size)
        """
        combined = torch.cat([h_prev, x_t], dim=-1)          # (batch, hidden+input)

        f_t     = torch.sigmoid(self.forget_gate(combined))   # eq. 4.7
        i_t     = torch.sigmoid(self.input_gate(combined))    # eq. 4.8
        c_tilde = torch.tanh(self.candidate_gate(combined))   # eq. 4.9
        o_t     = torch.sigmoid(self.output_gate(combined))   # eq. 4.10

        c_t = f_t * c_prev + i_t * c_tilde   # eq. 4.11
        h_t = o_t * torch.tanh(c_t)          # eq. 4.12
        return h_t, c_t

    def forward_with_reg(self, x_t: torch.Tensor,
                         h_prev: torch.Tensor, c_prev: torch.Tensor):
        """
        Un pas de temps + termes de régularisation, en un seul passage par porte.

        Utilise HybridKANLayer.forward_with_reg() : edge_activations est appelé
        1× par porte (au lieu de 3× avec forward + layer_l1_total + layer_entropy).

        Returns:
            h_t      : (batch, hidden_size)
            c_t      : (batch, hidden_size)
            l1_total : scalaire  Σ_portes Σ_ij |φ_ij|₁
            entropy  : scalaire  Σ_portes S(Φ_l)
        """
        combined = torch.cat([h_prev, x_t], dim=-1)

        f_raw, l1_f, ent_f = self.forget_gate.forward_with_reg(combined)
        i_raw, l1_i, ent_i = self.input_gate.forward_with_reg(combined)
        c_raw, l1_c, ent_c = self.candidate_gate.forward_with_reg(combined)
        o_raw, l1_o, ent_o = self.output_gate.forward_with_reg(combined)

        f_t     = torch.sigmoid(f_raw)
        i_t     = torch.sigmoid(i_raw)
        c_tilde = torch.tanh(c_raw)
        o_t     = torch.sigmoid(o_raw)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t, l1_f + l1_i + l1_c + l1_o, ent_f + ent_i + ent_c + ent_o

    def l1_norm(self) -> torch.Tensor:
        """Somme des normes L1 proxy des 4 portes (pour la loss, eq. 4.25)."""
        return (self.forget_gate.l1_norm()
                + self.input_gate.l1_norm()
                + self.candidate_gate.l1_norm()
                + self.output_gate.l1_norm())


class MKANScorer(nn.Module):
    """
    Pipeline complet section 4.2.4 (eq. 4.16) :
      1. Déroulé de TKANCell sur une fenêtre glissante de W transactions
      2. Projection linéaire + sigmoïde → score de risque continu ∈ (0, 1)

    score(x_{1:W}) = σ(w^T h_W + b)     (eq. 4.16)

    Args:
        input_size  : features par pas de temps
        hidden_size : dimension de l'état caché
        M, K, domain, mult_pairs : transmis à TKANCell

    Note :
        - États initiaux h_0, c_0 à zéro (convention standard, non précisé dans
          le mémoire).
        - Déroulé par boucle Python explicite pour lisibilité et débogage 
          à optimiser (vmap/JIT) si les volumes le nécessitent.
        - projection = nn.Linear classique, pas une couche KAN  fidèle à eq. 4.16.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 M: int = 8, K: int = 2, domain: float = 1.0,
                 mult_pairs: dict = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = TKANCell(input_size, hidden_size,
                                    M=M, K=K, domain=domain,
                                    mult_pairs=mult_pairs)
        self.projection  = nn.Linear(hidden_size, 1)   # w^T h_t + b (eq. 4.16)

    def forward(self, x_window: torch.Tensor,
                return_reg: bool = False):
        """
        Args:
            x_window   : (batch, W, input_size)
            return_reg : si True, retourne aussi les termes de régularisation
                         accumulés sur les W pas (Σ_t Σ_l |Φ_l|₁, Σ_t Σ_l S(Φ_l))

        Returns:
            score      : (batch,)  probabilité de fraude ∈ (0, 1)
            reg_l1     : scalaire (seulement si return_reg=True)
            reg_entropy: scalaire (seulement si return_reg=True)
        """
        batch, W, _ = x_window.shape
        device = x_window.device
        h_t = torch.zeros(batch, self.hidden_size, device=device, dtype=torch.float32)
        c_t = torch.zeros(batch, self.hidden_size, device=device, dtype=torch.float32)

        if return_reg:
            reg_l1      = torch.tensor(0.0, dtype=torch.float32, device=device)
            reg_entropy = torch.tensor(0.0, dtype=torch.float32, device=device)

        for t in range(W):
            x_t = x_window[:, t, :]
            if return_reg:
                # forward_with_reg : edge_activations appelé 1× par porte
                # (au lieu de 3× avec forward + layer_l1_total + layer_entropy)
                h_t, c_t, l1_t, ent_t = self.cell.forward_with_reg(x_t, h_t, c_t)
                reg_l1      = reg_l1      + l1_t
                reg_entropy = reg_entropy + ent_t
            else:
                h_t, c_t = self.cell(x_t, h_t, c_t)

        score = torch.sigmoid(self.projection(h_t)).squeeze(-1)

        if return_reg:
            return score, reg_l1, reg_entropy
        return score

    def l1_norm(self) -> torch.Tensor:
        """Proxy L1 (sans batch)  utiliser forward(return_reg=True) en entraînement."""
        return self.cell.l1_norm()
