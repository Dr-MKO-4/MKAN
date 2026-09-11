"""
gru_cell.py  Cellule récurrente GRU-KAN et scoreur associé (Niveau 1, facteur
cellule LSTM/GRU  protocole_benchmark.md §2.1).

Miroir direct de ../cell.py (TKANCell, MKANScorer) mais à 3 portes (reset,
update, candidate) au lieu de 4, chacune une GenericKANLayer (edges/layer.py)
au lieu d'une nn.Linear  même principe que TKANCell mais avec des bases
d'arête interchangeables (au lieu de la seule HybridEdgeFunction figée).

Équations GRU standard (convention PyTorch nn.GRUCell) :
    r_t = sigmoid(KAN_r([h_{t-1}, x_t]))                        (reset)
    z_t = sigmoid(KAN_z([h_{t-1}, x_t]))                        (update)
    n_t = tanh(KAN_n([r_t * h_{t-1}, x_t]))                     (candidate)
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}

match_gru_hidden_size() résout la taille cachée du GRU qui égalise le nombre
total de paramètres avec un LSTM-KAN de référence (3 portes contre 4, cf.
protocole §2.1)  condition d'équité pour comparer les deux cellules.
"""

import math
import torch
import torch.nn as nn

from edges import GenericKANLayer, BASIS_REGISTRY


def match_gru_hidden_size(input_size: int, hidden_size_lstm: int,
                           n_gates_lstm: int = 4, n_gates_gru: int = 3) -> int:
    """
    Résout hidden_size_gru tel que :
        n_gates_gru * h * (input_size + h) ~= n_gates_lstm * hidden_size_lstm * (input_size + hidden_size_lstm)

    (nombre de paramètres d'une porte ~ proportionnel à in_features * out_features,
    avec in_features = input_size + hidden_size pour la porte GRU/LSTM concaténée
    et out_features = hidden_size). Approximation à un facteur multiplicatif de
    base d'arête près (identique des deux côtés, donc s'annule).

    Returns:
        hidden_size_gru : int, arrondi à l'entier le plus proche.
    """
    target = n_gates_lstm * hidden_size_lstm * (input_size + hidden_size_lstm)
    a, b, c = n_gates_gru, n_gates_gru * input_size, -target
    h = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return max(1, round(h))


def _make_gate(concat_size: int, hidden_size: int, basis_name: str,
               basis_kwargs: dict, node_types=None) -> GenericKANLayer:
    cls = BASIS_REGISTRY[basis_name]
    kwargs = basis_kwargs if basis_kwargs is not None else cls.budget_kwargs
    basis = cls(in_features=concat_size, out_features=hidden_size, **kwargs)
    return GenericKANLayer(basis, in_features=concat_size, out_features=hidden_size,
                            node_types=node_types)


class GRUKANCell(nn.Module):
    """
    Cellule récurrente GRU-KAN (3 portes, cf. module docstring).

    Args:
        input_size   : nombre de features par pas de temps
        hidden_size  : dimension de l'état caché h_t
        gate_bases   : dict {"reset": nom_base, "update": nom_base, "candidate": nom_base}
                       nom_base doit exister dans edges.BASIS_REGISTRY. Défaut :
                       "hybrid" pour les 3 portes (parité avec TKANCell par défaut).
        gate_kwargs  : dict optionnel {nom_porte: kwargs de la base}. Si absent
                       pour une porte, utilise cls.budget_kwargs (budget iso B=12).
        mult_pairs   : cf. TKANCell  paires MultKAN par porte {nom_porte: {j: (i1,i2)}}.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 gate_bases: dict = None, gate_kwargs: dict = None,
                 mult_pairs: dict = None):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        concat_size      = input_size + hidden_size

        gate_bases  = gate_bases  or {}
        gate_kwargs = gate_kwargs or {}
        mult_pairs  = mult_pairs  or {}

        def node_types_for(gate):
            pairs = mult_pairs.get(gate)
            if not pairs:
                return None
            nt = ["add"] * hidden_size
            for j, (i1, i2) in pairs.items():
                nt[j] = (i1, i2)
            return nt

        self.reset_gate     = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("reset", "hybrid"),
                                          gate_kwargs.get("reset"),
                                          node_types_for("reset"))
        self.update_gate    = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("update", "hybrid"),
                                          gate_kwargs.get("update"),
                                          node_types_for("update"))
        self.candidate_gate = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("candidate", "hybrid"),
                                          gate_kwargs.get("candidate"),
                                          node_types_for("candidate"))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        combined = torch.cat([h_prev, x_t], dim=-1)
        r_t = torch.sigmoid(self.reset_gate(combined))
        z_t = torch.sigmoid(self.update_gate(combined))

        combined_n = torch.cat([r_t * h_prev, x_t], dim=-1)
        n_t = torch.tanh(self.candidate_gate(combined_n))

        h_t = (1 - z_t) * n_t + z_t * h_prev
        return h_t

    def forward_with_reg(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        combined = torch.cat([h_prev, x_t], dim=-1)
        r_raw, l1_r, ent_r = self.reset_gate.forward_with_reg(combined)
        z_raw, l1_z, ent_z = self.update_gate.forward_with_reg(combined)
        r_t = torch.sigmoid(r_raw)
        z_t = torch.sigmoid(z_raw)

        combined_n = torch.cat([r_t * h_prev, x_t], dim=-1)
        n_raw, l1_n, ent_n = self.candidate_gate.forward_with_reg(combined_n)
        n_t = torch.tanh(n_raw)

        h_t = (1 - z_t) * n_t + z_t * h_prev
        return h_t, l1_r + l1_z + l1_n, ent_r + ent_z + ent_n

    def l1_norm(self) -> torch.Tensor:
        return (self.reset_gate.l1_norm() + self.update_gate.l1_norm()
                + self.candidate_gate.l1_norm())


class GRUMKANScorer(nn.Module):
    """
    Pipeline complet GRU-KAN, miroir de MKANScorer (../cell.py) : déroulé de
    GRUKANCell sur une fenêtre glissante + projection linéaire + sigmoïde.

    score(x_{1:W}) = sigmoid(w^T h_W + b)   (mêmes conventions que eq. 4.16)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 gate_bases: dict = None, gate_kwargs: dict = None,
                 mult_pairs: dict = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUKANCell(input_size, hidden_size,
                                gate_bases=gate_bases, gate_kwargs=gate_kwargs,
                                mult_pairs=mult_pairs)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, x_window: torch.Tensor, return_reg: bool = False):
        batch, W, _ = x_window.shape
        device = x_window.device
        h_t = torch.zeros(batch, self.hidden_size, device=device, dtype=torch.float32)

        if return_reg:
            reg_l1      = torch.zeros(1, dtype=torch.float32, device=device)
            reg_entropy = torch.zeros(1, dtype=torch.float32, device=device)

        for t in range(W):
            x_t = x_window[:, t, :]
            if return_reg:
                h_t, l1_t, ent_t = self.cell.forward_with_reg(x_t, h_t)
                reg_l1.add_(l1_t)
                reg_entropy.add_(ent_t)
            else:
                h_t = self.cell(x_t, h_t)

        score = torch.sigmoid(self.projection(h_t)).squeeze(-1)
        if return_reg:
            return score, reg_l1, reg_entropy
        return score

    def l1_norm(self) -> torch.Tensor:
        return self.cell.l1_norm()
