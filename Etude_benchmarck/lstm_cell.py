"""
lstm_cell.py  Cellule LSTM-KAN configurable par base d'arête (Niveau 1).

TKANCell (../cell.py) est figée sur HybridKANLayer (arête hybride Gaussienne+
Fourier) : c'est la référence du mémoire, on n'y touche pas. ConfigurableTKANCell
en est un miroir structurel (mêmes 4 portes, mêmes eq. 4.7-4.12) mais où chaque
porte est une GenericKANLayer avec base interchangeable  nécessaire pour le
Niveau 1 du protocole (substitution par porte) et pour comparer LSTM-KAN et
GRU-KAN (gru_cell.py) à interface identique.

Avec gate_bases={"forget":"hybrid","input":"hybrid","candidate":"hybrid","output":"hybrid"}
et gate_kwargs=None (budget par défaut de chaque base), ConfigurableTKANCell est
numériquement équivalente à TKANCell (même formule, mêmes hyperparamètres M=8,K=2).
"""

import torch
import torch.nn as nn

from edges import GenericKANLayer, BASIS_REGISTRY


def _make_gate(concat_size: int, hidden_size: int, basis_name: str,
               basis_kwargs: dict, node_types=None) -> GenericKANLayer:
    cls = BASIS_REGISTRY[basis_name]
    kwargs = basis_kwargs if basis_kwargs is not None else cls.budget_kwargs
    basis = cls(in_features=concat_size, out_features=hidden_size, **kwargs)
    return GenericKANLayer(basis, in_features=concat_size, out_features=hidden_size,
                            node_types=node_types)


class ConfigurableTKANCell(nn.Module):
    """
    Cellule LSTM-KAN à 4 portes (eq. 4.7-4.12), base d'arête par porte au choix.

    Args:
        input_size, hidden_size : cf. TKANCell
        gate_bases  : dict {"forget","input","candidate","output": nom_base},
                      nom_base in edges.BASIS_REGISTRY. Défaut : "hybrid" partout
                      (= configuration de référence MKAN).
        gate_kwargs : dict optionnel {nom_porte: kwargs de la base}, sinon budget_kwargs.
        mult_pairs  : cf. TKANCell.
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

        self.forget_gate    = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("forget", "hybrid"),
                                          gate_kwargs.get("forget"), node_types_for("forget"))
        self.input_gate     = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("input", "hybrid"),
                                          gate_kwargs.get("input"), node_types_for("input"))
        self.candidate_gate = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("candidate", "hybrid"),
                                          gate_kwargs.get("candidate"), node_types_for("candidate"))
        self.output_gate    = _make_gate(concat_size, hidden_size,
                                          gate_bases.get("output", "hybrid"),
                                          gate_kwargs.get("output"), node_types_for("output"))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([h_prev, x_t], dim=-1)
        f_t     = torch.sigmoid(self.forget_gate(combined))
        i_t     = torch.sigmoid(self.input_gate(combined))
        c_tilde = torch.tanh(self.candidate_gate(combined))
        o_t     = torch.sigmoid(self.output_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def forward_with_reg(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([h_prev, x_t], dim=-1)
        f_raw, l1_f, ent_f = self.forget_gate.forward_with_reg(combined)
        i_raw, l1_i, ent_i = self.input_gate.forward_with_reg(combined)
        c_raw, l1_c, ent_c = self.candidate_gate.forward_with_reg(combined)
        o_raw, l1_o, ent_o = self.output_gate.forward_with_reg(combined)

        f_t, i_t, c_tilde, o_t = (torch.sigmoid(f_raw), torch.sigmoid(i_raw),
                                   torch.tanh(c_raw), torch.sigmoid(o_raw))
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t, l1_f + l1_i + l1_c + l1_o, ent_f + ent_i + ent_c + ent_o

    def l1_norm(self) -> torch.Tensor:
        return (self.forget_gate.l1_norm() + self.input_gate.l1_norm()
                + self.candidate_gate.l1_norm() + self.output_gate.l1_norm())


class ConfigurableMKANScorer(nn.Module):
    """Miroir de MKANScorer (../cell.py) au-dessus de ConfigurableTKANCell."""

    def __init__(self, input_size: int, hidden_size: int,
                 gate_bases: dict = None, gate_kwargs: dict = None,
                 mult_pairs: dict = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ConfigurableTKANCell(input_size, hidden_size,
                                          gate_bases=gate_bases, gate_kwargs=gate_kwargs,
                                          mult_pairs=mult_pairs)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, x_window: torch.Tensor, return_reg: bool = False):
        batch, W, _ = x_window.shape
        device = x_window.device
        h_t = torch.zeros(batch, self.hidden_size, device=device, dtype=torch.float32)
        c_t = torch.zeros(batch, self.hidden_size, device=device, dtype=torch.float32)

        if return_reg:
            reg_l1      = torch.zeros(1, dtype=torch.float32, device=device)
            reg_entropy = torch.zeros(1, dtype=torch.float32, device=device)

        for t in range(W):
            x_t = x_window[:, t, :]
            if return_reg:
                h_t, c_t, l1_t, ent_t = self.cell.forward_with_reg(x_t, h_t, c_t)
                reg_l1.add_(l1_t)
                reg_entropy.add_(ent_t)
            else:
                h_t, c_t = self.cell(x_t, h_t, c_t)

        score = torch.sigmoid(self.projection(h_t)).squeeze(-1)
        if return_reg:
            return score, reg_l1, reg_entropy
        return score

    def l1_norm(self) -> torch.Tensor:
        return self.cell.l1_norm()
