"""
layer.py  Couche KAN générique paramétrable par base d'arête (Étape 1).

GenericKANLayer factorise tout ce qui, dans HybridKANLayer (../hybrid_layer.py),
ne dépend PAS de la forme de phi_ij : agrégation additive (eq. 4.14) / MultKAN
(eq. 4.15), normes L1 exactes (eq. 2.19-2.20), entropie (section 4.3.3),
forward_with_reg. La seule chose que fournit une base (bases.py) est
`edge_activations(x) -> (batch, in_features, out_features)`.

Ainsi la boucle d'entraînement, la loss régularisée (loss.py) et l'audit
(audit.py) fonctionnent à l'identique quelle que soit la base branchée, comme
l'exige le protocole (protocole_benchmark.md §8, étape 1).
"""

import torch
import torch.nn as nn


class GenericKANLayer(nn.Module):
    """
    Couche KAN in_features x out_features avec base d'arête interchangeable.

    Args:
        basis        : nn.Module d'une des classes de bases.py, déjà instancié
                       avec (in_features, out_features, **basis_kwargs).
        in_features  : nombre de features en entrée
        out_features : nombre de nœuds en sortie
        node_types   : cf. HybridKANLayer  'add' (défaut), 'mult', ou [i1, i2, ...]
    """

    def __init__(self, basis: nn.Module, in_features: int, out_features: int,
                 node_types=None):
        super().__init__()
        self.basis        = basis
        self.in_features  = in_features
        self.out_features = out_features

        if node_types is None:
            node_types = ["add"] * out_features
        assert len(node_types) == out_features
        for nt in node_types:
            if nt not in ("add", "mult"):
                assert isinstance(nt, (list, tuple)) and len(nt) >= 2
                assert all(0 <= i < in_features for i in nt)
        self.node_types = node_types
        self.register_buffer("_add_mask", torch.tensor([nt == "add" for nt in node_types]))
        self.register_buffer("_add_idx", self._add_mask.nonzero(as_tuple=True)[0])
        self._mult_specs = {j: nt for j, nt in enumerate(node_types) if nt != "add"}

    def edge_activations(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_features) -> (batch, in_features, out_features)."""
        return self.basis(x)

    def _aggregate(self, edges: torch.Tensor, batch: int,
                    device, dtype) -> torch.Tensor:
        if not self._mult_specs:
            return edges.sum(dim=1)
        out = torch.zeros(batch, self.out_features, device=device, dtype=dtype)
        if len(self._add_idx) > 0:
            out[:, self._add_idx] = edges[:, :, self._add_idx].sum(dim=1)
        for j, spec in self._mult_specs.items():
            if spec == "mult":
                out[:, j] = edges[:, :, j].prod(dim=1)
            else:
                val = edges[:, spec[0], j]
                for idx in spec[1:]:
                    val = val * edges[:, idx, j]
                out[:, j] = val
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edges = self.edge_activations(x)
        return self._aggregate(edges, x.shape[0], x.device, x.dtype)

    def exact_l1_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Matrice L1 exacte par arête (eq. 2.19) : mean_batch |phi_ij(x_i)|."""
        return self.edge_activations(x).abs().mean(dim=0)

    def layer_l1_total(self, x: torch.Tensor) -> torch.Tensor:
        return self.exact_l1_norm(x).sum()

    def layer_entropy(self, x: torch.Tensor) -> torch.Tensor:
        l1_mat = self.exact_l1_norm(x)
        total  = l1_mat.sum() + 1e-12
        p      = (l1_mat / total).reshape(-1)
        return -(p * torch.log(p + 1e-12)).sum()

    def forward_with_reg(self, x: torch.Tensor):
        edges   = self.edge_activations(x)
        out     = self._aggregate(edges, x.shape[0], x.device, x.dtype)
        l1_mat  = edges.abs().mean(dim=0)
        l1_total = l1_mat.sum()
        p       = (l1_mat / (l1_total + 1e-12)).reshape(-1)
        entropy = -(p * torch.log(p + 1e-12)).sum()
        return out, l1_total, entropy

    def l1_norm(self) -> torch.Tensor:
        """Proxy par poids (non conforme à eq. 2.19, cf. HybridKANLayer.l1_norm)."""
        return self.basis.l1_norm()
