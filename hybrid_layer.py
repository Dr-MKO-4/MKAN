"""
hybrid_layer.py  Couche KAN vectorisée à fonctions d'arêtes hybrides (eq. 4.13–4.15, 4.17–4.18).

Contient :
  HybridKANLayer   couche complète in_features × out_features avec nœuds additifs
                    (eq. 4.14) et multiplicatifs MultKAN (eq. 4.15)
                    + normes L1 exactes (eq. 2.19–2.20) et entropie (section 4.3.3)
                    + extension de grille adaptative (eq. 4.17–4.18)
"""

import torch
import torch.nn as nn


class HybridKANLayer(nn.Module):
    """
    Couche KAN vectorisée à fonctions d'arêtes hybrides Gaussienne+Fourier (eq. 4.13),
    avec nœuds additifs (eq. 4.14) et nœuds multiplicatifs MultKAN (eq. 4.15).

    node_types : liste de longueur out_features. Pour chaque nœud de sortie j :
      - 'add'             → agrégation additive sur TOUTES les in_features (eq. 4.14)
      - 'mult'            → produit sur TOUTES les in_features (eq. 4.15, cas général)
      - [i1, i2, ...]     → produit sur les indices spécifiés (liste ≥ 2 éléments)
      Par défaut (node_types=None) : tous les nœuds sont additifs.

    Notes d'implémentation (section 4.2.2) :
      - Centres gaussiens partagés entre toutes les arêtes d'entrée (grille commune
        sur domaine normalisé  simplification assumée, documentée).
      - x doit arriver standardisé (pas de normalisation interne)  cohérent avec
        TopologyValidator.normalize() de la Phase 1.
      - extend_grid() casse le référencement de l'optimiseur (remplacement du
        nn.Parameter) : reconstruire l'optimiseur après extension.

    Args:
        in_features  : nombre de features en entrée
        out_features : nombre de nœuds en sortie
        node_types   : liste de 'add', 'mult', ou [i1, i2, ...] pour chaque nœud
        M            : nombre de centres gaussiens (défaut 8)
        K            : ordre de Fourier (défaut 2)
        domain       : demi-domaine normalisé (défaut 1.0)
        h            : largeur RBF ; par défaut espacement inter-centres
    """

    def __init__(self, in_features: int, out_features: int, node_types=None,
                 M: int = 8, K: int = 2, domain: float = 1.0, h: float = None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.M, self.K    = M, K

        if node_types is None:
            node_types = ["add"] * out_features
        assert len(node_types) == out_features, \
            f"node_types doit avoir {out_features} éléments"
        for nt in node_types:
            if nt not in ("add", "mult"):
                assert isinstance(nt, (list, tuple)) and len(nt) >= 2, \
                    "un nœud multiplicatif par indices doit spécifier une liste/tuple ≥ 2 éléments"
                assert all(0 <= i < in_features for i in nt), \
                    f"indices {nt} hors de [0, {in_features})"
        self.node_types  = node_types
        self.register_buffer("_add_mask", torch.tensor([nt == "add" for nt in node_types]))
        self._mult_specs = {j: nt for j, nt in enumerate(node_types) if nt != "add"}

        # Grille RBF partagée (buffers non appris)
        centers = torch.linspace(-domain, domain, M)
        self.register_buffer("centers", centers)
        if h is None:
            h = (2 * domain) / (M - 1)
        self.register_buffer("h", torch.tensor(float(h)))
        # Précompute -1/(2h²) : évite la division à chaque edge_activations (40×/batch)
        self.register_buffer("neg_half_over_h2", torch.tensor(-0.5 / float(h) ** 2))
        k_idx = torch.arange(1, K + 1, dtype=torch.float32)
        self.register_buffer("k_idx", k_idx)

        # Poids par arête (in_features, out_features, M ou K)  appris
        self.w_gauss   = nn.Parameter(torch.randn(in_features, out_features, M) * 0.1)
        self.a_fourier = nn.Parameter(torch.randn(in_features, out_features, K) * 0.1)
        self.b_fourier = nn.Parameter(torch.randn(in_features, out_features, K) * 0.1)

    # ── Activations d'arêtes ─────────────────────────────────────────────────

    def edge_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule phi_ij(x_i) pour toutes les arêtes (eq. 4.13).
        x : (batch, in_features) → sortie (batch, in_features, out_features)
        """
        # Composante gaussienne
        diff        = x.unsqueeze(-1) - self.centers          # (batch, in_features, M)
        gauss_basis = torch.exp(self.neg_half_over_h2 * diff ** 2)
        gauss_out   = torch.einsum("bim,ijm->bij", gauss_basis, self.w_gauss)

        # Composante Fourier
        kx        = x.unsqueeze(-1) * self.k_idx              # (batch, in_features, K)
        cos_kx, sin_kx = torch.cos(kx), torch.sin(kx)
        fourier_out = (torch.einsum("bik,ijk->bij", cos_kx, self.a_fourier) +
                       torch.einsum("bik,ijk->bij", sin_kx, self.b_fourier))

        return gauss_out + fourier_out                         # (batch, in_features, out_features)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_features) → (batch, out_features)"""
        edges = self.edge_activations(x)                       # (batch, in_features, out_features)

        # Fast-path : cas par défaut (tous les nœuds additifs)  pas de zeros ni nonzero
        if not self._mult_specs:
            return edges.sum(dim=1)

        batch = x.shape[0]
        out   = torch.zeros(batch, self.out_features, device=x.device, dtype=x.dtype)

        # Nœuds additifs (eq. 4.14) : somme sur toutes les in_features
        add_idx = self._add_mask.nonzero(as_tuple=True)[0]
        if len(add_idx) > 0:
            out[:, add_idx] = edges[:, :, add_idx].sum(dim=1)

        # Nœuds multiplicatifs (eq. 4.15) : Π_i φ_ij(x_i)
        for j, spec in self._mult_specs.items():
            if spec == "mult":
                # Produit sur TOUTES les in_features (cas général eq. 4.15)
                out[:, j] = edges[:, :, j].prod(dim=1)
            else:
                # Produit sur les indices spécifiés  spec est une liste [i1, i2, ...]
                val = edges[:, spec[0], j]
                for idx in spec[1:]:
                    val = val * edges[:, idx, j]
                out[:, j] = val

        return out

    # ── Régularisation (eq. 2.19–2.20, section 4.3.2–4.3.3) ─────────────────

    def exact_l1_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matrice L1 exacte par arête (eq. 2.19) : mean_batch |φ_ij(x_i)|

        Args:
            x : (batch, in_features)  batch courant

        Returns:
            l1_mat : (in_features, out_features)  importance par arête
        """
        edges = self.edge_activations(x)          # (batch, in_features, out_features)
        return edges.abs().mean(dim=0)            # (in_features, out_features)

    def layer_l1_total(self, x: torch.Tensor) -> torch.Tensor:
        """
        Norme L1 scalaire de la couche (eq. 2.20) : Σ_i Σ_j |φ_ij|₁
        Utilisé comme terme de régularisation dans ℓ_total (section 4.3.4).
        """
        return self.exact_l1_norm(x).sum()

    def layer_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entropie S(Φ_l) des contributions relatives des arêtes (section 4.3.3).

        S(Φ_l) = -Σ_i Σ_j p_ij · log(p_ij)
        où p_ij = |φ_ij|₁ / |Φ_l|₁

        Maximale si toutes les arêtes contribuent également (réseau dense) ;
        minimale si une seule arête concentre toute la contribution.
        La tension L1/entropie dans ℓ_total produit un réseau parcimonieux
        à contributions équiréparties  favorable à la régression symbolique.
        """
        l1_mat = self.exact_l1_norm(x)            # (in_features, out_features)
        total  = l1_mat.sum() + 1e-12
        p      = (l1_mat / total).view(-1)
        return -(p * torch.log(p + 1e-12)).sum()

    def forward_with_reg(self, x: torch.Tensor):
        """
        Forward + régularisation (L1 + entropie) en un seul appel à edge_activations.

        Remplace l'enchaînement forward() + layer_l1_total() + layer_entropy() qui
        appelait edge_activations 3× par porte par pas de temps (3× plus lent).

        Returns:
            out      : (batch, out_features)
            l1_total : scalaire  Σ_ij |φ_ij|₁  (eq. 2.20)
            entropy  : scalaire  S(Φ_l)         (section 4.3.3)
        """
        edges = self.edge_activations(x)   # (batch, in_features, out_features)  1 seul appel

        # Forward (identique à forward())
        if not self._mult_specs:
            out = edges.sum(dim=1)
        else:
            batch = x.shape[0]
            out   = torch.zeros(batch, self.out_features, device=x.device, dtype=x.dtype)
            add_idx = self._add_mask.nonzero(as_tuple=True)[0]
            if len(add_idx) > 0:
                out[:, add_idx] = edges[:, :, add_idx].sum(dim=1)
            for j, spec in self._mult_specs.items():
                if spec == "mult":
                    out[:, j] = edges[:, :, j].prod(dim=1)
                else:
                    val = edges[:, spec[0], j]
                    for idx in spec[1:]:
                        val = val * edges[:, idx, j]
                    out[:, j] = val

        # Régularisation dérivée des mêmes activations
        l1_mat   = edges.abs().mean(dim=0)          # (in_features, out_features)
        l1_total = l1_mat.sum()
        p        = (l1_mat / (l1_total + 1e-12)).view(-1)
        entropy  = -(p * torch.log(p + 1e-12)).sum()

        return out, l1_total, entropy

    def l1_norm(self) -> torch.Tensor:
        """
        Proxy de |Φ_l|₁ basé sur les poids (non conforme à eq. 2.19).
        Conservé pour compatibilité avec les appels sans batch.
        Utiliser layer_l1_total(x) pendant l'entraînement.
        """
        return (self.w_gauss.abs().sum()
                + self.a_fourier.abs().sum()
                + self.b_fourier.abs().sum())

    # ── Grid extension adaptative (eq. 4.17–4.18) ────────────────────────────

    def extend_grid(self, region: tuple, n_new: int) -> int:
        """
        Insère n_new nouveaux centres gaussiens dans [xl, xr] (eq. 4.17) et
        initialise leurs poids par interpolation linéaire des voisins (eq. 4.18).

        La grille est partagée entre toutes les features en entrée (simplification
        documentée section 4.2.2) : une extension en réponse à la dérive sur une
        feature étend la résolution pour toutes les arêtes de la couche.

        ⚠ Remplace le nn.Parameter w_gauss  reconstruire l'optimiseur après appel.

        Args:
            region : tuple (xl, xr)  intervalle où ajouter des centres
            n_new  : nombre de centres à insérer (> 0)

        Returns:
            nouveau M (nombre total de centres après extension)
        """
        xl, xr = region
        if n_new <= 0:
            return self.M

        new_centers  = torch.linspace(xl, xr, n_new + 2, device=self.centers.device)[1:-1]   # exclure les bords
        old_centers  = self.centers.clone()
        old_w_gauss  = self.w_gauss.data.clone()                  # (in, out, M_old)

        # Initialisation par interpolation linéaire (eq. 4.18)
        new_w = torch.zeros(self.in_features, self.out_features, len(new_centers),
                            device=old_w_gauss.device, dtype=old_w_gauss.dtype)
        for k in range(len(new_centers)):
            muk   = new_centers[k]
            diffs = old_centers - muk
            right_mask = diffs >= 0
            left_mask  = diffs <  0

            if not right_mask.any():
                new_w[:, :, k] = old_w_gauss[:, :, -1]
            elif not left_mask.any():
                new_w[:, :, k] = old_w_gauss[:, :, 0]
            else:
                i_left  = left_mask.nonzero(as_tuple=True)[0][-1].item()
                i_right = right_mask.nonzero(as_tuple=True)[0][0].item()
                mu_left  = old_centers[i_left]
                mu_right = old_centers[i_right]
                t = (muk - mu_left) / (mu_right - mu_left + 1e-12)
                new_w[:, :, k] = ((1 - t) * old_w_gauss[:, :, i_left]
                                  + t      * old_w_gauss[:, :, i_right])

        # Fusionner et trier par position
        merged_centers, sort_idx = torch.sort(torch.cat([old_centers, new_centers]))
        merged_w = torch.cat([old_w_gauss, new_w], dim=-1)[:, :, sort_idx]

        # Mettre à jour les buffers et le paramètre
        self.M = merged_centers.shape[0]
        self.register_buffer("centers", merged_centers)
        self.w_gauss = nn.Parameter(merged_w)

        return self.M
