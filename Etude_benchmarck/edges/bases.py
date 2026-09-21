"""
bases.py  Bases d'arête KAN alternatives (Étape 1, protocole_benchmark.md §8).

Chaque classe implémente UNIQUEMENT le calcul des activations d'arête :

    forward(x) : x (batch, in_features) -> (batch, in_features, out_features)

soit exactement ce que fait `HybridEdgeFunction`/`HybridKANLayer.edge_activations`
dans le modèle de référence. L'agrégation additive/multiplicative (nœuds MultKAN),
la régularisation L1/entropie et le calcul de forward_with_reg sont mutualisés dans
`GenericKANLayer` (layer.py)  une base n'a donc pas à les réimplémenter.

Toutes les bases sont calibrées au budget iso B=12 paramètres/arête (protocole
§3.1) via leurs valeurs par défaut ; un dict `native_kwargs` de réglage natif
optimal (tel que publié dans chaque papier) est fourni pour l'autre point de
mesure prescrit par le protocole.

Sources (voir amélioration_MKAN.md pour les fiches techniques complètes) :
    HybridBasis    référence MKAN (Gaussienne FastKAN + Fourier KAN-AD, eq. 4.13)
    GaussianBasis  FastKAN       (Li, 2024)
    RSWAFBasis     FasterKAN     (Ta, 2024  variante rapide introduite dans BSRBF-KAN)
    BSplineBasis   EfficientKAN  (Blealtan, 2024 ; Liu et al. 2024 pour la formule)
    ChebyshevBasis Chebyshev-KAN (Sidharth et al., 2024)
    WaveletBasis   Wav-KAN       (Bozorgasl & Chen, 2024)
    ReLUKANBasis   ReLU-KAN      (Qiu et al., 2024)
    FourierBasis   Fourier pur   (composante KAN-AD isolée)
    JacobiBasis    fKAN          (Aghaei, 2024  Jacobi fractionnaire simplifié)
    SincBasis      SincKAN       (interpolation de Sinc, Sugihara & Matsuo, 2004
                                  absente du registre historique, ajoutée pour le
                                  benchmark synthétique Niveau 0 de l'Étape 3)
    LinearBasis    baseline non-KAN (w·x, 1 paramètre/arête)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bmm_edges(basis_vals: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Contraction commune à toutes les bases "poids linéaires sur une famille de
    fonctions fixes/paramétriques" :

        basis_vals : (batch, in_features, n_basis)
        weight     : (in_features, out_features, n_basis)
        -> (batch, in_features, out_features) = sum_b basis_vals[...,b] * weight[...,b]

    Implémenté en bmm (pattern de hybrid_layer.py) plutôt qu'en einsum : compatible
    DirectML, une seule opération BLAS.
    """
    return torch.bmm(
        basis_vals.permute(1, 0, 2),
        weight.permute(0, 2, 1)
    ).permute(1, 0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Référence MKAN : Gaussienne (FastKAN) + Fourier (KAN-AD)
# ─────────────────────────────────────────────────────────────────────────────

class HybridBasis(nn.Module):
    """Réplique de HybridEdgeFunction (eq. 4.13). M + 2K paramètres/arête."""

    budget_kwargs = dict(M=8, K=2)
    native_kwargs = dict(M=8, K=2)

    def __init__(self, in_features: int, out_features: int,
                 M: int = 8, K: int = 2, domain: float = 1.0, h: float = None):
        super().__init__()
        centers = torch.linspace(-domain, domain, M)
        self.register_buffer("centers", centers)
        if h is None:
            h = 2 * domain / (M - 1)
        self.register_buffer("neg_half_over_h2", torch.tensor(-0.5 * float(h) ** -2))
        self.register_buffer("k_idx", torch.arange(1, K + 1, dtype=torch.float32))

        self.w_gauss   = nn.Parameter(torch.randn(in_features, out_features, M) * 0.1)
        self.a_fourier = nn.Parameter(torch.randn(in_features, out_features, K) * 0.1)
        self.b_fourier = nn.Parameter(torch.randn(in_features, out_features, K) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_col = x.unsqueeze(-1)
        diff  = x_col - self.centers
        gauss = torch.exp(self.neg_half_over_h2 * diff.square())
        kx    = x_col * self.k_idx
        gauss_out   = _bmm_edges(gauss, self.w_gauss)
        fourier_out = (_bmm_edges(torch.cos(kx), self.a_fourier)
                       + _bmm_edges(torch.sin(kx), self.b_fourier))
        return gauss_out + fourier_out

    def l1_norm(self) -> torch.Tensor:
        return (self.w_gauss.abs().sum() + self.a_fourier.abs().sum()
                + self.b_fourier.abs().sum())


# ─────────────────────────────────────────────────────────────────────────────
# FastKAN  RBF gaussienne (Li, 2024)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianBasis(nn.Module):
    """FastKAN : phi(x) = sum_m w_m exp(-(x-mu_m)^2 / 2h^2). M paramètres/arête."""

    budget_kwargs = dict(M=12)
    native_kwargs = dict(M=8)   # 8 centres RBF, réglage publié (Li, 2024)

    def __init__(self, in_features: int, out_features: int,
                 M: int = 12, domain: float = 1.0, h: float = None):
        super().__init__()
        centers = torch.linspace(-domain, domain, M)
        self.register_buffer("centers", centers)
        if h is None:
            h = 2 * domain / (M - 1)
        self.register_buffer("neg_half_over_h2", torch.tensor(-0.5 * float(h) ** -2))
        self.weight = nn.Parameter(torch.randn(in_features, out_features, M) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff  = x.unsqueeze(-1) - self.centers
        basis = torch.exp(self.neg_half_over_h2 * diff.square())
        return _bmm_edges(basis, self.weight)

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# FasterKAN  RSWAF (introduite dans BSRBF-KAN, Ta 2024)
# ─────────────────────────────────────────────────────────────────────────────

class RSWAFBasis(nn.Module):
    """FasterKAN : phi(x) = sum_m w_m [1 - tanh^2((x-mu_m)/h)]. M paramètres/arête."""

    budget_kwargs = dict(M=12)
    native_kwargs = dict(M=8)

    def __init__(self, in_features: int, out_features: int,
                 M: int = 12, domain: float = 1.0, h: float = None):
        super().__init__()
        centers = torch.linspace(-domain, domain, M)
        self.register_buffer("centers", centers)
        if h is None:
            h = 2 * domain / (M - 1)
        self.register_buffer("inv_h", torch.tensor(1.0 / float(h)))
        self.weight = nn.Parameter(torch.randn(in_features, out_features, M) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t     = (x.unsqueeze(-1) - self.centers) * self.inv_h
        basis = 1.0 - torch.tanh(t).square()
        return _bmm_edges(basis, self.weight)

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# EfficientKAN  B-splines d'ordre 3 (Blealtan, 2024)
# ─────────────────────────────────────────────────────────────────────────────

class BSplineBasis(nn.Module):
    """
    B-splines cubiques (ordre k=3) sur grille uniforme fixe de G intervalles.
    n_basis = G + k. G=9, k=3 -> 12 paramètres/arête.

    Note de source : la récursion de Cox-de Boor utilisée ici est la définition
    standard (non spécifique à un papier), reprise par le KAN original (Liu et al.,
    2024, arXiv:2404.19756  absent de latex/) et par l'implémentation EfficientKAN
    (Blealtan). Les papiers présents dans latex/ (ReLU-KAN §"B-splines Application in
    KAN", BSRBF-KAN) mentionnent explicitement ne pas redériver cette formule
    ("nous ne la détaillerons pas ici, la spécialisation et la complexité de sa
    définition n'étant pas centrales à l'idée de KAN"). Grille fixe (non adaptative) :
    l'extension de grille adaptative (eq. 4.17-4.18 de MKAN) n'est pas répliquée ici,
    seule la forme fonctionnelle B-spline l'est.
    """

    budget_kwargs = dict(G=9, k=3)
    native_kwargs = dict(G=5, k=3)   # grille "5 grilles" citée dans FastKAN vs efficient_kan

    def __init__(self, in_features: int, out_features: int,
                 G: int = 9, k: int = 3, domain: float = 1.0):
        super().__init__()
        self.k = k
        h = 2 * domain / G
        # Vecteur de nœuds étendu (G + 2k + 1 nœuds), partagé entre toutes les arêtes.
        grid = (torch.arange(-k, G + k + 1, dtype=torch.float32) * h - domain)
        grid = grid.unsqueeze(0).expand(in_features, -1).contiguous()   # (in, n_knots)
        self.register_buffer("grid", grid)
        n_basis = G + k
        self.weight = nn.Parameter(torch.randn(in_features, out_features, n_basis) * 0.1)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_features) -> (batch, in_features, G+k), récursion de Cox-de Boor."""
        grid = self.grid                                    # (in, n_knots)
        x_e  = x.unsqueeze(-1)                               # (batch, in, 1)
        bases = ((x_e >= grid[:, :-1]) & (x_e < grid[:, 1:])).to(x.dtype)   # ordre 0
        for kk in range(1, self.k + 1):
            left_num  = x_e - grid[:, : -(kk + 1)]
            left_den  = grid[:, kk:-1] - grid[:, : -(kk + 1)]
            right_num = grid[:, kk + 1:] - x_e
            right_den = grid[:, kk + 1:] - grid[:, 1:-kk]
            left  = left_num  / (left_den  + 1e-12) * bases[:, :, :-1]
            right = right_num / (right_den + 1e-12) * bases[:, :, 1:]
            bases = left + right
        return bases                                         # (batch, in, G+k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._b_splines(x.clamp(self.grid[0, self.k], self.grid[0, -self.k - 1]))
        return _bmm_edges(basis, self.weight)

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev-KAN (Sidharth et al., 2024)
# ─────────────────────────────────────────────────────────────────────────────

class ChebyshevBasis(nn.Module):
    """
    T_n(x) = cos(n arccos(x)), récurrence T_n = 2x T_{n-1} - T_{n-2}.
    Normalisation tanh obligatoire (papier, section ablation). degree+1 paramètres/arête.
    """

    budget_kwargs = dict(degree=11)
    native_kwargs = dict(degree=3)   # optimum publié (97.18% MNIST, ablation degré)

    def __init__(self, in_features: int, out_features: int, degree: int = 11):
        super().__init__()
        self.degree = degree
        # Initialisation Xavier (meilleure globale selon l'ablation du papier)
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.coeff = nn.Parameter(
            (torch.rand(in_features, out_features, degree + 1) * 2 - 1) * bound
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        T0 = torch.ones_like(x)
        T1 = x
        Ts = [T0, T1]
        for _ in range(2, self.degree + 1):
            Ts.append(2 * x * Ts[-1] - Ts[-2])
        T = torch.stack(Ts[: self.degree + 1], dim=-1)   # (batch, in, degree+1)
        return _bmm_edges(T, self.coeff)

    def l1_norm(self) -> torch.Tensor:
        return self.coeff.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# Wav-KAN (Bozorgasl & Chen, 2024)
# ─────────────────────────────────────────────────────────────────────────────

class WaveletBasis(nn.Module):
    """
    n_wavelets ondelettes translatées/mises à l'échelle par arête, poids w, translation
    tau, échelle s tous appris (3 paramètres par ondelette). n_wavelets=4 -> 12/arête.

    Formules exactes de la Table 1 de Wav-KAN (Bozorgasl & Chen, 2024), t=(x-tau)/s :
        mexican_hat : psi(t) = 2/(sqrt(3) pi^(1/4)) * (t^2 - 1) * exp(-t^2/2)
        morlet      : psi(t) = cos(w0 t) * exp(-t^2/2), w0 = 5
        dog         : psi(t) = -d/dt[exp(-t^2/2)] = t * exp(-t^2/2)
    Le signe de mexican_hat n'affecte pas l'expressivité (poids w appris, cf. note
    de bas de page du papier sur la convention pywavelets) mais est reproduit tel
    quel pour fidélité à la source. wavelet in {"mexican_hat","morlet","dog"} 
    Shannon/Bump exclues (le papier montre qu'elles sous-performent systématiquement).
    """

    budget_kwargs = dict(n_wavelets=4, wavelet="mexican_hat")
    native_kwargs = dict(n_wavelets=1, wavelet="mexican_hat")   # une ondelette, réglage papier

    _MEXICAN_HAT_NORM = 2.0 / (math.sqrt(3.0) * math.pi ** 0.25)

    def __init__(self, in_features: int, out_features: int,
                 n_wavelets: int = 4, wavelet: str = "mexican_hat"):
        super().__init__()
        assert wavelet in ("mexican_hat", "morlet", "dog")
        self.wavelet = wavelet
        self.tau    = nn.Parameter(torch.randn(in_features, out_features, n_wavelets) * 0.1)
        self.scale  = nn.Parameter(torch.ones(in_features, out_features, n_wavelets))
        self.weight = nn.Parameter(torch.randn(in_features, out_features, n_wavelets) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_col = x.unsqueeze(-1).unsqueeze(-1)                 # (batch, in, 1, 1)
        t = (x_col - self.tau) / (self.scale.abs() + 1e-3)    # (batch, in, out, n_wavelets)
        if self.wavelet == "mexican_hat":
            psi = self._MEXICAN_HAT_NORM * (t.square() - 1) * torch.exp(-0.5 * t.square())
        elif self.wavelet == "morlet":
            psi = torch.cos(5.0 * t) * torch.exp(-0.5 * t.square())
        else:  # dog
            psi = t * torch.exp(-0.5 * t.square())
        return (psi * self.weight).sum(dim=-1)

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# ReLU-KAN (Qiu et al., 2024)
# ─────────────────────────────────────────────────────────────────────────────

class ReLUKANBasis(nn.Module):
    """
    R_i(x) = [ReLU(e_i-x) * ReLU(x-s_i)]^2 * 16/(e_i-s_i)^4, s_i/e_i entraînables.
    3 paramètres par base (centre, largeur, poids) x n_bases. n_bases=4 -> 12/arête.
    """

    budget_kwargs = dict(n_bases=4)
    native_kwargs = dict(n_bases=4)

    def __init__(self, in_features: int, out_features: int,
                 n_bases: int = 4, domain: float = 1.0):
        super().__init__()
        centers = torch.linspace(-domain, domain, n_bases)
        self.center = nn.Parameter(centers.repeat(in_features, out_features, 1).clone())
        init_half = (2 * domain / n_bases) * 0.75
        self.width_raw = nn.Parameter(
            torch.full((in_features, out_features, n_bases),
                       math.log(math.exp(init_half) - 1.0))   # inverse softplus
        )
        self.weight = nn.Parameter(torch.randn(in_features, out_features, n_bases) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_col = x.unsqueeze(-1).unsqueeze(-1)                 # (batch, in, 1, 1)
        halfwidth = F.softplus(self.width_raw) + 1e-3         # (in, out, n_bases)
        s = self.center - halfwidth
        e = self.center + halfwidth
        left  = F.relu(e - x_col)
        right = F.relu(x_col - s)
        R = (left * right).square() * (16.0 / (e - s).pow(4))
        return (R * self.weight).sum(dim=-1)

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# Fourier pur (composante KAN-AD isolée)
# ─────────────────────────────────────────────────────────────────────────────

class FourierBasis(nn.Module):
    """
    phi(x) = sum_k a_k cos(kx) + b_k sin(kx). 2K paramètres/arête.

    Note de source : composante Fourier isolée de HybridBasis/eq. 4.13 du mémoire
    MKAN (KAN-AD, cf. amélioration_MKAN.md)  ne correspond à aucun des 6 papiers
    de latex/ pris isolément ; c'est le point de comparaison "sans la partie
    gaussienne" de la référence du mémoire, pas une base tierce à valider contre
    une source externe.
    """

    budget_kwargs = dict(K=6)
    native_kwargs = dict(K=6)

    def __init__(self, in_features: int, out_features: int, K: int = 6):
        super().__init__()
        self.register_buffer("k_idx", torch.arange(1, K + 1, dtype=torch.float32))
        self.a = nn.Parameter(torch.randn(in_features, out_features, K) * 0.1)
        self.b = nn.Parameter(torch.randn(in_features, out_features, K) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kx = x.unsqueeze(-1) * self.k_idx
        return _bmm_edges(torch.cos(kx), self.a) + _bmm_edges(torch.sin(kx), self.b)

    def l1_norm(self) -> torch.Tensor:
        return self.a.abs().sum() + self.b.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# SincKAN  interpolation de Sinc multi-résolution (Sugihara & Matsuo, 2004)
# ─────────────────────────────────────────────────────────────────────────────

class SincBasis(nn.Module):
    """
    SincKAN : interpolation de Sinc/Whittaker-Shannon, sinc(t) = sin(pi t)/(pi t)
    (sinc normalisé, sinc(0)=1 par convention torch.sinc). Convergence
    exponentielle garantie par le théorème de Sugihara & Matsuo (2004) pour
    f analytique à décroissance exponentielle sur une bande de Hardy H^1(D_d)
    (cf. step_1 §1.5, step_3 Proposition prop:sinc_convergence).

    Calibration budget iso B=12 (step_3, tableau~\\ref{tab:budget_iso}) :
    N=6 points d'interpolation également espacés, dupliqués sur DEUX largeurs
    de bande h1 (grossière, résolution basse fréquence) et h2 = h1/2 (fine,
    résolution haute fréquence) en parallèle, chacune pondérée par un jeu de
    coefficients indépendant :

        phi(x) = sum_{j in {1,2}} sum_{i=1}^{N} c_{i,j} sinc((x - x_i) / h_j)

    soit 2N = 12 coefficients apprenables/arête pour N=6. Absent du registre
    historique BASIS_REGISTRY (cf. extended_heuristic_search.py, probleme.md) ;
    implémenté ici spécifiquement pour le benchmark synthétique Niveau 0
    (Étape 3), qui exige les 9 bases théoriquement discutées en Étape 1/2.
    """

    budget_kwargs = dict(N=6)
    native_kwargs = dict(N=6)

    def __init__(self, in_features: int, out_features: int,
                 N: int = 6, domain: float = 1.0, h_coarse: float = None, h_fine: float = None):
        super().__init__()
        self.N = N
        centers = torch.linspace(-domain, domain, N)
        self.register_buffer("centers", centers)
        base_h = 2 * domain / (N - 1)
        h1 = h_coarse if h_coarse is not None else base_h
        h2 = h_fine   if h_fine   is not None else base_h * 0.5
        self.register_buffer("inv_h", torch.tensor([1.0 / float(h1), 1.0 / float(h2)]))
        self.weight = nn.Parameter(torch.randn(in_features, out_features, 2, N) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_col = x.unsqueeze(-1).unsqueeze(-1)                          # (batch, in, 1, 1)
        diff  = x_col - self.centers.view(1, 1, 1, -1)                 # (batch, in, 1, N)
        t     = diff * self.inv_h.view(1, 1, 2, 1)                     # (batch, in, 2, N)
        basis = torch.sinc(t)
        basis = basis.reshape(x.shape[0], x.shape[1], 2 * self.N)      # (batch, in, 2N)
        weight = self.weight.reshape(self.weight.shape[0], self.weight.shape[1], 2 * self.N)
        return _bmm_edges(basis, weight)

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# fKAN  Jacobi fractionnaire simplifié (Aghaei, 2024)
# ─────────────────────────────────────────────────────────────────────────────

class JacobiBasis(nn.Module):
    """
    fKAN (Aghaei, 2024), eq. JNB : phi(zeta) = J_q^(ELU(a,1), ELU(b,1))( phi_(0,1;sigma(g))(sigma(zeta)) )
    où phi_(0,1;gamma)(z) = 2*z^gamma - 1 (mapping fractionnaire de shift/scale, section
    "Fractional KAN"). alpha/beta/gamma entraînables et partagés par toutes les arêtes
    d'une même couche (limite le coût paramétrique à q+3 total, cf. protocole §3.1 :
    q=9 -> 12/arête en moyenne  le papier les définit par arête ; le partage est une
    simplification budgétaire documentée, pas une divergence de formule).

    Contraintes du papier : ELU(.,1) = torch.nn.functional.elu avec son alpha par
    défaut 1.0 (le "1" dans ELU(alpha,1) est le kappa de la définition de l'ELU
    elle-même, pas un facteur d'échelle appliqué à alpha) donne un codomaine
    (-1, +inf), d'où alpha, beta > -1 ; gamma in (0,1) via sigmoide.
    """

    budget_kwargs = dict(q=9)
    native_kwargs = dict(q=9)

    def __init__(self, in_features: int, out_features: int, q: int = 9):
        super().__init__()
        self.q = q
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.beta_raw  = nn.Parameter(torch.zeros(1))
        self.gamma_raw = nn.Parameter(torch.zeros(1))
        self.coeff = nn.Parameter(torch.randn(in_features, out_features, q) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.elu(self.alpha_raw) + 1e-2
        b = F.elu(self.beta_raw) + 1e-2
        g = torch.sigmoid(self.gamma_raw)
        sx = torch.sigmoid(x).clamp(min=1e-4, max=1 - 1e-4)
        z = 2.0 * sx.pow(g) - 1.0

        P0 = torch.ones_like(z)
        P1 = 0.5 * (a - b) + 0.5 * (a + b + 2) * z
        Ps = [P0, P1]
        for n in range(2, self.q):
            n_t = float(n)
            c1 = 2 * n_t * (n_t + a + b) * (2 * n_t + a + b - 2)
            c2 = (2 * n_t + a + b - 1) * (a.square() - b.square())
            c3 = (2 * n_t + a + b - 1) * (2 * n_t + a + b) * (2 * n_t + a + b - 2)
            c4 = 2 * (n_t + a - 1) * (n_t + b - 1) * (2 * n_t + a + b)
            Pn = ((c2 + c3 * z) * Ps[-1] - c4 * Ps[-2]) / (c1 + 1e-6)
            Ps.append(Pn)
        P = torch.stack(Ps[: self.q], dim=-1)      # (batch, in, q)
        return _bmm_edges(P, self.coeff)

    def l1_norm(self) -> torch.Tensor:
        return self.coeff.abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline non-KAN
# ─────────────────────────────────────────────────────────────────────────────

class LinearBasis(nn.Module):
    """phi(x) = w*x. 1 paramètre/arête  mesure le surcoût réel d'un KAN."""

    budget_kwargs = dict()
    native_kwargs = dict()

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight

    def l1_norm(self) -> torch.Tensor:
        return self.weight.abs().sum()


BASIS_REGISTRY = {
    "hybrid":    HybridBasis,
    "fastkan":   GaussianBasis,
    "fasterkan": RSWAFBasis,
    "efficientkan": BSplineBasis,
    "chebyshev": ChebyshevBasis,
    "wavkan":    WaveletBasis,
    "relukan":   ReLUKANBasis,
    "fourier":   FourierBasis,
    "fkan":      JacobiBasis,
    "sinckan":   SincBasis,
    "linear":    LinearBasis,
}
