"""
extended_heuristic_search.py  Étape 2 : extension de la recherche heuristique
à un espace hiérarchique conditionnel (base KAN par porte, cellule TKAN/GRU,
régime d'entrée) + fitness multi-objectifs normalisée avec pénalité de latence.

Référence : step_2_search_extension_and_analysis.tex (section "Étape 2").

─── Écarts documentés entre la théorie (step_2_*.tex) et le code réel ────────
Le tableau théorique de l'espace conditionnel (section "Formalisation de
l'espace de recherche hiérarchique conditionnel") énumère des bases (SincKAN,
rKAN à approximants de Padé (m,n), ReLU-KAN avec ordre p variable, ChebyKAN
avec choix de normalisation de domaine, Wav-KAN avec ondelette Shannon et
schéma d'initialisation Scale_Init) qui NE SONT PAS implémentées dans le
dépôt. Le dépôt réel (MKAN/Etude_benchmarck/edges/bases.py, protocole du
Niveau 1) définit un registre fermé de 10 bases interchangeables
(BASIS_REGISTRY : hybrid, fastkan, fasterkan, efficientkan, chebyshev,
wavkan, relukan, fourier, fkan, linear), chacune avec ses propres
hyperparamètres structurels réels (contraints par leur signature
__init__), et un plan de croisement par porte déjà défini
(niveau1_harness.GATE_CANDIDATES) qui NE correspond pas exactement à la
cartographie théorique Forget/Input → ReLU-KAN/rKAN/ChebyKAN,
Candidate → Wav-KAN/SincKAN, Output → FastKAN/EfficientKAN du prompt.

Conformément à la consigne « ne pas supposer qu'une classe ou une API
existe simplement parce qu'elle apparaît dans le prompt » et « créer
l'adaptateur minimal nécessaire plutôt que réécrire toute l'architecture »,
ce module :
  1. Construit ESPACE_CONDITIONNEL sur les 10 bases RÉELLEMENT
     instanciables (edges.BASIS_REGISTRY), en reprenant les noms de
     paramètres du tableau théorique quand une correspondance directe
     existe (ex. EfficientKAN : Grid_G, Spline_Degree_k → G, k ; FastKAN :
     N_centers, h_bandwidth → M, h).
  2. Documente explicitement, base par base (voir BASE_PARAM_SPACE et
     _DEVIATIONS_DOC ci-dessous), les hyperparamètres théoriques qui n'ont
     pas de contrepartie dans le code (SincKAN absent, rKAN/fKAN réduit au
     seul degré q de JacobiBasis, ReLU-KAN réduit à n_bases, ChebyKAN sans
     choix de normalisation, Wav-KAN sans Shannon ni Scale_Init).
  3. Réutilise le plan de croisement par porte du protocole Niveau 1
     (niveau1_harness.GATE_CANDIDATES / GRU_GATE_MAP), qui EST le résultat
     documenté d'une mise en compétition réelle, plutôt que de forcer la
     cartographie théorique du prompt en contrainte arbitraire (conforme à
     la consigne du prompt lui-même sur ce point).
  4. Réutilise l'infrastructure existante plutôt que de la dupliquer :
     GenericKANLayer/BASIS_REGISTRY (edges), ConfigurableMKANScorer /
     GRUMKANScorer + match_gru_hidden_size (lstm_cell.py, gru_cell.py),
     mkan_total_loss (loss.py), DMLAdam (optim.py), et surtout
     fit_symbolic_best / SYMBOLIC_LIBRARY (symbolic.py) pour R2_symbolic —
     le dépôt fait explicitement le choix de ne pas dépendre de scipy pour
     la régression symbolique (descente de gradient Adam sur
     c*f(a*x+b)+d) ; on conserve ce choix plutôt que d'introduire
     scipy.optimize.curve_fit en parallèle, ce qui dupliquerait une
     fonctionnalité déjà présente et testée avec un seuil (theta=1e-2,
     r2_threshold=0.99) qui correspond exactement à la spécification de
     l'Étape 2. sklearn.metrics EST utilisé (MCC, PR-AUC, Brier), tel
     qu'explicitement requis par la spécification de fitness.

`ExtendedHeuristicSearch` étend `RechercheHeuristique` (heuristic_search.py)
par sous-classement : les méthodes de génération/croisement/mutation
d'individus sont redéfinies pour l'espace hiérarchique conditionnel, mais
la classe reste un `RechercheHeuristique` valide (mêmes attributs de base,
mêmes conventions de nommage `_population`/`_scores`/`_journal`).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Ce module vit désormais dans MKAN/Etude_benchmarck/ (déplacé depuis MKAN/ —
# voir note de session) : edges, lstm_cell, gru_cell, niveau1_harness sont des
# modules FRÈRES (même dossier, résolus automatiquement) ; heuristic_search,
# loss, optim, symbolic vivent dans MKAN/ (le dossier parent) et doivent donc
# être explicitement ajoutés à sys.path.
_ETUDE_DIR = os.path.dirname(os.path.abspath(__file__))
_MKAN_DIR  = os.path.dirname(_ETUDE_DIR)
if _MKAN_DIR not in sys.path:
    sys.path.insert(0, _MKAN_DIR)

from heuristic_search import RechercheHeuristique
from loss import mkan_total_loss
from optim import DMLAdam

# symbolic.py fait partie du package MKAN et utilise un import relatif
# (`from .optim import DMLAdam`) : on l'importe comme sous-module d'un
# paquet "MKAN" minimal (sans exécuter MKAN/__init__.py, qui charge des
# dépendances optionnelles comme xgboost/kaleido non nécessaires ici).
import importlib.util as _importlib_util
if "MKAN" not in sys.modules:
    _spec = _importlib_util.spec_from_loader("MKAN", loader=None, is_package=True)
    _mkan_pkg = _importlib_util.module_from_spec(_spec)
    _mkan_pkg.__path__ = [_MKAN_DIR]
    sys.modules["MKAN"] = _mkan_pkg
from MKAN.symbolic import SYMBOLIC_LIBRARY

# ── Accès au registre de bases KAN interchangeables (Étape 1, Niveau 1) ──────
from edges import BASIS_REGISTRY                              # noqa: E402
from lstm_cell import ConfigurableMKANScorer                  # noqa: E402
from gru_cell import GRUMKANScorer, match_gru_hidden_size      # noqa: E402
from niveau1_harness import (                                 # noqa: E402
    build_regime_frame, standardize, build_windows,
    GATE_CANDIDATES as _NIVEAU1_GATE_CANDIDATES,
    GRU_GATE_MAP as _NIVEAU1_GRU_GATE_MAP,
)


def select_training_device(prefer: Optional[str] = None) -> torch.device:
    """
    Sélectionne le device d'ENTRAÎNEMENT (jamais celui de la mesure de latence,
    qui reste toujours CPU — cf. _measure_latency_ms). Cascade : prefer explicite
    > MPS (Apple Silicon, ex. M4) > CUDA > CPU.

    Ne concerne QUE l'entraînement/l'inférence de recherche : la contrainte de
    latence USSD (100 ms, batch=256) est une propriété du canal de production
    CPU-only, indépendante de la machine de développement — ce choix de device
    n'a donc aucune influence sur le protocole de fitness lui-même.
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════
# 1. Espace de recherche hiérarchique conditionnel
# ══════════════════════════════════════════════════════════════════════════

# Portes canoniques (nommage du mémoire, section 4.2.1) et leur porte GRU
# correspondante (GRUKANCell n'a que 3 portes : reset, update, candidate ;
# il n'existe pas d'équivalent GRU de la porte Output).
GATE_NAMES_TKAN = ["Forget", "Input", "Candidate", "Output"]
GATE_NAMES_GRU  = ["Forget", "Input", "Candidate"]     # Output absente en GRU

_CANON_TO_NIVEAU1 = {"Forget": "forget", "Input": "input",
                     "Candidate": "candidate", "Output": "output"}
_CANON_TO_GRU_GATE = {"Forget": "reset", "Input": "update", "Candidate": "candidate"}

# Bases candidates par porte : reprend le plan de croisement du protocole
# Niveau 1 (niveau1_harness.GATE_CANDIDATES), résultat documenté d'une
# analyse théorique (amélioration_MKAN.md) plutôt qu'une supposition.
GATE_CANDIDATES: dict = {
    canon: list(_NIVEAU1_GATE_CANDIDATES[_CANON_TO_NIVEAU1[canon]])
    for canon in GATE_NAMES_TKAN
}

# ── Hyperparamètres structurels par base (P(base), Table espace_conditionnel) ──
# Chaque entrée : {nom_theorique: valeurs}. `_kwarg_map` traduit ces noms
# théoriques vers les kwargs réels du constructeur de la base (bases.py).
# `_fixed_kwargs` documente les hyperparamètres théoriques SANS équivalent
# dans le code (non recherchés, valeur fixée par l'implémentation).
BASE_PARAM_SPACE: dict = {
    "hybrid": {
        "params": {"M": [4, 8, 12, 16, 24, 32], "K": [1, 2, 3, 4]},
        "kwarg_map": {"M": "M", "K": "K"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "fastkan": {
        # Table : FastKAN -> N_centers, h_bandwidth (GaussianBasis : M, h)
        "params": {"N_centers": [4, 8, 12, 16, 24, 32], "h_bandwidth": (0.05, 1.5)},
        "kwarg_map": {"N_centers": "M", "h_bandwidth": "h"},
        "continuous": {"h_bandwidth"},
        "fixed_kwargs": {},
    },
    "fasterkan": {
        # RSWAFBasis n'est pas dans la table théorique (FasterKAN, introduite
        # par BSRBF-KAN) mais fait partie du GATE_CANDIDATES réel (porte
        # Output notamment) : même paramétrisation que FastKAN (M, h).
        "params": {"N_centers": [4, 8, 12, 16, 24, 32], "h_bandwidth": (0.05, 1.5)},
        "kwarg_map": {"N_centers": "M", "h_bandwidth": "h"},
        "continuous": {"h_bandwidth"},
        "fixed_kwargs": {},
    },
    "efficientkan": {
        # Table : EfficientKAN -> Grid_G, Spline_Degree_k (BSplineBasis : G, k)
        "params": {"Grid_G": [3, 5, 8, 12, 16], "Spline_Degree_k": [2, 3, 4]},
        "kwarg_map": {"Grid_G": "G", "Spline_Degree_k": "k"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "chebyshev": {
        # Table : ChebyKAN -> n_degree, DomainNorm {tanh, Linear}.
        # DomainNorm n'existe pas dans ChebyshevBasis (normalisation tanh
        # câblée en dur) : écart documenté, DomainNorm n'est pas recherché.
        "params": {"Degree_n": [3, 5, 8, 12]},
        "kwarg_map": {"Degree_n": "degree"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "wavkan": {
        # Table : Wav-KAN -> psi_mother {Morlet,MexHat,DOG,Shannon},
        # M_wavelets, Scale_init {Linear,Log}.
        # Shannon exclue (WaveletBasis ne l'implémente pas ; le dépôt
        # documente ce choix comme fidèle à la source : Wav-KAN montre que
        # Shannon/Bump sous-performent systématiquement). Scale_init non
        # implémenté (scale toujours initialisée à 1) : écart documenté.
        "params": {"Mother_Wavelet": ["morlet", "mexican_hat", "dog"],
                   "M_wavelets": [3, 5, 8, 12]},
        "kwarg_map": {"Mother_Wavelet": "wavelet", "M_wavelets": "n_wavelets"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "relukan": {
        # Table : ReLU-KAN -> Grid_G, Order_p {2,3,4}. ReLUKANBasis n'a
        # qu'un seul hyperparamètre structurel (n_bases) ; l'exposant est
        # câblé à 2 dans la formule R_i(x) (Order_p non implémenté).
        "params": {"Grid_G": [4, 8, 12, 16, 24]},
        "kwarg_map": {"Grid_G": "n_bases"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "fourier": {
        # Composante Fourier isolée (KAN-AD) de la base hybride de référence.
        "params": {"K": [1, 2, 3, 4, 6, 8]},
        "kwarg_map": {"K": "K"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "fkan": {
        # Table : rKAN / fKAN -> m_Pade, n_Pade (approximants de Padé).
        # Le dépôt n'implémente pas rKAN (approximants de Padé) ; JacobiBasis
        # (fKAN, Aghaei 2024) est la seule base de cette famille disponible,
        # avec un unique degré structurel q. m_Pade/n_Pade n'ont pas
        # d'équivalent et ne sont donc pas recherchés (écart documenté).
        "params": {"Pade_q": [4, 6, 9, 12]},
        "kwarg_map": {"Pade_q": "q"},
        "continuous": set(),
        "fixed_kwargs": {},
    },
    "linear": {
        "params": {},
        "kwarg_map": {},
        "continuous": set(),
        "fixed_kwargs": {},
    },
}
assert set(BASE_PARAM_SPACE.keys()) == set(BASIS_REGISTRY.keys())

# ── theta_opt : hyperparamètres d'optimisation transversaux (eq. theta_opt) ──
THETA_OPT_SPACE: dict = {
    "hidden_size": [8, 16, 32, 64, 128],
    "lr":          [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
    "lam":         [1e-4, 1e-3, 1e-2, 1e-1],
    "mu1":         [0.1, 0.5, 1.0, 2.0],
    "mu2":         [0.1, 0.25, 0.5, 1.0, 2.0],
    "batch_size":  [32, 64, 128, 256],
    "W":           [5, 10, 15, 20],
}
# Hyperparamètres continus mutés par bruit gaussien à variance décroissante
# (section "Mutation", sigma_0=0.35, rho=0.97) : lr, lam, mu1, mu2 et
# h_bandwidth (spécifique aux bases FastKAN/FasterKAN). Les bornes de
# mutation sont le min/max de la grille THETA_OPT_SPACE / BASE_PARAM_SPACE.
CONTINUOUS_THETA_OPT = {"lr", "lam", "mu1", "mu2"}

CELL_TYPES     = ["TKANCell", "GRUKANCell"]
INPUT_REGIMES  = ["raw", "engineered"]                   # raw=brut, engineered=traité
_REGIME_TO_NIVEAU1 = {"raw": "brut", "engineered": "traite"}

# Représentation "documentaire" complète exigée par la spécification
# (Table espace_conditionnel) — dérivée de BASE_PARAM_SPACE/GATE_CANDIDATES,
# fournie pour audit/traçabilité (ne pilote pas directement sample_individual,
# qui utilise BASE_PARAM_SPACE + GATE_CANDIDATES par construction).
ESPACE_CONDITIONNEL: dict = {
    f"Base_{gate}": {
        base: BASE_PARAM_SPACE[base]["params"] for base in GATE_CANDIDATES[gate]
    }
    for gate in GATE_NAMES_TKAN
}
ESPACE_CONDITIONNEL.update({
    "Cell_Type":    CELL_TYPES,
    "Input_Regime": INPUT_REGIMES,
    **THETA_OPT_SPACE,
})


def hidden_size_gru_adjusted(hidden_size: int) -> int:
    """
    d_h^GRU = floor((4/3) * d_h)  (eq. ajustement_iso, section
    "Égalisation du budget paramétrique").

    3*d_h^GRU*(d_in+1) = 4*d_h*(d_in+1)  =>  d_h^GRU = (4/3)*d_h (ratio exact,
    eq. ajustement_exact) ; la troncature directe vers l'entier inférieur
    garantit P_GRU <= P_LSTM tout en restant la valeur entière la plus proche
    de ce budget parmi celles qui satisfont cette contrainte.

    CORRECTIF (ex-version de step_2_search_extension_and_analysis.tex utilisait
    floor(sqrt(4/3)*d_h) ≈ floor(1.1547*d_h), présentée comme une "approximation
    conservative couramment adoptée dans la littérature" — cette justification ne
    correspondait à aucune pratique documentée et sous-budgétait la GRUKANCell de
    ~16% par rapport à la troncature directe de 4/3 (erreur mathématique confirmée
    et corrigée dans le document de référence). Le dépôt possède par ailleurs
    match_gru_hidden_size() (gru_cell.py), qui résout l'égalité paramétrique exacte
    par porte — différente de cette formule fermée — utilisée par le protocole
    Niveau 1 (niveau1_harness.py) ; on ne la modifie pas.
    """
    return math.floor((4.0 / 3.0) * hidden_size)


# ══════════════════════════════════════════════════════════════════════════
# 2. Individu conditionnel : échantillonnage et validation
# ══════════════════════════════════════════════════════════════════════════

def _sample_base_hyperparams(base: str, rng: random.Random) -> dict:
    """Échantillonne uniquement les hyperparamètres actifs de `base`."""
    spec = BASE_PARAM_SPACE[base]
    out  = {}
    for name, domain in spec["params"].items():
        if name in spec["continuous"]:
            lo, hi = domain
            out[name] = rng.uniform(lo, hi)
        else:
            out[name] = rng.choice(domain)
    return out


def sample_individual(rng: Optional[random.Random] = None) -> dict:
    """
    Échantillonne un individu de l'espace hiérarchique conditionnel.

    Règle absolue (section 2) : seuls les hyperparamètres de la base
    effectivement choisie pour chaque porte sont stockés ; aucune porte
    n'a une base ni des hyperparamètres non applicables au type de
    cellule (Output absente si Cell_Type == "GRUKANCell").
    """
    rng = rng or random
    cell_type = rng.choice(CELL_TYPES)
    gate_names = GATE_NAMES_TKAN if cell_type == "TKANCell" else GATE_NAMES_GRU

    gates = {}
    for gate in gate_names:
        base = rng.choice(GATE_CANDIDATES[gate])
        gates[gate] = {"Base": base, **_sample_base_hyperparams(base, rng)}

    individual = {
        "Cell_Type":    cell_type,
        "Input_Regime": rng.choice(INPUT_REGIMES),
        "hidden_size":  rng.choice(THETA_OPT_SPACE["hidden_size"]),
        "lr":           rng.choice(THETA_OPT_SPACE["lr"]),
        "lam":          rng.choice(THETA_OPT_SPACE["lam"]),
        "mu1":          rng.choice(THETA_OPT_SPACE["mu1"]),
        "mu2":          rng.choice(THETA_OPT_SPACE["mu2"]),
        "batch_size":   rng.choice(THETA_OPT_SPACE["batch_size"]),
        "W":            rng.choice(THETA_OPT_SPACE["W"]),
        "gates":        gates,
    }
    return individual


def validate_individual(individual: dict, raise_on_error: bool = False) -> bool:
    """
    Vérifie : présence des bases, cohérence des sous-hyperparamètres,
    absence d'hyperparamètres d'une base inactive, validité des valeurs.
    """
    def fail(msg: str) -> bool:
        if raise_on_error:
            raise ValueError(msg)
        return False

    required_top = {"Cell_Type", "Input_Regime", "hidden_size", "lr", "lam",
                     "mu1", "mu2", "batch_size", "W", "gates"}
    if not required_top.issubset(individual.keys()):
        return fail(f"Champs manquants : {required_top - individual.keys()}")

    if individual["Cell_Type"] not in CELL_TYPES:
        return fail(f"Cell_Type invalide : {individual['Cell_Type']}")
    if individual["Input_Regime"] not in INPUT_REGIMES:
        return fail(f"Input_Regime invalide : {individual['Input_Regime']}")

    for key in ("hidden_size", "lr", "lam", "mu1", "mu2", "batch_size", "W"):
        if key not in THETA_OPT_SPACE:
            continue
        if key in CONTINUOUS_THETA_OPT:
            lo, hi = min(THETA_OPT_SPACE[key]), max(THETA_OPT_SPACE[key])
            if not (lo <= individual[key] <= hi):
                return fail(f"{key}={individual[key]} hors bornes [{lo},{hi}]")
        elif individual[key] not in THETA_OPT_SPACE[key]:
            return fail(f"{key}={individual[key]} absent de {THETA_OPT_SPACE[key]}")

    expected_gates = set(GATE_NAMES_TKAN if individual["Cell_Type"] == "TKANCell"
                         else GATE_NAMES_GRU)
    actual_gates = set(individual["gates"].keys())
    if actual_gates != expected_gates:
        return fail(f"Portes attendues {expected_gates}, reçues {actual_gates} "
                    f"(Cell_Type={individual['Cell_Type']})")

    for gate, gate_cfg in individual["gates"].items():
        base = gate_cfg.get("Base")
        if base not in GATE_CANDIDATES.get(gate, []):
            return fail(f"Base '{base}' non autorisée pour la porte {gate} "
                        f"(candidats : {GATE_CANDIDATES.get(gate)})")
        spec = BASE_PARAM_SPACE[base]
        expected_keys = {"Base", *spec["params"].keys()}
        actual_keys   = set(gate_cfg.keys())
        if actual_keys != expected_keys:
            extra   = actual_keys - expected_keys
            missing = expected_keys - actual_keys
            return fail(f"Porte {gate} (base {base}) : clés en trop {extra}, "
                        f"manquantes {missing} — hyperparamètre d'une base "
                        f"inactive détecté" if extra else
                        f"Porte {gate} (base {base}) : hyperparamètres manquants {missing}")
        for name, domain in spec["params"].items():
            val = gate_cfg[name]
            if name in spec["continuous"]:
                lo, hi = domain
                if not (lo <= val <= hi):
                    return fail(f"{gate}.{name}={val} hors bornes [{lo},{hi}]")
            elif val not in domain:
                return fail(f"{gate}.{name}={val} absent de {domain}")

    return True


def _individual_gate_kwargs(gate_cfg: dict) -> dict:
    """Traduit les hyperparamètres théoriques d'une porte vers les kwargs réels de la base."""
    base = gate_cfg["Base"]
    kwarg_map = BASE_PARAM_SPACE[base]["kwarg_map"]
    kwargs = {}
    for theo_name, real_name in kwarg_map.items():
        val = gate_cfg[theo_name]
        if isinstance(val, float) and real_name in ("M", "n_bases", "G", "k",
                                                      "degree", "n_wavelets", "q", "K"):
            val = int(round(val))
        kwargs[real_name] = val
    kwargs.update(BASE_PARAM_SPACE[base]["fixed_kwargs"])
    return kwargs


def build_model_from_individual(individual: dict, input_size: int) -> torch.nn.Module:
    """
    Reconstruit un scoreur (ConfigurableMKANScorer ou GRUMKANScorer, cf.
    Etude_benchmarck/{lstm_cell,gru_cell}.py) à partir d'un individu validé.

    Ces deux classes sont des miroirs structurels de MKANScorer
    (../cell.py) : même interface forward(x_window, return_reg=False/True)
    -> score (batch,) ∈ (0,1), donc compatibles avec mkan_total_loss et
    avec la boucle d'entraînement de train.ipynb sans adaptation.
    """
    gate_bases  = {}
    gate_kwargs = {}
    for gate, gate_cfg in individual["gates"].items():
        real_gate = (_CANON_TO_NIVEAU1[gate] if individual["Cell_Type"] == "TKANCell"
                    else _CANON_TO_GRU_GATE[gate])
        gate_bases[real_gate]  = gate_cfg["Base"]
        gate_kwargs[real_gate] = _individual_gate_kwargs(gate_cfg)

    hidden_size = int(individual["hidden_size"])
    if individual["Cell_Type"] == "TKANCell":
        return ConfigurableMKANScorer(input_size, hidden_size,
                                       gate_bases=gate_bases, gate_kwargs=gate_kwargs)
    h_gru = hidden_size_gru_adjusted(hidden_size)
    return GRUMKANScorer(input_size, h_gru,
                          gate_bases=gate_bases, gate_kwargs=gate_kwargs)


# ══════════════════════════════════════════════════════════════════════════
# 3. Fonction fitness (section 3)
# ══════════════════════════════════════════════════════════════════════════

def _safe_float(x, default=0.0) -> float:
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _measure_latency_ms(model: torch.nn.Module, W: int, input_size: int,
                         batch: int = 256, n_repeats: int = 3) -> float:
    """
    Latence d'inférence CPU (contrainte USSD, section 5) : time.perf_counter()
    sur un batch de 256 fenêtres, TOUJOURS sur CPU quel que soit le device
    d'entraînement (la contrainte de production est indépendante du device
    d'entraînement).
    """
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    x = torch.randn(batch, W, input_size, dtype=torch.float32)
    with torch.no_grad():
        model_cpu(x)  # warm-up
        best = float("inf")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model_cpu(x)
            best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def _sanitize_scores(scores: np.ndarray, context: str, max_bad_fraction: float = 0.01) -> np.ndarray:
    """
    Détecte puis répare les scores NaN/Inf produits par le modèle en
    validation. AVANT ce correctif, np.nan_to_num remplaçait silencieusement
    ces valeurs par 0.5/1.0/0.0 sans qu'aucun signal n'indique qu'une
    configuration produit des prédictions invalides — MCC/PR-AUC/Brier étaient
    alors calculés sur des scores partiellement fabriqués, avec un individu
    potentiellement défaillant recevant une fitness d'apparence normale.

    Cohérent avec le traitement d'une loss NaN pendant l'entraînement (échec
    immédiat, pas de réparation silencieuse) : au-delà de `max_bad_fraction`
    (1 % par défaut — tolère une poignée de cas limites numériques sans tuer
    des configurations par ailleurs valides), la configuration est déclarée
    instable et l'évaluation échoue explicitement plutôt que de continuer sur
    des données partiellement inventées.
    """
    bad_mask = np.isnan(scores) | np.isinf(scores)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        fraction = n_bad / len(scores)
        from tqdm.auto import tqdm
        tqdm.write(f"  [AVERTISSEMENT] {context} : {n_bad}/{len(scores)} scores NaN/Inf "
                  f"({fraction:.2%}) — configuration potentiellement instable.")
        if fraction > max_bad_fraction:
            raise RuntimeError(f"{context} : {n_bad}/{len(scores)} scores NaN/Inf "
                              f"({fraction:.2%}) dépasse le seuil de tolérance "
                              f"({max_bad_fraction:.2%}) — configuration instable rejetée.")
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _release_device_memory(device) -> None:
    """
    Force la libération de la mémoire GPU après une évaluation fitness (voir
    commentaire dans compute_fitness). gc.collect() casse les cycles de
    références autograd ; empty_cache() restitue les blocs libérés à
    MPS/CUDA (sans quoi l'allocateur "caching" de chaque backend les retient
    pour réutilisation interne, ce qui suffit à provoquer un OOM progressif
    sur MPS après quelques générations, la VRAM unifiée d'Apple Silicon étant
    partagée avec le reste du système).

    Ne lève JAMAIS : appelée depuis un bloc `finally` (compute_fitness,
    evaluate_best_on_full_validation) — si cette purge échouait elle-même
    (ex. empty_cache() instable sur un pilote MPS particulier), une exception
    non interceptée ICI remplacerait silencieusement le `return result` déjà
    construit par le try/except appelant, perdant un résultat de fitness
    parfaitement valide pour une raison de nettoyage mémoire sans rapport.
    """
    try:
        import gc
        gc.collect()
        device_type = device.type if isinstance(device, torch.device) else str(device)
        if device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:   # noqa: BLE001 — voir docstring : ne doit jamais remonter
        from tqdm.auto import tqdm
        tqdm.write(f"  [avertissement] libération mémoire device échouée (ignorée) : {exc}")


def _penalty_latency(latency_ms: float, alpha_lat: float = 10.0) -> float:
    if latency_ms <= 100.0:
        return 0.0
    return alpha_lat * ((latency_ms - 100.0) / 100.0) ** 2


# ══════════════════════════════════════════════════════════════════════════
# NSGA-II — tri par dominance contrainte (Deb, Pratab, Agarwal, Meyarivan,
# 2002, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
# IEEE Trans. Evolutionary Computation 6(2):182-197). Remplace la fitness
# scalaire agrégée comme mécanisme DE SÉLECTION de la recherche (élitisme,
# tournoi, injection de diversité, modèle EDA) — cf. ExtendedHeuristicSearch.
# La latence n'est PAS un 5e objectif libre à minimiser indéfiniment : elle
# reste une CONTRAINTE de seuil (100 ms, protocole du mémoire) traitée par le
# principe de dominance contrainte (§II.B de Deb et al.), fidèle à la
# sémantique opérationnelle (un individu à 5 ms n'est pas "meilleur" qu'un
# individu à 90 ms — les deux respectent la contrainte USSD).
# ══════════════════════════════════════════════════════════════════════════

LATENCY_CONSTRAINT_MS = 100.0   # seuil USSD du protocole — inchangé


def _objective_vector(fitness_components: dict) -> tuple:
    """4 objectifs à MAXIMISER (dominance de Pareto) : MCC_clipped, PR_AUC,
    -Brier (minimiser Brier <=> maximiser son opposé), R2_symbolic."""
    return (
        fitness_components["MCC_clipped"],
        fitness_components["PR_AUC"],
        -fitness_components["Brier"],
        fitness_components["R2_symbolic"],
    )


def _constraint_violation(fitness_components: dict,
                          threshold_ms: float = LATENCY_CONSTRAINT_MS) -> float:
    """Violation de la contrainte de latence (0.0 si respectée)."""
    return max(0.0, fitness_components["latency_ms"] - threshold_ms)


def _constrained_dominates(obj_a: tuple, cv_a: float, obj_b: tuple, cv_b: float) -> bool:
    """
    Principe de dominance contrainte (Deb et al., 2002, §II.B) : a domine b si
      (a) a est réalisable et b ne l'est pas, OU
      (b) les deux sont irréalisables et a viole moins la contrainte, OU
      (c) les deux sont réalisables et a domine b au sens de Pareto standard
          (meilleur ou égal sur tous les objectifs, strictement meilleur sur
          au moins un).
    """
    feas_a, feas_b = cv_a <= 0.0, cv_b <= 0.0
    if feas_a and not feas_b:
        return True
    if not feas_a and feas_b:
        return False
    if not feas_a and not feas_b:
        return cv_a < cv_b
    not_worse = all(a >= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a > b for a, b in zip(obj_a, obj_b))
    return not_worse and strictly_better


def _fast_non_dominated_sort(objectives: list, violations: list) -> list:
    """
    Tri non-dominé rapide (Deb et al., 2002, Algorithme 1) : partitionne les
    indices [0..n) en fronts F1 (non-dominés), F2 (dominés uniquement par des
    individus de F1), etc. Retourne une liste de fronts (listes d'indices) ;
    complexité O(M·N²), négligeable aux tailles de population utilisées ici.
    """
    n = len(objectives)
    dominated_by = [set() for _ in range(n)]
    domination_count = [0] * n
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _constrained_dominates(objectives[p], violations[p], objectives[q], violations[q]):
                dominated_by[p].add(q)
            elif _constrained_dominates(objectives[q], violations[q], objectives[p], violations[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def _crowding_distance(front_objectives: list) -> list:
    """
    Distance de crowding (Deb et al., 2002, Algorithme 2) au sein d'un front :
    mesure l'espacement d'un individu par rapport à ses voisins sur chaque
    objectif, pour favoriser la diversité le long du front lors de la
    sélection. Les points extrêmes de chaque objectif reçoivent une distance
    infinie (toujours préservés par l'opérateur de comparaison encombré).
    """
    m = len(front_objectives)
    if m == 0:
        return []
    if m <= 2:
        return [float("inf")] * m
    n_obj = len(front_objectives[0])
    distance = [0.0] * m
    for k in range(n_obj):
        order = sorted(range(m), key=lambda i: front_objectives[i][k])
        vmin, vmax = front_objectives[order[0]][k], front_objectives[order[-1]][k]
        distance[order[0]] = distance[order[-1]] = float("inf")
        span = (vmax - vmin) if vmax > vmin else 1.0
        for pos in range(1, m - 1):
            prev_v = front_objectives[order[pos - 1]][k]
            next_v = front_objectives[order[pos + 1]][k]
            distance[order[pos]] += (next_v - prev_v) / span
    return distance


def _compute_r2_symbolic(model: torch.nn.Module, x_pool: torch.Tensor,
                         theta: float = 0.01, r2_threshold: float = 0.99,
                         max_edges: int = 40, domain: float = 2.0) -> float:
    """
    R2_symbolic (section 3, R² symbolique) : élagage (arête survivante si
    L1 > theta) puis régression symbolique par gradient — mêmes familles de
    fonctions, même objectif c*f(a*x+b)+d, mêmes hyperparamètres (150 pas
    Adam, lr=0.05) que symbolic.fit_symbolic_best/SYMBOLIC_LIBRARY, mais
    ENTIÈREMENT VECTORISÉE EN TENSEURS PYTORCH plutôt qu'ajustée arête par
    arête : au lieu de E arêtes × 13 familles × 150 pas Adam SÉQUENTIELS
    (jusqu'à 78 000 micro-optimisations Python par évaluation fitness — le
    principal goulot d'étranglement mesuré de compute_fitness), les E arêtes
    survivantes sont ajustées SIMULTANÉMENT pour chaque famille : les
    paramètres a,b,c,d ont la forme (E,1) et une seule passe Adam à 150 pas
    traite les E arêtes en parallèle (13 passes au total, contre E×13).

    Équivalence garantie : la perte totale sommée sur les arêtes
    (`per_edge_loss.sum()`) a, pour le paramètre a[e]/b[e]/c[e]/d[e] d'une
    arête e, EXACTEMENT le même gradient que si cette arête était ajustée
    seule avec sa propre perte moyenne sur 100 points — aucune arête n'est
    couplée à une autre dans le calcul (produits/sommes strictement
    élément-par-élément), donc la trajectoire d'optimisation de chaque
    arête est identique au cas séquentiel. Une arête qui diverge (NaN/Inf)
    est masquée (perte remplacée par une constante) pour ne jamais
    contaminer par NaN le gradient des autres arêtes via la somme partagée,
    et reçoit R²=-∞ (même convention que fit_symbolic_candidate).

    max_edges plafonne le nombre d'arêtes survivantes effectivement
    évaluées symboliquement, pour borner le temps d'une évaluation fitness
    pendant la recherche — même logique de tractabilité que
    audit.prune_and_extract(top_k=...).

    Retourne 0.0 si aucune arête ne survit à l'élagage.
    """
    gate_modules = []
    for attr in ("forget_gate", "input_gate", "candidate_gate", "output_gate",
                "reset_gate", "update_gate"):
        m = getattr(model.cell, attr, None)
        if m is not None:
            gate_modules.append(m)

    survivors = []   # (layer, i, j, l1_importance)
    with torch.no_grad():
        for layer in gate_modules:
            edges = layer.edge_activations(x_pool)          # (batch, in, out)
            l1_mat = edges.abs().mean(dim=0)
            mask = l1_mat > theta
            for i, j in mask.nonzero(as_tuple=False).tolist():
                survivors.append((layer, i, j, float(l1_mat[i, j])))

    if not survivors:
        return 0.0

    survivors.sort(key=lambda s: -s[3])
    survivors = survivors[:max_edges]
    n_edges = len(survivors)

    # ── Extraction des courbes marginales (E passages forward bon marché :
    #    100 points chacun — négligeable face à l'ajustement ci-dessous) ──────
    x_grid = torch.linspace(-domain, domain, 100)
    curves = torch.empty(n_edges, 100)
    with torch.no_grad():
        for e, (layer, i, j, _) in enumerate(survivors):
            x = torch.zeros(100, layer.in_features)
            x[:, i] = x_grid
            curves[e] = layer.edge_activations(x)[:, i, j]

    # ── Ajustement symbolique vectorisé : toutes les arêtes en parallèle,
    #    une passe Adam (150 pas) par famille de fonctions ─────────────────
    best_r2 = torch.full((n_edges,), -float("inf"))
    x_row = x_grid.unsqueeze(0)   # (1, 100) — broadcast vers (E, 100)

    from tqdm.auto import tqdm
    for fn in tqdm(SYMBOLIC_LIBRARY.values(), desc="    R2_symbolic - familles",
                   unit="famille", leave=False):
        a = torch.nn.Parameter(torch.ones(n_edges, 1))
        b = torch.nn.Parameter(torch.zeros(n_edges, 1))
        c = torch.nn.Parameter(torch.ones(n_edges, 1))
        d = torch.nn.Parameter(torch.zeros(n_edges, 1))
        optimizer = DMLAdam([a, b, c, d], lr=0.05)

        diverged = torch.zeros(n_edges, dtype=torch.bool)
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True)
            pred = c * fn(a * x_row + b) + d                # (E, 100)
            bad = torch.isnan(pred).any(dim=1) | torch.isinf(pred).any(dim=1)
            diverged = diverged | bad
            per_edge_loss = (pred - curves).square().mean(dim=1)          # (E,)
            # Masquage AVANT réduction : empêche une arête NaN de contaminer
            # par la somme partagée le gradient des arêtes saines.
            per_edge_loss = torch.where(diverged, torch.zeros_like(per_edge_loss), per_edge_loss)
            per_edge_loss.sum().backward()
            optimizer.step()

        with torch.no_grad():
            pred = c * fn(a * x_row + b) + d
            ss_res = (curves - pred).square().sum(dim=1)
            ss_tot = (curves - curves.mean(dim=1, keepdim=True)).square().sum(dim=1)
            r2 = 1.0 - ss_res * (ss_tot + 1e-12).reciprocal()
            r2 = torch.where(diverged | torch.isnan(r2) | torch.isinf(r2),
                             torch.full_like(r2, -float("inf")), r2)
        best_r2 = torch.maximum(best_r2, r2)

    n_ok = int((best_r2 >= r2_threshold).sum().item())
    return n_ok / n_edges


def compute_fitness(
    config: dict,
    df_train_sample: pd.DataFrame,
    df_val_sample: pd.DataFrame,
    device,
    n_windows_eval: int = 50_000,
    n_epochs_eval: int = 10,
    seed: int = 42,
    theta_prune: float = 0.01,
    r2_threshold: float = 0.99,
    enable_latency_penalty: bool = True,
    window_cache: Optional[dict] = None,
) -> dict:
    """
    Fitness scientifique de l'Étape 2 (section 3) :
        Fitness = 0.40*max(0,MCC) + 0.25*PR_AUC - 0.15*Brier
                 + 0.10*R2_symbolic - 0.10*penalty_latency

    Args:
        config          : individu validé (cf. sample_individual/validate_individual)
        df_train_sample : DataFrame transactionnel brut (colonnes MoMTSim) — le pool
                           depuis lequel n_windows_eval fenêtres train sont tirées.
        df_val_sample   : idem pour la validation.
        device          : device d'ENTRAÎNEMENT (ex. "mps" sur Apple M4). La latence
                           est TOUJOURS mesurée sur CPU, indépendamment de ce paramètre
                           (cf. _measure_latency_ms) — la contrainte USSD de production
                           est une contrainte CPU, pas une propriété du device de dev.
        n_windows_eval  : nombre de fenêtres train/val sous-échantillonnées
                           (configurable, jamais codé en dur — section 4).
        n_epochs_eval   : nombre d'époques d'entraînement rapide (configurable).
        enable_latency_penalty : si False, latence et penalty_lat sont TOUJOURS
                           mesurées/calculées et conservées dans fitness_components
                           (traçabilité), mais penalty_lat n'est PAS soustraite de
                           fitness_total. Décision explicite de l'utilisateur pour ses
                           propres runs (recherche sur GPU M4, hors contrainte USSD
                           CPU de production) — le protocole du mémoire (seuil 100 ms,
                           pénalité quadratique, mesure CPU/batch=256) reste intact et
                           reproductible avec enable_latency_penalty=True (défaut).
        window_cache    : dict optionnel {(regime, W): (X_tr, y_tr, X_vl, y_vl, feature_cols)}
                           tenu par l'appelant (ExtendedHeuristicSearch._window_cache),
                           réutilisé entre individus partageant (Input_Regime, W).
                           Optimisation pure — AUCUN effet sur le résultat numérique :
                           `seed` est fixe sur toute une recherche, donc le sous-
                           échantillon de n_windows_eval fenêtres tiré pour un (regime, W)
                           donné est BIT-IDENTIQUE à chaque appel (même df source, même
                           transform, même graine) ; le cache mémorise ce résultat déjà-
                           subsamplé (~50 000 fenêtres, quelques dizaines de Mo) plutôt que
                           de refaire le passage complet sur les millions de transactions
                           brutes (construction des fenêtres compte par compte) à chaque
                           individu. None (défaut) = comportement inchangé, aucun cache.

    Returns:
        dict {'fitness_total', 'fitness_components', 'model_state', 'n_params',
              'hidden_size_gru_adjusted', 'error'}
        En cas d'échec, 'error' contient le message et 'fitness_total' vaut
        invalid_fitness (cf. ExtendedHeuristicSearch.invalid_fitness) — cette
        fonction ne lève jamais d'exception non contrôlée (section 14).
    """
    from sklearn.metrics import matthews_corrcoef, average_precision_score, brier_score_loss

    try:
        from niveau1_harness import PROCESSED_FEATURE_COLS  # noqa: F401  (documentation)
        regime = _REGIME_TO_NIVEAU1[config["Input_Regime"]]
        W = int(config["W"])
        cache_key = (regime, W)

        if window_cache is not None and cache_key in window_cache:
            X_tr, y_tr, X_vl, y_vl, feature_cols = window_cache[cache_key]
        else:
            df_tr, feature_cols = build_regime_frame(df_train_sample, regime)
            df_vl, _            = build_regime_frame(df_val_sample, regime)
            stats = standardize(df_tr, feature_cols)
            standardize(df_vl, feature_cols, stats=stats)

            X_tr, y_tr = build_windows(df_tr, W, feature_cols)
            X_vl, y_vl = build_windows(df_vl, W, feature_cols)
            if len(X_tr) == 0 or len(X_vl) == 0:
                raise ValueError("Aucune fenêtre construite (comptes < W transactions)")

            rng_np = np.random.default_rng(seed)
            if len(X_tr) > n_windows_eval:
                idx = np.sort(rng_np.choice(len(X_tr), n_windows_eval, replace=False))
                X_tr, y_tr = X_tr[idx], y_tr[idx]
            if len(X_vl) > n_windows_eval:
                idx = np.sort(rng_np.choice(len(X_vl), n_windows_eval, replace=False))
                X_vl, y_vl = X_vl[idx], y_vl[idx]

            # Mémorisé APRÈS sous-échantillonnage : ~50 000 fenêtres (~qq dizaines
            # de Mo), jamais le pool complet pré-subsampling (potentiellement
            # plusieurs millions de fenêtres sur featuresLog.parquet en entier).
            if window_cache is not None:
                window_cache[cache_key] = (X_tr, y_tr, X_vl, y_vl, feature_cols)

        input_size = len(feature_cols)
        torch.manual_seed(seed)
        model = build_model_from_individual(config, input_size).to(device)

        optimizer = DMLAdam(model.parameters(), lr=float(config["lr"]))
        batch_size = int(config["batch_size"])

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        n = X_tr_t.shape[0]

        model.train()
        from tqdm.auto import tqdm
        # Seule étape sans retour visuel jusqu'ici : jusqu'à n_epochs_eval ×
        # (n_windows_eval/batch_size) pas — ex. 10 × (50000/32) ≈ 15 600 pas
        # pour un petit batch_size, potentiellement la portion la plus longue
        # d'une évaluation individuelle. leave=False : disparaît une fois
        # l'individu terminé, la barre "individus" au-dessus reste la trace
        # persistante (cf. _evaluer_generation).
        epoch_pbar = tqdm(range(n_epochs_eval), desc="      entraînement", unit="ép", leave=False)
        for _ in epoch_pbar:
            perm = torch.randperm(n, device=device)
            # Accumulation en TENSEUR (pas de .item() par batch) : un .item()
            # par batch forcerait une synchronisation GPU à chaque pas (casse
            # le pipeline asynchrone MPS/CUDA — exactement le genre de coût
            # qu'on vient d'éliminer ailleurs). Une seule synchronisation par
            # époque (après la boucle), coût négligeable.
            # Non in-place (epoch_loss_sum = ... + ...) : mkan_total_loss peut
            # renvoyer un tenseur de forme (1,) (reg_l1/reg_entropy sont des
            # torch.zeros(1,...) dans MKANScorer.forward), incompatible avec
            # un += in-place sur un accumulateur scalaire strict.
            epoch_loss_sum = torch.zeros((), device=device)
            epoch_n_batches = 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad(set_to_none=True)
                loss_total, *_ = mkan_total_loss(
                    model, xb, yb, lam=float(config["lam"]),
                    mu1=float(config["mu1"]), mu2=float(config["mu2"]))
                if torch.isnan(loss_total) or torch.isinf(loss_total):
                    raise RuntimeError("loss NaN/Inf pendant l'entraînement")
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss_sum = epoch_loss_sum + loss_total.detach().reshape(())
                epoch_n_batches += 1
            epoch_pbar.set_postfix({"loss": f"{(epoch_loss_sum / max(1, epoch_n_batches)).item():.4f}"})

        model.eval()
        X_vl_t = torch.from_numpy(X_vl).to(device)
        with torch.no_grad():
            scores_parts = []
            for s in range(0, X_vl_t.shape[0], batch_size):
                scores_parts.append(model(X_vl_t[s:s + batch_size]))
            scores = torch.cat(scores_parts).cpu().numpy()
        scores = _sanitize_scores(scores, "compute_fitness (validation)")

        if len(np.unique(y_vl)) < 2:
            mcc_raw, pr_auc, brier = 0.0, 0.0, 1.0
        else:
            mcc_raw = _safe_float(matthews_corrcoef(y_vl, (scores >= 0.5).astype(int)))
            pr_auc  = _safe_float(average_precision_score(y_vl, scores))
            brier   = _safe_float(brier_score_loss(y_vl, scores), default=1.0)
        mcc_clipped = max(0.0, mcc_raw)

        # ── Latence (toujours CPU) + élagage/symbolique (sur un sous-pool) ──
        latency_ms = _measure_latency_ms(model, W, input_size)
        penalty_lat = _penalty_latency(latency_ms)

        with torch.no_grad():
            n_audit = min(256, X_vl_t.shape[0])
            audit_X = X_vl_t[:n_audit].to(device)
            h_t = torch.zeros(n_audit, model.hidden_size, device=device)
            if config["Cell_Type"] == "TKANCell":
                c_t = torch.zeros(n_audit, model.hidden_size, device=device)
                for t in range(W - 1):
                    h_t, c_t = model.cell(audit_X[:, t, :], h_t, c_t)
            else:
                for t in range(W - 1):
                    h_t = model.cell(audit_X[:, t, :], h_t)
            x_pool = torch.cat([h_t, audit_X[:, W - 1, :]], dim=1).cpu()

        r2_symbolic = _compute_r2_symbolic(model.to("cpu"), x_pool,
                                            theta=theta_prune, r2_threshold=r2_threshold)
        model.to(device)

        # penalty_lat est TOUJOURS mesurée/calculée (traçabilité, section 20) ; seule
        # sa contribution à fitness_total est conditionnelle à enable_latency_penalty.
        penalty_lat_applied = penalty_lat if enable_latency_penalty else 0.0
        fitness_total = (
            0.40 * mcc_clipped + 0.25 * pr_auc - 0.15 * brier
            + 0.10 * r2_symbolic - 0.10 * penalty_lat_applied
        )

        components = {
            "MCC_raw": mcc_raw, "MCC_clipped": mcc_clipped,
            "PR_AUC": pr_auc, "Brier": brier,
            "R2_symbolic": r2_symbolic,
            "latency_ms": latency_ms, "penalty_lat": penalty_lat,
            "penalty_lat_applied": enable_latency_penalty,
            "fitness_total": _safe_float(fitness_total, default=-10.0),
        }
        n_params = sum(p.numel() for p in model.parameters())
        h_gru = (hidden_size_gru_adjusted(int(config["hidden_size"]))
                 if config["Cell_Type"] == "GRUKANCell" else None)

        return {
            "fitness_total": components["fitness_total"],
            "fitness_components": components,
            "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "n_params": n_params,
            "hidden_size_gru_adjusted": h_gru,
            "input_size": input_size,
            "error": None,
        }

    except Exception as exc:   # noqa: BLE001 — section 14 : jamais planter la recherche
        return {
            "fitness_total": None,
            "fitness_components": None,
            "model_state": None,
            "n_params": None,
            "hidden_size_gru_adjusted": None,
            "input_size": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        }
    finally:
        # Libération explicite — critique sur MPS/CUDA (bug rapporté : "MPS backend
        # out of memory" après quelques générations). Chaque appel construit un
        # nouveau modèle + optimiseur + tenseurs d'entraînement sur `device` ; les
        # graphes autograd forment des cycles de références que le comptage de
        # références de Python ne collecte pas seul (il faut gc.collect()), et
        # l'allocateur "caching" de MPS/CUDA ne restitue pas la mémoire libérée au
        # driver tant que empty_cache() n'est pas appelé explicitement — sans ces
        # deux étapes, la mémoire GPU croît de façon monotone au fil des individus/
        # générations jusqu'à l'OOM. Le state_dict retourné est déjà cloné sur CPU
        # (ligne "model_state": ...) : cette purge n'affecte jamais la valeur renvoyée.
        _release_device_memory(device)


# ══════════════════════════════════════════════════════════════════════════
# 4. ExtendedHeuristicSearch — algorithme génétique sur l'espace conditionnel
# ══════════════════════════════════════════════════════════════════════════

class ExtendedHeuristicSearch(RechercheHeuristique):
    """
    Recherche heuristique étendue (Étape 2) sur l'espace hiérarchique
    conditionnel (base KAN par porte, cellule, régime d'entrée) avec la
    fonction de fitness multi-objectifs normalisée (section 3).

    Sous-classe RechercheHeuristique (heuristic_search.py) pour préserver
    sa compatibilité d'interface (attributs `_population`/`_scores`/
    `_journal`, propriétés `.history`/`.best`) ; les méthodes internes de
    génération/croisement/mutation/diversité/EDA sont redéfinies car
    l'espace de recherche n'est plus un simple dict {param: [valeurs]} —
    __init__ appelle le parent avec un espace factice à but de validation
    uniquement (jamais utilisé pour échantillonner un individu).
    """

    def __init__(
        self,
        population_size: int   = 12,
        elite_size:      int   = 2,
        n_generations:   int   = 15,
        mutation_rate:   float = 0.35,
        crossover_rate:  float = 0.9,
        tournament_size: int   = 3,
        refroidissement: float = 0.97,
        min_mutation_rate: float = 0.05,
        min_diversite:   float = 0.40,
        patience:        int   = 6,
        n_windows_eval:  int   = 50_000,
        n_epochs_eval:   int   = 10,
        seed:            int   = 42,
        scratch_dir:     Optional[str] = None,
        n_workers:       int   = 1,
        invalid_fitness: float = -10.0,
        maximize:        bool  = True,
        enforce_latency_constraint: bool = True,
        enable_latency_penalty:     bool = True,
    ) -> None:
        # espace factice : satisfait la validation du parent (non vide),
        # jamais consulté par les méthodes redéfinies ci-dessous.
        super().__init__(
            espace={"_placeholder": [0]}, fitness_fn=None,
            n_generations=n_generations, taille_population=population_size,
            elite_size=elite_size, tournament_size=tournament_size,
            mutation_rate=mutation_rate, refroidissement=refroidissement,
            min_mutation_rate=min_mutation_rate, min_diversite=min_diversite,
            patience=patience, maximize=maximize, n_workers=n_workers,
        )
        self.crossover_rate  = crossover_rate
        self.n_windows_eval  = n_windows_eval
        self.n_epochs_eval   = n_epochs_eval
        self.seed            = seed
        self.invalid_fitness = invalid_fitness
        # Contrainte de latence (100 ms, protocole du mémoire) appliquée via le
        # principe de dominance contrainte (Deb et al., 2002) : un individu qui
        # la viole est toujours dominé par un individu réalisable. Si False, la
        # contrainte est ignorée (tous les individus traités comme réalisables) —
        # décision explicite pour un usage hors contrainte USSD de production ;
        # la latence reste toujours mesurée/journalisée dans tous les cas.
        self.enforce_latency_constraint = enforce_latency_constraint
        # Propagé explicitement à compute_fitness (cf. _evaluer_un) — remplace
        # le monkeypatch de fonction module-level utilisé précédemment par
        # run_search.py (_ehs.compute_fitness = ...), fragile et incohérent
        # avec le vrai paramètre d'instance déjà utilisé pour
        # enforce_latency_constraint et pour HierarchicalSensitivityAnalyzer.
        self.enable_latency_penalty = enable_latency_penalty

        default_scratch = "/workspace/scratch"
        if scratch_dir is None:
            # /workspace/scratch est une convention de conteneur cloud absente
            # de cet environnement Windows : repli documenté sur un dossier
            # du dépôt si le chemin canonique n'est pas accessible en écriture.
            try:
                os.makedirs(default_scratch, exist_ok=True)
                scratch_dir = default_scratch
            except OSError:
                scratch_dir = os.path.join(_MKAN_DIR, "checkpoints", "extended_search")
        os.makedirs(scratch_dir, exist_ok=True)
        self.scratch_dir = scratch_dir

        self._rng = random.Random(seed)
        self._log_lock = threading.Lock()
        self._log_path = os.path.join(self.scratch_dir, "extended_search_log.jsonl")
        self._best_candidate_path = os.path.join(self.scratch_dir, "best_candidate.pt")
        self._best_config_path    = os.path.join(self.scratch_dir, "heuristic_best_config.json")
        self._checkpoint_path     = os.path.join(self.scratch_dir, "search_checkpoint.json")

        self._n_evaluations = 0
        self._best_individual: Optional[dict] = None
        self._best_result:     Optional[dict] = None
        self._input_size_used: Optional[int]  = None
        self._components: list = []   # fitness_components par individu de la génération courante
        self._window_cache: dict = {}   # (regime, W) -> fenêtres sous-échantillonnées déjà construites
        self._last_checkpointed_generation: Optional[int] = None
        self._validation_full: dict = {}   # rempli par evaluate_best_on_full_validation()

        # ── État NSGA-II (tri par dominance contrainte, Deb et al. 2002) ──────
        self._rank: list  = []    # rang de front par individu de la génération courante
        self._crowd: list = []    # distance de crowding par individu de la génération courante
        self._pareto_archive: list = []   # front non-dominé cumulé {individual, fitness_components}
        self._representative_changed_this_gen: bool = False
        self._pareto_front_path = os.path.join(self.scratch_dir, "pareto_front.json")

    # ── Génération d'individus (redéfinition) ───────────────────────────────

    def _individu_aleatoire(self) -> dict:
        return sample_individual(self._rng)

    # ── Diversité (redéfinition : égalité par sérialisation JSON canonique) ─

    @staticmethod
    def _individual_key(ind: dict) -> str:
        return json.dumps(ind, sort_keys=True, default=str)

    def _calculer_diversite(self) -> float:
        uniques = len({self._individual_key(ind) for ind in self._population})
        return uniques / len(self._population)

    def _injecter_diversite(self, n_inject: int) -> None:
        idx_tries = self._indices_tries()
        pires = idx_tries[-n_inject:]
        for i in pires:
            self._population[i] = self._individu_aleatoire()
            self._scores[i] = None
            self._components[i] = None

    # ── Sélection NSGA-II (redéfinition : dominance contrainte + crowding) ──
    #
    # RechercheHeuristique._elites() est HÉRITÉE telle quelle (non redéfinie
    # ici) : elle appelle self._indices_tries(), dont l'override ci-dessous
    # suffit par polymorphisme à la rendre NSGA-II-consciente sans dupliquer
    # son code — idem pour son usage dans _injecter_diversite() (ci-dessus)
    # et _eda_frequences() (plus bas).

    def _compute_rank_and_crowding(self) -> None:
        """
        Calcule rang de front (tri non-dominé) et distance de crowding pour
        CHAQUE individu de la population courante, à partir de
        self._components (fitness_components par individu). Appelée une fois
        par génération, juste après évaluation, avant toute sélection.
        Individus invalides (components=None, échec d'évaluation) : forcés au
        pire rang possible + crowding nul — jamais sélectionnés en élite/tournoi.
        """
        n = len(self._population)
        objs, cvs, valid_idx = [], [], []
        for i in range(n):
            comp = self._components[i]
            if comp is not None:
                objs.append(_objective_vector(comp))
                cvs.append(_constraint_violation(comp) if self.enforce_latency_constraint else 0.0)
                valid_idx.append(i)

        rank  = [len(valid_idx) + 1] * n   # pire rang par défaut (invalides)
        crowd = [0.0] * n

        if valid_idx:
            fronts = _fast_non_dominated_sort(objs, cvs)
            for front_rank, front in enumerate(fronts):
                cd = _crowding_distance([objs[k] for k in front])
                for local_k, k in enumerate(front):
                    global_i = valid_idx[k]
                    rank[global_i] = front_rank
                    crowd[global_i] = cd[local_k]

        self._rank, self._crowd = rank, crowd

    def _indices_tries(self) -> list:
        """
        Redéfinition NSGA-II de RechercheHeuristique._indices_tries() :
        opérateur de comparaison encombré (Deb et al., 2002, §III.A) — rang de
        front croissant, puis distance de crowding décroissante à rang égal —
        au lieu du tri par score scalaire du parent. _elites(), héritée,
        utilise transparemment ce nouvel ordre.
        """
        return sorted(range(len(self._population)),
                     key=lambda i: (self._rank[i], -self._crowd[i]))

    def _tournoi(self, n: int) -> list:
        """Sélection par tournoi NSGA-II : l'opérateur de comparaison encombré
        (rang, puis crowding) départage le groupe, remplace la comparaison par
        score scalaire du parent.

        Utilise self._rng (pas le module `random` global, contrairement à
        RechercheHeuristique._tournoi hérité) : reproductibilité complète sous
        seed fixe — tout le reste du tirage aléatoire propre à
        ExtendedHeuristicSearch (génération d'individus, croisement, mutation,
        EDA) passe déjà par self._rng ; seule la sélection par tournoi héritée
        du parent utilisait encore le générateur global avant cette redéfinition.
        """
        sel = []
        indices = list(range(len(self._population)))
        for _ in range(n):
            groupe = self._rng.sample(indices, self.tournament_size)
            gagnant = min(groupe, key=lambda i: (self._rank[i], -self._crowd[i]))
            sel.append(dict(self._population[gagnant]))
        return sel

    def _selectionner_parents(self) -> list:
        """
        NSGA-II : élites + tournoi uniquement (Deb et al., 2002) — la sélection
        par roulette du parent (proportionnelle à un score scalaire) n'a pas de
        sens pour une sélection multi-objectifs par dominance et est retirée
        (contrairement à RechercheHeuristique._selectionner_parents, qui
        combinait élites + roulette + tournoi).
        """
        reste = self.taille_pop - self.elite_size
        e_ind, _ = self._elites()
        return e_ind + self._tournoi(reste)

    # ── Croisement (section 6) ───────────────────────────────────────────────

    def _croiser(self, parent1: dict, parent2: dict) -> tuple:
        return self._croiser_un(parent1, parent2), self._croiser_un(parent2, parent1)

    def _croiser_un(self, p1: dict, p2: dict) -> dict:
        """
        Un enfant : theta_opt croisé point par point (Bernoulli 0.5 par clé).
        Par porte : même base chez les deux parents -> hyperparamètres
        croisés (Bernoulli 0.5 par hyperparamètre) ; bases différentes ->
        base du parent 1 avec probabilité 0.5 (sinon parent 2), puis
        ré-échantillonnage complet des hyperparamètres de la base retenue.
        Cell_Type hérité de p1 avec probabilité 0.5 (fixe le jeu de portes).
        """
        rng = self._rng
        child_cell = p1["Cell_Type"] if rng.random() < 0.5 else p2["Cell_Type"]
        child_regime = p1["Input_Regime"] if rng.random() < 0.5 else p2["Input_Regime"]

        child = {
            "Cell_Type": child_cell,
            "Input_Regime": child_regime,
        }
        for key in ("hidden_size", "lr", "lam", "mu1", "mu2", "batch_size", "W"):
            child[key] = p1[key] if rng.random() < 0.5 else p2[key]

        gate_names = GATE_NAMES_TKAN if child_cell == "TKANCell" else GATE_NAMES_GRU
        gates = {}
        for gate in gate_names:
            g1 = p1["gates"].get(gate)
            g2 = p2["gates"].get(gate)
            if g1 is not None and g2 is not None and g1["Base"] == g2["Base"]:
                base = g1["Base"]
                gate_cfg = {"Base": base}
                for name in BASE_PARAM_SPACE[base]["params"]:
                    gate_cfg[name] = g1[name] if rng.random() < 0.5 else g2[name]
                gates[gate] = gate_cfg
            else:
                source = g1 if (g1 is not None and rng.random() < 0.5) else g2
                if source is None:
                    source = g1 if g1 is not None else g2
                base = source["Base"] if source["Base"] in GATE_CANDIDATES[gate] \
                    else rng.choice(GATE_CANDIDATES[gate])
                gates[gate] = {"Base": base, **_sample_base_hyperparams(base, rng)}
        child["gates"] = gates
        return child

    # ── Mutation adaptative (section 6) ─────────────────────────────────────

    def _muter(self, individual: dict) -> dict:
        rng = self._rng
        mutant = copy.deepcopy(individual)
        sigma = self._mutation_rate_courant   # réutilisé comme sigma_t (cf. fit())

        for key in ("hidden_size", "batch_size", "W"):
            if rng.random() < self.mutation_rate:
                mutant[key] = rng.choice(THETA_OPT_SPACE[key])
        for key in CONTINUOUS_THETA_OPT:
            if rng.random() < self.mutation_rate:
                lo, hi = min(THETA_OPT_SPACE[key]), max(THETA_OPT_SPACE[key])
                mutant[key] = min(hi, max(lo, mutant[key] + rng.gauss(0.0, sigma * (hi - lo))))
        if rng.random() < self.mutation_rate:
            mutant["Input_Regime"] = rng.choice(INPUT_REGIMES)
        if rng.random() < self.mutation_rate:
            new_cell = rng.choice(CELL_TYPES)
            if new_cell != mutant["Cell_Type"]:
                mutant["Cell_Type"] = new_cell
                gate_names = GATE_NAMES_TKAN if new_cell == "TKANCell" else GATE_NAMES_GRU
                mutant["gates"] = {
                    g: {"Base": (b := rng.choice(GATE_CANDIDATES[g])),
                        **_sample_base_hyperparams(b, rng)}
                    for g in gate_names
                }

        for gate, gate_cfg in mutant["gates"].items():
            if rng.random() < self.mutation_rate:
                gate_cfg["Base"] = rng.choice(GATE_CANDIDATES[gate])
                new_params = _sample_base_hyperparams(gate_cfg["Base"], rng)
                mutant["gates"][gate] = {"Base": gate_cfg["Base"], **new_params}
                continue
            base = gate_cfg["Base"]
            spec = BASE_PARAM_SPACE[base]
            for name, domain in spec["params"].items():
                if rng.random() < self.mutation_rate:
                    if name in spec["continuous"]:
                        lo, hi = domain
                        gate_cfg[name] = min(hi, max(lo,
                            gate_cfg[name] + rng.gauss(0.0, sigma * (hi - lo))))
                    else:
                        gate_cfg[name] = rng.choice(domain)
        return mutant

    # ── EDA (section 6) ───────────────────────────────────────────────────────

    def _eda_frequences(self, n_best: int) -> dict:
        idx = self._indices_tries()[:n_best]
        best = [self._population[i] for i in idx]

        freq = {"Cell_Type": self._laplace_freq([b["Cell_Type"] for b in best], CELL_TYPES),
                "Input_Regime": self._laplace_freq([b["Input_Regime"] for b in best], INPUT_REGIMES)}
        for key in ("hidden_size", "batch_size", "W"):
            freq[key] = self._laplace_freq([b[key] for b in best], THETA_OPT_SPACE[key])
        for key in CONTINUOUS_THETA_OPT:
            vals = [b[key] for b in best]
            freq[key] = self._beta_params(vals, min(THETA_OPT_SPACE[key]), max(THETA_OPT_SPACE[key]))

        gate_freq = {}
        for gate in GATE_NAMES_TKAN:
            observed = [b["gates"][gate] for b in best if gate in b["gates"]]
            base_freq = self._laplace_freq([o["Base"] for o in observed], GATE_CANDIDATES[gate])
            per_base = {}
            for base in GATE_CANDIDATES[gate]:
                obs_base = [o for o in observed if o["Base"] == base]
                spec = BASE_PARAM_SPACE[base]
                param_dist = {}
                for name, domain in spec["params"].items():
                    vals = [o[name] for o in obs_base]
                    if name in spec["continuous"]:
                        lo, hi = domain
                        param_dist[name] = self._beta_params(vals, lo, hi)
                    else:
                        param_dist[name] = self._laplace_freq(vals, domain)
                per_base[base] = param_dist
            gate_freq[gate] = {"base": base_freq, "params": per_base}
        freq["gates"] = gate_freq
        return freq

    @staticmethod
    def _laplace_freq(observed: list, domain: list) -> dict:
        """Fréquences empiriques + lissage de Laplace(alpha=1)."""
        counts = {v: 1 for v in domain}
        for v in observed:
            counts[v] = counts.get(v, 1) + 1
        total = sum(counts.values())
        return {v: counts[v] / total for v in domain}

    @staticmethod
    def _beta_params(observed: list, lo: float, hi: float) -> dict:
        """
        Modèle Beta sur [lo, hi] par la méthode des moments sur les
        observations normalisées ; repli sur Beta(2,2) (uniforme centrée)
        si aucune observation ou variance nulle.
        """
        span = hi - lo if hi > lo else 1.0
        xs = [(v - lo) / span for v in observed] if observed else []
        if len(xs) < 2:
            return {"lo": lo, "hi": hi, "alpha": 2.0, "beta": 2.0}
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        m = min(max(m, 1e-3), 1 - 1e-3)
        var = max(var, 1e-6)
        common = m * (1 - m) / var - 1
        alpha = max(0.5, m * common)
        beta  = max(0.5, (1 - m) * common)
        return {"lo": lo, "hi": hi, "alpha": alpha, "beta": beta}

    def _generer_depuis_eda(self, freq: dict) -> dict:
        rng = self._rng

        def draw_categorical(dist: dict):
            items = list(dist.items())
            return rng.choices([k for k, _ in items], weights=[w for _, w in items], k=1)[0]

        def draw_beta(params: dict):
            x = rng.betavariate(params["alpha"], params["beta"])
            return params["lo"] + x * (params["hi"] - params["lo"])

        cell_type = draw_categorical(freq["Cell_Type"])
        individual = {
            "Cell_Type": cell_type,
            "Input_Regime": draw_categorical(freq["Input_Regime"]),
            "hidden_size": draw_categorical(freq["hidden_size"]),
            "batch_size": draw_categorical(freq["batch_size"]),
            "W": draw_categorical(freq["W"]),
        }
        for key in CONTINUOUS_THETA_OPT:
            individual[key] = draw_beta(freq[key])

        gate_names = GATE_NAMES_TKAN if cell_type == "TKANCell" else GATE_NAMES_GRU
        gates = {}
        for gate in gate_names:
            gfreq = freq["gates"][gate]
            base = draw_categorical(gfreq["base"])
            spec = BASE_PARAM_SPACE[base]
            gate_cfg = {"Base": base}
            for name, domain in spec["params"].items():
                dist = gfreq["params"][base][name]
                gate_cfg[name] = draw_beta(dist) if name in spec["continuous"] \
                    else draw_categorical(dist)
            gates[gate] = gate_cfg
        individual["gates"] = gates
        return individual

    # ── Boucle principale ────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame, device,
            input_size_hint: Optional[int] = None,
            resume_from_checkpoint: bool = False) -> dict:
        """
        Lance la recherche heuristique étendue. `df_train`/`df_val` sont les
        DataFrames transactionnels bruts (colonnes MoMTSim) ; compute_fitness
        en tire n_windows_eval fenêtres par évaluation (section 4).

        resume_from_checkpoint : si True et qu'un search_checkpoint.json existe
        dans scratch_dir, reprend exactement où la recherche précédente s'est
        arrêtée (population, scores, journal, meilleur individu, taux de
        mutation, compteur d'évaluations, état RNG) au lieu de repartir de
        zéro — permet d'interrompre puis relancer un run long (section 11).
        Si aucun checkpoint n'existe, démarre normalement (pas une erreur).
        """
        resumed = False
        if resume_from_checkpoint:
            resumed = self._restore_from_checkpoint()
            if resumed:
                print(f"  -> Reprise depuis le checkpoint : génération "
                     f"{self._last_checkpointed_generation + 1}/{self.n_generations}, "
                     f"{self._n_evaluations} évaluation(s) déjà réalisées, "
                     f"best={self._best_score:.4f}")
                # Le checkpoint capture l'état juste après évaluation de la génération
                # sauvegardée, AVANT la transition (diversité/refroidissement/EDA/
                # nouvelle population) vers la génération suivante : on rejoue cette
                # transition une fois pour obtenir une population start_gen cohérente.
                diversite_restauree = self._calculer_diversite()
                self._compute_rank_and_crowding()   # NSGA-II : requis avant _prepare_next_generation
                self._prepare_next_generation(diversite_restauree)

        if not resumed:
            self._initialiser()
            self._population = [self._individu_aleatoire() for _ in range(self.taille_pop)]
            self._components = [None] * self.taille_pop
            # _window_cache est clé par (regime, W), PAS par identité de df_train/
            # df_val : appeler fit() une seconde fois sur la MÊME instance avec des
            # données différentes (notebook, sweep programmatique) réutiliserait
            # silencieusement les fenêtres de l'appel précédent sans la moindre
            # erreur ni avertissement — résultats faux, aucun signal. Un nouveau
            # run() (resume=False) reconstruit forcément ses fenêtres à neuf ; un
            # resume garde le cache car il poursuit la MÊME recherche sur les
            # MÊMES données par construction.
            self._window_cache = {}

        start_gen = (self._last_checkpointed_generation + 1) if resumed else 0
        sans_amelioration = 0

        from tqdm.auto import tqdm
        _BAR_FMT = "{l_bar}{bar:28}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        gen_pbar = tqdm(range(start_gen, self.n_generations), desc="Générations",
                        unit="gén", leave=True, initial=start_gen, total=self.n_generations,
                        bar_format=_BAR_FMT, colour="cyan")

        for gen in gen_pbar:
            self._evaluer_generation(gen, df_train, df_val, device)
            self._compute_rank_and_crowding()   # NSGA-II : requis avant toute sélection

            ameliore = self._mettre_a_jour_best_extended()
            diversite = self._calculer_diversite()
            self._enregistrer_generation(gen, diversite, self._mutation_rate_courant)

            scores_valides = [s for s in self._scores if s is not None]
            moy = sum(scores_valides) / len(scores_valides) if scores_valides else float("nan")
            # Clés de postfix/print en ASCII (pas "μ") : certains terminaux (notamment
            # Windows avec la page de code cp1252 par défaut) plantent sur des
            # caractères hors de leur jeu de caractères lors de l'écriture tqdm —
            # constaté en pratique (UnicodeEncodeError sur "μ"/"→"/"↑"/"├" dans
            # gen_pbar.write). Les commentaires/docstrings du code gardent leurs
            # accents (jamais imprimés sur un terminal), seule la sortie console
            # réellement écrite au fil de l'exécution est restreinte à l'ASCII pour
            # rester robuste sur n'importe quelle plateforme/encodage de terminal.
            gen_pbar.set_postfix({"best": f"{self._best_score:.4f}", "moy": f"{moy:.4f}",
                                  "div": f"{diversite:.2f}", "mu": f"{self._mutation_rate_courant:.3f}"})
            # tqdm.write() (et non print()) : efface proprement la barre active, écrit
            # la ligne, puis la redessine en dessous — évite le rendu cassé/dupliqué
            # que print() provoque au milieu d'une barre tqdm ouverte (leave=True).
            gen_pbar.write(f"Gen {gen + 1:02d}/{self.n_generations} | best={self._best_score:.4f} | "
                          f"moy={moy:.4f} | div={diversite:.2f} | "
                          f"mu={self._mutation_rate_courant:.3f} | "
                          f"{'AMELIORATION' if ameliore else '='}")

            self._safe_disk_write("_save_checkpoint", self._save_checkpoint, gen)

            sans_amelioration = 0 if ameliore else sans_amelioration + 1
            if sans_amelioration >= self.patience:
                gen_pbar.write(f"  -> Arret anticipe : {self.patience} generations sans amelioration.")
                break

            self._prepare_next_generation(diversite)

        self._safe_disk_write("_write_best_config_json", self._write_best_config_json)
        self._safe_disk_write("export_pareto_front", self.export_pareto_front)
        return {"params": copy.deepcopy(self._best_individual), "score": self._best_score}

    def export_pareto_front(self, path: Optional[str] = None) -> str:
        """
        Écrit l'archive Pareto cumulée (front non-dominé, dominance contrainte
        latence) — le véritable résultat scientifique d'une recherche multi-
        objectifs NSGA-II, par opposition au représentant unique de
        heuristic_best_config.json (conservé pour compatibilité descendante
        avec sensitivity_analysis.py/results_export.py, qui attendent UNE
        configuration).
        """
        path = path or self._pareto_front_path
        payload = {
            "metadata": {
                "etape": 2, "methode": "NSGA-II (Deb et al., 2002), dominance contrainte",
                "latency_constraint_ms": LATENCY_CONSTRAINT_MS if self.enforce_latency_constraint else None,
                "n_front": len(self._pareto_archive),
                "objectives": ["MCC_clipped (max)", "PR_AUC (max)", "Brier (min)", "R2_symbolic (max)"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "front": [
                {"individual": c["individual"], "fitness_components": c["fitness_components"]}
                for c in self._pareto_archive
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return path

    def _prepare_next_generation(self, diversite: float) -> None:
        """
        Transition depuis la génération courante (déjà évaluée : self._population/
        _scores/_components complets) vers la suivante : injection de diversité,
        refroidissement de la mutation, modèle EDA, élites + enfants. Factorisé
        pour être rejouable après resume() (section 11 : reprise exacte, y compris
        la transition qui suit le dernier checkpoint sauvegardé).
        """
        if diversite < self.min_diversite:
            self._injecter_diversite(max(1, self.taille_pop // 4))

        self._mutation_rate_courant = max(self.min_mutation_rate,
                                          self._mutation_rate_courant * self.refroidissement)

        n_eda_best = min(max(2, self.elite_size * 2), len(self._population))
        freq = self._eda_frequences(n_eda_best)

        elite_idx = self._indices_tries()[:self.elite_size]
        e_ind    = [copy.deepcopy(self._population[i]) for i in elite_idx]
        e_scores = [self._scores[i] for i in elite_idx]
        e_comps  = [self._components[i] for i in elite_idx]
        parents = self._selectionner_parents()
        enfants = self._generer_enfants(parents, freq=freq)

        self._population = e_ind + enfants
        self._scores = e_scores + [None] * len(enfants)
        self._components = e_comps + [None] * len(enfants)

    def _evaluer_generation(self, gen: int, df_train, df_val, device) -> None:
        from tqdm.auto import tqdm

        self._representative_changed_this_gen = False
        to_eval = [(i, ind) for i, ind in enumerate(self._population) if self._scores[i] is None]
        pbar = tqdm(to_eval, desc=f"  |- gen {gen + 1}/{self.n_generations} individus",
                   unit="ind", leave=False, colour="green",
                   bar_format="{l_bar}{bar:28}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
        n_echecs = 0
        for i, ind in pbar:
            result = self._evaluer_un(gen, i, ind, df_train, df_val, device)
            self._scores[i] = result["fitness_total"] if result["fitness_total"] is not None \
                else self.invalid_fitness
            self._components[i] = result.get("fitness_components")
            if result.get("fitness_components") is None:
                n_echecs += 1
            pbar.set_postfix({"fitness": f"{self._scores[i]:.4f}", "base": ind.get("Cell_Type")})

        # Chaque échec individuel produit déjà une ligne [ERREUR], mais rien ne
        # signale explicitement un ÉCHEC SYSTÉMIQUE (tous les individus tombent
        # sur la même cause racine — mauvais fichier de données, schéma de
        # colonnes incorrect, device cassé...) : sans ce contrôle, la recherche
        # continue silencieusement génération après génération sans jamais
        # améliorer le front (patience finit par arrêter, mais sans indiquer
        # POURQUOI) plutôt que de signaler clairement le problème dès qu'il
        # survient.
        if to_eval and n_echecs == len(to_eval):
            tqdm.write(f"  [AVERTISSEMENT] gén={gen} : {n_echecs}/{len(to_eval)} individus ont "
                      f"ÉCHOUÉ (0 individu valide cette génération) — vérifier les fichiers de "
                      f"données/schéma/device : voir les lignes [ERREUR] ci-dessus pour la "
                      f"cause exacte (souvent la même pour tous).")

    def _evaluer_un(self, gen: int, individual_id: int, individual: dict,
                    df_train, df_val, device) -> dict:
        if not validate_individual(individual):
            result = {"fitness_total": self.invalid_fitness, "fitness_components": None,
                      "model_state": None, "error": "individu invalide (validate_individual)"}
        else:
            try:
                result = compute_fitness(individual, df_train, df_val, device,
                                         n_windows_eval=self.n_windows_eval,
                                         n_epochs_eval=self.n_epochs_eval, seed=self.seed,
                                         window_cache=self._window_cache,
                                         enable_latency_penalty=self.enable_latency_penalty)
            except Exception as exc:   # noqa: BLE001 — robustesse absolue (section 14)
                result = {"fitness_total": None, "fitness_components": None,
                          "model_state": None,
                          "error": f"exception non interceptée par compute_fitness : {exc}"}
            if result["fitness_total"] is None:
                # tqdm.write() (pas print()) : n'importe quelle barre active en cours
                # est proprement effacée/redessinée autour du message (évite le
                # rendu cassé signalé). Une seule ligne ici (type + message) ; la
                # trace complète reste dans extended_search_log.jsonl (_log_evaluation).
                _err = result.get("error") or ""
                from tqdm.auto import tqdm as _tqdm
                _tqdm.write(f"  [ERREUR] gén={gen} ind={individual_id} : {_err.splitlines()[0] if _err else '?'}")
                result = {**result, "fitness_total": self.invalid_fitness}

        with self._log_lock:
            self._n_evaluations += 1
            # Écritures disque protégées individuellement : un incident I/O (disque
            # plein, permission, verrou externe sur le fichier...) ne doit JAMAIS
            # interrompre toute la recherche en cours ni empêcher la mise à jour de
            # l'état EN MÉMOIRE (self._best_individual/_best_result restent la
            # source de vérité pour get_best_model()/export_top5() même si le
            # disque a momentanément refusé l'écriture) — seul un avertissement est
            # émis, la recherche continue.
            self._safe_disk_write("_log_evaluation", self._log_evaluation,
                                  gen, individual_id, individual, result)
            if (result["fitness_components"] is not None and
                    (self._best_result is None or
                     self._is_better_representative(result["fitness_components"],
                                                     self._best_result["fitness_components"]))):
                self._best_result = result
                self._best_individual = copy.deepcopy(individual)
                self._best_score = result["fitness_total"]
                self._representative_changed_this_gen = True
                self._safe_disk_write("_save_best_candidate", self._save_best_candidate)
                # heuristic_best_config.json à jour à CHAQUE amélioration (pas
                # seulement en fin de fit()) : un crash/kill en cours de route
                # laisse quand même un fichier canonique exploitable par
                # results_export.py, sans devoir attendre la fin de la recherche.
                self._safe_disk_write("_write_best_config_json", self._write_best_config_json)
        return result

    @staticmethod
    def _safe_disk_write(label: str, fn, *args) -> None:
        try:
            fn(*args)
        except OSError as exc:   # noqa: BLE001 — jamais interrompre la recherche pour un incident I/O
            from tqdm.auto import tqdm
            tqdm.write(f"  [AVERTISSEMENT] écriture disque échouée ({label}) : {exc} — "
                      "état en mémoire conservé, recherche poursuivie.")

    def _is_better_representative(self, cand_comp: dict, best_comp: dict) -> bool:
        """
        Règle de remplacement du représentant unique (best_candidate.pt /
        heuristic_best_config.json) — dominance contrainte D'ABORD (Deb et al.,
        2002), remplace la comparaison scalaire `>` de l'ancienne version.
        Si cand domine best (ou best viole la contrainte et pas cand) : cand
        remplace. Si best domine cand : jamais. Si mutuellement non-dominés
        (aucun ne domine l'autre) : départage par fitness_total (AHP,
        section~3.3.2) — SEUL point où le scalaire intervient encore, en
        tie-break documenté et non en décision primaire.
        """
        obj_c = _objective_vector(cand_comp)
        obj_b = _objective_vector(best_comp)
        if self.enforce_latency_constraint:
            cv_c = _constraint_violation(cand_comp)
            cv_b = _constraint_violation(best_comp)
        else:
            cv_c = cv_b = 0.0
        if _constrained_dominates(obj_c, cv_c, obj_b, cv_b):
            return True
        if _constrained_dominates(obj_b, cv_b, obj_c, cv_c):
            return False
        return cand_comp["fitness_total"] > best_comp["fitness_total"]

    def _log_evaluation(self, generation: int, individual_id: int, config: dict, result: dict) -> None:
        """Thread-safe (appelant détient déjà self._log_lock) : une ligne JSON par évaluation."""
        record = {
            "generation": generation, "individual_id": individual_id,
            "config": config,
            "fitness_components": result.get("fitness_components"),
            "fitness": result.get("fitness_total"),
            "error": result.get("error"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _mettre_a_jour_best_extended(self) -> bool:
        """
        Met à jour l'archive Pareto cumulée (section pareto_front.json) et
        retourne si le REPRÉSENTANT unique (best_candidate.pt) a changé
        pendant cette génération — la décision de remplacement elle-même a
        déjà eu lieu dans _evaluer_un (sous verrou, dominance contrainte,
        cf. _is_better_representative) ; ici on ne fait que lire le drapeau
        posé à ce moment-là et rafraîchir l'archive complète.
        """
        self._update_pareto_archive()
        return self._representative_changed_this_gen

    def _update_pareto_archive(self) -> None:
        """
        Recalcule l'archive Pareto cumulée = front non-dominé de (archive
        précédente ∪ individus valides de la génération courante), avec
        dominance contrainte (latence, section~3.4 du mémoire). Ne conserve
        que {individual, fitness_components} — jamais de state_dict (l'archive
        peut grossir sur toute la recherche ; seul le représentant unique
        conserve ses poids, cf. _is_better_representative/_save_best_candidate).
        """
        candidates = [
            {"individual": dict(self._population[i]), "fitness_components": dict(comp)}
            for i, comp in enumerate(self._components) if comp is not None
        ]
        pool = self._pareto_archive + candidates
        if not pool:
            return

        objs = [_objective_vector(c["fitness_components"]) for c in pool]
        cvs = [(_constraint_violation(c["fitness_components"]) if self.enforce_latency_constraint else 0.0)
               for c in pool]
        fronts = _fast_non_dominated_sort(objs, cvs)
        non_dominated = [pool[i] for i in fronts[0]] if fronts else []

        seen, archive = set(), []
        for c in non_dominated:
            key = self._individual_key(c["individual"])
            if key not in seen:
                seen.add(key)
                archive.append(c)
        self._pareto_archive = archive

    # ── Sauvegarde du meilleur modèle (section 8) ────────────────────────────

    def _save_best_candidate(self) -> None:
        if self._best_result is None or self._best_result["model_state"] is None:
            return
        payload = {
            "model_state": self._best_result["model_state"],
            "individual": self._best_individual,
            "input_size": self._best_result.get("input_size"),
            "n_params": self._best_result.get("n_params"),
            "hidden_size_gru_adjusted": self._best_result.get("hidden_size_gru_adjusted"),
            "fitness_components": self._best_result.get("fitness_components"),
        }
        tmp_path = self._best_candidate_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self._best_candidate_path)

    # ── Validation finale sur l'ensemble complet (section "Protocole
    #    d'évaluation sur 50 000 fenêtres") ──────────────────────────────────

    def evaluate_best_on_full_validation(
        self, df_train_for_stats: pd.DataFrame, df_val_full: pd.DataFrame, device="cpu",
    ) -> Optional[dict]:
        """
        Réévalue le meilleur modèle trouvé — poids déjà entraînés (best_candidate.pt),
        AUCUN ré-entraînement ici — sur la TOTALITÉ des fenêtres de df_val_full
        (pas de sous-échantillonnage, contrairement à compute_fitness). Conforme à
        step_2_search_extension_and_analysis.tex : « Une validation finale sur
        l'ensemble des 450 955 fenêtres de validation est conduite pour les 5
        meilleures configurations issues de la recherche. »

        Écart documenté : seule la MEILLEURE configuration est ré-évaluée ici (ses
        poids entraînés sont les seuls conservés par ExtendedHeuristicSearch —
        cf. _save_best_candidate) et non les 5 meilleures indépendamment, ce qui
        nécessiterait de conserver 5 state_dict complets pendant la recherche
        (non implémenté). export_top5() fournit les 5 configurations et leurs
        métriques de recherche (sous-échantillonnées) pour audit ; seule celle
        effectivement retenue (best_candidate.pt) est validée sur l'ensemble complet.

        Les statistiques de standardisation sont ajustées sur df_train_for_stats
        (même convention que compute_fitness : ajustement sur le train, jamais
        sur la validation — évite la fuite de données) puis appliquées à
        df_val_full.

        Returns:
            dict {'MCC', 'PR_AUC', 'Brier', 'AUC_ROC', 'n_windows'} ou None si
            aucun meilleur modèle n'est disponible. Stocké dans self._validation_full
            et repris tel quel dans heuristic_best_config.json (jamais de valeur
            fabriquée : si le calcul échoue, la clé reste absente/null).
        """
        if self._best_individual is None:
            self._warn_full_validation("aucun meilleur individu disponible (fit() non exécuté ?)")
            return None
        try:
            from sklearn.metrics import (matthews_corrcoef, average_precision_score,
                                         brier_score_loss, roc_auc_score)
            model = self.get_best_model()
            model.to(device).eval()

            regime = _REGIME_TO_NIVEAU1[self._best_individual["Input_Regime"]]
            df_tr, feature_cols = build_regime_frame(df_train_for_stats, regime)
            df_vl, _            = build_regime_frame(df_val_full, regime)
            stats = standardize(df_tr, feature_cols)
            standardize(df_vl, feature_cols, stats=stats)

            W = int(self._best_individual["W"])
            X_vl, y_vl = build_windows(df_vl, W, feature_cols)
            if len(X_vl) == 0:
                self._warn_full_validation("0 fenêtre construite sur df_val_full")
                return None

            batch_size = int(self._best_individual["batch_size"])
            # X_vl reste sur CPU ; seul le batch courant est transféré sur `device`
            # (jusqu'à 450 955 fenêtres en une seule fois serait ~qq centaines de Mo
            # à ~1 Go selon W — significatif sur la mémoire unifiée d'Apple Silicon,
            # partagée avec le reste du système).
            X_vl_cpu = torch.from_numpy(X_vl)
            scores_parts = []
            with torch.no_grad():
                for s in range(0, X_vl_cpu.shape[0], batch_size):
                    xb = X_vl_cpu[s:s + batch_size].to(device)
                    scores_parts.append(model(xb).cpu())
            scores = torch.cat(scores_parts).numpy()
            scores = _sanitize_scores(scores, "evaluate_best_on_full_validation")

            if len(np.unique(y_vl)) < 2:
                self._warn_full_validation("une seule classe présente dans df_val_full — "
                                           "métriques non calculables")
                return None

            result = {
                "MCC": _safe_float(matthews_corrcoef(y_vl, (scores >= 0.5).astype(int))),
                "PR_AUC": _safe_float(average_precision_score(y_vl, scores)),
                "Brier": _safe_float(brier_score_loss(y_vl, scores), default=1.0),
                "AUC_ROC": _safe_float(roc_auc_score(y_vl, scores)),
                "n_windows": int(len(X_vl)),
            }
            self._validation_full = result
            self._write_best_config_json()   # propage validation_full dans le fichier canonique
            return result
        except Exception as exc:   # noqa: BLE001 — ne jamais interrompre pour cette étape optionnelle
            self._warn_full_validation(f"{type(exc).__name__}: {exc}")
            return None
        finally:
            _release_device_memory(device)   # cf. commentaire dans compute_fitness

    @staticmethod
    def _warn_full_validation(msg: str) -> None:
        from tqdm.auto import tqdm
        tqdm.write(f"  [avertissement] validation finale sur l'ensemble complet impossible : {msg}")

    def get_best_model(self, path: Optional[str] = None) -> torch.nn.Module:
        """
        Reconstruit un scoreur entraînable/inférable à partir de
        best_candidate.pt : ConfigurableMKANScorer ou GRUMKANScorer, tous
        deux compatibles avec l'interface attendue par train.ipynb/
        mkan_total_loss (forward(x_window, return_reg=...) -> score (batch,)).
        """
        path = path or self._best_candidate_path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model_from_individual(payload["individual"], payload["input_size"])
        model.load_state_dict(payload["model_state"])
        return model

    # ── Fichier canonique de configuration (section 9) ──────────────────────

    def _write_best_config_json(self) -> None:
        if self._best_individual is None or self._best_result is None:
            return
        ind = self._best_individual
        comp = self._best_result["fitness_components"] or {}

        gates_json = {}
        for gate, gate_cfg in ind["gates"].items():
            base = gate_cfg["Base"]
            gates_json[gate] = {
                "base": base,
                "hyperparams": {k: v for k, v in gate_cfg.items() if k != "Base"},
            }

        config = {
            "metadata": {
                "etape": 2,
                "n_generations": len({r["generation"] for r in self._journal}) if self._journal else 0,
                "n_evaluations": self._n_evaluations,
                "n_windows_fitness": self.n_windows_eval,
                "seed": self.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Champs additionnels (n'altèrent pas le schéma canonique existant) :
                # la sélection interne (élitisme/tournoi) se fait par dominance
                # contrainte NSGA-II (Deb et al., 2002), pas par comparaison
                # scalaire ; ce fichier ne décrit qu'UN représentant du front
                # (le point le plus proche par dominance, départagé par fitness_total
                # AHP en cas de non-dominance mutuelle) — le front complet est dans
                # pareto_front.json.
                "selection_method": "NSGA-II (constrained-domination, Deb et al. 2002)",
                "pareto_front_file": os.path.basename(self._pareto_front_path),
                "pareto_front_size": len(self._pareto_archive),
            },
            "theta_opt": {
                "hidden_size": int(ind["hidden_size"]), "lr": float(ind["lr"]),
                "lam": float(ind["lam"]), "mu1": float(ind["mu1"]), "mu2": float(ind["mu2"]),
                "batch_size": int(ind["batch_size"]), "W": int(ind["W"]),
            },
            "theta_struct": {
                "cell_type": ind["Cell_Type"],
                "hidden_size_gru_adjusted": self._best_result.get("hidden_size_gru_adjusted"),
                "input_regime": ind["Input_Regime"],
                "gates": gates_json,
            },
            "fitness_components": {
                "MCC_raw": comp.get("MCC_raw"), "MCC_clipped": comp.get("MCC_clipped"),
                "PR_AUC": comp.get("PR_AUC"), "Brier": comp.get("Brier"),
                "R2_symbolic": comp.get("R2_symbolic"), "latency_ms": comp.get("latency_ms"),
                "penalty_lat": comp.get("penalty_lat"), "fitness_total": comp.get("fitness_total"),
            },
            "validation_full": dict(self._validation_full) if self._validation_full else {},
        }
        with open(self._best_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    # ── Checkpoint / reprise (section 11) ────────────────────────────────────

    def _save_checkpoint(self, generation: int) -> None:
        state = {
            "generation": generation,
            "population": self._population,
            "scores": self._scores,
            "journal": self._journal,
            "components": self._components,
            "best_individual": self._best_individual,
            "best_score": self._best_score,
            "mutation_rate_courant": self._mutation_rate_courant,
            "n_evaluations": self._n_evaluations,
            "rng_state": self._rng.getstate(),
            "best_candidate_path": self._best_candidate_path,
            "pareto_archive": self._pareto_archive,
        }
        tmp_path = self._checkpoint_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=_json_rng_default, indent=2)
        os.replace(tmp_path, self._checkpoint_path)

    def _restore_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Recharge l'état complet depuis search_checkpoint.json (population, scores,
        historique, meilleur individu, taux de mutation, compteur d'évaluations,
        état RNG) ainsi que best_candidate.pt (state_dict du meilleur modèle, trop
        volumineux pour le JSON) si présent. Retourne False si aucun checkpoint
        n'existe (première exécution — pas une erreur).
        """
        path = checkpoint_path or self._checkpoint_path
        if not os.path.exists(path):
            return False
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        self._population = state["population"]
        self._scores = state["scores"]
        self._journal = state["journal"]
        self._components = state.get("components", [None] * len(self._population))
        self._best_individual = state["best_individual"]
        self._best_score = state["best_score"]
        self._mutation_rate_courant = state["mutation_rate_courant"]
        self._n_evaluations = state["n_evaluations"]
        self._rng.setstate(_json_rng_restore(state["rng_state"]))
        self._last_checkpointed_generation = state["generation"]
        self._pareto_archive = state.get("pareto_archive", [])

        if os.path.exists(self._best_candidate_path):
            try:
                import torch as _torch
                payload = _torch.load(self._best_candidate_path, map_location="cpu", weights_only=False)
                _comp = payload.get("fitness_components") or {}
                self._best_result = {
                    "model_state": payload["model_state"], "input_size": payload.get("input_size"),
                    "n_params": payload.get("n_params"),
                    "hidden_size_gru_adjusted": payload.get("hidden_size_gru_adjusted"),
                    "fitness_components": _comp,
                    "fitness_total": _comp.get("fitness_total"),   # requis par _evaluer_un (comparaison)
                }
            except Exception as exc:   # noqa: BLE001 — reprise best-effort (section 11)
                print(f"  [avertissement] best_candidate.pt illisible au resume ({exc}) — "
                     "_best_result non restauré (get_best_model() restera indisponible).")
        return True

    def resume(self, checkpoint_path: Optional[str] = None) -> pd.DataFrame:
        """
        Recharge l'état disponible (population, scores, historique, meilleur
        individu, seed/RNG) et affiche/retourne un résumé par génération.
        Lève FileNotFoundError si aucun checkpoint n'existe (usage : inspection
        explicite d'un run passé — cf. fit(resume_from_checkpoint=True) pour
        reprendre une recherche interrompue sans erreur si rien n'est trouvé).
        """
        path = checkpoint_path or self._checkpoint_path
        if not self._restore_from_checkpoint(path):
            raise FileNotFoundError(f"Aucun checkpoint trouvé : {path}")

        rows = []
        for gen, group in pd.DataFrame(self._journal).groupby("generation"):
            idx = group["score"].idxmax()
            row = group.loc[idx].to_dict()
            gates = row.get("gates", {})
            comp = row.get("_fitness_components") or {}
            summary = {
                "generation": gen, "best_fitness": row.get("score"),
                "MCC": comp.get("MCC_raw"), "PR_AUC": comp.get("PR_AUC"),
                "Brier": comp.get("Brier"), "R2_symbolic": comp.get("R2_symbolic"),
                "latency_ms": comp.get("latency_ms"),
                "Base_Forget": gates.get("Forget", {}).get("Base") if isinstance(gates, dict) else None,
                "Base_Input": gates.get("Input", {}).get("Base") if isinstance(gates, dict) else None,
                "Base_Candidate": gates.get("Candidate", {}).get("Base") if isinstance(gates, dict) else None,
                "Base_Output": gates.get("Output", {}).get("Base") if isinstance(gates, dict) else None,
                "Cell_Type": row.get("Cell_Type"), "Input_Regime": row.get("Input_Regime"),
            }
            rows.append(summary)
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df

    def _enregistrer_generation(self, generation: int, diversite: float, mutation_rate: float) -> None:
        for ind, score, comp in zip(self._population, self._scores, self._components):
            record = {"generation": generation, "score": score,
                      "diversite": diversite, "mutation_rate": mutation_rate,
                      **ind, "_fitness_components": comp}
            self._journal.append(record)

    # ── Visualisation (section 12) ───────────────────────────────────────────

    def plot_convergence(self):
        import plotly.graph_objects as go

        df = self.history
        stats = df.groupby("generation")["score"].agg(["max", "mean", "std"]).reset_index()
        x = stats["generation"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=stats["mean"].tolist(), mode="lines",
                                 name="Fitness moyenne", line=dict(dash="dot", color="#FF9800")))
        fig.add_trace(go.Scatter(x=x, y=stats["max"].tolist(), mode="lines+markers",
                                 name="Meilleur de la génération", line=dict(color="#2196F3")))
        fig.update_layout(title="Convergence — recherche heuristique étendue (Étape 2)",
                          xaxis_title="Génération", yaxis_title="Fitness", template="plotly_white")
        out_path = os.path.join(self.scratch_dir, "fitness_convergence_extended.png")
        try:
            fig.write_image(out_path)
        except Exception as exc:   # kaleido peut manquer selon l'environnement
            print(f"  [avertissement] export PNG impossible ({exc}) — export HTML de repli.")
            fig.write_html(out_path.replace(".png", ".html"))
        return fig

    # ── Export Top 5 (section 13) ────────────────────────────────────────────

    def export_top5(self, path: Optional[str] = None) -> list:
        path = path or os.path.join(self.scratch_dir, "top5_configs.json")
        df = self.history.sort_values("score", ascending=False)
        seen, top = set(), []
        for _, row in df.iterrows():
            ind = {k: row[k] for k in
                  ("Cell_Type", "Input_Regime", "hidden_size", "lr", "lam", "mu1", "mu2",
                   "batch_size", "W", "gates")}
            key = self._individual_key(ind)
            if key in seen:
                continue
            seen.add(key)
            top.append({"individual": ind, "score": row["score"], "generation": int(row["generation"])})
            if len(top) == 5:
                break
        with open(path, "w", encoding="utf-8") as f:
            json.dump(top, f, indent=2, ensure_ascii=False, default=str)
        return top


def _json_rng_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"non sérialisable : {type(obj)}")


def _json_rng_restore(state):
    """random.getstate() attend (version:int, internal_state:tuple(625 ints), gauss_next)."""
    version, internal, gauss_next = state
    return (version, tuple(internal), gauss_next)
