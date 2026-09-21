"""
niveau0_benchmark_iso.py -- Benchmark synthetique iso-parametrique (Etape 3, Niveau 0).

Evalue chaque base KAN de facon isolee (GenericKANLayer 1->1, sans bruit de
sequence recurrente) sur les 8 fonctions synthetiques des 3 regimes fonctionnels
des portes de la cellule MKAN, a budget parametrique strictement identique
(B=12 parametres apprenables/arete pour toutes les bases).

Ce module implemente exactement le protocole formalise dans
step_3_benchmark_synthetique.tex (chapitre "Preselection Analytique des Bases
KAN par Regime Fonctionnel") :
  - eq. prob_variationnel / eq. rmse (Section "Formalisation du Probleme
    d'Approximation Iso-Parametrique")
  - tableau budget_iso (calibration B=12 par base, deja implementee dans
    edges/bases.py -> BASIS_REGISTRY, *_kwargs)
  - les 8 familles de fonctions (Section "Les Huit Familles de Fonctions
    Synthetiques en Trois Regimes", eq. fs1/fs2/fs3/fc1/fc2/fc3/fo1/fo2)
  - eq. gibbs (indice d'oscillation de Gibbs, Regime 1 uniquement -- calcule
    sur une grille dense DEDIEE de n_gibbs_dense=10000 points par defaut,
    distincte des n_points=1000 equidistants de l'entrainement/RMSE, pour ne
    pas rater un pic de suroscillation tres localise pres d'une discontinuite ;
    cf. _make_gibbs_probe)
  - Section "Protocole du Benchmark Synthetique" (4 metriques : RMSE, Gibbs,
    latence CPU -- mediane de latency_n_calls chronos individuels, plus
    robuste aux pics d'ordonnancement OS qu'une moyenne globale --, T_90%)

Latence : deux nombres distincts, jamais confondus.
  - latency_us (CPU, toujours mesuree) EST la metrique canonique du protocole
    (step_3.tex, contexte de production USSD/Mobile Money CEMAC sans GPU) --
    la SEULE utilisee par les criteres formels de preselection (eq.
    criteres_presel, niveau0_analysis.py/niveau0_presel_export.py).
  - latency_us_training_device (optionnelle, si --device != cpu) est un
    DIAGNOSTIC informatif sur l'accelerateur local (ex. MPS d'un Mac M4) :
    chronometree avec synchronisation explicite du device (_sync_device --
    indispensable sur un device asynchrone, sinon perf_counter() ne mesure
    que le temps de soumission de la commande, pas son execution reelle).
    Jamais utilisee pour une decision de presel : le deploiement cible reste
    CPU-only, et le comparer a un GPU de calcul (V100/A100) n'aurait de sens
    que pour un tout autre cas d'usage que celui de ce memoire.

Relation avec l'Etape 2 : le lien officiel est heuristic_best_config.json
(configuration d'elite Rang 1, GRUKANCell) et top5_configurations.json (dont
le Rang 2 TKANCell, seul candidat exploitant les 4 portes -- Output inclus,
d'ou la pertinence du Regime 3 / fo1-fo2). Ce module ne re-entraine PAS ces
cellules recurrentes ; il isole la question purement fonctionnelle "quelle
base KAN approxime le mieux quel type de signal, a budget egal ?" qui a
motive les choix de bases observes dans ces deux fichiers.

SincKAN (SincBasis, edges/bases.py) est absent du registre historique utilise
par l'Etape 2 (extended_heuristic_search.py, probleme.md) : il a ete
implemente specifiquement pour ce benchmark (cf. edges/bases.py) afin de
completer les 9 bases theoriquement discutees en Etape 1/2, calibre exactement
selon step_3 (N=6 points, deux largeurs de bande, 2N=12 coefficients).

Convention de normalisation d'entree (documentee ici pour transparence) :
chaque base de edges/bases.py attend nativement des entrees dans [-domain,
+domain] (domain=1.0 par defaut). Plutot que de reconfigurer le parametre
`domain` de chaque classe pour chaque fonction (fc2/fo2 vivent sur [-3,3],
fs3 sur [0,1]), le domaine reel [x_min, x_max] de chaque fonction est
reechantillonne affinement vers [-1,1] AVANT d'etre passe a la couche KAN ;
la fonction cible est evaluee sur le x REEL (non normalise). Une reparametrisation
affine du domaine d'entree ne change pas la difficulte d'approximation
(Gibbs, frequence relative, etc.), donc ce choix ne biaise pas la comparaison
inter-bases -- il est necessaire car les classes de edges/bases.py n'exposent
pas de decalage de domaine asymetrique (seulement une demi-largeur symetrique).

Strategie de performance (entrainement batche) : les n_seeds replicats d'un
(base, fonction) sont entraines EN UNE SEULE PASSE via une couche 1 entree ->
n_seeds sorties (chaque colonne de sortie possede ses propres B=12 parametres,
totalement independants des autres colonnes -- aucun terme croise dans la
perte ni dans le forward, donc strictement equivalent a n_seeds entrainements
separes), ce qui divise par n_seeds le nombre de passes de n_iterations
necessaires. EXCEPTION : fkan (JacobiBasis) partage 3 parametres (alpha/beta/
gamma) globalement entre TOUTES les sorties, par design documente de la classe
-- le batcher romprait a la fois le budget iso B=12/replicat (9*n_seeds+3, pas
12*n_seeds) et l'independance stricte entre graines. fkan est donc TOUJOURS
entraine par n_seeds passes sequentielles independantes (cf.
BASES_WITH_SHARED_PARAMS, _run_batch) -- seule base a ne pas beneficier du
gain de vitesse x5 du batching. La latence, elle, ne depend que de
l'architecture (forme des tenseurs), jamais des poids appris ni de la
fonction/graine : elle est mesuree UNE SEULE FOIS par base (9 mesures, y
compris fkan, toujours avec out_features=1) plutot qu'une fois par triplet (360).
Consequence sur la reproductibilite : l'initialisation des poids utilise une
graine UNIQUE partagee par le batch (--init_seed) au lieu de n_seeds appels
individuels torch.manual_seed(seed) ; le bruit de la famille fc3 reste, lui,
tire individuellement par replicat (cf. _make_dataset_batched). 'seed' dans
chaque enregistrement de sortie identifie donc un indice de replicat, pas un
appel de graine independant pour l'init des poids (metadata.batching_note).

Usage :
    python niveau0_benchmark_iso.py
    python niveau0_benchmark_iso.py --n_seeds 5 --n_iterations 2000 --out_dir extended_search
"""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from tqdm import tqdm

from edges import BASIS_REGISTRY, GenericKANLayer


def _select_training_device(prefer: str = None) -> torch.device:
    """Selectionne le device d'ENTRAINEMENT (jamais celui de la mesure de
    latence, qui reste TOUJOURS CPU -- cf. _measure_latency, protocole
    step_3.tex : latence mesuree en contexte de production USSD sans GPU,
    independant de la machine de developpement). Cascade : prefer explicite
    > MPS (Apple Silicon, ex. M4) > CUDA > CPU -- meme convention que
    extended_heuristic_search.select_training_device, reimplementee ici
    localement (pas d'import croise) pour garder ce module leger et
    independant de la chaine d'imports lourde de l'Etape 2.

    Par defaut (prefer=None -> "cpu" au niveau de main(), PAS "auto") : les
    modeles de ce benchmark sont des couches 1->n_seeds a 12*n_seeds
    parametres sur 1000-10000 points -- beaucoup trop petits pour amortir le
    cout de soumission de commandes Metal (MPS) ou CUDA sur 2000 iterations x
    72 combos ; le CPU (1 thread, cf. --num_threads) est empiriquement le
    plus rapide a cette echelle. --device mps/cuda reste disponible pour
    verifier ce compromis sur une machine donnee plutot que de l'imposer
    aveuglement."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Les 9 bases evaluees (step_3, tableau budget_iso) -- mapping label -> (cle du
# registre, kwargs additionnels qui surchargent budget_kwargs pour distinguer
# les variantes Wav-KAN).
# ─────────────────────────────────────────────────────────────────────────────

BASES_TO_EVALUATE = [
    ("relukan",       "relukan",      {}),
    ("efficientkan",  "efficientkan", {}),
    ("wavkan_dog",    "wavkan",       {"wavelet": "dog"}),
    ("wavkan_morlet", "wavkan",       {"wavelet": "morlet"}),
    ("chebyshev",     "chebyshev",    {}),
    ("fourier",       "fourier",      {}),
    ("fkan",          "fkan",         {}),
    ("sinckan",       "sinckan",      {}),
    ("fastkan",       "fastkan",      {}),
]

B_ISO = 12   # budget parametrique iso (parametres apprenables/arete)


# ─────────────────────────────────────────────────────────────────────────────
# Les 8 familles de fonctions synthetiques (step_3, eq. fs1..fo2)
# ─────────────────────────────────────────────────────────────────────────────

def _f_s1(x): return torch.sigmoid(100.0 * x)
def _f_s2(x): return (x.abs() <= 0.5).to(x.dtype)
def _f_s3(x): return torch.exp(-100.0 * x)
def _f_c1(x): return torch.sin(400.0 * math.pi * x)
def _f_c2(x): return torch.cos(5.0 * x) * torch.exp(-x.square() / 2.0)
def _f_c3_clean(x): return torch.sin(4.0 * math.pi * x) + 0.5 * torch.sin(40.0 * math.pi * x)
def _f_o1(x): return 2.0 * x.square() - 3.0 * x + 4.0
def _f_o2(x): return torch.tanh(x)

# (fonction, x_min, x_max, regime (1/2/3), porte(s) recommandee(s), gibbs applicable)
FUNCTIONS = {
    "fs1": (_f_s1,       -1.0, 1.0, 1, "Forget, Input", True),
    "fs2": (_f_s2,       -1.0, 1.0, 1, "Forget, Input", True),
    "fs3": (_f_s3,        0.0, 1.0, 1, "Forget, Input", True),
    "fc1": (_f_c1,       -1.0, 1.0, 2, "Candidate",      False),
    "fc2": (_f_c2,       -3.0, 3.0, 2, "Candidate",      False),
    "fc3": (_f_c3_clean, -1.0, 1.0, 2, "Candidate",      False),
    "fo1": (_f_o1,       -1.0, 1.0, 3, "Output",         False),
    "fo2": (_f_o2,       -3.0, 3.0, 3, "Output",         False),
}

FC3_NOISE_STD = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Entrainement batche (5 graines = 5 colonnes de sortie d'une meme couche) +
# mesure de latence (une seule fois par base, cf. docstring plus bas)
# ─────────────────────────────────────────────────────────────────────────────

def _build_layer(basis_key: str, extra_kwargs: dict, out_features: int = 1):
    cls = BASIS_REGISTRY[basis_key]
    kwargs = dict(cls.budget_kwargs)
    kwargs.update(extra_kwargs)
    basis = cls(in_features=1, out_features=out_features, **kwargs)
    n_params = sum(p.numel() for p in basis.parameters())
    expected = B_ISO * out_features
    if n_params != expected:
        raise RuntimeError(
            f"Calibration iso-parametrique rompue pour '{basis_key}' "
            f"(kwargs={kwargs}, out_features={out_features}) : {n_params} parametres "
            f"apprenables, attendu {expected} ({B_ISO}/sortie). "
            f"Verifier budget_kwargs dans edges/bases.py."
        )
    return GenericKANLayer(basis, in_features=1, out_features=out_features), n_params // out_features


def _normalize_domain(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    """Normalisation affine du domaine reel [x_min, x_max] -> [-1, 1] pour la
    couche KAN (cf. docstring du module). Division precalculee une seule fois
    (inv_range), multiplication dans le terme vectorise (convention du projet :
    multiplication plutot que division dans les chemins chauds, cf.
    extended_heuristic_search.py)."""
    if x_max > x_min:
        inv_range = 1.0 / (x_max - x_min)
        return 2.0 * (x - x_min) * inv_range - 1.0
    return x


def _make_dataset_batched(fn, x_min: float, x_max: float, n_points: int, seeds: list,
                           add_fc3_noise: bool):
    """x est partage entre les n_seeds colonnes (memes points d'evaluation) ; y a
    la forme (n_points, n_seeds). Pour fc3, chaque colonne s recoit SON PROPRE
    bruit, tire via torch.manual_seed(seeds[s]) -- reproductible independamment
    par graine, meme si l'entrainement lui-meme est batche (cf. docstring module)."""
    x = torch.linspace(x_min, x_max, n_points).unsqueeze(-1)
    if add_fc3_noise:
        cols = []
        for s in seeds:
            torch.manual_seed(s)
            noise = torch.randn(n_points, 1) * FC3_NOISE_STD
            cols.append(fn(x) + noise)
        y = torch.cat(cols, dim=1)
    else:
        y = fn(x).expand(n_points, len(seeds)).contiguous()
    # La cible y reste evaluee sur x REEL (non normalise) -- cf. docstring du module.
    x_norm = _normalize_domain(x, x_min, x_max)
    return x_norm, y


def _make_gibbs_probe(fn, x_min: float, x_max: float, n_dense: int):
    """Grille dense DEDIEE a l'indice de Gibbs, distincte de la grille
    d'entrainement/RMSE (n_points=1000, equidistante comme l'exige le
    protocole). Un maillage de 1000 points peut rater un pic de suroscillation
    tres localise pres d'une discontinuite (ex. x=0 pour f_s1, x=+-0.5 pour
    f_s2) -- n_dense=10000 par defaut resout le voisinage de la discontinuite
    ~10x plus finement, sans toucher au RMSE ni a l'entrainement (calcule
    UNE FOIS apres coup, en no_grad, jamais dans la boucle d'optimisation)."""
    x_dense = torch.linspace(x_min, x_max, n_dense).unsqueeze(-1)
    y_dense = fn(x_dense)
    x_dense_norm = _normalize_domain(x_dense, x_min, x_max)
    return x_dense_norm, y_dense


def _sync_device(device: torch.device) -> None:
    """Synchronisation explicite necessaire avant/apres chaque appel chronometre
    sur un device asynchrone (MPS/CUDA) : sans cela, perf_counter() ne mesure
    que le temps de SOUMISSION de la commande au driver Metal/CUDA (quelques
    microsecondes), pas le temps d'EXECUTION reel -- un chrono GPU non
    synchronise est un nombre scientifiquement faux, pas juste imprecis."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    # CPU : execution deja synchrone, rien a faire.


def _measure_latency(basis_key: str, extra_kwargs: dict,
                      latency_n_calls: int, latency_warmup: int,
                      device: torch.device = None, pbar_desc: str = None) -> float:
    """La latence de calcul de phi_B ne depend QUE de l'architecture (forme des
    tenseurs), jamais des valeurs de poids apprises, de la fonction cible ou de
    la graine -- mesuree ici UNE SEULE FOIS par base (couche 1->1 non entrainee,
    poids initiaux quelconques) plutot qu'une fois par triplet (base, fonction,
    graine), ce qui divise par 40 (8 fonctions x 5 graines) le nombre d'appels
    de mesure necessaires sans rien changer au resultat.

    Chaque appel est chronometre INDIVIDUELLEMENT (perf_counter avant/apres
    CHAQUE forward, pas un seul chrono global divise par n_calls) et la
    MEDIANE des n_calls temps est retenue plutot que la moyenne : plus robuste
    aux pics occasionnels d'ordonnancement OS/interruptions qui biaiseraient
    une moyenne sans affecter la latence typique reellement representative.

    device=None (CPU implicite) EST la mesure canonique du protocole
    (step_3.tex : contexte de production USSD sans GPU) -- c'est CELLE-CI, et
    UNIQUEMENT celle-ci, qui alimente le champ 'latency_us' des enregistrements
    et donc les criteres formels de preselection (eq. criteres_presel).
    device=torch.device('mps'/'cuda') est un DIAGNOSTIC informatif optionnel
    (cf. run_benchmark), jamais utilise pour une decision de preselection --
    la latence GPU/MPS ne remplace pas la contrainte reelle de deploiement."""
    device = device or torch.device("cpu")
    layer, _ = _build_layer(basis_key, extra_kwargs, out_features=1)
    layer = layer.to(device)
    layer.eval()
    forward_fn = layer.forward
    x_probe = torch.zeros(1, 1, device=device)
    with torch.no_grad():
        for _ in range(latency_warmup):
            forward_fn(x_probe)
        _sync_device(device)
        latencies_us = torch.empty(latency_n_calls)
        # leave=False : la barre disparait une fois la base terminee (elle est
        # imbriquee sous la barre "Mesure de latence (une fois/base)" ci-dessus,
        # qui, elle, reste affichee) -- evite d'empiler 9 (ou 18) barres figees.
        for i in tqdm(range(latency_n_calls), desc=pbar_desc or "latence", unit="appel",
                      leave=False):
            t_start = time.perf_counter()
            forward_fn(x_probe)
            _sync_device(device)
            t_end = time.perf_counter()
            latencies_us[i] = (t_end - t_start) * 1e6
    return latencies_us.median().item()


# Bases dont AU MOINS UN parametre est PARTAGE globalement (pas replique par sortie),
# par design documente de la classe elle-meme -- cf. JacobiBasis (fkan) : alpha_raw/
# beta_raw/gamma_raw sont des torch.zeros(1) uniques, pas (in_features, out_features, *).
# Consequence : une couche out_features=n_seeds pour une de ces bases (a) NE respecte
# PAS le budget iso B=12/replicat (le total ne scale pas lineairement avec out_features
# -- ex. fkan : 9*n_seeds + 3, PAS 12*n_seeds) ET (b) couple les gradients des n_seeds
# colonnes a travers ce parametre partage, ce qui romprait l'independance stricte entre
# replicats exigee par le protocole (5 graines INDEPENDANTES). Ces bases sont donc
# TOUJOURS entrainees par n_seeds passes sequentielles independantes (out_features=1
# chacune, cf. _run_batch, branche "sequentielle"), jamais batchees -- perte du gain de
# vitesse x5 pour CETTE base uniquement, pas pour les 8 autres.
BASES_WITH_SHARED_PARAMS = {"fkan"}


def _train_and_evaluate(basis_key: str, extra_kwargs: dict, out_features: int,
                         x_norm: torch.Tensor, y: torch.Tensor,
                         n_iterations: int, lr: float, grad_clip: float, device: torch.device,
                         fn, x_min: float, x_max: float, gibbs_applicable: bool,
                         n_gibbs_dense: int, pbar_desc: str):
    """Coeur d'entrainement partage par les deux branches de _run_batch (vectorisee,
    sequentielle) : entraine UNE couche a `out_features` colonnes de sortie
    sur (x_norm, y) pendant n_iterations, puis evalue RMSE/Gibbs/T_90% par colonne.
    Les poids sont TOUJOURS generes sur CPU (_build_layer, torch.randn sans device
    explicite) puis deplaces vers `device` via .to() -- jamais generes directement sur
    MPS/CUDA (les generateurs aleatoires CPU et MPS/CUDA divergent ; generer directement
    sur MPS donnerait des poids initiaux DIFFERENTS de ceux obtenus sur CPU pour le meme
    init_seed, rendant un run --device mps non comparable a un run --device cpu)."""
    layer, n_params = _build_layer(basis_key, extra_kwargs, out_features=out_features)
    layer = layer.to(device)

    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    loss_fn_none = nn.MSELoss(reduction="none")
    forward_fn = layer.forward

    with torch.no_grad():
        initial_loss_vec = loss_fn_none(forward_fn(x_norm), y).mean(dim=0)   # (out_features,)

    t90 = [None] * out_features
    converged = [False] * out_features
    threshold_vec = 0.10 * initial_loss_vec
    params = list(layer.parameters())
    zero_grad = opt.zero_grad
    step = opt.step
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    # leave=False : imbriquee sous une barre parente qui reste affichee -- disparait
    # une fois ce (sous-)combo termine. mininterval releve (0.2s) : 2000 iterations,
    # inutile de re-rendre la barre plus souvent que quelques fois par seconde.
    for it in tqdm(range(1, n_iterations + 1), desc=pbar_desc or "entrainement",
                   unit="it", leave=False, mininterval=0.2):
        zero_grad(set_to_none=True)
        pred = forward_fn(x_norm)
        per_col_loss = loss_fn_none(pred, y).mean(dim=0)   # (out_features,) -- independant/colonne
        loss = per_col_loss.sum()   # somme de termes independants : gradient par colonne
        loss.backward()             # identique a un entrainement individuel en reduction='mean'
        clip_grad_norm_(params, max_norm=grad_clip)
        step()
        if not all(converged):
            crossed = per_col_loss.detach() < threshold_vec
            for s in range(out_features):
                if not converged[s] and bool(crossed[s]):
                    t90[s] = it
                    converged[s] = True

    with torch.no_grad():
        pred_final = forward_fn(x_norm)
        rmse_vec = loss_fn_none(pred_final, y).mean(dim=0).sqrt()
        gibbs_vec = None
        if gibbs_applicable:
            # Grille dense DEDIEE (n_gibbs_dense points, cf. _make_gibbs_probe) --
            # distincte de la grille d'entrainement (n_points=1000) pour ne pas
            # rater un pic de suroscillation localise pres de la discontinuite.
            x_dense_norm, y_dense = _make_gibbs_probe(fn, x_min, x_max, n_gibbs_dense)
            x_dense_norm, y_dense = x_dense_norm.to(device), y_dense.to(device)
            pred_dense = forward_fn(x_dense_norm)
            gibbs_vec = pred_dense.abs().max(dim=0).values - y_dense.abs().max().item()

    return n_params, rmse_vec, gibbs_vec, t90, converged, initial_loss_vec


def _run_batch(basis_label: str, basis_key: str, extra_kwargs: dict,
               function_name: str, seeds: list, init_seed: int,
               n_points: int, n_iterations: int, lr: float, grad_clip: float,
               n_gibbs_dense: int, device: torch.device, pbar_desc: str = None) -> list:
    """Entraine les n_seeds graines d'un (base, fonction). Deux branches :

    - VECTORISEE (defaut, 8 des 9 bases) : une couche 1 entree -> n_seeds sorties,
      chaque colonne de sortie disposant de son propre jeu de B_ISO=12 parametres
      (totalement independants des autres colonnes, cf. forme (in_features,
      out_features, n_basis) de edges/bases.py) -- entrainer ce batch est
      mathematiquement identique a entrainer n_seeds couches 1->1 separement (aucun
      terme croise entre colonnes dans la perte ni dans le forward), mais divise par
      n_seeds le nombre de passes d'entrainement de 2000 iterations necessaires.

    - SEQUENTIELLE (BASES_WITH_SHARED_PARAMS, actuellement {"fkan"} uniquement) :
      n_seeds passes INDEPENDANTES separees (out_features=1 chacune), car au moins un
      parametre de la base est PARTAGE globalement (pas replique par sortie) -- le
      batcher romprait a la fois le budget iso B=12/replicat et l'independance stricte
      entre graines exigee par le protocole (cf. BASES_WITH_SHARED_PARAMS).

    Nuance de reproductibilite (choix valide explicitement, aucun resultat
    anterieur a preserver) : l'initialisation des poids utilise UNE seule graine
    (init_seed) -- pour le chemin vectorise, un seul tirage batche ; pour le chemin
    sequentiel, torch.manual_seed(init_seed) est appele UNE FOIS avant la boucle sur
    les graines (pas reinitialise a chaque graine), pour que les n_seeds tirages
    successifs restent des valeurs DIFFERENTES (diversite) tout en restant
    entierement deterministes. Dans les deux cas, ce n'est plus n_seeds appels
    individuels torch.manual_seed(seed). Le bruit de fc3, en revanche, reste tire
    colonne par colonne via torch.manual_seed(seeds[s]) (cf. _make_dataset_batched) :
    chaque replicat a donc toujours SON propre bruit reproductible, quelle que soit
    la strategie."""
    fn, x_min, x_max, regime, gate_hint, gibbs_applicable = FUNCTIONS[function_name]
    n_seeds = len(seeds)

    x_norm, y = _make_dataset_batched(fn, x_min, x_max, n_points, seeds,
                                       add_fc3_noise=(function_name == "fc3"))
    x_norm, y = x_norm.to(device), y.to(device)

    if basis_key in BASES_WITH_SHARED_PARAMS:
        torch.manual_seed(init_seed)
        n_params = None
        rmse_parts, gibbs_parts, t90, converged, initial_loss_parts = [], [], [], [], []
        for s in range(n_seeds):
            desc = f"{pbar_desc or basis_label} (graine {s + 1}/{n_seeds})"
            n_params, rmse_s, gibbs_s, t90_s, conv_s, init_s = _train_and_evaluate(
                basis_key, extra_kwargs, out_features=1,
                x_norm=x_norm, y=y[:, s:s + 1],
                n_iterations=n_iterations, lr=lr, grad_clip=grad_clip, device=device,
                fn=fn, x_min=x_min, x_max=x_max, gibbs_applicable=gibbs_applicable,
                n_gibbs_dense=n_gibbs_dense, pbar_desc=desc,
            )
            rmse_parts.append(rmse_s)
            gibbs_parts.append(gibbs_s)
            t90.extend(t90_s)
            converged.extend(conv_s)
            initial_loss_parts.append(init_s)
        rmse_vec = torch.cat(rmse_parts)
        gibbs_vec = torch.cat(gibbs_parts) if gibbs_applicable else None
        initial_loss_vec = torch.cat(initial_loss_parts)
    else:
        torch.manual_seed(init_seed)
        n_params, rmse_vec, gibbs_vec, t90, converged, initial_loss_vec = _train_and_evaluate(
            basis_key, extra_kwargs, out_features=n_seeds,
            x_norm=x_norm, y=y,
            n_iterations=n_iterations, lr=lr, grad_clip=grad_clip, device=device,
            fn=fn, x_min=x_min, x_max=x_max, gibbs_applicable=gibbs_applicable,
            n_gibbs_dense=n_gibbs_dense, pbar_desc=pbar_desc,
        )

    records = []
    for s, seed in enumerate(seeds):
        records.append({
            "base": basis_label,
            "function": function_name,
            "seed": seed,
            "regime": regime,
            "n_params": n_params,
            "rmse": float(rmse_vec[s].item()),
            "gibbs_index": float(gibbs_vec[s].item()) if gibbs_vec is not None else None,
            "t90_iterations": t90[s],
            "converged_90pct": converged[s],
            "initial_loss": float(initial_loss_vec[s].item()),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde continue (JSONL incremental + reprise automatique)
# ─────────────────────────────────────────────────────────────────────────────
#
# Un run complet est 9 mesures de latence + 72 combos (base, fonction) a 2000
# iterations chacun : potentiellement long sur CPU. Sans persistance
# incrementale, un crash ou un Ctrl+C a la combo 71/72 ferait TOUT reperdre
# (results n'etait accumule qu'en memoire, ecrit une seule fois a la toute
# fin). Chaque latence/combo termine est desormais ecrit IMMEDIATEMENT (append
# + flush + fsync) dans un fichier .jsonl ; au demarrage, ces fichiers sont
# relus et tout ce qui est deja present est saute plutot que recalcule.
#
# Une signature (hash des hyperparametres qui affectent le resultat) est
# ecrite sur chaque ligne : une ligne dont la signature ne correspond pas au
# run courant est ignoree (jamais melangee silencieusement a un run avec des
# hyperparametres differents -- ex. --n_iterations ou --lr changes entre deux
# lancements).

def _run_signature(n_points, n_iterations, seeds, lr, grad_clip, init_seed,
                    latency_n_calls, latency_warmup, n_gibbs_dense, device_type) -> str:
    payload = {
        "n_points": n_points, "n_iterations": n_iterations, "seeds": list(seeds),
        "lr": lr, "grad_clip": grad_clip, "init_seed": init_seed,
        "latency_n_calls": latency_n_calls, "latency_warmup": latency_warmup,
        "n_gibbs_dense": n_gibbs_dense,
        # Le device d'entrainement fait partie de la signature : les noyaux
        # CPU/MPS/CUDA ne sont pas garantis bit-identiques (ordre de sommation
        # different notamment) -- une reprise ne doit jamais melanger des
        # combos entraines sur des devices differents.
        "device_type": device_type,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())   # ecriture garantie sur disque avant de poursuivre


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration complete
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(seeds, n_points: int, n_iterations: int, lr: float, grad_clip: float,
                   latency_n_calls: int, latency_warmup: int, init_seed: int,
                   out_dir: str, n_gibbs_dense: int, device: torch.device,
                   resume: bool = True) -> list:
    run_signature = _run_signature(n_points, n_iterations, seeds, lr, grad_clip, init_seed,
                                    latency_n_calls, latency_warmup, n_gibbs_dense, device.type)
    latency_progress_path = os.path.join(out_dir, "niveau0_latency_progress.jsonl")
    combo_progress_path = os.path.join(out_dir, "niveau0_combo_progress.jsonl")

    latency_by_base = {}
    latency_device_by_base = {}   # diagnostic uniquement (cf. _measure_latency) -- jamais
                                   # utilise dans les criteres de preselection formels
    completed_combo_keys = set()
    results = []
    n_stale_ignored = 0

    if not resume:
        # --no_resume : repart de zero. Les anciens fichiers sont tronques (pas juste
        # ignores) pour eviter d'accumuler indefiniment des lignes dupliquees a chaque
        # nouveau lancement avec les memes hyperparametres.
        for path in (latency_progress_path, combo_progress_path):
            if os.path.exists(path):
                open(path, "w", encoding="utf-8").close()

    if resume:
        for row in _load_jsonl(latency_progress_path):
            if row.get("run_signature") != run_signature:
                n_stale_ignored += 1
                continue
            latency_by_base[row["base"]] = row["latency_us"]
            if row.get("latency_us_training_device") is not None:
                latency_device_by_base[row["base"]] = row["latency_us_training_device"]
        for row in _load_jsonl(combo_progress_path):
            if row.get("run_signature") != run_signature:
                n_stale_ignored += 1
                continue
            key = (row["base"], row["function"])
            if key not in completed_combo_keys:
                completed_combo_keys.add(key)
                results.extend(row["records"])
        if latency_by_base or completed_combo_keys:
            print(f"[reprise] {len(latency_by_base)}/{len(BASES_TO_EVALUATE)} latence(s) et "
                  f"{len(completed_combo_keys)}/{len(BASES_TO_EVALUATE) * len(FUNCTIONS)} "
                  f"combo(s) deja termine(s) retrouves (signature {run_signature}) -- reprise "
                  f"sans les relancer.")
        if n_stale_ignored:
            print(f"[avertissement] {n_stale_ignored} ligne(s) de progression avec une "
                  f"signature d'hyperparametres differente ignoree(s) (run precedent avec "
                  f"d'autres reglages -- non melange).")

    # Phase latence : mesuree UNE FOIS par base (9 mesures, pas 360 -- cf. _measure_latency).
    # latency_us (CPU) est TOUJOURS mesuree -- c'est la metrique canonique du protocole
    # (step_3.tex). Si --device n'est pas cpu, une SECONDE mesure diagnostique est prise
    # sur ce device (MPS/CUDA) et stockee a part, sans jamais remplacer ni influencer
    # latency_us dans les criteres formels de preselection.
    measure_device_diag = device.type != "cpu"
    bases_remaining = [c for c in BASES_TO_EVALUATE if c[0] not in latency_by_base
                        or (measure_device_diag and c[0] not in latency_device_by_base)]
    latency_pbar = tqdm(bases_remaining, desc="Mesure de latence (une fois/base)", unit="base")
    for basis_label, basis_key, extra_kwargs in latency_pbar:
        latency_pbar.set_postfix_str(basis_label)
        latency_us = latency_by_base.get(basis_label)
        if latency_us is None:
            latency_us = _measure_latency(basis_key, extra_kwargs, latency_n_calls, latency_warmup,
                                           pbar_desc=f"{basis_label} (cpu)")
            latency_by_base[basis_label] = latency_us
        latency_us_device = None
        if measure_device_diag:
            latency_us_device = latency_device_by_base.get(basis_label)
            if latency_us_device is None:
                latency_us_device = _measure_latency(
                    basis_key, extra_kwargs, latency_n_calls, latency_warmup,
                    device=device, pbar_desc=f"{basis_label} ({device.type})")
                latency_device_by_base[basis_label] = latency_us_device
        _append_jsonl(latency_progress_path, {
            "run_signature": run_signature, "base": basis_label, "latency_us": latency_us,
            "latency_us_training_device": latency_us_device,
            "training_device_type": device.type if measure_device_diag else None,
        })

    # Phase entrainement batche par combo (base, fonction)
    combos = [
        (basis_label, basis_key, extra_kwargs, function_name)
        for (basis_label, basis_key, extra_kwargs) in BASES_TO_EVALUATE
        for function_name in FUNCTIONS
    ]
    combos_remaining = [c for c in combos if (c[0], c[3]) not in completed_combo_keys]
    pbar = tqdm(combos_remaining, desc="Benchmark Niveau 0 (batch de graines)", unit="combo")
    for basis_label, basis_key, extra_kwargs, function_name in pbar:
        pbar.set_postfix_str(f"{basis_label}/{function_name} ({len(seeds)} graines batchees)")
        records = _run_batch(
            basis_label, basis_key, extra_kwargs, function_name, seeds, init_seed,
            n_points=n_points, n_iterations=n_iterations, lr=lr, grad_clip=grad_clip,
            n_gibbs_dense=n_gibbs_dense, device=device,
            pbar_desc=f"{basis_label}/{function_name}",
        )
        latency_us = latency_by_base[basis_label]
        latency_us_device = latency_device_by_base.get(basis_label)
        for record in records:
            record["latency_us"] = latency_us                       # canonique (CPU, protocole)
            record["latency_us_training_device"] = latency_us_device  # diagnostic uniquement
            record["training_device_type"] = device.type if measure_device_diag else None
        _append_jsonl(combo_progress_path, {
            "run_signature": run_signature, "base": basis_label, "function": function_name,
            "records": records,
        })
        results.extend(records)
    return results


def _load_etape2_link(project_root: str) -> dict:
    """Resume non-invasif du lien Etape 2 (n'echoue jamais le benchmark si absent)."""
    link = {"heuristic_best_config": None, "top5_rank2_tkan": None}
    best_path = os.path.join(project_root, "extended_search", "heuristic_best_config.json")
    top5_path = os.path.join(project_root, "extended_search", "top5_configurations.json")
    try:
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)
        link["heuristic_best_config"] = {
            "path": best_path,
            "cell_type": best["theta_struct"]["cell_type"],
            "gates": {g: v["base"] for g, v in best["theta_struct"]["gates"].items()},
            "MCC_raw": best["fitness_components"]["MCC_raw"],
        }
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[avertissement] heuristic_best_config.json non exploitable ({exc}).")
    try:
        with open(top5_path, encoding="utf-8") as f:
            top5 = json.load(f)
        rank2 = next((c for c in top5 if c.get("rank") == 2), None)
        if rank2 is not None:
            link["top5_rank2_tkan"] = {
                "path": top5_path,
                "cell_type": rank2["Cell_Type"],
                "Base_Forget": rank2["Base_Forget"],
                "Base_Input": rank2["Base_Input"],
                "Base_Candidate": rank2["Base_Candidate"],
                "Base_Output": rank2["Base_Output"],
                "latency_ms": rank2["latency_ms"],
            }
    except (OSError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        print(f"[avertissement] top5_configurations.json non exploitable ({exc}).")
    return link


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_points", type=int, default=1000)
    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--latency_n_calls", type=int, default=10000)
    parser.add_argument("--latency_warmup", type=int, default=50)
    parser.add_argument("--n_gibbs_dense", type=int, default=10000,
                         help="Nombre de points de la grille dense DEDIEE au calcul de "
                              "I_Gibbs (Regime 1 uniquement), distincte de la grille "
                              "d'entrainement/RMSE (n_points=1000, equidistante comme l'exige "
                              "le protocole) -- evite de rater un pic de suroscillation tres "
                              "localise pres d'une discontinuite.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                         help="Device d'ENTRAINEMENT. Defaut 'cpu' (PAS 'auto') : les modeles "
                              "de ce benchmark sont trop petits (12*n_seeds parametres) pour "
                              "amortir le cout de soumission de commandes MPS/CUDA sur 2000 "
                              "iterations x 72 combos -- CPU 1 thread est empiriquement le plus "
                              "rapide a cette echelle. 'mps' active l'acceleration native Apple "
                              "Silicon (ex. M4) ; 'auto' retombe sur la cascade MPS > CUDA > CPU. "
                              "La latence CANONIQUE du protocole (champ 'latency_us', utilisee "
                              "pour la preselection) reste TOUJOURS mesuree sur CPU quel que "
                              "soit ce choix. Si --device != cpu, une SECONDE latence, purement "
                              "diagnostique (champ 'latency_us_training_device'), est en plus "
                              "mesuree sur ce device -- jamais utilisee pour une decision.")
    parser.add_argument("--out_dir", type=str, default="extended_search",
                         help="Repertoire de scratch de sortie (relatif a ce script).")
    parser.add_argument("--out_name", type=str, default="niveau0_benchmark_results.json")
    parser.add_argument("--init_seed", type=int, default=0,
                         help="Graine d'initialisation des poids, PARTAGEE par les n_seeds "
                              "colonnes d'un meme batch (entrainement batche, cf. docstring "
                              "de _run_batch) -- distincte des graines de bruit de fc3, qui "
                              "restent individuelles par replicat.")
    parser.add_argument("--num_threads", type=int, default=1,
                         help="torch.set_num_threads() : les modeles ici sont des couches "
                              "1->1 a 12 parametres sur des tenseurs de 1000 points -- le "
                              "cout de synchronisation du pool de threads BLAS/dispatch de "
                              "PyTorch domine largement le calcul reel a cette echelle. "
                              "1 thread (defaut) est generalement le plus rapide ; "
                              "augmenter uniquement si le CPU est par ailleurs inutilise "
                              "et que des mesures montrent un gain reel sur cette machine.")
    parser.add_argument("--no_resume", action="store_true",
                         help="Ignore toute progression .jsonl existante (niveau0_latency_"
                              "progress.jsonl / niveau0_combo_progress.jsonl dans out_dir) et "
                              "relance tout depuis zero, meme si une reprise compatible existe.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    project_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)

    device = _select_training_device(args.device)
    if device.type == "cpu":
        print(f"Device d'entrainement : {device} (metrique canonique de latence).")
    else:
        print(f"Device d'entrainement : {device} -- latence canonique ('latency_us') mesuree "
              f"sur CPU comme toujours ; latence diagnostique additionnelle "
              f"('latency_us_training_device') mesuree sur {device}.")

    seeds = list(range(args.n_seeds))
    results = run_benchmark(
        seeds=seeds, n_points=args.n_points, n_iterations=args.n_iterations,
        lr=args.lr, grad_clip=args.grad_clip,
        latency_n_calls=args.latency_n_calls, latency_warmup=args.latency_warmup,
        init_seed=args.init_seed, out_dir=out_dir, resume=not args.no_resume,
        n_gibbs_dense=args.n_gibbs_dense, device=device,
    )

    etape2_link = _load_etape2_link(project_root)

    output = {
        "metadata": {
            "etape": 3,
            "level": 0,
            "B_iso": B_ISO,
            "n_points": args.n_points,
            "n_iterations": args.n_iterations,
            "seeds": seeds,
            "init_seed": args.init_seed,
            "batching_note": (
                "Les n_seeds graines d'un (base, fonction) sont entrainees en un seul "
                "batch (couche 1 entree -> n_seeds sorties, colonnes independantes) "
                "plutot que via n_seeds boucles sequentielles : 'seed' identifie un "
                "indice de replicat, pas un appel individuel torch.manual_seed(seed) "
                "pour l'initialisation des poids (init_seed unique, partagee). Le bruit "
                "de la famille fc3 reste tire individuellement par replicat via "
                "torch.manual_seed(seed)."
            ),
            "optimizer": "Adam",
            "lr": args.lr,
            "grad_clip_max_norm": args.grad_clip,
            "training_device": device.type,
            "latency_n_calls": args.latency_n_calls,
            "latency_warmup_calls": args.latency_warmup,
            "latency_device": "cpu",
            "latency_aggregation": "median",
            "latency_note": (
                "'latency_us' (CPU) est la metrique CANONIQUE du protocole "
                "(step_3.tex, USSD sans GPU) -- seule utilisee par les criteres "
                "formels de preselection. 'latency_us_training_device' "
                "(present uniquement si training_device != cpu) est un "
                "diagnostic informatif sur l'accelerateur local (ex. MPS "
                "d'un Mac M4), jamais utilise pour une decision."
            ),
            "n_gibbs_dense": args.n_gibbs_dense,
            "sequential_training_bases": sorted(BASES_WITH_SHARED_PARAMS),
            "sequential_training_note": (
                "Bases entrainees par n_seeds passes sequentielles independantes "
                "(out_features=1 chacune) plutot que par le batching vectorise "
                "habituel, car au moins un de leurs parametres est PARTAGE "
                "globalement entre toutes les sorties (ex. fkan/JacobiBasis : "
                "alpha/beta/gamma) -- le batching romprait le budget iso "
                "B=12/replicat ET l'independance stricte entre graines. "
                "Cf. BASES_WITH_SHARED_PARAMS dans le code."
            ),
            "torch_num_threads": args.num_threads,
            "sinckan_status": (
                "implemente specifiquement pour ce benchmark (SincBasis, edges/bases.py) "
                "-- absent du registre historique utilise par l'Etape 2 "
                "(extended_heuristic_search.py / BASIS_REGISTRY d'origine)."
            ),
            "etape2_link": etape2_link,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "bases_evaluated": [label for label, _, _ in BASES_TO_EVALUATE],
        "functions_evaluated": {
            name: {"regime": regime, "domain": [x_min, x_max], "gate_hint": gate_hint,
                   "gibbs_applicable": gibbs_applicable}
            for name, (_, x_min, x_max, regime, gate_hint, gibbs_applicable) in FUNCTIONS.items()
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{len(results)} enregistrements ecrits dans : {out_path}")
    print(f"Fichiers de progression conserves dans {out_dir} "
          f"(niveau0_latency_progress.jsonl, niveau0_combo_progress.jsonl) -- "
          f"supprimables sans risque une fois le JSON final valide, ou laisses "
          f"tels quels pour accelerer un futur run avec les memes hyperparametres.")


if __name__ == "__main__":
    main()
