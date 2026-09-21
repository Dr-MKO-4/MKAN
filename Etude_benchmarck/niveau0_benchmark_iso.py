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
  - eq. gibbs (indice d'oscillation de Gibbs, Regime 1 uniquement)
  - Section "Protocole du Benchmark Synthetique" (4 metriques : RMSE, Gibbs,
    latence CPU, T_90%)

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
necessaires. La latence, elle, ne depend que de l'architecture (forme des
tenseurs), jamais des poids appris ni de la fonction/graine : elle est mesuree
UNE SEULE FOIS par base (9 mesures) plutot qu'une fois par triplet (360).
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
import json
import math
import os
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from tqdm import tqdm

from edges import BASIS_REGISTRY, GenericKANLayer


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
    # Normalisation affine du domaine reel [x_min, x_max] -> [-1, 1] pour la
    # couche KAN (cf. docstring du module). La cible y reste evaluee sur x reel.
    # Division precalculee une seule fois (inv_range), multiplication dans le
    # terme vectorise (convention du projet : multiplication plutot que division
    # dans les chemins recurrents/chauds, cf. extended_heuristic_search.py).
    if x_max > x_min:
        inv_range = 1.0 / (x_max - x_min)
        x_norm = 2.0 * (x - x_min) * inv_range - 1.0
    else:
        x_norm = x
    return x_norm, y


def _measure_latency(basis_key: str, extra_kwargs: dict,
                      latency_n_calls: int, latency_warmup: int) -> float:
    """La latence de calcul de phi_B ne depend QUE de l'architecture (forme des
    tenseurs), jamais des valeurs de poids apprises, de la fonction cible ou de
    la graine -- mesuree ici UNE SEULE FOIS par base (couche 1->1 non entrainee,
    poids initiaux quelconques) plutot qu'une fois par triplet (base, fonction,
    graine), ce qui divise par 40 (8 fonctions x 5 graines) le nombre d'appels
    de mesure necessaires sans rien changer au resultat."""
    layer, _ = _build_layer(basis_key, extra_kwargs, out_features=1)
    layer.eval()
    forward_fn = layer.forward
    x_probe = torch.zeros(1, 1)
    with torch.no_grad():
        for _ in range(latency_warmup):
            forward_fn(x_probe)
        t_start = time.perf_counter()
        for _ in range(latency_n_calls):
            forward_fn(x_probe)
        t_end = time.perf_counter()
    return (t_end - t_start) / latency_n_calls * 1e6


def _run_batch(basis_label: str, basis_key: str, extra_kwargs: dict,
               function_name: str, seeds: list, init_seed: int,
               n_points: int, n_iterations: int, lr: float, grad_clip: float) -> list:
    """Entraine les n_seeds graines d'un (base, fonction) EN UNE SEULE PASSE :
    une couche 1 entree -> n_seeds sorties, chaque colonne de sortie disposant
    de son propre jeu de B_ISO=12 parametres (totalement independants des
    autres colonnes, cf. forme (in_features, out_features, n_basis) de
    edges/bases.py) -- entrainer ce batch est mathematiquement identique a
    entrainer n_seeds couches 1->1 separement (aucun terme croise entre
    colonnes dans la perte ni dans le forward), mais divise par n_seeds le
    nombre de passes d'entrainement de 2000 iterations necessaires.

    Nuance de reproductibilite (choix valide explicitement, aucun resultat
    anterieur a preserver) : l'initialisation des poids utilise UNE seule
    graine (init_seed) pour tout le tenseur batche, et non plus n_seeds appels
    individuels torch.manual_seed(seed). Le bruit de fc3, en revanche, reste
    tire colonne par colonne via torch.manual_seed(seeds[s]) (cf.
    _make_dataset_batched) : chaque replicat a donc toujours SON propre bruit
    reproductible, seule l'initialisation des poids est mutualisee dans le tirage
    aleatoire du batch plutot que rejouee individuellement."""
    fn, x_min, x_max, regime, gate_hint, gibbs_applicable = FUNCTIONS[function_name]
    n_seeds = len(seeds)

    x_norm, y = _make_dataset_batched(fn, x_min, x_max, n_points, seeds,
                                       add_fc3_noise=(function_name == "fc3"))

    torch.manual_seed(init_seed)
    layer, n_params = _build_layer(basis_key, extra_kwargs, out_features=n_seeds)

    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    loss_fn_none = nn.MSELoss(reduction="none")
    forward_fn = layer.forward

    with torch.no_grad():
        initial_loss_vec = loss_fn_none(forward_fn(x_norm), y).mean(dim=0)   # (n_seeds,)

    t90 = [None] * n_seeds
    converged = [False] * n_seeds
    threshold_vec = 0.10 * initial_loss_vec
    params = list(layer.parameters())
    zero_grad = opt.zero_grad
    step = opt.step
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    for it in range(1, n_iterations + 1):
        zero_grad(set_to_none=True)
        pred = forward_fn(x_norm)
        per_col_loss = loss_fn_none(pred, y).mean(dim=0)   # (n_seeds,) -- independant/colonne
        loss = per_col_loss.sum()   # somme de termes independants : gradient par colonne
        loss.backward()             # identique a un entrainement individuel en reduction='mean'
        clip_grad_norm_(params, max_norm=grad_clip)
        step()
        if not all(converged):
            crossed = per_col_loss.detach() < threshold_vec
            for s in range(n_seeds):
                if not converged[s] and bool(crossed[s]):
                    t90[s] = it
                    converged[s] = True

    with torch.no_grad():
        pred_final = forward_fn(x_norm)
        rmse_vec = loss_fn_none(pred_final, y).mean(dim=0).sqrt()
        gibbs_vec = None
        if gibbs_applicable:
            gibbs_vec = pred_final.abs().max(dim=0).values - y.abs().max(dim=0).values

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
# Orchestration complete
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(seeds, n_points: int, n_iterations: int, lr: float, grad_clip: float,
                   latency_n_calls: int, latency_warmup: int, init_seed: int) -> list:
    # Latence mesuree UNE FOIS par base (9 mesures, pas 360 -- cf. _measure_latency).
    latency_by_base = {}
    for basis_label, basis_key, extra_kwargs in tqdm(
            BASES_TO_EVALUATE, desc="Mesure de latence (une fois/base)", unit="base"):
        latency_by_base[basis_label] = _measure_latency(
            basis_key, extra_kwargs, latency_n_calls, latency_warmup)

    combos = [
        (basis_label, basis_key, extra_kwargs, function_name)
        for (basis_label, basis_key, extra_kwargs) in BASES_TO_EVALUATE
        for function_name in FUNCTIONS
    ]
    results = []
    pbar = tqdm(combos, desc="Benchmark Niveau 0 (batch de graines)", unit="combo")
    for basis_label, basis_key, extra_kwargs, function_name in pbar:
        pbar.set_postfix_str(f"{basis_label}/{function_name} ({len(seeds)} graines batchees)")
        records = _run_batch(
            basis_label, basis_key, extra_kwargs, function_name, seeds, init_seed,
            n_points=n_points, n_iterations=n_iterations, lr=lr, grad_clip=grad_clip,
        )
        latency_us = latency_by_base[basis_label]
        for record in records:
            record["latency_us"] = latency_us
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
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    project_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)

    seeds = list(range(args.n_seeds))
    results = run_benchmark(
        seeds=seeds, n_points=args.n_points, n_iterations=args.n_iterations,
        lr=args.lr, grad_clip=args.grad_clip,
        latency_n_calls=args.latency_n_calls, latency_warmup=args.latency_warmup,
        init_seed=args.init_seed,
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
            "latency_n_calls": args.latency_n_calls,
            "latency_warmup_calls": args.latency_warmup,
            "latency_device": "cpu",
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


if __name__ == "__main__":
    main()
