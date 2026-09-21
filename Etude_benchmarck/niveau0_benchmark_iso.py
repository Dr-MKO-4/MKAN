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
# Entrainement + mesure d'un triplet (base, fonction, graine)
# ─────────────────────────────────────────────────────────────────────────────

def _build_layer(basis_key: str, extra_kwargs: dict) -> GenericKANLayer:
    cls = BASIS_REGISTRY[basis_key]
    kwargs = dict(cls.budget_kwargs)
    kwargs.update(extra_kwargs)
    basis = cls(in_features=1, out_features=1, **kwargs)
    n_params = sum(p.numel() for p in basis.parameters())
    if n_params != B_ISO:
        raise RuntimeError(
            f"Calibration iso-parametrique rompue pour '{basis_key}' "
            f"(kwargs={kwargs}) : {n_params} parametres apprenables, attendu {B_ISO}. "
            f"Verifier budget_kwargs dans edges/bases.py."
        )
    return GenericKANLayer(basis, in_features=1, out_features=1), n_params


def _make_dataset(fn, x_min: float, x_max: float, n_points: int, seed: int,
                   add_fc3_noise: bool):
    x = torch.linspace(x_min, x_max, n_points).unsqueeze(-1)
    if add_fc3_noise:
        # Bruit tire APRES fixation de la graine, pour reproductibilite exacte
        # (le calcul de la grille x via linspace ne consomme pas le generateur).
        torch.manual_seed(seed)
        noise = torch.randn(n_points, 1) * FC3_NOISE_STD
        y = fn(x) + noise
    else:
        y = fn(x)
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


def _run_one(basis_label: str, basis_key: str, extra_kwargs: dict,
             function_name: str, seed: int,
             n_points: int, n_iterations: int, lr: float, grad_clip: float,
             latency_n_calls: int, latency_warmup: int) -> dict:
    fn, x_min, x_max, regime, gate_hint, gibbs_applicable = FUNCTIONS[function_name]

    x_norm, y = _make_dataset(fn, x_min, x_max, n_points, seed,
                               add_fc3_noise=(function_name == "fc3"))

    # Initialisation des poids reproductible et INDEPENDANTE du dataset construit
    # ci-dessus (evite toute confusion entre "graine du bruit" et "graine des poids").
    torch.manual_seed(seed)
    layer, n_params = _build_layer(basis_key, extra_kwargs)

    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    forward_fn = layer.forward   # bypass __call__ (aucun hook nn.Module enregistre ici) --
                                  # meme raisonnement que pour la mesure de latence plus bas,
                                  # applique ici a la boucle d'entrainement (720 000 appels
                                  # au total sur l'ensemble du benchmark).

    with torch.no_grad():
        initial_loss = loss_fn(forward_fn(x_norm), y).item()

    t90 = None
    converged = False
    threshold = 0.10 * initial_loss
    params = list(layer.parameters())
    # Lookups mis en cache en variables locales : dans CPython, une resolution de
    # nom local est plus rapide qu'une chaine d'attributs repetee (opt.zero_grad,
    # torch.nn.utils.clip_grad_norm_, opt.step) -- gain marginal par iteration,
    # mais la boucle tourne 2000 x 360 = 720 000 fois sur l'ensemble du benchmark.
    zero_grad = opt.zero_grad
    step = opt.step
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    for it in range(1, n_iterations + 1):
        zero_grad(set_to_none=True)   # evite le memset des gradients (plus rapide que
                                       # zero_grad() par defaut, cf. recommandation PyTorch
                                       # pour les modeles a peu de parametres/CPU)
        pred = forward_fn(x_norm)
        loss = loss_fn(pred, y)
        loss.backward()
        clip_grad_norm_(params, max_norm=grad_clip)
        step()
        if not converged and loss.item() < threshold:
            t90 = it
            converged = True

    with torch.no_grad():
        pred_final = forward_fn(x_norm)
        rmse = math.sqrt(loss_fn(pred_final, y).item())
        gibbs_index = None
        if gibbs_applicable:
            gibbs_index = (pred_final.abs().max().item()
                            - y.abs().max().item())

    # Latence CPU : appels consecutifs sur UN point fixe (le cout de calcul
    # d'une base KAN ne depend pas de la valeur de x, seulement de sa forme
    # architecturale ; un seul point suffit et evite le bruit de mesure lie
    # au rechargement d'un batch a chaque appel).
    # .forward() est appele directement (plutot que __call__) pour ecarter le
    # dispatch des hooks nn.Module (aucun hook n'est enregistre nulle part dans
    # ce projet) : la latence mesuree reflete alors le cout de calcul de la
    # base elle-meme, pas la machinerie generique de PyTorch autour.
    layer_cpu = layer.to("cpu")
    x_probe = x_norm[:1].detach().clone()
    layer_cpu.eval()
    forward = layer_cpu.forward
    with torch.no_grad():
        for _ in range(latency_warmup):
            forward(x_probe)
        t_start = time.perf_counter()
        for _ in range(latency_n_calls):
            forward(x_probe)
        t_end = time.perf_counter()
    latency_us = (t_end - t_start) / latency_n_calls * 1e6

    return {
        "base": basis_label,
        "function": function_name,
        "seed": seed,
        "regime": regime,
        "n_params": n_params,
        "rmse": rmse,
        "gibbs_index": gibbs_index,
        "latency_us": latency_us,
        "t90_iterations": t90,
        "converged_90pct": converged,
        "initial_loss": initial_loss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration complete
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(seeds, n_points: int, n_iterations: int, lr: float,
                   grad_clip: float, latency_n_calls: int, latency_warmup: int) -> list:
    triplets = [
        (basis_label, basis_key, extra_kwargs, function_name, seed)
        for (basis_label, basis_key, extra_kwargs) in BASES_TO_EVALUATE
        for function_name in FUNCTIONS
        for seed in seeds
    ]
    results = []
    pbar = tqdm(triplets, desc="Benchmark Niveau 0", unit="run")
    for basis_label, basis_key, extra_kwargs, function_name, seed in pbar:
        pbar.set_postfix_str(f"{basis_label}/{function_name}/seed={seed}")
        record = _run_one(
            basis_label, basis_key, extra_kwargs, function_name, seed,
            n_points=n_points, n_iterations=n_iterations, lr=lr, grad_clip=grad_clip,
            latency_n_calls=latency_n_calls, latency_warmup=latency_warmup,
        )
        results.append(record)
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
