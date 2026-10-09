"""
niveau1_benchmark_run.py -- Etape 4 (Niveau 1) : plan de croisement des
architectures heterogenes MKAN sur MoMTSim.

Deux strategies (--strategy), meme entrainement/metriques/qualite par run :
  - raw_first (DEFAUT) : phase 1 sur le regime raw seul (36 architectures x
    5 graines = 180 runs), puis extension du Top --top_n_engineered (defaut
    3) vers engineered (+15 runs pour N=3) = ~195 runs. Decision prise en
    cours de session : le plan factoriel integral (360 runs) s'est revele
    beaucoup trop long en pratique (52% de la phase d'entrainement en 18h de
    calcul reel) ; cette reduction ne touche NI epochs NI le nombre de
    graines NI la taille du dataset (aucune perte de qualite par run), et
    s'appuie sur un resultat DEJA observe empiriquement en Etape 2 (raw >
    engineered, MCC 0.9834 vs 0.9559 pour la config Rang 3 engineered).
  - full : plan factoriel integral, 36 architectures heterogenes x 2 regimes
    d'entree (brut/traite) x 5 graines = 360 runs. Aucun angle mort, mais
    nettement plus long -- a reserver a une machine dediee sur plusieurs jours.

Assemble les architectures heterogenes candidates (candidats Niveau 0 UNION
Elite Etape 2, verrouilles par la discussion), puis identifie la configuration
heterogene optimale et la compare au modele MKAN uniforme de reference
(base "hybrid" sur toutes les portes).

Ne reimplemente RIEN de l'entrainement recurrent : reutilise integralement
niveau1_harness.run_config() (deja le harness valide pour ce protocole) et,
pour les metriques manquantes a la formule de fitness reelle de l'Etape 2
(MCC, latence CPU, R2_symbolic), reutilise directement _measure_latency_ms /
_compute_r2_symbolic / _penalty_latency de extended_heuristic_search.py --
AUCUNE duplication de logique deja ecrite et testee.

Candidats par porte (UNION Retenues Niveau 0 u Elite Etape 2, verrouille) :
    Forget    : relukan, wavkan_dog                          (2)
    Input     : relukan, wavkan_dog, efficientkan             (3)
    Candidate : wavkan_morlet, wavkan_dog                     (2)
    Output    : wavkan_morlet, linear   (TKANCell uniquement) (2)
    GRUKANCell (3 portes)  : 2x3x2       = 12 architectures
    TKANCell   (4 portes)  : 2x3x2x2     = 24 architectures
    Total                  : 36 architectures x 2 regimes x 5 graines = 360 runs

Hyperparametres theta_opt (valeurs d'elite REELLES de l'Etape 2, PAS partagees
entre cellules -- verifie que mu2 differe reellement entre GRU et TKAN) :
    GRUKANCell : hidden_size=128 (reference LSTM pour match_gru_hidden_size,
                 PAS 170 -- l'adjustement reel du Niveau 1 est calcule
                 dynamiquement par match_gru_hidden_size(input_size, 128),
                 differente de la formule floor(4/3*128) de l'Etape 2, cf.
                 discussion et docstring de hidden_size_gru_adjusted),
                 W=5, lr=0.0001, batch_size=32, lam=0.0001, mu1=0.1,
                 mu2=0.6989017708691363 (heuristic_best_config.json, Rang 1).
    TKANCell   : hidden_size=32, W=10, lr=0.0001, batch_size=32, lam=0.0001,
                 mu1=0.1, mu2=1.6515820625005828 (top5_configurations.json,
                 Rang 2 -- PAS le mu2 du Rang 1, valeur differente verifiee).

Hyperparametres structurels par base (gate_kwargs) : quand une base a une
valeur d'elite REELLE et VERIFIEE dans heuristic_best_config.json /
top5_configurations.json pour au moins une porte, cette valeur est reutilisee
PARTOUT ou la base apparait (aucune autre donnee reelle disponible). Sinon
(base retenue uniquement au Niveau 0, jamais vue dans un config d'elite reel :
wavkan_morlet), le budget_kwargs iso B=12 de BASIS_REGISTRY sert de repli
documente -- jamais invente, jamais silencieux (cf. CANDIDATE_KWARGS).

Fitness (formule REELLE de l'Etape 2, reutilisee telle quelle, PAS la version
tronquee sans R2_symbolic proposee initialement). CORRECTIF (2026-10-01) :
utilise desormais le vecteur propre AHP non arrondi (AHP_WEIGHT_*, importes
depuis extended_heuristic_search.py) au lieu des poids arrondis a des
multiples de 0.05 utilises precedemment :
    fitness_total = AHP_WEIGHT_MCC*max(0,MCC) + AHP_WEIGHT_PR_AUC*PR_AUC
                    - AHP_WEIGHT_BRIER*Brier
                    + AHP_WEIGHT_R2_SYMBOLIC*R2_symbolic
                    - AHP_WEIGHT_LATENCY_PENALTY*penalty_latence
    soit, aux valeurs numeriques : 0.4147*max(0,MCC) + 0.2573*PR_AUC
                    - 0.1529*Brier + 0.0876*R2_symbolic - 0.0876*penalty_latence

Usage :
    python niveau1_benchmark_run.py
    python niveau1_benchmark_run.py --epochs 10 --device mps --n_seeds 5
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import tqdm

_ETUDE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ETUDE_DIR not in sys.path:
    sys.path.insert(0, _ETUDE_DIR)

from niveau1_harness import RunConfig, run_config, load_and_window, GRU_GATE_MAP  # noqa: E402
from extended_heuristic_search import (                          # noqa: E402
    _measure_latency_ms, _compute_r2_symbolic, _penalty_latency, _release_device_memory,
    AHP_WEIGHT_MCC, AHP_WEIGHT_PR_AUC, AHP_WEIGHT_BRIER,
    AHP_WEIGHT_R2_SYMBOLIC, AHP_WEIGHT_LATENCY_PENALTY,
)
from edges import BASIS_REGISTRY                                  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Candidats verrouilles par porte (discussion validee) + hyperparametres
# structurels sources (valeur reelle si disponible, sinon budget iso B=12).
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES_BY_GATE = {
    "Forget":    ["relukan", "wavkan_dog"],
    "Input":     ["relukan", "wavkan_dog", "efficientkan"],
    "Candidate": ["wavkan_morlet", "wavkan_dog"],
    "Output":    ["wavkan_morlet", "linear"],
}
GATE_TO_NIVEAU1_KEY = {"Forget": "forget", "Input": "input",
                        "Candidate": "candidate", "Output": "output"}

# label (niveau0/discussion) -> (cle BASIS_REGISTRY, kwargs structurels)
#   - relukan     : Grid_G=8 -> n_bases=8 (heuristic_best_config.json, Forget, Rang 1)
#   - efficientkan: Grid_G=3, Spline_Degree_k=4 (heuristic_best_config.json, Input, Rang 1)
#   - wavkan_dog  : M_wavelets=3 (heuristic_best_config.json Candidate Rang 1 ET
#                   top5_configurations.json Candidate Rang 2 -- valeur coherente
#                   dans les 2 configs d'elite reelles)
#   - wavkan_morlet, linear : jamais vus dans un config d'elite reel -> budget
#                   iso B=12 de BASIS_REGISTRY (repli documente, pas invente)
CANDIDATE_KWARGS = {
    "relukan":       {"n_bases": 8},
    "wavkan_dog":     {"wavelet": "dog", "n_wavelets": 3},
    "wavkan_morlet":  {**BASIS_REGISTRY["wavkan"].budget_kwargs, "wavelet": "morlet"},
    "efficientkan":  {"G": 3, "k": 4},
    "linear":        {},
}


def _basis_key(label: str) -> str:
    return "wavkan" if label.startswith("wavkan") else label


def _gate_kwargs(label: str) -> dict:
    return dict(CANDIDATE_KWARGS.get(label, {}))


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparametres theta_opt d'elite REELS, PAR TYPE DE CELLULE (jamais partages)
# ─────────────────────────────────────────────────────────────────────────────

CELL_THETA_OPT = {
    "gru": dict(hidden_size=128, W=5, lr=1e-4, batch_size=32,
                lam=1e-4, mu1=0.1, mu2=0.6989017708691363),
    "lstm": dict(hidden_size=32, W=10, lr=1e-4, batch_size=32,
                 lam=1e-4, mu1=0.1, mu2=1.6515820625005828),
}

REGIME_MAP = {"raw": "brut", "engineered": "traite"}   # niveau0/discussion -> niveau1_harness


def _build_gate_bases(cell_type: str, base_by_gate: dict) -> "tuple[dict, dict]":
    """base_by_gate : {"Forget": label, "Input": label, "Candidate": label,
    "Output": label (ignore si gru)} -> (gate_bases, gate_kwargs) au format
    niveau1_harness (cles reset/update/candidate pour gru, forget/input/
    candidate/output pour lstm/TKAN)."""
    gate_bases, gate_kwargs = {}, {}
    for gate_canon, label in base_by_gate.items():
        if cell_type == "gru":
            gate_key = GRU_GATE_MAP.get(GATE_TO_NIVEAU1_KEY[gate_canon])
            if gate_key is None:   # Output n'existe pas en GRU
                continue
        else:
            gate_key = GATE_TO_NIVEAU1_KEY[gate_canon]
        gate_bases[gate_key] = _basis_key(label)
        kw = _gate_kwargs(label)
        if kw:
            gate_kwargs[gate_key] = kw
    return gate_bases, gate_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Plan de croisement (36 architectures x 2 regimes x 5 graines)
# ─────────────────────────────────────────────────────────────────────────────

def build_architecture_grid() -> list:
    """36 architectures heterogenes (12 GRU + 24 TKAN), sans regime/graine."""
    archs = []
    for forget in CANDIDATES_BY_GATE["Forget"]:
        for inp in CANDIDATES_BY_GATE["Input"]:
            for cand in CANDIDATES_BY_GATE["Candidate"]:
                base_by_gate_gru = {"Forget": forget, "Input": inp, "Candidate": cand}
                archs.append({"cell_type": "gru", "base_by_gate": base_by_gate_gru})
                for outp in CANDIDATES_BY_GATE["Output"]:
                    base_by_gate_tkan = dict(base_by_gate_gru, Output=outp)
                    archs.append({"cell_type": "lstm", "base_by_gate": base_by_gate_tkan})
    return archs


def build_regime_grid(archs: list, regime_label: str, seeds: list) -> list:
    regime_niveau1 = REGIME_MAP[regime_label]
    grid = []
    for arch in archs:
        for seed in seeds:
            grid.append({
                "cell_type": arch["cell_type"],
                "base_by_gate": arch["base_by_gate"],
                "regime": regime_label,
                "regime_niveau1": regime_niveau1,
                "seed": seed,
            })
    return grid


def build_full_grid(seeds: list) -> list:
    """Plan factoriel integral (360 runs) -- conserve pour --strategy full."""
    archs = build_architecture_grid()
    grid = []
    for regime_label in REGIME_MAP:
        grid.extend(build_regime_grid(archs, regime_label, seeds))
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Fitness complete (metriques manquantes a run_config -- MCC deja ajoute a
# niveau1_harness.compute_metrics ; latence + R2_symbolic ici, en reutilisant
# extended_heuristic_search._measure_latency_ms / _compute_r2_symbolic).
# ─────────────────────────────────────────────────────────────────────────────

def _build_x_pool(model, cell_type: str, X_train_t: torch.Tensor, W: int,
                   n_audit: int = 256) -> torch.Tensor:
    """Reproduit exactement la construction de x_pool de
    extended_heuristic_search.compute_fitness (concat(h_t final, dernier pas
    de temps d'entree)) -- necessaire pour _compute_r2_symbolic, qui evalue
    les aretes sur leur distribution d'entree REELLE (concat_size =
    hidden_size + input_size), pas sur un bruit arbitraire."""
    n = min(n_audit, X_train_t.shape[0])
    audit_X = X_train_t[:n].to("cpu")
    model_cpu = model  # deja sur cpu au moment de l'appel (cf. compute_fitness_extras)
    h_t = torch.zeros(n, model_cpu.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        if cell_type == "lstm":
            c_t = torch.zeros(n, model_cpu.hidden_size, dtype=torch.float32)
            for t in range(W - 1):
                h_t, c_t = model_cpu.cell(audit_X[:, t, :], h_t, c_t)
        else:
            for t in range(W - 1):
                h_t = model_cpu.cell(audit_X[:, t, :], h_t)
    return torch.cat([h_t, audit_X[:, W - 1, :]], dim=1)


def compute_fitness_extras(metrics: dict, cell_type: str, W: int,
                            enable_latency_penalty: bool) -> dict:
    """Complete metrics (deja rempli par run_config(..., return_model=True),
    contient _model/_input_size/_X_train_pool) avec latency_ms, R2_symbolic,
    penalty_lat et fitness_total (formule REELLE 5 termes de l'Etape 2,
    reutilisee identique -- cf. docstring module)."""
    model = metrics.pop("_model")
    input_size = metrics.pop("_input_size")
    X_train_pool = metrics.pop("_X_train_pool")

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    latency_ms = _measure_latency_ms(model_cpu, W, input_size)
    x_pool = _build_x_pool(model_cpu, cell_type, X_train_pool, W)
    r2_symbolic = _compute_r2_symbolic(model_cpu, x_pool)
    penalty_lat = _penalty_latency(latency_ms)

    def _finite(x, default):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        return x if math.isfinite(x) else default

    mcc_clipped = max(0.0, _finite(metrics.get("mcc"), 0.0))
    pr_auc = _finite(metrics.get("auc_pr"), 0.0)
    # Ancien code : `metrics.get("brier", 1.0) or 1.0` transformait un Brier
    # parfait (0.0) en 1.0 (pire valeur) ; un NaN passait tel quel.
    brier = _finite(metrics.get("brier"), 1.0)
    penalty_applied = penalty_lat if enable_latency_penalty else 0.0

    fitness_total = (AHP_WEIGHT_MCC * mcc_clipped + AHP_WEIGHT_PR_AUC * pr_auc
                      - AHP_WEIGHT_BRIER * brier
                      + AHP_WEIGHT_R2_SYMBOLIC * r2_symbolic
                      - AHP_WEIGHT_LATENCY_PENALTY * penalty_applied)

    metrics.update({
        "mcc_clipped": mcc_clipped,
        "latency_ms": latency_ms,
        "r2_symbolic": r2_symbolic,
        "penalty_lat": penalty_lat,
        "fitness_total": fitness_total,
    })
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde continue (meme convention que niveau0_benchmark_iso.py)
# ─────────────────────────────────────────────────────────────────────────────

def _run_signature(epochs, n_windows_train, n_windows_val) -> str:
    """Le device n'en fait plus partie : avant, passer de mps a cpu (ou
    l'inverse) rendait TOUS les runs deja termines invisibles a la reprise."""
    payload = {"epochs": epochs, "n_windows_train": n_windows_train,
               "n_windows_val": n_windows_val}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _key_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # derniere ligne tronquee par un crash pendant l'ecriture :
                # on l'ignore au lieu de bloquer toute la reprise.
                print(f"[avertissement] ligne illisible ignoree dans {path}")
    return rows


def _append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _point_key(point: dict) -> str:
    bg = "|".join(f"{g}={point['base_by_gate'][g]}" for g in sorted(point["base_by_gate"]))
    return f"{point['cell_type']}::{bg}::{point['regime']}::seed{point['seed']}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def select_training_device(prefer: str = None) -> torch.device:
    """Cascade prefer > MPS > CUDA > CPU -- les cellules recurrentes completes
    (hidden_size jusqu'a 170, regression symbolique) sont assez grosses pour
    beneficier reellement de MPS sur Apple Silicon (contrairement au Niveau 0,
    ou les micro-modeles a 12 parametres rendaient MPS plus lent que CPU)."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train_path", type=str,
                         default=os.path.join("..", "..", "MOMTSIM", "config", "featuresLog.parquet"))
    parser.add_argument("--val_path", type=str,
                         default=os.path.join("..", "..", "data", "val_features.parquet"))
    parser.add_argument("--epochs", type=int, default=10,
                         help="Valeur REELLE de l'Etape 2 (n_epochs_eval).")
    parser.add_argument("--n_windows_train", type=int, default=50_000,
                         help="Fenetres d'entrainement tirees (graine fixe 42) -- 50 000 = "
                              "n_windows_eval REEL de l'Etape 2 (compute_fitness), donc "
                              "protocole identique a celui qui a produit theta_opt. 0 = "
                              "toutes les fenetres (~16 M : >10 jours PAR run).")
    parser.add_argument("--n_windows_val", type=int, default=0,
                         help="Fenetres de validation (0 = toutes, evaluees par lots).")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--strategy", type=str, default="raw_first",
                         choices=["raw_first", "full"])
    parser.add_argument("--top_n_engineered", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto",
                         choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--enable_latency_penalty", action="store_true", default=True)
    parser.add_argument("--no_latency_penalty", dest="enable_latency_penalty",
                         action="store_false")
    parser.add_argument("--checkpoint_every_min", type=float, default=10.0,
                         help="Frequence des checkpoints INTRA-run (minutes).")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                         help="Clipping du gradient (1.0 = valeur de l'Etape 2 ; 0 = aucun).")
    parser.add_argument("--out_dir", type=str, default="step4_outputs")
    parser.add_argument("--no_resume", action="store_true",
                         help="Repart de zero : l'ancien journal est ARCHIVE (renomme), "
                              "jamais efface.")
    args = parser.parse_args()

    device = select_training_device(args.device)
    print(f"Device d'entrainement : {device}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_root, args.out_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    model_dir = os.path.join(out_dir, "models")
    scores_dir = os.path.join(out_dir, "val_scores")
    for d in (out_dir, ckpt_dir, model_dir, scores_dir):
        os.makedirs(d, exist_ok=True)
    train_path = os.path.join(project_root, args.train_path)
    val_path = os.path.join(project_root, args.val_path)

    progress_path = os.path.join(out_dir, "step4_progress.jsonl")
    failures_path = os.path.join(out_dir, "step4_failures.jsonl")
    n_train = args.n_windows_train or None
    n_val = args.n_windows_val or None
    run_signature = _run_signature(args.epochs, n_train, n_val)

    if args.no_resume and os.path.exists(progress_path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = progress_path.replace(".jsonl", f"_archive_{stamp}.jsonl")
        os.replace(progress_path, archived)
        print(f"[no_resume] ancien journal archive : {archived}")

    completed = {}
    n_stale = 0
    for row in _load_jsonl(progress_path):
        if row.get("run_signature") != run_signature:
            n_stale += 1
            continue
        completed[row["key"]] = row["result"]
    if completed:
        print(f"[reprise] {len(completed)} run(s) deja termine(s) retrouve(s) "
              f"(signature {run_signature}).")
    if n_stale:
        print(f"[avertissement] {n_stale} ligne(s) de progression avec une signature "
              f"differente ignoree(s).")

    run_meta = {"epochs": args.epochs, "n_seeds": args.n_seeds, "device": device.type,
                "strategy": args.strategy, "n_windows_train": n_train,
                "n_windows_val": n_val, "run_signature": run_signature,
                "top_n_engineered": (args.top_n_engineered
                                     if args.strategy == "raw_first" else None)}

    def _refresh_reports():
        # Rapport + graphiques regeneres apres CHAQUE run : un crash ulterieur
        # ne fait jamais perdre ce qui est deja calcule. Ne doit jamais
        # interrompre le benchmark.
        try:
            from niveau1_report import write_reports
            write_reports(out_dir, run_signature=run_signature, metadata=run_meta,
                          figures=True, export_png=False, quiet=True)
        except Exception as exc:   # noqa: BLE001
            tqdm.write(f"  [avertissement] generation du rapport echouee (ignoree) : {exc}")

    seeds = list(range(1, args.n_seeds + 1))
    archs = build_architecture_grid()

    _data_cache = {}

    def _get_precomputed(regime_niveau1: str, W: int) -> dict:
        key = (regime_niveau1, W)
        if key not in _data_cache:
            data = load_and_window(train_path, val_path, regime_niveau1, W,
                                   n_windows_train=n_train, n_windows_val=n_val)
            np.save(os.path.join(scores_dir, f"y_val_{regime_niveau1}_W{W}.npy"), data["y_val"])
            # Pool d'audit R2_symbolic : 256 fenetres de validation tirees au
            # hasard (graine fixe, identiques pour tous les runs). Les 256
            # PREMIERES fenetres (tri par compte) ne couvriraient que quelques
            # comptes ; a l'Etape 2 elles venaient d'un sous-echantillon aleatoire.
            n_pool = min(256, len(data["X_val"]))
            pool_idx = np.sort(np.random.default_rng(0).choice(len(data["X_val"]), n_pool,
                                                               replace=False))
            data["x_audit_pool"] = torch.from_numpy(data["X_val"][pool_idx])
            _data_cache[key] = data
        return _data_cache[key]

    def _run_point(point: dict) -> dict:
        cell_theta = CELL_THETA_OPT[point["cell_type"]]
        gate_bases, gate_kwargs = _build_gate_bases(point["cell_type"], point["base_by_gate"])
        cfg = RunConfig(
            cell_type=point["cell_type"], regime=point["regime_niveau1"],
            gate_bases=gate_bases, gate_kwargs=gate_kwargs or None,
            hidden_size=cell_theta["hidden_size"], W=cell_theta["W"],
            epochs=args.epochs, lr=cell_theta["lr"], batch_size=cell_theta["batch_size"],
            lam=cell_theta["lam"], mu1=cell_theta["mu1"], mu2=cell_theta["mu2"],
            seed=point["seed"], device=device.type,
            clip_grad_norm=args.clip_grad_norm or None,
        )
        precomputed = _get_precomputed(point["regime_niveau1"], cell_theta["W"])
        h = _key_hash(point["key"])
        ckpt_path = os.path.join(ckpt_dir, f"{h}.pt")
        t0 = time.time()
        metrics = run_config(cfg, train_path, val_path, return_model=True,
                              precomputed=precomputed, checkpoint_path=ckpt_path,
                              checkpoint_every_s=args.checkpoint_every_min * 60.0,
                              return_scores=True)
        scores = metrics.pop("_val_scores")
        model = metrics["_model"]
        # x_pool de R2_symbolic construit sur des fenetres de VALIDATION, comme
        # extended_heuristic_search.compute_fitness -- l'ancien code prenait les
        # fenetres d'ENTRAINEMENT (cf. _get_precomputed).
        metrics["_X_train_pool"] = precomputed["x_audit_pool"]
        # Sauvegardes AVANT le calcul latence/R2_symbolic : si celui-ci plante,
        # le modele entraine n'est pas perdu (le checkpoint de fin d'entrainement
        # permet aussi de reprendre sans reentrainer).
        np.save(os.path.join(scores_dir, f"{h}.npy"), scores.astype(np.float32))
        torch.save(model.state_dict(), os.path.join(model_dir, f"{h}.pt"))
        metrics = compute_fitness_extras(metrics, point["cell_type"], cell_theta["W"],
                                          args.enable_latency_penalty)
        metrics["wall_time_s"] = time.time() - t0
        return {
            "cell_type": point["cell_type"], "regime": point["regime"],
            "base_by_gate": point["base_by_gate"], "seed": point["seed"],
            "key_hash": h, "y_val_file": f"y_val_{point['regime_niveau1']}_W{cell_theta['W']}.npy",
            "metrics": metrics,
        }

    def _run_one(point: dict):
        """Execute un point ; en cas d'echec, journalise et continue (le
        checkpoint intra-run est conserve pour la prochaine relance)."""
        try:
            result = _run_point(point)
        except KeyboardInterrupt:
            tqdm.write("\n[interruption] arret demande -- relancez la meme commande "
                       "pour reprendre (checkpoints conserves).")
            raise
        except Exception as exc:   # noqa: BLE001
            import traceback
            tqdm.write(f"  [ECHEC] {point['key']} : {exc!r} -- run suivant.")
            _append_jsonl(failures_path, {
                "run_signature": run_signature, "key": point["key"],
                "error": repr(exc), "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return None
        finally:
            _release_device_memory(device)
        _append_jsonl(progress_path, {
            "run_signature": run_signature, "key": point["key"], "result": result,
        })
        completed[point["key"]] = result
        ckpt = os.path.join(ckpt_dir, f"{_key_hash(point['key'])}.pt")
        if os.path.exists(ckpt):
            os.remove(ckpt)
        _refresh_reports()
        return result

    def _run_grid(points: list, desc: str) -> list:
        for p in points:
            p["key"] = _point_key(p)
        out = [completed[p["key"]] for p in points if p["key"] in completed]
        remaining = [p for p in points if p["key"] not in completed]
        pbar = tqdm(remaining, desc=desc, unit="run")
        for point in pbar:
            pbar.set_postfix_str(f"{point['cell_type']}/{point['regime']}/seed{point['seed']}")
            result = _run_one(point)
            if result is not None:
                out.append(result)
        return out

    from collections import defaultdict

    def _mean_fitness_by_arch(rows, with_regime):
        acc = defaultdict(list)
        for r in rows:
            arch = tuple(sorted(r["base_by_gate"].items()))
            k = (r["cell_type"], r["regime"], arch) if with_regime else (r["cell_type"], arch)
            acc[k].append(r["metrics"]["fitness_total"])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    if args.strategy == "full":
        results = _run_grid(build_full_grid(seeds),
                            f"Etape 4 -- Niveau 1 ({len(archs) * 2 * len(seeds)} runs)")
    else:
        raw_results = _run_grid(build_regime_grid(archs, "raw", seeds),
                                 f"Etape 4 phase 1/2 -- raw ({len(archs) * len(seeds)} runs)")
        raw_fitness = _mean_fitness_by_arch(raw_results, with_regime=False)
        top_archs_keys = sorted(raw_fitness, key=raw_fitness.get, reverse=True)[:args.top_n_engineered]
        top_archs = [{"cell_type": ct, "base_by_gate": dict(bg)} for ct, bg in top_archs_keys]
        print(f"[raw_first] Top {len(top_archs)} architectures (fitness raw) "
              f"etendues a engineered : "
              + "; ".join(f"{ct}/{dict(bg)}={raw_fitness[(ct, bg)]:.4f}"
                          for ct, bg in top_archs_keys))
        eng_results = _run_grid(build_regime_grid(top_archs, "engineered", seeds),
                                 f"Etape 4 phase 2/2 -- engineered top {len(top_archs)} "
                                 f"({len(top_archs) * len(seeds)} runs)")
        results = raw_results + eng_results

    if not results:
        print("Aucun run reussi -- voir", failures_path)
        return

    # ── Baseline MKAN uniforme (hybrid sur toutes les portes), meme cellule /
    #    regime / graines / theta_opt que la meilleure architecture heterogene.
    arch_fitness = _mean_fitness_by_arch(results, with_regime=True)
    best_cell_type, best_regime, _ = max(arch_fitness, key=arch_fitness.get)
    baseline_gate_names = (["Forget", "Input", "Candidate"] if best_cell_type == "gru"
                            else ["Forget", "Input", "Candidate", "Output"])
    for seed in tqdm(seeds, desc="Baseline MKAN uniforme (hybrid)", unit="seed"):
        b_point = {
            "cell_type": best_cell_type,
            "base_by_gate": {g: "hybrid" for g in baseline_gate_names},
            "regime": best_regime, "regime_niveau1": REGIME_MAP[best_regime],
            "seed": seed,
        }
        b_point["key"] = "baseline::" + _point_key(b_point)
        if b_point["key"] not in completed:
            _run_one(b_point)

    from niveau1_report import write_reports
    paths = write_reports(out_dir, run_signature=run_signature, metadata=run_meta,
                          figures=True, export_png=True)
    for label, p in paths.items():
        print(f"{label} : {p}")



if __name__ == "__main__":
    main()
