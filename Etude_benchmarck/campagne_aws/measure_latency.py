"""
measure_latency.py -- Mesure la latence de tous les modeles finis qui n'en ont pas
encore, machine AU REPOS (scheduler.py l'appelle entre deux phases, quand aucun
entraînement ne tourne), puis calcule leur fitness.

Mesure identique a l'Etape 4 (extended_heuristic_search._measure_latency_ms) :
passage avant sur un lot de 256 fenetres sur CPU, meilleur de 3 apres
echauffement. Nombre de threads fixe pour TOUS les modeles (--threads,
enregistre dans latency.json) : les latences ne se comparent qu'entre modeles
mesures sur la meme machine avec le meme reglage.

Fitness (formule de l'Etape 4, poids AHP inchanges) avec la penalite de latence
plafonnee et le seuil de 500 ms decides le 03/11 :
    F = 0,4147 max(0, MCC) + 0,2573 PR-AUC - 0,1529 Brier + 0,0876 R2_symb
        - 0,0876 min(1, max(0, L - T) / T),   T = 500 ms
R2_symb : seuil de travail principal 0,80 (decision du 27/10, config.R2_ALT).
"""

import argparse
import glob
import json
import os
import platform
import sys
import time

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import torch  # noqa: E402


def fitness(metrics: dict, latency_ms, r2_key: str = f"r2_symbolic_{config.R2_ALT:.2f}") -> dict:
    """Fitness de l'Etape 4 ; latency_ms=None donne la fitness SANS terme de
    latence (affichee comme provisoire tant que la mesure n'est pas faite)."""
    from extended_heuristic_search import (
        _penalty_latency, AHP_WEIGHT_MCC, AHP_WEIGHT_PR_AUC, AHP_WEIGHT_BRIER,
        AHP_WEIGHT_R2_SYMBOLIC, AHP_WEIGHT_LATENCY_PENALTY)

    def f(x, default):
        try:
            x = float(x)
            return x if x == x and abs(x) != float("inf") else default
        except (TypeError, ValueError):
            return default

    base = (AHP_WEIGHT_MCC * max(0.0, f(metrics.get("mcc"), 0.0))
            + AHP_WEIGHT_PR_AUC * f(metrics.get("auc_pr"), 0.0)
            - AHP_WEIGHT_BRIER * f(metrics.get("brier"), 1.0)
            + AHP_WEIGHT_R2_SYMBOLIC * f(metrics.get(r2_key), 0.0))
    if latency_ms is None:
        return {"fitness_sans_latence": base, "fitness_total": None, "penalty_lat": None}
    pen = _penalty_latency(latency_ms, threshold_ms=config.LATENCY_THRESHOLD_MS,
                           capped=config.LATENCY_CAPPED)
    return {"fitness_sans_latence": base, "penalty_lat": pen,
            "fitness_total": base - AHP_WEIGHT_LATENCY_PENALTY * pen}


def measure_pending(out_dir: str, threads: int, force: bool = False) -> int:
    from extended_heuristic_search import _measure_latency_ms
    from train_run import build_model
    torch.set_num_threads(threads)
    n = 0
    for done in sorted(glob.glob(os.path.join(out_dir, "runs", "*", "DONE"))):
        run_dir = os.path.dirname(done)
        lat_path = os.path.join(run_dir, "latency.json")
        if os.path.exists(lat_path) and not force:
            continue
        with open(os.path.join(run_dir, "spec.json"), encoding="utf-8") as f:
            if json.load(f).get("baseline"):
                from train_baseline import mesure_latence     # meme lot de 256 fenetres
                lat = mesure_latence(run_dir)
                with open(lat_path + ".tmp", "w", encoding="utf-8") as g:
                    json.dump({"latency_ms": lat, "threads": threads, "batch": 256, "repeats": 3,
                               "threshold_ms": config.LATENCY_THRESHOLD_MS,
                               "machine": platform.processor() or platform.machine(),
                               "measured": time.strftime("%Y-%m-%dT%H:%M:%S")}, g)
                os.replace(lat_path + ".tmp", lat_path)
                n += 1
                continue
        ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
        spec = ck["spec"]
        model = build_model(spec, ck["input_size"])
        model.load_state_dict(ck["state_dict"])
        lat = _measure_latency_ms(model, spec["W"], ck["input_size"])
        with open(lat_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"latency_ms": lat, "threads": threads, "batch": 256, "repeats": 3,
                       "threshold_ms": config.LATENCY_THRESHOLD_MS,
                       "machine": platform.processor() or platform.machine(),
                       "measured": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        os.replace(lat_path + ".tmp", lat_path)
        n += 1
        print(f"[latence] {spec['label']} : {lat:.1f} ms", flush=True)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out_dir", default=config.OUT_DIR)
    p.add_argument("--threads", type=int, default=int(os.environ.get("MKAN_LATENCY_THREADS", "4")))
    p.add_argument("--force", action="store_true", help="remesure aussi les modeles deja mesures")
    args = p.parse_args()
    print(f"{measure_pending(args.out_dir, args.threads, args.force)} modele(s) mesure(s).")


if __name__ == "__main__":
    main()
