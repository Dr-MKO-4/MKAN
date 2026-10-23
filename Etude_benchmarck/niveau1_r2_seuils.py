"""
niveau1_r2_seuils.py -- Recalcule R2_symbolic de chaque modele de l'Etape 4 a plusieurs
seuils d'ajustement (analyse de sensibilite), SANS reentrainement : les poids sont
relus dans step4_outputs/models/*.pt.

Le calcul est celui du benchmark, a l'identique :
    - meme fonction extended_heuristic_search._compute_r2_symbolic (seul r2_threshold change) ;
    - meme pool d'audit : 256 fenetres de VALIDATION tirees avec default_rng(0) parmi les
      fenetres de load_and_window(..., n_windows_train=50 000, n_windows_val=None),
      x_pool = concat(h_t final, dernier pas d'entree) via _build_x_pool.
Controle integre : au seuil 0,99 (celui du benchmark), la valeur recalculee doit egaler
r2_symbolic du JSON ; tout ecart est signale.

La regle de selection pre-enregistree reste au seuil 0,99 ; les autres seuils servent
d'analyse de sensibilite (niveau1_decision.py lit le fichier produit ici).

Usage :
    python niveau1_r2_seuils.py
    python niveau1_r2_seuils.py --thresholds 0.99 0.80 0.90
    python niveau1_r2_seuils.py --device cpu

Sorties (dans --out_dir) :
    r2_seuils_progress.jsonl   (1 ligne par modele, reprise automatique)
    r2_seuils.json             ({key_hash: {"r2_json", "r2": {seuil: valeur}, "fitness": {seuil: valeur}}})
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_ETUDE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ETUDE_DIR not in sys.path:
    sys.path.insert(0, _ETUDE_DIR)

from niveau1_harness import load_and_window                                    # noqa: E402
from niveau1_benchmark_run import (                                            # noqa: E402
    CELL_THETA_OPT, REGIME_MAP, _build_gate_bases, _build_x_pool, _append_jsonl, _load_jsonl,
)
from extended_heuristic_search import _compute_r2_symbolic, AHP_WEIGHT_R2_SYMBOLIC  # noqa: E402
from lstm_cell import ConfigurableMKANScorer                                   # noqa: E402
from gru_cell import GRUMKANScorer, match_gru_hidden_size                      # noqa: E402


def build_model(cell_type: str, base_by_gate: dict, input_size: int) -> torch.nn.Module:
    """Meme construction que niveau1_harness.run_config (memes bases, kwargs et tailles)."""
    theta = CELL_THETA_OPT[cell_type]
    gate_bases, gate_kwargs = _build_gate_bases(cell_type, base_by_gate)
    if cell_type == "lstm":
        return ConfigurableMKANScorer(input_size, theta["hidden_size"],
                                      gate_bases=gate_bases, gate_kwargs=gate_kwargs or None)
    h_gru = match_gru_hidden_size(input_size, theta["hidden_size"])
    return GRUMKANScorer(input_size, h_gru, gate_bases=gate_bases, gate_kwargs=gate_kwargs or None)


def audit_pool(train_path: str, val_path: str, regime_niveau1: str, W: int,
               n_windows_train: int) -> dict:
    """Reproduit _get_precomputed de niveau1_benchmark_run : memes fenetres, meme tirage."""
    data = load_and_window(train_path, val_path, regime_niveau1, W,
                           n_windows_train=n_windows_train, n_windows_val=None)
    n_pool = min(256, len(data["X_val"]))
    idx = np.sort(np.random.default_rng(0).choice(len(data["X_val"]), n_pool, replace=False))
    return {"x_audit_pool": torch.from_numpy(data["X_val"][idx]), "input_size": data["input_size"]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out_dir", default="step4_outputs")
    p.add_argument("--train_path", default=os.path.join("..", "..", "MOMTSIM", "config",
                                                        "featuresLog.parquet"))
    p.add_argument("--val_path", default=os.path.join("..", "..", "data", "val_features.parquet"))
    p.add_argument("--n_windows_train", type=int, default=50_000,
                   help="Doit etre celui du benchmark (metadata de step4_level1_results.json).")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.80])
    args = p.parse_args()

    out_dir = os.path.join(_ETUDE_DIR, args.out_dir)
    train_path = os.path.join(_ETUDE_DIR, args.train_path)
    val_path = os.path.join(_ETUDE_DIR, args.val_path)
    with open(os.path.join(out_dir, "step4_level1_results.json"), encoding="utf-8") as f:
        d = json.load(f)
    if d["metadata"].get("n_windows_train") != args.n_windows_train:
        print(f"[avertissement] n_windows_train={args.n_windows_train} differe du benchmark "
              f"({d['metadata'].get('n_windows_train')}) : le pool d'audit ne sera pas identique.")
    runs = d["results"] + d["baseline_results"]

    progress_path = os.path.join(out_dir, "r2_seuils_progress.jsonl")
    done = {row["key_hash"]: row for row in _load_jsonl(progress_path)
            if sorted(row["r2"]) == sorted(f"{t:.2f}" for t in args.thresholds)}
    todo = [r for r in runs if r["key_hash"] not in done]
    print(f"{len(done)} modele(s) deja traite(s), {len(todo)} a traiter.")

    pools = {}
    torch.set_grad_enabled(True)   # _compute_r2_symbolic ajuste par Adam
    for r in tqdm(sorted(todo, key=lambda r: (r["cell_type"], r["regime"])), desc="modeles"):
        cell, theta = r["cell_type"], CELL_THETA_OPT[r["cell_type"]]
        key = (REGIME_MAP[r["regime"]], theta["W"])
        if key not in pools:
            pools[key] = audit_pool(train_path, val_path, key[0], key[1], args.n_windows_train)
        pool = pools[key]
        model = build_model(cell, r["base_by_gate"], pool["input_size"])
        state = torch.load(os.path.join(out_dir, "models", r["key_hash"] + ".pt"),
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        x_pool = _build_x_pool(model, cell, pool["x_audit_pool"], theta["W"])
        r2 = {f"{t:.2f}": float(_compute_r2_symbolic(model, x_pool, r2_threshold=t))
              for t in args.thresholds}
        m = r["metrics"]
        row = {"key_hash": r["key_hash"], "r2_json": m["r2_symbolic"], "r2": r2,
               "fitness": {k: m["fitness_total"] + AHP_WEIGHT_R2_SYMBOLIC * (v - m["r2_symbolic"])
                           for k, v in r2.items()}}
        if "0.99" in r2 and abs(r2["0.99"] - m["r2_symbolic"]) > 1e-9:
            tqdm.write(f"  [ecart] {r['key_hash']} : R2 recalcule a 0,99 = {r2['0.99']:.4f}, "
                       f"JSON = {m['r2_symbolic']:.4f}")
        _append_jsonl(progress_path, row)
        done[r["key_hash"]] = row

    n_bad = sum(1 for row in done.values()
                if "0.99" in row["r2"] and abs(row["r2"]["0.99"] - row["r2_json"]) > 1e-9)
    out = {k: {kk: v[kk] for kk in ("r2_json", "r2", "fitness")} for k, v in done.items()}
    with open(os.path.join(out_dir, "r2_seuils.json"), "w", encoding="utf-8") as f:
        json.dump({"thresholds": [f"{t:.2f}" for t in args.thresholds], "models": out}, f, indent=1)
    print(f"r2_seuils.json ecrit ({len(out)} modeles). Controle au seuil 0,99 : "
          f"{len(out) - n_bad} identiques au JSON, {n_bad} ecart(s).")


if __name__ == "__main__":
    main()
