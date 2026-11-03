"""
report.py -- Rassemble les runs finis (result.json + latency.json) en un tableau
et un classement, regeneres apres chaque run : les resultats restent
exploitables a tout moment, meme en plein calcul.

Sorties (dans OUT_DIR) :
    runs.csv      une ligne par run fini (metriques, latence, fitness)
    summary.json  classement par architecture (moyenne et ecart-type sur les graines),
                  relu par le suivi web et par scheduler.py (Top 3 de la phase 2)
Tant qu'une latence manque, la fitness affichee est « sans latence » et marquee
provisoire.
"""

import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402
from measure_latency import fitness  # noqa: E402

COLUMNS = ["run_id", "phase", "label", "cell_type", "regime", "architecture", "seed",
           "hidden_size", "n_params", "mcc", "auc_pr", "auc_roc", "brier", "f1_best",
           "r2_symbolic", f"r2_symbolic_{config.R2_ALT:.2f}", "latency_ms", "penalty_lat",
           "fitness_sans_latence", "fitness_total", "epochs_run", "best_epoch",
           "train_time_s", "n_train_windows"]


def arch_string(spec: dict) -> str:
    return "/".join(f"{g}={spec['base_by_gate'][g]}" for g in sorted(spec["base_by_gate"]))


def collect(out_dir: str) -> list:
    rows = []
    for done in sorted(glob.glob(os.path.join(out_dir, "runs", "*", "DONE"))):
        run_dir = os.path.dirname(done)
        try:
            with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
                res = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        lat = None
        lat_path = os.path.join(run_dir, "latency.json")
        if os.path.exists(lat_path):
            with open(lat_path, encoding="utf-8") as f:
                lat = json.load(f)["latency_ms"]
        spec, m = res["spec"], res["metrics"]
        row = {"run_id": spec["run_id"], "phase": spec["phase"], "label": spec["label"],
               "cell_type": spec["cell_type"], "regime": spec["regime"],
               "architecture": arch_string(spec), "seed": spec["seed"],
               "hidden_size": spec["hidden_size"], "latency_ms": lat}
        for k in COLUMNS:
            if k in m:
                row[k] = m[k]
        row.update(fitness(m, lat))
        rows.append(row)
    return rows


def rank(rows: list, phases=None) -> list:
    groups = defaultdict(list)
    for r in rows:
        if phases and r["phase"] not in phases:
            continue
        groups[(r["phase"], r["cell_type"], r["regime"], r["architecture"])].append(r)
    table = []
    for (phase, cell, regime, arch), rs in groups.items():
        complete = all(r["fitness_total"] is not None for r in rs)
        key = "fitness_total" if complete else "fitness_sans_latence"
        vals = [r[key] for r in rs]

        def stat(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (None, None)
        table.append({"phase": phase, "cell_type": cell, "regime": regime, "architecture": arch,
                      "n_seeds": len(rs), "fitness": statistics.mean(vals),
                      "fitness_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                      "fitness_provisoire": not complete,
                      "mcc": stat("mcc"), "auc_pr": stat("auc_pr"), "brier": stat("brier"),
                      "r2_symbolic": stat("r2_symbolic"),
                      "r2_alt": stat(f"r2_symbolic_{config.R2_ALT:.2f}"),
                      "latency_ms": stat("latency_ms"), "n_params": stat("n_params"),
                      "base_by_gate": dict(x.split("=") for x in arch.split("/"))})
    table.sort(key=lambda t: -t["fitness"])
    return table


def write_reports(out_dir: str = config.OUT_DIR) -> dict:
    rows = collect(out_dir)
    tmp = os.path.join(out_dir, "runs.csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, os.path.join(out_dir, "runs.csv"))
    summary = {"updated": datetime.now(timezone.utc).isoformat(), "n_runs_done": len(rows),
               "latency_threshold_ms": config.LATENCY_THRESHOLD_MS,
               "ranking": rank(rows)}
    tmp = os.path.join(out_dir, "summary.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(out_dir, "summary.json"))
    return summary


if __name__ == "__main__":
    s = write_reports()
    print(f"{s['n_runs_done']} run(s) ; tete du classement :")
    for t in s["ranking"][:10]:
        flag = " (provisoire)" if t["fitness_provisoire"] else ""
        print(f"  {t['fitness']:.4f}{flag}  {t['phase']} {t['cell_type']}/{t['regime']}  {t['architecture']}")
