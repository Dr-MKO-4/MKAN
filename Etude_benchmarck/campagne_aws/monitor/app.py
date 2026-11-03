"""
monitor/app.py -- Suivi web en direct de la campagne (lecture seule).

Sert une page (index.html) et un point d'acces /api/state qui relit les fichiers
ecrits par scheduler.py, train_run.py, report.py et sentinel.py, plus la charge
de l'instance (psutil). Ecoute sur 127.0.0.1 uniquement : on l'ouvre depuis son
poste par un tunnel SSH, sans ouvrir de port sur Internet :
    ssh -i cle.pem -N -L 8050:127.0.0.1:8050 ubuntu@<ip-de-l-instance>
puis http://localhost:8050

Usage (sur l'instance) :
    python monitor/app.py            # ou : uvicorn ... (launch.sh)
"""

import glob
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

_MONITOR_DIR = os.path.dirname(os.path.abspath(__file__))
_CAMPAGNE_DIR = os.path.dirname(_MONITOR_DIR)
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import psutil  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

app = FastAPI(title="Suivi campagne MKAN")
psutil.cpu_percent(percpu=True)   # amorce : la premiere mesure psutil vaut toujours 0


def _read(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _age_s(iso: str):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def _system() -> dict:
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(config.OUT_DIR if os.path.exists(config.OUT_DIR) else "/")
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    return {"cpu_per_core": psutil.cpu_percent(percpu=True), "cpu_total": psutil.cpu_percent(),
            "n_cores": psutil.cpu_count(), "load_avg": load,
            "mem_used_gb": round((vm.total - vm.available) / 2**30, 1),
            "mem_total_gb": round(vm.total / 2**30, 1), "mem_percent": vm.percent,
            "disk_used_gb": round(du.used / 2**30, 1), "disk_total_gb": round(du.total / 2**30, 1),
            "n_train_procs": sum(1 for p in psutil.process_iter(["cmdline"])
                                 if p.info["cmdline"] and "train_run.py" in " ".join(p.info["cmdline"]))}


def _runs(sched: dict) -> list:
    running = set((sched or {}).get("running", []))
    out = []
    for path in glob.glob(os.path.join(config.OUT_DIR, "runs", "*", "status.json")):
        s = _read(path)
        if not s:
            continue
        age = _age_s(s.get("updated"))
        if s.get("state") in ("entrainement", "chargement", "evaluation") and s["run_id"] not in running:
            s["state"] = "en attente de reprise"
        s["stale"] = age is not None and age > 120 and s["run_id"] in running
        s["age_s"] = age
        out.append(s)
    order = {"entrainement": 0, "evaluation": 1, "chargement": 2, "en attente de reprise": 3,
             "interrompu": 4, "echec": 5, "fini": 6}
    out.sort(key=lambda s: (order.get(s.get("state"), 9), s.get("label", "")))
    return out


def _eta(sched: dict, runs: list):
    """Temps restant de la phase : duree moyenne des runs finis de la phase,
    multipliee par le travail restant, divisee par le nombre de processus."""
    if not sched or not sched.get("total"):
        return None
    durations = []
    for r in runs:
        if r.get("state") == "fini" and r.get("phase") == sched.get("phase"):
            res = _read(os.path.join(config.OUT_DIR, "runs", r["run_id"], "result.json"))
            if res:
                durations.append(res["metrics"].get("train_time_s", 0))
    if not durations:
        return None
    remaining = sched["total"] - sched.get("n_done", 0)
    return statistics.mean(durations) * remaining / max(1, sched.get("workers", 1))


@app.get("/api/state")
def state():
    sched = _read(os.path.join(config.OUT_DIR, "scheduler_state.json"), {})
    runs = _runs(sched)
    summary = _read(os.path.join(config.OUT_DIR, "summary.json"), {})
    return JSONResponse({
        "now": datetime.now(timezone.utc).isoformat(),
        "scheduler": sched, "scheduler_age_s": _age_s(sched.get("updated")),
        "runs": runs, "eta_phase_s": _eta(sched, runs),
        "ranking": (summary or {}).get("ranking", [])[:40],
        "n_runs_done": (summary or {}).get("n_runs_done", 0),
        "cost": _read(os.path.join(config.OUT_DIR, "cost_state.json"), {}),
        "sentinel": _read(os.path.join(config.OUT_DIR, "sentinel_state.json"), {}),
        "system": _system(), "budget_usd": config.BUDGET_USD,
        "latency_threshold_ms": config.LATENCY_THRESHOLD_MS,
        "stop": os.path.exists(config.STOP_FILE)})


@app.get("/")
def index():
    return FileResponse(os.path.join(_MONITOR_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MKAN_MONITOR_PORT", "8050")),
                log_level="warning")
