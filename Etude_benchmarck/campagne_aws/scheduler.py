"""
scheduler.py -- Orchestration de la campagne : repartit les runs sur tous les
coeurs de l'instance (un processus train_run.py par run, nombre de threads
fixe), reprend tout ce qui n'est pas fini, et enchaine les phases.

Nouvelle Etape 4 (meme plan que l'Etape 4 du 21/10, strategie raw_first) :
    1. brut      : 36 architectures heterogenes x 5 graines
    2. latence   : mesure de tous les modeles, machine au repos
    3. traite    : Top 3 des architectures (fitness moyenne du brut) x 5 graines
    4. latence
    5. reference : MKAN uniforme (hybrid sur toutes les portes) avec la cellule et
                   le regime de la meilleure architecture, 5 graines
    6. latence, rapport final
Entrainement final : --mode final --finalists finalistes.json (liste de
{"label", "cell_type", "regime", "base_by_gate"}), memes regles, jeu plus grand.

Reprise : relancer la meme commande. Les runs finis (DONE) sont sautes, les
autres repartent de leur point de reprise. Le fichier STOP (cree par
sentinel.py sur preavis spot ou budget atteint) fait sauvegarder et quitter
chaque run, et empeche d'en lancer de nouveaux.

Usage :
    python scheduler.py --workers 32 --threads 2
    python scheduler.py --mode final --finalists finalistes.json --workers 20 --threads 3
    python scheduler.py --workers 64 --threads auto   # threads par run calcules pour chaque phase

--threads auto : coeurs disponibles // nombre total de runs de la phase, entre 1 et
MAX_THREADS_PAR_RUN. Calcule sur la phase entiere (pas sur les runs restants) : un run garde
le meme nombre de threads a chaque reprise.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

GATE_SHORT = {"Forget": "F", "Input": "I", "Candidate": "C", "Output": "O"}
BASE_SHORT = {"wavkan_dog": "dog", "wavkan_morlet": "morlet", "efficientkan": "bspl",
              "relukan": "relu", "linear": "lin", "hybrid": "hyb"}


MAX_THREADS_PAR_RUN = 4      # le banc d'essai (bench_c8g.log) ne mesure le debit que jusqu'a 4 threads
COEURS_RESERVES = 2          # sentinelle et suivi


def threads_par_run(n_runs: int, coeurs: int) -> int:
    return max(1, min(MAX_THREADS_PAR_RUN, (coeurs - COEURS_RESERVES) // max(1, n_runs)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def make_spec(phase: str, cell_type: str, regime: str, base_by_gate: dict, seed: int,
              protocol: dict, data_dir: str, input_size: int) -> dict:
    from gru_cell import match_gru_hidden_size
    theta = config.CELL_THETA[cell_type]
    hidden = (config.HIDDEN_TKAN if cell_type == "lstm"
              else match_gru_hidden_size(input_size, config.HIDDEN_TKAN))
    arch = " ".join(f"{GATE_SHORT[g]}={BASE_SHORT.get(b, b)}" for g, b in base_by_gate.items())
    spec = {"phase": phase, "cell_type": cell_type, "regime": regime,
            "base_by_gate": base_by_gate, "seed": seed, "hidden_size": hidden,
            "W": theta["W"], "lr": theta["lr"], "batch_size": theta["batch_size"],
            "lam": theta["lam"], "mu1": theta["mu1"], "mu2": theta["mu2"],
            "clip_grad_norm": config.CLIP_GRAD_NORM, "grid_init": config.GRID_INIT,
            "max_epochs": protocol["max_epochs"], "patience": protocol["patience"],
            "min_delta": protocol["min_delta"], "data_dir": data_dir,
            "n_windows_train": protocol["n_windows_train"]}
    key = {k: v for k, v in spec.items() if k != "data_dir"}
    spec["run_id"] = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    spec["label"] = f"{'TKAN' if cell_type == 'lstm' else 'GRU'} {regime} {arch} s{seed}"
    return spec


def _cost_estimate(spec: dict) -> float:
    """Ordre de lancement : les runs les plus longs d'abord (aretes x pas de temps)."""
    n_gates = 4 if spec["cell_type"] == "lstm" else 3
    h = spec["hidden_size"]
    return n_gates * h * (h + 12) * spec["W"]


class Pool:
    def __init__(self, workers: int, threads: int, out_dir: str, mode: str):
        self.workers, self.threads, self.out_dir, self.mode = workers, threads, out_dir, mode
        self.auto = threads == "auto"
        self.running = {}        # run_id -> (Popen, spec, t0)
        self.failed, self.done = {}, set()
        self.phase, self.total = "", 0
        self.started = _now()

    def run_dir(self, spec):
        return os.path.join(self.out_dir, "runs", spec["run_id"])

    def is_done(self, spec):
        return os.path.exists(os.path.join(self.run_dir(spec), "DONE"))

    def save_state(self, queue):
        _write_json(os.path.join(self.out_dir, "scheduler_state.json"), {
            "mode": self.mode, "phase": self.phase, "workers": self.workers,
            "threads": self.threads, "total": self.total, "n_done": len(self.done),
            "running": [s["run_id"] for _, s, _ in self.running.values()],
            "queued": [s["run_id"] for s in queue], "failed": self.failed,
            "stop_requested": os.path.exists(config.STOP_FILE),
            "started": self.started, "updated": _now(), "pid": os.getpid()})

    def _launch(self, spec):
        rd = self.run_dir(spec)
        os.makedirs(rd, exist_ok=True)
        spec_path = os.path.join(rd, "spec.json")
        if not os.path.exists(spec_path):
            _write_json(spec_path, spec)
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads),
                   MKL_NUM_THREADS=str(self.threads), MKAN_EVAL_THREADS=str(self.eval_threads))
        log = open(os.path.join(rd, "log.txt"), "a", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, os.path.join(_CAMPAGNE_DIR, "train_run.py"),
                                 "--spec", spec_path, "--threads", str(self.threads)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        self.running[spec["run_id"]] = (proc, spec, time.time())

    def run_phase(self, phase: str, specs: list) -> list:
        """Execute tous les specs ; retourne les specs finis. Bloque jusqu'a la fin
        de la phase (ou jusqu'a l'arret demande)."""
        from report import write_reports
        self.phase = phase
        self.total = len(specs)
        if self.auto:
            self.threads = threads_par_run(len(specs), os.cpu_count() or 2)
        self.done = {s["run_id"] for s in specs if self.is_done(s)}
        queue = sorted([s for s in specs if s["run_id"] not in self.done],
                       key=_cost_estimate, reverse=True)
        attempts = {}
        print(f"[{phase}] {len(specs)} runs, {len(self.done)} deja finis, "
              f"{len(queue)} a faire, {self.workers} processus x {self.threads} threads", flush=True)
        while queue or self.running:
            stop = os.path.exists(config.STOP_FILE)
            while queue and not stop and len(self.running) < self.workers:
                self.eval_threads = self.threads
                if self.auto:       # coeurs libres en fin de phase, pour l'evaluation seulement
                    self.eval_threads = max(self.threads, ((os.cpu_count() or 2) - COEURS_RESERVES)
                                            // (len(queue) + len(self.running)))
                self._launch(queue.pop(0))
            for rid, (proc, spec, t0) in list(self.running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del self.running[rid]
                if code == 0 and self.is_done(spec):
                    self.done.add(rid)
                    try:
                        write_reports(self.out_dir)
                    except Exception as exc:   # noqa: BLE001
                        print(f"  [avertissement] rapport : {exc!r}", flush=True)
                elif code == 3 or stop:
                    pass      # interrompu proprement : repris a la prochaine relance
                else:
                    attempts[rid] = attempts.get(rid, 0) + 1
                    if attempts[rid] < 2:
                        print(f"  [echec] {spec['label']} (code {code}) : nouvelle tentative",
                              flush=True)
                        queue.append(spec)
                    else:
                        self.failed[rid] = spec["label"]
                        print(f"  [echec definitif] {spec['label']}", flush=True)
            self.save_state(queue)
            if stop and not self.running:
                print("[arret] fichier STOP : tous les runs ont sauvegarde leur etat.", flush=True)
                sys.exit(3)
            time.sleep(2.0)
        self.save_state([])
        return [s for s in specs if self.is_done(s)]


def latency_pass(out_dir: str):
    """Mesure isolee : appelee seulement quand aucun entrainement ne tourne."""
    from report import write_reports
    subprocess.run([sys.executable, os.path.join(_CAMPAGNE_DIR, "measure_latency.py"),
                    "--out_dir", out_dir], check=False)
    return write_reports(out_dir)


def _dataset(regime: str, W: int, n_train):
    from prepare_data import dataset_dir, dataset_ready, prepare
    path = dataset_dir(regime, W, n_train)
    if not dataset_ready(path):
        prepare([regime], [W], n_train)
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        return path, json.load(f)["input_size"]


def specs_for(phase, archs, regime, seeds, protocol):
    out = []
    for a in archs:
        W = config.CELL_THETA[a["cell_type"]]["W"]
        path, input_size = _dataset(regime, W, protocol["n_windows_train"] or None)
        for seed in seeds:
            out.append(make_spec(phase, a["cell_type"], regime, a["base_by_gate"], seed,
                                 protocol, path, input_size))
    return out


def run_etape4(pool: Pool):
    from niveau1_benchmark_run import build_architecture_grid
    p = config.ETAPE4
    seeds = list(range(1, p["n_seeds"] + 1))
    archs = build_architecture_grid()
    pool.run_phase("etape4_brut", specs_for("etape4_brut", archs, "brut", seeds, p))
    summary = latency_pass(pool.out_dir)
    brut = [t for t in summary["ranking"] if t["phase"] == "etape4_brut"]
    top = [{"cell_type": t["cell_type"], "base_by_gate": t["base_by_gate"]}
           for t in brut[:p["top_n_engineered"]]]
    print("[raw_first] Top etendu au regime traite : "
          + " ; ".join(f"{t['cell_type']} {t['architecture']} ({t['fitness']:.4f})"
                       for t in brut[:p["top_n_engineered"]]), flush=True)
    pool.run_phase("etape4_traite", specs_for("etape4_traite", top, "traite", seeds, p))
    summary = latency_pass(pool.out_dir)
    best = summary["ranking"][0]
    gates = (["Forget", "Input", "Candidate", "Output"] if best["cell_type"] == "lstm"
             else ["Forget", "Input", "Candidate"])
    ref = [{"cell_type": best["cell_type"], "base_by_gate": {g: "hybrid" for g in gates}}]
    pool.run_phase("etape4_reference",
                   specs_for("etape4_reference", ref, best["regime"], seeds, p))
    latency_pass(pool.out_dir)
    print("[etape4] terminee : voir summary.json et runs.csv", flush=True)


def run_final(pool: Pool, finalists_path: str):
    with open(finalists_path, encoding="utf-8") as f:
        finalists = json.load(f)
    p = config.FINAL
    specs = []
    for fin in finalists:
        specs += specs_for("final", [fin], fin["regime"], p["seeds"], p)
    pool.run_phase("final", specs)
    latency_pass(pool.out_dir)
    print("[final] termine", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Orchestration de la campagne AWS")
    ap.add_argument("--mode", choices=["etape4", "final"], default="etape4")
    ap.add_argument("--finalists", help="JSON des finalistes (mode final)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threads", type=lambda v: v if v == "auto" else int(v), default=2,
                    help="threads par run, ou auto (calcules pour chaque phase)")
    ap.add_argument("--out_dir", default=config.OUT_DIR)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "runs"), exist_ok=True)
    pool = Pool(args.workers, args.threads, args.out_dir, args.mode)
    if args.mode == "etape4":
        run_etape4(pool)
    else:
        if not args.finalists:
            ap.error("--finalists est requis en mode final")
        run_final(pool, args.finalists)


if __name__ == "__main__":
    main()
