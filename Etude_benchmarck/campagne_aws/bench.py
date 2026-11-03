"""
bench.py -- Banc d'essai de l'instance (environ 30 a 60 min) : fixe, par la
mesure et non par intuition, la taille cachee (budget de parametres commun
TKAN/GRU), le nombre de processus x threads, et le nombre de fenetres
d'entrainement qui tient dans le budget.

Mesures (donnees aleatoires de meme forme que les vraies : le debit d'un pas
d'entrainement ne depend pas des valeurs) :
  1. debit d'entrainement (fenetres/s) d'UN processus, pour chaque architecture
     type x taille cachee x nombre de threads ;
  2. debit TOTAL quand P processus tournent en meme temps (P x T = coeurs) :
     c'est lui qui compte pour la campagne ;
  3. latence de chaque architecture type x taille, machine au repos, avec la
     mesure de l'Etape 4 (lot de 256, meilleur de 3, MKAN_LATENCY_THREADS threads).
Puis estimation de la duree de l'Etape 4 et de l'entrainement final pour
plusieurs nombres de fenetres. Resultats : sorties/bench/bench_results.json.

Usage :
    python bench.py                       # tout
    python bench.py --hidden 32 48 64 --quick
"""

import argparse
import json
import os
import subprocess
import sys
import time

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

INPUT_SIZE = 12
ARCHS = {
    "TKAN_A": ("lstm", {"Forget": "wavkan_dog", "Input": "efficientkan",
                        "Candidate": "wavkan_dog", "Output": "linear"}),
    "TKAN_relu": ("lstm", {"Forget": "relukan", "Input": "relukan",
                           "Candidate": "wavkan_dog", "Output": "wavkan_morlet"}),
    "TKAN_uniforme": ("lstm", {g: "hybrid" for g in ("Forget", "Input", "Candidate", "Output")}),
    "GRU_gru": ("gru", {"Forget": "wavkan_dog", "Input": "efficientkan", "Candidate": "wavkan_dog"}),
}


def _spec(arch: str, hidden_tkan: int) -> dict:
    from gru_cell import match_gru_hidden_size
    cell, bbg = ARCHS[arch]
    theta = config.CELL_THETA[cell]
    h = hidden_tkan if cell == "lstm" else match_gru_hidden_size(INPUT_SIZE, hidden_tkan)
    return {"cell_type": cell, "base_by_gate": bbg, "hidden_size": h, **theta}


def worker(arch: str, hidden: int, threads: int, n_batches: int, warmup: int) -> dict:
    """Mesure dans un processus isole (lance par subprocess)."""
    import torch
    torch.set_num_threads(threads)
    from train_run import build_model
    from loss import mkan_total_loss
    spec = _spec(arch, hidden)
    torch.manual_seed(0)
    model = build_model(spec, INPUT_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    fwd = torch.compile(model, dynamic=False) if os.environ.get("MKAN_COMPILE") == "1" else model
    bs, W = spec["batch_size"], spec["W"]
    xs = torch.randn(n_batches + warmup, bs, W, INPUT_SIZE).clamp(-4, 4)
    ys = (torch.rand(n_batches + warmup, bs) < 0.2).float()
    t0 = None
    for b in range(n_batches + warmup):
        if b == warmup:
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, _, _, _ = mkan_total_loss(fwd, xs[b], ys[b], lam=spec["lam"], mu1=spec["mu1"],
                                        mu2=spec["mu2"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    dt = time.perf_counter() - t0
    return {"arch": arch, "hidden_tkan": hidden, "hidden": spec["hidden_size"], "threads": threads,
            "windows_per_s": n_batches * bs / dt,
            "n_params": sum(p.numel() for p in model.parameters())}


def _run_workers(arch, hidden, threads, n_proc, n_batches, warmup) -> list:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cmd = [sys.executable, __file__, "--worker", arch, str(hidden), str(threads),
           str(n_batches), str(warmup)]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
             for _ in range(n_proc)]
    out = []
    for p in procs:
        stdout, _ = p.communicate()
        out.append(json.loads(stdout.strip().splitlines()[-1]))
    return out


def latency(arch: str, hidden: int, threads: int) -> float:
    import torch
    torch.set_num_threads(threads)
    from train_run import build_model
    from extended_heuristic_search import _measure_latency_ms
    spec = _spec(arch, hidden)
    model = build_model(spec, INPUT_SIZE)
    return _measure_latency_ms(model, spec["W"], INPUT_SIZE)


def estimate(results: dict, n_cores: int) -> list:
    """Duree estimee de l'Etape 4 (180 runs brut + 15 traite + 5 reference) et de
    l'entrainement final (4 finalistes x 5 graines), pour chaque taille et nombre
    de fenetres, a partir du debit total mesure en concurrence."""
    rows = []
    for hidden, conc in results["concurrency"].items():
        rate = {}
        for arch, v in conc.items():
            rate[arch] = v["total_windows_per_s"] / v["n_proc"]   # debit par processus en charge
        n_proc = next(iter(conc.values()))["n_proc"]
        tkan = (rate.get("TKAN_A", 0) + rate.get("TKAN_relu", 0) + rate.get("TKAN_uniforme", 0)) / 3
        gru = rate.get("GRU_gru", tkan)
        for n_train in (200_000, 300_000, 500_000):
            for epochs in (8, 15):
                runs = [("tkan", 120 + 10 + 5), ("gru", 60 + 5)]
                proc_hours = sum(n * n_train * epochs / (tkan if c == "tkan" else gru)
                                 for c, n in runs) / 3600
                wall = proc_hours / n_proc
                rows.append({"hidden_tkan": int(hidden), "n_train": n_train, "epochs_moy": epochs,
                             "n_proc": n_proc, "etape4_heures": round(wall, 1)})
        for n_final in (1_000_000, 2_000_000):
            for epochs in (15, 30):
                proc_hours = 20 * n_final * epochs / ((tkan * 3 + gru) / 4) / 3600
                rows.append({"hidden_tkan": int(hidden), "n_train_final": n_final,
                             "epochs_moy": epochs, "n_proc": min(n_proc, 20),
                             "final_heures": round(proc_hours / min(n_proc, 20), 1)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--worker", nargs=5, metavar=("ARCH", "H", "T", "NB", "WARM"))
    ap.add_argument("--hidden", nargs="+", type=int, default=[32, 48, 64])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--quick", action="store_true", help="moins de lots par mesure")
    ap.add_argument("--combos", nargs="+", metavar="PxT",
                    help="mesure seulement le debit total de ces repartitions (ex. 32x1 32x2 16x4), "
                         "en completant bench_results.json")
    ap.add_argument("--combo_batches", type=int, default=40)
    args = ap.parse_args()
    if args.worker:
        a, h, t, nb, w = args.worker
        print(json.dumps(worker(a, int(h), int(t), int(nb), int(w))))
        return

    n_cores = os.cpu_count() or 1
    nb, warm = (40, 5) if args.quick else (150, 15)
    out_dir = os.path.join(config.OUT_DIR, "bench")
    os.makedirs(out_dir, exist_ok=True)
    res = {"n_cores": n_cores, "single": [], "concurrency": {}, "latency": [],
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = os.path.join(out_dir, "bench_results.json")

    def save():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    if args.combos:
        # Repartitions processus x threads mesurees directement (le meilleur
        # debit par coeur en processus seul ne predit pas le debit en charge :
        # 64 vCPU = 32 coeurs physiques, bande passante memoire partagee).
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                res = json.load(f)
        res.setdefault("combos", {})
        for h in args.hidden:
            for arch in ARCHS:
                for combo in args.combos:
                    p, t = (int(v) for v in combo.split("x"))
                    key = f"{h}|{arch}|{combo}"
                    if key in res["combos"]:
                        continue
                    rs = _run_workers(arch, h, t, p, args.combo_batches, 5)
                    tot = sum(r["windows_per_s"] for r in rs)
                    res["combos"][key] = {"hidden_tkan": h, "arch": arch, "n_proc": p, "threads": t,
                                          "total_windows_per_s": tot}
                    print(f"  {arch:14s} h={h:3d} {combo:>5s} : {tot:8.0f} fen/s au total "
                          f"({tot / p:6.1f} par processus)", flush=True)
                    save()
        print(f"\nResultats : {path}")
        return

    print("1/3 latence (machine au repos)", flush=True)
    lat_threads = int(os.environ.get("MKAN_LATENCY_THREADS", "4"))
    for h in args.hidden:
        for arch in ARCHS:
            r = subprocess.run([sys.executable, "-c",
                                f"import sys; sys.path.insert(0, {_CAMPAGNE_DIR!r}); import bench; "
                                f"print(bench.latency({arch!r}, {h}, {lat_threads}))"],
                               capture_output=True, text=True)
            lat = float(r.stdout.strip().splitlines()[-1])
            res["latency"].append({"arch": arch, "hidden_tkan": h, "latency_ms": lat,
                                   "threads": lat_threads})
            print(f"  {arch:14s} h={h:3d} : {lat:7.1f} ms", flush=True)
            save()

    print("2/3 debit d'un processus seul", flush=True)
    for h in args.hidden:
        for arch in ARCHS:
            for t in args.threads:
                r = _run_workers(arch, h, t, 1, nb, warm)[0]
                res["single"].append(r)
                print(f"  {arch:14s} h={h:3d} T={t} : {r['windows_per_s']:7.0f} fen/s "
                      f"({r['n_params']} param.)", flush=True)
                save()

    print("3/3 debit total en concurrence", flush=True)
    for h in args.hidden:
        best_t = {}
        for arch in ARCHS:
            # threads retenus : meilleur debit PAR COEUR en processus seul
            cand = [r for r in res["single"] if r["arch"] == arch and r["hidden_tkan"] == h]
            best = max(cand, key=lambda r: r["windows_per_s"] / r["threads"])
            t = best["threads"]
            n_proc = max(1, n_cores // t)
            rs = _run_workers(arch, h, t, n_proc, nb, warm)
            tot = sum(r["windows_per_s"] for r in rs)
            best_t[arch] = {"threads": t, "n_proc": n_proc, "total_windows_per_s": tot}
            print(f"  {arch:14s} h={h:3d} : {n_proc} x {t} threads -> {tot:8.0f} fen/s au total",
                  flush=True)
        res["concurrency"][str(h)] = best_t
        save()

    res["estimations"] = estimate(res, n_cores)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()
    print("\nEstimations (heures de calcul, machine entiere) :")
    for r in res["estimations"]:
        print("  " + json.dumps(r, ensure_ascii=False))
    print(f"\nResultats : {path}")


if __name__ == "__main__":
    main()
