"""
niveau1_final_training.py -- Entrainement final des finalistes de l'Etape 4, jusqu'a
convergence (arret anticipe), avec les memes donnees, hyperparametres et graines que
le plan de croisement.

Pourquoi : a 10 epoques, la perte d'entrainement baisse encore de 3 a 9 % par epoque
pour toutes les architectures (step4_level1_results.json, loss_history) ; le
classement de l'Etape 4 compare donc des modeles non convergés.

Finalistes (issus de l'analyse de step4_outputs, cf. niveau1_decision.py) :
    A        TKAN brut  Forget=wavkan_dog Input=efficientkan Candidate=wavkan_dog Output=linear
             (retenue par la regle pre-enregistree du chapitre 4)
    B        TKAN brut  Forget=relukan    Input=efficientkan Candidate=wavkan_dog Output=linear
             (retenue par l'AHP normalise ; elite de la recherche NSGA-II)
    uniforme TKAN brut  hybrid sur les quatre portes (reference de la regle 3)
    gru      GRU brut   reset=wavkan_dog update=efficientkan candidate=wavkan_dog
             (meilleur MCC de l'Etape 4)

Protocole de chaque entrainement (finaliste x graine) :
    - donnees : load_and_window(..., n_windows_train, n_windows_val=None), meme tirage
      que le benchmark (graine 42) ; hyperparametres CELL_THETA_OPT de la cellule ;
      perte mkan_total_loss ; Adam ; ecretage du gradient a 1 ;
    - arret anticipe : apres chaque epoque, perte de prediction (weighted_bce) sur un
      sous-ensemble FIXE de --n_monitor fenetres de validation (default_rng(1), le meme
      pour tous les finalistes) ; on garde les poids de la meilleure epoque et on
      s'arrete apres --patience epoques sans amelioration (ou a --max_epochs) ;
    - evaluation : poids de la meilleure epoque, TOUTE la validation, memes metriques,
      latence, R2_symbolic (seuil 0,99 et --r2_alt) et fitness que le benchmark
      (compute_fitness_extras).
Le jeu de TEST n'est pas utilise ici (regle 4 du chapitre 4 : il ne sert qu'une fois,
a la configuration finalement retenue).

Reprise : etat complet sauvegarde a chaque fin d'epoque (modele, optimiseur, meilleurs
poids, historique, generateurs aleatoires) ; relancer la meme commande reprend a
l'epoque suivante. Ctrl+C perd au plus l'epoque en cours.

Usage :
    python niveau1_final_training.py
    python niveau1_final_training.py --only A B uniforme --seeds 1 2 3
    python niveau1_final_training.py --max_epochs 80 --patience 8 --device mps

Sorties (dans --out_dir, par defaut step4_final/) :
    final_progress.jsonl, final_results.json, models/<id>.pt, val_scores/<id>.npy,
    y_val_<regime>_W<W>.npy, checkpoints/<id>.pt
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

from niveau1_harness import load_and_window, compute_metrics                    # noqa: E402
from niveau1_benchmark_run import (                                              # noqa: E402
    CELL_THETA_OPT, REGIME_MAP, _build_x_pool, compute_fitness_extras, select_training_device,
    _append_jsonl, _load_jsonl,
)
from niveau1_r2_seuils import build_model                                       # noqa: E402
from extended_heuristic_search import _compute_r2_symbolic, _release_device_memory  # noqa: E402
from loss import mkan_total_loss, weighted_bce                                  # noqa: E402

FINALISTS = {
    "A": ("lstm", "raw", {"Forget": "wavkan_dog", "Input": "efficientkan",
                          "Candidate": "wavkan_dog", "Output": "linear"}),
    "B": ("lstm", "raw", {"Forget": "relukan", "Input": "efficientkan",
                          "Candidate": "wavkan_dog", "Output": "linear"}),
    "uniforme": ("lstm", "raw", {"Forget": "hybrid", "Input": "hybrid",
                                 "Candidate": "hybrid", "Output": "hybrid"}),
    "gru": ("gru", "raw", {"Forget": "wavkan_dog", "Input": "efficientkan",
                           "Candidate": "wavkan_dog"}),
}


def _run_id(name: str, seed: int, protocol: dict) -> str:
    payload = json.dumps({"finalist": name, "spec": FINALISTS[name], "seed": seed, **protocol},
                         sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _predict(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            parts.append(model(torch.from_numpy(X[s:s + batch]).to(device)).float().cpu().numpy())
    return np.concatenate(parts)


def _monitor_loss(model, X: np.ndarray, y: np.ndarray, device, batch: int = 4096) -> float:
    """Perte de prediction (weighted_bce, celle de l'entrainement) moyennee par lot."""
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch]).to(device)
            yb = torch.from_numpy(y[s:s + batch]).to(device)
            tot += float(weighted_bce(model(xb), yb)) * len(xb)
            n += len(xb)
    return tot / max(1, n)


def train_one(name: str, seed: int, data: dict, args, device, out_dir: str, run_id: str) -> dict:
    cell, regime, base_by_gate = FINALISTS[name]
    theta = CELL_THETA_OPT[cell]
    ckpt_path = os.path.join(out_dir, "checkpoints", run_id + ".pt")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cell, base_by_gate, data["input_size"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=theta["lr"])
    X_tr, y_tr = data["X_train_t"], data["y_train_t"]
    n = X_tr.shape[0]
    n_batches = math.ceil(n / theta["batch_size"])

    state = dict(epoch=0, best_loss=float("inf"), best_epoch=0, wait=0, best_model=None,
                 train_loss=[], monitor_loss=[], train_time_s=0.0)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        state = ck["state"]
        tqdm.write(f"  [reprise] {name} graine {seed} : epoque {state['epoch']} terminee")

    stopped = state["wait"] >= args.patience
    while not stopped and state["epoch"] < args.max_epochs:
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        loss_sum, n_seen = torch.zeros((), device=device), 0
        for b in tqdm(range(n_batches), desc=f"{name} s{seed} ep{state['epoch'] + 1}",
                      unit="batch", leave=False, mininterval=1.0):
            idx = perm[b * theta["batch_size"]:(b + 1) * theta["batch_size"]]
            xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss, _, _, _ = mkan_total_loss(model, xb, yb, lam=theta["lam"], mu1=theta["mu1"],
                                            mu2=theta["mu2"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            opt.step()
            loss_sum = loss_sum + loss.detach().reshape(())
            n_seen += 1
        train_loss = (loss_sum / max(1, n_seen)).item()
        if not math.isfinite(train_loss):
            raise RuntimeError(f"perte NaN/Inf a l'epoque {state['epoch'] + 1}")
        mon = _monitor_loss(model, data["X_monitor"], data["y_monitor"], device)
        state["epoch"] += 1
        state["train_loss"].append(train_loss)
        state["monitor_loss"].append(mon)
        state["train_time_s"] += time.time() - t0
        if mon < state["best_loss"] - args.min_delta:
            state.update(best_loss=mon, best_epoch=state["epoch"], wait=0,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            state["wait"] += 1
        stopped = state["wait"] >= args.patience
        tqdm.write(f"  {name} s{seed} ep{state['epoch']:3d}  train={train_loss:.4f}  "
                   f"val(suivi)={mon:.4f}  meilleure=ep{state['best_epoch']}  "
                   f"sans amelioration={state['wait']}/{args.patience}")
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                      "state": state}, ckpt_path)

    # ── Evaluation des poids de la meilleure epoque sur toute la validation ──
    model.load_state_dict(state["best_model"])
    scores = _predict(model, data["X_val"], device)
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    metrics = compute_metrics(data["y_val"], scores)
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    x_pool = _build_x_pool(model_cpu, cell, data["x_audit_pool"], theta["W"])
    r2_alt = float(_compute_r2_symbolic(model_cpu, x_pool, r2_threshold=args.r2_alt))
    metrics.update({"n_params": sum(p.numel() for p in model.parameters()),
                    "n_train_windows": int(n), "n_val_windows": int(len(data["X_val"])),
                    "_model": model_cpu, "_input_size": data["input_size"],
                    "_X_train_pool": data["x_audit_pool"]})
    metrics = compute_fitness_extras(metrics, cell, theta["W"], enable_latency_penalty=True)
    metrics.update({f"r2_symbolic_{args.r2_alt:.2f}": r2_alt,
                    "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
                    "stopped_early": state["wait"] >= args.patience,
                    "train_loss_history": state["train_loss"],
                    "monitor_loss_history": state["monitor_loss"],
                    "train_time_s": state["train_time_s"]})
    np.save(os.path.join(out_dir, "val_scores", run_id + ".npy"), scores.astype(np.float32))
    torch.save(state["best_model"], os.path.join(out_dir, "models", run_id + ".pt"))
    return {"finalist": name, "cell_type": cell, "regime": regime, "base_by_gate": base_by_gate,
            "seed": seed, "run_id": run_id,
            "y_val_file": f"y_val_{REGIME_MAP[regime]}_W{theta['W']}.npy", "metrics": metrics}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--only", nargs="+", choices=list(FINALISTS), default=list(FINALISTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min_delta", type=float, default=1e-4,
                   help="Amelioration minimale de la perte de suivi pour remettre la patience a zero.")
    p.add_argument("--n_windows_train", type=int, default=50_000,
                   help="50 000 = meme echantillon que l'Etape 4 (seul l'entrainement s'allonge).")
    p.add_argument("--n_monitor", type=int, default=50_000,
                   help="Fenetres de validation suivies pour l'arret anticipe.")
    p.add_argument("--r2_alt", type=float, default=0.80,
                   help="Seuil R2 supplementaire (analyse de sensibilite ; la fitness reste a 0,99).")
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--train_path", default=os.path.join("..", "..", "MOMTSIM", "config",
                                                        "featuresLog.parquet"))
    p.add_argument("--val_path", default=os.path.join("..", "..", "data", "val_features.parquet"))
    p.add_argument("--out_dir", default="step4_final")
    args = p.parse_args()

    device = select_training_device(args.device)
    out_dir = os.path.join(_ETUDE_DIR, args.out_dir)
    for sub in ("", "models", "val_scores", "checkpoints"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    train_path = os.path.join(_ETUDE_DIR, args.train_path)
    val_path = os.path.join(_ETUDE_DIR, args.val_path)
    protocol = {"max_epochs": args.max_epochs, "patience": args.patience,
                "min_delta": args.min_delta, "n_windows_train": args.n_windows_train,
                "n_monitor": args.n_monitor, "clip_grad_norm": args.clip_grad_norm}
    print(f"Device : {device} ; protocole : {protocol}")

    progress_path = os.path.join(out_dir, "final_progress.jsonl")
    done = {row["run_id"]: row for row in _load_jsonl(progress_path)}
    data_cache = {}
    for name in args.only:
        cell, regime, _ = FINALISTS[name]
        W = CELL_THETA_OPT[cell]["W"]
        key = (REGIME_MAP[regime], W)
        for seed in args.seeds:
            run_id = _run_id(name, seed, protocol)
            if run_id in done:
                continue
            if key not in data_cache:
                d = load_and_window(train_path, val_path, key[0], W,
                                    n_windows_train=args.n_windows_train, n_windows_val=None)
                np.save(os.path.join(out_dir, f"y_val_{key[0]}_W{W}.npy"), d["y_val"])
                n_pool = min(256, len(d["X_val"]))
                pool_idx = np.sort(np.random.default_rng(0).choice(len(d["X_val"]), n_pool,
                                                                   replace=False))
                d["x_audit_pool"] = torch.from_numpy(d["X_val"][pool_idx])
                n_mon = min(args.n_monitor, len(d["X_val"]))
                mon_idx = np.sort(np.random.default_rng(1).choice(len(d["X_val"]), n_mon,
                                                                  replace=False))
                d["X_monitor"], d["y_monitor"] = d["X_val"][mon_idx], d["y_val"][mon_idx]
                data_cache[key] = d
            t0 = time.time()
            try:
                result = train_one(name, seed, data_cache[key], args, device, out_dir, run_id)
            except KeyboardInterrupt:
                print("\n[interruption] relancer la meme commande pour reprendre.")
                raise
            finally:
                _release_device_memory(device)
            result["metrics"]["wall_time_s"] = time.time() - t0
            row = {"run_id": run_id, "protocol": protocol, "result": result,
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            _append_jsonl(progress_path, row)
            done[run_id] = row
            m = result["metrics"]
            print(f"[fini] {name} graine {seed} : {m['epochs_run']} epoques (meilleure "
                  f"{m['best_epoch']}), MCC={m['mcc']:.4f}, R2={m['r2_symbolic']:.3f}, "
                  f"latence={m['latency_ms']:.0f} ms, fitness={m['fitness_total']:.4f}")

    rows = [r["result"] for r in done.values() if r["protocol"] == protocol]
    with open(os.path.join(out_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "finalists": {k: list(v) for k, v in FINALISTS.items()},
                   "results": rows}, f, indent=1)
    print(f"final_results.json : {len(rows)} entrainement(s).")


if __name__ == "__main__":
    main()
