"""
train_run.py -- Entraine UN modele (une architecture, une graine) de la campagne,
dans son propre processus, et peut reprendre exactement ou il s'est arrete.

Lance par scheduler.py (un processus par run, nombre de threads fixe) ou a la main :
    python train_run.py --spec sorties/runs/<run_id>/spec.json --threads 2

Protocole (identique a l'Etape 4 sauf decisions du 03/11 consignees dans
DECISIONS.md) : perte mkan_total_loss, Adam, ecretage du gradient a 1, lot 32,
lr 1e-4 ; grille des bases recalee sur les donnees ; arret anticipe sur la perte
de prediction (weighted_bce) d'un sous-ensemble FIXE de fenetres de validation,
poids de la meilleure epoque conserves ; evaluation sur TOUTE la validation.
La latence n'est PAS mesuree ici (processus concurrents sur la machine) :
measure_latency.py la mesure ensuite, machine au repos.

Fichiers du run (sorties/runs/<run_id>/) :
    spec.json        specification (ecrite par scheduler.py)
    status.json      etat en direct (progression, pertes) relu par le suivi web
    checkpoint.pt    point de reprise (toutes les CHECKPOINT_EVERY_S et a chaque epoque)
    model.pt         poids de la meilleure epoque + grille + pretraitement
    val_scores.npy   scores sur toute la validation
    edge_profiles.npz  profils d'activation des aretes (mesure de derive ulterieure)
    result.json      metriques (sans latence)
    DONE             marque de fin
Codes de sortie : 0 fini, 3 interrompu proprement (fichier STOP), 1 echec.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402  (ajoute Etude_benchmarck et MKAN au chemin)

import numpy as np  # noqa: E402
import torch  # noqa: E402

EXIT_DONE, EXIT_FAILED, EXIT_INTERRUPTED = 0, 1, 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_model(spec: dict, input_size: int) -> torch.nn.Module:
    """Meme construction que l'Etape 4 (niveau1_benchmark_run._build_gate_bases),
    taille cachee prise dans la spec."""
    from niveau1_benchmark_run import _build_gate_bases
    from lstm_cell import ConfigurableMKANScorer
    from gru_cell import GRUMKANScorer
    gate_bases, gate_kwargs = _build_gate_bases(spec["cell_type"], spec["base_by_gate"])
    cls = ConfigurableMKANScorer if spec["cell_type"] == "lstm" else GRUMKANScorer
    return cls(input_size, spec["hidden_size"], gate_bases=gate_bases,
               gate_kwargs=gate_kwargs or None)


def load_dataset(path: str) -> dict:
    d = {name: np.load(os.path.join(path, name + ".npy"), mmap_mode="r")
         for name in ("X_train", "y_train", "X_val", "y_val", "monitor_idx", "audit_idx",
                      "grid_samples")}
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        d["meta"] = json.load(f)
    with open(os.path.join(path, "preprocessing.json"), encoding="utf-8") as f:
        d["preprocessing"] = json.load(f)
    return d


class StatusWriter:
    """Ecrit status.json au plus toutes les STATUS_EVERY_S secondes (ou de force)."""

    def __init__(self, path: str, base: dict):
        self.path, self.base, self.last = path, base, 0.0

    def __call__(self, force: bool = False, **fields) -> None:
        self.base.update(fields)
        now = time.time()
        if force or now - self.last >= config.STATUS_EVERY_S:
            self.base["updated"] = _now()
            _write_json(self.path, self.base)
            self.last = now


def _gate_layers(model):
    for attr in ("forget_gate", "input_gate", "candidate_gate", "output_gate",
                 "reset_gate", "update_gate"):
        layer = getattr(model.cell, attr, None)
        if layer is not None:
            yield attr, layer


@torch.no_grad()
def edge_profiles(model, input_size: int) -> dict:
    """Profils d'activation des aretes, sans reentrainement possible plus tard :
    phi_ij evaluee sur une grille de N_PROFILE_POINTS valeurs de chaque entree
    ([-1, 1] pour les colonnes h, [-4, 4] pour les colonnes x), plus la matrice
    L1 exacte (eq. 2.19) sur le pool d'audit. Une arete phi_ij ne depend que de
    x_i, donc une seule passe par porte suffit."""
    out = {}
    for name, layer in _gate_layers(model):
        n_in = layer.in_features
        n_h = n_in - input_size
        g = torch.linspace(-1.0, 1.0, config.N_PROFILE_POINTS)
        grid = torch.empty(config.N_PROFILE_POINTS, n_in)
        grid[:, :n_h] = g[:, None]
        grid[:, n_h:] = 4.0 * g[:, None]
        out[f"{name}_grid"] = grid.numpy().astype(np.float32)
        out[f"{name}_phi"] = layer.edge_activations(grid).numpy().astype(np.float16)
    return out


@torch.no_grad()
def _l1_on_pool(model, cell_type: str, x_pool: torch.Tensor, W: int) -> dict:
    from niveau1_benchmark_run import _build_x_pool
    concat = _build_x_pool(model, cell_type, x_pool, W)
    return {f"{name}_l1": layer.exact_l1_norm(concat).numpy().astype(np.float32)
            for name, layer in _gate_layers(model)}


def train(spec: dict, run_dir: str, threads: int) -> int:
    torch.set_num_threads(threads)
    from loss import mkan_total_loss, weighted_bce
    from edges import fit_grids_to_data

    status = StatusWriter(os.path.join(run_dir, "status.json"), {
        "run_id": spec["run_id"], "phase": spec["phase"], "label": spec["label"],
        "state": "chargement", "pid": os.getpid(), "threads": threads, "started": _now(),
        "epoch": 0, "max_epochs": spec["max_epochs"], "batch": 0, "n_batches": 0,
        "train_loss": [], "monitor_loss": [], "best_epoch": 0, "wait": 0,
        "patience": spec["patience"]})
    status(force=True)

    data = load_dataset(spec["data_dir"])
    input_size = data["meta"]["input_size"]
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_mon = np.ascontiguousarray(data["X_val"][data["monitor_idx"]])
    y_mon = np.ascontiguousarray(data["y_val"][data["monitor_idx"]])
    n, bs = X_tr.shape[0], spec["batch_size"]
    n_batches = math.ceil(n / bs)

    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    model = build_model(spec, input_size)
    if spec["grid_init"] == "data":
        fit_grids_to_data(model, torch.from_numpy(np.array(data["grid_samples"])))
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    gen = torch.Generator().manual_seed(spec["seed"])

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    st = dict(epoch=0, batch=0, perm=None, loss_sum=0.0, n_seen=0, best_loss=float("inf"),
              best_epoch=0, wait=0, best_model=None, train_loss=[], monitor_loss=[],
              train_time_s=0.0)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])     # grille comprise (buffers)
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        st = ck["state"]
        print(f"[reprise] {spec['label']} : epoque {st['epoch'] + 1}, lot {st['batch']}", flush=True)

    t_session = time.time()
    time_before = st["train_time_s"]

    def save_checkpoint():
        st["train_time_s"] = time_before + (time.time() - t_session)
        _atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "generator": gen.get_state(), "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(), "state": st, "spec": spec},
                           ckpt_path)

    stop_requested = {"flag": False}

    def _on_signal(signum, frame):   # SIGTERM (arret de l'instance) : sauvegarde propre
        stop_requested["flag"] = True
    signal.signal(signal.SIGTERM, _on_signal)

    def must_stop() -> bool:
        return stop_requested["flag"] or os.path.exists(config.STOP_FILE)

    status(state="entrainement", n_batches=n_batches, train_loss=st["train_loss"],
           monitor_loss=st["monitor_loss"], best_epoch=st["best_epoch"], wait=st["wait"],
           epoch=st["epoch"], force=True)
    last_ck = time.time()
    stopped = st["wait"] >= spec["patience"]
    while not stopped and st["epoch"] < spec["max_epochs"]:
        model.train()
        if st["perm"] is None:
            st["perm"] = torch.randperm(n, generator=gen)
            st["batch"], st["loss_sum"], st["n_seen"] = 0, 0.0, 0
        perm = st["perm"]
        loss_acc = torch.zeros(())
        n_acc = 0
        t_ep, seen_ep = time.time(), 0
        for b in range(st["batch"], n_batches):
            idx = np.sort(perm[b * bs:(b + 1) * bs].numpy())
            xb = torch.from_numpy(np.asarray(X_tr[idx]))
            yb = torch.from_numpy(np.asarray(y_tr[idx]))
            opt.zero_grad(set_to_none=True)
            loss, _, _, _ = mkan_total_loss(model, xb, yb, lam=spec["lam"], mu1=spec["mu1"],
                                            mu2=spec["mu2"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec["clip_grad_norm"])
            opt.step()
            loss_acc += loss.detach().reshape(())
            n_acc += 1
            seen_ep += len(idx)
            st["batch"] = b + 1
            if n_acc % 200 == 0 or b + 1 == n_batches:
                st["loss_sum"] += float(loss_acc)
                st["n_seen"] += n_acc
                loss_acc.zero_()
                n_acc = 0
                if not math.isfinite(st["loss_sum"]):
                    raise RuntimeError(f"perte NaN/Inf, epoque {st['epoch'] + 1}, lot {b}")
                rate = seen_ep / max(1e-6, time.time() - t_ep)
                status(batch=st["batch"], running_loss=st["loss_sum"] / max(1, st["n_seen"]),
                       windows_per_s=rate, eta_epoch_s=(n - st["batch"] * bs) / max(rate, 1e-6))
                if must_stop():
                    save_checkpoint()
                    status(state="interrompu", force=True)
                    return EXIT_INTERRUPTED
                if time.time() - last_ck >= config.CHECKPOINT_EVERY_S:
                    save_checkpoint()
                    last_ck = time.time()
        # Ordre des operations de fin d'epoque : la perte d'entrainement est la
        # moyenne des pertes de lot (comme l'Etape 4).
        train_loss = st["loss_sum"] / max(1, st["n_seen"])
        model.eval()
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for s in range(0, len(X_mon), 4096):
                xb = torch.from_numpy(X_mon[s:s + 4096])
                yb = torch.from_numpy(y_mon[s:s + 4096])
                tot += float(weighted_bce(model(xb), yb)) * len(xb)
                cnt += len(xb)
        mon = tot / max(1, cnt)
        st["epoch"] += 1
        st["train_loss"].append(train_loss)
        st["monitor_loss"].append(mon)
        if mon < st["best_loss"] - spec["min_delta"]:
            st.update(best_loss=mon, best_epoch=st["epoch"], wait=0,
                      best_model={k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            st["wait"] += 1
        st["perm"] = None
        stopped = st["wait"] >= spec["patience"]
        save_checkpoint()
        last_ck = time.time()
        status(epoch=st["epoch"], batch=0, train_loss=st["train_loss"],
               monitor_loss=st["monitor_loss"], best_epoch=st["best_epoch"], wait=st["wait"],
               force=True)
        print(f"  {spec['label']} ep{st['epoch']:3d} train={train_loss:.4f} suivi={mon:.4f} "
              f"meilleure=ep{st['best_epoch']} attente={st['wait']}/{spec['patience']}", flush=True)
        if must_stop():
            status(state="interrompu", force=True)
            return EXIT_INTERRUPTED

    # ── Evaluation des poids de la meilleure epoque sur toute la validation ──
    status(state="evaluation", force=True)
    from niveau1_harness import compute_metrics
    from niveau1_benchmark_run import _build_x_pool
    from extended_heuristic_search import _compute_r2_symbolic
    model.load_state_dict(st["best_model"])
    model.eval()
    X_val, y_val = data["X_val"], np.asarray(data["y_val"])
    parts = []
    with torch.no_grad():
        for s in range(0, len(X_val), 4096):
            parts.append(model(torch.from_numpy(np.asarray(X_val[s:s + 4096]))).numpy())
    scores = np.concatenate(parts)
    n_bad = int((~np.isfinite(scores)).sum())
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    metrics = compute_metrics(y_val, scores)
    x_audit = torch.from_numpy(np.ascontiguousarray(X_val[data["audit_idx"]]))
    x_pool = _build_x_pool(model, spec["cell_type"], x_audit, spec["W"])
    metrics["r2_symbolic"] = float(_compute_r2_symbolic(model, x_pool,
                                                        r2_threshold=config.R2_THRESHOLD))
    metrics[f"r2_symbolic_{config.R2_ALT:.2f}"] = float(
        _compute_r2_symbolic(model, x_pool, r2_threshold=config.R2_ALT))
    metrics.update({"n_params": sum(p.numel() for p in model.parameters()),
                    "n_bad_scores": n_bad, "n_train_windows": int(n),
                    "n_val_windows": int(len(X_val)), "epochs_run": st["epoch"],
                    "best_epoch": st["best_epoch"],
                    "stopped_early": st["wait"] >= spec["patience"],
                    "train_loss_history": st["train_loss"],
                    "monitor_loss_history": st["monitor_loss"],
                    "train_time_s": time_before + (time.time() - t_session)})

    np.save(os.path.join(run_dir, "val_scores.npy"), scores.astype(np.float32))
    np.savez_compressed(os.path.join(run_dir, "edge_profiles.npz"),
                        **edge_profiles(model, input_size),
                        **_l1_on_pool(model, spec["cell_type"], x_audit, spec["W"]))
    torch.save({"state_dict": st["best_model"], "spec": spec, "input_size": input_size,
                "preprocessing": data["preprocessing"], "data_meta": data["meta"]},
               os.path.join(run_dir, "model.pt"))
    _write_json(os.path.join(run_dir, "result.json"),
                {"run_id": spec["run_id"], "spec": spec, "metrics": metrics,
                 "finished": _now()})
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write(_now())
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    status(state="fini", mcc=metrics.get("mcc"), auc_pr=metrics.get("auc_pr"), force=True)
    print(f"[fini] {spec['label']} : MCC={metrics.get('mcc', float('nan')):.4f} "
          f"R2={metrics['r2_symbolic']:.3f} epoques={st['epoch']}", flush=True)
    return EXIT_DONE


def main():
    p = argparse.ArgumentParser(description="Entraine un run de la campagne AWS")
    p.add_argument("--spec", required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    run_dir = os.path.dirname(os.path.abspath(args.spec))
    try:
        code = train(spec, run_dir, args.threads)
    except Exception as exc:   # noqa: BLE001
        _write_json(os.path.join(run_dir, "error.json"),
                    {"error": repr(exc), "traceback": traceback.format_exc(), "time": _now()})
        try:
            with open(os.path.join(run_dir, "status.json"), encoding="utf-8") as f:
                s = json.load(f)
            s.update(state="echec", error=repr(exc), updated=_now())
            _write_json(os.path.join(run_dir, "status.json"), s)
        except Exception:   # noqa: BLE001
            pass
        traceback.print_exc()
        code = EXIT_FAILED
    sys.exit(code)


if __name__ == "__main__":
    main()
