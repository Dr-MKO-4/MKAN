"""
train_baseline.py -- Entraine UNE baseline classique (XGBoost, MLP ou LSTM) sur les memes fenetres que
les modeles KAN, avec les memes fichiers de run que train_run.py (spec, status, result, DONE).

    XGBoost : agregats de fenetre de MKAN/baselines.make_xgb_features, arret sur les fenetres de suivi.
    MLP     : fenetre aplatie, couches cachees CACHE_MLP, Adam, meme perte ponderee.
    LSTM    : MKAN/baselines.LSTMBaseline, deux couches de taille h (celle de la TKAN).
Lot, taux d'apprentissage, epoques, patience et fenetres de suivi : ceux de l'entrainement final.
Reprise : point de reprise des poids a chaque epoque et toutes les CHECKPOINT_EVERY_S (MLP, LSTM) ;
XGBoost repart du debut s'il est interrompu (quelques minutes).
"""

import argparse
import json
import math
import os
import pickle
import signal
import sys
import time
import traceback

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_run import EXIT_DONE, EXIT_FAILED, EXIT_INTERRUPTED, StatusWriter, _atomic_torch_save, _now, \
    _write_json, load_dataset  # noqa: E402

CACHE_MLP = (256, 128)


class MLPBaseline(torch.nn.Module):
    def __init__(self, n_entree: int):
        super().__init__()
        couches, n = [torch.nn.Flatten()], n_entree
        for c in CACHE_MLP:
            couches += [torch.nn.Linear(n, c), torch.nn.ReLU()]
            n = c
        self.reseau = torch.nn.Sequential(*couches, torch.nn.Linear(n, 1))

    def forward(self, x):
        return torch.sigmoid(self.reseau(x)).squeeze(-1)


def construire(spec: dict, input_size: int) -> torch.nn.Module:
    if spec["baseline"] == "mlp":
        return MLPBaseline(spec["W"] * input_size)
    from baselines import LSTMBaseline
    return LSTMBaseline(input_size, hidden_size=spec["hidden_size"])


def _finir(spec, run_dir, status, scores, y_val, extra, t0):
    from niveau1_harness import compute_metrics
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    metrics = compute_metrics(y_val, scores)
    metrics.update({"n_val_windows": int(len(y_val)), "train_time_s": time.time() - t0, **extra})
    np.save(os.path.join(run_dir, "val_scores.npy"), scores.astype(np.float32))
    _write_json(os.path.join(run_dir, "result.json"),
                {"run_id": spec["run_id"], "spec": spec, "metrics": metrics, "finished": _now()})
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write(_now())
    ck = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(ck):
        os.remove(ck)
    status(state="fini", mcc=metrics.get("mcc"), auc_pr=metrics.get("auc_pr"), force=True)
    print(f"[fini] {spec['label']} : MCC={metrics.get('mcc', float('nan')):.4f}", flush=True)
    return EXIT_DONE


def entrainer_xgboost(spec, run_dir, data, status, t0, threads):
    import xgboost as xgb
    from baselines import make_xgb_features
    status(state="entrainement", force=True)
    y = np.asarray(data["y_train"])
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    mod = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
                            eval_metric="logloss", tree_method="hist", early_stopping_rounds=30,
                            random_state=spec["seed"], n_jobs=threads, verbosity=0)
    X_mon = np.ascontiguousarray(data["X_val"][data["monitor_idx"]])
    y_mon = np.ascontiguousarray(data["y_val"][data["monitor_idx"]])
    mod.fit(make_xgb_features(np.asarray(data["X_train"])), y,
            eval_set=[(make_xgb_features(X_mon), y_mon)], verbose=False)
    status(state="evaluation", force=True)
    parts = [mod.predict_proba(make_xgb_features(np.asarray(data["X_val"][s:s + 65536])))[:, 1]
             for s in range(0, len(data["X_val"]), 65536)]
    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump(mod, f)
    return _finir(spec, run_dir, status, np.concatenate(parts), np.asarray(data["y_val"]),
                  {"n_arbres": int(mod.best_iteration + 1), "n_train_windows": int(len(y))}, t0)


def entrainer_reseau(spec, run_dir, data, status, t0):
    from loss import weighted_bce
    input_size = data["meta"]["input_size"]
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_mon = np.ascontiguousarray(data["X_val"][data["monitor_idx"]])
    y_mon = np.ascontiguousarray(data["y_val"][data["monitor_idx"]])
    n, bs = X_tr.shape[0], spec["batch_size"]
    n_batches = math.ceil(n / bs)
    torch.manual_seed(spec["seed"])
    model = construire(spec, input_size)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    gen = torch.Generator().manual_seed(spec["seed"])
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    st = dict(epoch=0, batch=0, perm=None, best_loss=float("inf"), best_epoch=0, wait=0, best_model=None,
              train_loss=[], monitor_loss=[], loss_sum=0.0, n_seen=0)
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["state"]

    def sauver():
        _atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "generator": gen.get_state(), "torch_rng": torch.get_rng_state(), "state": st},
                           ckpt)

    arret = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: arret.update(flag=True))
    stop = lambda: arret["flag"] or os.path.exists(config.STOP_FILE)    # noqa: E731
    status(state="entrainement", n_batches=n_batches, epoch=st["epoch"], force=True)
    dernier = time.time()
    while st["wait"] < spec["patience"] and st["epoch"] < spec["max_epochs"]:
        model.train()
        if st["perm"] is None:
            st.update(perm=torch.randperm(n, generator=gen), batch=0, loss_sum=0.0, n_seen=0)
        for b in range(st["batch"], n_batches):
            idx = np.sort(st["perm"][b * bs:(b + 1) * bs].numpy())
            xb, yb = torch.from_numpy(np.asarray(X_tr[idx])), torch.from_numpy(np.asarray(y_tr[idx]))
            opt.zero_grad(set_to_none=True)
            perte = weighted_bce(model(xb), yb)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec["clip_grad_norm"])
            opt.step()
            st["loss_sum"] += float(perte)
            st["n_seen"] += 1
            st["batch"] = b + 1
            if b % 200 == 0:
                status(batch=st["batch"], running_loss=st["loss_sum"] / st["n_seen"])
                if stop():
                    sauver()
                    status(state="interrompu", force=True)
                    return EXIT_INTERRUPTED
                if time.time() - dernier >= config.CHECKPOINT_EVERY_S:
                    sauver()
                    dernier = time.time()
        model.eval()
        with torch.no_grad():
            tot = sum(float(weighted_bce(model(torch.from_numpy(X_mon[s:s + 4096])),
                                         torch.from_numpy(y_mon[s:s + 4096]))) * len(X_mon[s:s + 4096])
                      for s in range(0, len(X_mon), 4096))
        mon = tot / len(X_mon)
        st["epoch"] += 1
        st["train_loss"].append(st["loss_sum"] / max(1, st["n_seen"]))
        st["monitor_loss"].append(mon)
        if mon < st["best_loss"] - spec["min_delta"]:
            st.update(best_loss=mon, best_epoch=st["epoch"], wait=0,
                      best_model={k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            st["wait"] += 1
        st["perm"] = None
        sauver()
        dernier = time.time()
        status(epoch=st["epoch"], batch=0, train_loss=st["train_loss"], monitor_loss=st["monitor_loss"],
               best_epoch=st["best_epoch"], wait=st["wait"], force=True)
        if stop():
            status(state="interrompu", force=True)
            return EXIT_INTERRUPTED

    status(state="evaluation", force=True)
    model.load_state_dict(st["best_model"])
    model.eval()
    X_val = data["X_val"]
    with torch.no_grad():
        scores = np.concatenate([model(torch.from_numpy(np.asarray(X_val[s:s + 4096]))).numpy()
                                 for s in range(0, len(X_val), 4096)])
    torch.save({"state_dict": st["best_model"], "spec": spec, "input_size": input_size,
                "preprocessing": data["preprocessing"]}, os.path.join(run_dir, "model.pt"))
    return _finir(spec, run_dir, status, scores, np.asarray(data["y_val"]),
                  {"n_params": sum(p.numel() for p in model.parameters()), "epochs_run": st["epoch"],
                   "best_epoch": st["best_epoch"], "n_train_windows": int(n),
                   "train_loss_history": st["train_loss"], "monitor_loss_history": st["monitor_loss"]}, t0)


def mesure_latence(run_dir: str, batch: int = 256, repetitions: int = 3) -> float:
    """Meilleur de 3 passages sur un lot de 256 fenetres, apres echauffement (ms)."""
    with open(os.path.join(run_dir, "spec.json"), encoding="utf-8") as f:
        spec = json.load(f)
    with open(os.path.join(spec["data_dir"], "meta.json"), encoding="utf-8") as f:
        input_size = json.load(f)["input_size"]
    x = torch.rand(batch, spec["W"], input_size)
    if spec["baseline"] == "xgboost":
        from baselines import make_xgb_features
        with open(os.path.join(run_dir, "model.pkl"), "rb") as f:
            mod = pickle.load(f)
        arr = x.numpy()
        appel = lambda: mod.predict_proba(make_xgb_features(arr))    # noqa: E731
    else:
        ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
        model = construire(spec, input_size)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        def appel():
            with torch.no_grad():
                model(x)
    appel()
    mesures = []
    for _ in range(repetitions):
        t = time.perf_counter()
        appel()
        mesures.append(1000.0 * (time.perf_counter() - t))
    return min(mesures)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--spec", required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    run_dir = os.path.dirname(os.path.abspath(a.spec))
    status = StatusWriter(os.path.join(run_dir, "status.json"), {
        "run_id": spec["run_id"], "phase": spec["phase"], "label": spec["label"], "state": "chargement",
        "pid": os.getpid(), "threads": a.threads, "started": _now(), "epoch": 0,
        "max_epochs": spec["max_epochs"], "batch": 0, "n_batches": 0, "train_loss": [], "monitor_loss": [],
        "best_epoch": 0, "wait": 0, "patience": spec["patience"]})
    status(force=True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    try:
        data = load_dataset(spec["data_dir"])
        if spec["baseline"] == "xgboost":
            code = entrainer_xgboost(spec, run_dir, data, status, t0, a.threads)
        else:
            code = entrainer_reseau(spec, run_dir, data, status, t0)
    except Exception as exc:   # noqa: BLE001
        _write_json(os.path.join(run_dir, "error.json"),
                    {"error": repr(exc), "traceback": traceback.format_exc(), "time": _now()})
        status(state="echec", error=repr(exc), force=True)
        traceback.print_exc()
        code = EXIT_FAILED
    sys.exit(code)


if __name__ == "__main__":
    main()
