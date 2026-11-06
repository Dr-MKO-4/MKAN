"""
derive.py -- Experience S3 : Grid Extension et derive, sur les finalistes, apres l'entrainement final.

    D_ref   : fenetres de validation hors suivi (decision du 03/11 : le test reste intact)
    D_derive: jeu derive genere avec MoMTSim, coupe en D_adapt (50 % premiers pas, 10 % finaux de ces pas
              pour l'arret anticipe) et D_eval (50 % suivants)
    Bras    : fige | affinage | extension + affinage, 5 graines (les modeles finaux de chaque graine)

Detection (memoire 4.8) : pour chaque variable de x, histogrammes a 20 classes sur [-4, 4] du dernier pas ;
derive si JS(P_ref || P_cur) > tau = 0,05 ; region = enveloppe des classes ou p_cur / p_ref > kappa = 2.
Affinage : lr final / 10, memes epoques et patience. Mesures : MCC, AUC-PR, rappel du scenario derive sur
D_eval (adaptation) et D_ref (oubli), au seuil 0,5.

Usage :
    python derive.py preparer --drift_path derive.parquet --data_dir donnees/<jeu>   # une fois par jeu
    python derive.py lancer --run_dir sorties/runs/<final> --bras extension --out_dir sorties/derive/...
"""

import argparse
import json
import os
import signal
import sys

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

N_CLASSES, TAU, KAPPA, DELTA, BORNE = 20, 0.05, 2.0, 4, 4.0
EXIT_DONE, EXIT_INTERRUPTED = 0, 3


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a[a > 0] * np.log(a[a > 0] / b[a > 0])).sum())    # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def regions_de_derive(x_ref: np.ndarray, x_cur: np.ndarray, colonnes: list) -> list:
    """x_ref, x_cur : (n, F), dernier pas des fenetres. Retourne les variables en derive et leur region."""
    bornes = np.linspace(-BORNE, BORNE, N_CLASSES + 1)
    sortie = []
    for k, nom in enumerate(colonnes):
        r = np.histogram(x_ref[:, k], bornes)[0].astype(np.float64)
        c = np.histogram(x_cur[:, k], bornes)[0].astype(np.float64)
        r, c = r / max(r.sum(), 1.0), c / max(c.sum(), 1.0)
        d = js(r, c)
        ligne = {"variable": nom, "indice": k, "js": d, "derive": bool(d > TAU), "region": None}
        if d > TAU:
            classes = np.nonzero(c / (r + 1e-6) > KAPPA)[0]
            if len(classes):
                ligne["region"] = [float(bornes[classes[0]]), float(bornes[classes[-1] + 1])]
        sortie.append(ligne)
    return sortie


def _fenetres(arr: dict, marque: np.ndarray, W: int):
    """Toutes les fenetres de longueur W dont le DERNIER pas verifie marque (indices des lignes triees)."""
    debuts = []
    for s, e in zip(arr["starts"], arr["ends"]):
        if e - s >= W:
            debuts.append(np.arange(s, e - W + 1))
    debuts = np.concatenate(debuts) if debuts else np.zeros(0, dtype=np.int64)
    debuts = debuts[marque[debuts + W - 1]]
    return debuts


def preparer(a):
    """Applique au jeu derive le pretraitement du jeu de reference (statistiques du train)."""
    from niveau1_harness import (ACCOUNT_COL, TIME_COL, build_regime_frame, prepare_window_arrays,
                                 read_momtsim_parquet, standardize)
    with open(os.path.join(a.data_dir, "preprocessing.json"), encoding="utf-8") as f:
        pre = json.load(f)
    with open(os.path.join(a.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    brut = read_momtsim_parquet(a.drift_path)
    scen = pd.read_parquet(a.drift_path, columns=["nameOrig", "step", "fraudScenario"])
    frame, _ = build_regime_frame(brut, pre["regime"], pre.get("preproc", "B"), {"m_j": pre.get("m_j", {})})
    cols = pre["feature_cols"]
    standardize(frame, cols, stats={c: (v["mean"], v["std"]) for c, v in pre["standardize_stats"].items()})
    # Meme tri que prepare_window_arrays, pour aligner le scenario et le pas sur les lignes.
    ordre = frame.sort_values([ACCOUNT_COL, TIME_COL], kind="mergesort").index.to_numpy()
    pas = frame[TIME_COL].to_numpy()[ordre]
    scenario = scen["fraudScenario"].fillna("").astype(str).to_numpy()[ordre]
    arr = prepare_window_arrays(frame, cols)
    W = meta["W"]
    pivot = np.quantile(np.unique(pas), 0.5)
    fin_adapt = np.quantile(np.unique(pas[pas <= pivot]), 0.9)
    parts = {"adapt": pas <= fin_adapt, "suivi": (pas > fin_adapt) & (pas <= pivot), "eval": pas > pivot}
    out = os.path.join(a.data_dir, "derive")
    os.makedirs(out, exist_ok=True)
    for nom, masque in parts.items():
        d = _fenetres(arr, masque, W)
        np.save(os.path.join(out, f"X_{nom}.npy"), arr["feat"][d[:, None] + np.arange(W)])
        np.save(os.path.join(out, f"y_{nom}.npy"), arr["targ"][d + W - 1])
        np.save(os.path.join(out, f"scenario_{nom}.npy"), scenario[d + W - 1])
    X_val = np.load(os.path.join(a.data_dir, "X_val.npy"), mmap_mode="r")
    garde = np.ones(len(X_val), dtype=bool)
    garde[np.load(os.path.join(a.data_dir, "monitor_idx.npy"))] = False
    idx = np.nonzero(garde)[0]
    np.save(os.path.join(out, "ref_idx.npy"), idx)
    x_ref = np.ascontiguousarray(X_val[idx[:200_000], -1, :])
    x_cur = np.load(os.path.join(out, "X_adapt.npy"), mmap_mode="r")[:, -1, :]
    reg = regions_de_derive(x_ref, np.asarray(x_cur), cols)
    with open(os.path.join(out, "regions.json"), "w", encoding="utf-8") as f:
        json.dump({"tau": TAU, "kappa": KAPPA, "delta": DELTA, "variables": reg}, f, indent=1, ensure_ascii=False)
    print(f"[derive] {sum(r['derive'] for r in reg)} variables en derive -> {out}", flush=True)


def _scores(model, X, lot=4096) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.asarray(X[s:s + lot]))).numpy()
                               for s in range(0, len(X), lot)])


def _mesures(model, X, y, scenario, cible: str) -> dict:
    from niveau1_harness import compute_metrics
    s = np.nan_to_num(_scores(model, X), nan=0.5).clip(0.0, 1.0)
    m = compute_metrics(np.asarray(y), s)
    res = {k: m.get(k) for k in ("mcc", "auc_pr", "brier")}
    if scenario is not None:
        c = np.char.find(np.char.lower(scenario.astype(str)), cible) >= 0
        res["rappel_scenario"] = float((s[c] >= 0.5).mean()) if c.any() else None
        res["n_scenario"] = int(c.sum())
    return res


def lancer(a):
    from loss import mkan_total_loss, weighted_bce
    from edges import etendre_modele
    from train_run import _atomic_torch_save, _write_json, build_model
    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(a.out_dir, "DONE")):
        return EXIT_DONE
    ck0 = torch.load(os.path.join(a.run_dir, "model.pt"), map_location="cpu", weights_only=False)
    spec = ck0["spec"]
    data_dir = spec["data_dir"]
    der = os.path.join(data_dir, "derive")
    with open(os.path.join(der, "regions.json"), encoding="utf-8") as f:
        regions = [r for r in json.load(f)["variables"] if r["region"]]
    charge = lambda n: np.load(os.path.join(der, n + ".npy"), mmap_mode="r")    # noqa: E731
    X_ad, y_ad = charge("X_adapt"), charge("y_adapt")
    X_su, y_su = np.asarray(charge("X_suivi")), np.asarray(charge("y_suivi"))
    model = build_model(spec, ck0["input_size"])
    model.load_state_dict(ck0["state_dict"])
    if a.bras == "extension":
        g = np.random.default_rng(spec["seed"])
        echant = np.asarray(X_ad[np.sort(g.choice(len(X_ad), min(20_000, len(X_ad)), replace=False)), -1, :])
        echant = np.concatenate([np.load(os.path.join(data_dir, "grid_samples.npy")), echant])
        etendre_modele(model, [(r["indice"], *r["region"]) for r in regions], ck0["input_size"], DELTA,
                       torch.from_numpy(echant))
    if a.bras != "fige":
        lr, bs = spec["lr"] / 10, spec["batch_size"]
        opt = torch.optim.Adam(model.parameters(), lr=lr)   # recree apres l'extension
        gen = torch.Generator().manual_seed(spec["seed"])
        n, n_lots = len(X_ad), int(np.ceil(len(X_ad) / bs))
        chemin = os.path.join(a.out_dir, "checkpoint.pt")
        st = dict(epoch=0, wait=0, best=float("inf"), best_model=None)
        if os.path.exists(chemin):
            ck = torch.load(chemin, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            gen.set_state(ck["gen"])
            st = ck["st"]
        arret = {"f": False}
        signal.signal(signal.SIGTERM, lambda *_: arret.update(f=True))
        stop = lambda: arret["f"] or os.path.exists(config.STOP_FILE)    # noqa: E731
        while st["wait"] < spec["patience"] and st["epoch"] < spec["max_epochs"]:
            model.train()
            perm = torch.randperm(n, generator=gen).numpy()
            for b in range(n_lots):      # une epoque interrompue est refaite (point de reprise par epoque)
                idx = np.sort(perm[b * bs:(b + 1) * bs])
                perte, _, _, _ = mkan_total_loss(model, torch.from_numpy(np.asarray(X_ad[idx])),
                                                 torch.from_numpy(np.asarray(y_ad[idx])), lam=spec["lam"],
                                                 mu1=spec["mu1"], mu2=spec["mu2"])
                opt.zero_grad(set_to_none=True)
                perte.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec["clip_grad_norm"])
                opt.step()
                if b % 500 == 0 and stop():
                    return EXIT_INTERRUPTED
            model.eval()
            with torch.no_grad():
                suivi = float(np.mean([float(weighted_bce(model(torch.from_numpy(X_su[s:s + 4096])),
                                                          torch.from_numpy(y_su[s:s + 4096])))
                                       for s in range(0, len(X_su), 4096)]))
            st["epoch"] += 1
            if suivi < st["best"] - spec["min_delta"]:
                st.update(best=suivi, wait=0, best_model={k: v.clone() for k, v in model.state_dict().items()})
            else:
                st["wait"] += 1
            _atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "gen": gen.get_state(), "st": st}, chemin)
            if stop():
                return EXIT_INTERRUPTED
        if st["best_model"] is not None:
            model.load_state_dict(st["best_model"])
    model.eval()
    X_val = np.load(os.path.join(data_dir, "X_val.npy"), mmap_mode="r")
    y_val = np.load(os.path.join(data_dir, "y_val.npy"), mmap_mode="r")
    ref = np.load(os.path.join(der, "ref_idx.npy"))
    res = {"bras": a.bras, "seed": spec["seed"], "run_final": spec["run_id"], "label": spec["label"],
           "regions": regions,
           "D_eval": _mesures(model, charge("X_eval"), charge("y_eval"), np.asarray(charge("scenario_eval")), a.cible),
           "D_ref": _mesures(model, X_val[ref], y_val[ref], None, a.cible)}
    _write_json(os.path.join(a.out_dir, "resultat.json"), res)
    torch.save({"state_dict": model.state_dict(), "spec": spec, "bras": a.bras},
               os.path.join(a.out_dir, "model.pt"))
    with open(os.path.join(a.out_dir, "DONE"), "w") as f:
        f.write("ok")
    print(f"[derive] {spec['label']} {a.bras} : {res['D_eval']}", flush=True)
    return EXIT_DONE


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    s = p.add_subparsers(dest="action", required=True)
    q = s.add_parser("preparer")
    q.add_argument("--drift_path", required=True)
    q.add_argument("--data_dir", required=True)
    q = s.add_parser("lancer")
    q.add_argument("--run_dir", required=True, help="run final (modele d'une graine)")
    q.add_argument("--bras", required=True, choices=["fige", "affinage", "extension"])
    q.add_argument("--out_dir", required=True)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--cible", default="split", help="partie du nom du scenario derive, en minuscules")
    a = p.parse_args()
    if a.action == "preparer":
        preparer(a)
    else:
        sys.exit(lancer(a))


if __name__ == "__main__":
    main()
