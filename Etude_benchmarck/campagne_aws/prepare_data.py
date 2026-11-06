"""
prepare_data.py -- Prepare UNE FOIS les fenetres de la campagne et les ecrit sur
disque ; chaque processus d'entrainement les relit ensuite sans copie
(np.load(..., mmap_mode="r")), au lieu de relire le parquet et de refaire le
fenetrage dans chacun des dizaines de processus.

Memes fonctions que l'Etape 4 (niveau1_harness) : build_regime_frame (NaN -> 0,
sign(x)*log1p(|x|) sur les colonnes non binaires), standardize (statistiques du
train, ecretage a [-4, 4]), fenetrage par compte avec l'etiquette de la DERNIERE
transaction, tirage de n fenetres d'entrainement avec la graine 42, toute la
validation. Le jeu de test n'est pas lu.

Pour chaque (regime, W, n) : dossier <DATA_DIR>/<regime>_W<W>_n<n|all>/ avec
    X_train.npy y_train.npy X_val.npy y_val.npy
    monitor_idx.npy  (fenetres de validation suivies pour l'arret anticipe, rng(1))
    audit_idx.npy    (256 fenetres de validation pour R2_symbolic, rng(0))
    grid_samples.npy (valeurs x_t pour recaler la grille des bases, rng(7))
    preprocessing.json (etat complet du pretraitement, relu par le tableau de bord)
    meta.json        (ecrit en dernier : marque de jeu complet)

Usage :
    python prepare_data.py                              # Etape 4 : brut+traite, W=10 et 5
    python prepare_data.py --regimes brut --W 10 --n_train 1000000   # entrainement final
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import config
from niveau1_harness import (read_momtsim_parquet, build_regime_frame, d_ks_normale, standardize,
                             prepare_window_arrays, sample_windows_from_arrays,
                             BINARY_COLS, ACTION_ONEHOT_COLS)


def dataset_dir(regime: str, W: int, n_train, preproc: str = "B", drop=()) -> str:
    """Le nom historique est garde pour le pretraitement B sans colonne retiree."""
    nom = f"{regime}_W{W}_n{n_train or 'all'}"
    if preproc != "B":
        nom += f"_p{preproc}"
    if drop:
        nom += "_sans_" + "_".join(sorted(drop))
    return os.path.join(config.DATA_DIR, nom)


def dataset_ready(path: str) -> bool:
    return os.path.exists(os.path.join(path, "meta.json"))


def _file_fingerprint(path: str) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    st = os.stat(path)
    return {"path": os.path.relpath(path, config.ROOT_DIR), "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "sha256": h.hexdigest()}


def _save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


ETAPES_PREPROC = {
    "A": "log(1 + max(x - m_j, 0)) sur les colonnes continues de D_KS >= 0,15 (train)",
    "B": "sign(x)*log1p(|x|) sur les colonnes non binaires",
    "C": "aucune transformation avant la standardisation",
}


def prepare(regimes, Ws, n_train, n_monitor=50_000, n_audit=256, force=False,
            preproc: str = "B", drop=()) -> list:
    """preproc, drop : memes fenetres et memes tirages que le jeu de reference (les graines et le
    fenetrage ne dependent ni des valeurs ni des colonnes), seules les colonnes changent."""
    todo = [(r, W) for r in regimes for W in Ws
            if force or not dataset_ready(dataset_dir(r, W, n_train, preproc, drop))]
    if not todo:
        print("Tous les jeux demandes existent deja.")
        return []
    sources = {"train": _file_fingerprint(config.TRAIN_PATH),
               "val": _file_fingerprint(config.VAL_PATH)}
    df_train_raw = read_momtsim_parquet(config.TRAIN_PATH)
    df_val_raw = read_momtsim_parquet(config.VAL_PATH)
    done = []
    for regime in sorted({r for r, _ in todo}):
        t0 = time.time()
        etat = {}
        df_train, cols = build_regime_frame(df_train_raw, regime, preproc, etat)
        df_val, _ = build_regime_frame(df_val_raw, regime, preproc, etat)
        cols = [c for c in cols if c not in set(drop)]
        d_ks = {c: d_ks_normale(df_train[c].to_numpy(), 42 + j) for j, c in enumerate(cols)
                if c not in BINARY_COLS and c not in ACTION_ONEHOT_COLS}
        stats = standardize(df_train, cols)
        standardize(df_val, cols, stats=stats)
        arr_train = prepare_window_arrays(df_train, cols)
        arr_val = prepare_window_arrays(df_val, cols)
        del df_train, df_val
        preprocessing = {
            "regime": regime, "feature_cols": cols,
            "steps": ["fillna(0)",
                      ETAPES_PREPROC[preproc],
                      "standardisation (moyenne, ecart-type du train)",
                      "ecretage a [-4, 4]"],
            "log_excluded_cols": [c for c in cols if c in BINARY_COLS or c in ACTION_ONEHOT_COLS],
            "standardize_stats": {c: {"mean": m, "std": s} for c, (m, s) in stats.items()},
            "clip": [-4.0, 4.0],
            "action_categories_onehot": [c for c in cols if c in ACTION_ONEHOT_COLS],
            "source_code": "niveau1_harness.build_regime_frame + standardize",
            "preproc": preproc, "colonnes_retirees": sorted(drop),
            "d_ks_apres_transformation": d_ks,
            "d_ks_avant_transformation": etat.get("d_ks", {}), "m_j": etat.get("m_j", {}),
        }
        for r, W in todo:
            if r != regime:
                continue
            out = dataset_dir(regime, W, n_train, preproc, drop)
            os.makedirs(out, exist_ok=True)
            X_tr, y_tr, n_tr_full = sample_windows_from_arrays(arr_train, W, None, n_train, 42)
            X_va, y_va, n_va_full = sample_windows_from_arrays(arr_val, W, None, None, 43)
            n_mon = min(n_monitor, len(X_va))
            mon_idx = np.sort(np.random.default_rng(1).choice(len(X_va), n_mon, replace=False))
            aud_idx = np.sort(np.random.default_rng(0).choice(len(X_va), min(n_audit, len(X_va)),
                                                              replace=False))
            flat = X_tr.reshape(-1, X_tr.shape[-1])
            g_idx = np.random.default_rng(7).choice(len(flat), min(config.N_GRID_SAMPLES, len(flat)),
                                                    replace=False)
            for name, a in (("X_train", X_tr), ("y_train", y_tr), ("X_val", X_va), ("y_val", y_va),
                            ("monitor_idx", mon_idx), ("audit_idx", aud_idx),
                            ("grid_samples", flat[np.sort(g_idx)])):
                _save(os.path.join(out, name + ".npy"), a)
            _write_json(os.path.join(out, "preprocessing.json"), preprocessing)
            meta = {"regime": regime, "W": W, "input_size": len(cols),
                    "n_train": int(len(X_tr)), "n_train_available": int(n_tr_full),
                    "n_val": int(len(X_va)), "n_val_available": int(n_va_full),
                    "fraud_rate_train": float(y_tr.mean()), "fraud_rate_val": float(y_va.mean()),
                    "n_monitor": int(n_mon), "sample_seed_train": 42,
                    "label_rule": "etiquette de la derniere transaction de la fenetre",
                    "sources": sources,
                    "created": datetime.now(timezone.utc).isoformat()}
            _write_json(os.path.join(out, "meta.json"), meta)
            print(f"[donnees] {regime} W={W} : train {len(X_tr):,}/{n_tr_full:,}, "
                  f"val {len(X_va):,}, fraude train {y_tr.mean():.3f} -> {out}", flush=True)
            done.append(out)
        print(f"[donnees] regime {regime} prepare en {time.time() - t0:.0f} s", flush=True)
    return done


def main():
    p = argparse.ArgumentParser(description="Preparation des fenetres de la campagne AWS")
    p.add_argument("--regimes", nargs="+", default=["brut", "traite"], choices=["brut", "traite"])
    p.add_argument("--W", nargs="+", type=int, default=[10, 5])
    p.add_argument("--n_train", type=int, default=config.ETAPE4["n_windows_train"],
                   help="0 = toutes les fenetres d'entrainement")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    prepare(args.regimes, args.W, args.n_train or None, force=args.force)


if __name__ == "__main__":
    main()
