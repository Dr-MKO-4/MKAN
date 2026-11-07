"""
evaluation_test.py -- Evaluation unique sur le jeu de test (regle 4 du chapitre 4), en dernier, une seule fois.

Ne reentraine rien. Modeles evalues : tous les runs finis des phases demandees (defaut : final et baselines),
chaque graine separement. Pretraitement fige : statistiques du train relues dans preprocessing.json. Fenetrage
identique a la validation (toutes les fenetres, etiquette du dernier pas). MCC au seuil 0,5, sans reglage.

Sorties dans <out_dir>/test/ :
    VERROU              liste des runs evalues, ecrite avant de lire le test
    scores/<run_id>.npy scores du run ; y_<regime>_W<W>.npy etiquettes
    test_runs.csv       une ligne par run
    test_resume.json    moyenne et ecart-type par architecture

Une interruption se reprend : relancer la meme commande complete les runs manquants. Une liste de runs differente
de celle du VERROU est refusee : on n'ajoute pas un modele apres avoir vu le test.

    python evaluation_test.py                          # affiche les runs, ne lit pas le test
    python evaluation_test.py --test_path ../../../data/test_features.parquet --une_seule_fois
    python evaluation_test.py --verifier_sur_val --phases etape4_traite --max_runs 1
        # controle sans toucher au test : refait un run sur la validation et compare a result.json
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from datetime import datetime, timezone

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

METRIQUES = ("mcc", "auc_pr", "auc_roc", "brier", "f1_best", "precision_at_recall_0.5",
             "precision_at_recall_0.8")


def lister_runs(out_dir: str, phases: list) -> list:
    specs = []
    for done in sorted(glob.glob(os.path.join(out_dir, "runs", "*", "DONE"))):
        with open(os.path.join(os.path.dirname(done), "spec.json"), encoding="utf-8") as f:
            spec = json.load(f)
        if spec["phase"] in phases:
            specs.append(spec)
    return specs


def fenetres(chemin: str, spec: dict, cache_df: dict, cache_fen: dict):
    """Memes operations que prepare_data.prepare, avec les statistiques du train."""
    from niveau1_harness import (read_momtsim_parquet, build_regime_frame, standardize,
                                 prepare_window_arrays, sample_windows_from_arrays)
    cle = (spec["regime"], spec["W"])
    if cle in cache_fen:
        return cache_fen[cle]
    with open(os.path.join(spec["data_dir"], "preprocessing.json"), encoding="utf-8") as f:
        pre = json.load(f)
    if pre.get("preproc", "B") != "B" or pre.get("colonnes_retirees"):
        raise ValueError(f"{spec['label']} : seul le pretraitement B complet est prevu ici")
    if spec["regime"] not in cache_df:
        if "brut" not in cache_df:
            cache_df["brut"] = read_momtsim_parquet(chemin)     # jeu lu une fois
        df, cols = build_regime_frame(cache_df["brut"], spec["regime"], "B", {})
        if cols != pre["feature_cols"]:
            raise ValueError(f"colonnes differentes du train pour le regime {spec['regime']}")
        standardize(df, cols, stats={c: (s["mean"], s["std"]) for c, s in pre["standardize_stats"].items()})
        cache_df[spec["regime"]] = prepare_window_arrays(df, cols)
    X, y, _ = sample_windows_from_arrays(cache_df[spec["regime"]], spec["W"], None, None, 43)
    cache_fen[cle] = (X, y)
    return X, y


def scores_run(spec: dict, run_dir: str, X: np.ndarray) -> np.ndarray:
    if spec.get("baseline") == "xgboost":
        from baselines import make_xgb_features
        with open(os.path.join(run_dir, "model.pkl"), "rb") as f:
            mod = pickle.load(f)
        parts = [mod.predict_proba(make_xgb_features(X[s:s + 65536]))[:, 1] for s in range(0, len(X), 65536)]
    else:
        ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
        if spec.get("baseline"):
            from train_baseline import construire
            model = construire(spec, X.shape[-1])
        else:
            from train_run import build_model
            model = build_model(spec, ck["input_size"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            parts = [model(torch.from_numpy(X[s:s + 4096])).numpy() for s in range(0, len(X), 4096)]
    s = np.nan_to_num(np.concatenate(parts).astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    return s.clip(0.0, 1.0)


def architecture(spec: dict) -> str:
    return f"{spec['cell_type']} {spec['regime']} W={spec['W']} " + " ".join(
        f"{g}={b}" for g, b in sorted(spec["base_by_gate"].items()))


def resumer(lignes: list) -> dict:
    groupes = {}
    for l in lignes:
        groupes.setdefault(l["architecture"], []).append(l)
    return {a: {"n_graines": len(g),
                **{m: [float(np.mean([x[m] for x in g])), float(np.std([x[m] for x in g], ddof=0))]
                   for m in METRIQUES}}
            for a, g in groupes.items()}


def verrouiller(chemin: str, run_ids: list, source: str) -> None:
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            deja = json.load(f)
        if sorted(deja["run_ids"]) != sorted(run_ids):
            raise SystemExit("Refus : la liste des runs differe de celle du VERROU, le test a deja ete ouvert "
                             "pour une autre liste.")
        print(f"Reprise de l'evaluation commencee le {deja['debut']}", flush=True)
        return
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"debut": datetime.now(timezone.utc).isoformat(), "source": source, "run_ids": run_ids}, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out_dir", default=config.OUT_DIR)
    p.add_argument("--phases", nargs="+", default=["final", "exp_baselines"])
    p.add_argument("--test_path")
    p.add_argument("--une_seule_fois", action="store_true", help="confirme l'ouverture du test")
    p.add_argument("--verifier_sur_val", action="store_true")
    p.add_argument("--max_runs", type=int, help="avec --verifier_sur_val seulement")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    from niveau1_harness import compute_metrics

    specs = lister_runs(a.out_dir, a.phases)
    for ph in a.phases:
        print(f"{ph} : {sum(s['phase'] == ph for s in specs)} runs finis", flush=True)
    if not specs:
        raise SystemExit("Aucun run a evaluer.")
    torch.set_num_threads(a.threads)

    if a.verifier_sur_val:
        cache_df, cache_fen = {}, {}
        for spec in specs[:a.max_runs]:
            run_dir = os.path.join(a.out_dir, "runs", spec["run_id"])
            X, y = fenetres(config.VAL_PATH, spec, cache_df, cache_fen)
            m = compute_metrics(y, scores_run(spec, run_dir, X))
            with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
                ref = json.load(f)["metrics"]
            ecart = max(abs(m[k] - ref[k]) for k in ("mcc", "auc_pr", "brier"))
            print(f"{spec['label']} : MCC {m['mcc']:.6f} (result.json {ref['mcc']:.6f}), ecart max {ecart:.2e}",
                  flush=True)
            if ecart > 1e-4:
                raise SystemExit("Ecart avec la validation d'origine : ne pas ouvrir le test.")
        print("Controle sur la validation reussi. Le test n'a pas ete lu.", flush=True)
        return

    if not a.une_seule_fois or not a.test_path:
        print("Rien n'est lu : ajouter --test_path et --une_seule_fois pour ouvrir le test.", flush=True)
        return

    sortie = os.path.join(a.out_dir, "test")
    os.makedirs(os.path.join(sortie, "scores"), exist_ok=True)
    verrouiller(os.path.join(sortie, "VERROU"), [s["run_id"] for s in specs], a.test_path)
    cache_df, cache_fen = {}, {}
    lignes = []
    for i, spec in enumerate(specs, 1):
        chemin = os.path.join(sortie, "scores", spec["run_id"] + ".npy")
        X, y = fenetres(a.test_path, spec, cache_df, cache_fen)
        if os.path.exists(chemin):
            s = np.load(chemin).astype(np.float64)
        else:
            s = scores_run(spec, os.path.join(a.out_dir, "runs", spec["run_id"]), X)
            np.save(chemin + ".tmp.npy", s.astype(np.float32))
            os.replace(chemin + ".tmp.npy", chemin)
        np.save(os.path.join(sortie, "scores", f"y_{spec['regime']}_W{spec['W']}.npy"), y)
        m = compute_metrics(y, s)
        lignes.append({"run_id": spec["run_id"], "label": spec["label"], "phase": spec["phase"],
                       "architecture": architecture(spec), "seed": spec["seed"], "n_test": len(y),
                       **{k: m[k] for k in METRIQUES}})
        print(f"[{i}/{len(specs)}] {spec['label']} : MCC={m['mcc']:.4f} PR-AUC={m['auc_pr']:.4f}", flush=True)
    with open(os.path.join(sortie, "test_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    with open(os.path.join(sortie, "test_resume.json"), "w", encoding="utf-8") as f:
        json.dump(resumer(lignes), f, indent=1, ensure_ascii=False)
    print(f"-> {sortie}", flush=True)


if __name__ == "__main__":
    main()
