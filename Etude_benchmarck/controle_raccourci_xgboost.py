"""
controle_raccourci_xgboost.py -- Controle du raccourci par les soldes, version executable sur un poste de travail.

Pourquoi ce script (audit du 12/11/2026, AUDIT_TRAVAUX_2026-11-12.md § 9.1) : le controle lance sur AWS les 08-09/11
retirait les colonnes de solde brutes d'un modele du regime traite, qui ne les contient pas ; il n'a donc rien controle.
Reentrainer une cellule KAN coute environ 14 heures sur 64 coeurs. XGBoost, lui, s'entraine en quelques minutes sur les
memes fenetres : on mesure donc, au niveau des DONNEES, ce que la detection perd quand les soldes sont retires.

Variantes (memes fenetres, memes graines 1 a 5, memes hyperparametres que campagne_aws/train_baseline.py) :
    brut, W = 10   : complet ; sans les soldes de l'emetteur ; sans aucun solde
    traite, W = 5  : complet ; sans les quatre variables derivees des soldes (delta_B_orig, delta_B_dest, r1, r2)

Limite a ecrire avec le resultat : c'est un controle sur XGBoost. Il dit si l'information des soldes est necessaire a
la detection dans ces donnees ; il ne dit pas ce qu'une cellule KAN en ferait. Validation uniquement, test non lu.

Usage (depuis MKAN/Etude_benchmarck) :  python controle_raccourci_xgboost.py  [--n-train 1000000] [--threads 8]
Sorties : analyse_banc/final_2026-11-08/raccourci_xgboost_runs.csv, raccourci_xgboost.csv, ANALYSE_RACCOURCI_XGBOOST.md,
          memoire/tableaux_banc/tab_raccourci_xgboost.tex
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, ".."))
from niveau1_harness import (read_momtsim_parquet, build_regime_frame, standardize, prepare_window_arrays,  # noqa: E402
                             sample_windows_from_arrays, compute_metrics)

TRAIN_PATH = os.path.join(RACINE, "MOMTSIM", "config", "featuresLog.parquet")
VAL_PATH = os.path.join(RACINE, "data", "val_features.parquet")
EMETTEUR = ["oldBalanceOrig", "newBalanceOrig"]
TOUS = EMETTEUR + ["oldBalanceDest", "newBalanceDest"]
DERIVEES = ["delta_B_orig", "delta_B_dest", "r1", "r2"]
PLAN = [("brut", 10, [("complet", []), ("sans les soldes de l'émetteur", EMETTEUR), ("sans aucun solde", TOUS)]),
        ("traite", 5, [("complet", []), ("sans les variables dérivées des soldes", DERIVEES)])]
SEEDS = [1, 2, 3, 4, 5]


def fr(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


def agregats(X):
    """Memes agregats que MKAN/baselines.make_xgb_features : derniere valeur, moyenne, ecart-type, |fin - debut|."""
    return np.concatenate([X[:, -1, :], X.mean(axis=1), X.std(axis=1), np.abs(X[:, -1, :] - X[:, 0, :])], axis=1)


def holm(p):
    p = np.asarray(p, dtype=float)
    ordre, adj, courant = np.argsort(p), np.empty(len(p)), 0.0
    for rang, i in enumerate(ordre):
        courant = max(courant, (len(p) - rang) * p[i])
        adj[i] = min(1.0, courant)
    return adj


def main():
    import xgboost as xgb
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(ICI, "analyse_banc", "final_2026-11-08"))
    ap.add_argument("--tex", default=os.path.join(RACINE, "memoire", "tableaux_banc"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chemin_runs = os.path.join(a.out, "raccourci_xgboost_runs.csv")
    lignes = pd.read_csv(chemin_runs).to_dict("records") if os.path.exists(chemin_runs) else []
    faits = {(l["regime"], l["variante"], l["seed"]) for l in lignes}

    df_train_raw = read_momtsim_parquet(TRAIN_PATH)
    df_val_raw = read_momtsim_parquet(VAL_PATH)
    for regime, W, variantes in PLAN:
        if all((regime, v, s) in faits for v, _ in variantes for s in SEEDS):
            continue
        t0 = time.time()
        etat = {}
        df_train, cols = build_regime_frame(df_train_raw, regime, "B", etat)
        df_val, _ = build_regime_frame(df_val_raw, regime, "B", etat)
        st = standardize(df_train, cols)
        standardize(df_val, cols, stats=st)
        arr_train, arr_val = prepare_window_arrays(df_train, cols), prepare_window_arrays(df_val, cols)
        del df_train, df_val
        X_tr, y_tr, _ = sample_windows_from_arrays(arr_train, W, None, a.n_train, 42)
        F_tr = agregats(X_tr); del X_tr, arr_train
        X_va, y_va, _ = sample_windows_from_arrays(arr_val, W, None, None, 43)
        F_va = agregats(X_va); del X_va, arr_val
        mon = np.sort(np.random.default_rng(1).choice(len(F_va), min(50_000, len(F_va)), replace=False))
        nf = len(cols)
        print(f"[{regime} W={W}] {len(F_tr)} fenetres d'entrainement, {len(F_va)} de validation, "
              f"{nf} variables, prepare en {time.time() - t0:.0f} s", flush=True)
        neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        for variante, retirees in variantes:
            absentes = [c for c in retirees if c not in cols]
            if absentes:
                raise SystemExit(f"{regime} : colonnes absentes {absentes} (le controle ne retirerait rien)")
            gardees = [j for j, c in enumerate(cols) if c not in retirees]
            idx = np.concatenate([np.array(gardees) + k * nf for k in range(4)])
            for seed in SEEDS:
                if (regime, variante, seed) in faits:
                    continue
                t1 = time.time()
                mod = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
                                        eval_metric="logloss", tree_method="hist", early_stopping_rounds=30,
                                        random_state=seed, n_jobs=a.threads, verbosity=0)
                mod.fit(F_tr[:, idx], y_tr, eval_set=[(F_va[mon][:, idx], y_va[mon])], verbose=False)
                s = np.concatenate([mod.predict_proba(F_va[i:i + 262144][:, idx])[:, 1]
                                    for i in range(0, len(F_va), 262144)])
                m = compute_metrics(y_va, s)
                p = s >= 0.5
                lignes.append({"regime": regime, "W": W, "variante": variante, "seed": seed,
                               "n_variables": len(gardees), "colonnes_retirees": " ".join(retirees),
                               "mcc": m["mcc"], "auc_pr": m["auc_pr"], "brier": m["brier"],
                               "perf": 0.4147 * max(0.0, m["mcc"]) + 0.2573 * m["auc_pr"] - 0.1529 * m["brier"],
                               "rappel": float(p[y_va == 1].mean()), "taux_faux_positifs": float(p[y_va == 0].mean()),
                               "n_arbres": int(mod.best_iteration + 1), "n_train": int(len(y_tr)), "n_val": int(len(y_va))})
                pd.DataFrame(lignes).to_csv(chemin_runs, index=False)
                print(f"  {variante}, graine {seed} : MCC {m['mcc']:.4f}, PR-AUC {m['auc_pr']:.4f} "
                      f"({time.time() - t1:.0f} s)", flush=True)
        del F_tr, F_va

    d = pd.DataFrame(lignes)
    res = []
    for (regime, W), g in d.groupby(["regime", "W"]):
        ref = g[g.variante == "complet"].set_index("seed")
        for variante, gv in g.groupby("variante", sort=False):
            gv = gv.set_index("seed")
            l = {"regime": regime, "W": W, "variante": variante, "n_variables": int(gv.n_variables.iloc[0]),
                 "mcc": gv.mcc.mean(), "mcc_sd": gv.mcc.std(ddof=1), "auc_pr": gv.auc_pr.mean(), "perf": gv.perf.mean(),
                 "rappel": gv.rappel.mean(), "taux_faux_positifs": gv.taux_faux_positifs.mean()}
            if variante != "complet":
                e = gv.mcc - ref.mcc
                l.update(ecart_mcc=e.mean(), ecart_mcc_sd=e.std(ddof=1), p=stats.ttest_rel(gv.mcc, ref.mcc).pvalue)
            res.append(l)
    r = pd.DataFrame(res)
    masque = r.p.notna()
    r.loc[masque, "p_holm"] = holm(r.loc[masque, "p"].to_numpy())
    r.to_csv(os.path.join(a.out, "raccourci_xgboost.csv"), index=False)

    def pv(x):
        return "" if pd.isna(x) else ("< 0,001" if x < 0.001 else fr(x, 3))

    noms = {"brut": "brut", "traite": "traité"}
    md = ["# Contrôle du raccourci par les soldes, sur XGBoost", "",
          f"Validation uniquement, {len(SEEDS)} graines, {int(d.n_train.iloc[0])} fenêtres d'entraînement. "
          "Écart de MCC apparié par graine à la version complète, p corrigés par Holm.", "",
          "| Régime | W | Variante | Variables | MCC | PR-AUC | Rappel | Faux positifs | Écart de MCC | p corrigé |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for _, l in r.iterrows():
        ecart = "" if pd.isna(l.get("ecart_mcc")) else f"{fr(l.ecart_mcc)} ± {fr(l.ecart_mcc_sd)}"
        md.append(f"| {noms[l.regime]} | {l.W} | {l.variante} | {l.n_variables} | {fr(l.mcc)} ± {fr(l.mcc_sd)} | "
                  f"{fr(l.auc_pr)} | {fr(l.rappel)} | {fr(l.taux_faux_positifs, 5)} | {ecart} | {pv(l.get('p_holm'))} |")
        tex.append(" & ".join([f"{noms[l.regime]}, $W = {l.W}$", l.variante, str(l.n_variables),
                               f"{fr(l.mcc)} $\\pm$ {fr(l.mcc_sd)}", fr(l.auc_pr), fr(l.rappel),
                               ecart.replace("±", "$\\pm$") or "", pv(l.get("p_holm")).replace("<", "$<$")]) + " \\\\")
    md += ["", "Lecture : une chute nette du MCC sans les soldes signifie que la détection s'appuie sur l'arithmétique "
               "des soldes du simulateur. Contrôle fait sur XGBoost, pas sur les cellules KAN."]
    with open(os.path.join(a.out, "ANALYSE_RACCOURCI_XGBOOST.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.tex, "tab_raccourci_xgboost.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                "\\caption{Contrôle du raccourci par les soldes sur XGBoost : détection avec et sans les colonnes de solde, "
                "moyenne $\\pm$ écart-type sur cinq graines, écart de MCC apparié à la version complète et $p$ corrigé "
                "par la méthode de Holm (validation)}\n\\label{tab:raccourci-xgboost}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
                "\\textbf{Régime} & \\textbf{Variante} & \\textbf{Variables} & \\textbf{MCC} & \\textbf{PR-AUC} & "
                "\\textbf{Rappel} & \\textbf{Écart de MCC} & $p$ \\\\\n\\midrule\n" + "\n".join(tex)
                + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
