"""
comparaison_shap.py -- XGBoost explique par TreeSHAP contre les fonctions d'arete des finalistes (validation uniquement).

Audit du 12/11/2026, point 2.5. Question : un XGBoost explique par SHAP s'appuie-t-il sur les memes variables que les
cellules KAN, et que gagne-t-on a l'une ou l'autre explication ?

  1. TreeSHAP (Lundberg et al., 2020) exact, via xgboost pred_contribs, sur n fenetres de validation tirees avec une
     graine fixe, pour le XGBoost du regime traite a W = 5, graine 1. Importance d'une variable = moyenne de |SHAP| sur
     ses quatre agregats de fenetre (derniere valeur, moyenne, ecart-type, |fin - debut|).
  2. Importance d'une variable pour une cellule KAN = somme des normes L1 des aretes actives qui en partent, toutes
     portes confondues (fichier fidelite_aretes.csv produit par fidelite_formule.py).
  3. Accord des deux classements (Spearman) et des cinq premieres variables ; additivite de TreeSHAP verifiee (somme
     des contributions + biais = marge du modele).

Usage (depuis MKAN/Etude_benchmarck) :  python comparaison_shap.py [--n 20000]
Sorties : analyse_banc/final_2026-11-08/comparaison_shap.csv, ANALYSE_SHAP.md, memoire/tableaux_banc/tab_shap.tex
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
sys.path.insert(0, ICI)
from niveau1_harness import (read_momtsim_parquet, build_regime_frame, standardize, prepare_window_arrays,  # noqa: E402
                             sample_windows_from_arrays)

FINAL = os.path.join(ICI, "campagne_aws", "resultats_instance", "final_2026-11-08")
OUT = os.path.join(ICI, "analyse_banc", "final_2026-11-08")
NOMS = {"delta_B_orig": "$\\Delta B_{orig}$", "delta_B_dest": "$\\Delta B_{dest}$", "r1": "$r_1$", "r2": "$r_2$",
        "flag_anomalie": "drapeau d'anomalie", "delta_commission": "écart de commission",
        "var_agent_split": "variance agent-client", "rho_rupture": "ratio de rupture", "rho_refund": "ratio de remboursement",
        "v1h": "vélocité sur 1 h", "flag_nuit": "drapeau nocturne", "rho_nouveau": "part de nouveaux destinataires"}
KAN = {"GRU traite F=dog I=dog C=dog": "GRUKANCell DoG/DoG/DoG", "GRU traite F=dog I=relu C=dog": "GRUKANCell DoG/ReLU-KAN/DoG",
       "GRU traite F=hyb I=hyb C=hyb": "GRUKANCell hybride"}


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    import xgboost as xgb
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    a = ap.parse_args()
    runs = pd.read_csv(os.path.join(FINAL, "runs.csv"))
    rid = runs[runs.label == "baseline xgboost traite W=5 s1"].run_id.iloc[0]
    with open(os.path.join(FINAL, "runs", rid, "model.pkl"), "rb") as f:
        mod = pickle.load(f)

    etat = {}
    df_tr, cols = build_regime_frame(read_momtsim_parquet(os.path.join(RACINE, "MOMTSIM", "config", "featuresLog.parquet")),
                                     "traite", "B", etat)
    df_va, _ = build_regime_frame(read_momtsim_parquet(os.path.join(RACINE, "data", "val_features.parquet")), "traite", "B", etat)
    st = standardize(df_tr, cols)
    standardize(df_va, cols, stats=st)
    del df_tr
    X, y, _ = sample_windows_from_arrays(prepare_window_arrays(df_va, cols), 5, None, a.n, 43)
    F = np.concatenate([X[:, -1, :], X.mean(axis=1), X.std(axis=1), np.abs(X[:, -1, :] - X[:, 0, :])], axis=1)
    d = xgb.DMatrix(F)
    booster = mod.get_booster()
    contrib = booster.predict(d, pred_contribs=True, iteration_range=(0, mod.best_iteration + 1))
    marge = booster.predict(d, output_margin=True, iteration_range=(0, mod.best_iteration + 1))
    ecart_additivite = float(np.abs(contrib.sum(axis=1) - marge).max())
    p = mod.predict_proba(F)[:, 1]
    nf = len(cols)
    shap_abs = np.abs(contrib[:, :-1]).mean(axis=0).reshape(4, nf).sum(axis=0)
    imp = pd.DataFrame({"variable": cols, "shap_xgboost": shap_abs})
    imp["part_shap"] = imp.shap_xgboost / imp.shap_xgboost.sum()
    # nombre de contributions necessaires pour expliquer 90 % de |SHAP| d'une decision
    tri = np.sort(np.abs(contrib[:, :-1]), axis=1)[:, ::-1]
    cum = np.cumsum(tri, axis=1) / tri.sum(axis=1, keepdims=True)
    k90 = (cum < 0.9).sum(axis=1) + 1

    ar = pd.read_csv(os.path.join(OUT, "fidelite_aretes.csv"))
    lignes = []
    for m, nom in KAN.items():
        s = ar[(ar.modele == m) & ar.entree.isin(cols)].groupby("entree").l1.sum()
        imp[nom] = imp.variable.map(s).fillna(0.0)
        imp["part_" + nom] = imp[nom] / imp[nom].sum()
        rho = stats.spearmanr(imp.shap_xgboost, imp[nom])
        top_x = set(imp.nlargest(5, "shap_xgboost").variable)
        top_k = set(imp.nlargest(5, nom).variable)
        lignes.append((nom, rho.statistic, rho.pvalue, len(top_x & top_k)))
    imp = imp.sort_values("shap_xgboost", ascending=False)
    imp.to_csv(os.path.join(OUT, "comparaison_shap.csv"), index=False)

    md = ["# XGBoost expliqué par TreeSHAP contre fonctions d'arête des cellules KAN", "",
          f"Validation uniquement, {len(y)} fenêtres (W = 5, régime traité), XGBoost graine 1. Écart maximal d'additivité "
          f"de TreeSHAP (somme des contributions moins marge du modèle) : {ecart_additivite:.2e}.",
          f"Contributions nécessaires pour couvrir 90 % de |SHAP| d'une décision : médiane {int(np.median(k90))}, "
          f"9e décile {int(np.percentile(k90, 90))} (sur {F.shape[1]}).", "",
          "| Variable | Part SHAP XGBoost | " + " | ".join(f"Part L1 {n}" for n in KAN.values()) + " |",
          "|---|---|" + "---|" * len(KAN)]
    for _, r in imp.iterrows():
        md.append(f"| {r.variable} | {fr(100 * r.part_shap, 1)} % | "
                  + " | ".join(f"{fr(100 * r['part_' + n], 1)} %" for n in KAN.values()) + " |")
    md += ["", "| Cellule KAN | Spearman avec SHAP | p | Variables communes dans les 5 premières |", "|---|---|---|---|"]
    md += [f"| {n} | {fr(rho)} | {fr(pv)} | {c} |" for n, rho, pv, c in lignes]
    with open(os.path.join(OUT, "ANALYSE_SHAP.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    tex = [" & ".join([NOMS.get(r.variable, r.variable), fr(100 * r.part_shap, 1)]
                      + [fr(100 * r["part_" + n], 1) for n in KAN.values()]) + " \\\\" for _, r in imp.iterrows()]
    tex.append("\\midrule")
    tex.append(" & ".join(["Spearman avec SHAP", ""] + [fr(rho, 2) for _, rho, _, _ in lignes]) + " \\\\")
    with open(os.path.join(RACINE, "memoire", "tableaux_banc", "tab_shap.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                "\\caption{Part de chaque variable dans l'explication : moyenne de $|$SHAP$|$ de XGBoost (régime traité, "
                f"$W = 5$, {len(y)} fenêtres de validation) et part de la norme L1 des arêtes qui partent de la variable "
                "dans chaque GRUKANCell finaliste, en \\%}\n\\label{tab:shap}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}l" + "r" * (1 + len(KAN)) + "@{}}\n\\toprule\n"
                "\\textbf{Variable} & \\textbf{SHAP XGBoost} & " + " & ".join(f"\\textbf{{{n}}}" for n in KAN.values())
                + " \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    print("\n".join(md))
    print("k90 moyenne", k90.mean(), "MCC echantillon", float(np.corrcoef(p >= 0.5, y)[0, 1]))


if __name__ == "__main__":
    main()
