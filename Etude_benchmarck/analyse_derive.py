"""
analyse_derive.py -- Expérience de dérive (Split Deposit plus irrégulier) : trois bras (modèle figé, affinage seul,
extension de grille puis affinage) x 4 finalistes x 5 graines, sur les modèles de l'entraînement final.

Règle de décision fixée avant l'exécution (mémoire, ch. 4, Grid Extension) : l'extension n'est jugée utile que si elle
améliore la détection sur les données dérivées (D_eval) par rapport à l'affinage seul, sans dégrader la détection sur la
référence (D_ref, une partie de la validation) de plus d'un écart-type entre graines. Le test n'est pas lu.

Sorties : analyse_banc/derive/ANALYSE_DERIVE.md, memoire/tableaux_banc/tab_derive.tex
Usage (depuis MKAN/Etude_benchmarck) : python analyse_derive.py [--dossier ...]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
JOLI = {"GRU traite F=fasterkan I=fasterkan C=dog": "GRUKANCell FasterKAN/FasterKAN/DoG (retenu)",
        "GRU traite F=dog I=dog C=dog": "GRUKANCell DoG/DoG/DoG", "GRU traite F=dog I=relu C=dog": "GRUKANCell DoG/ReLU-KAN/DoG",
        "GRU traite F=hyb I=hyb C=hyb": "GRUKANCell hybride", "TKAN brut F=relu I=bspl C=dog O=lin": "TKANCell B (brut)"}
BRAS = ["fige", "affinage", "extension"]
NOM_BRAS = {"fige": "figé", "affinage": "affinage seul", "extension": "extension et affinage"}


def fr(x, nd=4):
    return "n.d." if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")


def perf(m):
    return 0.4147 * max(0.0, m["mcc"]) + 0.2573 * m["auc_pr"] - 0.1529 * m["brier"]


def holm(p):
    p = np.asarray(p, float)
    o, adj, c = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        c = max(c, (len(p) - k) * p[i])
        adj[i] = min(1.0, c)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default=os.path.join(ICI, "campagne_aws", "resultats_instance", "campagne_v2", "derive"))   # 75 runs : 4 finalistes + modèle retenu
    a = ap.parse_args()
    lignes = []
    for f in sorted(glob.glob(os.path.join(a.dossier, "runs", "*", "resultat.json"))):
        r = json.load(open(f, encoding="utf-8"))
        mod = re.sub(r"\s+s\d+$", "", r["label"])
        lignes.append({"modele": mod, "bras": r["bras"], "seed": int(r["seed"]),
                       "mcc_eval": r["D_eval"]["mcc"], "perf_eval": perf(r["D_eval"]), "rappel_split": r["D_eval"].get("rappel_scenario"),
                       "mcc_ref": r["D_ref"]["mcc"], "perf_ref": perf(r["D_ref"]),
                       "n_regions": sum(1 for g in r.get("regions", []) if g.get("derive"))})
    d = pd.DataFrame(lignes)
    if d.empty:
        raise SystemExit("aucun resultat.json")
    agg = d.groupby(["modele", "bras"]).agg(n=("seed", "count"), mcc_eval=("mcc_eval", "mean"), mcc_eval_sd=("mcc_eval", "std"),
                                            perf_eval=("perf_eval", "mean"), rappel=("rappel_split", "mean"),
                                            mcc_ref=("mcc_ref", "mean"), mcc_ref_sd=("mcc_ref", "std"),
                                            perf_ref=("perf_ref", "mean"), perf_ref_sd=("perf_ref", "std")).reset_index()
    comp = []
    for mod, g in d.groupby("modele"):
        P = {b: g[g.bras == b].set_index("seed") for b in BRAS}
        for x, y in (("extension", "affinage"), ("affinage", "fige"), ("extension", "fige")):
            s = P[x].index.intersection(P[y].index)
            de = P[x].loc[s, "perf_eval"] - P[y].loc[s, "perf_eval"]
            dr = P[x].loc[s, "perf_ref"] - P[y].loc[s, "perf_ref"]
            dsp = P[x].loc[s, "rappel_split"] - P[y].loc[s, "rappel_split"]
            comp.append({"modele": mod, "comparaison": f"{x} - {y}", "n": len(s),
                         "ecart_eval": de.mean(), "sd_eval": de.std(ddof=1), "p_eval": stats.ttest_rel(P[x].loc[s, "perf_eval"], P[y].loc[s, "perf_eval"]).pvalue,
                         "ecart_ref": dr.mean(), "sd_ref": dr.std(ddof=1), "ecart_rappel_split": dsp.mean(),
                         "sd_ref_entre_graines": P[y].loc[s, "perf_ref"].std(ddof=1)})
    c = pd.DataFrame(comp)
    c["p_eval_holm"] = np.nan
    for comp_nom, g in c.groupby("comparaison"):
        c.loc[g.index, "p_eval_holm"] = holm(g.p_eval.to_numpy())
    regle = []
    for _, r in c[c.comparaison == "extension - affinage"].iterrows():
        utile = bool(r.ecart_eval > 0 and r.p_eval_holm < 0.05 and r.ecart_ref > -r.sd_ref_entre_graines)
        regle.append((r.modele, utile))
    out = os.path.join(ICI, "analyse_banc", "derive")
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "derive_runs.csv"), index=False)
    c.to_csv(os.path.join(out, "derive_comparaisons.csv"), index=False)
    md = ["# Expérience de dérive", "", f"{len(d)} runs ({d.modele.nunique()} modèles x {d.bras.nunique()} bras x graines). "
          f"Régions en dérive détectées par run : {int(d.n_regions.min())} à {int(d.n_regions.max())}.", "",
          "| Modèle | Bras | n | perf D_eval | MCC D_eval | Rappel Split Deposit | perf D_ref | MCC D_ref |", "|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        md.append(f"| {JOLI.get(r.modele, r.modele)} | {NOM_BRAS[r.bras]} | {r.n} | {fr(r.perf_eval)} | {fr(r.mcc_eval)} ± {fr(r.mcc_eval_sd)} | "
                  f"{fr(r.rappel)} | {fr(r.perf_ref)} | {fr(r.mcc_ref)} ± {fr(r.mcc_ref_sd)} |")
    md += ["", "| Modèle | Comparaison | Écart perf D_eval | p (Holm) | Écart rappel Split | Écart perf D_ref | É.-t. D_ref entre graines |",
           "|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        md.append(f"| {JOLI.get(r.modele, r.modele)} | {r.comparaison} | {fr(r.ecart_eval, 5)} ± {fr(r.sd_eval, 5)} | {fr(r.p_eval_holm, 3)} | "
                  f"{fr(r.ecart_rappel_split)} | {fr(r.ecart_ref, 5)} | {fr(r.sd_ref_entre_graines, 5)} |")
    md += ["", "Règle de décision (extension utile) : " + ", ".join(f"{JOLI.get(m, m)} : {'oui' if u else 'non'}" for m, u in regle)]
    with open(os.path.join(out, "ANALYSE_DERIVE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    tex = []
    for mod in agg.modele.unique():
        for b in BRAS:
            r = agg[(agg.modele == mod) & (agg.bras == b)]
            if r.empty:
                continue
            r = r.iloc[0]
            tex.append(" & ".join([JOLI.get(mod, mod) if b == "fige" else "", NOM_BRAS[b], f"{fr(r.mcc_eval)} $\\pm$ {fr(r.mcc_eval_sd)}",
                                   fr(r.rappel), fr(r.perf_eval), f"{fr(r.mcc_ref)} $\\pm$ {fr(r.mcc_ref_sd)}"]) + " \\\\")
        tex.append("\\midrule")
    tex = tex[:-1]
    with open(os.path.join(RACINE, "memoire", "tableaux_banc", "tab_derive.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                "\\caption{Expérience de dérive : détection sur le jeu dérivé (MCC, rappel du Split Deposit, score perf) et sur la "
                "référence tirée de la validation (MCC), moyenne $\\pm$ écart-type sur cinq graines, pour chaque bras}\n"
                "\\label{tab:derive}\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
                "\\textbf{Modèle} & \\textbf{Bras} & \\textbf{MCC dérivé} & \\textbf{Rappel Split Deposit} & \\textbf{perf dérivé} & "
                "\\textbf{MCC référence} \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
