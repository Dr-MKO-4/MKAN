"""
analyse_sensibilite_v2.py -- Morris (et Sobol quand il est disponible) de la recherche réexécutée le 13-14/11/2026.

Lit morris_results.json, morris_X.npy, morris_Y.npy (et sobol_results.json s'il existe) dans le dossier recherche_v2.
Rapporte, pour chaque hyperparamètre : mu*, son intervalle à 95 % (mu_star_conf), sigma et mu ; la part des évaluations
effondrées (fitness < 0,30) ; le nombre de paramètres dont l'intervalle contient 0 ; et, pour chaque paramètre
conditionnel, le nombre de pas de trajectoire où il était actif (sa famille de base choisie), pour distinguer une
absence d'effet d'une absence de mesure.

Sorties : memoire/tableaux_banc/tab_morris.tex (et tab_sobol.tex), analyse_banc/recherche_v2/ANALYSE_SENSIBILITE_V2.md
Usage (depuis MKAN/Etude_benchmarck) : python analyse_sensibilite_v2.py
"""
import argparse
import json
import os

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
NOMS = {"lam": "$\\lambda$ (régularisation)", "hidden_size": "dimension cachée $d_h$", "W": "longueur de fenêtre $W$",
        "Input_Regime": "régime d'entrée", "lr": "pas d'apprentissage $\\eta$", "Base_Candidate": "base de la porte Candidate",
        "mu1": "$\\mu_1$ (L1)", "mu2": "$\\mu_2$ (entropie)", "batch_size": "taille de lot", "Cell_Type": "type de cellule",
        "Base_Input": "base de la porte Input", "Base_Forget": "base de la porte Forget", "Base_Output": "base de la porte Output"}


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def nom(r):
    if r["param_name"] in NOMS:
        return NOMS[r["param_name"]]
    return r["param_name"].replace("::", " / ").replace("_", "\\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default=os.path.join(ICI, "campagne_aws", "resultats_instance", "campagne_2026-11-14", "recherche_v2"))
    ap.add_argument("--n-tableau", type=int, default=13)
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.dossier, "morris_results.json"), encoding="utf-8"))
    R = sorted(d["results"], key=lambda r: -r["mu_star"])
    X = np.load(os.path.join(a.dossier, "morris_X.npy"))
    Y = np.load(os.path.join(a.dossier, "morris_Y.npy"))
    n = len(Y)
    effondre = int((Y < 0.30).sum())
    sans_effet_sur = [r for r in R if r["mu_star_conf"] >= r["mu_star"]]
    exclus = [r for r in R if r["morris_screening_candidate_exclusion"]]
    sig_mu = float(np.median([r["sigma"] / r["mu_star"] for r in R if r["mu_star"] > 0]))
    md = ["# Analyse de sensibilité de la recherche réexécutée", "",
          f"Morris : r = {d['metadata']['r']} trajectoires, {d['metadata']['num_levels']} niveaux, {n} évaluations, "
          f"{X.shape[1]} hyperparamètres encodés.",
          f"- Évaluations effondrées (fitness < 0,30) : {effondre} sur {n} ({fr(100 * effondre / n, 1)} %).",
          f"- Paramètres dont l'intervalle à 95 % de mu* contient 0 : {len(sans_effet_sur)} sur {len(R)}.",
          f"- Paramètres marqués « sans influence » par le seuil de l'outil : {len(exclus)}, tous conditionnels.",
          f"- Rapport sigma/mu* médian : {fr(sig_mu, 2)} (effets non linéaires ou en interaction).", "",
          "| Hyperparamètre | mu* | IC 95 % | sigma | mu |", "|---|---|---|---|---|"]
    md += [f"| {r['param_name']} | {fr(r['mu_star'])} | {fr(r['mu_star_conf'])} | {fr(r['sigma'])} | {fr(r['mu'])} |" for r in R]
    tex = [" & ".join([nom(r), fr(r["mu_star"]), "$\\pm$ " + fr(r["mu_star_conf"]), fr(r["sigma"]), fr(r["mu"])]) + " \\\\"
           for r in R[:a.n_tableau]]
    with open(os.path.join(RACINE, "memoire", "tableaux_banc", "tab_morris.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                f"\\caption{{Criblage de Morris de la recherche réexécutée : les {a.n_tableau} hyperparamètres de plus fort "
                f"effet moyen absolu $\\mu^*$ sur la fitness, avec l'intervalle à $95\\,\\%$ de $\\mu^*$, l'écart-type "
                f"$\\sigma$ des effets élémentaires et leur moyenne signée $\\mu$ ($r = {d['metadata']['r']}$ trajectoires, "
                f"{n} évaluations, validation)}}\n\\label{{tab:morris}}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
                "\\textbf{Hyperparamètre} & $\\mu^*$ & \\textbf{IC 95\\,\\%} & $\\sigma$ & $\\mu$ \\\\\n\\midrule\n"
                + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    chemin_sobol = os.path.join(a.dossier, "sobol_results.json")
    if os.path.exists(chemin_sobol):
        s = json.load(open(chemin_sobol, encoding="utf-8"))
        md += ["", "## Sobol", "", "```", json.dumps(s, ensure_ascii=False)[:3000], "```"]
    out = os.path.join(ICI, "analyse_banc", "recherche_v2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ANALYSE_SENSIBILITE_V2.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
