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
        from scipy import stats
        s = json.load(open(chemin_sobol, encoding="utf-8"))
        S = s["results"]
        noms = [r["param_name"] for r in S]
        XS = np.load(os.path.join(a.dossier, "sobol_X.npy"))
        YS = np.load(os.path.join(a.dossier, "sobol_Y.npy"))
        D = XS.shape[1]
        G = D + 2
        N = len(YS) // G
        Ya = YS[0::G]
        # parametre jamais modifie dans sa decision : la sortie ne change sur aucun groupe A -> AB_j
        jamais = [noms[j] for j in range(D) if np.all(np.abs(YS[1 + j::G] - Ya) < 1e-12)]
        bascule = {}
        for nm in ("Cell_Type", "Input_Regime"):
            if nm in noms:
                j = noms.index(nm)
                bascule[nm] = int(np.sum((XS[0::G, j] < 0.5) != (XS[G - 1::G, j] < 0.5)))
        sig = [r["param_name"] for r in S if r["S_Ti"] - r["S_Ti_conf_95"] > 0]
        mstar = {r["param_name"]: r["mu_star"] for r in R}
        commun = [n_ for n_ in noms if n_ in mstar]
        rho = stats.spearmanr([mstar[n_] for n_ in commun], [next(r["S_Ti"] for r in S if r["param_name"] == n_) for n_ in commun])
        Ss = sorted(S, key=lambda r: -r["S_Ti"])
        md += ["", "## Sobol", "",
               f"N = {N} points de base, {len(YS)} évaluations, second ordre non calculé.",
               f"- Évaluations effondrées (fitness < 0,30) : {int((YS < 0.30).sum())} sur {len(YS)}.",
               f"- Indices de premier ordre négatifs : {sum(r['S_i'] < 0 for r in S)} sur {D} ; somme des indices totaux : "
               f"{fr(sum(r['S_Ti'] for r in S), 2)}.",
               f"- Indices totaux dont la borne basse à 95 % dépasse 0 : {len(sig)} ({', '.join(sig)}).",
               f"- Paramètres dont la modification ne change jamais la sortie dans l'échantillon : {', '.join(jamais)}.",
               f"- Groupes où le paramètre binaire change de valeur entre A et B (sur {N}) : {bascule}.",
               f"- Corrélation de Spearman entre mu* de Morris et S_T de Sobol : {fr(rho.statistic, 2)} (p = {fr(rho.pvalue, 3)}).", "",
               "| Hyperparamètre | S_i | IC 95 % | S_Ti | IC 95 % | Part active |", "|---|---|---|---|---|---|"]
        md += [f"| {r['param_name']} | {fr(r['S_i'])} | {fr(r['S_i_conf_95'])} | {fr(r['S_Ti'])} | {fr(r['S_Ti_conf_95'])} | "
               f"{fr(r['P_active'], 2)} |" for r in Ss]
        texs = [" & ".join([nom(r), fr(r["S_i"]), "$\\pm$ " + fr(r["S_i_conf_95"]), fr(r["S_Ti"]), "$\\pm$ " + fr(r["S_Ti_conf_95"])])
                + " \\\\" for r in Ss[:a.n_tableau]]
        with open(os.path.join(RACINE, "memoire", "tableaux_banc", "tab_sobol.tex"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                    f"\\caption{{Indices de Sobol de la recherche réexécutée, à titre indicatif ($N = {N}$, {len(YS)} évaluations) : "
                    f"les {a.n_tableau} plus forts indices totaux $S_{{Ti}}$, avec l'indice de premier ordre $S_i$ et les "
                    "intervalles à $95\\,\\%$ (bootstrap, validation)}\n\\label{tab:sobol}\n"
                    "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
                    "\\textbf{Hyperparamètre} & $S_i$ & \\textbf{IC 95\\,\\%} & $S_{Ti}$ & \\textbf{IC 95\\,\\%} \\\\\n\\midrule\n"
                    + "\n".join(texs) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    out = os.path.join(ICI, "analyse_banc", "recherche_v2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ANALYSE_SENSIBILITE_V2.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
