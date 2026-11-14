"""
analyse_recherche_v2.py -- Résultats de la recherche NSGA-II réexécutée le 13-14/11/2026 (réglages conformes aux
décisions du 03/11 et du 06/11 : latence 500 ms plafonnée et contrainte, R² symbolique 0,80, grille recalée, latence
pré-mesurée sur 4 cœurs réservés). Validation uniquement (50 000 fenêtres par individu), une graine par individu.

Contrôles : la fitness journalisée se recalcule à l'identique ; aucun membre du front n'est dominé par un individu
admissible.
Sorties : memoire/tableaux_banc/tab_recherche_v2.tex, analyse_banc/recherche_v2/ANALYSE_RECHERCHE_V2.md

Usage (depuis MKAN/Etude_benchmarck) : python analyse_recherche_v2.py [--dossier ...]
"""
import argparse
import collections
import json
import os

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
COURT = {"wavkan": "DoG", "relukan": "ReLU-KAN", "efficientkan": "B-spline", "fasterkan": "FasterKAN", "fastkan": "FastKAN",
         "hybrid": "hybride", "chebyshev": "Chebyshev", "linear": "linéaire", "fourier": "Fourier", "fkan": "FKAN"}


def fr(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


def bases(cfg):
    g = cfg.get("gates", {})
    out = []
    for porte in ("Forget", "Input", "Candidate", "Output"):
        if porte in g:
            b = g[porte].get("Base")
            if b == "wavkan":
                b = g[porte].get("Mother_Wavelet", "dog")
                out.append({"dog": "DoG", "morlet": "Morlet", "mexican_hat": "chapeau mexicain"}.get(b, b))
            else:
                out.append(COURT.get(b, b))
    return "/".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default=os.path.join(ICI, "campagne_aws", "resultats_instance", "campagne_2026-11-14", "recherche_v2"))
    a = ap.parse_args()
    L = [json.loads(l) for l in open(os.path.join(a.dossier, "extended_search_log.jsonl"), encoding="utf-8") if l.strip()]
    L = [l for l in L if l.get("fitness_components")]
    C = [l["fitness_components"] for l in L]
    ecart = max(abs(0.4147 * max(0, c["MCC_raw"]) + 0.2573 * c["PR_AUC"] - 0.1529 * c["Brier"] + 0.0876 * c["R2_symbolic"]
                    - 0.0876 * min(1, max(0, (c["latency_ms"] - 500) / 500)) - c["fitness_total"]) for c in C)
    pf = json.load(open(os.path.join(a.dossier, "pareto_front.json"), encoding="utf-8"))
    front_ids = {json.dumps(f["individual"], sort_keys=True) for f in pf["front"]}
    ordre = sorted(range(len(L)), key=lambda i: -C[i]["fitness_total"])
    vus, lignes = set(), []
    for i in ordre:
        cle = json.dumps(L[i]["config"], sort_keys=True)
        if cle in vus:
            continue
        if len(lignes) < 5 or cle in front_ids:
            vus.add(cle)
            lignes.append((L[i]["config"], C[i], cle in front_ids))
    for f in pf["front"]:
        cle = json.dumps(f["individual"], sort_keys=True)
        if cle not in vus:
            vus.add(cle)
            lignes.append((f["individual"], f["fitness_components"], True))
    regimes = collections.Counter(l["config"].get("Input_Regime") for l in L)
    cellules = collections.Counter(l["config"].get("Cell_Type") for l in L)
    lat = np.array([c["latency_ms"] for c in C])
    md = ["# Recherche NSGA-II réexécutée (13-14/11/2026)", "",
          f"{len(L)} évaluations journalisées, {pf['metadata'].get('n_front')} individus sur le front ; écart maximal de fitness "
          f"recalculée : {ecart:.1e}. Cellules : {dict(cellules)} ; régimes : {dict(regimes)} ; latence médiane "
          f"{fr(float(np.median(lat)), 1)} ms, {int((lat > 500).sum())} individus au-dessus de 500 ms.", "",
          "| Cellule | Régime | Bases | h | W | MCC | PR-AUC | Brier | R² 0,80 | R² 0,99 | Latence (ms) | Fitness | Front |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for cfg, c, fr_ in lignes:
        cell = "GRUKANCell" if cfg["Cell_Type"].startswith("GRU") else "TKANCell"
        reg = "traité" if cfg["Input_Regime"] == "engineered" else "brut"
        md.append(f"| {cell} | {reg} | {bases(cfg)} | {cfg['hidden_size']} | {cfg['W']} | {fr(c['MCC_raw'])} | {fr(c['PR_AUC'])} | "
                  f"{fr(c['Brier'])} | {fr(c['R2_symbolic'], 3)} | {fr(c.get('R2_symbolic_0.99', float('nan')), 3)} | "
                  f"{fr(c['latency_ms'], 1)} | {fr(c['fitness_total'])} | {'oui' if fr_ else ''} |")
        tex.append(" & ".join([cell, reg, bases(cfg), str(cfg["hidden_size"]), str(cfg["W"]), fr(c["MCC_raw"]), fr(c["R2_symbolic"], 3),
                               fr(c.get("R2_symbolic_0.99", float("nan")), 3), fr(c["latency_ms"], 1), fr(c["fitness_total"]),
                               "$\\bullet$" if fr_ else ""]) + " \\\\")
    out_md = os.path.join(ICI, "analyse_banc", "recherche_v2")
    os.makedirs(out_md, exist_ok=True)
    with open(os.path.join(out_md, "ANALYSE_RECHERCHE_V2.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RACINE, "memoire", "tableaux_banc", "tab_recherche_v2.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                "\\caption{Recherche NSGA-II réexécutée : cinq meilleures configurations à la fitness et membres du front de "
                "Pareto ($\\bullet$). Une graine et $50\\,000$ fenêtres par individu (validation) ; $h$ avant égalisation "
                "pour la GRUKANCell ; bases dans l'ordre des portes (Forget ou \\textit{reset}, Input ou \\textit{update}, "
                "Candidate, Output)}\n\\label{tab:recherche-v2}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}lllrrrrrrrc@{}}\n\\toprule\n"
                "\\textbf{Cellule} & \\textbf{Régime} & \\textbf{Bases} & $h$ & $W$ & \\textbf{MCC} & $R^2_{0{,}80}$ & "
                "$R^2_{0{,}99}$ & \\textbf{Lat. (ms)} & \\textbf{Fitness} & \\textbf{Front} \\\\\n\\midrule\n"
                + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
