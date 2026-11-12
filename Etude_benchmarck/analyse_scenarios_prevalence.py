"""
analyse_scenarios_prevalence.py -- Analyses sans reentrainement, sur la VALIDATION uniquement (le test n'est pas lu).

Audit du 12/11/2026 (contextes_sessions/AUDIT_TRAVAUX_2026-11-12.md, points 2.1 et 2.6). Trois analyses :

  1. Rappel par scenario de fraude (ATO, REFUND, FAKE_CRED, SPLIT_DEP, SMURFING) de chaque finaliste et de chaque
     modele de reference, au seuil de decision 0,5, a partir des scores deja sauvegardes (val_scores.npy).
  2. Precision attendue a une proportion de fraude realiste. Le rappel (TPR) et le taux de faux positifs (FPR) ne
     dependent pas de la proportion de fraude ; la precision, si. Pour une proportion pi :
         precision(pi) = pi * TPR / (pi * TPR + (1 - pi) * FPR)
     et le nombre d'alertes pour un million de transactions = 1e6 * (pi * TPR + (1 - pi) * FPR).
  3. Base unique contre bases mixtes, sur les GRUKANCell du plan de croisement en regime brut : ecart apparie par
     graine de chaque architecture mixte a la GRU DoG/DoG/DoG (seule architecture a base unique du plan), p corriges
     par la methode de Holm.

Controle d'integrite avant tout calcul : les fenetres de validation sont reconstruites dans l'ordre de prepare_data
(tri stable par compte puis pas, toutes les fenetres completes) ; pour chaque entrainement, le nombre de scores doit
etre egal au nombre de fenetres et le MCC recalcule doit retomber sur celui de result.json. Un entrainement qui echoue
a ce controle est ecarte et signale.

Usage (a lancer par Miguel, depuis MKAN/Etude_benchmarck) :
    python analyse_scenarios_prevalence.py
    python analyse_scenarios_prevalence.py --final campagne_aws/resultats_instance/final_2026-11-08 \
        --etape4 campagne_aws/resultats_instance/etape4_2026-11-06 --out analyse_banc/final_2026-11-08

Sorties (dans --out) : scenarios_par_run.csv, scenarios_par_modele.csv, prevalence_par_modele.csv,
base_unique_contre_mixtes.csv, ANALYSE_SCENARIOS.md, figures/scenarios_rappel.png (si kaleido est installe).
Tableaux LaTeX : memoire/tableaux_banc/tab_scenarios_rappel.tex, tab_prevalence.tex, tab_base_unique.tex.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
VAL_PATH = os.path.join(RACINE, "data", "val_features.parquet")
ACCOUNT_COL, TIME_COL, TARGET_COL, SCEN_COL = "nameOrig", "step", "isFraud", "fraudScenario"
SCENARIOS = ["ATO", "SMURFING", "SPLIT_DEP", "REFUND", "FAKE_CRED"]
NOMS_SCEN = {"ATO": "Account Takeover", "SMURFING": "Smurfing", "SPLIT_DEP": "Split Deposit",
             "REFUND": "Refund Fraud", "FAKE_CRED": "Fake Credentials"}
PREVALENCES = [0.001, 0.01, 0.05]
SEUIL = 0.5
UNIQUE = "Candidate=wavkan_dog/Forget=wavkan_dog/Input=wavkan_dog"


def fr(x, nd=4):
    """Nombre au format francais (virgule decimale)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.d."
    return f"{x:.{nd}f}".replace(".", ",")


def fenetres_validation(W):
    """Etiquette et scenario de la derniere transaction de chaque fenetre complete de longueur W, dans l'ordre de
    prepare_data.prepare (niveau1_harness.prepare_window_arrays + sample_windows_from_arrays sans tirage)."""
    df = pd.read_parquet(VAL_PATH, engine="fastparquet", columns=[ACCOUNT_COL, TIME_COL, TARGET_COL, SCEN_COL])
    df = df.sort_values([ACCOUNT_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    y = df[TARGET_COL].to_numpy().astype(np.int8)
    scen = df[SCEN_COL].astype(object).where(df[SCEN_COL].notna(), "LEGITIME").to_numpy()
    acct = df[ACCOUNT_COL].to_numpy()
    _, starts = np.unique(acct, return_index=True)
    ends = np.append(starts[1:], len(acct))
    n_win = np.maximum(ends - starts - W + 1, 0)
    total = int(n_win.sum())
    g = np.arange(total)
    cum = np.cumsum(n_win)
    acc_idx = np.searchsorted(cum, g, side="right")
    dernier = starts[acc_idx] + (g - (cum[acc_idx] - n_win[acc_idx])) + W - 1
    return y[dernier], scen[dernier]


def mcc(y, p):
    tp = float(((y == 1) & (p == 1)).sum()); tn = float(((y == 0) & (p == 0)).sum())
    fp = float(((y == 0) & (p == 1)).sum()); fn = float(((y == 1) & (p == 0)).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def modele(label):
    """Libelle sans la graine (« ... s3 » -> « ... »)."""
    return re.sub(r"\s+s\d+$", "", label)


def joli(m):
    """Libelle lisible d'un modele : « baseline xgboost traite W=5 » -> « XGBoost, traité »."""
    reg = "traité" if "traite" in m else "brut"
    for cle, nom in (("xgboost", "XGBoost"), ("lstm", "LSTM"), ("mlp", "MLP")):
        if cle in m:
            return f"{nom}, {reg}"
    courts = {"dog": "DoG", "relu": "ReLU-KAN", "bspl": "B-spline", "hyb": "hybride", "lin": "linéaire", "morl": "Morlet"}
    bases = [courts.get(x.split("=")[1], x.split("=")[1]) for x in m.split() if "=" in x]
    cellule = "GRUKANCell" if m.startswith("GRU") else "TKANCell"
    return f"{cellule} {'/'.join(bases)}, {reg}"


def pv(x):
    return "< 0,001" if x < 0.001 else fr(x, 3)


def holm(p):
    p = np.asarray(p, dtype=float)
    ordre = np.argsort(p)
    adj = np.empty_like(p)
    courant = 0.0
    for rang, i in enumerate(ordre):
        courant = max(courant, (len(p) - rang) * p[i])
        adj[i] = min(1.0, courant)
    return adj


def analyse_scores(final_dir, tolerance):
    runs = pd.read_csv(os.path.join(final_dir, "runs.csv"))
    runs = runs[runs.phase.isin(["final", "exp_baselines"])]
    finalistes_path = os.path.join(final_dir, "finalistes_4.json")      # 4 finalistes (A traite retire le 09/11)
    if not os.path.exists(finalistes_path):
        finalistes_path = os.path.join(final_dir, "finalistes.json")
    gardes = None
    if os.path.exists(finalistes_path):
        with open(finalistes_path, encoding="utf-8") as f:
            gardes = json.load(f)
    cache, lignes, ecartes = {}, [], []
    for _, r in runs.iterrows():
        d = os.path.join(final_dir, "runs", r.run_id)
        chemin = os.path.join(d, "val_scores.npy")
        if not os.path.exists(chemin):
            ecartes.append((r.label, "val_scores.npy absent"))
            continue
        with open(os.path.join(d, "spec.json"), encoding="utf-8") as f:
            spec = json.load(f)
        if r.phase == "final" and gardes is not None:
            cles = [(a["cell_type"], a["regime"], json.dumps(a["base_by_gate"], sort_keys=True)) for a in gardes]
            cle = (spec["cell_type"], spec["regime"], json.dumps(spec.get("base_by_gate", {}), sort_keys=True))
            if cle not in cles:
                continue                      # architecture retiree de l'analyse (decision du 09/11)
        W = int(spec["W"])
        if W not in cache:
            print(f"Reconstruction des fenetres de validation, W = {W} ...", flush=True)
            cache[W] = fenetres_validation(W)
        y, scen = cache[W]
        s = np.load(chemin)
        if len(s) != len(y):
            ecartes.append((r.label, f"{len(s)} scores pour {len(y)} fenetres"))
            continue
        p = (s >= SEUIL).astype(np.int8)
        m = mcc(y, p)
        if abs(m - float(r.mcc)) > tolerance:
            ecartes.append((r.label, f"MCC recalcule {m:.6f} contre {float(r.mcc):.6f} dans runs.csv"))
            continue
        pos, neg = y == 1, y == 0
        tpr = float(p[pos].mean()); fpr = float(p[neg].mean())
        ligne = {"run_id": r.run_id, "phase": r.phase, "label": r.label, "modele": modele(r.label), "seed": int(r.seed),
                 "regime": spec["regime"], "W": W, "mcc": m, "rappel": tpr, "taux_faux_positifs": fpr,
                 "precision_observee": float(y[p == 1].mean()) if p.any() else float("nan"),
                 "prevalence_observee": float(pos.mean())}
        for sc in SCENARIOS:
            masque = scen == sc
            ligne[f"n_{sc}"] = int(masque.sum())
            ligne[f"rappel_{sc}"] = float(p[masque].mean()) if masque.any() else float("nan")
        for pi in PREVALENCES:
            alertes = pi * tpr + (1 - pi) * fpr
            ligne[f"precision_{pi}"] = pi * tpr / alertes if alertes > 0 else float("nan")
            ligne[f"alertes_par_million_{pi}"] = 1e6 * alertes
            ligne[f"fausses_alertes_par_million_{pi}"] = 1e6 * (1 - pi) * fpr
        lignes.append(ligne)
        print(f"  {r.label} : rappel {tpr:.4f}, faux positifs {fpr:.5f}", flush=True)
    return pd.DataFrame(lignes), ecartes


def base_unique(etape4_dir):
    from scipy import stats
    d = pd.read_csv(os.path.join(etape4_dir, "runs.csv"))
    d = d[(d.phase == "etape4_brut") & (d.cell_type == "gru")].copy()
    d["perf"] = 0.4147 * d.mcc.clip(lower=0) + 0.2573 * d.auc_pr - 0.1529 * d.brier
    perf = d.pivot(index="seed", columns="architecture", values="perf")
    r2 = d.pivot(index="seed", columns="architecture", values="r2_symbolic_0.80")
    if UNIQUE not in perf.columns:
        return pd.DataFrame()
    lignes = []
    for c in perf.columns:
        if c == UNIQUE:
            continue
        dp, dr = perf[c] - perf[UNIQUE], r2[c] - r2[UNIQUE]
        lignes.append({"architecture": c, "n": int(dp.notna().sum()),
                       "ecart_perf": dp.mean(), "sd_perf": dp.std(ddof=1), "p_perf": stats.ttest_rel(perf[c], perf[UNIQUE]).pvalue,
                       "ecart_r2": dr.mean(), "sd_r2": dr.std(ddof=1), "p_r2": stats.ttest_rel(r2[c], r2[UNIQUE]).pvalue,
                       "perf_mixte": perf[c].mean(), "perf_unique": perf[UNIQUE].mean(),
                       "r2_mixte": r2[c].mean(), "r2_unique": r2[UNIQUE].mean()})
    out = pd.DataFrame(lignes)
    out["p_perf_holm"] = holm(out.p_perf.to_numpy())
    out["p_r2_holm"] = holm(out.p_r2.to_numpy())
    return out.sort_values("ecart_perf", ascending=False)


def nom_archi(a):
    courts = {"wavkan_dog": "DoG", "wavkan_morlet": "Morlet", "relukan": "ReLU-KAN", "efficientkan": "B-spline",
              "linear": "linéaire", "hybrid": "hybride"}
    portes = dict(x.split("=") for x in a.split("/"))
    return "/".join(courts.get(portes[k], portes[k]) for k in ("Forget", "Input", "Candidate", "Output") if k in portes)


def tableau(chemin, legende, etiquette, entetes, lignes, colonnes):
    with open(chemin, "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n")
        f.write(f"\\caption{{{legende}}}\n\\label{{{etiquette}}}\n\\begin{{adjustbox}}{{max width=\\textwidth}}\n")
        f.write(f"\\begin{{tabular}}{{@{{}}{colonnes}@{{}}}}\n\\toprule\n")
        f.write(" & ".join(entetes) + " \\\\\n\\midrule\n")
        for l in lignes:
            f.write(" & ".join(l) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = os.path.join(ICI, "campagne_aws", "resultats_instance")
    ap.add_argument("--final", default=os.path.join(base, "final_2026-11-08"))
    ap.add_argument("--etape4", default=os.path.join(base, "etape4_2026-11-06"))
    ap.add_argument("--out", default=os.path.join(ICI, "analyse_banc", "final_2026-11-08"))
    ap.add_argument("--tex", default=os.path.join(RACINE, "memoire", "tableaux_banc"))
    ap.add_argument("--tolerance", type=float, default=1e-4, help="ecart de MCC admis au controle d'integrite")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    par_run, ecartes = analyse_scores(a.final, a.tolerance)
    if par_run.empty:
        raise SystemExit("Aucun entrainement n'a passe le controle d'integrite : " + str(ecartes))
    par_run.to_csv(os.path.join(a.out, "scenarios_par_run.csv"), index=False)

    cols_scen = [f"rappel_{s}" for s in SCENARIOS]
    cols_prev = [c for c in par_run.columns if c.startswith(("precision_0", "alertes_par", "fausses_alertes"))]
    num = ["mcc", "rappel", "taux_faux_positifs", "precision_observee"] + cols_scen + cols_prev
    g = par_run.groupby(["modele", "regime", "W"])
    moy = g[num].mean().add_suffix("_moy"); sd = g[num].std(ddof=1).add_suffix("_sd")
    par_modele = pd.concat([moy, sd, g.size().rename("n_graines")], axis=1).reset_index()
    par_modele = par_modele.sort_values(["W", "regime", "mcc_moy"], ascending=[True, False, False])
    par_modele.to_csv(os.path.join(a.out, "scenarios_par_modele.csv"), index=False)
    par_modele[["modele", "regime", "W", "n_graines"] + [c + "_moy" for c in ["rappel", "taux_faux_positifs"] + cols_prev]] \
        .to_csv(os.path.join(a.out, "prevalence_par_modele.csv"), index=False)

    bu = base_unique(a.etape4)
    bu.to_csv(os.path.join(a.out, "base_unique_contre_mixtes.csv"), index=False)

    effectifs = {s: int(par_run[f"n_{s}"].max()) for s in SCENARIOS}
    md = ["# Rappel par scénario, précision à prévalence réaliste, base unique contre bases mixtes", "",
          "Validation uniquement, seuil de décision 0,5. Le jeu de test n'a pas été lu.", "",
          f"Entraînements analysés : {len(par_run)}. Écartés par le contrôle d'intégrité : {len(ecartes)}."]
    md += [f"- écarté : {l} ({pourquoi})" for l, pourquoi in ecartes]
    md += ["", "## 1. Rappel par scénario (moyenne sur les graines)", "",
           "Effectifs de fenêtres par scénario (plus grande valeur entre W = 5 et W = 10) : "
           + ", ".join(f"{NOMS_SCEN[s]} {effectifs[s]}" for s in SCENARIOS) + ".", "",
           "| Modèle | Régime | W | " + " | ".join(NOMS_SCEN[s] for s in SCENARIOS) + " | Rappel global |",
           "|---|---|---|" + "---|" * (len(SCENARIOS) + 1)]
    for _, r in par_modele.iterrows():
        md.append(f"| {joli(r.modele)} | {r.regime} | {r.W} | " + " | ".join(fr(r[f"rappel_{s}_moy"]) for s in SCENARIOS)
                  + f" | {fr(r.rappel_moy)} |")
    md += ["", "Lecture : un rappel faible sur un scénario désigne la typologie que le modèle laisse passer. "
               "Fake Credentials ne compte que quelques centaines de fenêtres : son rappel est très incertain.", "",
           "## 2. Précision attendue selon la proportion de fraude", "",
           "| Modèle | W | Rappel | Taux de faux positifs | "
           + " | ".join(f"Précision à {fr(100 * pi, 1)} %" for pi in PREVALENCES) + " | "
           + " | ".join(f"Fausses alertes par million à {fr(100 * pi, 1)} %" for pi in PREVALENCES) + " |",
           "|---|---|---|---|" + "---|" * (2 * len(PREVALENCES))]
    for _, r in par_modele.iterrows():
        md.append(f"| {joli(r.modele)} | {r.W} | {fr(r.rappel_moy)} | {fr(r.taux_faux_positifs_moy, 5)} | "
                  + " | ".join(fr(r[f"precision_{pi}_moy"]) for pi in PREVALENCES) + " | "
                  + " | ".join(fr(r[f"fausses_alertes_par_million_{pi}_moy"], 0) for pi in PREVALENCES) + " |")
    md += ["", "Lecture : le rappel et le taux de faux positifs sont mesurés sur la validation (environ 20 % de fraude) "
               "et supposés inchangés ; seule la proportion de fraude varie. Cette hypothèse suppose que les fraudes "
               "et les transactions légitimes gardent la même distribution, ce que la simulation ne garantit pas.", "",
           "## 3. Base unique contre bases mixtes (GRUKANCell, régime brut, plan de croisement)", ""]
    if bu.empty:
        md.append("Architecture à base unique absente de runs.csv.")
    else:
        md += [f"Référence : GRU DoG/DoG/DoG (score perf {fr(bu.perf_unique.iloc[0])}, "
               f"R² symbolique {fr(bu.r2_unique.iloc[0], 3)}). Écarts appariés par graine, p corrigés par Holm.", "",
               "| Architecture mixte | Écart de score perf | p corrigé | Écart de R² symbolique | p corrigé |",
               "|---|---|---|---|---|"]
        for _, r in bu.iterrows():
            md.append(f"| {nom_archi(r.architecture)} | {fr(r.ecart_perf, 5)} ± {fr(r.sd_perf, 5)} | {pv(r.p_perf_holm)} | "
                      f"{fr(r.ecart_r2, 3)} ± {fr(r.sd_r2, 3)} | {pv(r.p_r2_holm)} |")
    with open(os.path.join(a.out, "ANALYSE_SCENARIOS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")

    os.makedirs(a.tex, exist_ok=True)
    tableau(os.path.join(a.tex, "tab_scenarios_rappel.tex"),
            "Rappel par scénario de fraude au seuil $0{,}5$, moyenne sur les graines (validation)", "tab:scenarios-rappel",
            ["\\textbf{Modèle}", "$W$"] + [f"\\textbf{{{NOMS_SCEN[s]}}}" for s in SCENARIOS] + ["\\textbf{Global}"],
            [[joli(r.modele), str(r.W)] + [fr(r[f"rappel_{s}_moy"]) for s in SCENARIOS] + [fr(r.rappel_moy)]
             for _, r in par_modele.iterrows()], "lr" + "r" * (len(SCENARIOS) + 1))
    tableau(os.path.join(a.tex, "tab_prevalence.tex"),
            "Précision attendue selon la proportion de fraude, à rappel et taux de faux positifs mesurés sur la validation",
            "tab:prevalence",
            ["\\textbf{Modèle}", "$W$", "\\textbf{Rappel}", "\\textbf{Faux positifs}"]
            + [f"\\textbf{{Précision à {fr(100 * pi, 1)}\\,\\%}}" for pi in PREVALENCES],
            [[r.modele.replace("_", "\\_"), str(r.W), fr(r.rappel_moy), fr(r.taux_faux_positifs_moy, 5)]
             + [fr(r[f"precision_{pi}_moy"]) for pi in PREVALENCES] for _, r in par_modele.iterrows()],
            "lrrr" + "r" * len(PREVALENCES))
    if not bu.empty:
        tableau(os.path.join(a.tex, "tab_base_unique.tex"),
                "GRUKANCell à bases mixtes contre la GRUKANCell à base unique DoG, régime brut : écarts appariés sur "
                "cinq graines, $p$ corrigés par la méthode de Holm (validation)", "tab:base-unique",
                ["\\textbf{Architecture mixte}", "\\textbf{Écart perf}", "$p$", "\\textbf{Écart} $R^2_{0{,}80}$", "$p$"],
                [[nom_archi(r.architecture), f"{fr(r.ecart_perf, 5)} $\\pm$ {fr(r.sd_perf, 5)}", pv(r.p_perf_holm).replace("<", "$<$"),
                  f"{fr(r.ecart_r2, 3)} $\\pm$ {fr(r.sd_r2, 3)}", pv(r.p_r2_holm).replace("<", "$<$")] for _, r in bu.iterrows()], "lrrrr")

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for _, r in par_modele.iterrows():
            fig.add_trace(go.Bar(name=f"{r.modele} (W={r.W})", x=[NOMS_SCEN[s] for s in SCENARIOS],
                                 y=[r[f"rappel_{s}_moy"] for s in SCENARIOS],
                                 error_y=dict(type="data", array=[r[f"rappel_{s}_sd"] for s in SCENARIOS])))
        fig.update_layout(barmode="group", template="simple_white", width=1400, height=650,
                          yaxis_title="Rappel au seuil 0,5 (validation)", legend=dict(orientation="h", y=-0.25),
                          font=dict(family="Segoe UI, Arial", size=14))
        fig.write_image(os.path.join(a.out, "figures", "scenarios_rappel.png"), scale=2)
        fig.write_image(os.path.join(RACINE, "memoire", "figure", "final_06_scenarios.png"), scale=2)
    except Exception as e:                                   # kaleido absent : les tableaux suffisent
        print("Figure non produite :", e)

    print(f"\nTermine : {len(par_run)} entrainements analyses, {len(ecartes)} ecartes. Rapport : "
          f"{os.path.join(a.out, 'ANALYSE_SCENARIOS.md')}")


if __name__ == "__main__":
    main()
