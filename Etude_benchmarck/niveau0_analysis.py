"""
niveau0_analysis.py -- Analyse et matrice de qualification (Etape 3, Niveau 0,
module 3.2).

Lit les resultats bruts produits par niveau0_benchmark_iso.py (module 3.1) et
produit :
  - une agregation statistique sur les 5 graines par (base, famille de fonctions)
  - un classement par famille avec notation qualitative (++ / + / +- / - / --)
  - deux heatmaps Plotly (convention du projet, pas matplotlib) -- RMSE log-scale,
    I_Gibbs Regime 1 -- chacune ecrite en .html (interactif) et .png (via kaleido,
    pour insertion dans le memoire LaTeX)
  - un rapport de synthese texte : bases retenues/rejetees par porte, avec
    verification de coherence par rapport a l'Etape 2 (heuristic_best_config.json)

Aucun resultat n'est fabrique : si le fichier de resultats bruts est absent ou
incomplet, ce module le signale explicitement (via des avertissements) plutot
que d'inventer des valeurs.

Usage :
    python niveau0_analysis.py
    python niveau0_analysis.py --raw_path extended_search/niveau0_benchmark_results.json \\
                                --out_dir niveau0_analysis
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constantes du protocole (step_3_benchmark_synthetique.tex)
# ─────────────────────────────────────────────────────────────────────────────

GIBBS_THRESHOLD = 0.05
GRADE_THRESHOLDS = [
    (1.10, "++"),
    (1.50, "+"),
    (2.50, "+-"),   # "+-" represente le symbole "pm" (evite tout probleme d'encodage console)
    (5.00, "-"),
]
GRADE_BEYOND = "--"

GATE_TO_REGIME = {"Forget": 1, "Input": 1, "Candidate": 2, "Output": 3}
REGIME_FUNCTIONS = {1: ["fs1", "fs2", "fs3"], 2: ["fc1", "fc2", "fc3"], 3: ["fo1", "fo2"]}
REGIME_ORDER = ["fs1", "fs2", "fs3", "fc1", "fc2", "fc3", "fo1", "fo2"]

# Bases plebiscitees par l'Etape 2 (heuristic_best_config.json, configuration
# d'elite Rang 1) -- verification de coherence obligatoire.
ETAPE2_ELITE_BY_GATE = {
    "Forget": "relukan",
    "Input": "efficientkan",
    "Candidate": "wavkan_dog",
}

CRITERIA_LATENCY_FACTOR = 3.0
CRITERIA_T90_MAX = 2000

# fc1 (sin(400*pi*x), 200 oscillations) et fc3 (composante a 40*pi) sont exclues
# du critere T_90% -- constat empirique, pas une supposition : sur un run reel a
# n_iterations=10000 (5x le budget initial), les 45 essais (9 bases x 5 graines)
# sur fc1 sont TOUS restes a RMSE ~= 0.7067 (= RMS(sin(400*pi*x)) = 1/sqrt(2),
# la solution triviale "predire ~0 partout") et t90_median EXACTEMENT au
# plafond de censure (aucune progression, meme partielle). fc3 montre le meme
# mur pour 8 bases sur 9. C'est un mur de CAPACITE (B=12 parametres ne peuvent
# pas representer un contenu a si haute frequence -- argument de type
# Nyquist/echantillonnage), pas un probleme de vitesse de convergence : aucun
# nombre d'iterations supplementaires ne change ce resultat (deja verifie
# empiriquement, cf. session du 2026-09-22). RMSE/latence restent evalues
# normalement sur fc1/fc3 ; seul T_90% est exclu pour ces deux familles,
# conserve pour fc2 (Morlet, effectivement apprenable a ce budget).
T90_EXCLUDED_FUNCTIONS = {"fc1", "fc3"}


def _grade_from_ratio(ratio: float) -> str:
    for bound, grade in GRADE_THRESHOLDS:
        if ratio <= bound:
            return grade
    return GRADE_BEYOND


# ─────────────────────────────────────────────────────────────────────────────
# Chargement + agregation
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_results(raw_path: str) -> dict:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Fichier de resultats bruts introuvable : {raw_path}\n"
            f"Lancer d'abord niveau0_benchmark_iso.py pour le produire."
        )
    with open(raw_path, encoding="utf-8") as f:
        return json.load(f)


def build_records_df(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw["results"])
    if df.empty:
        raise ValueError("Le fichier de resultats bruts ne contient aucun enregistrement.")
    return df


def aggregate(df: pd.DataFrame, n_iterations: int) -> pd.DataFrame:
    """Agrege sur les graines pour chaque (base, function). Les runs non
    convergents (t90_iterations=None) sont censures a n_iterations (convention
    standard pour une metrique de type 'temps jusqu'a evenement' : on ne peut
    pas ignorer silencieusement ces essais sans biaiser la mediane vers le bas)."""
    rows = []
    for (base, function), grp in df.groupby(["base", "function"]):
        rmse = grp["rmse"].astype(float)
        gibbs_vals = grp["gibbs_index"].dropna().astype(float)
        t90_censored = grp["t90_iterations"].fillna(n_iterations).astype(float)
        n_converged = int(grp["converged_90pct"].sum())
        rows.append({
            "base": base,
            "function": function,
            "regime": int(grp["regime"].iloc[0]),
            "n_seeds": len(grp),
            "rmse_median": float(rmse.median()),
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std(ddof=0)),
            "rmse_min": float(rmse.min()),
            "rmse_max": float(rmse.max()),
            "gibbs_max": float(gibbs_vals.max()) if len(gibbs_vals) else None,
            "latency_mean_us": float(grp["latency_us"].mean()),
            "t90_median": float(t90_censored.median()),
            "n_converged": n_converged,
        })
    return pd.DataFrame(rows)


def rank_and_grade(agg: pd.DataFrame, gibbs_threshold: float = GIBBS_THRESHOLD) -> tuple:
    """Retourne (ranking_df, qualification_df) : classement par famille +
    notation qualitative avec penalite Gibbs absolue (Regime 1).

    gibbs_threshold : seuil du protocole (0.05 par defaut, step_3.tex). Parametre
    explicite plutot que constante figee : un assouplissement documente (ex. via
    --gibbs_threshold en CLI) doit se voir dans la signature de l'appel, jamais
    en editant silencieusement GIBBS_THRESHOLD en dur dans le code."""
    ranking_rows = []
    for function, grp in agg.groupby("function"):
        grp = grp.sort_values("rmse_median").reset_index(drop=True)
        rmse_min = grp["rmse_median"].iloc[0]
        for rank, row in grp.iterrows():
            ratio = row["rmse_median"] / rmse_min if rmse_min > 0 else float("inf")
            grade = _grade_from_ratio(ratio)
            ranking_rows.append({
                "function": function,
                "regime": row["regime"],
                "rank": rank + 1,
                "base": row["base"],
                "rmse_median": row["rmse_median"],
                "ratio_to_min": ratio,
                "grade_before_gibbs": grade,
            })
    ranking_df = pd.DataFrame(ranking_rows)

    # Penalite Gibbs absolue : une base qui depasse gibbs_threshold sur N'IMPORTE
    # QUELLE famille du Regime 1 recoit -- pour TOUTES les familles du Regime 1.
    gibbs_regime1 = agg[agg["regime"] == 1]
    gibbs_violation_bases = set(
        gibbs_regime1.loc[gibbs_regime1["gibbs_max"] > gibbs_threshold, "base"]
    )

    def _final_grade(row):
        if row["regime"] == 1 and row["base"] in gibbs_violation_bases:
            return GRADE_BEYOND
        return row["grade_before_gibbs"]

    ranking_df["grade"] = ranking_df.apply(_final_grade, axis=1)
    ranking_df["gibbs_penalty_applied"] = (
        (ranking_df["regime"] == 1) & ranking_df["base"].isin(gibbs_violation_bases)
    )

    qualification_df = ranking_df.pivot(index="base", columns="function", values="grade")
    ordered_cols = [c for c in REGIME_ORDER if c in qualification_df.columns]
    qualification_df = qualification_df[ordered_cols]

    return ranking_df, qualification_df, gibbs_violation_bases


# ─────────────────────────────────────────────────────────────────────────────
# Criteres de preselection formels (4 conditions simultanees, eq. criteres_presel)
# ─────────────────────────────────────────────────────────────────────────────

def preselect_by_gate(agg: pd.DataFrame, gibbs_violation_bases: set,
                       gibbs_threshold: float = GIBBS_THRESHOLD,
                       latency_factor: float = CRITERIA_LATENCY_FACTOR,
                       t90_max: int = CRITERIA_T90_MAX) -> dict:
    """gibbs_threshold/latency_factor/t90_max : seuils du protocole (step_3.tex),
    passes explicitement (jamais lus depuis une constante modifiee en dur) pour
    qu'un assouplissement documente (CLI) soit tracable de bout en bout jusque
    dans les justifications textuelles produites plus bas."""
    overall_latency = agg.groupby("base")["latency_mean_us"].mean()
    fastest_base = overall_latency.idxmin()
    fastest_latency = overall_latency.min()
    latency_limit = latency_factor * fastest_latency

    result = {}
    for gate, regime in GATE_TO_REGIME.items():
        functions = REGIME_FUNCTIONS[regime]
        sub = agg[agg["function"].isin(functions)]
        rmse_avg = sub.groupby("base")["rmse_median"].mean()

        # T_90% exclut fc1/fc3 (mur de capacite empirique, cf. T90_EXCLUDED_FUNCTIONS) --
        # RMSE/Gibbs/latence restent, eux, evalues sur TOUTES les familles du regime.
        t90_functions = [f for f in functions if f not in T90_EXCLUDED_FUNCTIONS]
        t90_excluded = [f for f in functions if f in T90_EXCLUDED_FUNCTIONS]
        if t90_functions:
            t90_sub = agg[agg["function"].isin(t90_functions)]
            t90_avg = t90_sub.groupby("base")["t90_median"].median().reindex(rmse_avg.index)
        else:
            # Toutes les familles du regime sont exclues (cas non rencontre avec la
            # configuration actuelle) : le critere T_90% ne peut pas s'appliquer,
            # traite comme toujours satisfait plutot que de planter sur un groupby vide.
            t90_avg = pd.Series(0.0, index=rmse_avg.index)

        per_base = pd.DataFrame({"rmse_avg": rmse_avg, "t90_avg": t90_avg})
        per_base["latency"] = overall_latency.reindex(per_base.index)
        q1 = per_base["rmse_avg"].quantile(0.25)

        bases_decisions = {}
        for base, row in per_base.iterrows():
            reasons_fail = []
            rmse_ok = row["rmse_avg"] <= q1
            if not rmse_ok:
                reasons_fail.append(
                    f"RMSE moyen ({row['rmse_avg']:.4g}) hors du 1er quartile "
                    f"(seuil Q1={q1:.4g})"
                )
            gibbs_ok = True
            if regime == 1 and base in gibbs_violation_bases:
                gibbs_ok = False
                reasons_fail.append(
                    f"I_Gibbs > {gibbs_threshold} sur au moins une famille du Regime 1"
                )
            latency_ok = row["latency"] <= latency_limit
            if not latency_ok:
                reasons_fail.append(
                    f"latence ({row['latency']:.4g} us) > {latency_factor}x "
                    f"la plus rapide ({fastest_base}={fastest_latency:.4g} us, "
                    f"seuil={latency_limit:.4g} us)"
                )
            t90_ok = row["t90_avg"] < t90_max
            if not t90_ok:
                t90_scope = (f"sur {'/'.join(t90_functions)} uniquement ({'/'.join(t90_excluded)} "
                             f"exclues : mur de capacite B=12 empirique, cf. T90_EXCLUDED_FUNCTIONS)"
                             if t90_excluded else f"sur {'/'.join(t90_functions)}")
                reasons_fail.append(
                    f"T_90% median ({row['t90_avg']:.0f}) >= {t90_max} iterations {t90_scope}"
                )
            retained = rmse_ok and gibbs_ok and latency_ok and t90_ok
            bases_decisions[base] = {
                "retained": bool(retained),
                "rmse_avg": float(row["rmse_avg"]),
                "t90_avg": float(row["t90_avg"]),
                "latency_us": float(row["latency"]),
                "reasons_fail": reasons_fail,
            }
        result[gate] = {
            "functions": functions,
            "q1_threshold_rmse": float(q1),
            "fastest_base": fastest_base,
            "fastest_latency_us": float(fastest_latency),
            "latency_limit_us": float(latency_limit),
            "gibbs_threshold": gibbs_threshold,
            "latency_factor": latency_factor,
            "t90_max": t90_max,
            "t90_functions": t90_functions,
            "t90_excluded_functions": t90_excluded,
            "bases": bases_decisions,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Heatmaps (Plotly -- convention du projet, pas matplotlib) : chaque figure est
# ecrite en .html (interactif, toujours possible) ET en .png (statique, via
# kaleido, pour insertion directe dans le memoire LaTeX). Meme convention que le
# notebook analyse_resultats_recherche.ipynb (save_fig()) -- si kaleido est
# indisponible, seul le .html est produit et un avertissement est imprime,
# jamais un echec silencieux ni un plantage du reste du pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, out_path_no_ext: str) -> None:
    os.makedirs(os.path.dirname(out_path_no_ext) or ".", exist_ok=True)
    html_path = out_path_no_ext + ".html"
    png_path = out_path_no_ext + ".png"
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, scale=2)
    except Exception as exc:   # kaleido peut manquer selon l'environnement
        print(f"[avertissement] export PNG impossible pour {out_path_no_ext} ({exc}) -- "
              f"HTML seul disponible ({html_path}).")


def plot_rmse_heatmap(agg: pd.DataFrame, out_path_no_ext: str) -> None:
    import plotly.graph_objects as go

    pivot = agg.pivot(index="base", columns="function", values="rmse_median")
    cols = [c for c in REGIME_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    bases = list(pivot.index)
    data = pivot.values.astype(float)
    data_safe = np.where(data <= 0, np.nextafter(0, 1), data)
    log_data = np.log10(data_safe)

    fig = go.Figure(go.Heatmap(
        z=log_data, x=cols, y=bases,
        colorscale="Viridis_r",
        text=[[f"{v:.3g}" for v in row] for row in data],
        texttemplate="%{text}", textfont={"size": 10},
        colorbar=dict(title="RMSE (log10)"),
        hovertemplate="base=%{y}<br>fonction=%{x}<br>RMSE=%{text}<extra></extra>",
    ))

    # Separateurs visuels entre les 3 regimes (fs*/fc*/fo*)
    prev_regime = None
    for j, c in enumerate(cols):
        regime = 1 if c.startswith("fs") else (2 if c.startswith("fc") else 3)
        if prev_regime is not None and regime != prev_regime:
            fig.add_shape(type="line", x0=j - 0.5, x1=j - 0.5, y0=-0.5, y1=len(bases) - 0.5,
                          line=dict(color="black", width=2))
        prev_regime = regime

    fig.update_layout(
        title="RMSE median (echelle log10) -- Benchmark synthetique iso-parametrique Niveau 0",
        xaxis_title="Fonction", yaxis_title="Base",
        yaxis=dict(autorange="reversed"),   # 1ere base en haut (convention matplotlib/imshow)
        template="plotly_white",
        height=max(350, 55 * len(bases) + 150), width=max(500, 130 * len(cols) + 200),
    )
    _save_fig(fig, out_path_no_ext)


def plot_gibbs_heatmap(agg: pd.DataFrame, out_path_no_ext: str,
                        gibbs_threshold: float = GIBBS_THRESHOLD) -> None:
    import plotly.graph_objects as go

    regime1_cols = REGIME_FUNCTIONS[1]
    sub = agg[agg["function"].isin(regime1_cols)]
    pivot = sub.pivot(index="base", columns="function", values="gibbs_max")
    pivot = pivot[[c for c in regime1_cols if c in pivot.columns]]
    bases = list(pivot.index)
    cols = list(pivot.columns)
    data = pivot.values.astype(float)

    vmax = max(float(np.nanmax(data)), gibbs_threshold * 1.5)
    vmin = min(float(np.nanmin(data)), 0.0)
    # Point milieu de l'echelle de couleur cale sur le seuil critique (equivalent
    # de TwoSlopeNorm(vcenter=gibbs_threshold)) -- proportion normalisee dans [0,1].
    mid = (gibbs_threshold - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    mid = min(max(mid, 0.0), 1.0)
    colorscale = [[0.0, "green"], [mid, "yellow"], [1.0, "red"]]

    fig = go.Figure(go.Heatmap(
        z=data, x=cols, y=bases, zmin=vmin, zmax=vmax, colorscale=colorscale,
        text=[[f"{v:.3g}" for v in row] for row in data],
        texttemplate="%{text}", textfont={"size": 11},
        colorbar=dict(title="I_Gibbs"),
        hovertemplate="base=%{y}<br>fonction=%{x}<br>I_Gibbs=%{text}<extra></extra>",
    ))

    # Cadre rouge autour des cellules qui depassent le seuil critique.
    for i, base in enumerate(bases):
        for j, col in enumerate(cols):
            if data[i, j] > gibbs_threshold:
                fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                              line=dict(color="red", width=3), fillcolor="rgba(0,0,0,0)")

    fig.update_layout(
        title=f"Indice de Gibbs I_Gibbs -- Regime 1 uniquement "
              f"(seuil critique = {gibbs_threshold}, cadre rouge = depassement)",
        xaxis_title="Fonction", yaxis_title="Base",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=max(350, 55 * len(bases) + 150), width=max(500, 160 * len(cols) + 200),
    )
    _save_fig(fig, out_path_no_ext)


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de synthese (texte)
# ─────────────────────────────────────────────────────────────────────────────

def write_report(preselection: dict, out_path: str, raw_metadata: dict) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("RAPPORT DE SYNTHESE -- Preselection des bases KAN par porte (Niveau 0)")
    lines.append("=" * 78)
    lines.append(f"Genere le : {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Base sur {raw_metadata.get('n_points')} points, "
                 f"{raw_metadata.get('n_iterations')} iterations, "
                 f"{len(raw_metadata.get('seeds', []))} graines.")
    any_gate = next(iter(preselection.values()), None)
    if any_gate is not None:
        lines.append(
            f"Seuils de preselection utilises pour ce rapport : "
            f"I_Gibbs <= {any_gate.get('gibbs_threshold', GIBBS_THRESHOLD)} ; "
            f"latence <= {any_gate.get('latency_factor', CRITERIA_LATENCY_FACTOR)}x "
            f"la plus rapide ; T_90% < {any_gate.get('t90_max', CRITERIA_T90_MAX)} "
            f"iterations. (Valeurs par defaut du protocole step_3.tex sauf si "
            f"surchargees explicitement via --gibbs_threshold/--latency_factor/--t90_max.)"
        )
    lines.append("")

    for gate, info in preselection.items():
        lines.append("-" * 78)
        lines.append(f"PORTE : {gate}  (Regime {GATE_TO_REGIME[gate]}, "
                     f"familles : {', '.join(info['functions'])})")
        lines.append("-" * 78)
        retained = [b for b, d in info["bases"].items() if d["retained"]]
        rejected = [b for b, d in info["bases"].items() if not d["retained"]]

        lines.append(f"Seuil RMSE (1er quartile) : {info['q1_threshold_rmse']:.4g}")
        lines.append(f"Base la plus rapide : {info['fastest_base']} "
                     f"({info['fastest_latency_us']:.4g} us) -- seuil latence "
                     f"({CRITERIA_LATENCY_FACTOR}x) = {info['latency_limit_us']:.4g} us")
        if info.get("t90_excluded_functions"):
            lines.append(
                f"[NOTE] Critere T_90% calcule sur {', '.join(info['t90_functions'])} "
                f"uniquement -- {', '.join(info['t90_excluded_functions'])} exclue(s) : "
                f"mur de capacite empirique (B=12 insuffisant pour representer ce contenu "
                f"haute frequence, confirme par RMSE ~= RMS(cible) et zero convergence sur "
                f"les 45 essais meme a 5x le budget d'iterations initial -- cf. "
                f"T90_EXCLUDED_FUNCTIONS dans le code). RMSE/latence restent, eux, evalues "
                f"sur toutes les familles du regime."
            )
        lines.append("")

        lines.append(f"Bases RETENUES ({len(retained)}) : {', '.join(sorted(retained)) or 'aucune'}")
        lines.append("")
        lines.append("Bases REJETEES :")
        if not rejected:
            lines.append("  (aucune)")
        for b in sorted(rejected):
            d = info["bases"][b]
            lines.append(f"  - {b} : " + " ; ".join(d["reasons_fail"]))
        lines.append("")

        # Verification de coherence Etape 2
        elite = ETAPE2_ELITE_BY_GATE.get(gate)
        if elite is not None:
            if elite in retained:
                lines.append(f"[OK] Coherence Etape 2 : la base plebiscitee ('{elite}') "
                             f"figure bien parmi les bases retenues.")
            else:
                d = info["bases"].get(elite)
                reason = " ; ".join(d["reasons_fail"]) if d else "base absente du benchmark Niveau 0"
                lines.append(f"[AVERTISSEMENT] Ecart Etape 2 / Niveau 0 : la base plebiscitee "
                             f"par l'Etape 2 ('{elite}') n'est PAS retenue par la preselection "
                             f"Niveau 0 sur la porte {gate}. Raison(s) : {reason}")
        else:
            lines.append("(Porte Output : applicable a la TKANCell (Rang 2) uniquement -- "
                         "la GRUKANCell d'elite (Rang 1) n'a pas de porte Output.)")
        lines.append("")

    lines.append("=" * 78)
    lines.append("LIMITE METHODOLOGIQUE")
    lines.append("=" * 78)
    lines.append(
        "Les portes Forget et Input partagent exactement les memes familles de "
        "fonctions synthetiques (Regime 1) : le benchmark Niveau 0, qui evalue "
        "les bases de facon ISOLEE (sans la dynamique recurrente complete), ne "
        "peut donc PAS les distinguer -- les deux portes recoivent le meme "
        "ensemble de bases retenues. La discrimination fine observee dans "
        "l'Etape 2 (relukan sur Forget, efficientkan sur Input, alors que les "
        "deux bases appartiennent au meme ensemble retenu ici) provient de la "
        "dynamique temporelle de la cellule recurrente complete, hors du perimetre "
        "de ce filtre Niveau 0. Ceci est attendu et documente : le role du "
        "Niveau 0 est de reduire l'espace de recherche combinatoire (filtre "
        "large), pas de se substituer a la recherche heuristique de l'Etape 2 "
        "(decision fine)."
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_path", type=str,
                         default=os.path.join("extended_search", "niveau0_benchmark_results.json"))
    parser.add_argument("--out_dir", type=str, default="niveau0_analysis")
    parser.add_argument("--gibbs_threshold", type=float, default=GIBBS_THRESHOLD,
                         help=f"Seuil I_Gibbs du protocole (defaut step_3.tex = {GIBBS_THRESHOLD}). "
                              f"Un assouplissement documente (ex. 0.10) doit passer par ce flag, "
                              f"jamais par une edition silencieuse de la constante dans le code.")
    parser.add_argument("--latency_factor", type=float, default=CRITERIA_LATENCY_FACTOR,
                         help=f"Facteur du critere de latence (defaut = {CRITERIA_LATENCY_FACTOR}x "
                              f"la base la plus rapide).")
    parser.add_argument("--t90_max", type=int, default=CRITERIA_T90_MAX,
                         help=f"Budget T_90% maximal en iterations (defaut = {CRITERIA_T90_MAX}, "
                              f"doit correspondre a --n_iterations du run niveau0_benchmark_iso.py "
                              f"sous-jacent pour rester coherent).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(project_root, args.raw_path)
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw = load_raw_results(raw_path)
    df = build_records_df(raw)
    n_iterations = raw["metadata"]["n_iterations"]

    agg = aggregate(df, n_iterations)
    ranking_df, qualification_df, gibbs_violation_bases = rank_and_grade(
        agg, gibbs_threshold=args.gibbs_threshold)
    preselection = preselect_by_gate(
        agg, gibbs_violation_bases, gibbs_threshold=args.gibbs_threshold,
        latency_factor=args.latency_factor, t90_max=args.t90_max)

    agg_path = os.path.join(out_dir, "niveau0_aggregation.csv")
    ranking_path = os.path.join(out_dir, "niveau0_ranking.csv")
    qualif_path = os.path.join(out_dir, "niveau0_qualification_matrix.csv")
    rmse_heatmap_stem = os.path.join(out_dir, "niveau0_heatmap_rmse")
    gibbs_heatmap_stem = os.path.join(out_dir, "niveau0_heatmap_gibbs")
    report_path = os.path.join(out_dir, "niveau0_rapport_synthese.txt")

    agg.to_csv(agg_path, index=False, encoding="utf-8")
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8")
    qualification_df.to_csv(qualif_path, encoding="utf-8")
    plot_rmse_heatmap(agg, rmse_heatmap_stem)
    plot_gibbs_heatmap(agg, gibbs_heatmap_stem, gibbs_threshold=args.gibbs_threshold)
    write_report(preselection, report_path, raw["metadata"])

    print(f"Agregation ecrite : {agg_path}")
    print(f"Classement ecrit : {ranking_path}")
    print(f"Matrice de qualification ecrite : {qualif_path}")
    print(f"Heatmap RMSE ecrite : {rmse_heatmap_stem}.html / .png")
    print(f"Heatmap Gibbs ecrite : {gibbs_heatmap_stem}.html / .png")
    print(f"Rapport de synthese ecrit : {report_path}")

    return {
        "agg": agg, "ranking_df": ranking_df, "qualification_df": qualification_df,
        "preselection": preselection, "gibbs_violation_bases": gibbs_violation_bases,
        "paths": {
            "aggregation_csv": agg_path, "ranking_csv": ranking_path,
            "qualification_csv": qualif_path,
            "rmse_heatmap_html": rmse_heatmap_stem + ".html",
            "rmse_heatmap_png": rmse_heatmap_stem + ".png",
            "gibbs_heatmap_html": gibbs_heatmap_stem + ".html",
            "gibbs_heatmap_png": gibbs_heatmap_stem + ".png",
            "report_txt": report_path,
        },
    }


if __name__ == "__main__":
    main()
