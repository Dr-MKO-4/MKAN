"""
niveau1_decision.py -- Graphiques d'aide a la decision de l'Etape 4 (Niveau 1),
construits UNIQUEMENT a partir des sorties deja sur disque (aucun reentrainement) :

    step4_outputs/step4_level1_results.json   (195 runs heterogenes + 5 baseline)
    step4_outputs/models/*.pt                 (poids finaux, pour la figure 7 seulement)

Usage :
    python niveau1_decision.py
    python niveau1_decision.py --out_dir step4_outputs --no_png
    python niveau1_decision.py --no_r2_portes        # saute la figure 7 (torch inutile)

Sorties (dans <out_dir>/figures_decision/) :
    fig1_poids_ahp_vs_influence.{html,png}
    fig2_ecart_uniforme_par_composante.{html,png}
    fig3_arbitrage_mcc_r2.{html,png}
    fig4_robustesse_des_rangs.{html,png}
    fig5_graine_par_graine.{html,png}
    fig6_gru_par_portes.{html,png}
    fig7_r2_symbolique_par_porte.{html,png}
    DECISION_ETAPE4.md      (tableaux chiffres qui accompagnent les figures)

Definitions (toutes appliquees aux moyennes sur les 5 graines) :
    perf      = w_MCC*max(0,MCC) + w_PR*PR-AUC - w_Brier*Brier   (poids AHP du benchmark)
    V1        = fitness du benchmark (perf + w_R2*R2_symb - w_lat*P_lat)
    V2        = perf + w_R2*R2_symb                 (sans penalite de latence)
    V3        = perf - w_lat*P_lat                  (sans R2_symb)
    V4        = perf                                (detection seule)
    V5        = AHP normalise : chaque critere (MCC, PR-AUC, Brier, R2_symb) ramene
                sur [0, 1] par min-max sur les architectures faisables, pondere par
                les poids AHP, la latence etant traitee comme contrainte (<= seuil)
                et non comme terme -- les infaisables ne sont pas classees.
    Regle pre-enregistree (chapitre 4, "Regle de selection") : un ecart de fitness
    moyenne inferieur a l'ecart-type entre graines est non concluant ; parmi les
    ex aequo, la moins couteuse en nombre de parametres puis en latence est retenue.
    L'ecart-type de reference est le plus grand des deux (meilleure, candidate).

Figure 7 : R2 symbolique PAR PORTE, recalcule a partir des poids (courbes d'arete
sur [-2, 2] comme _compute_r2_symbolic, ajustement c*f(a*x+b)+d par recherche
exhaustive sur (a, b) et (c, d) en forme close -- borne haute de l'ajustement
Adam du benchmark ; importance L1 calculee sur la grille et non sur x_pool).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

_ETUDE_DIR = os.path.dirname(os.path.abspath(__file__))
_MKAN_DIR = os.path.dirname(_ETUDE_DIR)
if _MKAN_DIR not in sys.path:
    sys.path.insert(0, _MKAN_DIR)


def _read_constants(path: str, names: tuple) -> dict:
    """Lit les constantes de niveau module dans le SOURCE de extended_heuristic_search.py
    (ast, sans l'executer) : memes valeurs que le benchmark, sans importer torch, que
    ce module charge des sa premiere ligne (figures 1 a 6 utilisables sans torch)."""
    import ast
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"constantes introuvables dans {path} : {sorted(missing)}")
    return found


_C = _read_constants(os.path.join(_ETUDE_DIR, "extended_heuristic_search.py"), (
    "AHP_WEIGHT_MCC", "AHP_WEIGHT_PR_AUC", "AHP_WEIGHT_BRIER", "AHP_WEIGHT_R2_SYMBOLIC",
    "AHP_WEIGHT_LATENCY_PENALTY", "LATENCY_CONSTRAINT_MS"))
AHP_WEIGHT_MCC = _C["AHP_WEIGHT_MCC"]
AHP_WEIGHT_PR_AUC = _C["AHP_WEIGHT_PR_AUC"]
AHP_WEIGHT_BRIER = _C["AHP_WEIGHT_BRIER"]
AHP_WEIGHT_R2_SYMBOLIC = _C["AHP_WEIGHT_R2_SYMBOLIC"]
AHP_WEIGHT_LATENCY_PENALTY = _C["AHP_WEIGHT_LATENCY_PENALTY"]
LATENCY_CONSTRAINT_MS = _C["LATENCY_CONSTRAINT_MS"]

ACCENT = "#2f6fd0"      # architecture retenue par la regle pre-enregistree
SECOND = "#d0742f"      # architecture retenue par l'AHP normalise
NEUTRAL = "#a3a8b0"     # autres architectures
FAINT = "#d6d9de"       # architectures infaisables
BASELINE = "#3c3f44"    # MKAN uniforme (hybrid)
GRU_COL = "#7a5cc0"     # GRUKANCell
BAD = "#c0392b"
INK = "#2b2d31"

ABBR = {"relukan": "ReLU", "efficientkan": "B-spl", "wavkan_dog": "DoG",
        "wavkan_morlet": "Morlet", "linear": "Lin", "hybrid": "Hyb"}
GATES = ["Forget", "Input", "Candidate", "Output"]
VARIANTS = [("V1", "Fitness du<br>benchmark"), ("V2", "Sans pénalité<br>de latence"),
            ("V3", "Sans R²<sub>symb</sub>"), ("V4", "Détection<br>seule"),
            ("V5", "AHP normalisé,<br>latence en contrainte")]
GATE_NOTE = "Ordre des portes : Forget/Input/Candidate/Output (GRU : reset/update/candidate)."


# ─────────────────────────────────────────────────────────────────────────────
# Chargement et agregation
# ─────────────────────────────────────────────────────────────────────────────

def _arch_label(r: dict) -> str:
    b = r["base_by_gate"]
    cell = "GRU" if r["cell_type"] == "gru" else "TKAN"
    regime = "brut" if r["regime"] == "raw" else "traité"
    prefix = "Uniforme " if b.get("Forget") == "hybrid" else ""
    return f"{prefix}{cell}·{regime} " + "/".join(ABBR[b[g]] for g in GATES if g in b)


def _perf(m: dict) -> float:
    return (AHP_WEIGHT_MCC * max(0.0, m["mcc"]) + AHP_WEIGHT_PR_AUC * m["auc_pr"]
            - AHP_WEIGHT_BRIER * m["brier"])


def load_runs(out_dir: str) -> list:
    with open(os.path.join(out_dir, "step4_level1_results.json"), encoding="utf-8") as f:
        d = json.load(f)
    runs = []
    for r in d["results"] + d["baseline_results"]:
        m = r["metrics"]
        perf = _perf(m)
        runs.append(dict(
            key_hash=r["key_hash"], arch=_arch_label(r), cell=r["cell_type"], regime=r["regime"],
            base_by_gate=r["base_by_gate"], seed=r["seed"],
            baseline=r["base_by_gate"].get("Forget") == "hybrid",
            mcc=m["mcc"], pr=m["auc_pr"], brier=m["brier"], r2=m["r2_symbolic"],
            lat=m["latency_ms"], pen=m["penalty_lat"], npar=m["n_params"],
            V1=m["fitness_total"], V2=perf + AHP_WEIGHT_R2_SYMBOLIC * m["r2_symbolic"],
            V3=perf - AHP_WEIGHT_LATENCY_PENALTY * m["penalty_lat"], V4=perf,
            c_mcc=AHP_WEIGHT_MCC * max(0.0, m["mcc"]), c_pr=AHP_WEIGHT_PR_AUC * m["auc_pr"],
            c_brier=-AHP_WEIGHT_BRIER * m["brier"], c_r2=AHP_WEIGHT_R2_SYMBOLIC * m["r2_symbolic"],
            c_pen=-AHP_WEIGHT_LATENCY_PENALTY * m["penalty_lat"],
        ))
    return runs


def attach_r2_thresholds(runs: list, out_dir: str, threshold: str = "0.80") -> bool:
    """Si niveau1_r2_seuils.py a ete lance, ajoute V6 = fitness avec R2_symb au seuil
    `threshold` (analyse de sensibilite ; la regle reste au seuil 0,99)."""
    path = os.path.join(out_dir, "r2_seuils.json")
    if not os.path.exists(path):
        print(f"[info] {path} absent : variante R2 au seuil {threshold} non tracee "
              f"(lancer niveau1_r2_seuils.py).")
        return False
    with open(path, encoding="utf-8") as f:
        models = json.load(f)["models"]
    if any(r["key_hash"] not in models or threshold not in models[r["key_hash"]]["r2"]
           for r in runs):
        print(f"[avertissement] {path} incomplet pour le seuil {threshold} : variante ignoree.")
        return False
    for r in runs:
        m = models[r["key_hash"]]
        r["r2_alt"] = m["r2"][threshold]
        r["V6"] = r["V1"] + AHP_WEIGHT_R2_SYMBOLIC * (m["r2"][threshold] - r["r2"])
    VARIANTS.append(("V6", f"R²<sub>symb</sub> au<br>seuil {threshold.replace('.', ',')}"))
    return True


def aggregate(runs: list) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[r["arch"]].append(r)
    extra = ["r2_alt", "V6"] if "V6" in runs[0] else []
    agg = {}
    for arch, rs in by.items():
        a = dict(arch=arch, n=len(rs), cell=rs[0]["cell"], regime=rs[0]["regime"],
                 baseline=rs[0]["baseline"], base_by_gate=rs[0]["base_by_gate"],
                 npar=rs[0]["npar"], seeds=sorted(r["seed"] for r in rs))
        for k in ["mcc", "pr", "brier", "r2", "lat", "pen", "V1", "V2", "V3", "V4",
                  "c_mcc", "c_pr", "c_brier", "c_r2", "c_pen"] + extra:
            v = np.array([r[k] for r in rs], dtype=float)
            a[k] = float(v.mean())
            a[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        a["feasible"] = a["lat"] <= LATENCY_CONSTRAINT_MS
        agg[arch] = a
    _add_normalized_ahp(agg)
    for vid, _ in VARIANTS:
        vals = sorted((a[vid] for a in agg.values() if a.get(vid) is not None), reverse=True)
        for a in agg.values():
            a["rank_" + vid] = (vals.index(a[vid]) + 1) if a.get(vid) is not None else None
    return agg


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def _add_normalized_ahp(agg: dict) -> None:
    feas = [a for a in agg.values() if a["feasible"]]
    w = np.array([AHP_WEIGHT_MCC, AHP_WEIGHT_PR_AUC, AHP_WEIGHT_BRIER, AHP_WEIGHT_R2_SYMBOLIC])
    z = np.c_[_minmax(np.array([max(0.0, a["mcc"]) for a in feas])),
              _minmax(np.array([a["pr"] for a in feas])),
              1.0 - _minmax(np.array([a["brier"] for a in feas])),
              _minmax(np.array([a["r2"] for a in feas]))]
    score = z @ w / w.sum()
    for a in agg.values():
        a["V5"] = None
    for a, s in zip(feas, score):
        a["V5"] = float(s)


def pareto_feasible(agg: dict) -> set:
    """Front de Pareto des architectures faisables sur (max(0,MCC), PR-AUC, -Brier, R2),
    dominance contrainte du chapitre 4 (les infaisables sont toutes dominees)."""
    feas = [a for a in agg.values() if a["feasible"]]
    X = np.array([[max(0.0, a["mcc"]), a["pr"], -a["brier"], a["r2"]] for a in feas])
    front = set()
    for i, a in enumerate(feas):
        dominated = any((X[j] >= X[i]).all() and (X[j] > X[i]).any()
                        for j in range(len(feas)) if j != i)
        if not dominated:
            front.add(a["arch"])
    return front


def preregistered_rule(agg: dict) -> dict:
    ordered = sorted(agg.values(), key=lambda a: a["V1"], reverse=True)
    best = ordered[0]
    ties = [a for a in ordered if best["V1"] - a["V1"] < max(best["V1_sd"], a["V1_sd"])]
    ties_strict = [a for a in ordered if best["V1"] - a["V1"] < best["V1_sd"]]
    winner = min(ties, key=lambda a: (a["npar"], a["lat"]))
    return dict(best=best, ties=ties, ties_strict=ties_strict, winner=winner)


# ─────────────────────────────────────────────────────────────────────────────
# Figures (Plotly -- convention du projet : .html interactif + .png via kaleido)
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, out_path_no_ext: str, export_png: bool = True) -> str:
    os.makedirs(os.path.dirname(out_path_no_ext) or ".", exist_ok=True)
    html_path = out_path_no_ext + ".html"
    fig.write_html(html_path)
    if export_png:
        try:
            fig.write_image(out_path_no_ext + ".png", scale=2)
        except Exception as exc:   # kaleido peut manquer selon l'environnement
            print(f"[avertissement] export PNG impossible pour {out_path_no_ext} ({exc}) -- "
                  f"HTML seul disponible ({html_path}).")
    print(html_path)
    return html_path


def _layout(fig, title: str, subtitle: str, height: int, width: int = 1000, **kw) -> None:
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.01, xanchor="left"),
        template="plotly_white", height=height, width=width,
        font=dict(color=INK, size=12), margin=dict(t=90, l=10, r=20, b=60), **kw)


def fig_poids_ahp_vs_influence(agg: dict, path: str, png: bool) -> dict:
    """Part de chaque critere dans les poids AHP vs part dans l'ecart reel de fitness
    entre architectures (poids x etendue du critere sur les TKAN et la reference)."""
    import plotly.graph_objects as go
    pool = [a for a in agg.values() if a["cell"] == "lstm"]
    crit = [("MCC", AHP_WEIGHT_MCC, [max(0.0, a["mcc"]) for a in pool]),
            ("PR-AUC", AHP_WEIGHT_PR_AUC, [a["pr"] for a in pool]),
            ("Brier", AHP_WEIGHT_BRIER, [a["brier"] for a in pool]),
            ("R²<sub>symb</sub>", AHP_WEIGHT_R2_SYMBOLIC, [a["r2"] for a in pool]),
            ("Pénalité de latence", AHP_WEIGHT_LATENCY_PENALTY, [a["pen"] for a in pool])]
    wsum = sum(c[1] for c in crit)
    eff = [c[1] * (max(c[2]) - min(c[2])) for c in crit]
    esum = sum(eff)
    names = [c[0] for c in crit][::-1]
    w_pct = [100 * c[1] / wsum for c in crit][::-1]
    e_pct = [100 * e / esum for e in eff][::-1]
    spans = [max(c[2]) - min(c[2]) for c in crit][::-1]
    fig = go.Figure()
    fig.add_bar(y=names, x=w_pct, orientation="h", name="Poids AHP", marker_color=NEUTRAL,
                text=[f"{v:.1f} %" for v in w_pct], textposition="outside",
                hovertemplate="%{y}<br>poids AHP = %{x:.1f} %<extra></extra>")
    fig.add_bar(y=names, x=e_pct, orientation="h", name="Influence réelle sur le classement",
                marker_color=ACCENT, text=[f"{v:.1f} %" for v in e_pct], textposition="outside",
                customdata=spans,
                hovertemplate="%{y}<br>influence = %{x:.1f} %<br>étendue entre architectures = "
                              "%{customdata:.4f}<extra></extra>")
    _layout(fig, "Les poids AHP ne sont pas ceux qui départagent les architectures",
            f"Influence réelle = poids × étendue du critère sur les {len(pool)} TKANCell "
            "(dont la référence uniforme), en part du total. GRU exclus.", 420,
            barmode="group", xaxis=dict(title="Part (%)", range=[0, 1.15 * max(w_pct + e_pct)]),
            legend=dict(orientation="h", y=-0.18))
    _save_fig(fig, path, png)
    return dict(zip([c[0] for c in crit], zip([100 * c[1] / wsum for c in crit],
                                                [100 * e / esum for e in eff])))


def fig_ecart_uniforme(agg: dict, rule: dict, path: str, png: bool) -> None:
    """Ecart de chaque composante de fitness a la reference uniforme, pour les TKANCell."""
    import plotly.graph_objects as go
    base = next(a for a in agg.values() if a["baseline"])
    rows = sorted([a for a in agg.values() if a["cell"] == "lstm" and not a["baseline"]],
                  key=lambda a: a["V1"])
    labels = [("<b>%s</b>" % a["arch"]) if a is rule["winner"] else a["arch"] for a in rows]
    comps = [("c_mcc", "MCC", "#2f6fd0"), ("c_pr", "PR-AUC", "#6d9be0"),
             ("c_brier", "Brier", "#9fbde9"), ("c_r2", "R²<sub>symb</sub>", "#d0742f"),
             ("c_pen", "Pénalité de latence", BAD)]
    fig = go.Figure()
    for k, name, col in comps:
        fig.add_bar(y=labels, x=[a[k] - base[k] for a in rows], orientation="h", name=name,
                    marker_color=col,
                    hovertemplate="%{y}<br>" + name + " : %{x:+.4f}<extra></extra>")
    fig.add_scatter(y=labels, x=[a["V1"] - base["V1"] for a in rows], mode="markers",
                    name="Écart de fitness total", marker=dict(symbol="diamond", size=9, color=INK),
                    hovertemplate="%{y}<br>écart total : %{x:+.4f}<extra></extra>")
    fig.add_vline(x=0, line_color=BASELINE, line_width=1)
    _layout(fig, "D'où vient l'écart de fitness à la référence uniforme",
            "Contribution pondérée de chaque critère, moins celle du MKAN uniforme (hybrid), "
            "moyenne sur 5 graines. " + GATE_NOTE, max(520, 24 * len(rows) + 180),
            barmode="relative", xaxis=dict(title="Écart à la référence uniforme (points de fitness)",
                                           zeroline=False),
            yaxis=dict(tickfont=dict(size=10)), legend=dict(orientation="h", y=-0.08))
    _save_fig(fig, path, png)


def fig_arbitrage(agg: dict, rule: dict, b_norm: dict, front: set, path: str, png: bool) -> None:
    """MCC vs R2_symb, barres d'erreur entre graines, faisabilite et front de Pareto."""
    import plotly.graph_objects as go
    xmin = 0.95
    shown = [a for a in agg.values() if a["mcc"] >= xmin]
    hidden = [a for a in agg.values() if a["mcc"] < xmin]
    groups = [
        ("TKANCell faisable (≤ %d ms)" % LATENCY_CONSTRAINT_MS,
         lambda a: a["cell"] == "lstm" and not a["baseline"] and a["feasible"], NEUTRAL, "circle"),
        ("TKANCell infaisable", lambda a: a["cell"] == "lstm" and not a["baseline"]
         and not a["feasible"], FAINT, "circle"),
        ("GRUKANCell (infaisable)", lambda a: a["cell"] == "gru", GRU_COL, "square"),
        ("MKAN uniforme (référence)", lambda a: a["baseline"], BASELINE, "diamond"),
    ]
    fig = go.Figure()
    sizeref = max(a["npar"] for a in shown) / 30.0
    for name, pred, col, sym in groups:
        pts = [a for a in shown if pred(a)]
        if not pts:
            continue
        fig.add_scatter(
            x=[a["mcc"] for a in pts], y=[a["r2"] for a in pts], mode="markers", name=name,
            error_x=dict(array=[a["mcc_sd"] for a in pts], color=col, thickness=1, width=0),
            error_y=dict(array=[a["r2_sd"] for a in pts], color=col, thickness=1, width=0),
            marker=dict(color=col, symbol=sym, size=[max(7, a["npar"] / sizeref) for a in pts],
                        line=dict(color=[INK if a["arch"] in front else "rgba(0,0,0,0)"
                                         for a in pts], width=1.5)),
            text=[a["arch"] for a in pts],
            customdata=[[a["lat"], a["npar"], a["V1"]] for a in pts],
            hovertemplate="%{text}<br>MCC = %{x:.4f}<br>R² symb = %{y:.3f}<br>latence = "
                          "%{customdata[0]:.0f} ms<br>paramètres = %{customdata[1]:,}<br>"
                          "fitness = %{customdata[2]:.4f}<extra></extra>")
    fr = sorted((agg[k] for k in front), key=lambda a: a["mcc"])
    fig.add_scatter(x=[a["mcc"] for a in fr], y=[a["r2"] for a in fr], mode="lines",
                    line=dict(color=INK, width=1, dash="dot"), name="Front de Pareto (faisables)",
                    hoverinfo="skip")
    for a, txt, col in [(rule["winner"], "Règle pré-enregistrée", ACCENT),
                        (b_norm, "AHP normalisé", SECOND),
                        (next(x for x in agg.values() if x["baseline"]), "Référence uniforme",
                         BASELINE)]:
        fig.add_annotation(x=a["mcc"], y=a["r2"], text=f"<b>{txt}</b><br>{a['arch']}",
                           showarrow=True, arrowhead=0, ax=60 if a["mcc"] < 0.972 else -60,
                           ay=-40, font=dict(size=10, color=col), arrowcolor=col)
    note = (f"{len(hidden)} architectures GRU effondrées (MCC < {xmin}) hors cadre. "
            if hidden else "")
    _layout(fig, "Arbitrage détection / interprétabilité sous contrainte de latence",
            "Moyenne ± écart-type sur 5 graines ; taille ∝ nombre de paramètres ; contour noir = "
            "front de Pareto des faisables sur (MCC, PR-AUC, −Brier, R²). " + note + GATE_NOTE,
            640, xaxis=dict(title="MCC (validation, seuil 0,5)",
                            range=[xmin, max(a["mcc"] + a["mcc_sd"] for a in shown) + 0.002]),
            yaxis=dict(title="R² symbolique (part des arêtes ajustées à R² ≥ 0,99)",
                       range=[-0.05, 1.05]),
            legend=dict(orientation="h", y=-0.15))
    _save_fig(fig, path, png)


def fig_robustesse(agg: dict, rule: dict, b_norm: dict, path: str, png: bool) -> None:
    import plotly.graph_objects as go
    rows = sorted(agg.values(), key=lambda a: a["rank_V1"])
    def lab(a):
        if a is rule["winner"] or a is b_norm or a["baseline"]:
            return f"<b>{a['arch']}</b>"
        return a["arch"]
    z, txt = [], []
    for a in rows:
        z.append([a["rank_" + v] if a["rank_" + v] is not None else np.nan for v, _ in VARIANTS])
        txt.append([str(a["rank_" + v]) if a["rank_" + v] is not None else "—" for v, _ in VARIANTS])
    fig = go.Figure(go.Heatmap(
        z=z, x=[name for _, name in VARIANTS], y=[lab(a) for a in rows], text=txt,
        texttemplate="%{text}", colorscale=[[0, ACCENT], [0.35, "#cfdcf1"], [1, "#f4f5f7"]],
        zmin=1, zmax=len(rows), colorbar=dict(title="Rang"),
        hovertemplate="%{y}<br>%{x}<br>rang %{text}<extra></extra>"))
    _layout(fig, "Le rang de chaque architecture selon la manière de juger",
            "Rang 1 = meilleur. V5 ne classe que les architectures faisables (— = latence > "
            f"{LATENCY_CONSTRAINT_MS:.0f} ms). En gras : règle pré-enregistrée, AHP normalisé, "
            "référence uniforme.", 26 * len(rows) + 200,
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)), xaxis=dict(side="top"))
    _save_fig(fig, path, png)


def fig_graines(runs: list, agg: dict, rule: dict, b_norm: dict, path: str, png: bool) -> None:
    """Comparaison appariee graine par graine (meme graine = meme ligne)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    base = next(a for a in agg.values() if a["baseline"])
    gru_best = max((a for a in agg.values() if a["cell"] == "gru"), key=lambda a: a["mcc"])
    models = [(base, "Référence<br>uniforme", BASELINE), (rule["winner"], "Règle<br>pré-enregistrée",
              ACCENT), (b_norm, "AHP<br>normalisé", SECOND), (gru_best, "Meilleur GRU<br>(MCC)",
              GRU_COL)]
    by = {(r["arch"], r["seed"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    metrics = [("mcc", "MCC"), ("r2", "R² symbolique"), ("lat", "Latence CPU (ms)")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[1] for m in metrics],
                        horizontal_spacing=0.08)
    xs = [m[1] for m in models]
    for c, (k, _) in enumerate(metrics, start=1):
        for s in seeds:
            fig.add_scatter(x=xs, y=[by[(m[0]["arch"], s)][k] for m in models], mode="lines",
                            line=dict(color=FAINT, width=1), showlegend=False, hoverinfo="skip",
                            row=1, col=c)
        for m, x, col in models:
            fig.add_scatter(x=[x] * len(seeds), y=[by[(m["arch"], s)][k] for s in seeds],
                            mode="markers", marker=dict(color=col, size=8), showlegend=False,
                            text=[f"{m['arch']}<br>graine {s}" for s in seeds],
                            hovertemplate="%{text}<br>%{y:.4f}<extra></extra>", row=1, col=c)
    fig.add_hline(y=LATENCY_CONSTRAINT_MS, line_dash="dash", line_color=BAD, row=1, col=3,
                  annotation_text=f"contrainte {LATENCY_CONSTRAINT_MS:.0f} ms",
                  annotation_position="top left")
    fig.update_yaxes(type="log", row=1, col=3)
    _layout(fig, "Graine par graine : ce que gagne et perd chaque candidat",
            "Chaque ligne grise relie les résultats d'une même graine. "
            f"Retenues : {rule['winner']['arch']} (règle) ; {b_norm['arch']} (AHP normalisé) ; "
            f"{gru_best['arch']} (GRU).", 480, width=1150)
    _save_fig(fig, path, png)


def fig_gru(agg: dict, path: str, png: bool) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    gru = [a for a in agg.values() if a["cell"] == "gru"]
    resets = ["relukan", "wavkan_dog"]
    updates = ["relukan", "wavkan_dog", "efficientkan"]
    cands = ["wavkan_dog", "wavkan_morlet"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Candidate = {ABBR[c]}" for c in cands],
                        horizontal_spacing=0.12)
    for ci, cand in enumerate(cands, start=1):
        z, txt = [], []
        for rs in resets:
            zr, tr = [], []
            for up in updates:
                a = next((g for g in gru if g["base_by_gate"] == {"Forget": rs, "Input": up,
                                                                    "Candidate": cand}), None)
                zr.append(a["mcc"] if a else np.nan)
                tr.append(f"MCC {a['mcc']:.3f}<br>{a['lat']:.0f} ms" if a else "—")
            z.append(zr)
            txt.append(tr)
        fig.add_trace(go.Heatmap(z=z, x=[ABBR[u] for u in updates], y=[ABBR[r] for r in resets],
                                 text=txt, texttemplate="%{text}", zmin=0, zmax=1,
                                 colorscale=[[0, BAD], [0.6, "#f3d9c9"], [1, ACCENT]],
                                 showscale=(ci == 2), colorbar=dict(title="MCC"),
                                 hovertemplate="reset %{y} · update %{x}<br>%{text}<extra></extra>"),
                      row=1, col=ci)
    fig.update_xaxes(title_text="Porte update (Input)")
    fig.update_yaxes(title_text="Porte reset (Forget)", col=1)
    _layout(fig, "GRUKANCell : l'effondrement suit la base ReLU-KAN",
            "MCC moyen et latence CPU moyenne sur 5 graines, h = 149, W = 5. "
            "Toutes les configurations dépassent la contrainte de latence.", 420)
    _save_fig(fig, path, png)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 : R2 symbolique par porte, recalcule a partir des poids
# ─────────────────────────────────────────────────────────────────────────────

def _basis_from_state(prefix: str, sd: dict, label: str, in_f: int, out_f: int):
    """Reconstruit la base d'une porte a partir de ses poids (hyperparametres lus
    dans les formes des tenseurs) et charge l'etat en mode strict."""
    from edges import BASIS_REGISTRY
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if label == "hybrid":
        basis = BASIS_REGISTRY["hybrid"](in_f, out_f, M=sub["w_gauss"].shape[-1],
                                         K=sub["a_fourier"].shape[-1])
    elif label.startswith("wavkan"):
        basis = BASIS_REGISTRY["wavkan"](in_f, out_f, n_wavelets=sub["tau"].shape[-1],
                                         wavelet=label.split("_", 1)[1])
    elif label == "relukan":
        basis = BASIS_REGISTRY["relukan"](in_f, out_f, n_bases=sub["weight"].shape[-1])
    elif label == "efficientkan":
        n_basis, n_knots = sub["weight"].shape[-1], sub["grid"].shape[-1]
        k = n_knots - n_basis - 1
        basis = BASIS_REGISTRY["efficientkan"](in_f, out_f, G=n_basis - k, k=k)
    elif label == "linear":
        basis = BASIS_REGISTRY["linear"](in_f, out_f)
    else:
        raise ValueError(label)
    basis.load_state_dict(sub, strict=True)
    return basis.eval()


def _symbolic_bank(x, n_grid: int = 61):
    """Toutes les courbes f(a*x+b) de SYMBOLIC_LIBRARY sur une grille (a, b), centrees
    et normees : le R2 de c*f(a*x+b)+d en forme close est alors le carre du produit
    scalaire avec la courbe d'arete centree et normee."""
    import torch
    import importlib.util as _u
    if "MKAN" not in sys.modules:
        spec = _u.spec_from_loader("MKAN", loader=None, is_package=True)
        pkg = _u.module_from_spec(spec)
        pkg.__path__ = [_MKAN_DIR]
        sys.modules["MKAN"] = pkg
    from MKAN.symbolic import SYMBOLIC_LIBRARY
    a = torch.linspace(-4, 4, n_grid)
    a = a[a.abs() > 1e-6]
    b = torch.linspace(-4, 4, n_grid)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    z = aa.reshape(-1, 1) * x + bb.reshape(-1, 1)
    bank = []
    for fn in SYMBOLIC_LIBRARY.values():
        g = fn(z)
        g = g - g.mean(dim=1, keepdim=True)
        nrm = g.norm(dim=1, keepdim=True)
        ok = torch.isfinite(g).all(dim=1) & (nrm[:, 0] > 1e-8)
        bank.append((g / nrm.clamp(min=1e-12))[ok])
    return torch.cat(bank)


def r2_par_porte(out_dir: str, runs: list, archs: list) -> list:
    import torch
    torch.set_grad_enabled(False)
    x = torch.linspace(-2.0, 2.0, 100)      # meme domaine que _compute_r2_symbolic
    bank = _symbolic_bank(x)
    gate_attr = {"Forget": "forget_gate", "Input": "input_gate", "Candidate": "candidate_gate",
                 "Output": "output_gate"}
    out = []
    for arch in archs:
        for r in [r for r in runs if r["arch"] == arch]:
            sd = torch.load(os.path.join(out_dir, "models", r["key_hash"] + ".pt"),
                            map_location="cpu", weights_only=True)
            curves, tags = [], []
            for g in GATES:
                prefix = f"cell.{gate_attr[g]}.basis."
                out_f = sd[f"cell.{gate_attr[g]}._add_mask"].shape[0]      # un noeud par sortie
                in_f = next(v for k, v in sd.items()
                            if k.startswith(prefix) and v.ndim >= 2).shape[0]  # input + hidden
                basis = _basis_from_state(prefix, sd, r["base_by_gate"][g], in_f, out_f)
                edges = basis(x.unsqueeze(1).expand(-1, in_f))          # (100, in, out)
                curves.append(edges.reshape(100, -1).T)
                tags += [g] * (in_f * out_f)
            Y = torch.cat(curves)
            Yc = Y - Y.mean(dim=1, keepdim=True)
            Yc = Yc / Yc.norm(dim=1, keepdim=True).clamp(min=1e-12)
            best = torch.cat([((bank @ Yc[i:i + 512].T) ** 2).max(dim=0).values
                              for i in range(0, len(Yc), 512)])
            top = torch.argsort(Y.abs().mean(dim=1), descending=True)[:40]
            top_tags = Counter(tags[i] for i in top.tolist())
            tags_arr = np.array(tags)
            for g in GATES:
                m = torch.from_numpy(tags_arr == g)
                out.append(dict(arch=arch, seed=r["seed"], gate=g, base=r["base_by_gate"][g],
                                r2_median=float(best[m].median()),
                                share_099=float((best[m] >= 0.99).float().mean()),
                                top40=top_tags.get(g, 0),
                                r2_top40_proxy=float((best[top] >= 0.99).float().mean()),
                                r2_json=r["r2"]))
    return out


def fig_r2_portes(rows: list, archs: list, colors: list, path: str, png: bool) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=[
        "R² médian par arête", "Part des arêtes à R² ≥ 0,99", "Origine des 40 arêtes les plus fortes"])
    for arch, col in zip(archs, colors):
        rs = [r for r in rows if r["arch"] == arch]
        med = [np.mean([r["r2_median"] for r in rs if r["gate"] == g]) for g in GATES]
        sh = [np.mean([r["share_099"] for r in rs if r["gate"] == g]) for g in GATES]
        top = [np.mean([r["top40"] for r in rs if r["gate"] == g]) for g in GATES]
        bases = [ABBR[next(r["base"] for r in rs if r["gate"] == g)] for g in GATES]
        hover = [f"{arch}<br>{g} = {b}" for g, b in zip(GATES, bases)]
        fig.add_bar(x=GATES, y=med, name=arch, marker_color=col, legendgroup=arch, text=hover,
                    hovertemplate="%{text}<br>%{y:.3f}<extra></extra>", row=1, col=1)
        fig.add_bar(x=GATES, y=sh, name=arch, marker_color=col, legendgroup=arch, text=hover,
                    showlegend=False, hovertemplate="%{text}<br>%{y:.1%}<extra></extra>",
                    row=1, col=2)
        fig.add_bar(x=GATES, y=top, name=arch, marker_color=col, legendgroup=arch, text=hover,
                    showlegend=False, hovertemplate="%{text}<br>%{y:.1f} arêtes sur 40<extra></extra>",
                    row=1, col=3)
    fig.add_hline(y=0.99, line_dash="dash", line_color=BAD, row=1, col=1,
                  annotation_text="seuil 0,99", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", row=1, col=2)
    fig.update_yaxes(range=[0, 40], row=1, col=3)
    _layout(fig, "Pourquoi le R² symbolique favorise une porte de sortie linéaire",
            "Recalculé à partir des poids enregistrés (moyenne sur 5 graines) : courbes d'arête sur "
            "[−2, 2], meilleur ajustement c·f(a·x+b)+d sur la bibliothèque symbolique.", 470,
            width=1250, barmode="group", legend=dict(orientation="h", y=-0.18))
    _save_fig(fig, path, png)


# ─────────────────────────────────────────────────────────────────────────────
# Recapitulatif chiffre
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(path: str, agg: dict, rule: dict, b_norm: dict, front: set, influence: dict,
                  r2rows: list) -> None:
    f4 = lambda v: f"{v:.4f}"
    L = ["# Decision -- Etape 4 (Niveau 1)", "",
         "Genere par niveau1_decision.py a partir de step4_level1_results.json.", "",
         "## Regle pre-enregistree (chapitre 4)", "",
         f"- Meilleure fitness moyenne : {rule['best']['arch']} = {f4(rule['best']['V1'])} "
         f"+/- {f4(rule['best']['V1_sd'])}",
         f"- Ex aequo (ecart < max des deux ecarts-types) : {len(rule['ties'])} ; "
         f"(ecart < ecart-type de la meilleure) : {len(rule['ties_strict'])}",
         "", "| Architecture | Fitness | Ecart-type | Ecart a la meilleure | Parametres | Latence (ms) | MCC | R2 symb |",
         "|---|---|---|---|---|---|---|---|"]
    for a in rule["ties"]:
        L.append(f"| {a['arch']} | {f4(a['V1'])} | {f4(a['V1_sd'])} | "
                 f"{f4(rule['best']['V1'] - a['V1'])} | {a['npar']:,} | {a['lat']:.1f} | "
                 f"{f4(a['mcc'])} | {a['r2']:.3f} |")
    L += ["", f"**Retenue par la regle (parametres puis latence) : {rule['winner']['arch']}**",
          "", "## Variantes de jugement", "",
          f"- Retenue par l'AHP normalise (V5, latence en contrainte) : {b_norm['arch']}",
          f"- Front de Pareto des faisables : {len(front)} architectures", "",
          "| Architecture | " + " | ".join(v for v, _ in VARIANTS)
          + " | MCC | R2 symb | Latence (ms) | Parametres | Pareto |",
          "|---|" + "---|" * (len(VARIANTS) + 5)]
    for a in sorted(agg.values(), key=lambda a: a["rank_V1"]):
        rk = " | ".join(str(a["rank_" + v]) if a["rank_" + v] is not None else "-" for v, _ in VARIANTS)
        L.append(f"| {a['arch']} | {rk} | {f4(a['mcc'])} | {a['r2']:.3f} | {a['lat']:.1f} | "
                 f"{a['npar']:,} | {'oui' if a['arch'] in front else ''} |")
    L += ["", "## Poids AHP et influence reelle (TKANCell + reference)", "",
          "| Critere | Poids AHP (%) | Influence reelle (%) |", "|---|---|---|"]
    for k, (w, e) in influence.items():
        L.append(f"| {k.replace('<sub>', '_').replace('</sub>', '')} | {w:.1f} | {e:.1f} |")
    if r2rows:
        L += ["", "## R2 symbolique par porte (recalcule a partir des poids)", "",
              "| Architecture | Graine | Porte | Base | R2 median | Part R2>=0.99 | Arretes du top 40 | R2 estime (top 40) | R2 du JSON |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in r2rows:
            L.append(f"| {r['arch']} | {r['seed']} | {r['gate']} | {r['base']} | {r['r2_median']:.3f} | "
                     f"{r['share_099']:.2f} | {r['top40']} | {r['r2_top40_proxy']:.3f} | {r['r2_json']:.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out_dir", default=os.path.join(_ETUDE_DIR, "step4_outputs"))
    p.add_argument("--no_png", action="store_true", help="HTML seulement (kaleido non requis)")
    p.add_argument("--no_r2_portes", action="store_true", help="sauter la figure 7")
    args = p.parse_args()
    png = not args.no_png
    fig_dir = os.path.join(args.out_dir, "figures_decision")

    runs = load_runs(args.out_dir)
    attach_r2_thresholds(runs, args.out_dir)
    agg = aggregate(runs)
    rule = preregistered_rule(agg)
    b_norm = max((a for a in agg.values() if a["V5"] is not None), key=lambda a: a["V5"])
    front = pareto_feasible(agg)
    print(f"Regle pre-enregistree : {rule['winner']['arch']} ({len(rule['ties'])} ex aequo)")
    print(f"AHP normalise         : {b_norm['arch']}")

    influence = fig_poids_ahp_vs_influence(agg, os.path.join(fig_dir, "fig1_poids_ahp_vs_influence"), png)
    fig_ecart_uniforme(agg, rule, os.path.join(fig_dir, "fig2_ecart_uniforme_par_composante"), png)
    fig_arbitrage(agg, rule, b_norm, front, os.path.join(fig_dir, "fig3_arbitrage_mcc_r2"), png)
    fig_robustesse(agg, rule, b_norm, os.path.join(fig_dir, "fig4_robustesse_des_rangs"), png)
    fig_graines(runs, agg, rule, b_norm, os.path.join(fig_dir, "fig5_graine_par_graine"), png)
    fig_gru(agg, os.path.join(fig_dir, "fig6_gru_par_portes"), png)

    r2rows = []
    if not args.no_r2_portes:
        base = next(a for a in agg.values() if a["baseline"])
        w = rule["winner"]["base_by_gate"]
        morlet_out = next((a for a in agg.values() if a["cell"] == "lstm" and a["regime"] == "raw"
                           and a["base_by_gate"] == dict(w, Output="wavkan_morlet")), None)
        archs = [base["arch"], rule["winner"]["arch"], b_norm["arch"]]
        colors = [BASELINE, ACCENT, SECOND]
        if morlet_out is not None:
            archs.append(morlet_out["arch"])
            colors.append(NEUTRAL)
        try:
            import torch  # noqa: F401  (seule la figure 7 en a besoin)
        except (ImportError, OSError) as exc:
            print(f"[avertissement] torch ne se charge pas ({exc}) -- figure 7 sautee, "
                  f"figures 1 a 6 et DECISION_ETAPE4.md produites quand meme.")
        else:
            r2rows = r2_par_porte(args.out_dir, runs, archs)
            fig_r2_portes(r2rows, archs, colors,
                          os.path.join(fig_dir, "fig7_r2_symbolique_par_porte"), png)

    write_summary(os.path.join(fig_dir, "DECISION_ETAPE4.md"), agg, rule, b_norm, front,
                  influence, r2rows)


if __name__ == "__main__":
    main()
