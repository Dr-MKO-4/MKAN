"""
niveau1_courbes_aretes.py -- Lecture des fonctions apprises phi(x) des aretes d'entree des
modeles de l'Etape 4, SANS reentrainement : les poids sont relus dans step4_outputs/models/*.pt.

Ce que le script laisse decider au modele (rien n'est impose a priori) :
    1. Importance des variables. Pour chaque variable d'entree k (indice hidden_size + k du
       vecteur [h, x]) et chaque porte, importance = somme sur les unites j de la norme L1
       moyenne |phi_kj| sur le pool d'audit du benchmark (256 fenetres de validation,
       default_rng(0)), en ne comptant que les aretes survivantes (L1 > theta = 0,01,
       comme _compute_r2_symbolic). Les variables tracees ensuite sont les variables
       numeriques classees par importance.
    2. Forme des courbes. Chaque arete survivante tracee est ajustee par c*f(a*z+b)+d sur la
       bibliotheque symbolique, a l'identique de _compute_r2_symbolic (grille [-2, 2],
       100 points, 150 pas Adam, lr=0,05) ; formule retenue si R2_e >= seuil (0,80,
       cf. DECISIONS.md). La lecture vient de la formule : point caracteristique z* = -b/a
       (centre de transition pour sigmoid/tanh, coude pour softplus/|x|/sqrt/log,
       extremum pour x^2), converti en unites d'origine, et sens de variation de la courbe
       sur la plage P1-P99 observee.
    3. Effet net sur le score. Les fonctions phi alimentent les portes, pas le score : la
       variable du dernier pas des 256 fenetres est remplacee par chaque valeur d'une
       grille (ou chaque type d'operation pour les variables one-hot), les autres entrees
       restant inchangees, et le score moyen est trace.

Axe horizontal en unites d'origine (FCFA) par inversion de la transformation d'entree
z = (sign(x) log1p|x| - mu_train) / std_train ; mu_train, std_train recalcules sur le jeu
d'entrainement de l'Etape 4.

Usage :
    python niveau1_courbes_aretes.py
    python niveau1_courbes_aretes.py --archs A dog --n_vars 3 --top_k 2
    python niveau1_courbes_aretes.py --seed 3 --no_png

Sorties (dans --out_dir, par defaut step4_outputs/figures_aretes/) :
    importance_variables.{html,png}     importance variable x porte, par architecture
    <arch>_courbes_phi.{html,png}       courbes phi des aretes les plus fortes, variable x porte
    score_variables_numeriques.{html,png}  effet net de chaque variable numerique sur le score
    score_type_operation.{html,png}     score moyen selon le type d'operation du dernier pas
    aretes_resume.json                  importances, aretes (formule, R2_e, point
                                        caracteristique, sens de variation), effets nets
"""

import argparse
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

_ETUDE_DIR = os.path.dirname(os.path.abspath(__file__))
if _ETUDE_DIR not in sys.path:
    sys.path.insert(0, _ETUDE_DIR)

from niveau1_harness import (                                                  # noqa: E402
    ACTION_ONEHOT_COLS, RAW_NUMERIC_COLS, build_regime_frame, read_momtsim_parquet, standardize,
)
from niveau1_benchmark_run import CELL_THETA_OPT, REGIME_MAP, _build_x_pool    # noqa: E402
from niveau1_r2_seuils import audit_pool, build_model                          # noqa: E402
import extended_heuristic_search as ehs                                        # noqa: E402

# Architectures finalistes (DECISIONS.md), toutes en regime brut.
ARCHS = {
    "A":        ("lstm", {"Forget": "wavkan_dog", "Input": "efficientkan",
                          "Candidate": "wavkan_dog", "Output": "linear"}),
    "B":        ("lstm", {"Forget": "relukan", "Input": "efficientkan",
                          "Candidate": "wavkan_dog", "Output": "linear"}),
    "dog":      ("lstm", {"Forget": "wavkan_dog", "Input": "wavkan_dog",
                          "Candidate": "wavkan_dog", "Output": "linear"}),
    "uniforme": ("lstm", {"Forget": "hybrid", "Input": "hybrid",
                          "Candidate": "hybrid", "Output": "hybrid"}),
    "gru":      ("gru",  {"Forget": "wavkan_dog", "Input": "efficientkan",
                          "Candidate": "wavkan_dog"}),
}
ARCH_LABELS = {
    "A": "A : DoG / EfficientKAN / DoG / Lin.",
    "B": "B : ReLU-KAN / EfficientKAN / DoG / Lin.",
    "dog": "DoG / DoG / DoG / Lin.",
    "uniforme": "Uniforme : hybride x 4",
    "gru": "GRU : DoG / EfficientKAN / DoG",
}
GATES = {
    "lstm": [("forget_gate", "Oubli"), ("input_gate", "Entrée"),
             ("candidate_gate", "Candidate"), ("output_gate", "Sortie")],
    "gru":  [("reset_gate", "Reset"), ("update_gate", "Mise à jour"),
             ("candidate_gate", "Candidate")],
}
FEATURE_LABELS = {
    "amount": "Montant (FCFA)",
    "oldBalanceOrig": "Solde émetteur avant (FCFA)",
    "newBalanceOrig": "Solde émetteur après (FCFA)",
    "oldBalanceDest": "Solde destinataire avant (FCFA)",
    "newBalanceDest": "Solde destinataire après (FCFA)",
    **{c: c.replace("action_", "Opération ") for c in ACTION_ONEHOT_COLS},
}
COLORS = ["#14377B", "#2E86C8", "#38A1E8", "#7A8CA8", "#0B1F47"]
FONT = dict(family="Montserrat, Helvetica, Arial, sans-serif", color="#14377B", size=12)

THETA_PRUNE = 0.01   # survie d'une arete (identique a _compute_r2_symbolic)
FIT_DOMAIN = 2.0     # grille de l'ajustement symbolique (identique au benchmark)
TOL_MONOTONE = 0.05  # variation de sens contraire < 5 % de l'amplitude -> courbe monotone
# Familles dont z* = -b/a a un sens : nature du point caracteristique.
POINT_CARACTERISTIQUE = {
    "sigmoid(x)": "centre de transition", "tanh(x)": "centre de transition",
    "softplus(x)": "coude", "|x|": "coude", "sqrt(|x|)": "coude", "log(|x|+eps)": "coude",
    "x^2": "extremum",
}


def _fr(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def _fcfa(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " FCFA"


def to_original(z, mean: float, std: float) -> np.ndarray:
    """Inverse de build_regime_frame + standardize (variable numerique) : z -> FCFA."""
    u = np.asarray(z) * std + mean
    return np.sign(u) * np.expm1(np.abs(u))


def to_z(x: float, mean: float, std: float) -> float:
    return (np.sign(x) * np.log1p(abs(x)) - mean) / std


def train_stats(train_path: str) -> dict:
    """mu_train / std_train du regime brut, calcules comme dans load_and_window."""
    df, cols = build_regime_frame(read_momtsim_parquet(train_path), "brut")
    stats = standardize(df, cols)
    del df
    return stats


def pick_run(runs: list, cell: str, bases: dict, seed) -> dict:
    """Run de l'Etape 4 correspondant a l'architecture ; seed=None -> graine de MCC median."""
    cands = [r for r in runs if r["cell_type"] == cell and r["regime"] == "raw"
             and r["base_by_gate"] == bases]
    if not cands:
        raise ValueError(f"aucun run pour {cell} {bases}")
    if seed is not None:
        return next(r for r in cands if r["seed"] == seed)
    cands.sort(key=lambda r: r["metrics"]["mcc"])
    return cands[len(cands) // 2]


def fit_symbolic_per_edge(curves: torch.Tensor) -> list:
    """Meme ajustement que _compute_r2_symbolic (vectorise, 150 pas DMLAdam, lr=0,05),
    mais renvoie pour chaque arete la meilleure famille, ses parametres et son R2_e."""
    n_edges = curves.shape[0]
    x_row = torch.linspace(-FIT_DOMAIN, FIT_DOMAIN, curves.shape[1]).unsqueeze(0)
    best = [{"famille": None, "r2": -float("inf"), "a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
            for _ in range(n_edges)]
    with torch.enable_grad():
        for name, fn in ehs.SYMBOLIC_LIBRARY.items():
            a = torch.nn.Parameter(torch.ones(n_edges, 1))
            b = torch.nn.Parameter(torch.zeros(n_edges, 1))
            c = torch.nn.Parameter(torch.ones(n_edges, 1))
            d = torch.nn.Parameter(torch.zeros(n_edges, 1))
            optimizer = ehs.DMLAdam([a, b, c, d], lr=0.05)
            diverged = torch.zeros(n_edges, dtype=torch.bool)
            for _ in range(150):
                optimizer.zero_grad(set_to_none=True)
                pred = c * fn(a * x_row + b) + d
                diverged = diverged | torch.isnan(pred).any(dim=1) | torch.isinf(pred).any(dim=1)
                loss = (pred - curves).square().mean(dim=1)
                torch.where(diverged, torch.zeros_like(loss), loss).sum().backward()
                optimizer.step()
            with torch.no_grad():
                pred = c * fn(a * x_row + b) + d
                ss_res = (curves - pred).square().sum(dim=1)
                ss_tot = (curves - curves.mean(dim=1, keepdim=True)).square().sum(dim=1)
                r2 = 1.0 - ss_res * (ss_tot + 1e-12).reciprocal()
                r2 = torch.where(diverged | torch.isnan(r2) | torch.isinf(r2),
                                 torch.full_like(r2, -float("inf")), r2)
            for e in range(n_edges):
                if float(r2[e]) > best[e]["r2"]:
                    best[e] = {"famille": name, "r2": float(r2[e]), "a": float(a[e]),
                               "b": float(b[e]), "c": float(c[e]), "d": float(d[e])}
    return best


def read_edge(fit: dict, y_obs: np.ndarray, stats: tuple, r2_thr: float) -> dict:
    """Lecture d'une arete : point caracteristique tire de la formule (si retenue et situe
    dans le domaine d'ajustement) et sens de variation sur la plage observee."""
    out = {"point_caracteristique": None, "point_fcfa": None}
    nature = POINT_CARACTERISTIQUE.get(fit["famille"])
    if fit["r2"] >= r2_thr and nature and abs(fit["a"]) > 1e-6:
        z_star = -fit["b"] / fit["a"]
        if -FIT_DOMAIN <= z_star <= FIT_DOMAIN:
            out = {"point_caracteristique": nature,
                   "point_fcfa": float(to_original(z_star, *stats))}
    dy = np.diff(y_obs)
    amp = float(np.ptp(y_obs)) or 1e-12
    rise, fall = float(dy[dy > 0].sum()), float(-dy[dy < 0].sum())
    if fall <= TOL_MONOTONE * amp:
        sens = "croissante"
    elif rise <= TOL_MONOTONE * amp:
        sens = "décroissante"
    else:
        k = int(np.argmax(y_obs)) if rise >= fall else int(np.argmin(y_obs))
        sens = "non monotone"
        out["extremum_observe_index"] = k
    return {**out, "sens": sens, "amplitude": amp}


def label_edge(e: dict, r2_thr: float) -> str:
    f, rd = e["fit"], e["lecture"]
    s = f"u{e['unite']} : {f['famille']}, R² {_fr(f['r2'], 2)}"
    if f["r2"] < r2_thr:
        return s + " (non retenue)"
    s += f", {rd['sens']}"
    if rd["point_fcfa"] is not None:
        s += f", {rd['point_caracteristique']} ≈ {_fcfa(rd['point_fcfa'])}"
    return s


def gate_input_importance(model, cell: str, x_pool: torch.Tensor) -> dict:
    """{attr: (l1 (n_inputs, out), mask survivants)} restreint aux entrees x."""
    h = model.hidden_size
    out = {}
    with torch.no_grad():
        for attr, _ in GATES[cell]:
            l1 = getattr(model.cell, attr).edge_activations(x_pool).abs().mean(dim=0)[h:, :]
            out[attr] = (l1, l1 > THETA_PRUNE)
    return out


def edge_curve(layer, i: int, j: int, z: np.ndarray) -> np.ndarray:
    """phi_ij sur la grille z, autres entrees a zero (convention de _compute_r2_symbolic)."""
    x = torch.zeros(len(z), layer.in_features)
    x[:, i] = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    with torch.no_grad():
        return layer.edge_activations(x)[:, i, j].numpy()


def _save_fig(fig, out_path_no_ext: str, export_png: bool = True) -> None:
    os.makedirs(os.path.dirname(out_path_no_ext) or ".", exist_ok=True)
    fig.write_html(out_path_no_ext + ".html")
    if export_png:
        try:
            fig.write_image(out_path_no_ext + ".png", scale=2)
        except Exception as exc:   # kaleido peut manquer selon l'environnement
            print(f"[avertissement] export PNG impossible pour {out_path_no_ext} ({exc}).")
    print(out_path_no_ext + ".html")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out_dir", default=os.path.join("step4_outputs", "figures_aretes"))
    p.add_argument("--results_dir", default="step4_outputs")
    # Jeux du 8 aout, ceux de l'Etape 4 (DECISIONS.md, 2026-10-31).
    p.add_argument("--train_path", default=os.path.join(
        "..", "..", "MOMTSIM", "config", "archives", "featuresLog_20261027_004532.parquet"))
    p.add_argument("--val_path", default=os.path.join(
        "..", "..", "data", "archives", "val_features_20261027_004542.parquet"))
    p.add_argument("--n_windows_train", type=int, default=50_000)
    p.add_argument("--archs", nargs="+", default=["A", "B", "dog", "uniforme"],
                   choices=list(ARCHS))
    p.add_argument("--seed", type=int, default=None,
                   help="Graine du modele ; par defaut la graine de MCC median.")
    p.add_argument("--n_vars", type=int, default=len(RAW_NUMERIC_COLS),
                   help="Variables numeriques tracees, par importance decroissante.")
    p.add_argument("--top_k", type=int, default=2, help="Aretes tracees par variable et porte.")
    p.add_argument("--r2_threshold", type=float, default=0.80)
    p.add_argument("--no_png", action="store_true")
    args = p.parse_args()

    out_dir = os.path.join(_ETUDE_DIR, args.out_dir)
    results_dir = os.path.join(_ETUDE_DIR, args.results_dir)
    with open(os.path.join(results_dir, "step4_level1_results.json"), encoding="utf-8") as f:
        d = json.load(f)
    runs = d["results"] + d["baseline_results"]

    print("[stats] mu_train / std_train du regime brut ...", flush=True)
    stats = train_stats(os.path.join(_ETUDE_DIR, args.train_path))
    cols = list(stats)                                   # ordre des colonnes du modele
    num_idx = [cols.index(c) for c in RAW_NUMERIC_COLS]
    act_idx = [cols.index(c) for c in ACTION_ONEHOT_COLS]

    pools, summary, loaded = {}, {}, []
    for arch in args.archs:
        cell, bases = ARCHS[arch]
        run = pick_run(runs, cell, bases, args.seed)
        W = CELL_THETA_OPT[cell]["W"]
        key = (REGIME_MAP["raw"], W)
        if key not in pools:
            pools[key] = audit_pool(os.path.join(_ETUDE_DIR, args.train_path),
                                    os.path.join(_ETUDE_DIR, args.val_path),
                                    key[0], W, args.n_windows_train)
        pool = pools[key]
        model = build_model(cell, run["base_by_gate"], pool["input_size"])
        model.load_state_dict(torch.load(os.path.join(results_dir, "models", run["key_hash"] + ".pt"),
                                         map_location="cpu", weights_only=True), strict=True)
        model.eval()
        x_pool = _build_x_pool(model, cell, pool["x_audit_pool"], W)
        imp = gate_input_importance(model, cell, x_pool)
        loaded.append((arch, cell, run, model, pool, imp))
        print(f"[{arch}] run {run['key_hash']} (graine {run['seed']})", flush=True)

    # ── 1. Importance variable x porte (aretes survivantes seulement) ────────────
    fig = make_subplots(rows=1, cols=len(loaded), horizontal_spacing=0.10,
                        subplot_titles=[ARCH_LABELS[a] for a, *_ in loaded])
    for col, (arch, cell, run, model, pool, imp) in enumerate(loaded, start=1):
        gates = GATES[cell]
        mat = np.array([[float((imp[a][0][k] * imp[a][1][k]).sum()) for a, _ in gates]
                        for k in range(len(cols))])
        n_surv = np.array([[int(imp[a][1][k].sum()) for a, _ in gates] for k in range(len(cols))])
        rel = mat / (mat.sum() or 1.0)
        fig.add_trace(go.Heatmap(
            z=rel, x=[g[1] for g in gates], y=[FEATURE_LABELS[c] for c in cols],
            colorscale=[[0, "#F7F7FB"], [0.5, "#38A1E8"], [1, "#14377B"]],
            text=[[f"{_fr(100 * v, 1)} %<br>{n} ar." for v, n in zip(r, nr)]
                  for r, nr in zip(rel, n_surv)],
            texttemplate="%{text}", textfont=dict(size=9), showscale=(col == len(loaded)),
            colorbar=dict(title="Part de L1")), row=1, col=col)
        fig.update_yaxes(autorange="reversed", showticklabels=(col == 1), row=1, col=col)
        summary[arch] = {
            "key_hash": run["key_hash"], "seed": run["seed"], "mcc": run["metrics"]["mcc"],
            "importance": {c: {g[1]: {"part_l1": float(rel[k, n]), "aretes_survivantes": int(n_surv[k, n])}
                               for n, g in enumerate(gates)} for k, c in enumerate(cols)},
            "variables": {}}
    fig.update_layout(
        title=dict(text=("Importance des arêtes d'entrée par variable et par porte"
                         "<br><sup>Part de la norme L1 totale des arêtes d'entrée survivantes "
                         f"(L1 > {_fr(THETA_PRUNE)}), nombre d'arêtes survivantes ; "
                         "lot d'audit de 256 fenêtres</sup>"), x=0.01),
        template="plotly_white", font=FONT, height=560, width=420 * len(loaded))
    _save_fig(fig, os.path.join(out_dir, "importance_variables"), not args.no_png)

    # ── 2. Courbes phi des aretes les plus fortes, variables numeriques par importance ──
    for arch, cell, run, model, pool, imp in loaded:
        gates = GATES[cell]
        h = model.hidden_size
        total = {k: sum(float((imp[a][0][k] * imp[a][1][k]).sum()) for a, _ in gates) for k in num_idx}
        ranked = [k for k in sorted(num_idx, key=lambda k: -total[k]) if total[k] > 0][:args.n_vars]
        if not ranked:
            print(f"[{arch}] aucune arete numerique survivante")
            continue

        # Grilles par variable : axe trace (>= 1 FCFA pour l'echelle log) et plage observee.
        grids = {}
        for k in ranked:
            mean, std = stats[cols[k]]
            obs = pool["x_audit_pool"][:, :, k].numpy().ravel()
            z_lo = max(-4.0, to_z(1.0, mean, std))
            p1, p5, p95, p99 = np.percentile(np.clip(obs, z_lo, 4.0), [1, 5, 95, 99])
            grids[k] = dict(z_plot=np.linspace(z_lo, 4.0, 200), z_obs=np.linspace(p1, p99, 200),
                            band=to_original([p5, p95], mean, std), stats=(mean, std))

        edges = []
        for k in ranked:
            for attr, glabel in gates:
                l1, mask = imp[attr]
                units = [j for j in torch.argsort(l1[k], descending=True).tolist() if mask[k, j]]
                layer = getattr(model.cell, attr)
                for j in units[:args.top_k]:
                    edges.append({"k": k, "attr": attr, "porte": glabel, "unite": j,
                                  "importance_l1": float(l1[k, j]),
                                  "y_plot": edge_curve(layer, h + k, j, grids[k]["z_plot"]),
                                  "y_obs": edge_curve(layer, h + k, j, grids[k]["z_obs"]),
                                  "y_fit": torch.as_tensor(edge_curve(
                                      layer, h + k, j, np.linspace(-FIT_DOMAIN, FIT_DOMAIN, 100)))})
        for e, fit in zip(edges, fit_symbolic_per_edge(torch.stack([e["y_fit"] for e in edges]))):
            e["fit"] = fit
            e["lecture"] = read_edge(fit, e["y_obs"], grids[e["k"]]["stats"], args.r2_threshold)

        fig = make_subplots(rows=len(ranked), cols=len(gates), horizontal_spacing=0.05,
                            vertical_spacing=0.09,
                            column_titles=[f"Porte {g[1]}" for g in gates],
                            row_titles=[FEATURE_LABELS[cols[k]] for k in ranked])
        z_fit = np.linspace(-FIT_DOMAIN, FIT_DOMAIN, 100)
        for r, k in enumerate(ranked, start=1):
            g = grids[k]
            x_plot = to_original(g["z_plot"], *g["stats"])
            for c, (attr, _) in enumerate(gates, start=1):
                # Axe log : Plotly attend les bornes des formes en log10.
                fig.add_vrect(x0=np.log10(max(g["band"][0], 1.0)), x1=np.log10(g["band"][1]),
                              fillcolor="#38A1E8", opacity=0.10, line_width=0, row=r, col=c)
                cell_edges = [e for e in edges if e["k"] == k and e["attr"] == attr]
                for n, e in enumerate(cell_edges):
                    color = COLORS[n % len(COLORS)]
                    fig.add_trace(go.Scatter(x=x_plot, y=e["y_plot"], mode="lines",
                                             name=label_edge(e, args.r2_threshold), showlegend=False,
                                             line=dict(color=color, width=2.2)), row=r, col=c)
                    f = e["fit"]
                    fn = ehs.SYMBOLIC_LIBRARY[f["famille"]]
                    y_sym = (f["c"] * fn(torch.as_tensor(f["a"] * z_fit + f["b"], dtype=torch.float32))
                             + f["d"]).numpy()
                    fig.add_trace(go.Scatter(x=to_original(z_fit, *g["stats"]), y=y_sym, mode="lines",
                                             showlegend=False, hoverinfo="skip",
                                             line=dict(color=color, width=1.3, dash="dot")),
                                  row=r, col=c)
                n_sub = (r - 1) * len(gates) + c
                ax = "" if n_sub == 1 else str(n_sub)
                txt = "<br>".join(label_edge(e, args.r2_threshold) for e in cell_edges) \
                    or "aucune arête survivante"
                fig.add_annotation(text=txt, xref=f"x{ax} domain", yref=f"y{ax} domain",
                                   x=0.02, y=0.98, xanchor="left", yanchor="top", showarrow=False,
                                   align="left", font=dict(size=8), bgcolor="rgba(255,255,255,0.8)")
                fig.update_xaxes(type="log", row=r, col=c)
        fig.update_layout(
            title=dict(text=(f"Fonctions φ apprises des arêtes d'entrée, {ARCH_LABELS[arch]} "
                             f"(graine {run['seed']}, MCC {_fr(run['metrics']['mcc'], 4)})"
                             f"<br><sup>Variables par importance décroissante ; {args.top_k} arêtes les "
                             "plus fortes par porte ; trait plein : φ apprise, pointillé : formule "
                             "symbolique sur z ∈ [-2, 2] ; bande : P5-P95 observé ; axe vertical : "
                             "contribution à la pré-activation de l'unité</sup>"), x=0.01),
            template="plotly_white", font=FONT, height=300 * len(ranked) + 140,
            width=380 * len(gates), margin=dict(t=110))
        _save_fig(fig, os.path.join(out_dir, f"{arch}_courbes_phi"), not args.no_png)

        for k in ranked:
            summary[arch]["variables"][cols[k]] = {
                "importance_rang": ranked.index(k) + 1,
                "plage_observee_p1_p99_fcfa": to_original(grids[k]["z_obs"][[0, -1]], *grids[k]["stats"]).tolist(),
                "aretes": [{"porte": e["porte"], "unite": e["unite"], "importance_l1": e["importance_l1"],
                            "formule": {kk: e["fit"][kk] for kk in ("famille", "a", "b", "c", "d")},
                            "r2_e": e["fit"]["r2"], "formule_retenue": e["fit"]["r2"] >= args.r2_threshold,
                            **{kk: v for kk, v in e["lecture"].items() if kk != "extremum_observe_index"}}
                           for e in edges if e["k"] == k]}
        for e in edges:
            print(f"   {cols[e['k']]:>15} -> {e['porte']:<11} {label_edge(e, args.r2_threshold)}")

    # ── 3. Effet net sur le score ──────────────────────────────────────────────
    fig = make_subplots(rows=1, cols=len(num_idx), horizontal_spacing=0.04,
                        subplot_titles=[FEATURE_LABELS[cols[k]] for k in num_idx], shared_yaxes=True)
    fig_act = go.Figure()
    for n, (arch, cell, run, model, pool, imp) in enumerate(loaded):
        X = pool["x_audit_pool"]
        color = COLORS[n % len(COLORS)]
        summary[arch]["effet_net"] = {}
        for c, k in enumerate(num_idx, start=1):
            mean, std = stats[cols[k]]
            z_grid = np.linspace(max(-4.0, to_z(1.0, mean, std)), 4.0, 50)
            with torch.no_grad():
                s = []
                for z in z_grid:
                    Xz = X.clone()
                    Xz[:, -1, k] = float(z)
                    s.append(float(model(Xz).mean()))
            x_grid = to_original(z_grid, mean, std)
            summary[arch]["effet_net"][cols[k]] = {"x_fcfa": x_grid.tolist(), "score_moyen": s}
            fig.add_trace(go.Scatter(x=x_grid, y=s, mode="lines", name=ARCH_LABELS[arch],
                                     legendgroup=arch, showlegend=(c == 1),
                                     line=dict(color=color, width=2.2)), row=1, col=c)
            fig.update_xaxes(type="log", row=1, col=c)
        # Type d'operation du dernier pas : one-hot standardise (valeurs 0 et 1 -> z).
        s_act = []
        for k in act_idx:
            Xz = X.clone()
            for kk in act_idx:
                mean, std = stats[cols[kk]]
                Xz[:, -1, kk] = float(((1.0 if kk == k else 0.0) - mean) / std)
            with torch.no_grad():
                s_act.append(float(model(Xz).mean()))
        summary[arch]["effet_net"]["type_operation"] = dict(zip(ACTION_ONEHOT_COLS, s_act))
        fig_act.add_trace(go.Bar(x=[FEATURE_LABELS[c] for c in ACTION_ONEHOT_COLS], y=s_act,
                                 name=ARCH_LABELS[arch], marker_color=color))
    fig.update_yaxes(title_text="Score de fraude moyen", row=1, col=1)
    fig.update_layout(
        title=dict(text=("Effet net de chaque variable numérique sur le score de fraude"
                         "<br><sup>Valeur du dernier pas remplacée sur les 256 fenêtres d'audit, "
                         "autres entrées inchangées (les combinaisons obtenues ne respectent pas "
                         "toujours la cohérence montant/soldes)</sup>"), x=0.01),
        template="plotly_white", font=FONT, height=460, width=330 * len(num_idx),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    _save_fig(fig, os.path.join(out_dir, "score_variables_numeriques"), not args.no_png)
    fig_act.update_layout(
        title=dict(text=("Score de fraude moyen selon le type d'opération du dernier pas"
                         "<br><sup>Type d'opération imposé sur les 256 fenêtres d'audit, "
                         "autres entrées inchangées</sup>"), x=0.01),
        yaxis_title="Score de fraude moyen", barmode="group",
        template="plotly_white", font=FONT, height=460, width=1000,
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    _save_fig(fig_act, os.path.join(out_dir, "score_type_operation"), not args.no_png)

    with open(os.path.join(out_dir, "aretes_resume.json"), "w", encoding="utf-8") as f:
        json.dump({"r2_threshold": args.r2_threshold, "theta_prune": THETA_PRUNE,
                   "architectures": summary}, f, indent=1, ensure_ascii=False)
    print(os.path.join(out_dir, "aretes_resume.json"))


if __name__ == "__main__":
    main()
