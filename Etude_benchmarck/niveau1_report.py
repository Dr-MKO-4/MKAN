"""
niveau1_report.py -- Rapport + graphiques de l'Etape 4 (Niveau 1) reconstruits
UNIQUEMENT a partir de ce qui est deja sauvegarde sur disque, sans aucun
reentrainement :

    step4_outputs/step4_progress.jsonl   (1 ligne par run termine)
    step4_outputs/val_scores/*.npy       (scores de validation par run)
    step4_outputs/models/*.pt            (poids finaux par run)

Appele automatiquement par niveau1_benchmark_run.py apres CHAQUE run (le
rapport reflete donc toujours l'etat courant, meme partiel), et utilisable
seul a tout moment, y compris pendant que le benchmark tourne :

    python niveau1_report.py
    python niveau1_report.py --out_dir step4_outputs --no_figures

Sorties (dans out_dir) :
    step4_level1_results.json, step4_runs.csv,
    best_heterogeneous_config.json, RAPPORT_SYNTHESE_ETAPE4.md,
    figures/fitness_par_architecture.{html,png}, figures/courbes_loss_top5.{html,png},
    figures/courbes_pr_meilleure_vs_baseline.{html,png}  (Plotly + kaleido)
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

ACCENT = "#2f6fd0"      # meilleure architecture heterogene
NEUTRAL = "#a3a8b0"     # autres architectures
BASELINE = "#3c3f44"    # MKAN uniforme (hybrid)
INK = "#2b2d31"


def _load_jsonl(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # ligne tronquee par un crash
    return rows


def _arch_label(r: dict) -> str:
    cell = "GRU" if r["cell_type"] == "gru" else "TKAN"
    order = ["Forget", "Input", "Candidate", "Output"]
    bases = "/".join(r["base_by_gate"][g] for g in order if g in r["base_by_gate"])
    return f"{cell} [{r['regime']}] {bases}"


def _arch_key(r: dict) -> tuple:
    return (r["cell_type"], r["regime"], tuple(sorted(r["base_by_gate"].items())))


def load_results(out_dir: str, run_signature: str = None):
    rows = _load_jsonl(os.path.join(out_dir, "step4_progress.jsonl"))
    if not rows:
        return [], [], None
    if run_signature is None:
        # signature la plus representee (en cas de journal mixte)
        run_signature = Counter(r.get("run_signature") for r in rows).most_common(1)[0][0]
    latest = {}
    for r in rows:
        if r.get("run_signature") == run_signature:
            latest[r["key"]] = r["result"]   # derniere ecriture gagnante
    results = [v for k, v in latest.items() if not k.startswith("baseline::")]
    baseline = [v for k, v in latest.items() if k.startswith("baseline::")]
    return results, baseline, run_signature


def aggregate(results: list) -> list:
    """[(arch_key, label, fitness_moy, fitness_std, n_graines, runs)] tries."""
    by = defaultdict(list)
    for r in results:
        by[_arch_key(r)].append(r)
    agg = []
    for k, runs in by.items():
        fit = np.array([x["metrics"]["fitness_total"] for x in runs], dtype=float)
        agg.append((k, _arch_label(runs[0]), float(np.mean(fit)),
                    float(np.std(fit, ddof=1)) if len(fit) > 1 else 0.0, len(fit), runs))
    agg.sort(key=lambda a: -a[2])
    return agg


def _metric_mean(runs, name):
    vals = [r["metrics"].get(name) for r in runs]
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Figures (Plotly -- convention du projet, pas matplotlib) : .html interactif
# + .png via kaleido (insertion LaTeX), meme convention que niveau0_analysis.py.
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, out_path_no_ext: str, export_png: bool = True) -> str:
    os.makedirs(os.path.dirname(out_path_no_ext) or ".", exist_ok=True)
    html_path = out_path_no_ext + ".html"
    fig.write_html(html_path)
    if not export_png:
        return html_path
    try:
        fig.write_image(out_path_no_ext + ".png", scale=2)
    except Exception as exc:   # kaleido peut manquer selon l'environnement
        print(f"[avertissement] export PNG impossible pour {out_path_no_ext} ({exc}) -- "
              f"HTML seul disponible ({html_path}).")
    return html_path


def make_figures(out_dir: str, agg: list, baseline: list, export_png: bool = True) -> dict:
    import plotly.graph_objects as go

    fig_dir = os.path.join(out_dir, "figures")
    paths = {}

    # 1. Fitness moyenne (+/- ecart-type entre graines) par architecture x regime
    labels = [f"{a[1]}  (n={a[4]})" for a in agg]
    fig = go.Figure(go.Bar(
        x=[a[2] for a in agg], y=labels, orientation="h",
        error_x=dict(type="data", array=[a[3] for a in agg], color=INK, thickness=1, width=3),
        marker_color=[ACCENT] + [NEUTRAL] * (len(agg) - 1),
        customdata=[[a[3], _metric_mean(a[5], "mcc"), _metric_mean(a[5], "auc_pr"),
                     _metric_mean(a[5], "brier")] for a in agg],
        hovertemplate=("%{y}<br>fitness=%{x:.4f} +/- %{customdata[0]:.4f}"
                       "<br>MCC=%{customdata[1]:.4f}<br>PR-AUC=%{customdata[2]:.4f}"
                       "<br>Brier=%{customdata[3]:.4f}<extra></extra>"),
        showlegend=False,
    ))
    if baseline:
        b_mean = float(np.mean([r["metrics"]["fitness_total"] for r in baseline]))
        fig.add_vline(x=b_mean, line_dash="dash", line_color=BASELINE, line_width=1.5,
                      annotation_text=f"MKAN uniforme hybrid ({b_mean:.4f})",
                      annotation_position="bottom right")
    lo = min(a[2] - a[3] for a in agg)
    hi = max(a[2] + a[3] for a in agg)
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    fig.update_layout(
        title="Etape 4 -- fitness par architecture heterogene (moyenne +/- ecart-type entre graines)",
        xaxis_title="Fitness totale (AHP)", xaxis_range=[lo - pad, hi + pad],
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        template="plotly_white", height=max(400, 26 * len(agg) + 160), width=1000,
    )
    paths["Figure fitness"] = _save_fig(fig, os.path.join(fig_dir, "fitness_par_architecture"), export_png)

    # 2. Courbes de perte (moyenne entre graines) des 5 meilleures architectures
    top = [a for a in agg[:5] if any(r["metrics"].get("loss_history") for r in a[5])]
    if top:
        fig = go.Figure()
        for rank, a in enumerate(top):
            hist = [r["metrics"]["loss_history"] for r in a[5] if r["metrics"].get("loss_history")]
            L = min(len(h) for h in hist)
            curve = np.mean([h[:L] for h in hist], axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(1, L + 1)), y=curve, mode="lines+markers",
                name=f"#{rank + 1} {a[1]}",
                line=dict(width=2, color=ACCENT if rank == 0 else NEUTRAL),
                marker=dict(size=8), opacity=1.0 if rank == 0 else 0.6,
                hovertemplate="epoque %{x}<br>perte=%{y:.5f}<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(title="Convergence -- Top 5 architectures (perte moyenne entre graines)",
                          xaxis_title="Epoque", yaxis_title="Perte d'entrainement totale",
                          template="plotly_white", height=500, width=1000,
                          hovermode="x unified")
        paths["Figure convergence"] = _save_fig(fig, os.path.join(fig_dir, "courbes_loss_top5"), export_png)

    # 3. Courbes precision-rappel : meilleure architecture vs baseline (une
    #    courbe par graine), depuis les scores de validation sauvegardes.
    from sklearn.metrics import precision_recall_curve
    scores_dir = os.path.join(out_dir, "val_scores")

    def _pr_runs(runs, max_points=2000):
        out = []
        for r in runs:
            h, yf = r.get("key_hash"), r.get("y_val_file")
            sp = os.path.join(scores_dir, f"{h}.npy") if h else None
            yp = os.path.join(scores_dir, yf) if yf else None
            if sp and yp and os.path.exists(sp) and os.path.exists(yp):
                y, s = np.load(yp), np.load(sp)
                if len(y) == len(s) and len(np.unique(y)) == 2:
                    prec, rec, _ = precision_recall_curve(y, s)
                    # sous-echantillonnage des points (HTML leger sur des millions de scores)
                    idx = np.unique(np.linspace(0, len(rec) - 1, max_points).astype(int))
                    out.append((r["seed"], rec[idx], prec[idx]))
        return out

    best_pr = _pr_runs(agg[0][5]) if agg else []
    base_pr = _pr_runs(baseline)
    if best_pr or base_pr:
        fig = go.Figure()
        for i, (seed, rec, prec) in enumerate(best_pr):
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", legendgroup="best",
                                     name=f"Heterogene : {agg[0][1]}", showlegend=(i == 0),
                                     line=dict(color=ACCENT, width=2), opacity=0.8,
                                     hovertemplate=f"graine {seed}<br>rappel=%{{x:.3f}}<br>"
                                                   f"precision=%{{y:.3f}}<extra></extra>"))
        for i, (seed, rec, prec) in enumerate(base_pr):
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", legendgroup="base",
                                     name="MKAN uniforme (hybrid)", showlegend=(i == 0),
                                     line=dict(color=BASELINE, width=2, dash="dash"), opacity=0.8,
                                     hovertemplate=f"graine {seed}<br>rappel=%{{x:.3f}}<br>"
                                                   f"precision=%{{y:.3f}}<extra></extra>"))
        fig.update_layout(title="Precision-rappel en validation (une courbe par graine)",
                          xaxis_title="Rappel", yaxis_title="Precision",
                          xaxis_range=[0, 1], yaxis_range=[0, 1.02],
                          template="plotly_white", height=550, width=750,
                          legend=dict(x=0.02, y=0.05))
        paths["Figure precision-rappel"] = _save_fig(
            fig, os.path.join(fig_dir, "courbes_pr_meilleure_vs_baseline"), export_png)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Rapport
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_reports(out_dir: str, run_signature: str = None, metadata: dict = None,
                  figures: bool = True, export_png: bool = True,
                  quiet: bool = False) -> dict:
    """figures=False : JSON/CSV/rapport seulement. export_png=False : figures
    en HTML seul (kaleido peut se figer -- jamais pendant le benchmark)."""
    results, baseline, run_signature = load_results(out_dir, run_signature)
    paths = {}
    if not results:
        if not quiet:
            print("Aucun run termine dans", out_dir)
        return paths
    agg = aggregate(results)
    best = agg[0]
    now = datetime.now(timezone.utc).isoformat()

    meta = dict(metadata or {})
    meta.update({"run_signature": run_signature, "n_runs_termines": len(results),
                 "n_runs_baseline": len(baseline), "genere_le": now})

    p = os.path.join(out_dir, "step4_level1_results.json")
    _atomic_write(p, json.dumps({"metadata": meta, "results": results,
                                 "baseline_results": baseline},
                                indent=2, ensure_ascii=False))
    paths["Resultats"] = p

    cols = ["mcc", "auc_pr", "auc_roc", "brier", "r2_symbolic", "latency_ms",
            "fitness_total", "train_time_s", "n_train_windows", "n_val_windows"]
    p = os.path.join(out_dir, "step4_runs.csv")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "architecture", "cell_type", "regime", "seed"] + cols)
        for kind, rows in (("heterogene", results), ("baseline", baseline)):
            for r in rows:
                w.writerow([kind, _arch_label(r), r["cell_type"], r["regime"], r["seed"]]
                           + [r["metrics"].get(c) for c in cols])
    os.replace(tmp, p)
    paths["CSV par run"] = p

    b_mean = (float(np.mean([r["metrics"]["fitness_total"] for r in baseline]))
              if baseline else None)
    best_cfg = {
        "cell_type": best[0][0], "regime": best[0][1], "base_by_gate": dict(best[0][2]),
        "fitness_mean": best[2], "fitness_std": best[3], "n_seeds": best[4],
        "baseline": ({"fitness_mean": b_mean, "n_seeds": len(baseline),
                      "gain_vs_baseline": best[2] - b_mean} if baseline else None),
    }
    p = os.path.join(out_dir, "best_heterogeneous_config.json")
    _atomic_write(p, json.dumps(best_cfg, indent=2, ensure_ascii=False))
    paths["Meilleure config"] = p

    fig_paths = {}
    if figures:
        try:
            fig_paths = make_figures(out_dir, agg, baseline, export_png)
        except Exception as exc:   # noqa: BLE001
            if not quiet:
                print(f"[avertissement] figures non generees : {exc}")
    paths.update(fig_paths)

    lines = [
        "# Rapport de synthese -- Etape 4 (Niveau 1)",
        "",
        f"Genere le {now} -- {len(results)} runs termines, {len(agg)} architectures x "
        f"regime, signature {run_signature}.",
    ]
    if metadata:
        lines.append(f"Parametres : {json.dumps(metadata, ensure_ascii=False)}")
    lines += [
        "",
        "## Meilleure architecture heterogene (a ce stade)",
        "",
        f"- {best[1]}",
        f"- Fitness moyenne : {best[2]:.4f} +/- {best[3]:.4f} ({best[4]} graine(s))",
        f"- MCC moyen : {_metric_mean(best[5], 'mcc'):.4f} -- PR-AUC moyen : "
        f"{_metric_mean(best[5], 'auc_pr'):.4f} -- Brier moyen : {_metric_mean(best[5], 'brier'):.4f}",
        "",
        "## Comparaison au MKAN uniforme (hybrid sur toutes les portes)",
        "",
    ]
    if baseline:
        lines += [f"- Fitness moyenne baseline : {b_mean:.4f} ({len(baseline)} graine(s))",
                  f"- Gain heterogene - uniforme : {best[2] - b_mean:+.4f}"]
    else:
        lines.append("- Baseline pas encore calculee.")
    lines += [
        "",
        "## Classement (architecture x regime)",
        "",
        "| Rang | Architecture | n | Fitness moy. | Ecart-type | MCC | PR-AUC | Brier | R2 symb. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, a in enumerate(agg, start=1):
        lines.append(f"| {i} | {a[1]} | {a[4]} | {a[2]:.4f} | {a[3]:.4f} | "
                     f"{_metric_mean(a[5], 'mcc'):.4f} | {_metric_mean(a[5], 'auc_pr'):.4f} | "
                     f"{_metric_mean(a[5], 'brier'):.4f} | {_metric_mean(a[5], 'r2_symbolic'):.4f} |")
    if fig_paths:
        lines += ["", "## Figures", ""]
        lines += [f"- {k} : `{os.path.relpath(v, out_dir)}`" for k, v in fig_paths.items()]
    p = os.path.join(out_dir, "RAPPORT_SYNTHESE_ETAPE4.md")
    _atomic_write(p, "\n".join(lines) + "\n")
    paths["Rapport"] = p
    return paths


def main():
    parser = argparse.ArgumentParser(description="Rapport Etape 4 depuis les sauvegardes.")
    parser.add_argument("--out_dir", type=str, default="step4_outputs")
    parser.add_argument("--signature", type=str, default=None,
                        help="run_signature a retenir (defaut : la plus representee).")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    paths = write_reports(out_dir, run_signature=args.signature,
                          figures=not args.no_figures)
    for k, v in paths.items():
        print(f"{k} : {v}")


if __name__ == "__main__":
    main()
