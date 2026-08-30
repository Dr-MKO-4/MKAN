"""
comparison_viz.py  Visualisations de comparaison MKAN / LSTM / XGBoost (section 5.x).

  plot_roc_pr_comparison   courbes ROC + Précision-Rappel sur un même canvas
  plot_metrics_table       tableau MCC / AUC-ROC / AP / F1 pour les 3 modèles
  compute_all_metrics      calcule toutes les métriques à partir des probas

Usage :
    from MKAN.comparison_viz import plot_roc_pr_comparison, plot_metrics_table
    from MKAN.baselines import predict_proba

    probas = {
        "MKAN":    predict_proba(mkan_model,  X_test, device, "mkan"),
        "LSTM":    predict_proba(lstm_model,  X_test, device, "lstm"),
        "XGBoost": predict_proba(xgb_model,   X_test, device, "xgb"),
    }
    fig = plot_roc_pr_comparison(probas, y_test)
    fig.show()
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, f1_score, brier_score_loss,
)

_COLORS = {
    "MKAN":    "#2196F3",
    "LSTM":    "#FF9800",
    "XGBoost": "#4CAF50",
}
_DASH = {
    "MKAN":    "solid",
    "LSTM":    "dash",
    "XGBoost": "dot",
}


def compute_all_metrics(probas: dict[str, np.ndarray],
                        y_true: np.ndarray,
                        threshold: float = 0.5) -> dict[str, dict]:
    """
    Calcule MCC, AUC-ROC, Average Precision et F1 pour chaque modèle.

    Args:
        probas    : {"MKAN": array(N,), "LSTM": ..., "XGBoost": ...}
        y_true    : labels binaires (N,)
        threshold : seuil de binarisation pour MCC et F1 (défaut 0.5)

    Returns:
        {"MKAN": {"MCC": ..., "AUC": ..., "AP": ..., "F1": ...}, ...}
    """
    results = {}
    for name, proba in probas.items():
        pred_bin    = (proba >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_true, proba)
        results[name] = {
            "MCC":   float(matthews_corrcoef(y_true, pred_bin)),
            "AUC":   float(auc(fpr, tpr)),
            "PRAUC": float(average_precision_score(y_true, proba)),
            "Brier": float(brier_score_loss(y_true, proba)),
            "F1":    float(f1_score(y_true, pred_bin, zero_division=0)),
        }
    return results


def plot_roc_pr_comparison(probas: dict[str, np.ndarray],
                            y_true: np.ndarray) -> go.Figure:
    """
    Figure 2 panneaux : courbe ROC (gauche) + courbe Précision-Rappel (droite).

    La courbe PR est plus informative que la ROC sur des données déséquilibrées
    (fraude rare) — les deux sont présentées pour couvrir les conventions académiques
    et réglementaires.

    Args:
        probas : {"MKAN": array(N,), "LSTM": ..., "XGBoost": ...}
        y_true : labels binaires (N,)
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Courbe ROC", "Courbe Précision-Rappel"],
        horizontal_spacing=0.12,
    )

    for name, proba in probas.items():
        color = _COLORS.get(name, "#999999")
        dash  = _DASH.get(name, "solid")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val     = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC = {auc_val:.3f})",
            line=dict(color=color, width=2.5, dash=dash),
        ), row=1, col=1)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap           = average_precision_score(y_true, proba)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name}  (AP = {ap:.3f})",
            line=dict(color=color, width=2.5, dash=dash),
            showlegend=False,
        ), row=1, col=2)

    # Référence aléatoire ROC
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color="lightgray", width=1),
        name="Aléatoire", showlegend=True,
    ), row=1, col=1)

    # Référence naïve PR (taux de fraude)
    fraud_rate = float(y_true.mean())
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[fraud_rate, fraud_rate], mode="lines",
        line=dict(dash="dot", color="lightgray", width=1),
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Taux faux positifs (FPR)", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Taux vrais positifs (TPR)", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Rappel",    row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text="Précision", row=1, col=2, range=[0, 1])
    fig.update_layout(
        title     = "Comparaison MKAN / LSTM / XGBoost — Jeu de test",
        height    = 520,
        width     = 1100,
        legend    = dict(x=0.02, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
        font      = dict(size=13),
    )
    return fig


def plot_metrics_table(metrics: dict[str, dict]) -> go.Figure:
    """
    Tableau comparatif des métriques pour les 3 modèles.

    Entrée : sortie de compute_all_metrics().
    """
    model_names = list(metrics.keys())

    # Même ordre que le visualiseur MKAN : MCC, PR-AUC, Brier, puis AUC-ROC et F1
    metric_defs = [
        ("MCC ↑",       "MCC",   False),
        ("PR-AUC ↑",    "PRAUC", False),
        ("Brier ↓",     "Brier", True),   # True = plus bas est mieux (fond inversé)
        ("AUC-ROC ↑",   "AUC",   False),
        ("F1 ↑",        "F1",    False),
    ]

    col_metric = [label for label, _, _ in metric_defs]
    data_cols  = []
    bg_cols    = []
    bg_base    = ["#f0f4ff", "#fff8f0", "#f0fff4"]

    for mn in model_names:
        col_vals = []
        col_bg   = []
        for _, key, lower_is_better in metric_defs:
            val = metrics[mn][key]
            col_vals.append(f"{val:.4f}")
            # Mettre en vert la meilleure valeur par métrique
            best = (min if lower_is_better else max)(
                metrics[m][key] for m in model_names
            )
            col_bg.append("#d4edda" if abs(val - best) < 1e-9 else bg_base[model_names.index(mn)])
        data_cols.append(col_vals)
        bg_cols.append(col_bg)

    n_metrics = len(metric_defs)
    fig = go.Figure(go.Table(
        header=dict(
            values     = ["<b>Métrique</b>"] + [f"<b>{m}</b>" for m in model_names],
            fill_color = ["#1f2937"] + [_COLORS.get(m, "#555") for m in model_names],
            font_color = "white",
            align      = "center",
            height     = 32,
        ),
        cells=dict(
            values     = [col_metric] + data_cols,
            fill_color = [["#f9f9f9"] * n_metrics] + bg_cols,
            align      = "center",
            height     = 28,
            font       = dict(size=13),
        ),
    ))
    fig.update_layout(
        title  = "Métriques de comparaison — Jeu de test  (vert = meilleur)",
        height = 300,
        margin = dict(t=50, b=10, l=10, r=10),
    )
    return fig


def plot_drift_timeline(js_scores: list[float],
                        epochs:     list[int],
                        threshold:  float = 0.05,
                        feature_name: str = "feature") -> go.Figure:
    """
    Évolution de la divergence JS sur les époques d'entraînement.
    Met en évidence les dépassements du seuil theta_drift.

    Args:
        js_scores    : liste de valeurs JS par époque
        epochs       : numéros d'époques correspondants
        threshold    : seuil de dérive theta_drift (défaut 0.05)
        feature_name : nom de la feature analysée (pour le titre)
    """
    js_arr = np.array(js_scores)
    above  = js_arr > threshold

    fig = go.Figure()

    # Zones de dérive (fond rouge)
    in_drift = False
    start_e  = None
    for i, ep in enumerate(epochs):
        if above[i] and not in_drift:
            in_drift = True
            start_e  = ep
        elif not above[i] and in_drift:
            fig.add_vrect(x0=start_e, x1=epochs[i - 1],
                          fillcolor="rgba(244,67,54,0.12)",
                          layer="below", line_width=0,
                          annotation_text="Dérive", annotation_position="top left",
                          annotation_font_size=10)
            in_drift = False
    if in_drift:
        fig.add_vrect(x0=start_e, x1=epochs[-1],
                      fillcolor="rgba(244,67,54,0.12)",
                      layer="below", line_width=0)

    # Courbe JS
    fig.add_trace(go.Scatter(
        x=epochs, y=js_arr, mode="lines+markers",
        name="JS divergence",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=5, color=np.where(above, "#F44336", "#2196F3").tolist()),
    ))

    # Seuil
    fig.add_hline(y=threshold, line_dash="dash",
                  line_color="#F44336", line_width=1.5,
                  annotation_text=f"θ_drift = {threshold}",
                  annotation_position="bottom right",
                  annotation_font_color="#F44336")

    fig.update_layout(
        title  = f"Détection de dérive JS — {feature_name}",
        xaxis_title = "Époque",
        yaxis_title = "Divergence Jensen-Shannon",
        height = 380,
        legend = dict(x=0.01, y=0.99),
    )
    return fig
