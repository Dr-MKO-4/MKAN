"""
explainability.py  Explication de décision MKAN pour audit COBAC (section 4.4.7).

  explain_transaction      décompose la décision sur une fenêtre de transactions
  plot_decision_explanation  figure Plotly : contributions + jauge de score
  plot_symbolic_report      visualisation textuelle du rapport symbolique

L'objectif est de fournir une explication opposable : "cette transaction a été
signalée à risque X% parce que φ(r1) ≈ f₁(x) et φ(delta_B_orig) ≈ f₂(x)",
cohérent avec les exigences d'auditabilité COBAC (section 2.5.5).
"""

from __future__ import annotations

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def explain_transaction(
    model,
    x_window:      torch.Tensor,
    feature_names: list[str],
    threshold:     float = 0.5,
) -> dict:
    """
    Calcule la contribution de chaque feature à la décision pour une fenêtre.

    La contribution d'une feature i est définie comme la moyenne (sur les W pas
    de temps et les 4 portes) des activations d'arêtes |φ_ij(x_i)|₁  cohérent
    avec la norme L1 utilisée pour l'élagage (eq. 2.19).

    Args:
        model         : MKANScorer entraîné
        x_window      : (W, input_size) ou (1, W, input_size)
        feature_names : noms des features originales (longueur input_size)
        threshold     : seuil de décision (défaut 0.5)

    Returns:
        dict :
          score           float ∈ (0,1)
          verdict         "FRAUDE" | "NORMALE"
          threshold       float
          top_features    liste triée [{label, contribution}, ...]
          gate_contribs   {gate_name: array(concat_size,)} contributions par porte
          concat_labels   noms des entrées du vecteur [h, x]
    """
    if x_window.dim() == 2:
        x_window = x_window.unsqueeze(0)

    hidden_size   = model.hidden_size
    input_size    = model.cell.input_size
    concat_labels = [f"h[{k}]" for k in range(hidden_size)] + list(feature_names)
    concat_size   = hidden_size + input_size

    orig_dev = next(model.parameters()).device
    on_dml   = str(orig_dev.type) != "cpu"
    if on_dml:
        model.cpu()          # bascule CPU : évite le conflit DirectML / no_grad

    xw = x_window.float().cpu()
    W  = xw.shape[1]

    gate_contribs = {g: np.zeros(concat_size)
                     for g in ["forget", "input", "candidate", "output"]}

    model.eval()
    inv_W = 1.0 / W
    with torch.no_grad():
        h_t = torch.zeros(1, hidden_size)
        c_t = torch.zeros(1, hidden_size)

        for t in range(W):
            x_t      = xw[:, t, :]
            combined = torch.cat([h_t, x_t], dim=-1)   # (1, concat_size)

            for gate_name in ["forget", "input", "candidate", "output"]:
                gate = getattr(model.cell, f"{gate_name}_gate")
                edges = gate.edge_activations(combined)  # (1, concat_size, hidden_size)
                contrib = edges.abs().mean(dim=-1).squeeze(0).cpu().numpy()
                gate_contribs[gate_name] += contrib * inv_W

            h_t, c_t = model.cell(x_t, h_t, c_t)

        score = torch.sigmoid(model.projection(h_t)).squeeze().item()

    # Contribution globale : somme sur les 4 portes, normalisée
    global_contrib = sum(gate_contribs.values())
    total          = global_contrib.sum() + 1e-12
    global_norm    = global_contrib * (1.0 / total)

    if on_dml:
        model.to(orig_dev)   # remettre sur DirectML après inférence

    order       = np.argsort(-global_norm)
    top_features = [
        {"label": concat_labels[i], "contribution": float(global_norm[i])}
        for i in order
    ]

    return {
        "score":         score,
        "verdict":       "FRAUDE" if score >= threshold else "NORMALE",
        "threshold":     threshold,
        "top_features":  top_features,
        "gate_contribs": {g: v.tolist() for g, v in gate_contribs.items()},
        "concat_labels": concat_labels,
    }


def plot_decision_explanation(explanation: dict, top_k: int = 12) -> go.Figure:
    """
    Figure 2 panneaux : contributions des features (barres) + jauge de score.

    Conçu pour être insérable dans un rapport COBAC : une seule image auto-explicative
    sans nécessiter de connaissance technique du modèle.

    Args:
        explanation : sortie de explain_transaction()
        top_k       : nombre de features à afficher (défaut 12)
    """
    score   = explanation["score"]
    verdict = explanation["verdict"]
    top_f   = explanation["top_features"][:top_k]

    labels   = [f["label"]        for f in top_f]
    contribs = [f["contribution"]  for f in top_f]
    colors   = [
        "#F44336" if "h[" not in lbl else "#90A4AE"
        for lbl in labels
    ]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=[
            "Contributions à la décision (normalisées)",
            "Score de risque",
        ],
        horizontal_spacing=0.08,
        specs=[[{"type": "xy"}, {"type": "indicator"}]],
    )

    # Barres horizontales
    fig.add_trace(go.Bar(
        x      = contribs[::-1],
        y      = labels[::-1],
        orientation = "h",
        marker_color = colors[::-1],
        text   = [f"{c:.3f}" for c in contribs[::-1]],
        textposition = "outside",
        name   = "Contribution",
    ), row=1, col=1)

    # Jauge
    fig.add_trace(go.Indicator(
        mode  = "gauge+number",
        value = score * 100,
        number = dict(suffix="%", font=dict(size=30)),
        title  = dict(
            text = f"Score de risque<br><b style='color:{'#F44336' if verdict=='FRAUDE' else '#4CAF50'}'>{verdict}</b>",
            font = dict(size=15),
        ),
        gauge = dict(
            axis  = dict(range=[0, 100], ticksuffix="%"),
            bar   = dict(color="#F44336" if verdict == "FRAUDE" else "#4CAF50", thickness=0.3),
            steps = [
                dict(range=[0,  50], color="#E8F5E9"),
                dict(range=[50, 100], color="#FFEBEE"),
            ],
            threshold = dict(
                line      = dict(color="black", width=3),
                thickness = 0.8,
                value     = explanation["threshold"] * 100,
            ),
        ),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Contribution relative", range=[0, max(contribs) * 1.2],
                     row=1, col=1)
    fig.update_layout(
        title  = (f"Explication de décision MKAN  Score : {score:.3f}  ({verdict})  "
                  f"[seuil {explanation['threshold']:.2f}]"),
        height = 520,
        width  = 1000,
        showlegend = False,
        font   = dict(size=12),
    )
    return fig


def plot_symbolic_report(report: dict, feature_names: list[str],
                          hidden_size: int) -> go.Figure:
    """
    Visualise le rapport symbolique extrait (sortie de extract_full_model_report).

    Affiche pour chaque porte les arêtes actives avec leur formule symbolique
    et leur importance L1, sous forme de tableau coloré par gate.

    Args:
        report        : sortie de extract_full_model_report()
        feature_names : noms des features originales
        hidden_size   : taille de l'état caché
    """
    gate_colors = {
        "forget":    "rgba(33,150,243,0.15)",
        "input":     "rgba(76,175,80,0.15)",
        "candidate": "rgba(255,152,0,0.15)",
        "output":    "rgba(156,39,176,0.15)",
    }

    rows_data = []
    for gate_name, edges in report.items():
        for e in edges:
            rows_data.append({
                "Porte":       gate_name.capitalize(),
                "Entrée":      e["input"],
                "Sortie":      e["output"],
                "Importance":  f"{e['l1_importance']:.4f}",
                "Formule":     e["formula"],
                "R²":          f"{e['r2']:.4f}" if e["r2"] is not None else "",
                "Symbolif.":   "✓" if e["symbolifiable"] else "✗",
                "_color":      gate_colors.get(gate_name, "#999"),
            })

    if not rows_data:
        fig = go.Figure()
        fig.update_layout(title="Rapport symbolique  aucune arête active")
        return fig

    fig = go.Figure(go.Table(
        header=dict(
            values    = ["<b>Porte</b>", "<b>Entrée</b>", "<b>Sortie</b>",
                         "<b>|φ|₁</b>", "<b>Formule symbolique</b>",
                         "<b>R²</b>", "<b>Symbolif.</b>"],
            fill_color = "#1f2937",
            font_color = "white",
            align      = "left",
            height     = 32,
        ),
        cells=dict(
            values=[
                [r["Porte"]      for r in rows_data],
                [r["Entrée"]     for r in rows_data],
                [r["Sortie"]     for r in rows_data],
                [r["Importance"] for r in rows_data],
                [r["Formule"]    for r in rows_data],
                [r["R²"]         for r in rows_data],
                [r["Symbolif."]  for r in rows_data],
            ],
            fill_color = [
                [r["_color"] for r in rows_data],
                ["#f9f9f9"] * len(rows_data),
                ["#f9f9f9"] * len(rows_data),
                ["#f9f9f9"] * len(rows_data),
                ["#f0f4ff"] * len(rows_data),
                ["#f9f9f9"] * len(rows_data),
                [("#e8f5e9" if r["Symbolif."] == "✓" else "#ffebee") for r in rows_data],
            ],
            align  = ["left"] * 7,
            height = 26,
            font   = dict(size=12),
        ),
    ))
    fig.update_layout(
        title  = "Rapport d'audit symbolique COBAC  Arêtes actives par porte T-KAN",
        height = max(300, 60 + 26 * len(rows_data)),
        margin = dict(t=50, b=10, l=10, r=10),
    )
    return fig
