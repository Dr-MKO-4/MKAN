"""
visualizer.py — Visualisations Plotly pour MKAN (section 6.3 / section 4.4.6).

7 méthodes de visualisation :
  plot_loss_curves           — perte totale décomposée (pred + reg), échelle log
  plot_regularization_detail — Σ|Φ_l|₁ (eq. 2.20) et ΣS(Φ_l) (eq. 2.21) séparément
  plot_metrics               — MCC, PR-AUC, Score de Brier sur la validation
  plot_training_dashboard    — vue d'ensemble combinée 6 panneaux
  plot_edge_heatmap          — heatmap L1 de toutes les arêtes d'une porte T-KAN
  plot_edge_functions        — courbes individuelles avec décomposition Gaussienne/Fourier
  plot_pruning_summary       — bilan du nombre d'arêtes survivantes par porte (eq. 4.28)

Usage minimal :
    from MKAN import MKANVisualizer
    viz = MKANVisualizer(model, history, feature_names=FEATURE_COLS)
    viz.plot_training_dashboard().show()
    viz.plot_edge_heatmap("input", x_pool_audit).show()
    viz.plot_edge_functions("forget", x_pool_audit).show()
    viz.plot_pruning_summary(x_pool_audit).show()
"""
from __future__ import annotations

import math
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .audit import edge_l1_matrix, prune_mask, evaluate_edge_curve


# Palette de couleurs cohérente entre toutes les figures
PALETTE = {
    "total":    "#1f2937",
    "pred":     "#2563eb",
    "reg":      "#dc2626",
    "l1":       "#dc2626",
    "entropy":  "#d97706",
    "mcc":      "#059669",
    "prauc":    "#7c3aed",
    "brier":    "#db2777",
    "gauss":    "#2563eb",
    "fourier":  "#d97706",
    "combined": "#1f2937",
}


class MKANVisualizer:
    """
    Regroupe l'ensemble des visualisations Plotly pour :
      (1) le suivi d'entraînement : loss décomposée, régularisation, métriques ;
      (2) l'interprétabilité post-entraînement : structure des arêtes survivantes
          et courbes des fonctions d'activation individuelles (section 4.4.6).

    Le paramètre x_pool (attendu : tenseur (N, hidden+input) des entrées réelles
    des portes — voir cellule « Pool d'audit » dans train.ipynb) permet le calcul
    des normes L1 exactes sur données réelles (eq. 2.19), conformément à la
    politique de transparence COBAC (section 2.5.5).
    """

    def __init__(
        self,
        model,
        history:       dict,
        feature_names: list | None = None,
        hidden_size:   int  | None = None,
    ):
        self.model         = model
        self.history       = history
        self.feature_names = list(feature_names) if feature_names else []
        self.hidden_size   = hidden_size if hidden_size is not None else model.hidden_size
        self.gate_names    = ["forget", "input", "candidate", "output"]
        self.gate_map      = {
            "forget":    model.cell.forget_gate,
            "input":     model.cell.input_gate,
            "candidate": model.cell.candidate_gate,
            "output":    model.cell.output_gate,
        }
        # Labels explicites du vecteur concaténé [h_{t-1}, x_t] :
        # distinction état récurrent (non directement auditable)
        # vs feature transactionnelle métier (auditable COBAC, section 1.2.4).
        self.concat_labels = (
            [f"h[{k}]" for k in range(self.hidden_size)]
            + list(self.feature_names)
        )

    # ── Axe temporel (compatible avec ou sans clé "epoch" dans history) ─────
    def _epochs(self) -> list:
        h = self.history
        if "epoch" in h and h["epoch"]:
            return h["epoch"]
        ref = h.get("loss", h.get("total", []))
        return list(range(1, len(ref) + 1))

    # ── 1. Courbes de perte décomposées (eq. 4.27) ──────────────────────────
    def plot_loss_curves(self) -> go.Figure:
        """Perte totale, prédictive et de régularisation en fonction de l'époque."""
        h    = self.history
        ep   = self._epochs()
        loss = h.get("loss",      h.get("total", []))
        pred = h.get("pred_loss", h.get("pred",  []))
        reg  = h.get("reg", [l - p for l, p in zip(loss, pred)])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ep, y=loss, mode="lines+markers",
            name="ℓ_total (perte totale)",
            line=dict(color=PALETTE["total"], width=2.5)))
        fig.add_trace(go.Scatter(
            x=ep, y=pred, mode="lines+markers",
            name="ℓ_pred (BCE pondérée, eq. 3.1)",
            line=dict(color=PALETTE["pred"], width=2, dash="dash")))
        fig.add_trace(go.Scatter(
            x=ep, y=reg, mode="lines+markers",
            name="ℓ_reg = λ(μ₁·L1 + μ₂·S) (eq. 4.27)",
            line=dict(color=PALETTE["reg"], width=2, dash="dot")))
        fig.update_layout(
            title="Décomposition de la fonction de perte MKAN (eq. 4.27)",
            xaxis_title="Époque",
            yaxis_title="Valeur de la perte (échelle log)",
            yaxis_type="log",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig

    # ── 2. Régularisation : L1 (eq. 2.20) et entropie (eq. 2.21) séparément ─
    def plot_regularization_detail(self) -> go.Figure:
        """Dynamique des deux termes de régularisation au fil de l'entraînement."""
        h   = self.history
        ep  = self._epochs()
        l1  = h.get("l1",      h.get("l1_raw",      []))
        ent = h.get("entropy", h.get("entropy_raw",  []))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Norme L1 du réseau (Σ|Φ_l|₁, eq. 2.20) — sparsification",
                "Entropie des couches (ΣS(Φ_l), eq. 2.21) — équirépartition",
            ),
        )
        fig.add_trace(go.Scatter(x=ep, y=l1, mode="lines+markers",
                                  line=dict(color=PALETTE["l1"], width=2.5),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep, y=ent, mode="lines+markers",
                                  line=dict(color=PALETTE["entropy"], width=2.5),
                                  showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="Époque", row=1, col=1)
        fig.update_xaxes(title_text="Époque", row=1, col=2)
        fig.update_yaxes(title_text="Σ|Φ_l|₁",         row=1, col=1)
        fig.update_yaxes(title_text="Entropie (nats)",  row=1, col=2)
        fig.update_layout(
            template="plotly_white",
            title="Dynamique des termes de régularisation (sparsification vs équirépartition)",
        )
        return fig

    # ── 3. Métriques de validation (MCC, PR-AUC, Brier) ────────────────────
    def plot_metrics(self) -> go.Figure:
        """MCC, PR-AUC (ou AUC-ROC), Score de Brier (ou F1) sur la validation."""
        h  = self.history
        ep = self._epochs()
        val_mcc   = h.get("val_mcc",   [])
        val_prauc = h.get("val_prauc", h.get("val_auc", []))
        val_brier = h.get("val_brier", h.get("val_f1",  []))

        prauc_lbl = "PR-AUC"  if "val_prauc" in h else "AUC-ROC"
        brier_lbl = "Brier ↓" if "val_brier" in h else "F1"

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "MCC (Matthews Correlation Coefficient)",
                prauc_lbl,
                brier_lbl,
            ),
        )
        fig.add_trace(go.Scatter(x=ep, y=val_mcc, mode="lines+markers",
                                  line=dict(color=PALETTE["mcc"],   width=2.5),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep, y=val_prauc, mode="lines+markers",
                                  line=dict(color=PALETTE["prauc"], width=2.5),
                                  showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=ep, y=val_brier, mode="lines+markers",
                                  line=dict(color=PALETTE["brier"], width=2.5),
                                  showlegend=False), row=1, col=3)
        fig.update_yaxes(range=[-1, 1], title_text="MCC",       row=1, col=1)
        fig.update_yaxes(range=[0, 1],  title_text=prauc_lbl,   row=1, col=2)
        fig.update_yaxes(              title_text=brier_lbl,    row=1, col=3)
        for c in (1, 2, 3):
            fig.update_xaxes(title_text="Époque", row=1, col=c)
        fig.update_layout(
            template="plotly_white",
            title="Métriques de performance sur la validation (section 3.4)",
        )
        return fig

    # ── 4. Dashboard combiné (6 panneaux) ────────────────────────────────────
    def plot_training_dashboard(self) -> go.Figure:
        """Vue d'ensemble : loss, régularisation et métriques sur une seule figure."""
        h  = self.history
        ep = self._epochs()
        loss  = h.get("loss",      h.get("total",       []))
        pred  = h.get("pred_loss", h.get("pred",        []))
        l1    = h.get("l1",        h.get("l1_raw",      []))
        ent   = h.get("entropy",   h.get("entropy_raw", []))
        mcc   = h.get("val_mcc",   [])
        prauc = h.get("val_prauc", h.get("val_auc",     []))
        brier = h.get("val_brier", h.get("val_f1",      []))

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Perte totale vs prédictive",
                "Norme L1 (sparsification)",
                "Entropie (équirépartition)",
                "MCC",
                "PR-AUC / AUC-ROC",
                "Brier / F1",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.08,
        )
        fig.add_trace(go.Scatter(x=ep, y=loss, name="ℓ_total",
                                  line=dict(color=PALETTE["total"], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ep, y=pred, name="ℓ_pred",
                                  line=dict(color=PALETTE["pred"], width=2, dash="dash")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=ep, y=l1,  showlegend=False,
                                  line=dict(color=PALETTE["l1"],      width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=ep, y=ent, showlegend=False,
                                  line=dict(color=PALETTE["entropy"], width=2)), row=1, col=3)
        fig.add_trace(go.Scatter(x=ep, y=mcc,   showlegend=False,
                                  line=dict(color=PALETTE["mcc"],   width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=ep, y=prauc, showlegend=False,
                                  line=dict(color=PALETTE["prauc"], width=2)), row=2, col=2)
        fig.add_trace(go.Scatter(x=ep, y=brier, showlegend=False,
                                  line=dict(color=PALETTE["brier"], width=2)), row=2, col=3)
        fig.update_yaxes(type="log",    row=1, col=1)
        fig.update_yaxes(range=[-1, 1], row=2, col=1)
        fig.update_yaxes(range=[0, 1],  row=2, col=2)
        fig.update_layout(
            template="plotly_white",
            height=650,
            title="Tableau de bord d'entraînement MKAN",
            legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
        )
        return fig

    # ── 5. Heatmap des normes L1 par arête (eq. 2.19) ───────────────────────
    def compute_gate_l1_matrix(
        self, gate_name: str, x_pool: torch.Tensor
    ) -> np.ndarray:
        """Calcule la matrice (in_features, out_features) des |φ_ij|₁."""
        return edge_l1_matrix(self.gate_map[gate_name], x_pool)

    def plot_edge_heatmap(
        self, gate_name: str, x_pool: torch.Tensor
    ) -> go.Figure:
        """
        Heatmap de l'importance L1 de chaque arête (i → j) d'une porte T-KAN.

        Permet d'identifier visuellement quelles features ou quels états cachés
        influencent le plus le comportement d'une porte — diagnostic auditabilité.
        """
        l1_mat    = self.compute_gate_l1_matrix(gate_name, x_pool)
        out_labels = [f"h_out[{j}]" for j in range(l1_mat.shape[1])]
        fig = go.Figure(go.Heatmap(
            z=l1_mat,
            x=out_labels,
            y=self.concat_labels,
            colorscale="Reds",
            colorbar=dict(title="|φ_ij|₁"),
        ))
        fig.update_layout(
            title=(f"Structure des arêtes — porte «{gate_name}» "
                   f"(norme L1 individuelle, eq. 2.19)"),
            xaxis_title="Neurone de sortie (composante de l'état caché)",
            yaxis_title="Entrée [état caché h | features transactionnelles x]",
            template="plotly_white",
            height=max(400, 22 * len(self.concat_labels)),
        )
        return fig

    # ── 6. Courbes individuelles avec décomposition Gaussienne / Fourier ─────
    def plot_edge_functions(
        self,
        gate_name: str,
        x_pool:    torch.Tensor,
        theta:     float = 1e-2,
        top_k:     int   = 6,
        domain:    float = 2.0,
    ) -> go.Figure:
        """
        Trace les courbes φ_ij(x) des top_k arêtes les plus importantes d'une porte,
        en décomposant la contribution Gaussienne (FastKAN) et Fourier (KAN-AD).

        Cette décomposition est au cœur de l'interprétabilité MKAN (section 4.2.2) :
        elle permet de voir si une arête donnée est dominée par un comportement
        localisé (Gaussien) ou périodique / robuste (Fourier).
        """
        layer  = self.gate_map[gate_name]
        l1_mat = self.compute_gate_l1_matrix(gate_name, x_pool)
        mask   = prune_mask(l1_mat, theta=theta)

        surviving = np.argwhere(mask)
        if len(surviving) == 0:
            raise ValueError(
                f"Aucune arête ne survit à l'élagage (theta={theta}) sur '{gate_name}'. "
                f"Réduisez theta ou utilisez un x_pool plus représentatif."
            )

        importances = l1_mat[mask]
        order       = np.argsort(-importances)[:top_k]
        top_edges   = surviving[order]

        n     = len(top_edges)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        titles = [
            f"{self.concat_labels[i]} → h_out[{j}]  (|φ|₁={l1_mat[i, j]:.3f})"
            for i, j in top_edges
        ]
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        dev       = layer.centers.device
        x_grid    = torch.linspace(-domain, domain, 200, device=dev)
        x_grid_np = x_grid.cpu().numpy()

        for idx, (i, j) in enumerate(top_edges):
            r, c   = idx // ncols + 1, idx % ncols + 1
            full_y = evaluate_edge_curve(layer, int(i), int(j), x_grid)

            # Décomposition interne Gaussienne vs Fourier (section 4.2.2)
            with torch.inference_mode():
                batch       = torch.zeros(len(x_grid), layer.in_features, device=dev)
                batch[:, i] = x_grid
                diff        = batch.unsqueeze(-1) - layer.centers       # (N, in, M)
                gauss_basis = torch.exp(-(diff ** 2) / (2 * layer.h ** 2))
                gauss_only  = (gauss_basis[:, i, :] @ layer.w_gauss[i, j, :]).cpu().numpy()
                kx          = batch.unsqueeze(-1) * layer.k_idx         # (N, in, K)
                fourier_only = (
                    torch.cos(kx)[:, i, :] @ layer.a_fourier[i, j, :]
                    + torch.sin(kx)[:, i, :] @ layer.b_fourier[i, j, :]
                ).cpu().numpy()

            show = (idx == 0)
            fig.add_trace(go.Scatter(
                x=x_grid_np, y=gauss_only, mode="lines",
                line=dict(color=PALETTE["gauss"],    width=1.3, dash="dot"),
                name="Composante Gaussienne (FastKAN)",
                showlegend=show), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=x_grid_np, y=fourier_only, mode="lines",
                line=dict(color=PALETTE["fourier"],  width=1.3, dash="dash"),
                name="Composante Fourier (KAN-AD)",
                showlegend=show), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=x_grid_np, y=full_y, mode="lines",
                line=dict(color=PALETTE["combined"], width=2.5),
                name="φ_ij(x) combinée",
                showlegend=show), row=r, col=c)

        fig.update_layout(
            template="plotly_white",
            height=320 * nrows,
            title=(f"Fonctions d'arête survivantes — porte «{gate_name}» "
                   f"({n} arêtes, θ={theta})"),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0.15),
        )
        return fig

    # ── 7. Bilan d'élagage par porte (eq. 4.28) ─────────────────────────────
    def plot_pruning_summary(
        self, x_pool: torch.Tensor, theta: float = 1e-2
    ) -> go.Figure:
        """
        Nombre d'arêtes totales vs survivantes par porte T-KAN après élagage (eq. 4.28).

        Permet de visualiser le degré de sparsification atteint : un réseau correctement
        régularisé devrait conserver un petit nombre d'arêtes par porte, chacune
        correspondant à une relation métier identifiable.
        """
        totals, survivors = [], []
        for gname in self.gate_names:
            l1_mat = self.compute_gate_l1_matrix(gname, x_pool)
            mask   = prune_mask(l1_mat, theta=theta)
            totals.append(mask.size)
            survivors.append(int(mask.sum()))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=self.gate_names, y=totals,
            name="Arêtes totales",
            marker_color="#e5e7eb"))
        fig.add_trace(go.Bar(
            x=self.gate_names, y=survivors,
            name="Arêtes survivantes (après élagage)",
            marker_color=PALETTE["l1"]))
        fig.update_layout(
            barmode="overlay",
            template="plotly_white",
            title=f"Sparsification par porte après élagage (θ={theta}, eq. 4.28)",
            xaxis_title="Porte T-KAN",
            yaxis_title="Nombre d'arêtes",
        )
        return fig
