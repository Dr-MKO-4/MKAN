#!/usr/bin/env python3
"""
export_figures.py
Convertit tous les fichiers HTML Plotly du run MKAN en PNG via kaleido.
Deux formats : wide (1400×600) pour figures de comparaison, square (900×500) pour autres.
Usage : python export_figures.py [--html-dir DIR] [--out-dir DIR]
"""

import argparse
import sys
from pathlib import Path

# ─── kaleido requis ──────────────────────────────────────────────────────────
try:
    import plotly.io as pio
except ImportError:
    sys.exit("Installez : pip install plotly kaleido")

# ─── Configuration des figures ───────────────────────────────────────────────
# (nom_base_html : (largeur, hauteur))
WIDE   = (1400, 600)
SQUARE = (900,  500)

FIGURE_SPECS = {
    # Courbes d'entraînement
    "loss_curves":          WIDE,
    "training_curves":      WIDE,
    "val_metrics":          WIDE,
    "regularization":       WIDE,
    "roc_pr_comparison":    WIDE,
    # Tableaux & matrices
    "metrics_table":        WIDE,
    "confusion_matrix":     SQUARE,
    "pruning_summary":      SQUARE,
    # Fonctions d'arête
    "edge_functions_forget":    WIDE,
    "edge_functions_input":     WIDE,
    "edge_functions_candidate": WIDE,
    "edge_functions_output":    WIDE,
    # Heatmaps d'activation
    "heatmap_forget":    SQUARE,
    "heatmap_input":     SQUARE,
    "heatmap_candidate": SQUARE,
    "heatmap_output":    SQUARE,
    # Dérive & explication
    "drift_by_feature":      SQUARE,
    "decision_explanation":  SQUARE,
    "symbolic_report":       WIDE,
    # Dashboard global
    "dashboard":             (1600, 900),
}

def stem_to_key(stem: str) -> str:
    """Mappe 'xxxxxxxx-loss_curves' → 'loss_curves'."""
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 else stem

def _find_json_end(content: str, start: int) -> int:
    """
    Retourne l'indice du caractère fermant du JSON (objet ou tableau)
    commençant à `start`, en comptant les accolades/crochets imbriqués
    et en ignorant les caractères dans les chaînes.
    Retourne -1 si non trouvé.
    """
    open_char  = content[start]
    close_char = '}' if open_char == '{' else ']'
    depth       = 0
    in_str      = False
    escape_next = False

    for i in range(start, len(content)):
        c = content[i]
        if escape_next:
            escape_next = False
            continue
        if in_str:
            if c == '\\':
                escape_next = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _extract_plotly_data_layout(content: str):
    """
    Extrait (data_str, layout_str) depuis un fichier HTML Plotly.
    Supporte :
      - Plotly.newPlot("id", data, layout, config)   format classique
      - Plotly.react("id", data, layout, config)
      - <script type="application/json">{"data":...}</script>  format récent
    Retourne (None, None) si introuvable.
    """
    import re

    # ── Stratégie 1 : balise <script type="application/json"> ────────────────
    m = re.search(
        r'<script type=["\']application/json["\'][^>]*>(.*?)</script>',
        content, re.DOTALL
    )
    if m:
        raw = m.group(1).strip()
        try:
            obj = __import__("json").loads(raw)
            data_str   = __import__("json").dumps(obj.get("data",   []))
            layout_str = __import__("json").dumps(obj.get("layout", {}))
            return data_str, layout_str
        except Exception:
            pass

    # ── Stratégie 2 : Plotly.newPlot / Plotly.react ───────────────────────────
    # rfind() : prend la DERNIÈRE occurrence (l'appel réel, pas la doc de la lib)
    for func in ("Plotly.newPlot(", "Plotly.react("):
        idx = content.rfind(func)
        if idx == -1:
            continue

        i = idx + len(func)

        # Passer le premier argument (id du div — chaîne entre guillemets)
        while i < len(content) and content[i] in ' \t\n\r':
            i += 1
        if i >= len(content) or content[i] not in ('"', "'"):
            continue
        quote = content[i]
        i += 1
        while i < len(content):
            if content[i] == '\\':
                i += 2
                continue
            if content[i] == quote:
                i += 1
                break
            i += 1

        # Passer la virgule et les espaces
        while i < len(content) and content[i] in ' \t\n\r,':
            i += 1

        # Extraire le tableau data [...]
        if i >= len(content) or content[i] not in ('[', '{'):
            continue
        end_data = _find_json_end(content, i)
        if end_data == -1:
            continue
        data_str = content[i:end_data + 1]

        # Passer la virgule et les espaces
        j = end_data + 1
        while j < len(content) and content[j] in ' \t\n\r,':
            j += 1

        # Extraire l'objet layout {...}
        layout_str = "{}"
        if j < len(content) and content[j] == '{':
            end_layout = _find_json_end(content, j)
            if end_layout != -1:
                layout_str = content[j:end_layout + 1]

        return data_str, layout_str

    return None, None


def export_html_to_png(html_path: Path, out_dir: Path) -> None:
    import json
    key  = stem_to_key(html_path.stem)
    w, h = FIGURE_SPECS.get(key, SQUARE)
    out  = out_dir / f"{key}.png"

    content = html_path.read_text(encoding="utf-8", errors="ignore")
    data_str, layout_str = _extract_plotly_data_layout(content)

    if data_str is None:
        print(f"  [SKIP] données Plotly introuvables : {html_path.name}")
        return

    try:
        data   = json.loads(data_str)
        layout = json.loads(layout_str)
    except json.JSONDecodeError as e:
        print(f"  [SKIP] JSON invalide ({e}) : {html_path.name}")
        return

    import plotly.graph_objects as go
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        width=w, height=h,
        font=dict(family="DejaVu Sans, sans-serif", size=13),
        margin=dict(l=60, r=40, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.write_image(str(out), format="png", scale=2)
    print(f"  [OK]  {out.name}  ({w}×{h} px, scale=2)")

def main():
    parser = argparse.ArgumentParser(description="Exporte les figures HTML Plotly en PNG.")
    parser.add_argument("--html-dir", default="MKAN/checkpoints", help="Dossier source des HTML")
    parser.add_argument("--out-dir",  default="MKAN/checkpoints", help="Dossier destination PNG")
    args = parser.parse_args()

    html_dir = Path(args.html_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        sys.exit(f"Aucun fichier HTML trouvé dans : {html_dir}")

    print(f"{'='*60}")
    print(f"Export MKAN figures → PNG")
    print(f"Source : {html_dir}   ({len(html_files)} fichiers)")
    print(f"Dest.  : {out_dir}")
    print(f"{'='*60}")

    ok = 0
    for hp in html_files:
        try:
            export_html_to_png(hp, out_dir)
            ok += 1
        except Exception as exc:
            print(f"  [ERR] {hp.name}: {exc}")

    print(f"{'='*60}")
    print(f"Terminé : {ok}/{len(html_files)} figures exportées.")
    print(f"PNGs disponibles dans : {out_dir.resolve()}")

if __name__ == "__main__":
    main()