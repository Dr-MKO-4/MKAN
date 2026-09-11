"""
results_export.py  Étape 2 : consolidation, tables, graphiques et rapport de
synthèse à partir des sorties de extended_heuristic_search.py et
sensitivity_analysis.py.

─── Constat d'inventaire (première opération, section 2) ───────────────────
Sur cet environnement Windows, /workspace/scratch/ n'existe pas (convention
de conteneur cloud absente localement — même constat que dans
extended_heuristic_search.py et sensitivity_analysis.py, qui replient alors
respectivement sur MKAN/checkpoints/extended_search/ et
MKAN/checkpoints/sensitivity/). À la date d'écriture de ce module, AUCUNE
recherche Étape 2 réelle n'a encore été exécutée dans ce dépôt : ni
/workspace/scratch/ ni MKAN/checkpoints/{extended_search,sensitivity}/ ne
contiennent de extended_search_log.jsonl, heuristic_best_config.json,
sensitivity_rankings.json, morris_results.json, sobol_evaluations.npy ou
sensitivity_report.md — seuls des artefacts de l'Étape 1 (audit_report.json,
mkan_final.pt, *.html) existent dans MKAN/checkpoints/. Step2ResultsExporter
est donc conçu, comme l'exige la spécification, pour fonctionner sur un
répertoire source PARTIEL ou VIDE : load_all_results() ne lève jamais
d'exception fatale sur un fichier manquant, et validate_outputs() rapporte
explicitement les sorties impossibles à produire faute de source — aucun
résultat scientifique n'est fabriqué par ce module.

─── Compatibilité de formats (section 19) ───────────────────────────────────
_normalize_config()/_normalize_metrics() tolèrent les variations réelles
entre les trois sources possibles :
  - extended_search_log.jsonl : individu au format natif de
    ExtendedHeuristicSearch (Cell_Type, Input_Regime, gates={gate:{"Base":…}}).
  - heuristic_best_config.json : theta_opt + theta_struct.gates={gate:{"base":…,
    "hyperparams":{…}}} (format scientifique canonique de l'Étape 2, prioritaire
    pour la compatibilité avec les étapes suivantes — jamais réécrit avec une
    structure différente par ce module).
  - top5_configs.json / représentation aplatie (sensitivity_analysis.
    _flatten_individual) : clés "Base_<Gate>" et "<Gate>::<Base>::<Param>".
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_heuristic_search import hidden_size_gru_adjusted, GATE_NAMES_TKAN  # noqa: E402

logger = logging.getLogger("results_export")

# ── Palette accessible (section 10) — ne pas en ajouter d'autres sans nécessité ──
PALETTE = {
    "Critique":   (0.85, 0.33, 0.10),
    "Interactif": (0.94, 0.66, 0.13),
    "Neutre":     (0.50, 0.50, 0.50),
    "Redondant":  (0.17, 0.51, 0.73),
}

CANONICAL_METRIC_KEYS = ["MCC_raw", "MCC_clipped", "PR_AUC", "Brier",
                         "R2_symbolic", "latency_ms", "penalty_lat", "fitness_total"]

TABLE_COLUMNS = ["Rang", "MCC", "PR_AUC", "Brier", "R2_symbolic", "Latence_ms",
                 "Fitness_totale", "Base_Forget", "Base_Input", "Base_Candidate",
                 "Base_Output", "Cell_Type", "Input_Regime", "hidden_size", "lr",
                 "lam", "mu1", "mu2", "W", "batch_size"]

EXPECTED_FILES = ["extended_search_log.jsonl", "best_candidate.pt",
                  "heuristic_best_config.json", "sensitivity_rankings.json",
                  "morris_results.json", "sobol_evaluations.npy",
                  "sensitivity_report.md", "sobol_results.json"]


# ══════════════════════════════════════════════════════════════════════════
# Normalisation (section 19)
# ══════════════════════════════════════════════════════════════════════════

def _normalize_metrics(raw: Optional[dict]) -> dict:
    """
    Normalise un dict de métriques quel que soit son format d'origine.
    Ne JAMAIS remplacer MCC_raw par MCC_clipped ; MCC_clipped n'est dérivé
    de MCC_raw (max(0, MCC_raw), formule imposée par l'Étape 2) que s'il est
    absent de la source — jamais une métrique inventée, une conséquence
    déterministe déjà spécifiée ailleurs dans le projet. Toute clé absente
    des deux sources reste None (jamais 0 — section 20).
    """
    if not raw:
        return {k: None for k in CANONICAL_METRIC_KEYS}
    mcc_raw = raw.get("MCC_raw", raw.get("MCC"))
    mcc_clipped = raw.get("MCC_clipped")
    if mcc_clipped is None and mcc_raw is not None:
        mcc_clipped = max(0.0, mcc_raw)
    return {
        "MCC_raw": mcc_raw, "MCC_clipped": mcc_clipped,
        "PR_AUC": raw.get("PR_AUC"), "Brier": raw.get("Brier"),
        "R2_symbolic": raw.get("R2_symbolic"),
        "latency_ms": raw.get("latency_ms"), "penalty_lat": raw.get("penalty_lat"),
        "fitness_total": raw.get("fitness_total", raw.get("fitness")),
    }


def _normalize_config(raw: dict) -> dict:
    """
    Normalise une configuration vers le format canonique
    {Cell_Type, Input_Regime, hidden_size, lr, lam, mu1, mu2, batch_size, W,
     gates: {gate: {"Base": base, **hyperparams actifs}}}.
    """
    if not raw:
        return {}
    cell_type = raw.get("Cell_Type", raw.get("cell_type"))
    input_regime = raw.get("Input_Regime", raw.get("input_regime"))

    theta_opt_src = raw.get("theta_opt", raw)
    out = {"Cell_Type": cell_type, "Input_Regime": input_regime}
    for key in ("hidden_size", "lr", "lam", "mu1", "mu2", "batch_size", "W"):
        out[key] = theta_opt_src.get(key, raw.get(key))

    gates = {}
    if isinstance(raw.get("gates"), dict) and raw["gates"] and \
            all("Base" in v or "base" in v for v in raw["gates"].values()):
        # Format natif ExtendedHeuristicSearch (ou variante base/Base).
        for gate, cfg in raw["gates"].items():
            base = cfg.get("Base", cfg.get("base"))
            params = {k: v for k, v in cfg.items() if k not in ("Base", "base")}
            gates[gate] = {"Base": base, **params}
    elif isinstance(raw.get("theta_struct"), dict):
        # heuristic_best_config.json : theta_struct.gates[gate] = {base, hyperparams}
        for gate, cfg in raw["theta_struct"].get("gates", {}).items():
            gates[gate] = {"Base": cfg.get("base"), **(cfg.get("hyperparams") or {})}
        out["Cell_Type"] = out["Cell_Type"] or raw["theta_struct"].get("cell_type")
        out["Input_Regime"] = out["Input_Regime"] or raw["theta_struct"].get("input_regime")
    else:
        # Représentation aplatie : "Base_<Gate>" + "<Gate>::<Base>::<Param>"
        for gate in GATE_NAMES_TKAN:
            base_key = f"Base_{gate}"
            if base_key not in raw:
                continue
            base = raw[base_key]
            prefix = f"{gate}::{base}::"
            params = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
            gates[gate] = {"Base": base, **params}
    out["gates"] = gates
    return out


def _flatten_bases(config: dict) -> dict:
    gates = config.get("gates", {})
    return {f"Base_{g}": gates.get(g, {}).get("Base") for g in GATE_NAMES_TKAN}


# ══════════════════════════════════════════════════════════════════════════
# Step2ResultsExporter
# ══════════════════════════════════════════════════════════════════════════

class Step2ResultsExporter:
    """
    Consolide, exporte (tables/JSON/CSV/PNG) et rapporte les résultats de
    l'Étape 2 (ExtendedHeuristicSearch + HierarchicalSensitivityAnalyzer).

    Args:
        scratch_dir : répertoire source ET destination des exports (défaut
                     "/workspace/scratch/" — le CLI expose --output_dir).
    """

    def __init__(self, scratch_dir: str = "/workspace/scratch/") -> None:
        self.scratch_dir = scratch_dir
        self.warnings: list = []
        self.results: Optional[dict] = None

    # ── Inventaire (section 2, PREMIÈRE opération) ──────────────────────────

    def inventory(self) -> dict:
        inv = {}
        for fname in EXPECTED_FILES:
            path = os.path.join(self.scratch_dir, fname)
            inv[fname] = os.path.exists(path)
        extra = []
        if os.path.isdir(self.scratch_dir):
            known = set(EXPECTED_FILES)
            for entry in os.listdir(self.scratch_dir):
                if entry not in known and os.path.isfile(os.path.join(self.scratch_dir, entry)):
                    extra.append(entry)
        inv["_extra_files"] = sorted(extra)
        inv["_directory_exists"] = os.path.isdir(self.scratch_dir)
        return inv

    # ── Chargement défensif (section 3) ─────────────────────────────────────

    def load_all_results(self) -> dict:
        inv = self.inventory()
        if not inv["_directory_exists"]:
            self._warn(f"Répertoire introuvable : {self.scratch_dir}")

        results = {
            "heuristic_log": self._load_jsonl("extended_search_log.jsonl"),
            "best_candidate": self._load_torch("best_candidate.pt"),
            "heuristic_best_config": self._load_json("heuristic_best_config.json"),
            "sensitivity_rankings": self._load_json("sensitivity_rankings.json"),
            "morris_results": self._load_json("morris_results.json"),
            "sobol_evaluations": self._load_npy("sobol_evaluations.npy"),
            "sensitivity_report": self._load_text("sensitivity_report.md"),
            "sobol_results": self._load_json("sobol_results.json"),
        }
        self.results = results
        return results

    def _path(self, fname: str) -> str:
        return os.path.join(self.scratch_dir, fname)

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def _load_jsonl(self, fname: str) -> Optional[list]:
        path = self._path(fname)
        if not os.path.exists(path):
            self._warn(f"Fichier manquant : {fname}")
            return None
        records = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    self._warn(f"{fname} ligne {i} illisible : {exc}")
        return records

    def _load_json(self, fname: str) -> Optional[dict]:
        path = self._path(fname)
        if not os.path.exists(path):
            self._warn(f"Fichier manquant : {fname}")
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            self._warn(f"{fname} illisible (JSON invalide) : {exc}")
            return None

    def _load_torch(self, fname: str):
        path = self._path(fname)
        if not os.path.exists(path):
            self._warn(f"Fichier manquant : {fname}")
            return None
        try:
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:   # noqa: BLE001
            self._warn(f"{fname} illisible : {exc}")
            return None

    def _load_npy(self, fname: str) -> Optional[np.ndarray]:
        path = self._path(fname)
        if not os.path.exists(path):
            self._warn(f"Fichier manquant : {fname}")
            return None
        try:
            return np.load(path, allow_pickle=True)
        except Exception as exc:   # noqa: BLE001
            self._warn(f"{fname} illisible : {exc}")
            return None

    def _load_text(self, fname: str) -> Optional[str]:
        path = self._path(fname)
        if not os.path.exists(path):
            self._warn(f"Fichier manquant : {fname}")
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    # ── Consolidation (section 4) ────────────────────────────────────────────

    def _iter_evaluations(self) -> list:
        """Liste normalisée [(config, metrics, record)] à partir de heuristic_log."""
        log = (self.results or {}).get("heuristic_log")
        if not log:
            return []
        out = []
        for rec in log:
            if rec.get("fitness_components") is None and rec.get("error"):
                continue
            config = _normalize_config(rec.get("config", {}))
            metrics = _normalize_metrics(rec.get("fitness_components")
                                         or ({"fitness_total": rec.get("fitness")}
                                             if rec.get("fitness") is not None else None))
            out.append((config, metrics, rec))
        return out

    def _best_config_normalized(self) -> "tuple[Optional[dict], Optional[dict]]":
        """Retourne (config, metrics) normalisés depuis heuristic_best_config.json, sinon
        depuis la meilleure évaluation du journal."""
        cfg = (self.results or {}).get("heuristic_best_config")
        if cfg:
            return (_normalize_config(cfg),
                   _normalize_metrics(cfg.get("fitness_components")))
        evals = self._iter_evaluations()
        if not evals:
            return None, None
        best = max(evals, key=lambda t: (t[1]["fitness_total"] if t[1]["fitness_total"] is not None
                                         else float("-inf")))
        return best[0], best[1]

    # ── Top-K (section 5) ────────────────────────────────────────────────────

    def export_tables(self, output_dir: str, top_k: int = 5) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        evals = self._iter_evaluations()

        rows = []
        seen = set()
        if evals:
            ranked = sorted(evals, key=lambda t: (t[1]["fitness_total"]
                                                   if t[1]["fitness_total"] is not None
                                                   else float("-inf")), reverse=True)
            for config, metrics, _ in ranked:
                key = json.dumps(config, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((config, metrics))
                if len(rows) >= top_k:
                    break
        else:
            config, metrics = self._best_config_normalized()
            if config:
                rows = [(config, metrics)]
                self._warn("extended_search_log.jsonl absent : top-k réduit à la "
                          "seule heuristic_best_config.json disponible.")

        json_rows = []
        md_lines = [f"# Top-{top_k} configurations — Étape 2 MKAN", ""]
        header = TABLE_COLUMNS + ["Hyperparametres_conditionnels_actifs"]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for rank, (config, metrics) in enumerate(rows, start=1):
            bases = _flatten_bases(config)
            active_hp = {g: {k: v for k, v in c.items() if k != "Base"}
                        for g, c in config.get("gates", {}).items()}

            def fmt(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "N/A"

            md_lines.append("| " + " | ".join([
                str(rank), fmt(metrics["MCC_raw"]), fmt(metrics["PR_AUC"]), fmt(metrics["Brier"]),
                fmt(metrics["R2_symbolic"]), fmt(metrics["latency_ms"]), fmt(metrics["fitness_total"]),
                str(bases["Base_Forget"]), str(bases["Base_Input"]), str(bases["Base_Candidate"]),
                str(bases["Base_Output"]), str(config.get("Cell_Type")), str(config.get("Input_Regime")),
                str(config.get("hidden_size")), fmt(config.get("lr")), fmt(config.get("lam")),
                fmt(config.get("mu1")), fmt(config.get("mu2")), str(config.get("W")),
                str(config.get("batch_size")), json.dumps(active_hp, default=str),
            ]) + " |")

            json_rows.append({
                "rank": rank, "MCC_raw": metrics["MCC_raw"], "MCC_clipped": metrics["MCC_clipped"],
                "PR_AUC": metrics["PR_AUC"], "Brier": metrics["Brier"],
                "R2_symbolic": metrics["R2_symbolic"], "latency_ms": metrics["latency_ms"],
                "fitness_total": metrics["fitness_total"],
                **bases, "Cell_Type": config.get("Cell_Type"), "Input_Regime": config.get("Input_Regime"),
                "hidden_size": config.get("hidden_size"), "lr": config.get("lr"), "lam": config.get("lam"),
                "mu1": config.get("mu1"), "mu2": config.get("mu2"), "W": config.get("W"),
                "batch_size": config.get("batch_size"), "gates": config.get("gates", {}),
            })

        base_name = "top5_configurations" if top_k == 5 else f"top{top_k}_configurations"
        md_path = os.path.join(output_dir, f"{base_name}.md")
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, indent=2, ensure_ascii=False, default=str)
        return {"md_path": md_path, "json_path": json_path, "n_rows": len(json_rows)}

    # ── Table de sensibilité (section 7) ─────────────────────────────────────

    def export_hyperparameter_importance(self, output_dir: str) -> Optional[dict]:
        os.makedirs(output_dir, exist_ok=True)
        rankings_payload = (self.results or {}).get("sensitivity_rankings")
        if not rankings_payload:
            self._warn("sensitivity_rankings.json absent : hyperparameter_importance non généré.")
            return None

        rankings = sorted(rankings_payload["rankings"], key=lambda r: -(r["S_Ti"] or 0.0))
        md_lines = ["# Importance des hyperparamètres — Étape 2 MKAN", "",
                   "| Hyperparamètre | S_i | S_Ti | S_Ti/S_i | mu_star_Morris | Statut | Valeur_fixe_si_redondant |",
                   "|---|---|---|---|---|---|---|"]
        json_rows = []
        for r in rankings:
            s_i = r.get("S_i")
            ratio = r.get("ratio_STi_Si")
            if ratio is None and s_i is not None and r.get("S_Ti") is not None:
                ratio = r["S_Ti"] / (abs(s_i) + 1e-9)   # protection S_i ≈ 0

            def fmt(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "N/A"

            md_lines.append(f"| {r['param_name']} | {fmt(s_i)} | {fmt(r.get('S_Ti'))} | "
                           f"{fmt(ratio)} | {fmt(r.get('mu_star'))} | {r.get('status')} | "
                           f"{r.get('fixed_value') if r.get('fixed_value') is not None else 'N/A'} |")
            json_rows.append({"param_name": r["param_name"], "gate": r.get("gate"),
                             "base_context": r.get("base_context"), "S_i": s_i,
                             "S_Ti": r.get("S_Ti"), "S_Ti_over_S_i": ratio,
                             "mu_star_Morris": r.get("mu_star"), "status": r.get("status"),
                             "fixed_value": r.get("fixed_value")})

        md_path = os.path.join(output_dir, "hyperparameter_importance.md")
        json_path = os.path.join(output_dir, "hyperparameter_importance.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, indent=2, ensure_ascii=False, default=str)
        return {"md_path": md_path, "json_path": json_path, "n_rows": len(json_rows)}

    # ── Search history (section 8) ───────────────────────────────────────────

    def export_search_history(self, output_dir: str) -> Optional[dict]:
        os.makedirs(output_dir, exist_ok=True)
        evals = self._iter_evaluations()
        if not evals:
            self._warn("extended_search_log.jsonl absent/vide : search_history.csv non généré.")
            return None

        rows = []
        for config, metrics, rec in evals:
            bases = _flatten_bases(config)
            row = {
                "generation": rec.get("generation"), "individual_id": rec.get("individual_id"),
                "fitness": metrics["fitness_total"], "MCC_raw": metrics["MCC_raw"],
                "MCC_clipped": metrics["MCC_clipped"], "PR_AUC": metrics["PR_AUC"],
                "Brier": metrics["Brier"], "R2_symbolic": metrics["R2_symbolic"],
                "latency_ms": metrics["latency_ms"], "penalty_lat": metrics["penalty_lat"],
                **bases, "Cell_Type": config.get("Cell_Type"), "Input_Regime": config.get("Input_Regime"),
                "hidden_size": config.get("hidden_size"), "lr": config.get("lr"),
                "lam": config.get("lam"), "mu1": config.get("mu1"), "mu2": config.get("mu2"),
                "W": config.get("W"), "batch_size": config.get("batch_size"),
            }
            for gate, cfg in config.get("gates", {}).items():
                for pname, val in cfg.items():
                    if pname == "Base":
                        continue
                    row[f"{gate}__{pname}"] = val
            rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "search_history.csv")
        df.to_csv(path, index=False)
        return {"path": path, "n_rows": len(df)}

    # ── Graphiques (sections 9-14) ───────────────────────────────────────────

    def plot_sensitivity_indices(self, output_dir: str, dpi: int = 300) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        produced = {}
        produced.update(self._plot_sobol_s1(output_dir, dpi))
        produced.update(self._plot_sobol_st(output_dir, dpi))
        produced.update(self._plot_morris(output_dir, dpi))
        produced.update(self._plot_convergence(output_dir, dpi))
        return produced

    def _sensitivity_rows(self) -> list:
        payload = (self.results or {}).get("sensitivity_rankings")
        return payload["rankings"] if payload else []

    def _plot_sobol_s1(self, output_dir: str, dpi: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [r for r in self._sensitivity_rows() if r.get("S_i") is not None]
        path = os.path.join(output_dir, "sobol_barplot_S1.png")
        if not rows:
            self._warn("sensitivity_rankings.json absent/vide : sobol_barplot_S1.png non généré.")
            return {}
        rows = sorted(rows, key=lambda r: -r["S_i"])

        names = [r["param_name"] for r in rows]
        values = [r["S_i"] for r in rows]
        colors = [PALETTE.get(r.get("status"), PALETTE["Neutre"]) for r in rows]
        errs = [((r["S_Ti_conf_95"][1] - r["S_Ti_conf_95"][0]) / 2 if r.get("S_i_conf_95") is None
                and r.get("S_Ti_conf_95") else None) for r in rows]
        errs = [r.get("S_i_conf_95") and (r["S_i_conf_95"][1] - r["S_i_conf_95"][0]) / 2 for r in rows]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(rows))))
        y_pos = np.arange(len(rows))
        ax.barh(y_pos, values, xerr=[e if e else 0 for e in errs], color=colors,
               capsize=3 if any(errs) else 0)
        ax.axvline(0.10, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("S_i")
        ax.set_title("Indices de Sobol premier ordre (S_i) — Étape 2 MKAN")
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return {"sobol_barplot_S1": path}

    def _plot_sobol_st(self, output_dir: str, dpi: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [r for r in self._sensitivity_rows() if r.get("S_Ti") is not None]
        path = os.path.join(output_dir, "sobol_barplot_ST.png")
        if not rows:
            self._warn("sensitivity_rankings.json absent/vide : sobol_barplot_ST.png non généré.")
            return {}
        rows = sorted(rows, key=lambda r: -r["S_Ti"])

        names = [r["param_name"] for r in rows]
        s_i = [r.get("S_i") or 0.0 for r in rows]
        s_ti = [r["S_Ti"] for r in rows]
        colors = [PALETTE.get(r.get("status"), PALETTE["Neutre"]) for r in rows]
        errs = [r.get("S_Ti_conf_95") and (r["S_Ti_conf_95"][1] - r["S_Ti_conf_95"][0]) / 2 for r in rows]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(rows))))
        y_pos = np.arange(len(rows))
        ax.barh(y_pos, s_ti, color=colors, alpha=0.9, label="S_Ti",
               xerr=[e if e else 0 for e in errs], capsize=3 if any(errs) else 0)
        ax.barh(y_pos, s_i, color="black", alpha=0.35, label="S_i", height=0.4)
        ax.axvline(0.10, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, max(s_ti) * 1.1 if s_ti else 1.0))
        ax.set_xlabel("Indice de Sobol")
        ax.set_title("Indices de Sobol totaux (S_Ti) — Étape 2 MKAN")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return {"sobol_barplot_ST": path}

    def _plot_morris(self, output_dir: str, dpi: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        payload = (self.results or {}).get("morris_results")
        path = os.path.join(output_dir, "morris_scatter.png")
        if not payload or not payload.get("results"):
            self._warn("morris_results.json absent/vide : morris_scatter.png non généré.")
            return {}

        threshold = payload["metadata"]["threshold"]   # 0.05*std(fitness), réellement calculé
        rows = payload["results"]
        mu_star = [r["mu_star"] for r in rows]
        sigma = [r["sigma"] for r in rows]
        status_by_name = {r["param_name"]: r for r in self._sensitivity_rows()}
        colors = []
        for r in rows:
            s = status_by_name.get(r["param_name"], {}).get("status")
            colors.append(PALETTE.get(s, PALETTE["Neutre"]))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(mu_star, sigma, c=colors, alpha=0.8, edgecolors="black", linewidths=0.3)
        for r in rows:
            if r["mu_star"] > threshold or r["sigma"] > threshold:
                ax.annotate(r["param_name"], (r["mu_star"], r["sigma"]), fontsize=6,
                           xytext=(3, 3), textcoords="offset points")
        ax.axvline(threshold, color="grey", linestyle=":", linewidth=1)
        ax.axhline(threshold, color="grey", linestyle=":", linewidth=1)
        x_max = max(mu_star + [threshold]) * 1.15 + 1e-9
        y_max = max(sigma + [threshold]) * 1.15 + 1e-9
        ax.text(x_max * 0.98, y_max * 0.98, "Interactifs", ha="right", va="top",
               fontsize=8, color="grey")
        ax.text(x_max * 0.98, threshold * 0.5, "Linéaires importants", ha="right", va="center",
               fontsize=8, color="grey")
        ax.text(threshold * 0.5, threshold * 0.5, "Négligeables", ha="center", va="center",
               fontsize=8, color="grey")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("mu* (importance moyenne)")
        ax.set_ylabel("sigma (non-linéarité / interaction)")
        ax.set_title("Analyse Morris — Criblage des hyperparamètres MKAN")
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return {"morris_scatter": path}

    def _plot_convergence(self, output_dir: str, dpi: int) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        log = (self.results or {}).get("heuristic_log")
        path = os.path.join(output_dir, "fitness_convergence_extended.png")
        if not log:
            self._warn("extended_search_log.jsonl absent/vide : fitness_convergence_extended.png non généré.")
            return {}

        by_gen: dict = {}
        for rec in log:
            gen = rec.get("generation")
            fit = rec.get("fitness")
            if gen is None or fit is None:
                continue
            by_gen.setdefault(gen, []).append(fit)

        gens = sorted(by_gen.keys())
        cum = 0
        x_cum, y_best, y_mean, y_std = [], [], [], []
        running_best = float("-inf")
        for gen in gens:
            scores = by_gen[gen]
            cum += len(scores)
            running_best = max(running_best, max(scores))
            x_cum.append(cum)
            y_best.append(running_best)
            y_mean.append(float(np.mean(scores)))
            y_std.append(float(np.std(scores)))

        grid_events = [rec for rec in log if isinstance(rec.get("config"), dict)
                      and rec["config"].get("event") == "grid_extension"]

        n_windows = None
        cfg = (self.results or {}).get("heuristic_best_config")
        if cfg:
            n_windows = cfg.get("metadata", {}).get("n_windows_fitness")
        n_windows_label = f"{n_windows:,}".replace(",", " ") if n_windows else "50 000"

        fig, ax = plt.subplots(figsize=(10, 6))
        y_mean_arr, y_std_arr = np.array(y_mean), np.array(y_std)
        ax.fill_between(x_cum, y_mean_arr - y_std_arr, y_mean_arr + y_std_arr,
                        color="#FF9800", alpha=0.15, label="±1σ")
        ax.plot(x_cum, y_mean, linestyle=":", color="#FF9800", label="Fitness moyenne")
        ax.plot(x_cum, y_best, marker="o", color="#2196F3", label="Fitness best (cumulatif)")
        for ev in grid_events:   # jamais inventé : uniquement si présent dans le log
            ax.axvline(x=ev.get("individual_id", 0), color="green", linestyle="--", alpha=0.6)
            ax.annotate("Grid Extension", (ev.get("individual_id", 0), max(y_best)),
                       fontsize=7, color="green")
        ax.set_xlabel("Numéro d'évaluation cumulé")
        ax.set_ylabel("Fitness")
        ax.set_title(f"Convergence de la recherche heuristique étendue — "
                    f"{n_windows_label} fenêtres de validation")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return {"fitness_convergence_extended": path}

    # ── Rapport de synthèse (section 15) ─────────────────────────────────────

    def generate_synthesis_report(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        config, metrics = self._best_config_normalized()
        rankings = self._sensitivity_rows()
        by_name = {r["param_name"]: r for r in rankings}

        def fmt(x, na="N/A"):
            return f"{x:.4f}" if isinstance(x, (int, float)) else na

        lines = ["# Résultats de l'Étape 2 — Recherche Heuristique Étendue et Analyse de Sensibilité", ""]

        # ── 1. Configuration optimale ──
        lines.append("## 1. Configuration optimale retenue\n")
        if config:
            bases = _flatten_bases(config)
            lines.append(f"- **Cellule** : {config.get('Cell_Type', 'N/A')}")
            lines.append(f"- **Régime d'entrée** : {config.get('Input_Regime', 'N/A')}")
            lines.append(f"- hidden_size={config.get('hidden_size')}, lr={config.get('lr')}, "
                        f"lam={config.get('lam')}, mu1={config.get('mu1')}, mu2={config.get('mu2')}, "
                        f"batch_size={config.get('batch_size')}, W={config.get('W')}")
            lines.append("")
            lines.append("| Porte | Base | Hyperparamètres |")
            lines.append("|---|---|---|")
            for gate in GATE_NAMES_TKAN:
                cfg = config.get("gates", {}).get(gate)
                if cfg is None:
                    lines.append(f"| {gate} | N/A (absente — {config.get('Cell_Type')}) | |")
                else:
                    hp = {k: v for k, v in cfg.items() if k != "Base"}
                    lines.append(f"| {gate} | {cfg['Base']} | {json.dumps(hp, default=str)} |")
            lines.append("")
            lines.append(f"MCC_raw={fmt(metrics['MCC_raw'])} | MCC_clipped={fmt(metrics['MCC_clipped'])} | "
                        f"PR_AUC={fmt(metrics['PR_AUC'])} | Brier={fmt(metrics['Brier'])} | "
                        f"R2_symbolic={fmt(metrics['R2_symbolic'])} | latency_ms={fmt(metrics['latency_ms'])} | "
                        f"fitness_total={fmt(metrics['fitness_total'])}")
        else:
            lines.append("Aucune configuration disponible (heuristic_best_config.json et "
                        "extended_search_log.jsonl absents).")
        lines.append("")

        # ── 2. Justification par la sensibilité ──
        lines.append("## 2. Justification par l'analyse de sensibilité\n")
        if config and rankings:
            flat = {"hidden_size": config.get("hidden_size"), "lr": config.get("lr"),
                   "lam": config.get("lam"), "mu1": config.get("mu1"), "mu2": config.get("mu2"),
                   "batch_size": config.get("batch_size"), "W": config.get("W")}
            for g, cfg in config.get("gates", {}).items():
                flat[f"Base_{g}"] = cfg["Base"]
                for k, v in cfg.items():
                    if k != "Base":
                        flat[f"{g}::{cfg['Base']}::{k}"] = v
            lines.append("| Hyperparamètre | Valeur | S_i | S_Ti | Ratio | Statut | Interprétation |")
            lines.append("|---|---|---|---|---|---|---|")
            for name, val in flat.items():
                r = by_name.get(name)
                if r is None:
                    continue
                interp = ("dépend du contexte de base" if r.get("base_context") else "global")
                ratio = r["S_Ti"] / (abs(r["S_i"]) + 1e-9) if r.get("S_i") is not None else None
                lines.append(f"| {name} | {val} | {fmt(r.get('S_i'))} | {fmt(r.get('S_Ti'))} | "
                            f"{fmt(ratio)} | {r.get('status')} | {interp} |")
        else:
            lines.append("sensitivity_rankings.json ou la configuration optimale sont indisponibles.")
        lines.append("")

        # ── 3. Hyperparamètres fixés ──
        lines.append("## 3. Hyperparamètres fixés (redondants)\n")
        redundant = [r for r in rankings if r.get("status") == "Redondant"]
        if redundant:
            lines.append("| Nom | Valeur fixe | S_Ti | mu_star | Justification |")
            lines.append("|---|---|---|---|---|")
            for r in redundant:
                lines.append(f"| {r['param_name']} | {r.get('fixed_value')} | {fmt(r.get('S_Ti'))} | "
                            f"{fmt(r.get('mu_star'))} | S_Ti < 0.02 et critère Morris faible |")
        else:
            lines.append("Aucun hyperparamètre redondant identifié (ou sensitivity_rankings.json absent).")
        lines.append("")

        # ── 4. Interactions critiques ──
        lines.append("## 4. Interactions critiques\n")
        interactive = [r for r in rankings if r.get("status") == "Interactif"]
        if interactive:
            lines.append("| Hyperparamètre | S_i | S_Ti | S_Ti - S_i | Implication |")
            lines.append("|---|---|---|---|---|")
            for r in interactive:
                delta = (r["S_Ti"] - r["S_i"]) if r.get("S_i") is not None and r.get("S_Ti") is not None else None
                lines.append(f"| {r['param_name']} | {fmt(r.get('S_i'))} | {fmt(r.get('S_Ti'))} | "
                            f"{fmt(delta)} | contribution en interaction agrégée avec le reste de "
                            f"l'espace (pas une paire identifiée) |")
            calc2 = (self.results or {}).get("sobol_results", {})
            calc2 = calc2.get("metadata", {}).get("calc_second_order") if calc2 else None
            if not calc2:
                lines.append("")
                lines.append("> calc_second_order=False : aucune paire d'interaction exacte n'est "
                            "affirmée à partir de S_Ti/S_i seul (absence d'indices S2).")
        else:
            lines.append("Aucun hyperparamètre interactif identifié (ou sensitivity_rankings.json absent).")
        lines.append("")

        # ── 5. Brut vs Traité ──
        lines.append("## 5. Comparaison Brut vs Traité\n")
        lines.extend(self._compare_by_group("Input_Regime", {"raw": "Brut", "engineered": "Traité"}))
        lines.append("")

        # ── 6. TKANCell vs GRUKANCell ──
        lines.append("## 6. TKANCell vs GRUKANCell\n")
        lines.extend(self._compare_cells())

        report = "\n".join(lines)
        path = os.path.join(output_dir, "etape2_synthesis.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        return report

    @staticmethod
    def _fmt_num(x, spec: str = ".4f", na: str = "N/A") -> str:
        return format(x, spec) if isinstance(x, (int, float)) else na

    def _compare_by_group(self, field: str, labels: dict) -> list:
        evals = self._iter_evaluations()
        if not evals:
            return ["extended_search_log.jsonl indisponible — comparaison non calculable."]
        groups: dict = {}
        for config, metrics, _ in evals:
            key = config.get(field)
            if key is None or metrics["fitness_total"] is None:
                continue
            groups.setdefault(key, []).append(metrics)

        rows = ["| Régime | n évaluations | fitness (best) | MCC (best) | PR_AUC (best) | Brier (best) |",
               "|---|---|---|---|---|---|"]
        best_by_group = {}
        for key, label in labels.items():
            metrics_list = groups.get(key)
            if not metrics_list:
                rows.append(f"| {label} | 0 | N/A (non évalué) | N/A | N/A | N/A |")
                continue
            best = max(metrics_list, key=lambda m: m["fitness_total"])
            best_by_group[key] = best
            rows.append(f"| {label} | {len(metrics_list)} | {self._fmt_num(best['fitness_total'])} | "
                       f"{self._fmt_num(best['MCC_raw'])} | {self._fmt_num(best['PR_AUC'])} | "
                       f"{self._fmt_num(best['Brier'])} |")

        keys = list(labels.keys())
        if all(k in best_by_group for k in keys):
            b0, b1 = best_by_group[keys[0]], best_by_group[keys[1]]
            d_mcc = (b1["MCC_raw"] - b0["MCC_raw"]) if isinstance(b0["MCC_raw"], (int, float)) and \
                isinstance(b1["MCC_raw"], (int, float)) else None
            d_fit = b1["fitness_total"] - b0["fitness_total"]
            rows.append("")
            rows.append(f"Δ MCC ({labels[keys[1]]} - {labels[keys[0]]}) = "
                       f"{d_mcc:.4f}" if d_mcc is not None else "Δ MCC = N/A")
            rows.append(f"Δ fitness ({labels[keys[1]]} - {labels[keys[0]]}) = {d_fit:.4f}")
        else:
            rows.append("")
            rows.append("Comparaison incomplète : au moins un régime n'a pas été évalué dans le "
                       "journal disponible — aucun résultat inventé pour le régime manquant.")
        return rows

    def _compare_cells(self) -> list:
        evals = self._iter_evaluations()
        if not evals:
            return ["extended_search_log.jsonl indisponible — comparaison non calculable."]
        by_cell: dict = {}
        for config, metrics, _ in evals:
            ct = config.get("Cell_Type")
            if ct is None or metrics["fitness_total"] is None:
                continue
            by_cell.setdefault(ct, []).append((config, metrics))

        rows = ["| Cellule | hidden_size | hidden_size_gru_adjusted | fitness (best) | MCC (best) |",
               "|---|---|---|---|---|"]
        best_tkan = max(by_cell.get("TKANCell", []), key=lambda t: t[1]["fitness_total"], default=None)
        best_gru = max(by_cell.get("GRUKANCell", []), key=lambda t: t[1]["fitness_total"], default=None)

        if best_tkan:
            c, m = best_tkan
            rows.append(f"| TKANCell | {c.get('hidden_size')} | — | "
                       f"{self._fmt_num(m['fitness_total'])} | {self._fmt_num(m['MCC_raw'])} |")
        else:
            rows.append("| TKANCell | — | — | N/A (non évalué) | N/A |")

        if best_gru:
            c, m = best_gru
            h_adj = hidden_size_gru_adjusted(c.get("hidden_size")) if c.get("hidden_size") else None
            rows.append(f"| GRUKANCell | {c.get('hidden_size')} | {h_adj} | "
                       f"{self._fmt_num(m['fitness_total'])} | {self._fmt_num(m['MCC_raw'])} |")
        else:
            rows.append("| GRUKANCell | — | — | N/A (non évalué) | N/A |")

        if best_tkan and best_gru:
            d_fit = best_gru[1]["fitness_total"] - best_tkan[1]["fitness_total"]
            rows.append("")
            rows.append(f"Δ fitness (GRUKANCell - TKANCell) = {d_fit:.4f}")
        else:
            rows.append("")
            rows.append("Comparaison incomplète : au moins une cellule n'a pas été évaluée dans le "
                       "journal disponible.")
        return rows

    # ── Validation (section 16) ──────────────────────────────────────────────

    def validate_outputs(self, output_dir: str, top_k: int = 5) -> dict:
        errors, warnings_ = [], list(self.warnings)

        def check_json(path, label):
            if not os.path.exists(path):
                errors.append(f"{label} manquant : {path}")
                return None
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                errors.append(f"{label} invalide : {exc}")
                return None

        def check_png(path, label):
            if not os.path.exists(path):
                warnings_.append(f"{label} non généré (source indisponible) : {path}")
                return
            size = os.path.getsize(path)
            if size <= 10 * 1024:
                errors.append(f"{label} trop petit ({size} o) : {path}")

        base_name = "top5_configurations" if top_k == 5 else f"top{top_k}_configurations"
        top_json = check_json(os.path.join(output_dir, f"{base_name}.json"), "top-k JSON")
        if top_json is not None:
            if len(top_json) != top_k and len(top_json) > 0:
                warnings_.append(f"top-k JSON contient {len(top_json)} entrées, attendu {top_k} "
                                "(possible si moins d'évaluations distinctes existent).")
            required_keys = {"rank", "fitness_total", "Base_Forget", "Base_Input",
                            "Base_Candidate", "Base_Output", "Cell_Type"}
            for entry in top_json:
                missing = required_keys - entry.keys()
                if missing:
                    errors.append(f"top-k JSON : entrée rang {entry.get('rank')} sans les clés {missing}")

        for fname, label in [("hyperparameter_importance.json", "hyperparameter_importance JSON"),
                             ("etape2_synthesis.md", "rapport de synthèse")]:
            path = os.path.join(output_dir, fname)
            if fname.endswith(".json"):
                check_json(path, label)
            elif not os.path.exists(path):
                warnings_.append(f"{label} non généré : {path}")

        for fname, label in [("sobol_barplot_S1.png", "sobol_barplot_S1"),
                             ("sobol_barplot_ST.png", "sobol_barplot_ST"),
                             ("morris_scatter.png", "morris_scatter"),
                             ("fitness_convergence_extended.png", "fitness_convergence_extended")]:
            check_png(os.path.join(output_dir, fname), label)

        csv_path = os.path.join(output_dir, "search_history.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                required_cols = {"generation", "individual_id", "fitness"}
                missing = required_cols - set(df.columns)
                if missing:
                    errors.append(f"search_history.csv : colonnes manquantes {missing}")
                n_expected = len(self._iter_evaluations())
                if n_expected and len(df) != n_expected:
                    warnings_.append(f"search_history.csv : {len(df)} lignes vs "
                                    f"{n_expected} évaluations normalisées disponibles.")
            except Exception as exc:   # noqa: BLE001
                errors.append(f"search_history.csv illisible : {exc}")
        else:
            warnings_.append(f"search_history.csv non généré : {csv_path}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings_}


# ══════════════════════════════════════════════════════════════════════════
# CLI (section 17)
# ══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export des résultats de l'Étape 2 MKAN.")
    parser.add_argument("--output_dir", default="/workspace/scratch/")
    parser.add_argument("--format", choices=["tables", "plots", "report", "all"], default="all")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    exporter = Step2ResultsExporter(scratch_dir=args.output_dir)
    exporter.load_all_results()

    produced_anything = False
    try:
        if args.format in ("tables", "all"):
            exporter.export_tables(args.output_dir, top_k=args.top_k)
            exporter.export_hyperparameter_importance(args.output_dir)
            exporter.export_search_history(args.output_dir)
            produced_anything = True
        if args.format in ("plots", "all"):
            exporter.plot_sensitivity_indices(args.output_dir, dpi=args.dpi)
            produced_anything = True
        if args.format in ("report", "all"):
            exporter.generate_synthesis_report(args.output_dir)
            produced_anything = True
    except Exception as exc:   # noqa: BLE001
        logger.error(f"Échec de l'export : {exc}")
        return 1

    validation = exporter.validate_outputs(args.output_dir, top_k=args.top_k)
    for w in validation["warnings"]:
        logger.warning(w)
    for e in validation["errors"]:
        logger.error(e)

    if not produced_anything:
        return 1
    return 0 if validation["valid"] or exporter.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
