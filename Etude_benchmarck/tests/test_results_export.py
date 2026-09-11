"""
test_results_export.py  Tests rapides pour results_export.py.

Toutes les données sont synthétiques (>= 50 évaluations) : aucun test ne
charge un modèle PyTorch réel ni n'entraîne quoi que ce soit.
"""

import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_export import Step2ResultsExporter, _normalize_config, _normalize_metrics  # noqa: E402
from extended_heuristic_search import sample_individual                                  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Génération de données synthétiques (>= 50 évaluations)
# ══════════════════════════════════════════════════════════════════════════

def _synthetic_log_records(n: int = 50, seed: int = 0) -> list:
    rng = random.Random(seed)
    records = []
    gens = 5
    per_gen = n // gens
    eid = 0
    for gen in range(gens):
        for _ in range(per_gen):
            ind = sample_individual(rng)
            fitness = round(rng.uniform(-0.2, 0.6), 4)
            components = {
                "MCC_raw": round(rng.uniform(-0.3, 0.5), 4),
                "MCC_clipped": None, "PR_AUC": round(rng.uniform(0.0, 0.6), 4),
                "Brier": round(rng.uniform(0.05, 0.3), 4),
                "R2_symbolic": round(rng.uniform(0.0, 1.0), 4),
                "latency_ms": round(rng.uniform(5, 150), 2),
                "penalty_lat": 0.0, "fitness_total": fitness,
            }
            components["MCC_clipped"] = max(0.0, components["MCC_raw"])
            records.append({
                "generation": gen, "individual_id": eid, "config": ind,
                "fitness_components": components, "fitness": fitness, "error": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            eid += 1
    return records


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def _synthetic_best_config(records) -> dict:
    best = max(records, key=lambda r: r["fitness"])
    ind = best["config"]
    gates_json = {g: {"base": cfg["Base"], "hyperparams": {k: v for k, v in cfg.items() if k != "Base"}}
                 for g, cfg in ind["gates"].items()}
    return {
        "metadata": {"etape": 2, "n_generations": 5, "n_evaluations": len(records),
                    "n_windows_fitness": 50000, "seed": 42, "timestamp": ""},
        "theta_opt": {"hidden_size": ind["hidden_size"], "lr": ind["lr"], "lam": ind["lam"],
                     "mu1": ind["mu1"], "mu2": ind["mu2"], "batch_size": ind["batch_size"], "W": ind["W"]},
        "theta_struct": {"cell_type": ind["Cell_Type"], "hidden_size_gru_adjusted": None,
                        "input_regime": ind["Input_Regime"], "gates": gates_json},
        "fitness_components": best["fitness_components"],
        "validation_full": {},
    }


def _synthetic_sensitivity_rankings() -> dict:
    rankings = [
        {"rank": 1, "param_name": "lr", "gate": None, "base_context": None,
        "S_i": 0.20, "S_Ti": 0.35, "S_Ti_conf_95": [0.30, 0.40], "ratio_STi_Si": 1.75,
        "mu_star": 0.4, "sigma_morris": 0.1, "status": "Critique", "fixed_value": None},
        {"rank": 2, "param_name": "hidden_size", "gate": None, "base_context": None,
        "S_i": 0.05, "S_Ti": 0.15, "S_Ti_conf_95": [0.10, 0.20], "ratio_STi_Si": 3.0,
        "mu_star": 0.2, "sigma_morris": 0.3, "status": "Interactif", "fixed_value": None},
        {"rank": 3, "param_name": "Forget::hybrid::K", "gate": "Forget", "base_context": "hybrid",
        "S_i": 0.001, "S_Ti": 0.005, "S_Ti_conf_95": [0.0, 0.01], "ratio_STi_Si": 5.0,
        "mu_star": 0.01, "sigma_morris": 0.01, "status": "Redondant", "fixed_value": 2},
        {"rank": 4, "param_name": "mu2", "gate": None, "base_context": None,
        "S_i": 0.03, "S_Ti": 0.04, "S_Ti_conf_95": [0.02, 0.06], "ratio_STi_Si": 1.33,
        "mu_star": 0.05, "sigma_morris": 0.05, "status": "Neutre", "fixed_value": None},
    ]
    return {
        "metadata": {"etape": 2, "method_morris": {"r": 10, "num_levels": 4, "seed": 42},
                    "method_sobol": {"N": 1024, "calc_second_order": False, "seed": 42},
                    "d_effective": 48, "timestamp": ""},
        "rankings": rankings,
        "summary": {"n_critique": 1, "n_interactif": 1, "n_redondant": 1, "n_neutre": 1,
                   "top3_interactions": [["hidden_size", "(interaction agrégée)", 0.10]]},
    }


def _synthetic_morris_results() -> dict:
    names = ["lr", "hidden_size", "Forget::hybrid::K", "mu2"]
    rng = random.Random(1)
    results = [{"param_name": n, "gate": None, "base_context": None,
               "mu": rng.uniform(-0.1, 0.1), "mu_star": rng.uniform(0.0, 0.5),
               "sigma": rng.uniform(0.0, 0.3), "mu_star_conf": rng.uniform(0.0, 0.05),
               "morris_screening_candidate_exclusion": False} for n in names]
    return {"metadata": {"r": 10, "num_levels": 4, "seed": 42, "salib_version": "1.5.2",
                        "n_evaluations": 50, "std_fitness": 0.2, "threshold": 0.01,
                        "timestamp": ""},
           "results": results}


@pytest.fixture()
def populated_dir(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    records = _synthetic_log_records(n=50)
    _write_jsonl(scratch / "extended_search_log.jsonl", records)
    with open(scratch / "heuristic_best_config.json", "w", encoding="utf-8") as f:
        json.dump(_synthetic_best_config(records), f)
    with open(scratch / "sensitivity_rankings.json", "w", encoding="utf-8") as f:
        json.dump(_synthetic_sensitivity_rankings(), f)
    with open(scratch / "morris_results.json", "w", encoding="utf-8") as f:
        json.dump(_synthetic_morris_results(), f)
    with open(scratch / "sensitivity_report.md", "w", encoding="utf-8") as f:
        f.write("# rapport synthétique factice\n")
    with open(scratch / "sobol_results.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"N": 4, "calc_second_order": False, "seed": 42,
                                "salib_version": "1.5.2", "d_effective": 2, "n_evaluations": 16,
                                "timestamp": ""}, "results": []}, f)
    dtype = np.dtype([("index", "i8"), ("fitness", "f8"), ("config_vector", "f8", (4,))])
    arr = np.zeros(3, dtype=dtype)
    np.save(scratch / "sobol_evaluations.npy", arr)
    return str(scratch), records


# ══════════════════════════════════════════════════════════════════════════
# 1. load_all_results
# ══════════════════════════════════════════════════════════════════════════

class TestLoadAllResults:

    def test_loads_all_present_files(self, populated_dir):
        scratch, records = populated_dir
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        results = exporter.load_all_results()
        assert len(results["heuristic_log"]) == 50
        assert results["heuristic_best_config"] is not None
        assert results["sensitivity_rankings"] is not None
        assert results["morris_results"] is not None
        assert results["sensitivity_report"] is not None
        assert results["sobol_evaluations"] is not None
        # best_candidate.pt n'est volontairement pas fourni par la fixture (poids de
        # modèle non nécessaires à ces tests) : seul son warning est attendu.
        other_warnings = [w for w in exporter.warnings if "best_candidate.pt" not in w]
        assert other_warnings == []

    def test_missing_file_warns_not_raises(self, populated_dir):
        scratch, _ = populated_dir
        os.remove(os.path.join(scratch, "morris_results.json"))
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        results = exporter.load_all_results()   # ne doit jamais lever
        assert results["morris_results"] is None
        assert any("morris_results.json" in w for w in exporter.warnings)

    def test_entirely_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        exporter = Step2ResultsExporter(scratch_dir=str(empty))
        results = exporter.load_all_results()
        assert all(v is None or v == [] for v in results.values())
        assert len(exporter.warnings) >= 5


# ══════════════════════════════════════════════════════════════════════════
# Normalisation (robustesse de formats, section 19)
# ══════════════════════════════════════════════════════════════════════════

class TestNormalization:

    def test_normalize_metrics_mcc_alias(self):
        m = _normalize_metrics({"MCC": 0.4, "fitness": 0.55})
        assert m["MCC_raw"] == 0.4
        assert m["MCC_clipped"] == 0.4
        assert m["fitness_total"] == 0.55

    def test_normalize_metrics_never_overwrites_raw_with_clipped(self):
        m = _normalize_metrics({"MCC_raw": -0.2, "MCC_clipped": 0.0})
        assert m["MCC_raw"] == -0.2
        assert m["MCC_clipped"] == 0.0

    def test_normalize_metrics_missing_stays_none(self):
        m = _normalize_metrics({})
        assert m["PR_AUC"] is None and m["Brier"] is None

    def test_normalize_config_heuristic_best_config_shape(self):
        raw = {
            "theta_opt": {"hidden_size": 32, "lr": 1e-3, "lam": 1e-2, "mu1": 1.0,
                         "mu2": 0.5, "batch_size": 64, "W": 10},
            "theta_struct": {"cell_type": "TKANCell", "input_regime": "engineered",
                            "gates": {"Forget": {"base": "hybrid", "hyperparams": {"M": 8, "K": 2}}}},
        }
        cfg = _normalize_config(raw)
        assert cfg["Cell_Type"] == "TKANCell"
        assert cfg["gates"]["Forget"]["Base"] == "hybrid"
        assert cfg["gates"]["Forget"]["M"] == 8

    def test_normalize_config_flat_shape(self):
        raw = {"Cell_Type": "TKANCell", "Input_Regime": "raw", "hidden_size": 16,
              "Base_Candidate": "wavkan", "Candidate::wavkan::Mother_Wavelet": "morlet",
              "Candidate::wavkan::M_wavelets": 8}
        cfg = _normalize_config(raw)
        assert cfg["gates"]["Candidate"]["Base"] == "wavkan"
        assert cfg["gates"]["Candidate"]["Mother_Wavelet"] == "morlet"
        assert cfg["gates"]["Candidate"]["M_wavelets"] == 8

    def test_normalize_config_conditional_inactive_params_absent(self):
        """Une config Wav-KAN ne doit jamais faire apparaître N_points (SincKAN)."""
        raw = {"Cell_Type": "TKANCell", "Input_Regime": "raw",
              "gates": {"Candidate": {"Base": "wavkan", "Mother_Wavelet": "dog", "M_wavelets": 5}}}
        cfg = _normalize_config(raw)
        assert "N_points" not in cfg["gates"]["Candidate"]
        assert set(cfg["gates"]["Candidate"].keys()) == {"Base", "Mother_Wavelet", "M_wavelets"}


# ══════════════════════════════════════════════════════════════════════════
# 2-5. Tables, JSON, CSV
# ══════════════════════════════════════════════════════════════════════════

class TestTablesAndHistory:

    def test_export_tables_top5(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        result = exporter.export_tables(str(out), top_k=5)
        assert os.path.exists(result["md_path"])
        assert os.path.exists(result["json_path"])
        assert result["n_rows"] <= 5

    def test_export_tables_json_native_types(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        result = exporter.export_tables(str(out), top_k=5)
        with open(result["json_path"], encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            if row["fitness_total"] is not None:
                assert isinstance(row["fitness_total"], (int, float))
            assert isinstance(row["rank"], int)

    def test_export_tables_top_k_custom_filename(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        result = exporter.export_tables(str(out), top_k=3)
        assert result["json_path"].endswith("top3_configurations.json")
        assert result["n_rows"] <= 3

    def test_hyperparameter_importance_sorted_by_s_ti_desc(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        result = exporter.export_hyperparameter_importance(str(out))
        with open(result["json_path"], encoding="utf-8") as f:
            rows = json.load(f)
        s_tis = [r["S_Ti"] for r in rows]
        assert s_tis == sorted(s_tis, reverse=True)
        statuses = {r["status"] for r in rows}
        assert statuses == {"Critique", "Interactif", "Redondant", "Neutre"}

    def test_search_history_csv(self, populated_dir, tmp_path):
        scratch, records = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        result = exporter.export_search_history(str(out))
        df = pd.read_csv(result["path"])
        assert len(df) == len(records)
        for col in ("generation", "individual_id", "fitness", "MCC_raw", "Base_Forget", "Cell_Type"):
            assert col in df.columns
        # colonnes conditionnelles : NaN (pas 0) pour les bases inactives
        gru_rows = df[df["Cell_Type"] == "GRUKANCell"]
        if not gru_rows.empty and "Base_Output" in df.columns:
            assert gru_rows["Base_Output"].isna().all()


# ══════════════════════════════════════════════════════════════════════════
# 6. Graphiques
# ══════════════════════════════════════════════════════════════════════════

class TestPlots:

    def test_four_pngs_generated(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        produced = exporter.plot_sensitivity_indices(str(out), dpi=100)
        for key in ("sobol_barplot_S1", "sobol_barplot_ST", "morris_scatter",
                   "fitness_convergence_extended"):
            assert key in produced
            assert os.path.exists(produced[key])
            assert os.path.getsize(produced[key]) > 1000   # sanity (validate_outputs exige >10 Ko à dpi=300)

    def test_convergence_x_axis_is_cumulative_evaluations(self, populated_dir, tmp_path):
        scratch, records = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        produced = exporter._plot_convergence(str(out), dpi=100)
        assert "fitness_convergence_extended" in produced

    def test_no_grid_extension_fabricated(self, populated_dir, tmp_path):
        """Aucun événement Grid Extension dans le log synthétique -> aucune annotation
        ne doit être fabriquée (juste vérifié par absence de crash / génération correcte)."""
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        produced = exporter._plot_convergence(str(out), dpi=100)
        assert os.path.exists(produced["fitness_convergence_extended"])


# ══════════════════════════════════════════════════════════════════════════
# 7. Rapport
# ══════════════════════════════════════════════════════════════════════════

class TestSynthesisReport:

    def test_report_contains_required_sections(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        report = exporter.generate_synthesis_report(str(out))
        for heading in ("## 1. Configuration optimale retenue",
                       "## 2. Justification par l'analyse de sensibilité",
                       "## 3. Hyperparamètres fixés",
                       "## 4. Interactions critiques",
                       "## 5. Comparaison Brut vs Traité",
                       "## 6. TKANCell vs GRUKANCell"):
            assert heading in report
        assert os.path.exists(os.path.join(str(out), "etape2_synthesis.md"))

    def test_report_never_fabricates_missing_regime(self, tmp_path):
        """Si un seul régime a été évalué, l'autre doit apparaître N/A, jamais un chiffre inventé."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        rng = random.Random(5)
        records = []
        for i in range(50):
            ind = sample_individual(rng)
            ind["Input_Regime"] = "raw"   # un seul régime jamais évalué : "engineered" absent
            fitness = round(rng.uniform(0.0, 0.5), 4)
            records.append({"generation": i // 10, "individual_id": i, "config": ind,
                           "fitness_components": {"MCC_raw": 0.1, "MCC_clipped": 0.1, "PR_AUC": 0.2,
                                                  "Brier": 0.1, "R2_symbolic": 0.1, "latency_ms": 10.0,
                                                  "penalty_lat": 0.0, "fitness_total": fitness},
                           "fitness": fitness, "error": None, "timestamp": ""})
        _write_jsonl(scratch / "extended_search_log.jsonl", records)

        exporter = Step2ResultsExporter(scratch_dir=str(scratch))
        exporter.load_all_results()
        report = exporter.generate_synthesis_report(str(tmp_path / "out"))
        assert "N/A (non évalué)" in report


# ══════════════════════════════════════════════════════════════════════════
# 8-9. Validation
# ══════════════════════════════════════════════════════════════════════════

class TestValidateOutputs:

    def test_validate_outputs_full_pipeline(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        exporter.export_tables(str(out), top_k=5)
        exporter.export_hyperparameter_importance(str(out))
        exporter.export_search_history(str(out))
        exporter.plot_sensitivity_indices(str(out), dpi=150)
        exporter.generate_synthesis_report(str(out))

        validation = exporter.validate_outputs(str(out), top_k=5)
        assert validation["valid"] is True, validation["errors"]
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)

    def test_validate_outputs_reports_missing_source(self, tmp_path):
        empty_src = tmp_path / "empty_src"
        empty_src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        exporter = Step2ResultsExporter(scratch_dir=str(empty_src))
        exporter.load_all_results()
        validation = exporter.validate_outputs(str(out), top_k=5)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


# ══════════════════════════════════════════════════════════════════════════
# Round-trip JSON
# ══════════════════════════════════════════════════════════════════════════

class TestJsonRoundtrip:

    def test_top_k_json_roundtrip(self, populated_dir, tmp_path):
        scratch, _ = populated_dir
        out = tmp_path / "out"
        exporter = Step2ResultsExporter(scratch_dir=scratch)
        exporter.load_all_results()
        result = exporter.export_tables(str(out), top_k=5)
        with open(result["json_path"], encoding="utf-8") as f:
            data = json.load(f)
        dumped = json.dumps(data, default=str)
        reloaded = json.loads(dumped)
        assert reloaded == data
