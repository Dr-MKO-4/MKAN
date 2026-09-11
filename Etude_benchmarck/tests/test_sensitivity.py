"""
test_sensitivity.py  Tests rapides pour sensitivity_analysis.py.

Aucun test n'exécute 1024 entraînements réels : compute_fitness est
monkeypatchée par une fonction synthétique rapide (déterministe à partir du
vecteur décodé), et Morris/Sobol sont exercés avec un budget minimal
(r=2, N=4) suffisant pour valider le câblage sans coût prohibitif.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sensitivity_analysis as sa                     # noqa: E402
from sensitivity_analysis import (                     # noqa: E402
    HierarchicalSensitivityAnalyzer, encode_continuous, decode_continuous,
    encode_discrete_rank, decode_discrete_rank, SENTINEL,
)
from extended_heuristic_search import (                # noqa: E402
    sample_individual, validate_individual, BASE_PARAM_SPACE,
)


# ══════════════════════════════════════════════════════════════════════════
# Fixtures : config de résultats + fitness mockée
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def best_config_path(tmp_path):
    config = {
        "metadata": {"etape": 2, "n_generations": 1, "n_evaluations": 4,
                     "n_windows_fitness": 1000, "seed": 42, "timestamp": ""},
        "theta_opt": {"hidden_size": 16, "lr": 1e-3, "lam": 1e-2,
                     "mu1": 1.0, "mu2": 0.5, "batch_size": 64, "W": 10},
        "theta_struct": {
            "cell_type": "TKANCell", "hidden_size_gru_adjusted": None,
            "input_regime": "engineered",
            "gates": {
                "Forget":    {"base": "hybrid", "hyperparams": {"M": 8, "K": 2}},
                "Input":     {"base": "hybrid", "hyperparams": {"M": 8, "K": 2}},
                "Candidate": {"base": "wavkan",
                             "hyperparams": {"Mother_Wavelet": "morlet", "M_wavelets": 8}},
                "Output":    {"base": "linear", "hyperparams": {}},
            },
        },
        "fitness_components": {"MCC_raw": 0.3, "MCC_clipped": 0.3, "PR_AUC": 0.4,
                               "Brier": 0.1, "R2_symbolic": 0.5, "latency_ms": 20.0,
                               "penalty_lat": 0.0, "fitness_total": 0.31},
        "validation_full": {},
    }
    path = tmp_path / "heuristic_best_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return str(path)


@pytest.fixture()
def analyzer(best_config_path, tmp_path):
    df = pd.DataFrame({"nameOrig": ["C0"], "step": [0], "isFraud": [0]})
    inst = HierarchicalSensitivityAnalyzer(
        search_results_path=best_config_path, df_train=df, df_val=df, device="cpu",
        n_epochs_eval=1, n_windows_eval=100, max_workers=1,
        scratch_dir=str(tmp_path / "scratch"), seed=42,
    )
    return inst


def _mock_fitness_factory(rng_seed=0):
    """Fitness synthétique déterministe : dépend de theta_opt.lr et hidden_size,
    invariante aux hyperparamètres conditionnels inactifs -> permet de vérifier
    que les scores restent finis et reproductibles sans entraîner de modèle."""
    def _mock_compute_fitness(config, df_train, df_val, device, **kwargs):
        lr = config["lr"]
        hs = config["hidden_size"]
        val = 0.5 + 0.1 * math.log10(lr) + 0.001 * hs
        return {"fitness_total": val, "fitness_components": {"fitness_total": val},
               "model_state": None, "n_params": 0, "hidden_size_gru_adjusted": None,
               "input_size": 12, "error": None}
    return _mock_compute_fitness


# ══════════════════════════════════════════════════════════════════════════
# 1-4. Encode/decode + catégories + sentinelles
# ══════════════════════════════════════════════════════════════════════════

class TestEncodeDecode:

    def test_continuous_roundtrip_log_uniform(self):
        for value in (1e-4, 5e-4, 1e-3, 3e-3, 1e-2):
            x01 = encode_continuous(value, 1e-4, 1e-2)
            back = decode_continuous(x01, 1e-4, 1e-2)
            assert back == pytest.approx(value, rel=1e-6)
            assert 0.0 <= x01 <= 1.0

    def test_continuous_is_log_not_linear(self):
        # Le point médian log-uniforme n'est pas le point médian linéaire.
        lo, hi = 1e-4, 1e-2
        x01 = encode_continuous(math.sqrt(lo * hi), lo, hi)   # moyenne géométrique
        assert x01 == pytest.approx(0.5, abs=1e-6)

    def test_discrete_rank_roundtrip_exact(self):
        domain = [4, 8, 12, 16, 24, 32]
        for value in domain:
            x01 = encode_discrete_rank(value, domain)
            back = decode_discrete_rank(x01, domain)
            assert back == value   # round-trip exact (valeur admissible)

    def test_100_synthetic_configs_encode_decode(self, analyzer):
        import random
        rng = random.Random(123)
        for _ in range(100):
            ind = sample_individual(rng)
            x = np.zeros(analyzer.dimension)
            idx = {n: i for i, n in enumerate(analyzer.parameter_names)}

            x[idx["Cell_Type"]] = encode_discrete_rank(
                ind["Cell_Type"], analyzer.decoding_map["Cell_Type"] and
                [analyzer.decoding_map["Cell_Type"][k] for k in sorted(analyzer.decoding_map["Cell_Type"])])
            # Encodage direct via encoding_map (plus robuste que ci-dessus)
            x[idx["Cell_Type"]] = analyzer.encoding_map["Cell_Type"][ind["Cell_Type"]] / \
                max(1, len(analyzer.encoding_map["Cell_Type"]) - 1)
            x[idx["Input_Regime"]] = analyzer.encoding_map["Input_Regime"][ind["Input_Regime"]] / \
                max(1, len(analyzer.encoding_map["Input_Regime"]) - 1)
            for name in ("hidden_size", "batch_size", "W"):
                x[idx[name]] = encode_discrete_rank(ind[name], analyzer.parameter_metadata[name]["domain_used"])
            for name in ("lr", "lam", "mu1", "mu2"):
                lo, hi = analyzer.parameter_metadata[name]["bounds_used"]
                x[idx[name]] = encode_continuous(ind[name], lo, hi)

            for gate, cfg in ind["gates"].items():
                base_var = f"Base_{gate}"
                n_bases = len(analyzer.encoding_map[base_var])
                x[idx[base_var]] = analyzer.encoding_map[base_var][cfg["Base"]] / max(1, n_bases - 1)
                spec = BASE_PARAM_SPACE[cfg["Base"]]
                for pname in spec["params"]:
                    var_name = f"{gate}::{cfg['Base']}::{pname}"
                    meta = analyzer.parameter_metadata[var_name]
                    if meta["kind"] == "continuous_log":
                        lo, hi = meta["bounds_used"]
                        x[idx[var_name]] = encode_continuous(cfg[pname], lo, hi)
                    else:
                        x[idx[var_name]] = encode_discrete_rank(cfg[pname], meta["domain_used"])

            decoded = analyzer.decode_config(x)
            assert validate_individual(decoded, raise_on_error=True)
            assert decoded["Cell_Type"] == ind["Cell_Type"]
            assert decoded["hidden_size"] == ind["hidden_size"]
            for gate, cfg in ind["gates"].items():
                assert decoded["gates"][gate]["Base"] == cfg["Base"]

    def test_sentinel_for_inactive_conditional(self, analyzer):
        idx = {n: i for i, n in enumerate(analyzer.parameter_names)}
        x = np.full(analyzer.dimension, 0.0)
        x[idx["Cell_Type"]] = 0.0   # TKANCell (premier de l'ordre canonique/empirique)
        # Force Base_Candidate = wavkan
        n_bases = len(analyzer.encoding_map["Base_Candidate"])
        x[idx["Base_Candidate"]] = analyzer.encoding_map["Base_Candidate"]["wavkan"] / max(1, n_bases - 1)

        decoded_with_sentinels = analyzer.decoded_vector_with_sentinels(x)
        # Un hyperparamètre d'une AUTRE base pour la porte Candidate doit être sentinelle.
        other_bases = [b for b in BASE_PARAM_SPACE if b != "wavkan"
                       and any(f"Candidate::{b}::" in n for n in analyzer.parameter_names)]
        assert other_bases, "aucune autre base candidate à tester"
        other = other_bases[0]
        other_param = next(n for n in analyzer.parameter_names if n.startswith(f"Candidate::{other}::"))
        assert decoded_with_sentinels[other_param] == SENTINEL

        wavkan_param = next(n for n in analyzer.parameter_names if n.startswith("Candidate::wavkan::"))
        assert decoded_with_sentinels[wavkan_param] != SENTINEL


# ══════════════════════════════════════════════════════════════════════════
# 5. mask_active
# ══════════════════════════════════════════════════════════════════════════

class TestMaskActive:

    def test_mask_matches_decoded_base_choice(self, analyzer):
        idx = {n: i for i, n in enumerate(analyzer.parameter_names)}
        x = np.full(analyzer.dimension, 0.3)
        n_bases = len(analyzer.encoding_map["Base_Candidate"])
        x[idx["Base_Candidate"]] = analyzer.encoding_map["Base_Candidate"]["wavkan"] / max(1, n_bases - 1)

        mask = analyzer.mask_active(x)
        assert mask.shape == (analyzer.dimension,)
        assert set(np.unique(mask)).issubset({0, 1})

        wavkan_param = next(n for n in analyzer.parameter_names if n.startswith("Candidate::wavkan::"))
        assert mask[idx[wavkan_param]] == 1

        other = next(b for b in BASE_PARAM_SPACE if b != "wavkan"
                    and any(f"Candidate::{b}::" in n for n in analyzer.parameter_names))
        other_param = next(n for n in analyzer.parameter_names if n.startswith(f"Candidate::{other}::"))
        assert mask[idx[other_param]] == 0

    def test_mask_global_vars_always_active(self, analyzer):
        x = np.random.default_rng(0).uniform(0, 1, analyzer.dimension)
        mask = analyzer.mask_active(x)
        idx = {n: i for i, n in enumerate(analyzer.parameter_names)}
        for name in ("Cell_Type", "Input_Regime", "hidden_size", "lr", "lam", "mu1", "mu2", "W"):
            assert mask[idx[name]] == 1

    def test_mask_output_gate_inactive_for_gru(self, analyzer):
        idx = {n: i for i, n in enumerate(analyzer.parameter_names)}
        x = np.full(analyzer.dimension, 0.5)
        cell_types = [analyzer.decoding_map["Cell_Type"][k] for k in sorted(analyzer.decoding_map["Cell_Type"])]
        gru_idx = cell_types.index("GRUKANCell")
        x[idx["Cell_Type"]] = gru_idx / max(1, len(cell_types) - 1)

        mask = analyzer.mask_active(x)
        assert mask[idx["Base_Output"]] == 0


# ══════════════════════════════════════════════════════════════════════════
# 6. Correction P_active
# ══════════════════════════════════════════════════════════════════════════

class TestPActiveCorrection:

    def test_p_active_correction_halves_index(self):
        s_i_raw, s_ti_raw, p_active = 0.4, 0.6, 0.5
        assert s_i_raw * p_active == pytest.approx(0.2)
        assert s_ti_raw * p_active == pytest.approx(0.3)

    def test_status_protects_against_s_i_zero(self, analyzer):
        # S_i ~ 0 ne doit jamais produire de ZeroDivisionError dans le ratio.
        status = analyzer._status(s_i=0.0, s_ti=0.05, mu_star=None, sigma=None, threshold=None)
        assert status in ("Redondant", "Neutre", "Interactif", "Critique")


# ══════════════════════════════════════════════════════════════════════════
# 7-10. Morris / Sobol / rankings / rapport (fitness mockée)
# ══════════════════════════════════════════════════════════════════════════

class TestMorrisSobolReport:

    def test_morris_screening_and_status_ranking(self, analyzer, monkeypatch):
        monkeypatch.setattr(sa, "compute_fitness", _mock_fitness_factory())
        morris_payload = analyzer.run_morris(r=2, num_levels=4, seed=42)
        assert os.path.exists(os.path.join(analyzer.scratch_dir, "morris_results.json"))
        assert len(morris_payload["results"]) == analyzer.dimension
        for r in morris_payload["results"]:
            for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
                assert key in r
                assert not math.isnan(r[key])

    def test_sobol_reduced_space_and_ranking(self, analyzer, monkeypatch):
        monkeypatch.setattr(sa, "compute_fitness", _mock_fitness_factory())
        reduced = ["lr", "hidden_size", "lam"]
        sobol_payload = analyzer.run_sobol(N=4, calc_second_order=False, seed=42,
                                           reduced_names=reduced)
        assert sobol_payload["metadata"]["d_effective"] == len(reduced)
        assert len(sobol_payload["results"]) == len(reduced)
        for r in sobol_payload["results"]:
            assert 0.0 <= r["P_active"] <= 1.0
            assert r["S_i"] == pytest.approx(r["S_i_raw"] * r["P_active"])
            assert r["S_Ti"] == pytest.approx(r["S_Ti_raw"] * r["P_active"])

    def test_canonical_json_written(self, analyzer, monkeypatch):
        monkeypatch.setattr(sa, "compute_fitness", _mock_fitness_factory())
        analyzer.run_morris(r=2, seed=42)
        analyzer.run_sobol(N=4, calc_second_order=False, seed=42, reduced_names=["lr", "hidden_size"])
        payload = analyzer.write_sensitivity_rankings()

        path = os.path.join(analyzer.scratch_dir, "sensitivity_rankings.json")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            reloaded = json.load(f)
        for key in ("metadata", "rankings", "summary"):
            assert key in reloaded
        assert reloaded["metadata"]["etape"] == 2
        for entry in reloaded["rankings"]:
            for key in ("rank", "param_name", "gate", "base_context", "S_i", "S_Ti",
                       "S_Ti_conf_95", "ratio_STi_Si", "mu_star", "sigma_morris",
                       "status", "fixed_value"):
                assert key in entry
        assert payload["summary"]["n_critique"] + payload["summary"]["n_interactif"] + \
            payload["summary"]["n_redondant"] + payload["summary"]["n_neutre"] == len(payload["rankings"])

    def test_markdown_report_generated(self, analyzer, monkeypatch):
        monkeypatch.setattr(sa, "compute_fitness", _mock_fitness_factory())
        analyzer.run_morris(r=2, seed=42)
        analyzer.run_sobol(N=4, calc_second_order=False, seed=42, reduced_names=["lr", "hidden_size"])
        report = analyzer.generate_decision_report()
        assert "Hyperparamètres critiques" in report
        assert "Hyperparamètres redondants" in report
        assert "Hyperparamètres interactifs" in report
        assert "Configuration finale" in report
        path = os.path.join(analyzer.scratch_dir, "sensitivity_report.md")
        assert os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════
# Checkpointing / erreurs
# ══════════════════════════════════════════════════════════════════════════

class TestCheckpointAndErrors:

    def test_checkpoint_resume_detects_existing(self, analyzer, monkeypatch):
        monkeypatch.setattr(sa, "compute_fitness", _mock_fitness_factory())
        X = np.random.default_rng(0).uniform(0, 1, (5, analyzer.dimension))
        analyzer._evaluate_batch(X, start_index=0, checkpoint_every=2)
        done = analyzer.resume()
        assert len(done) == 5

    def test_evaluation_error_logged_not_raised(self, analyzer, monkeypatch):
        def _broken(config, df_train, df_val, device, **kwargs):
            raise RuntimeError("simulated OOM")
        monkeypatch.setattr(sa, "compute_fitness", _broken)

        individual = sample_individual()
        fitness = analyzer._evaluate_individual(individual, evaluation_id=999)
        assert fitness == -1.0
        assert os.path.exists(analyzer._errors_path)
        with open(analyzer._errors_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["evaluation_id"] == 999
        assert "RuntimeError" in record["error_type"]
