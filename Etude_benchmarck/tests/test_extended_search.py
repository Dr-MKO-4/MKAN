"""
test_extended_search.py  Tests rapides pour extended_heuristic_search.py.

Aucun de ces tests ne lance une vraie recherche de plusieurs générations :
compute_fitness() est appelée au plus une fois par test, avec n_epochs_eval=1
et une poignée de comptes synthétiques (>= 100 fenêtres), et les tests
d'algorithme génétique (croisement/mutation/diversité) opèrent sur des
individus déjà en mémoire (pas de fitness_fn).
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_heuristic_search import (          # noqa: E402
    ExtendedHeuristicSearch, GATE_CANDIDATES, GATE_NAMES_TKAN, GATE_NAMES_GRU,
    BASE_PARAM_SPACE, THETA_OPT_SPACE, CONTINUOUS_THETA_OPT,
    sample_individual, validate_individual, build_model_from_individual,
    hidden_size_gru_adjusted, compute_fitness,
)


# ══════════════════════════════════════════════════════════════════════════
# Données synthétiques (colonnes MoMTSim minimales pour build_regime_frame)
# ══════════════════════════════════════════════════════════════════════════

ACTIONS = ["DEPOSIT", "CASH_IN", "DEBIT", "PAYMENT", "TRANSFER", "CASH_OUT", "REFUND"]


def make_synthetic_df(n_accounts: int = 40, n_tx_per_account: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for acc in range(n_accounts):
        for t in range(n_tx_per_account):
            rows.append({
                "nameOrig": f"C{acc}",
                "step": t,
                "isFraud": int(rng.random() < 0.15),
                "action": rng.choice(ACTIONS),
                "amount": float(rng.uniform(1, 1000)),
                "oldBalanceOrig": float(rng.uniform(0, 5000)),
                "newBalanceOrig": float(rng.uniform(0, 5000)),
                "oldBalanceDest": float(rng.uniform(0, 5000)),
                "newBalanceDest": float(rng.uniform(0, 5000)),
                "delta_B_orig": float(rng.normal()),
                "delta_B_dest": float(rng.normal()),
                "r1": float(rng.normal()),
                "r2": float(rng.normal()),
                "flag_anomalie": int(rng.random() < 0.1),
                "delta_commission": float(rng.normal()),
                "var_agent_split": float(rng.normal()),
                "rho_rupture": float(rng.normal()),
                "rho_refund": float(rng.normal()),
                "v1h": float(rng.normal()),
                "flag_nuit": int(rng.random() < 0.3),
                "rho_nouveau": float(rng.normal()),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df():
    return make_synthetic_df()


def _minimal_individual(cell_type="TKANCell", regime="engineered", seed=0):
    import random
    rng = random.Random(seed)
    ind = sample_individual(rng)
    ind["Cell_Type"] = cell_type
    ind["Input_Regime"] = regime
    ind["hidden_size"] = 8
    ind["W"] = 5
    ind["batch_size"] = 32
    gate_names = GATE_NAMES_TKAN if cell_type == "TKANCell" else GATE_NAMES_GRU
    gates = {}
    for gate in gate_names:
        base = "hybrid"
        gates[gate] = {"Base": base, **{k: v[0] if isinstance(v, list) else v[0]
                                        for k, v in BASE_PARAM_SPACE[base]["params"].items()}}
    ind["gates"] = gates
    return ind


# ══════════════════════════════════════════════════════════════════════════
# 1-2. sample_individual / validate_individual
# ══════════════════════════════════════════════════════════════════════════

class TestSampleAndValidate:

    def test_sample_individual_valid(self):
        import random
        rng = random.Random(1)
        for _ in range(30):
            ind = sample_individual(rng)
            assert validate_individual(ind, raise_on_error=True)

    def test_sample_individual_gate_count_matches_cell_type(self):
        import random
        rng = random.Random(2)
        for _ in range(30):
            ind = sample_individual(rng)
            expected = set(GATE_NAMES_TKAN if ind["Cell_Type"] == "TKANCell" else GATE_NAMES_GRU)
            assert set(ind["gates"].keys()) == expected

    def test_no_inactive_base_hyperparams_leak(self):
        """Un individu Wav-KAN ne doit jamais contenir N_points (SincKAN) ou tout
        autre hyperparamètre appartenant à une autre base."""
        import random
        rng = random.Random(3)
        for _ in range(50):
            ind = sample_individual(rng)
            for gate_cfg in ind["gates"].values():
                base = gate_cfg["Base"]
                allowed = {"Base", *BASE_PARAM_SPACE[base]["params"].keys()}
                assert set(gate_cfg.keys()) == allowed

    def test_validate_rejects_foreign_hyperparam(self):
        ind = _minimal_individual()
        ind["gates"]["Candidate"] = {"Base": "wavkan", "Mother_Wavelet": "morlet",
                                     "M_wavelets": 8, "N_points": 32}   # N_points = SincKAN
        assert validate_individual(ind) is False

    def test_validate_rejects_bad_base_for_gate(self):
        ind = _minimal_individual()
        # "linear" n'est pas dans GATE_CANDIDATES["Forget"]
        ind["gates"]["Forget"] = {"Base": "linear"}
        assert validate_individual(ind) is False

    def test_validate_rejects_wrong_gate_set_for_gru(self):
        ind = _minimal_individual(cell_type="GRUKANCell")
        ind["gates"]["Output"] = {"Base": "hybrid", "M": 8, "K": 2}   # Output n'existe pas en GRU
        assert validate_individual(ind) is False

    def test_validate_accepts_all_registered_bases_per_gate(self):
        for gate, bases in GATE_CANDIDATES.items():
            for base in bases:
                ind = _minimal_individual()
                spec = BASE_PARAM_SPACE[base]
                params = {}
                for name, domain in spec["params"].items():
                    params[name] = (sum(domain) / 2 if name in spec["continuous"] else domain[0])
                ind["gates"][gate] = {"Base": base, **params}
                assert validate_individual(ind, raise_on_error=True)


# ══════════════════════════════════════════════════════════════════════════
# 3-5. compute_fitness
# ══════════════════════════════════════════════════════════════════════════

class TestComputeFitness:

    def test_compute_fitness_synthetic_enough_windows(self, synthetic_df):
        ind = _minimal_individual()
        # 40 comptes x (8-5+1)=4 fenêtres = 160 >= 100
        result = compute_fitness(ind, synthetic_df, synthetic_df, device="cpu",
                                 n_windows_eval=1000, n_epochs_eval=1)
        assert result["error"] is None, result["error"]
        comp = result["fitness_components"]
        for key in ("MCC_raw", "MCC_clipped", "PR_AUC", "Brier", "R2_symbolic",
                   "latency_ms", "penalty_lat", "fitness_total"):
            assert key in comp
            assert isinstance(comp[key], float)
            assert not math.isnan(comp[key]) and not math.isinf(comp[key])

    def test_mcc_clipped_is_nonneg_max_of_raw(self, synthetic_df):
        ind = _minimal_individual()
        result = compute_fitness(ind, synthetic_df, synthetic_df, device="cpu",
                                 n_windows_eval=1000, n_epochs_eval=1)
        comp = result["fitness_components"]
        assert comp["MCC_clipped"] == max(0.0, comp["MCC_raw"])
        assert comp["MCC_clipped"] >= 0.0

    def test_latency_penalty_zero_under_threshold(self):
        from extended_heuristic_search import _penalty_latency
        assert _penalty_latency(50.0) == 0.0
        assert _penalty_latency(100.0) == 0.0

    def test_latency_penalty_quadratic_above_threshold(self):
        from extended_heuristic_search import _penalty_latency
        assert _penalty_latency(200.0) == pytest.approx(10.0 * 1.0 ** 2)
        assert _penalty_latency(150.0) == pytest.approx(10.0 * 0.5 ** 2)

    def test_compute_fitness_invalid_config_does_not_raise(self, synthetic_df):
        ind = _minimal_individual()
        ind["gates"]["Forget"]["Base"] = "does_not_exist"
        result = compute_fitness(ind, synthetic_df, synthetic_df, device="cpu",
                                 n_windows_eval=1000, n_epochs_eval=1)
        assert result["error"] is not None
        assert result["fitness_total"] is None


# ══════════════════════════════════════════════════════════════════════════
# 6-9. Croisement / mutation
# ══════════════════════════════════════════════════════════════════════════

class TestGeneticOperators:

    def _search(self):
        return ExtendedHeuristicSearch(population_size=6, elite_size=2, n_generations=1, seed=7)

    def test_crossover_same_base(self):
        search = self._search()
        p1 = _minimal_individual(seed=1)
        p2 = _minimal_individual(seed=2)
        p1["gates"]["Candidate"] = {"Base": "wavkan", "Mother_Wavelet": "morlet", "M_wavelets": 3}
        p2["gates"]["Candidate"] = {"Base": "wavkan", "Mother_Wavelet": "dog", "M_wavelets": 12}
        e1, e2 = search._croiser(p1, p2)
        for child in (e1, e2):
            assert validate_individual(child, raise_on_error=True)
            assert child["gates"]["Candidate"]["Base"] == "wavkan"
            assert child["gates"]["Candidate"]["Mother_Wavelet"] in ("morlet", "dog")

    def test_crossover_different_bases(self):
        search = self._search()
        p1 = _minimal_individual(seed=3)
        p2 = _minimal_individual(seed=4)
        p1["gates"]["Candidate"] = {"Base": "wavkan", "Mother_Wavelet": "morlet", "M_wavelets": 3}
        p2["gates"]["Candidate"] = {"Base": "chebyshev", "Degree_n": 5}
        e1, e2 = search._croiser(p1, p2)
        for child in (e1, e2):
            assert validate_individual(child, raise_on_error=True)
            base = child["gates"]["Candidate"]["Base"]
            assert base in ("wavkan", "chebyshev")
            # aucun hyperparamètre étranger à la base retenue ne doit fuiter
            allowed = {"Base", *BASE_PARAM_SPACE[base]["params"].keys()}
            assert set(child["gates"]["Candidate"].keys()) == allowed

    def test_mutation_continuous_stays_in_bounds(self):
        search = self._search()
        search._mutation_rate_courant = 1.0   # force la mutation de tous les continus
        ind = _minimal_individual()
        mutant = search._muter(ind)
        for key in CONTINUOUS_THETA_OPT:
            lo, hi = min(THETA_OPT_SPACE[key]), max(THETA_OPT_SPACE[key])
            assert lo <= mutant[key] <= hi
        assert validate_individual(mutant, raise_on_error=True)

    def test_mutation_discrete_replacement(self):
        search = self._search()
        search._mutation_rate_courant = 1.0
        ind = _minimal_individual()
        mutant = search._muter(ind)
        assert mutant["hidden_size"] in THETA_OPT_SPACE["hidden_size"]
        assert mutant["W"] in THETA_OPT_SPACE["W"]
        assert mutant["batch_size"] in THETA_OPT_SPACE["batch_size"]
        assert validate_individual(mutant, raise_on_error=True)


# ══════════════════════════════════════════════════════════════════════════
# 10. Diversité
# ══════════════════════════════════════════════════════════════════════════

class TestDiversity:

    def test_diversity_all_identical(self):
        search = ExtendedHeuristicSearch(population_size=5, elite_size=1, n_generations=1)
        ind = _minimal_individual()
        search._population = [ind for _ in range(5)]
        assert search._calculer_diversite() == pytest.approx(1.0 / 5.0)

    def test_diversity_all_distinct(self):
        import random
        search = ExtendedHeuristicSearch(population_size=5, elite_size=1, n_generations=1)
        rng = random.Random(0)
        search._population = [sample_individual(rng) for _ in range(5)]
        assert search._calculer_diversite() > 0.0


# ══════════════════════════════════════════════════════════════════════════
# 11-12. Checkpoint et reconstruction du meilleur modèle
# ══════════════════════════════════════════════════════════════════════════

class TestCheckpointAndBestModel:

    def test_checkpoint_save_and_resume(self, tmp_path):
        search = ExtendedHeuristicSearch(population_size=4, elite_size=1, n_generations=1,
                                         scratch_dir=str(tmp_path))
        search._initialiser()
        search._population = [_minimal_individual(seed=i) for i in range(4)]
        search._scores = [0.1, 0.5, 0.3, 0.2]
        search._journal = [
            {"generation": 0, "score": s, "diversite": 1.0, "mutation_rate": 0.35, **ind}
            for ind, s in zip(search._population, search._scores)
        ]
        search._best_individual = search._population[1]
        search._best_score = 0.5
        search._n_evaluations = 4
        search._save_checkpoint(0)

        assert os.path.exists(search._checkpoint_path)

        reloaded = ExtendedHeuristicSearch(population_size=4, elite_size=1, n_generations=1,
                                           scratch_dir=str(tmp_path))
        df = reloaded.resume(checkpoint_path=search._checkpoint_path)
        assert reloaded._best_score == 0.5
        assert reloaded._n_evaluations == 4
        assert len(df) == 1
        assert "Base_Forget" in df.columns

    def test_get_best_model_reconstructs_scorer(self, tmp_path):
        search = ExtendedHeuristicSearch(population_size=4, elite_size=1, n_generations=1,
                                         tournament_size=2, scratch_dir=str(tmp_path))
        ind = _minimal_individual()
        input_size = 12
        model = build_model_from_individual(ind, input_size)
        search._best_individual = ind
        search._best_result = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "n_params": sum(p.numel() for p in model.parameters()),
            "hidden_size_gru_adjusted": None,
            "fitness_components": {"fitness_total": 0.42},
        }
        search._save_best_candidate()

        rebuilt = search.get_best_model()
        x = torch.randn(3, ind["W"], input_size)
        with torch.no_grad():
            score = rebuilt(x)
        assert score.shape == (3,)
        assert torch.all((score >= 0) & (score <= 1))


# ══════════════════════════════════════════════════════════════════════════
# 13. hidden_size_gru_adjusted = floor((4/3) * hidden_size)
# ══════════════════════════════════════════════════════════════════════════

class TestGruAdjustment:

    @pytest.mark.parametrize("hidden_size", THETA_OPT_SPACE["hidden_size"])
    def test_formula_matches_floor_4_3(self, hidden_size):
        expected = math.floor((4.0 / 3.0) * hidden_size)
        assert hidden_size_gru_adjusted(hidden_size) == expected

    def test_known_values_from_reference_table(self):
        # Table iso_param (corrigée) du document step_2_search_extension_and_analysis.tex
        assert hidden_size_gru_adjusted(8) == 10
        assert hidden_size_gru_adjusted(16) == 21
        assert hidden_size_gru_adjusted(32) == 42
        assert hidden_size_gru_adjusted(64) == 85
        assert hidden_size_gru_adjusted(128) == 170

    def test_build_gru_model_uses_adjusted_hidden_size(self):
        ind = _minimal_individual(cell_type="GRUKANCell")
        ind["hidden_size"] = 16
        model = build_model_from_individual(ind, input_size=12)
        assert model.hidden_size == hidden_size_gru_adjusted(16) == 21
