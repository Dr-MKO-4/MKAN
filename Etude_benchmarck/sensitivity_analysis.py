"""
sensitivity_analysis.py  Étape 2 : analyse de sensibilité hiérarchique
(Sobol + Morris) sur l'espace conditionnel de ExtendedHeuristicSearch.

Référence : step_2_search_extension_and_analysis.tex, section
"Analyse de sensibilité hiérarchique : méthodes de Sobol et Morris".

─── Principe ──────────────────────────────────────────────────────────────
SALib exige un espace numérique de dimension fixe (`problem = {"num_vars",
"names", "bounds"}`). L'espace conditionnel de MKAN ne l'est pas : les
hyperparamètres P(Base_g) n'existent que si Base_g est choisie, et la porte
Output elle-même n'existe pas si Cell_Type = "GRUKANCell". La solution
retenue (cf. section "Traitement des variables conditionnelles" du document)
est un espace numérique de dimension fixe où :
  - chaque variable est encodée dans [0, 1] (catégorielle : rang normalisé
    sur un ordre déterministe ; continue : log-uniforme ; discrète : rang
    normalisé sur la grille théorique/empirique) ;
  - une variable conditionnelle inactive pour l'individu décodé porte la
    valeur sentinelle -9999 dans le VECTEUR DÉCODÉ EN UNITÉS RÉELLES (jamais
    dans le vecteur [0,1] envoyé à SALib, qui reste toujours défini) ;
  - mask_active() indique quelles composantes sont réellement actives pour
    un tirage donné, ce qui permet la correction P_active des indices de
    Sobol/Morris (section "Attention aux variables conditionnelles").

Ce module réutilise directement l'infrastructure de extended_heuristic_search.py
(BASE_PARAM_SPACE, GATE_CANDIDATES, THETA_OPT_SPACE, validate_individual,
compute_fitness, hidden_size_gru_adjusted) : un vecteur décodé produit
exactement un individu au format ExtendedHeuristicSearch, évalué par la
MÊME compute_fitness() que la recherche heuristique (protocole différent :
n_epochs_eval/n_windows_eval réduits pour l'analyse de sensibilité, jamais
les 50 000 fenêtres / 10 époques de la recherche finale).

─── Écart documenté ────────────────────────────────────────────────────────
Le tableau théorique mentionne SincKAN comme candidat de la porte Candidate.
Comme documenté dans extended_heuristic_search.py, SincKAN n'est pas
implémenté dans le dépôt (BASIS_REGISTRY n'a pas d'entrée correspondante) :
aucune analyse intra-groupe SincKAN n'est produite (section "ne jamais
prétendre avoir calculé un groupe si aucune évaluation suffisante n'est
disponible") ; les groupes intra-Candidate réellement analysés sont ceux de
GATE_CANDIDATES["Candidate"] (hybrid, wavkan, relukan, chebyshev, fourier, fkan).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extended_heuristic_search import (               # noqa: E402
    BASE_PARAM_SPACE, GATE_CANDIDATES, GATE_NAMES_TKAN, GATE_NAMES_GRU,
    THETA_OPT_SPACE, CONTINUOUS_THETA_OPT, CELL_TYPES, INPUT_REGIMES,
    validate_individual, compute_fitness,
)

SENTINEL = -9999.0


# ══════════════════════════════════════════════════════════════════════════
# 1. Construction dynamique de l'espace numérique fixe
# ══════════════════════════════════════════════════════════════════════════

def _global_categorical_specs() -> dict:
    return {"Cell_Type": list(CELL_TYPES), "Input_Regime": list(INPUT_REGIMES)}


def _build_parameter_space() -> "tuple[list, dict]":
    """
    Construit (parameter_names, parameter_metadata) de façon entièrement
    dérivée de extended_heuristic_search (jamais de dimension codée en dur).

    Ordre canonique : variables globales, puis pour chaque porte (ordre
    GATE_NAMES_TKAN) : le sélecteur Base_<porte>, puis pour chaque base
    candidate (ordre GATE_CANDIDATES[porte]) chacun de ses hyperparamètres
    structurels (ordre BASE_PARAM_SPACE[base]["params"]).
    """
    names: list = []
    meta: dict = {}

    for name, domain in _global_categorical_specs().items():
        names.append(name)
        meta[name] = {"kind": "categorical", "domain": list(domain),
                      "gate": None, "base_context": None}
    for name in ("hidden_size", "batch_size", "W"):
        names.append(name)
        meta[name] = {"kind": "discrete_rank", "domain": list(THETA_OPT_SPACE[name]),
                      "gate": None, "base_context": None}
    for name in ("lr", "lam", "mu1", "mu2"):
        lo, hi = min(THETA_OPT_SPACE[name]), max(THETA_OPT_SPACE[name])
        names.append(name)
        meta[name] = {"kind": "continuous_log", "bounds": (lo, hi),
                      "gate": None, "base_context": None}

    for gate in GATE_NAMES_TKAN:
        base_var = f"Base_{gate}"
        names.append(base_var)
        meta[base_var] = {"kind": "categorical", "domain": list(GATE_CANDIDATES[gate]),
                          "gate": gate, "base_context": None, "is_base_selector": True}
        for base in GATE_CANDIDATES[gate]:
            spec = BASE_PARAM_SPACE[base]
            for pname, domain in spec["params"].items():
                var_name = f"{gate}::{base}::{pname}"
                names.append(var_name)
                if pname in spec["continuous"]:
                    lo, hi = domain
                    meta[var_name] = {"kind": "continuous_log", "bounds": (lo, hi),
                                      "gate": gate, "base_context": base, "param": pname}
                else:
                    meta[var_name] = {"kind": "discrete_rank", "domain": list(domain),
                                      "gate": gate, "base_context": base, "param": pname}
    return names, meta


# ══════════════════════════════════════════════════════════════════════════
# 2-6. Encodage / décodage
# ══════════════════════════════════════════════════════════════════════════

def encode_continuous(value: float, lower: float, upper: float) -> float:
    """Encodage log-uniforme dans [0,1] : log(lower)/log(upper), PAS d'interpolation linéaire."""
    if upper <= lower:
        return 0.5
    log_lo, log_hi = math.log(lower), math.log(upper)
    value = min(max(value, lower), upper)
    return (math.log(value) - log_lo) / (log_hi - log_lo)


def decode_continuous(value01: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return lower
    value01 = min(max(value01, 0.0), 1.0)
    log_lo, log_hi = math.log(lower), math.log(upper)
    # clip final : exp(log(upper)) peut dépasser `upper` de quelques ULP en
    # float64 (round-trip lr=1e-2 -> ~1.0000000000000004e-2), ce qui ferait
    # échouer validate_individual sur une borne stricte.
    return min(max(math.exp(log_lo + value01 * (log_hi - log_lo)), lower), upper)


def _rank_index(value01: float, n: int) -> int:
    if n <= 1:
        return 0
    return int(round(min(max(value01, 0.0), 1.0) * (n - 1)))


def encode_discrete_rank(value, domain: list) -> float:
    """Encodage par rang normalisé : indice dans la grille / (taille-1)."""
    idx = domain.index(value)
    return 0.0 if len(domain) <= 1 else idx / (len(domain) - 1)


def decode_discrete_rank(value01: float, domain: list):
    return domain[_rank_index(value01, len(domain))]


class HierarchicalSensitivityAnalyzer:
    """
    Analyse de sensibilité hiérarchique (Sobol + Morris) sur l'espace
    conditionnel de ExtendedHeuristicSearch.

    Args:
        search_results_path : heuristic_best_config.json OU top5_configs.json
                               (le format est détecté automatiquement).
        df_train, df_val     : DataFrames transactionnels bruts (mêmes colonnes
                               que compute_fitness de extended_heuristic_search).
        device               : device d'entraînement demandé par l'appelant.
                               Note (section 15, sérialisation multiprocessing) :
                               les évaluations parallèles s'exécutent TOUJOURS
                               sur CPU dans les workers, indépendamment de ce
                               paramètre — un contexte CUDA/DirectML ne se
                               partage pas entre processus, et le protocole de
                               sensibilité (5 époques, 10k fenêtres) reste
                               rapide sur CPU. device n'est conservé que pour
                               un usage séquentiel (max_workers=1) explicite.
        n_epochs_eval, n_windows_eval : protocole RÉDUIT de la sensibilité
                               (distinct du protocole final 50k/10 époques
                               de la recherche heuristique — jamais confondus).
        max_workers          : ProcessPoolExecutor (défaut 2, configurable).
    """

    def __init__(
        self,
        search_results_path: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        device="cpu",
        n_epochs_eval: int = 5,
        n_windows_eval: int = 10_000,
        max_workers: int = 2,
        scratch_dir: Optional[str] = None,
        log_path: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.df_train = df_train
        self.df_val = df_val
        self.device = device
        self.n_epochs_eval = n_epochs_eval
        self.n_windows_eval = n_windows_eval
        self.max_workers = max_workers
        self.seed = seed

        default_scratch = "/workspace/scratch"
        if scratch_dir is None:
            try:
                os.makedirs(default_scratch, exist_ok=True)
                scratch_dir = default_scratch
            except OSError:
                scratch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "checkpoints", "sensitivity")
        os.makedirs(scratch_dir, exist_ok=True)
        self.scratch_dir = scratch_dir

        self.best_config, self.top_configs = self._load_search_results(search_results_path)

        self.parameter_names, self.parameter_metadata = _build_parameter_space()
        self.dimension = len(self.parameter_names)
        self.parameter_bounds = [[0.0, 1.0] for _ in self.parameter_names]  # espace SALib normalisé

        self._log_path = log_path or os.path.join(self.scratch_dir, "..",
                                                   "extended_search", "extended_search_log.jsonl")
        self._fallback_log: list = []
        self._apply_empirical_bounds_and_order()

        self.encoding_map, self.decoding_map = self._build_category_maps()

        self._errors_path = os.path.join(self.scratch_dir, "evaluation_errors.jsonl")
        self._checkpoint_path = os.path.join(self.scratch_dir, "sobol_evaluations.npy")
        self._n_evaluated_total = 0

    # ── Chargement des résultats de la recherche (section 18) ───────────────

    @staticmethod
    def _load_search_results(path: str) -> "tuple[Optional[dict], list]":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "theta_struct" in data:
            # heuristic_best_config.json
            ind = data["theta_struct"]
            individual = {
                "Cell_Type": ind["cell_type"], "Input_Regime": ind["input_regime"],
                **data["theta_opt"],
                "gates": {g: {"Base": cfg["base"], **cfg["hyperparams"]}
                         for g, cfg in ind["gates"].items()},
            }
            return individual, [individual]
        if isinstance(data, list):
            # top5_configs.json (ExtendedHeuristicSearch.export_top5)
            configs = [entry["individual"] for entry in data]
            return (configs[0] if configs else None), configs
        raise ValueError(f"Format non reconnu pour search_results_path={path}")

    # ── Bornes/ordre empiriques (sections 2 et 7) ────────────────────────────

    def _apply_empirical_bounds_and_order(self) -> None:
        """
        Si extended_search_log.jsonl est disponible, restreint les bornes
        continues aux min/max observés et réordonne les catégories par
        fréquence dans le top-20% ; sinon, repli déterministe sur les bornes
        théoriques (loggé dans self._fallback_log, section 7).
        """
        records = self._read_log_records()
        for name, meta in self.parameter_metadata.items():
            if meta["kind"] == "continuous_log":
                values = self._extract_values(records, name)
                if len(values) >= 5:
                    meta["bounds_used"] = (min(values), max(values))
                    meta["bounds_source"] = "empirique"
                else:
                    meta["bounds_used"] = meta["bounds"]
                    meta["bounds_source"] = "theorique (fallback)"
                    self._fallback_log.append(
                        f"{name} : {len(values)} observation(s) < 5 -> bornes théoriques {meta['bounds']}")
            elif meta["kind"] == "discrete_rank":
                values = self._extract_values(records, name)
                theoretical = meta["domain"]
                if len(values) >= 5:
                    observed = sorted({v for v in values if v in theoretical}, key=theoretical.index)
                    meta["domain_used"] = observed if observed else theoretical
                    meta["domain_source"] = "empirique" if observed else "theorique (fallback)"
                else:
                    meta["domain_used"] = theoretical
                    meta["domain_source"] = "theorique (fallback)"
                    self._fallback_log.append(
                        f"{name} : {len(values)} observation(s) < 5 -> grille théorique {theoretical}")
            elif meta["kind"] == "categorical":
                meta["domain_used"] = meta["domain"]   # ordre traité séparément (top-20%)

    def _read_log_records(self) -> list:
        path = self._log_path
        if not os.path.exists(path):
            self._fallback_log.append(f"extended_search_log.jsonl introuvable ({path}) — "
                                      "repli théorique pour toutes les variables.")
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _extract_values(records: list, var_name: str) -> list:
        """var_name = 'lr' (global) ou 'Forget::fastkan::h_bandwidth' (conditionnel)."""
        values = []
        parts = var_name.split("::")
        for rec in records:
            cfg = rec.get("config")
            if not isinstance(cfg, dict):
                continue
            if len(parts) == 1:
                if parts[0] in cfg and not isinstance(cfg[parts[0]], dict):
                    values.append(cfg[parts[0]])
            else:
                gate, base, pname = parts
                gate_cfg = cfg.get("gates", {}).get(gate)
                if gate_cfg and gate_cfg.get("Base") == base and pname in gate_cfg:
                    values.append(gate_cfg[pname])
        return values

    def _build_category_maps(self) -> "tuple[dict, dict]":
        """
        encoding_map[name] = {catégorie: indice} ; decoding_map[name] = {indice: catégorie}.
        Ordre : fréquence décroissante dans le top-20% du journal si dispo
        (>= 5 observations), sinon ordre canonique de l'espace de recherche
        (loggé, section 2 : « jamais un encodage arbitraire différent entre
        deux exécutions » — l'ordre canonique est fixe et déterministe).
        """
        records = self._read_log_records()
        valid = [r for r in records if isinstance(r.get("fitness"), (int, float))]
        top20: list = []
        if valid:
            scores = sorted((r["fitness"] for r in valid), reverse=True)
            cutoff = scores[max(0, int(len(scores) * 0.20) - 1)]
            top20 = [r for r in valid if r["fitness"] >= cutoff]

        encoding_map, decoding_map = {}, {}
        for name, meta in self.parameter_metadata.items():
            if meta["kind"] != "categorical":
                continue
            canonical = list(meta["domain"])
            observed = self._extract_values(top20, name) if len(top20) >= 5 else []
            if len(observed) >= 5:
                counts = {c: 0 for c in canonical}
                for v in observed:
                    if v in counts:
                        counts[v] += 1
                order = sorted(canonical, key=lambda c: (-counts[c], canonical.index(c)))
                source = "frequence top-20%"
            else:
                order = canonical
                source = "ordre canonique (fallback, < 5 observations top-20%)"
                self._fallback_log.append(f"{name} (catégorielle) : {source}")
            meta["category_order"] = order
            meta["category_order_source"] = source
            encoding_map[name] = {c: i for i, c in enumerate(order)}
            decoding_map[name] = {i: c for i, c in enumerate(order)}
        return encoding_map, decoding_map

    # ── Décodage / masque d'activation (sections 3-4) ────────────────────────

    def decode_config(self, x_vector: np.ndarray) -> dict:
        """
        Décode un vecteur [0,1]^d en un individu au format
        extended_heuristic_search (Cell_Type, Input_Regime, theta_opt, gates).
        Ignore les hyperparamètres conditionnels inactifs (sentinelle -9999).
        """
        x_vector = np.asarray(x_vector, dtype=float)
        idx = {name: i for i, name in enumerate(self.parameter_names)}

        cell_type = self.decoding_map["Cell_Type"][_rank_index(
            x_vector[idx["Cell_Type"]], len(self.decoding_map["Cell_Type"]))]
        input_regime = self.decoding_map["Input_Regime"][_rank_index(
            x_vector[idx["Input_Regime"]], len(self.decoding_map["Input_Regime"]))]

        individual = {"Cell_Type": cell_type, "Input_Regime": input_regime}
        for name in ("hidden_size", "batch_size", "W"):
            meta = self.parameter_metadata[name]
            individual[name] = decode_discrete_rank(x_vector[idx[name]], meta["domain_used"])
        for name in ("lr", "lam", "mu1", "mu2"):
            lo, hi = self.parameter_metadata[name]["bounds_used"]
            individual[name] = decode_continuous(x_vector[idx[name]], lo, hi)

        applicable_gates = set(GATE_NAMES_TKAN if cell_type == "TKANCell" else GATE_NAMES_GRU)
        gates = {}
        for gate in GATE_NAMES_TKAN:
            base_var = f"Base_{gate}"
            n_bases = len(self.decoding_map[base_var])
            base = self.decoding_map[base_var][_rank_index(x_vector[idx[base_var]], n_bases)]
            if gate not in applicable_gates:
                continue
            gate_cfg = {"Base": base}
            spec = BASE_PARAM_SPACE[base]
            for pname, domain in spec["params"].items():
                var_name = f"{gate}::{base}::{pname}"
                meta = self.parameter_metadata[var_name]
                if meta["kind"] == "continuous_log":
                    lo, hi = meta["bounds_used"]
                    gate_cfg[pname] = decode_continuous(x_vector[idx[var_name]], lo, hi)
                else:
                    gate_cfg[pname] = decode_discrete_rank(x_vector[idx[var_name]], meta["domain_used"])
            gates[gate] = gate_cfg
        individual["gates"] = gates
        return individual

    def mask_active(self, x_vector: np.ndarray) -> np.ndarray:
        """Masque binaire (dimension = self.dimension) : 1 si actif pour ce tirage, 0 sinon."""
        individual = self.decode_config(x_vector)
        cell_type = individual["Cell_Type"]
        applicable_gates = set(GATE_NAMES_TKAN if cell_type == "TKANCell" else GATE_NAMES_GRU)
        active_base = {gate: individual["gates"][gate]["Base"]
                       for gate in individual.get("gates", {})}

        mask = np.zeros(self.dimension, dtype=int)
        for i, name in enumerate(self.parameter_names):
            meta = self.parameter_metadata[name]
            gate, base_ctx = meta.get("gate"), meta.get("base_context")
            if gate is None:
                mask[i] = 1                                            # variable globale
            elif base_ctx is None:
                mask[i] = 1 if gate in applicable_gates else 0          # sélecteur Base_<gate>
            else:
                mask[i] = int(gate in applicable_gates and active_base.get(gate) == base_ctx)
        return mask

    def decoded_vector_with_sentinels(self, x_vector: np.ndarray) -> dict:
        """Vecteur décodé en unités réelles ; variables inactives -> SENTINEL (-9999)."""
        mask = self.mask_active(x_vector)
        individual = self.decode_config(x_vector)
        out = {}
        for i, name in enumerate(self.parameter_names):
            if not mask[i]:
                out[name] = SENTINEL
                continue
            meta = self.parameter_metadata[name]
            gate, base_ctx = meta.get("gate"), meta.get("base_context")
            if name == "Cell_Type":
                out[name] = individual["Cell_Type"]
            elif name == "Input_Regime":
                out[name] = individual["Input_Regime"]
            elif gate is None:
                out[name] = individual[name]
            elif base_ctx is None:
                out[name] = individual["gates"][gate]["Base"]
            else:
                out[name] = individual["gates"][gate][meta["param"]]
        return out

    # ── Évaluation d'un individu (section 17 : robustesse) ───────────────────

    def _evaluate_individual(self, individual: dict, evaluation_id: int) -> float:
        try:
            if not validate_individual(individual):
                raise ValueError("individu invalide (validate_individual)")
            result = compute_fitness(individual, self.df_train, self.df_val, "cpu",
                                     n_windows_eval=self.n_windows_eval,
                                     n_epochs_eval=self.n_epochs_eval, seed=self.seed)
            if result["fitness_total"] is None:
                raise RuntimeError(result.get("error", "compute_fitness a échoué"))
            fitness = result["fitness_total"]
            if math.isnan(fitness) or math.isinf(fitness):
                raise ValueError("fitness NaN/Inf")
            return float(fitness)
        except Exception as exc:   # noqa: BLE001 — section 17 : jamais interrompre l'analyse
            self._log_evaluation_error(evaluation_id, exc)
            return -1.0

    def _log_evaluation_error(self, evaluation_id: int, exc: Exception) -> None:
        record = {
            "evaluation_id": evaluation_id, "error_type": type(exc).__name__,
            "message": f"{exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Batch d'évaluations avec parallélisation + checkpointing ────────────

    def _evaluate_batch(self, X: np.ndarray, start_index: int = 0,
                        checkpoint_every: int = 50) -> np.ndarray:
        """
        Évalue chaque ligne de X (tirages [0,1]^d), avec reprise automatique
        depuis sobol_evaluations.npy (section 16) et checkpoint tous les
        `checkpoint_every` évaluations.
        """
        n = X.shape[0]
        done = self._load_checkpoint()
        Y = np.full(n, np.nan, dtype=float)
        for idx, fitness in done.items():
            if 0 <= idx - start_index < n:
                Y[idx - start_index] = fitness

        pending = [(start_index + i, X[i]) for i in range(n) if np.isnan(Y[i])]
        if not pending:
            return Y

        results: dict = {}
        if self.max_workers > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers,
                                     initializer=_init_worker,
                                     initargs=(self,)) as pool:
                for eval_id, fitness in pool.map(_worker_evaluate, pending):
                    results[eval_id] = fitness
                    if len(results) % checkpoint_every == 0:
                        self._append_checkpoint(results)
                        results = {}
        else:
            for eval_id, x_row in pending:
                fitness = self._evaluate_individual(self.decode_config(x_row), eval_id)
                results[eval_id] = fitness
                if len(results) % checkpoint_every == 0:
                    self._append_checkpoint(results)
                    results = {}
        if results:
            self._append_checkpoint(results)

        merged = self._load_checkpoint()
        for i in range(n):
            gid = start_index + i
            if gid in merged:
                Y[i] = merged[gid]
        return Y

    def _load_checkpoint(self) -> dict:
        if not os.path.exists(self._checkpoint_path):
            return {}
        arr = np.load(self._checkpoint_path, allow_pickle=True)
        return {int(row["index"]): float(row["fitness"]) for row in arr}

    def _append_checkpoint(self, new_results: dict) -> None:
        existing = self._load_checkpoint()
        existing.update(new_results)
        dtype = np.dtype([("index", "i8"), ("fitness", "f8"),
                          ("config_vector", "f8", (self.dimension,))])
        # config_vector non re-matérialisé ici (coût mémoire) : rempli à 0 ;
        # seul (index, fitness) est garanti pour la reprise (section 16).
        rows = np.zeros(len(existing), dtype=dtype)
        for i, (k, v) in enumerate(sorted(existing.items())):
            rows[i] = (k, v, np.zeros(self.dimension))
        tmp = self._checkpoint_path + ".tmp"
        np.save(tmp, rows, allow_pickle=True)
        os.replace(tmp + ".npy" if os.path.exists(tmp + ".npy") else tmp, self._checkpoint_path)

    def resume(self) -> dict:
        """Détecte automatiquement les évaluations déjà disponibles (section 16)."""
        done = self._load_checkpoint()
        print(f"{len(done)} évaluation(s) déjà disponible(s) dans {self._checkpoint_path}")
        return done

    # ── SALib : version installée (section 10) ───────────────────────────────

    @staticmethod
    def _salib_version() -> str:
        try:
            from importlib.metadata import version
            return version("SALib")
        except Exception:   # noqa: BLE001
            return "inconnue"

    def _problem(self) -> dict:
        return {"num_vars": self.dimension, "names": self.parameter_names,
               "bounds": self.parameter_bounds}

    # ── Morris (section 8-9) ─────────────────────────────────────────────────

    def run_morris(self, r: int = 10, num_levels: int = 4, seed: int = 42) -> dict:
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze

        problem = self._problem()
        X = morris_sample.sample(problem, N=r, num_levels=num_levels, seed=seed)
        Y = self._evaluate_batch(X, start_index=0)

        Si = morris_analyze.analyze(problem, X, Y, num_levels=num_levels, seed=seed)

        std_fitness = float(np.std(Y[~np.isnan(Y)])) if np.any(~np.isnan(Y)) else 0.0
        threshold = 0.05 * std_fitness

        results = []
        for i, name in enumerate(self.parameter_names):
            mu_star, sigma = float(Si["mu_star"][i]), float(Si["sigma"][i])
            candidate_exclusion = bool(mu_star < threshold and sigma < threshold)
            results.append({
                "param_name": name, "gate": self.parameter_metadata[name].get("gate"),
                "base_context": self.parameter_metadata[name].get("base_context"),
                "mu": float(Si["mu"][i]), "mu_star": mu_star, "sigma": sigma,
                "mu_star_conf": float(Si["mu_star_conf"][i]),
                # Criblage préalable UNIQUEMENT : ne pas qualifier de "redondant Sobol"
                # avant l'analyse Sobol (section 9).
                "morris_screening_candidate_exclusion": candidate_exclusion,
            })

        payload = {
            "metadata": {"r": r, "num_levels": num_levels, "seed": seed,
                        "salib_version": self._salib_version(),
                        "n_evaluations": int(len(Y)), "std_fitness": std_fitness,
                        "threshold": threshold,
                        "timestamp": datetime.now(timezone.utc).isoformat()},
            "results": results,
        }
        with open(os.path.join(self.scratch_dir, "morris_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        self._plot_morris(results)
        self._morris_results = payload
        return payload

    def _plot_morris(self, results: list) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [avertissement] matplotlib indisponible — morris_plot.png non généré.")
            return
        mu_star = [r["mu_star"] for r in results]
        sigma = [r["sigma"] for r in results]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(mu_star, sigma, alpha=0.7)
        for r in results:
            if r["morris_screening_candidate_exclusion"] is False and r["mu_star"] > 0:
                ax.annotate(r["param_name"], (r["mu_star"], r["sigma"]), fontsize=6)
        ax.set_xlabel("mu*  (importance moyenne)")
        ax.set_ylabel("sigma  (non-linéarité / interaction)")
        ax.set_title("Criblage de Morris — espace conditionnel MKAN (Étape 2)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.scratch_dir, "morris_plot.png"), dpi=150)
        plt.close(fig)

    # ── Sobol (section 10-11) ─────────────────────────────────────────────────

    def run_sobol(self, N: int = 1024, calc_second_order: bool = False, seed: int = 42,
                  reduced_names: Optional[list] = None) -> dict:
        """
        reduced_names : sous-ensemble de self.parameter_names à échantillonner
        (les autres sont fixés au point médian [0,1]->valeur centrale, cf.
        section "coût nominal N*(d_reduit+2)"). None = espace complet.
        """
        # SALib 1.5.2 : SALib.sample.saltelli est dépréciée au profit de
        # SALib.sample.sobol (API vérifiée par inspect.signature avant
        # implémentation, section 10 : « avant d'utiliser une API SALib,
        # inspecter la version installée » — saltelli.sample() n'a d'ailleurs
        # pas de paramètre `seed`, contrairement à sobol.sample()).
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze

        active_names = reduced_names or self.parameter_names
        fixed = {n: 0.5 for n in self.parameter_names if n not in active_names}

        problem = {"num_vars": len(active_names), "names": active_names,
                  "bounds": [[0.0, 1.0]] * len(active_names)}
        X_reduced = sobol_sample.sample(problem, N=N, calc_second_order=calc_second_order, seed=seed)

        idx_full = {name: i for i, name in enumerate(self.parameter_names)}
        X_full = np.tile(0.5, (X_reduced.shape[0], self.dimension))
        for j, name in enumerate(active_names):
            X_full[:, idx_full[name]] = X_reduced[:, j]
        for name, val in fixed.items():
            X_full[:, idx_full[name]] = val

        Y = self._evaluate_batch(X_full, start_index=0)
        masks = np.array([self.mask_active(X_full[i]) for i in range(X_full.shape[0])])

        Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, seed=seed)

        results = []
        for j, name in enumerate(active_names):
            i_full = idx_full[name]
            p_active = float(np.mean(masks[:, i_full]))
            s_i_raw, s_ti_raw = float(Si["S1"][j]), float(Si["ST"][j])
            s_i_conf = float(Si["S1_conf"][j]) if "S1_conf" in Si else None
            s_ti_conf = float(Si["ST_conf"][j]) if "ST_conf" in Si else None
            results.append({
                "param_name": name, "gate": self.parameter_metadata[name].get("gate"),
                "base_context": self.parameter_metadata[name].get("base_context"),
                "S_i_raw": s_i_raw, "S_Ti_raw": s_ti_raw,
                "P_active": p_active,
                # Correction P_active (section 11) : appliquée SEULEMENT aux variables
                # conditionnelles (base_context is not None) ; variables globales/sélecteurs
                # de base ont P_active=1 par construction (toujours actives).
                "S_i": s_i_raw * p_active, "S_Ti": s_ti_raw * p_active,
                "S_i_conf_95": s_i_conf, "S_Ti_conf_95": s_ti_conf,
            })

        payload = {
            "metadata": {"N": N, "calc_second_order": calc_second_order, "seed": seed,
                        "salib_version": self._salib_version(),
                        "d_effective": len(active_names),
                        "n_evaluations": int(len(Y)),
                        "timestamp": datetime.now(timezone.utc).isoformat()},
            "results": results,
        }
        with open(os.path.join(self.scratch_dir, "sobol_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        self._sobol_results = payload
        return payload

    # ── Analyse intra-groupe (section 12) ────────────────────────────────────

    def run_intragroup_analysis(self, gate: str = "Candidate", r: int = 10,
                                num_levels: int = 4, seed: int = 42) -> dict:
        """
        Analyse de sensibilité conduite séparément pour chaque base réellement
        candidate de `gate` (GATE_CANDIDATES[gate], déterminé depuis
        extended_heuristic_search — jamais une liste théorique en dur).
        Les autres variables (theta_opt, autres portes) sont fixées à la
        configuration de référence (self.best_config si disponible, sinon
        valeurs médianes de la grille théorique).
        """
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze

        base_center = self.best_config or self._default_individual()
        intragroup = {}
        for base in GATE_CANDIDATES[gate]:
            spec = BASE_PARAM_SPACE[base]
            if not spec["params"]:
                intragroup[base] = {"status": "aucun hyperparamètre structurel (base sans P(base))"}
                continue

            local_names = list(spec["params"].keys())
            local_bounds = [[0.0, 1.0]] * len(local_names)
            problem = {"num_vars": len(local_names), "names": local_names, "bounds": local_bounds}
            X_local = morris_sample.sample(problem, N=r, num_levels=num_levels, seed=seed)

            Y = np.empty(X_local.shape[0], dtype=float)
            for row_i in range(X_local.shape[0]):
                individual = self._individual_fixed_except_gate(base_center, gate, base,
                                                                 local_names, X_local[row_i])
                Y[row_i] = self._evaluate_individual(individual, evaluation_id=1_000_000 + row_i)

            if np.all(np.isnan(Y)) or len(Y) < len(local_names) + 1:
                intragroup[base] = {"status": "évaluations insuffisantes — groupe non analysé",
                                    "base_context": base, "gate": gate}
                continue

            Si = morris_analyze.analyze(problem, X_local, Y, num_levels=num_levels, seed=seed)
            intragroup[base] = {
                "base_context": base, "gate": gate,
                "params": [
                    {"param_name": name, "base_context": base, "gate": gate,
                    "mu_star": float(Si["mu_star"][i]), "sigma": float(Si["sigma"][i])}
                    for i, name in enumerate(local_names)
                ],
            }

        with open(os.path.join(self.scratch_dir, f"intragroup_{gate}.json"), "w", encoding="utf-8") as f:
            json.dump(intragroup, f, indent=2, ensure_ascii=False, default=str)
        return intragroup

    def _default_individual(self) -> dict:
        import random
        from extended_heuristic_search import sample_individual
        return sample_individual(random.Random(self.seed))

    def _individual_fixed_except_gate(self, base_center: dict, gate: str, base: str,
                                      local_names: list, x_local: np.ndarray) -> dict:
        individual = json.loads(json.dumps(base_center, default=str))  # copie profonde simple
        individual.setdefault("gates", {})
        spec = BASE_PARAM_SPACE[base]
        gate_cfg = {"Base": base}
        for j, pname in enumerate(local_names):
            domain = spec["params"][pname]
            if pname in spec["continuous"]:
                lo, hi = domain
                gate_cfg[pname] = decode_continuous(x_local[j], lo, hi)
            else:
                gate_cfg[pname] = decode_discrete_rank(x_local[j], list(domain))
        individual["gates"][gate] = gate_cfg
        applicable = GATE_NAMES_TKAN if individual["Cell_Type"] == "TKANCell" else GATE_NAMES_GRU
        individual["gates"] = {g: c for g, c in individual["gates"].items() if g in applicable}
        for g in applicable:
            if g not in individual["gates"]:
                fallback_base = GATE_CANDIDATES[g][0]
                fspec = BASE_PARAM_SPACE[fallback_base]
                individual["gates"][g] = {"Base": fallback_base,
                                          **{n: (d[0] if n not in fspec["continuous"] else sum(d) / 2)
                                             for n, d in fspec["params"].items()}}
        return individual

    # ── Rapport de décision (section 13) ─────────────────────────────────────

    def _status(self, s_i: float, s_ti: float, mu_star: Optional[float], sigma: Optional[float],
               threshold: Optional[float]) -> str:
        eps = 1e-9
        if s_ti > 0.10:
            return "Critique"
        morris_low = (mu_star is not None and threshold is not None
                      and mu_star < threshold and sigma < threshold)
        if s_ti < 0.02 and (morris_low or mu_star is None):
            return "Redondant"
        if abs(s_i) > eps and (s_ti / (abs(s_i) + eps)) > 2:
            return "Interactif"
        return "Neutre"

    def generate_decision_report(self, sobol_payload: Optional[dict] = None,
                                 morris_payload: Optional[dict] = None) -> str:
        sobol_payload = sobol_payload or getattr(self, "_sobol_results", None)
        morris_payload = morris_payload or getattr(self, "_morris_results", None)
        if sobol_payload is None:
            raise RuntimeError("run_sobol() doit être exécuté avant generate_decision_report().")

        morris_by_name = {r["param_name"]: r for r in morris_payload["results"]} if morris_payload else {}
        threshold = morris_payload["metadata"]["threshold"] if morris_payload else None

        rows = []
        for r in sobol_payload["results"]:
            m = morris_by_name.get(r["param_name"])
            status = self._status(r["S_i"], r["S_Ti"], m["mu_star"] if m else None,
                                  m["sigma"] if m else None, threshold)
            rows.append({**r, "status": status,
                        "mu_star": m["mu_star"] if m else None, "sigma": m["sigma"] if m else None,
                        "ratio": r["S_Ti"] / (abs(r["S_i"]) + 1e-9)})

        critiques   = sorted([r for r in rows if r["status"] == "Critique"], key=lambda r: -r["S_Ti"])
        redondants  = [r for r in rows if r["status"] == "Redondant"]
        interactifs = sorted([r for r in rows if r["status"] == "Interactif"], key=lambda r: -r["ratio"])

        lines = ["# Rapport de décision — Analyse de sensibilité hiérarchique (Étape 2)", ""]
        lines.append(f"SALib version : {sobol_payload['metadata']['salib_version']} | "
                    f"N Sobol = {sobol_payload['metadata']['N']} | "
                    f"d_effective = {sobol_payload['metadata']['d_effective']}")
        lines.append("")
        lines.append("> Les indices ci-dessous mesurent la part de variance de la fitness "
                    "expliquée dans la distribution explorée : ce ne sont PAS des preuves "
                    "causales. Pour les hyperparamètres conditionnels, la sensibilité "
                    "dépend du contexte de base (P_active < 1).")
        lines.append("")

        lines.append("## 1. Hyperparamètres critiques (S_Ti > 0.10)\n")
        lines.append("| Hyperparamètre | S_i | S_Ti | S_Ti/S_i | Interprétation |")
        lines.append("|---|---|---|---|---|")
        for r in critiques:
            interp = ("conditionnel (P_active={:.2f}, contexte={})".format(r["P_active"], r["base_context"])
                      if r["base_context"] else "global")
            lines.append(f"| {r['param_name']} | {r['S_i']:.4f} | {r['S_Ti']:.4f} | "
                        f"{r['ratio']:.2f} | {interp} |")
        if not critiques:
            lines.append("| — aucun — | | | | |")
        lines.append("")

        lines.append("## 2. Hyperparamètres redondants (S_Ti < 0.02, faible importance Morris)\n")
        lines.append("| Hyperparamètre | S_Ti | Valeur fixe recommandée | Justification |")
        lines.append("|---|---|---|---|")
        for r in redondants:
            fixed_val = self._fixed_value_for(r["param_name"])
            lines.append(f"| {r['param_name']} | {r['S_Ti']:.4f} | {fixed_val} | "
                        f"S_Ti et effet élémentaire de Morris (mu*, sigma) tous deux "
                        f"sous le seuil 0.05·σ(fitness) |")
        if not redondants:
            lines.append("| — aucun — | | | |")
        lines.append("")

        lines.append("## 3. Hyperparamètres interactifs (S_Ti/S_i > 2)\n")
        lines.append("| Hyperparamètre | S_i | S_Ti | Ratio | S_Ti - S_i |")
        lines.append("|---|---|---|---|---|")
        for r in interactifs:
            lines.append(f"| {r['param_name']} | {r['S_i']:.4f} | {r['S_Ti']:.4f} | "
                        f"{r['ratio']:.2f} | {r['S_Ti'] - r['S_i']:.4f} |")
        if interactifs:
            lines.append("")
            lines.append(f"> calc_second_order={sobol_payload['metadata']['calc_second_order']} : "
                        "S_Ti - S_i mesure une interaction AGRÉGÉE avec le reste de l'espace, "
                        "pas une paire d'hyperparamètres identifiée. Aucune paire spécifique "
                        "n'est affirmée sans indices d'ordre 2 (S2).")
        else:
            lines.append("| — aucun — | | | | |")
        lines.append("")

        lines.append("## 4. Configuration finale retenue\n")
        if self.best_config:
            flat = self._flatten_individual(self.best_config)
            lines.append("| Hyperparamètre | Valeur | S_i | S_Ti | Ratio | Statut |")
            lines.append("|---|---|---|---|---|---|")
            by_name = {r["param_name"]: r for r in rows}
            for name, val in flat.items():
                r = by_name.get(name)
                if r is None:
                    continue
                lines.append(f"| {name} | {val} | {r['S_i']:.4f} | {r['S_Ti']:.4f} | "
                            f"{r['ratio']:.2f} | {r['status']} |")
            lines.append("")
            lines.append("La configuration retenue privilégie les valeurs des hyperparamètres "
                        "critiques identifiées ci-dessus ; les hyperparamètres redondants sont "
                        "fixés à leur valeur par défaut pour réduire l'espace de recherche des "
                        "étapes suivantes.")
        else:
            lines.append("Aucune configuration de référence chargée (search_results_path).")

        report = "\n".join(lines)
        with open(os.path.join(self.scratch_dir, "sensitivity_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        self._decision_rows = rows
        return report

    def _fixed_value_for(self, param_name: str):
        parts = param_name.split("::")
        if len(parts) == 1:
            meta = self.parameter_metadata[param_name]
            domain = meta.get("domain_used") or meta.get("domain")
            return domain[len(domain) // 2] if domain else None
        gate, base, pname = parts
        spec = BASE_PARAM_SPACE[base]
        domain = spec["params"][pname]
        if pname in spec["continuous"]:
            lo, hi = domain
            return (lo + hi) / 2
        return domain[len(domain) // 2]

    @staticmethod
    def _flatten_individual(individual: dict) -> dict:
        flat = {"Cell_Type": individual["Cell_Type"], "Input_Regime": individual["Input_Regime"]}
        for k in ("hidden_size", "lr", "lam", "mu1", "mu2", "batch_size", "W"):
            flat[k] = individual[k]
        for gate, cfg in individual.get("gates", {}).items():
            base = cfg["Base"]
            flat[f"Base_{gate}"] = base
            for pname, val in cfg.items():
                if pname == "Base":
                    continue
                flat[f"{gate}::{base}::{pname}"] = val
        return flat

    # ── Fichier canonique (section 14) ───────────────────────────────────────

    def write_sensitivity_rankings(self, sobol_payload: Optional[dict] = None,
                                   morris_payload: Optional[dict] = None) -> dict:
        sobol_payload = sobol_payload or getattr(self, "_sobol_results", None)
        morris_payload = morris_payload or getattr(self, "_morris_results", None)
        if sobol_payload is None:
            raise RuntimeError("run_sobol() doit être exécuté avant write_sensitivity_rankings().")

        morris_by_name = {r["param_name"]: r for r in morris_payload["results"]} if morris_payload else {}
        threshold = morris_payload["metadata"]["threshold"] if morris_payload else None

        rankings = []
        rows_sorted = sorted(sobol_payload["results"], key=lambda r: -r["S_Ti"])
        for rank, r in enumerate(rows_sorted, start=1):
            m = morris_by_name.get(r["param_name"])
            status = self._status(r["S_i"], r["S_Ti"], m["mu_star"] if m else None,
                                  m["sigma"] if m else None, threshold)
            rankings.append({
                "rank": rank, "param_name": r["param_name"], "gate": r["gate"],
                "base_context": r["base_context"],
                "S_i": r["S_i"], "S_Ti": r["S_Ti"],
                "S_Ti_conf_95": [r["S_Ti"] - (r["S_Ti_conf_95"] or 0.0),
                                 r["S_Ti"] + (r["S_Ti_conf_95"] or 0.0)] if r["S_Ti_conf_95"] is not None else None,
                "ratio_STi_Si": r["S_Ti"] / (abs(r["S_i"]) + 1e-9),
                "mu_star": m["mu_star"] if m else None, "sigma_morris": m["sigma"] if m else None,
                "status": status,
                "fixed_value": self._fixed_value_for(r["param_name"]) if status == "Redondant" else None,
            })

        n_critique   = sum(1 for r in rankings if r["status"] == "Critique")
        n_interactif = sum(1 for r in rankings if r["status"] == "Interactif")
        n_redondant  = sum(1 for r in rankings if r["status"] == "Redondant")
        n_neutre     = sum(1 for r in rankings if r["status"] == "Neutre")
        top3 = sorted(
            [(r["param_name"], r["S_Ti"] - r["S_i"]) for r in rankings if r["S_Ti"] - r["S_i"] > 0],
            key=lambda t: -t[1])[:3]
        top3_interactions = [[name, "(interaction agrégée, non identifiée par paire)", delta]
                             for name, delta in top3]

        payload = {
            "metadata": {
                "etape": 2,
                "method_morris": (morris_payload["metadata"] if morris_payload
                                  else {"r": None, "num_levels": None, "seed": None}),
                "method_sobol": sobol_payload["metadata"],
                "d_effective": sobol_payload["metadata"]["d_effective"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "rankings": rankings,
            "summary": {"n_critique": n_critique, "n_interactif": n_interactif,
                       "n_redondant": n_redondant, "n_neutre": n_neutre,
                       "top3_interactions": top3_interactions},
        }
        with open(os.path.join(self.scratch_dir, "sensitivity_rankings.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return payload


# ══════════════════════════════════════════════════════════════════════════
# Support ProcessPoolExecutor (section 15) — fonctions au niveau module,
# picklables ; l'initializer place l'analyzer/les DataFrames une seule fois
# par worker plutôt que de les sérialiser à chaque tâche.
# ══════════════════════════════════════════════════════════════════════════

_WORKER_ANALYZER: Optional[HierarchicalSensitivityAnalyzer] = None


def _init_worker(analyzer: HierarchicalSensitivityAnalyzer) -> None:
    global _WORKER_ANALYZER
    _WORKER_ANALYZER = analyzer


def _worker_evaluate(item) -> "tuple[int, float]":
    eval_id, x_row = item
    individual = _WORKER_ANALYZER.decode_config(x_row)
    fitness = _WORKER_ANALYZER._evaluate_individual(individual, eval_id)
    return eval_id, fitness
