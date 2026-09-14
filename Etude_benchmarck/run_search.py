"""
run_search.py  Script exécutable (pas un notebook) qui lance la recherche
heuristique étendue (ExtendedHeuristicSearch, Étape 2) sur les données réelles
du projet, avec reprise automatique sur interruption.

Sources de données (branchées sur le workflow réel, cf. train.ipynb) :
    Train : MOMTSIM/config/featuresLog.parquet   (~5,5 M tx, pool d'entraînement)
    Val   : data/val_features.parquet            (150 K clients, seed 1001, ~1,65 M tx)
    Test  : data/test_features.parquet           (150 K clients, seed 1002, ~1,65 M tx —
                                                    chargé mais PAS utilisé par la fitness ;
                                                    réservé à une évaluation finale hors
                                                    recherche, comme dans train.ipynb)

Préprocessing : compute_fitness() (extended_heuristic_search.py) construit déjà
lui-même les régimes "raw"/"engineered" par appel à niveau1_harness.
build_regime_frame + standardize + build_windows pour CHAQUE individu (le
régime dépend de Input_Regime, un hyperparamètre structurel recherché — il ne
peut donc pas être pré-calculé une fois pour toutes en dehors de la fitness).
Ce pipeline est une simplification DÉLIBÉRÉE et documentée (niveau1_harness.py)
du pipeline de train.ipynb (TopologyValidator + test de Kolmogorov-Smirnov par
colonne) : train.ipynb ne traite que le régime "engineered" à 12 features avec
un modèle d'architecture FIXE, alors que l'Étape 2 doit comparer "raw" et
"engineered" pour une architecture VARIABLE (recherchée) — les deux pipelines
coexistent pour des besoins différents, aucun des deux n'est remplacé ici.

Device d'entraînement : sélection automatique MPS (Apple Silicon, ex. M4) >
CUDA > CPU via select_training_device() — SANS AUCUN effet sur la latence de
fitness, qui reste toujours mesurée sur CPU/batch=256 (contrainte USSD de
production, indépendante de la machine de développement).

Pénalité de latence : DÉSACTIVÉE PAR DÉFAUT dans ce script (--enable-latency-
penalty pour la réactiver)  La latence
reste mesurée et journalisée (CPU, batch=256, seuil 100 ms) pour audit ; elle
ne soustrait simplement rien à la fitness par défaut ICI. Le protocole complet
du mémoire (pénalité activée) reste reproductible via --enable-latency-penalty.

Reprise / checkpointing : --resume (activé par défaut) recharge automatiquement
search_checkpoint.json s'il existe et continue exactement où la recherche
précédente s'est arrêtée (population, scores, historique, meilleur individu,
taux de mutation, état RNG) — Ctrl+C puis relancer la même commande reprend le
run sans perte. Tous les artefacts (extended_search_log.jsonl,
heuristic_best_config.json, best_candidate.pt, fitness_convergence_extended.png)
sont réécrits à chaque génération, donc consultables/graphables pendant que le
run est en cours ou après une interruption.

Usage :
    python run_search.py
    python run_search.py --n_generations 20 --population_size 16 --device mps
    python run_search.py --enable-latency-penalty      # protocole intégral du mémoire
    python run_search.py --no-resume                   # forcer un nouveau run
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_heuristic_search import ExtendedHeuristicSearch, select_training_device  # noqa: E402

_MODELISATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_TRAIN_PATH = os.path.join(_MODELISATION_ROOT, "MOMTSIM", "config", "featuresLog.parquet")
DEFAULT_VAL_PATH   = os.path.join(_MODELISATION_ROOT, "data", "val_features.parquet")
DEFAULT_TEST_PATH  = os.path.join(_MODELISATION_ROOT, "data", "test_features.parquet")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recherche heuristique étendue MKAN (Étape 2).")
    p.add_argument("--train_path", default=DEFAULT_TRAIN_PATH)
    p.add_argument("--val_path", default=DEFAULT_VAL_PATH)
    p.add_argument("--test_path", default=DEFAULT_TEST_PATH)
    p.add_argument("--scratch_dir", default=None,
                   help="Défaut : /workspace/scratch/ si accessible, sinon "
                        "MKAN/checkpoints/extended_search/ (cf. ExtendedHeuristicSearch).")
    p.add_argument("--device", default=None,
                   help="Device d'entraînement. Défaut : détection auto MPS > CUDA > CPU "
                        "(sans effet sur la mesure de latence, toujours CPU).")
    p.add_argument("--population_size", type=int, default=12)
    p.add_argument("--elite_size", type=int, default=2)
    p.add_argument("--n_generations", type=int, default=15)
    p.add_argument("--tournament_size", type=int, default=3)
    p.add_argument("--mutation_rate", type=float, default=0.35)
    p.add_argument("--crossover_rate", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--n_windows_eval", type=int, default=50_000,
                   help="Protocole Étape 2 : 50 000 fenêtres train ET val par évaluation "
                        "(ne pas modifier sans raison méthodologique).")
    p.add_argument("--n_epochs_eval", type=int, default=10,
                   help="Protocole Étape 2 : 10 époques par évaluation fitness.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-latency-penalty", dest="enable_latency_penalty",
                   action="store_true", default=False,
                   help="Réactive la pénalité de latence dans la fitness (protocole "
                        "intégral du mémoire, seuil 100 ms). Désactivée par défaut dans "
                        "ce script (recherche sur GPU M4, hors contrainte USSD CPU de "
                        "production) ; la latence reste toujours mesurée/journalisée.")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                   help="Ignore un éventuel checkpoint existant et repart de zéro.")
    p.add_argument("--skip-full-validation", dest="full_validation", action="store_false",
                   default=True, help="Ne pas ré-évaluer le meilleur modèle sur "
                                      "l'ensemble complet de data/val_features.parquet.")
    p.add_argument("--export", action="store_true", default=True,
                   help="Lance results_export.py (--format all) sur scratch_dir en fin de run.")
    p.add_argument("--no-export", dest="export", action="store_false")

    p.add_argument("--skip-sensitivity", dest="run_sensitivity", action="store_false",
                   default=True,
                   help="Ne pas lancer l'analyse de sensibilité (Sobol+Morris) après la "
                        "recherche. Activée par défaut — coûteuse (jusqu'à N*(d+2) + "
                        "r*(d+1) évaluations, d≈48 : potentiellement des dizaines de "
                        "milliers d'entraînements avec les valeurs par défaut).")
    p.add_argument("--sensitivity_N", type=int, default=1024,
                   help="Protocole Étape 2 : taille d'échantillon Sobol (défaut 1024).")
    p.add_argument("--sensitivity_r", type=int, default=10,
                   help="Protocole Étape 2 : trajectoires Morris (défaut 10).")
    p.add_argument("--sensitivity_n_epochs_eval", type=int, default=5,
                   help="Protocole de sensibilité (RÉDUIT, distinct du protocole de "
                        "recherche 50k/10 époques) : 5 époques par défaut.")
    p.add_argument("--sensitivity_n_windows_eval", type=int, default=10_000,
                   help="Protocole de sensibilité (RÉDUIT) : 10 000 fenêtres par défaut.")
    p.add_argument("--sensitivity_max_workers", type=int, default=2,
                   help="Processus parallèles pour Sobol/Morris (les workers de "
                        "sensitivity_analysis.py évaluent toujours sur CPU, jamais MPS/CUDA, "
                        "pour éviter la contention multi-process sur un device GPU).")
    return p


def _load_datasets(args) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    checks = [("train", args.train_path), ("val", args.val_path), ("test", args.test_path)]
    missing = [(label, os.path.abspath(path)) for label, path in checks if not os.path.exists(path)]
    if missing:
        lines = [f"  - {label} : {path}" for label, path in missing]
        raise FileNotFoundError(
            "Fichier(s) de données introuvable(s) :\n" + "\n".join(lines) +
            f"\n\nChemin racine déduit (2 niveaux au-dessus de run_search.py) : "
            f"{_MODELISATION_ROOT}\n"
            "Si ton arborescence locale diffère de MKAN/Etude_benchmarck/run_search.py, "
            "passe les chemins explicitement, ex. :\n"
            "  python run_search.py \\\n"
            "    --train_path \"/chemin/vers/MOMTSIM/config/featuresLog.parquet\" \\\n"
            "    --val_path   \"/chemin/vers/data/val_features.parquet\" \\\n"
            "    --test_path  \"/chemin/vers/data/test_features.parquet\""
        )

    t0 = time.time()
    print(f"Chargement train : {args.train_path}")
    df_train = pd.read_parquet(args.train_path)
    print(f"  {len(df_train):,} transactions, {df_train['nameOrig'].nunique():,} comptes "
         f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    print(f"Chargement val   : {args.val_path}")
    df_val = pd.read_parquet(args.val_path)
    print(f"  {len(df_val):,} transactions, {df_val['nameOrig'].nunique():,} comptes "
         f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    print(f"Chargement test  : {args.test_path}  (chargé, non utilisé par la fitness)")
    df_test = pd.read_parquet(args.test_path)
    print(f"  {len(df_test):,} transactions ({time.time() - t0:.1f}s)")

    return df_train, df_val, df_test


def _run_sensitivity_analysis(args, search, df_train, df_val) -> None:
    """
    Lance l'analyse de sensibilité hiérarchique (Sobol + Morris) sur la
    configuration trouvée par la recherche, à la suite de celle-ci.
    Protocole RÉDUIT et DISTINCT de la recherche (n_epochs_eval/n_windows_eval
    par défaut 5/10 000, jamais confondu avec 10/50 000) — cf. sensitivity_analysis.py.
    Ne fait jamais échouer le run principal : toute erreur est journalisée puis
    ignorée (la recherche et ses artefacts restent valides sans elle).
    """
    best_config_path = os.path.join(search.scratch_dir, "heuristic_best_config.json")
    if not os.path.exists(best_config_path):
        print("\n[avertissement] heuristic_best_config.json introuvable — "
             "analyse de sensibilité ignorée (aucun individu valide trouvé ?).")
        return

    from sensitivity_analysis import HierarchicalSensitivityAnalyzer

    print("\n" + "=" * 70)
    print("Analyse de sensibilité hiérarchique (Sobol + Morris)")
    print("=" * 70)

    analyzer = HierarchicalSensitivityAnalyzer(
        search_results_path=best_config_path, df_train=df_train, df_val=df_val,
        device="cpu",   # les workers évaluent toujours sur CPU (cf. --sensitivity_max_workers)
        n_epochs_eval=args.sensitivity_n_epochs_eval,
        n_windows_eval=args.sensitivity_n_windows_eval,
        max_workers=args.sensitivity_max_workers,
        scratch_dir=search.scratch_dir, seed=args.seed,
        # Même bascule que la recherche elle-même (attribut d'instance sérialisé
        # vers les workers ProcessPoolExecutor — fonctionne en séquentiel ET en
        # parallèle, contrairement à un monkeypatch de fonction module-level qui
        # ne serait pas vu par des processus enfants réimportant le module).
        enable_latency_penalty=args.enable_latency_penalty,
    )
    d = analyzer.dimension
    n_morris = args.sensitivity_r * (d + 1)
    n_sobol = args.sensitivity_N * (d + 2)
    print(f"Dimension effective de l'espace : {d}")
    print(f"Morris : r={args.sensitivity_r} trajectoires -> {n_morris:,} évaluations")
    print(f"Sobol  : N={args.sensitivity_N} -> {n_sobol:,} évaluations")
    print(f"Total estimé : {n_morris + n_sobol:,} entraînements "
         f"({args.sensitivity_n_epochs_eval} époques, "
         f"{args.sensitivity_n_windows_eval:,} fenêtres chacun, CPU, "
         f"{args.sensitivity_max_workers} processus en parallèle)")
    print(f"Reprise automatique en cas d'interruption via "
         f"{os.path.join(search.scratch_dir, 'sobol_evaluations.npy')}")

    try:
        morris_payload = analyzer.run_morris(r=args.sensitivity_r, seed=args.seed)
        sobol_payload = analyzer.run_sobol(N=args.sensitivity_N, calc_second_order=False,
                                           seed=args.seed)
        analyzer.write_sensitivity_rankings(sobol_payload, morris_payload)
        analyzer.generate_decision_report(sobol_payload, morris_payload)
        print(f"\nAnalyse de sensibilité terminée -> sensitivity_rankings.json, "
             f"sensitivity_report.md, morris_results.json, sobol_results.json "
             f"dans {search.scratch_dir}")
    except Exception as exc:   # noqa: BLE001 — n'invalide jamais la recherche elle-même
        print(f"  [avertissement] analyse de sensibilité échouée : {exc}")


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    device = select_training_device(prefer=args.device)
    print(f"Device d'entraînement : {device}  "
         f"(latence de fitness toujours mesurée sur CPU, indépendamment de ce choix)")

    df_train, df_val, df_test = _load_datasets(args)   # noqa: F841 (df_test conservé pour audit ultérieur)

    search = ExtendedHeuristicSearch(
        population_size=args.population_size, elite_size=args.elite_size,
        n_generations=args.n_generations, tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate, crossover_rate=args.crossover_rate,
        patience=args.patience, n_windows_eval=args.n_windows_eval,
        n_epochs_eval=args.n_epochs_eval, seed=args.seed,
        scratch_dir=args.scratch_dir,
    )
    print(f"Sorties écrites dans : {search.scratch_dir}")
    print(f"Pénalité de latence dans la fitness : "
         f"{'ACTIVÉE (protocole intégral)' if args.enable_latency_penalty else 'DÉSACTIVÉE (mesurée/journalisée quand même)'}")

    # enable_latency_penalty est un paramètre de compute_fitness (pas de ExtendedHeuristicSearch) ;
    # on le propage via une fermeture légère pour ne pas modifier la signature de fit()/_evaluer_un.
    import extended_heuristic_search as _ehs
    _original_compute_fitness = _ehs.compute_fitness

    def _compute_fitness_with_flag(config, df_tr, df_vl, dev, **kwargs):
        kwargs.setdefault("enable_latency_penalty", args.enable_latency_penalty)
        return _original_compute_fitness(config, df_tr, df_vl, dev, **kwargs)

    _ehs.compute_fitness = _compute_fitness_with_flag
    try:
        best = search.fit(df_train, df_val, device=device, resume_from_checkpoint=args.resume)
    finally:
        _ehs.compute_fitness = _original_compute_fitness

    print("\n" + "=" * 70)
    print(f"Meilleure configuration : {best['params']}")
    print(f"Score de fitness        : {best['score']:.4f}")
    print("=" * 70)

    if args.full_validation:
        print("\nValidation finale du meilleur modèle sur l'ensemble complet de "
             f"{args.val_path} (pas de sous-échantillonnage)...")
        result = search.evaluate_best_on_full_validation(df_train, df_val, device=device)
        if result:
            print(f"  MCC={result['MCC']:.4f}  PR_AUC={result['PR_AUC']:.4f}  "
                 f"Brier={result['Brier']:.4f}  AUC_ROC={result['AUC_ROC']:.4f}  "
                 f"n_windows={result['n_windows']:,}")

    try:
        search.export_top5()
        search.plot_convergence()
    except Exception as exc:   # noqa: BLE001 — ne jamais faire échouer le run pour de la restitution
        print(f"  [avertissement] export top5/plot_convergence : {exc}")

    if args.run_sensitivity:
        _run_sensitivity_analysis(args, search, df_train, df_val)

    if args.export:
        try:
            import results_export
            rc = results_export.main(["--output_dir", search.scratch_dir, "--format", "all"])
            print(f"results_export terminé (code {rc}) — voir {search.scratch_dir}")
        except Exception as exc:   # noqa: BLE001
            print(f"  [avertissement] results_export a échoué : {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
