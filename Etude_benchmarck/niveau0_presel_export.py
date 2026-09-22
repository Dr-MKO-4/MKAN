"""
niveau0_presel_export.py -- Export presel_bases_level0.json (Etape 3, Niveau 0,
module 3.3).

Assemble le fichier officiel de preselection qui sert d'entree au Niveau 1
(Etape 4), a partir :
  - des resultats bruts de niveau0_benchmark_iso.py (module 3.1)
  - de l'analyse/matrice de qualification de niveau0_analysis.py (module 3.2,
    reimportee directement -- AUCUNE logique de decision n'est dupliquee ici,
    pour eviter toute divergence entre le rapport 3.2 et ce JSON officiel)
  - de heuristic_best_config.json / top5_configurations.json (Etape 2)

Le fichier produit est auto-documente : chaque base retenue ou rejetee porte
une justification textuelle explicite, et la coherence avec l'Etape 2 est
verifiee et rapportee (AVERTISSEMENT explicite en cas d'ecart, jamais silencieux).

Usage :
    python niveau0_presel_export.py
    python niveau0_presel_export.py --out_dir extended_search
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import niveau0_analysis as n0a


N_BASE_FAMILIES = 8    # families "top-niveau" du choix de base (Wav-KAN compte pour UNE
                        # famille, dog/morlet etant un hyperparametre interne, cf. Etape 2
                        # GATE_CANDIDATES et heuristic_best_config.json : base="wavkan",
                        # hyperparams={"Mother_Wavelet": ...}) -- coherent avec le chiffre
                        # deja etabli dans step_3_benchmark_synthetique.tex (4096 = 8^4).
N_GATES = 4             # superset TKANCell (Forget, Input, Candidate, Output) ; la
                        # GRUKANCell d'elite (Rang 1) n'a que 3 portes (pas d'Output).


def _family_of(base_label: str) -> str:
    if base_label.startswith("wavkan"):
        return "wavkan"
    return base_label


def _md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _retained_justification(gate: str, base: str, decision: dict, gate_info: dict) -> str:
    """Lit les seuils REELLEMENT utilises depuis gate_info (rempli par
    n0a.preselect_by_gate avec les valeurs effectivement appliquees pour ce run --
    jamais les constantes par defaut n0a.GIBBS_THRESHOLD/etc., qui seraient fausses
    si l'appelant a surcharge --gibbs_threshold/--latency_factor/--t90_max)."""
    parts = [
        f"RMSE moyen ({decision['rmse_avg']:.4g}) dans le 1er quartile des bases",
    ]
    if n0a.GATE_TO_REGIME[gate] == 1:
        parts.append(f"I_Gibbs <= {gate_info['gibbs_threshold']} sur les 3 familles du Regime 1")
    parts.append(f"latence ({decision['latency_us']:.4g} us) <= "
                 f"{gate_info['latency_factor']}x la plus rapide")
    t90_scope = ""
    if gate_info.get("t90_excluded_functions"):
        t90_scope = (f" (calcule sur {', '.join(gate_info['t90_functions'])} uniquement -- "
                     f"{', '.join(gate_info['t90_excluded_functions'])} exclue(s) du critere "
                     f"T_90% : mur de capacite empirique B=12, cf. T90_EXCLUDED_FUNCTIONS)")
    parts.append(f"T_90% median ({decision['t90_avg']:.0f}) < {gate_info['t90_max']} "
                 f"iterations{t90_scope}")
    justification = "Retenue : " + " ; ".join(parts) + "."
    elite = n0a.ETAPE2_ELITE_BY_GATE.get(gate)
    if elite == base:
        justification += " Coherente avec la base plebiscitee par l'Etape 2 sur cette porte."
    return justification


def _rejected_justification(base: str, decision: dict) -> str:
    return "Rejetee : " + " ; ".join(decision["reasons_fail"]) + "."


def build_presel_by_gate(preselection: dict) -> dict:
    presel = {}
    for gate, info in preselection.items():
        retained_bases = []
        rejected_bases = []
        for base, decision in sorted(info["bases"].items()):
            if decision["retained"]:
                retained_bases.append({
                    "base": base,
                    "justification": _retained_justification(gate, base, decision, info),
                })
            else:
                rejected_bases.append({
                    "base": base,
                    "justification": _rejected_justification(base, decision),
                })

        coherence_check = None
        elite = n0a.ETAPE2_ELITE_BY_GATE.get(gate)
        if elite is not None:
            retained_labels = {b["base"] for b in retained_bases}
            if elite in retained_labels:
                coherence_check = {
                    "etape2_elite_base": elite,
                    "status": "OK",
                    "explanation": (
                        f"La base plebiscitee par l'Etape 2 sur la porte {gate} ('{elite}') "
                        f"figure bien parmi les bases retenues par le Niveau 0."
                    ),
                }
            else:
                d = info["bases"].get(elite)
                reason = " ; ".join(d["reasons_fail"]) if d else "base absente du benchmark Niveau 0"
                coherence_check = {
                    "etape2_elite_base": elite,
                    "status": "AVERTISSEMENT",
                    "explanation": (
                        f"La base plebiscitee par l'Etape 2 sur la porte {gate} ('{elite}') "
                        f"n'est PAS retenue par la preselection Niveau 0. Raison(s) : {reason}"
                    ),
                }

        entry = {
            "retained_bases": retained_bases,
            "rejected_bases": rejected_bases,
            "coherence_check": coherence_check,
        }
        if gate == "Output":
            entry["applicable_to"] = (
                "TKANCell (Rang 2 des top5_configurations.json) uniquement -- "
                "la GRUKANCell d'elite (Rang 1, heuristic_best_config.json) n'a pas de porte Output."
            )
        presel[gate] = entry
    return presel


def compute_search_space_reduction(presel_by_gate: dict) -> dict:
    per_gate_family_counts = {}
    warnings = []
    for gate, entry in presel_by_gate.items():
        families = {_family_of(b["base"]) for b in entry["retained_bases"]}
        per_gate_family_counts[gate] = len(families)
        if len(families) == 0:
            warnings.append(f"Porte {gate} : AUCUNE base retenue (espace apres reduction = 0).")

    combinations_before = N_BASE_FAMILIES ** N_GATES
    combinations_after = 1
    for gate in ("Forget", "Input", "Candidate", "Output"):
        combinations_after *= max(per_gate_family_counts.get(gate, 0), 0)

    reduction_pct = (
        (1.0 - combinations_after / combinations_before) * 100.0
        if combinations_before > 0 else None
    )

    return {
        "n_base_families": N_BASE_FAMILIES,
        "n_gates": N_GATES,
        "per_gate_retained_family_count": per_gate_family_counts,
        "combinations_before": combinations_before,
        "combinations_after": combinations_after,
        "reduction_pct": reduction_pct,
        "n_niveau1_runs_avoided": combinations_before - combinations_after,
        "warnings": warnings,
        "note": (
            "combinations_before suppose l'espace superset a 4 portes de la TKANCell "
            "(la GRUKANCell d'elite n'a que 3 portes, donc un espace reel plus petit "
            "pour cette architecture). Wav-KAN compte pour UNE famille (dog/morlet est "
            "un hyperparametre interne a la famille, pas un choix de base separe au "
            "niveau de la recherche Etape 2)."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_path", type=str,
                         default=os.path.join("extended_search", "niveau0_benchmark_results.json"))
    parser.add_argument("--best_config_path", type=str,
                         default=os.path.join("extended_search", "heuristic_best_config.json"))
    parser.add_argument("--top5_path", type=str,
                         default=os.path.join("extended_search", "top5_configurations.json"))
    parser.add_argument("--out_dir", type=str, default="extended_search")
    parser.add_argument("--out_name", type=str, default="presel_bases_level0.json")
    parser.add_argument("--gibbs_threshold", type=float, default=n0a.GIBBS_THRESHOLD,
                         help=f"Doit correspondre au --gibbs_threshold utilise pour "
                              f"niveau0_analysis.py (defaut = {n0a.GIBBS_THRESHOLD}) -- une "
                              f"incoherence entre les deux produirait un presel_bases_level0.json "
                              f"qui ne correspond pas au rapport de synthese.")
    parser.add_argument("--latency_factor", type=float, default=n0a.CRITERIA_LATENCY_FACTOR)
    parser.add_argument("--t90_max", type=int, default=n0a.CRITERIA_T90_MAX)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(project_root, args.raw_path)
    best_config_path = os.path.join(project_root, args.best_config_path)
    top5_path = os.path.join(project_root, args.top5_path)
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)

    raw = n0a.load_raw_results(raw_path)
    df = n0a.build_records_df(raw)
    agg = n0a.aggregate(df, raw["metadata"]["n_iterations"])
    _, _, gibbs_violation_bases = n0a.rank_and_grade(agg, gibbs_threshold=args.gibbs_threshold)
    preselection = n0a.preselect_by_gate(
        agg, gibbs_violation_bases, gibbs_threshold=args.gibbs_threshold,
        latency_factor=args.latency_factor, t90_max=args.t90_max)
    presel_by_gate = build_presel_by_gate(preselection)
    reduction = compute_search_space_reduction(presel_by_gate)

    best_config = _load_json(best_config_path)
    fitness = best_config["fitness_components"]
    etape2_link = {
        "heuristic_best_config_path": best_config_path,
        "cell_type": best_config["theta_struct"]["cell_type"],
        "MCC_raw": fitness["MCC_raw"],
        "fitness_total": fitness["fitness_total"],
        "gates": {g: v["base"] for g, v in best_config["theta_struct"]["gates"].items()},
    }
    if os.path.exists(top5_path):
        top5 = _load_json(top5_path)
        rank2 = next((c for c in top5 if c.get("rank") == 2), None)
        if rank2 is not None:
            etape2_link["top5_rank2_tkan"] = {
                "path": top5_path,
                "cell_type": rank2["Cell_Type"],
                "Base_Forget": rank2["Base_Forget"],
                "Base_Input": rank2["Base_Input"],
                "Base_Candidate": rank2["Base_Candidate"],
                "Base_Output": rank2["Base_Output"],
                "latency_ms": rank2["latency_ms"],
                "MCC_raw": rank2["MCC_raw"],
            }

    output = {
        "metadata": {
            "etape": 3,
            "level": 0,
            "source_files": {
                "raw_results": raw_path,
                "heuristic_best_config": best_config_path,
                "top5_configurations": top5_path if os.path.exists(top5_path) else None,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "B_iso": raw["metadata"]["B_iso"],
            "n_seeds": len(raw["metadata"]["seeds"]),
            "raw_results_md5": _md5_of_file(raw_path),
            "etape2_link": etape2_link,
            "preselection_thresholds": {
                "gibbs_threshold": args.gibbs_threshold,
                "latency_factor": args.latency_factor,
                "t90_max": args.t90_max,
                "is_protocol_default": (
                    args.gibbs_threshold == n0a.GIBBS_THRESHOLD
                    and args.latency_factor == n0a.CRITERIA_LATENCY_FACTOR
                    and args.t90_max == n0a.CRITERIA_T90_MAX
                ),
                "protocol_defaults": {
                    "gibbs_threshold": n0a.GIBBS_THRESHOLD,
                    "latency_factor": n0a.CRITERIA_LATENCY_FACTOR,
                    "t90_max": n0a.CRITERIA_T90_MAX,
                },
                "t90_excluded_functions": sorted(n0a.T90_EXCLUDED_FUNCTIONS),
                "t90_exclusion_justification": (
                    "fc1 (sin(400*pi*x), 200 oscillations) et fc3 (composante a 40*pi) "
                    "sont exclues du critere T_90% pour toutes les portes : constat "
                    "empirique (pas une supposition) -- a n_iterations=10000 (5x le budget "
                    "initial de 2000), les 45 essais (9 bases x 5 graines) sur fc1 sont "
                    "TOUS restes a RMSE ~= 0.7067 (= RMS(sin(400*pi*x)), solution triviale "
                    "'predire ~0 partout') avec t90_median exactement au plafond de censure "
                    "(zero progression, meme partielle). fc3 montre le meme mur pour 8 bases "
                    "sur 9. C'est un mur de CAPACITE (B=12 parametres insuffisants pour "
                    "representer un contenu a si haute frequence -- argument de type "
                    "Nyquist/echantillonnage), pas un probleme de vitesse de convergence : "
                    "aucun nombre d'iterations supplementaires ne change ce resultat. "
                    "RMSE/Gibbs/latence restent evalues normalement sur fc1/fc3."
                ),
            },
        },
        "presel_by_gate": presel_by_gate,
        "search_space_reduction": reduction,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"presel_bases_level0.json ecrit : {out_path}")
    for w in reduction["warnings"]:
        print(f"[avertissement] {w}")
    return output


if __name__ == "__main__":
    main()
