"""
verifier_protocole.py -- Controle AVANT tout lancement de la chaine (13/11, apres l'ecart de latence de NSGA-II
et le Sobol a 51 200 entrainements).

1. Conformite : les parametres EFFECTIFS de la chaine (chaine_complete.sh, lus par le meme analyseur d'arguments que
   run_search.py, et config.py) sont compares aux decisions de memoire/DECISIONS.md :
     - latence : seuil 500 ms, penalite min(1, (L-T)/T) appliquee, contrainte de dominance au meme seuil (03/11) ;
     - graines du final : 1 a 5 ; prétraitement B pour la nouvelle Etape 4 (03/11).
2. Cout : pour chaque etape non finie (pas de sorties/chaine/<etape>.ok), nombre d'entrainements, heures et dollars
   estimes, a partir des durees reelles (runs.csv) et du prix horaire (cost_state.json), plus le cout de Sobol /
   Morris (N x (d + 2) evaluations, d = 48) au rythme mesure.
Sortie : un tableau et un code de retour (0 si tout est conforme, 1 sinon). Lecture seule.

Usage (sur l'instance ou en local, depuis campagne_aws/) :
    python verifier_protocole.py [--sorties sorties] [--rythme-eval-h 50] [--coeurs 64]
"""

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import config  # noqa: E402

DECISION = {"seuil_ms": 500.0, "plafonnee": True, "graines": [1, 2, 3, 4, 5]}
DIMENSION_SOBOL = 48                       # espace effectif de la recherche (journal du 12/11)
DUREE_PHASE = {"preproc": "final", "multkan": "final", "derive": "final"}   # reference de duree par etape
RUNS_PAR_ETAPE = {"preproc": 40, "multkan": 60}     # A et C x 20 ; K = 1, 2, 3 x 20


def valeurs_config(texte: str) -> str:
    """Remplace chaque "$(python -c 'import config; print(config.X)')" de la chaine par la valeur de config.X."""
    return re.sub(r"\"?\$\(python -c '[^']*config\.(\w+)\)'\)\"?", lambda m: str(getattr(config, m.group(1))), texte)


def args_recherche(chaine: str) -> list:
    """Arguments de run_search.py dans RECHERCHE=(...) de chaine_complete.sh, variables du script remplacees."""
    # La valeur lue dans config.py par la chaine ("$(python -c '...LATENCY_THRESHOLD_MS...')") est remplacee par
    # celle que la chaine obtiendra, avant toute autre lecture (ses parentheses fermeraient sinon le tableau).
    chaine = valeurs_config(chaine)
    m = re.search(r"^RECHERCHE=\((.*?)\)\s*$", chaine, re.S | re.M)
    if not m:
        return []
    brut = " ".join(l.split("#")[0] for l in m.group(1).splitlines())
    brut = brut.replace("$OUT", "sorties").replace('"', "")
    jetons = shlex.split(brut)
    return jetons[jetons.index("../run_search.py") + 1:] if "../run_search.py" in jetons else []


def analyser(jetons: list):
    sys.path.insert(0, os.path.dirname(ICI))
    try:
        from run_search import _build_arg_parser
    except Exception as exc:  # noqa: BLE001
        return None, f"run_search.py non importable ({exc!r})"
    try:
        return _build_arg_parser().parse_args(jetons), None
    except SystemExit:
        return None, "arguments refuses par run_search.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sorties", default=os.path.join(ICI, "sorties"))
    ap.add_argument("--rythme-eval-h", type=float, default=50.0,
                    help="evaluations de recherche par heure (Morris du 12-13/11 : 490 en ~9,5 h, 10 processus)")
    ap.add_argument("--coeurs", type=int, default=64)
    ap.add_argument("--rythme-nsga2-h", type=float, default=168 / 20.7,
                    help="individus NSGA-II par heure (recherche du 12/11 : 168 en 20,7 h, 10 processus)")
    a = ap.parse_args()
    chaine = open(os.path.join(ICI, "chaine_complete.sh"), encoding="utf-8").read()
    ok, lignes = True, []

    def controle(nom, condition, detail=""):
        nonlocal ok
        ok &= bool(condition)
        lignes.append(f"  [{'OK' if condition else 'ECART'}] {nom}{' : ' + detail if detail else ''}")

    # ── 1. Conformite ──
    controle("config.LATENCY_THRESHOLD_MS = 500", config.LATENCY_THRESHOLD_MS == DECISION["seuil_ms"],
             str(config.LATENCY_THRESHOLD_MS))
    controle("config.LATENCY_CAPPED", getattr(config, "LATENCY_CAPPED", False) is True)
    controle("graines du final 1 a 5", list(config.FINAL["seeds"]) == DECISION["graines"], str(config.FINAL["seeds"]))
    jetons = args_recherche(chaine)
    controle("RECHERCHE trouvee dans chaine_complete.sh", bool(jetons))
    r, err = analyser(jetons) if jetons else (None, "pas d'arguments")
    if r is None:
        controle("arguments de run_search.py", False, err)
    else:
        controle("recherche : penalite appliquee (--enable-latency-penalty)", r.enable_latency_penalty)
        controle("recherche : penalite plafonnee (--latency-penalty-capped)", r.latency_penalty_capped)
        controle("recherche : seuil = config.LATENCY_THRESHOLD_MS", r.latency_threshold_ms == config.LATENCY_THRESHOLD_MS,
                 f"{r.latency_threshold_ms} ms")
        controle("recherche : contrainte de dominance active", r.enforce_latency_constraint)
        controle("recherche : R2 de la fitness = config.R2_ALT (0,80)", r.r2_threshold == config.R2_ALT,
                 str(r.r2_threshold))
        controle("recherche : grille des bases = config.GRID_INIT (data)", r.grid_init == config.GRID_INIT,
                 r.grid_init)
        rech = re.search(r"^RECH=\(env (.*?)\"\$\{RECHERCHE", valeurs_config(chaine), re.S | re.M)
        fils = re.search(r"MKAN_LATENCY_THREADS=(\d+)", rech.group(1)) if rech else None
        controle("recherche : latence mesuree a config.LATENCY_THREADS fils",
                 fils is not None and int(fils.group(1)) == config.LATENCY_THREADS,
                 fils.group(1) if fils else "MKAN_LATENCY_THREADS absent")
        ligne_sobol = re.search(r"^\s*etape_plafonnee (sobol\w*) .*$", chaine, re.M)
        sobol_actif = ligne_sobol is not None
        n_N = re.search(r"--sensitivity_N (\d+)", ligne_sobol.group(0)) if ligne_sobol else None
        N = int(n_N.group(1)) if n_N else r.sensitivity_N
        n_sobol = N * (DIMENSION_SOBOL + 2)
        n_morris = r.sensitivity_r * (DIMENSION_SOBOL + 1)
        coeurs = re.search(r"MKAN_LATENCY_CORES=(\S+)", rech.group(1)) if rech else None
        controle("recherche : latence mesuree sur config.LATENCY_CORES (reserves)",
                 coeurs is not None and coeurs.group(1) == config.LATENCY_CORES,
                 coeurs.group(1) if coeurs else "MKAN_LATENCY_CORES absent")
        controle("chaine confinee a config.TRAIN_CORES (taskset)",
                 re.search(r"^taskset -a -p -c " + re.escape(config.TRAIN_CORES), valeurs_config(chaine), re.M)
                 is not None, config.TRAIN_CORES)

    # ── 2. Cout des etapes restantes ──
    prix = None
    try:
        cs = json.load(open(os.path.join(a.sorties, "cost_state.json"), encoding="utf-8"))
        prix, deja = cs.get("price_usd_h"), cs.get("total_usd")
        arret = cs.get("stop_at_usd")
    except (OSError, ValueError):
        deja = arret = None
    durees = {}
    try:
        with open(os.path.join(a.sorties, "runs.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("train_time_s"):
                    durees.setdefault(row["phase"], []).append(float(row["train_time_s"]) / 3600)
    except OSError:
        pass
    h_final = statistics.mean(durees["final"]) if durees.get("final") else None
    couts = []
    finies = {f[:-3] for f in os.listdir(os.path.join(a.sorties, "chaine"))} if os.path.isdir(
        os.path.join(a.sorties, "chaine")) else set()
    for etape, n in RUNS_PAR_ETAPE.items():
        if etape in finies or h_final is None:
            continue
        n -= sum(len(v) for k, v in durees.items() if k.startswith(f"exp_{etape}"))   # runs deja finis
        if n <= 0:
            continue
        threads = 2
        paralleles = max(1, min(20, a.coeurs // threads))
        h = n / paralleles * h_final          # vagues de runs de duree moyenne du final
        couts.append((etape, f"{n} entrainements", h))
    if r is not None:
        # File de recherche : NSGA-II refait (population x generations individus), puis Morris et Sobol, en serie.
        h_rech, quoi = 0.0, []
        if "nsga2_v2" not in finies and re.search(r"^\s*etape_plafonnee nsga2_v2", chaine, re.M):
            n_ind = r.population_size * r.n_generations
            h_rech += n_ind / a.rythme_nsga2_h
            quoi.append(f"NSGA-II {n_ind}")
        if sobol_actif and ligne_sobol.group(1) not in finies:
            h_rech += (n_morris + n_sobol) / a.rythme_eval_h
            quoi.append(f"Morris {n_morris} + Sobol {n_sobol} (N={N})")
        if quoi:
            couts.append(("recherche", " ; ".join(quoi), h_rech))
    lignes.append("")
    lignes.append(f"  Cout deja engage : {deja} $ ; arret de la sentinelle : {arret} $ ; prix : {prix} $/h")
    total_h = 0.0
    for etape, quoi, h in couts:
        total_h = max(total_h, h)
        lignes.append(f"  {etape:<10} {quoi:<52} ~{h:7.1f} h" + (f"  ~{h * prix:6.1f} $" if prix else ""))
    if couts and prix:
        reste = (arret - deja) if (arret and deja is not None) else None
        tient = reste is None or total_h * prix <= reste
        controle("budget : la plus longue file tient avant l'arret de la sentinelle", tient,
                 f"~{total_h:.0f} h soit ~{total_h * prix:.0f} $ pour {reste:.0f} $ restants" if reste else "")
    if r is not None and sobol_actif:
        controle("Sobol : moins de 5 000 evaluations (sinon, decision explicite de Miguel)", n_sobol < 5000,
                 f"{n_sobol} evaluations + Morris {n_morris}")
    print("Controle du protocole et du cout avant lancement")
    print("\n".join(lignes))
    print("CONFORME" if ok else "NON CONFORME : ne rien lancer sans accord explicite")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
