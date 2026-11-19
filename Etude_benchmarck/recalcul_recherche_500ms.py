"""
recalcul_recherche_500ms.py -- Recherche NSGA-II de la campagne (13/11) relue selon la decision du 03/11, sans
reentrainement.

La recherche lancee par la chaine (run_search.py) a garde la contrainte de dominance a 100 ms
(LATENCY_CONSTRAINT_MS) et n'a applique aucune penalite de latence (--enable-latency-penalty absent). La decision du
2026-11-03 (memoire/DECISIONS.md) fixe : seuil T = 500 ms, penalite min(1, (L - T)/T) soustraite avec le poids AHP
0,0876, contrainte de dominance a 500 ms. Chaque individu evalue a sa latence et ses composantes journalisees
(extended_search_log.jsonl) : on recalcule ici, sur les memes individus,
  * la fitness selon la decision : 0,4147 max(0, MCC) + 0,2573 PR-AUC - 0,1529 Brier + 0,0876 R2 - 0,0876 pen(L) ;
  * les fronts de dominance contrainte (Deb et al. 2002, par. II.B) avec T = 500 ms, sur les 4 objectifs de
    extended_heuristic_search._objective_vector (MCC ecrete, PR-AUC, -Brier, R2 symbolique).
Limite (a declarer) : la trajectoire de la recherche (selection, tournois, elitisme) a ete guidee par la contrainte
a 100 ms ; seul le classement des individus deja evalues est corrige, pas les individus que la recherche aurait
explores autrement.
Autres limites (relecture de la session memoire, 13/11), non corrigeables sans reentrainement : la fitness journalisee
repose sur le R2 symbolique au seuil 0,99 (et non 0,80), sur la grille des bases [-1, 1] (non recalee) et sur une
latence pre-mesuree a 1 fil (et non 4). Ce recalcul n'est donc pas sur l'echelle du plan et du final : il corrige
seulement la contrainte et la penalite de latence.

Usage :
    python recalcul_recherche_500ms.py --dossier campagne_aws/resultats_instance/campagne_2026-11-13/recherche
Sorties (dans --dossier) : recalcul_500ms.json, recalcul_500ms.md
"""

import argparse
import json
import os

# Poids AHP (vecteur propre non arrondi) et seuils : memes valeurs que extended_heuristic_search.py et
# campagne_aws/config.py (LATENCY_THRESHOLD_MS = 500.0). Recopies ici pour ne pas importer torch/sklearn.
W_MCC, W_PR, W_BRIER, W_R2, W_LAT = 0.4147, 0.2573, 0.1529, 0.0876, 0.0876
T_ANCIEN, T_DECISION = 100.0, 500.0


def penalite(l_ms: float, t: float = T_DECISION) -> float:
    """Decision du 03/11 : min(1, (L - T)/T) au-dela du seuil, 0 en dessous."""
    return 0.0 if l_ms <= t else min(1.0, (l_ms - t) / t)


def fitness(c: dict, pen: float) -> float:
    return (W_MCC * c["MCC_clipped"] + W_PR * c["PR_AUC"] - W_BRIER * c["Brier"]
            + W_R2 * c["R2_symbolic"] - W_LAT * pen)


def objectifs(c: dict) -> tuple:
    return (c["MCC_clipped"], c["PR_AUC"], -c["Brier"], c["R2_symbolic"])


def domine(a, cva, b, cvb) -> bool:
    """Dominance contrainte (Deb et al. 2002) : realisable > irrealisable ; deux irrealisables : moindre violation ;
    deux realisables : Pareto (>= partout, > quelque part)."""
    if cva == 0 and cvb > 0:
        return True
    if cva > 0 and cvb > 0:
        return cva < cvb
    if cva > 0:
        return False
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def fronts(ind: list, t: float) -> list:
    """Rang de front (1 = non domine) de chaque individu, contrainte de latence a t ms."""
    obj = [objectifs(i["c"]) for i in ind]
    cv = [max(0.0, i["c"]["latency_ms"] - t) for i in ind]
    n = len(ind)
    domines_par = [0] * n
    domine_liste = [[] for _ in range(n)]
    for p in range(n):
        for q in range(n):
            if p != q and domine(obj[p], cv[p], obj[q], cv[q]):
                domine_liste[p].append(q)
            elif p != q and domine(obj[q], cv[q], obj[p], cv[p]):
                domines_par[p] += 1
    rang, courant, r = [0] * n, [p for p in range(n) if domines_par[p] == 0], 1
    while courant:
        suivant = []
        for p in courant:
            rang[p] = r
            for q in domine_liste[p]:
                domines_par[q] -= 1
                if domines_par[q] == 0:
                    suivant.append(q)
        courant, r = suivant, r + 1
    return rang


def resume_config(cfg: dict) -> str:
    g = cfg.get("gates", {})
    bases = "/".join(str(g[k].get("Base")) for k in g)
    return (f"{cfg.get('Cell_Type')} {cfg.get('Input_Regime')} h={cfg.get('hidden_size')} W={cfg.get('W')} "
            f"bases={bases}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True, help="dossier contenant extended_search_log.jsonl")
    a = ap.parse_args()
    ind = []
    with open(os.path.join(a.dossier, "extended_search_log.jsonl"), encoding="utf-8") as f:
        for ligne in f:
            d = json.loads(ligne)
            c = d.get("fitness_components")
            if d.get("error") or not c or c.get("latency_ms") is None:
                continue
            ind.append({"gen": d["generation"], "id": d["individual_id"], "cfg": d["config"], "c": c,
                        "fitness_journal": d["fitness"]})

    # Controle : sans penalite, la formule redonne la fitness journalisee (penalite non appliquee pendant le run).
    ecart = max(abs(fitness(i["c"], 0.0) - i["fitness_journal"]) for i in ind)
    for i in ind:
        i["pen_decision"] = penalite(i["c"]["latency_ms"])
        i["fitness_decision"] = fitness(i["c"], i["pen_decision"])
    r100, r500 = fronts(ind, T_ANCIEN), fronts(ind, T_DECISION)
    for i, x, y in zip(ind, r100, r500):
        i["front_100ms"], i["front_500ms"] = x, y

    realisables = [i for i in ind if i["c"]["latency_ms"] <= T_DECISION]
    classement = sorted(realisables, key=lambda i: -i["fitness_decision"])
    front1 = sorted([i for i in ind if i["front_500ms"] == 1], key=lambda i: -i["fitness_decision"])
    lignes = lambda L: [{  # noqa: E731
        "generation": i["gen"], "individu": i["id"], "config": resume_config(i["cfg"]),
        "latence_ms": round(i["c"]["latency_ms"], 1), "MCC": round(i["c"]["MCC_clipped"], 4),
        "PR_AUC": round(i["c"]["PR_AUC"], 4), "Brier": round(i["c"]["Brier"], 4),
        "R2_symbolique": round(i["c"]["R2_symbolic"], 4), "penalite_decision": round(i["pen_decision"], 4),
        "fitness_journal_sans_penalite": round(i["fitness_journal"], 4),
        "fitness_decision": round(i["fitness_decision"], 4),
        "front_100ms": i["front_100ms"], "front_500ms": i["front_500ms"]} for i in L]

    sortie = {
        "source": "extended_search_log.jsonl", "n_individus": len(ind),
        "controle_formule_ecart_max": ecart,
        "n_entre_100_et_500ms": sum(T_ANCIEN < i["c"]["latency_ms"] <= T_DECISION for i in ind),
        "n_au_dela_500ms": sum(i["c"]["latency_ms"] > T_DECISION for i in ind),
        "front1_100ms": sum(r == 1 for r in r100), "front1_500ms": sum(r == 1 for r in r500),
        "entrent_dans_front1_a_500ms": sum(i["front_500ms"] == 1 and i["front_100ms"] != 1 for i in ind),
        "sortent_du_front1_a_500ms": sum(i["front_100ms"] == 1 and i["front_500ms"] != 1 for i in ind),
        "front1_500ms_detail": lignes(front1),
        "top10_fitness_decision_realisables": lignes(classement[:10]),
        "limite": "trajectoire de la recherche guidee par la contrainte a 100 ms ; seul le classement des "
                  "individus evalues est corrige",
    }
    with open(os.path.join(a.dossier, "recalcul_500ms.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=1)

    md = ["# Recherche NSGA-II relue selon la décision du 03/11 (sans réentraînement)", "",
          f"Individus évalués : {len(ind)}. Contrôle : la formule redonne la fitness journalisée à {ecart:.1e} près.",
          f"Latence entre 100 et 500 ms : {sortie['n_entre_100_et_500ms']} individus ; au-delà de 500 ms : "
          f"{sortie['n_au_dela_500ms']}.",
          f"Front 1 : {sortie['front1_100ms']} individus à 100 ms, {sortie['front1_500ms']} à 500 ms "
          f"({sortie['entrent_dans_front1_a_500ms']} entrent, {sortie['sortent_du_front1_a_500ms']} sortent).", "",
          "Limite : la trajectoire de la recherche a été guidée par la contrainte à 100 ms ; seul le classement des "
          "individus déjà évalués est corrigé. La fitness journalisée repose aussi sur le R² au seuil 0,99 (et non "
          "0,80), la grille des bases [-1, 1] (non recalée) et une latence pré-mesurée à 1 fil (et non 4) : ce "
          "recalcul n'est pas sur l'échelle du plan et du final.", "",
          "## Front 1 à 500 ms (classé par fitness de la décision)", "",
          "| gén. | ind. | configuration | latence (ms) | MCC | PR-AUC | Brier | R² | pénalité | fitness | front à 100 ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sortie["front1_500ms_detail"]:
        md.append(f"| {r['generation']} | {r['individu']} | {r['config']} | {r['latence_ms']} | {r['MCC']} | "
                  f"{r['PR_AUC']} | {r['Brier']} | {r['R2_symbolique']} | {r['penalite_decision']} | "
                  f"{r['fitness_decision']} | {r['front_100ms']} |")
    md += ["", "## 10 meilleures fitness de la décision (individus sous 500 ms)", "",
           "| gén. | ind. | configuration | latence (ms) | fitness | front à 500 ms | front à 100 ms |",
           "|---|---|---|---|---|---|---|"]
    for r in sortie["top10_fitness_decision_realisables"]:
        md.append(f"| {r['generation']} | {r['individu']} | {r['config']} | {r['latence_ms']} | "
                  f"{r['fitness_decision']} | {r['front_500ms']} | {r['front_100ms']} |")
    with open(os.path.join(a.dossier, "recalcul_500ms.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
