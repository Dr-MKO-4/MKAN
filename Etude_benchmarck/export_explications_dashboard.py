"""
export_explications_dashboard.py -- Exporte, pour chaque modèle du tableau de bord, ce que les analyses du mémoire ont
mesuré sur ce modèle précis (validation seulement), afin que le rapport COBAC et l'aide affichent les mêmes chiffres que
le mémoire :
  - métriques de validation du run (MCC, PR-AUC, Brier, score perf, R² symbolique norme L1 et attribution, latence AWS) ;
  - fidélité des formules (fidelite_formule.py) : arêtes actives, part symbolisable, accord des décisions après élagage,
    après remplacement des arêtes par leur formule, et pour les formules seules ;
  - les six arêtes les plus fortes partant d'une variable d'entrée, avec leur meilleure formule et son R² ;
  - rappel par scénario et précision attendue à 0,1 % et 1 % de fraude (analyse_scenarios_prevalence.py).

Sortie : MKAN/checkpoints/explications_tableau_de_bord.json (lu par dashboard/callbacks_extra.py).
Usage (depuis MKAN/Etude_benchmarck) : python export_explications_dashboard.py
"""
import json
import os

import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ICI, "..", "checkpoints")
FINAL = os.path.join(ICI, "campagne_aws", "resultats_instance", "final_2026-11-08")
CAMP = os.path.join(ICI, "campagne_aws", "resultats_instance", "campagne_2026-11-13")
V2 = os.path.join(ICI, "campagne_aws", "resultats_instance", "campagne_v2")
ANA = os.path.join(ICI, "analyse_banc", "final_2026-11-08")
VARIANTES = {
    "reseau": "réseau complet",
    "elagage": "élagage, L1 > 0,01",
    "formules_dans_domaine_080": "arêtes à R² ≥ 0,80 remplacées sur [-2, 2] seulement, apprises au-delà",
    "formules_080": "arêtes à R² ≥ 0,80 remplacées par leur formule, les autres apprises",
    "formules_seules_080": "formule seule : arêtes à R² ≥ 0,80, les autres supprimées",
    "aretes_40": "40 arêtes les plus fortes",
}


def main():
    cat = json.load(open(os.path.join(CKPT, "modeles_tableau_de_bord.json"), encoding="utf-8"))
    # campagne_v2 contient en plus la GRU FasterKAN/FasterKAN/DoG (modèle retenu le 18/11), entraînée après le final
    runs = pd.concat([pd.read_csv(os.path.join(FINAL, "runs.csv")), pd.read_csv(os.path.join(V2, "runs.csv"))])
    runs = runs.drop_duplicates("run_id").set_index("run_id")
    att = pd.read_csv(os.path.join(CAMP, "r2_attribution.csv")).set_index("run_id")
    fid = pd.read_csv(os.path.join(ANA, "fidelite_formule.csv"))
    are = pd.read_csv(os.path.join(ANA, "fidelite_aretes.csv"))
    sce = pd.read_csv(os.path.join(ANA, "scenarios_par_run.csv")).set_index("run_id")
    out = {}
    for m in cat["modeles"]:
        rid = m.get("run_id")
        if not rid or rid not in runs.index:
            continue
        r = runs.loc[rid]
        label = r["label"].rsplit(" s", 1)[0]
        e = {"run_id": rid, "label": label, "graine": int(r["seed"]),
             "validation": {"mcc": float(r["mcc"]), "auc_pr": float(r["auc_pr"]), "brier": float(r["brier"]),
                            "perf": float(0.4147 * max(0, r["mcc"]) + 0.2573 * r["auc_pr"] - 0.1529 * r["brier"]),
                            "r2_l1_080": float(r["r2_symbolic_0.80"]), "r2_l1_099": float(r["r2_symbolic"]),
                            "r2_attribution_080": float(att.loc[rid, "r2_attribution_0.80"]) if rid in att.index else None,
                            "latence_aws_ms": float(r["latency_ms"])}}
        f = fid[fid.modele == label]
        if len(f):
            acc = {}
            for cle, var in VARIANTES.items():
                v = f[f.variante == var]
                if len(v):
                    acc[cle] = {"aretes": int(v.aretes_actives.iloc[0]), "accord": float(v.accord_decisions.iloc[0]),
                                "mcc": float(v.mcc.iloc[0])}
            a = are[are.modele == label]
            top = a[~a.entree.str.startswith("h[")].nlargest(6, "l1")
            e["fidelite"] = {"aretes_total": int(f.aretes_total.iloc[0]), "aretes_actives": int(len(a)),
                             "part_symb_080": float((a.r2 >= 0.80).mean()), "part_symb_099": float((a.r2 >= 0.99).mean()),
                             "variantes": acc, "n_fenetres": 20000, "graine_analysee": int(f.seed.iloc[0]),
                             "aretes_fortes": [{"porte": t.porte, "variable": t.entree, "l1": float(t.l1), "famille": t.famille,
                                                "a": float(t.a), "b": float(t.b), "c": float(t.c), "d": float(t.d),
                                                "r2": float(t.r2)} for t in top.itertuples()]}
        if rid in sce.index:
            s = sce.loc[rid]
            e["scenarios"] = {k: float(s[f"rappel_{k}"]) for k in ("ATO", "SMURFING", "SPLIT_DEP", "REFUND", "FAKE_CRED")}
            e["prevalence"] = {"taux_faux_positifs": float(s["taux_faux_positifs"]), "rappel": float(s["rappel"]),
                               "precision_0.1pc": float(s["precision_0.001"]), "precision_1pc": float(s["precision_0.01"]),
                               "fausses_alertes_par_million_0.1pc": float(s["fausses_alertes_par_million_0.001"])}
        out[m["fichier"]] = e
    chemin = os.path.join(CKPT, "explications_tableau_de_bord.json")
    with open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"source": "analyses du mémoire sur la validation (fidélité et scénarios mesurés sur les finalistes, "
                             "pas sur la GRU FasterKAN ; test : voir le mémoire)", "modeles": out}, fh,
                  ensure_ascii=False, indent=1)
    print(chemin, ":", ", ".join(f"{k} ({'fidélité' if 'fidelite' in v else 'sans fidélité'})" for k, v in out.items()))


if __name__ == "__main__":
    main()
