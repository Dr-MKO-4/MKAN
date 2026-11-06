"""
analyse_raccourci.py -- Test du raccourci par les soldes, partie donnees : part des transactions dont le
solde emetteur se reconcilie avec le montant, par classe et par scenario, sur le train et la validation.

Reconciliation : | |oldBalanceOrig - newBalanceOrig| - amount | <= tolerance. Si elle separe les classes
presque parfaitement, le modele peut detecter la fraude par cette egalite comptable plutot que par le
comportement ; le reentrainement sans soldes (scheduler.py --mode raccourci) mesure ce qu'il en reste.

Usage : python analyse_raccourci.py --out sorties/raccourci.json
"""

import argparse
import json
import os
import sys

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLONNES = ["amount", "oldBalanceOrig", "newBalanceOrig", "isFraud", "fraudScenario", "action"]


def taux_reconciliation(df: pd.DataFrame, tolerance: float) -> dict:
    ecart = (df["oldBalanceOrig"] - df["newBalanceOrig"]).abs() - df["amount"]
    ok = ecart.abs().to_numpy() <= tolerance
    frauduleux = df["isFraud"].to_numpy().astype(bool)

    def part(masque):
        return {"n": int(masque.sum()), "taux": float(ok[masque].mean()) if masque.any() else None}

    scenarios = df["fraudScenario"].fillna("aucun").astype(str).to_numpy()
    return {"tolerance": tolerance, "total": part(np.ones(len(df), dtype=bool)),
            "legitime": part(~frauduleux), "fraude": part(frauduleux),
            "par_scenario": {s: part((scenarios == s) & frauduleux)
                             for s in sorted(set(scenarios[frauduleux]))},
            "par_action_et_classe": {f"{a}/{'fraude' if c else 'legitime'}":
                                     part((df["action"].to_numpy() == a) & (frauduleux == c))
                                     for a in sorted(df["action"].unique()) for c in (False, True)}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default=os.path.join(config.OUT_DIR, "raccourci.json"))
    p.add_argument("--tolerance", type=float, default=0.01)
    a = p.parse_args()
    res = {}
    for nom, chemin in (("train", config.TRAIN_PATH), ("validation", config.VAL_PATH)):
        res[nom] = taux_reconciliation(pd.read_parquet(chemin, columns=COLONNES), a.tolerance)
        print(f"{nom} : fraude {res[nom]['fraude']['taux']}, legitime {res[nom]['legitime']['taux']}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
