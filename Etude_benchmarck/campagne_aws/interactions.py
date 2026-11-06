"""
interactions.py -- Statistique d'interaction H de Friedman et Popescu (2008) entre variables,
calculee sur un finaliste entraine sans MultKAN, pour choisir les paires des noeuds produit.

    H2_ab = sum_i [PD_ab(x_a^i, x_b^i) - PD_a(x_a^i) - PD_b(x_b^i)]^2 / sum_i PD_ab(x_a^i, x_b^i)^2

PD : dependance partielle centree du score, en ne faisant varier que a (et b) au DERNIER pas de la
fenetre ; 500 fenetres de validation tirees avec la graine 0 ; variables continues de x seulement
(pas les colonnes h ni les indicatrices). Sortie : paires classees par H decroissant.

Usage (scheduler.py l'appelle une fois par finaliste) :
    python interactions.py --run_dir sorties/runs/<run_id> --data_dir donnees/<jeu> --out h.json
"""

import argparse
import json
import os
import sys
from itertools import combinations

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


@torch.no_grad()
def _pd(model, X: torch.Tensor, colonnes: tuple, batch: int = 4096) -> np.ndarray:
    """Dependance partielle centree aux points observes : pour chaque fenetre i, les variables de
    `colonnes` de toutes les fenetres prennent au dernier pas la valeur de la fenetre i."""
    n = X.shape[0]
    valeurs = X[:, -1, list(colonnes)].clone()
    pd = np.empty(n, dtype=np.float64)
    tampon = X.clone()
    for i in range(n):
        tampon[:, -1, list(colonnes)] = valeurs[i]
        total = 0.0
        for s in range(0, n, batch):
            total += float(model(tampon[s:s + batch]).sum())
        pd[i] = total / n
    return pd - pd.mean()


def h_interactions(model, X: torch.Tensor, indices: list) -> list:
    """indices : positions des variables retenues dans x. Retourne [{i, j, H}] par H decroissant."""
    pd1 = {a: _pd(model, X, (a,)) for a in indices}
    sortie = []
    for a, b in combinations(indices, 2):
        pd2 = _pd(model, X, (a, b))
        den = float((pd2 ** 2).sum())
        h2 = float(((pd2 - pd1[a] - pd1[b]) ** 2).sum() / den) if den > 0 else 0.0
        sortie.append({"i": int(a), "j": int(b), "H": float(np.sqrt(h2))})
    sortie.sort(key=lambda p: -p["H"])
    return sortie


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    from train_run import build_model
    from niveau1_harness import BINARY_COLS, ACTION_ONEHOT_COLS

    torch.set_num_threads(a.threads)
    ck = torch.load(os.path.join(a.run_dir, "model.pt"), map_location="cpu", weights_only=False)
    model = build_model(ck["spec"], ck["input_size"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cols = ck["preprocessing"]["feature_cols"]
    indices = [i for i, c in enumerate(cols) if c not in BINARY_COLS and c not in ACTION_ONEHOT_COLS]
    X_val = np.load(os.path.join(a.data_dir, "X_val.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(X_val), min(a.n, len(X_val)), replace=False))
    X = torch.from_numpy(np.ascontiguousarray(X_val[idx]))
    paires = h_interactions(model, X, indices)
    for q in paires:
        q["variables"] = [cols[q["i"]], cols[q["j"]]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"run_id": ck["spec"]["run_id"], "n_fenetres": len(idx), "paires": paires}, f,
                  indent=1, ensure_ascii=False)
    os.replace(tmp, a.out)
    print(f"[interactions] {len(paires)} paires -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
