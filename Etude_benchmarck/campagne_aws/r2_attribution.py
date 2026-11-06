"""
r2_attribution.py -- R2 symbolique recalcule SANS reentrainement sur les 40 aretes de plus fort score
d'attribution (KAN 2.0, Liu et al. 2024, section 4.1, Eq. 9), hors porte de sortie.

Le score de KAN 2.0 se calcule de la sortie vers l'entree : B_ij = A_j * sigma(phi_ij) / sigma(noeud j),
sigma = ecart-type sur les entrees reelles. KAN 2.0 le definit pour un reseau sans recurrence ; ici chaque
porte est une couche, et A_j, l'importance du noeud j pour le score de fraude, est mesuree par
sigma(noeud j) * moyenne |d score / d noeud j| sur les fenetres d'audit (adaptation de ce travail).

Sortie : runs/<run_id>/r2_attribution.json, puis r2_attribution.csv dans le dossier des sorties.
Usage : python r2_attribution.py --workers 8      (repris : les runs deja faits sont sautes)
"""

import argparse
import concurrent.futures as cf
import csv
import glob
import json
import multiprocessing as mp
import os
import sys

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

PORTES = ("forget_gate", "input_gate", "candidate_gate", "reset_gate", "update_gate")   # sans output_gate


def importance_noeuds(model, x_fenetres: torch.Tensor) -> dict:
    """A_j par porte : sigma(sortie du noeud j) * moyenne |d score / d noeud j|, moyennes sur les pas."""
    captures = {}
    crochets = []

    def crochet(nom):
        def f(_mod, _ent, sortie):
            sortie.retain_grad()
            captures.setdefault(nom, []).append(sortie)
        return f

    for nom in PORTES + ("output_gate",):
        couche = getattr(model.cell, nom, None)
        if couche is not None:
            crochets.append(couche.register_forward_hook(crochet(nom)))
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(True)      # active le graphe : les sorties des portes en dependent
        model.zero_grad(set_to_none=True)
        model(x_fenetres).sum().backward()
    for c in crochets:
        c.remove()
    sortie = {}
    n = x_fenetres.shape[0]
    for nom, pas in captures.items():
        a = [s.detach().std(dim=0) * (s.grad.abs().sum(dim=0) / n) for s in pas if s.grad is not None]
        if a:
            sortie[nom] = torch.stack(a).mean(dim=0)
    return sortie


@torch.no_grad()
def aretes_par_attribution(model, x_pool: torch.Tensor, A: dict, theta: float = 0.01) -> list:
    """Memes aretes admissibles que _compute_r2_symbolic (L1 > theta), classees par B_ij."""
    survivants = []
    for nom in PORTES:
        couche = getattr(model.cell, nom, None)
        if couche is None or nom not in A:
            continue
        aretes = couche.edge_activations(x_pool)                 # (lot, in, out)
        l1 = aretes.abs().mean(dim=0)
        sigma_noeud = couche(x_pool).std(dim=0)
        b = A[nom][None, :] * aretes.std(dim=0) / (sigma_noeud[None, :] + 1e-12)
        for i, j in (l1 > theta).nonzero(as_tuple=False).tolist():
            survivants.append((couche, i, j, float(b[i, j])))
    survivants.sort(key=lambda s: -s[3])
    return survivants


def traiter(run_dir: str, threads: int = 2) -> str:
    sortie = os.path.join(run_dir, "r2_attribution.json")
    if os.path.exists(sortie):
        return run_dir
    from extended_heuristic_search import _compute_r2_symbolic
    from niveau1_benchmark_run import _build_x_pool
    from train_run import build_model
    torch.set_num_threads(threads)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    spec = ck["spec"]
    model = build_model(spec, ck["input_size"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    X_val = np.load(os.path.join(spec["data_dir"], "X_val.npy"), mmap_mode="r")
    idx = np.load(os.path.join(spec["data_dir"], "audit_idx.npy"))
    x = torch.from_numpy(np.ascontiguousarray(X_val[idx]))
    A = importance_noeuds(model, x)
    x_pool = _build_x_pool(model, spec["cell_type"], x, spec["W"])
    surv = aretes_par_attribution(model, x_pool, A)[:40]
    res = {"run_id": spec["run_id"], "n_aretes": len(surv), "r2_attribution": {}}
    for seuil in (config.R2_ALT, config.R2_THRESHOLD):
        res["r2_attribution"][f"{seuil:.2f}"] = float(
            _compute_r2_symbolic(model, x_pool, r2_threshold=seuil, survivors=surv))
    tmp = sortie + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(tmp, sortie)
    return run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out_dir", default=config.OUT_DIR)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a_faire = []
    for done in sorted(glob.glob(os.path.join(a.out_dir, "runs", "*", "DONE"))):
        run_dir = os.path.dirname(done)
        with open(os.path.join(run_dir, "spec.json"), encoding="utf-8") as f:
            if not json.load(f).get("baseline"):
                a_faire.append(run_dir)
    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, _ in enumerate(pool.map(traiter, a_faire), 1):
            print(f"[r2_attribution] {i}/{len(a_faire)}", flush=True)
    lignes = []
    for run_dir in a_faire:
        with open(os.path.join(run_dir, "r2_attribution.json"), encoding="utf-8") as f:
            r = json.load(f)
        lignes.append({"run_id": r["run_id"], "n_aretes": r["n_aretes"],
                       **{f"r2_attribution_{k}": v for k, v in r["r2_attribution"].items()}})
    chemin = os.path.join(a.out_dir, "r2_attribution.csv")
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    print(f"[r2_attribution] -> {chemin}", flush=True)


if __name__ == "__main__":
    main()
