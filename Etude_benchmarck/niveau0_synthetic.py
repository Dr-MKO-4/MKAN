"""
niveau0_synthetic.py  Approximation unitaire d'arête (protocole_benchmark.md §4,
Niveau 0).

Entraîne chaque base d'arête (budget iso B=12/arête, cf. edges/bases.py) comme
une GenericKANLayer 1->1 sur chacune des 6 familles synthétiques du protocole,
sur 5 graines, et rapporte la MSE finale moyenne +/- écart-type.

Usage :
    python niveau0_synthetic.py
    python niveau0_synthetic.py --epochs 300 --n_points 1000 --out results/niveau0.md
"""

import argparse
import math
import statistics as stats

import torch
import torch.nn as nn

from edges import BASIS_REGISTRY, GenericKANLayer


# ─────────────────────────────────────────────────────────────────────────────
# Familles synthétiques (protocole §4, Niveau 0)
# ─────────────────────────────────────────────────────────────────────────────

def _target_seuil_dur(x):
    return torch.sigmoid(20.0 * (x - 0.3))

def _target_seuil_double(x):
    return ((x > -0.2) & (x < 0.5)).float()

def _target_quasi_lineaire(x):
    return 0.8 * x + 0.05

def _target_haute_frequence(x):
    return torch.sin(5 * math.pi * x) + x

def _target_multi_echelle(x):
    return torch.sin(2 * math.pi * x) + 0.3 * torch.sin(20 * math.pi * x)

def _target_bruit(x, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    noise = torch.randn(x.shape, generator=g) * 0.1
    return _target_multi_echelle(x) + noise


FAMILIES = {
    "seuil_dur":         (_target_seuil_dur,        "Forget, Input"),
    "seuil_double":      (_target_seuil_double,     "Forget, Input"),
    "quasi_lineaire":    (_target_quasi_lineaire,   "Output"),
    "haute_frequence":   (_target_haute_frequence,  "Candidate"),
    "multi_echelle":     (_target_multi_echelle,    "Candidate"),
    "bruit":             (None,                     "Candidate (robustesse)"),   # cas spécial : seed-dependent
}


def make_dataset(family: str, n_points: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n_points, 1, generator=g) * 2 - 1)   # uniforme sur [-1, 1]
    if family == "bruit":
        y = _target_bruit(x, seed)
    else:
        fn, _ = FAMILIES[family]
        y = fn(x)
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement d'une base sur une famille
# ─────────────────────────────────────────────────────────────────────────────

def train_one(basis_name: str, family: str, seed: int,
              n_points: int, epochs: int, lr: float = 1e-2, use_native: bool = False):
    torch.manual_seed(seed)
    cls = BASIS_REGISTRY[basis_name]
    kwargs = cls.native_kwargs if use_native else cls.budget_kwargs
    basis = cls(in_features=1, out_features=1, **kwargs)
    layer = GenericKANLayer(basis, in_features=1, out_features=1)

    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x, y = make_dataset(family, n_points, seed)
    x_val, y_val = make_dataset(family, max(200, n_points // 5), seed + 500)

    best_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        pred = layer(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        val_mse = loss_fn(layer(x_val), y_val).item()
    return val_mse


def run_niveau0(epochs: int, n_points: int, seeds=(0, 1, 2, 3, 4), use_native: bool = False):
    results = {}   # {family: {basis: (mean, std)}}
    for family in FAMILIES:
        results[family] = {}
        for basis_name in BASIS_REGISTRY:
            mses = [train_one(basis_name, family, s, n_points, epochs, use_native=use_native)
                    for s in seeds]
            results[family][basis_name] = (stats.mean(mses), stats.pstdev(mses))
    return results


def format_markdown(results: dict, title: str) -> str:
    families = list(results.keys())
    bases = list(BASIS_REGISTRY.keys())
    lines = [f"# {title}", "", "MSE de validation (moyenne ± écart-type, 5 graines).", ""]
    header = "| Base | " + " | ".join(families) + " |"
    sep    = "|---|" + "---|" * len(families)
    lines += [header, sep]
    for basis_name in bases:
        row = [basis_name]
        for family in families:
            mean, std = results[family][basis_name]
            row.append(f"{mean:.4g} ± {std:.4g}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n_points", type=int, default=1000)
    parser.add_argument("--out", type=str, default="results/niveau0_synthetic_budget.md")
    parser.add_argument("--native", action="store_true",
                         help="utilise native_kwargs (réglage publié) au lieu du budget iso B=12")
    args = parser.parse_args()

    res = run_niveau0(args.epochs, args.n_points, use_native=args.native)
    title = ("Niveau 0  réglage natif optimal" if args.native
              else "Niveau 0  budget iso B=12 paramètres/arête")
    md = format_markdown(res, title)
    print(md)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nÉcrit dans {args.out}")
