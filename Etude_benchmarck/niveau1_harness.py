"""
niveau1_harness.py  Substitution par porte, croisée LSTM/GRU x brut/traité
(protocole_benchmark.md §2 et §4, Niveau 1).

Construit une cellule (ConfigurableTKANCell ou GRUKANCell) avec une base d'arête
choisie par porte, sur l'un des deux régimes d'entrée (brut ou traité  §2.2),
l'entraîne avec la loss régularisée de mkan (loss.py) et rapporte les métriques
du protocole (§5).

Une configuration = un point du plan de croisement à 108 cellules (27 substitutions
de porte x 4 combinaisons cellule/régime, §2.3). Ce module expose :

  - build_regime_frame(df, regime)      : construit les features (brut ou traité)
  - build_windows(df, W, feature_cols)  : fenêtrage par compte (miroir de train.ipynb)
  - run_config(cfg)                     : entraîne UNE configuration, retourne les métriques
  - CLI                                 : lance run_config depuis la ligne de commande,
                                           pour dispatcher les 540 runs (108 x 5 graines)
                                           en jobs indépendants (cluster / boucle shell).

Ce module ne lance PAS lui-même les 540 runs (coût de calcul, §2.3) : il fournit
le harness validé (cf. smoke test dans le README de ce dossier) que la boucle
d'exécution du protocole (§8) doit appeler une fois par configuration.
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from lstm_cell import ConfigurableMKANScorer
from gru_cell import GRUMKANScorer, match_gru_hidden_size

import sys, os
sys.path.insert(0, os.path.abspath(".."))
from loss import mkan_total_loss   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Régimes d'entrée (§2.2)
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED_FEATURE_COLS = [
    "delta_B_orig", "delta_B_dest", "r1", "r2", "flag_anomalie",
    "delta_commission", "var_agent_split", "rho_rupture", "rho_refund",
    "v1h", "flag_nuit", "rho_nouveau",
]
RAW_NUMERIC_COLS = ["amount", "oldBalanceOrig", "newBalanceOrig",
                     "oldBalanceDest", "newBalanceDest"]
ACTION_CATEGORIES = ["DEPOSIT", "CASH_IN", "DEBIT", "PAYMENT",
                     "TRANSFER", "CASH_OUT", "REFUND"]

TARGET_COL  = "isFraud"
ACCOUNT_COL = "nameOrig"
TIME_COL    = "step"


BINARY_COLS = {"flag_anomalie", "flag_nuit"}
ACTION_ONEHOT_COLS = [f"action_{cat}" for cat in ACTION_CATEGORIES]


def build_regime_frame(df: pd.DataFrame, regime: str) -> "tuple[pd.DataFrame, list]":
    """
    Construit les colonnes de features du régime demandé.

    regime : "traite" (features issues du feature engineering, eq. 3.8-3.19)
             "brut"   (colonnes transactionnelles + encodage one-hot de `action`)

    Prétraitement (avant standardize()) :
      - NaN -> 0.0 : sémantiquement correct pour les features conditionnelles
        (delta_commission, var_agent_split... NaN quand la feature ne s'applique
        pas à la transaction, cf. train.ipynb) ;
      - log1p signé (sign(x)*log1p(|x|)) sur les colonnes non binaires : ces
        features (montants, ratios, rho_*) sont fortement asymétriques, cohérent
        avec la transformation appliquée dans train.ipynb après test de
        normalité KS (eq. 4.5), simplifiée ici en un log-transform inconditionnel
        plutôt qu'un test par colonne (TopologyValidator non ré-importé).

    Returns:
        df augmenté des colonnes finales, liste des noms de colonnes finales
    """
    df = df.copy()
    if regime == "traite":
        cols = PROCESSED_FEATURE_COLS
    elif regime == "brut":
        for cat in ACTION_CATEGORIES:
            df[f"action_{cat}"] = (df["action"] == cat).astype(np.float32)
        cols = RAW_NUMERIC_COLS + ACTION_ONEHOT_COLS
    else:
        raise ValueError(f"regime inconnu : {regime}")

    df[cols] = df[cols].fillna(0.0)
    for c in cols:
        if c in BINARY_COLS or c in ACTION_ONEHOT_COLS:
            continue
        x = df[c].to_numpy(dtype=np.float64)
        df[c] = np.sign(x) * np.log1p(np.abs(x))
    return df, cols


def standardize(df: pd.DataFrame, cols: list, stats: Optional[dict] = None) -> dict:
    """Standardisation en place ; stats={col: (mean, std)} calculés si None, sinon réutilisés."""
    new_stats = {}
    for c in cols:
        if stats is None:
            mean = float(df[c].mean())
            std  = float(df[c].std()) or 1.0
        else:
            mean, std = stats[c]
        df[c] = ((df[c] - mean) / std).astype(np.float32).clip(-4, 4)
        new_stats[c] = (mean, std)
    return new_stats


def build_windows(df: pd.DataFrame, W: int, feature_cols: list,
                   max_accounts: Optional[int] = None):
    """
    Fenêtres glissantes par compte (miroir de make_windows, train.ipynb).
    max_accounts : sous-échantillonnage pour smoke test / débogage rapide.
    """
    df = df.sort_values([ACCOUNT_COL, TIME_COL]).reset_index(drop=True)
    feat = df[feature_cols].values.astype(np.float32)
    targ = df[TARGET_COL].values.astype(np.float32)
    acct = df[ACCOUNT_COL].values
    step = df[TIME_COL].values

    _, unique_starts = np.unique(acct, return_index=True)
    unique_ends = np.append(unique_starts[1:], len(acct))
    if max_accounts is not None:
        unique_starts = unique_starts[:max_accounts]
        unique_ends   = unique_ends[:max_accounts]

    X_list, y_list = [], []
    for start, end in zip(unique_starts, unique_ends):
        n = end - start
        if n < W:
            continue
        order = np.argsort(step[start:end])
        Xacc = feat[start:end][order]
        yacc = targ[start:end][order]
        for i in range(n - W + 1):
            X_list.append(Xacc[i:i + W])
            y_list.append(yacc[i + W - 1])

    if not X_list:
        return np.zeros((0, W, len(feature_cols)), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration et exécution d'un point du plan de croisement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    cell_type:   str            # "lstm" | "gru"
    regime:      str            # "brut" | "traite"
    gate_bases:  dict           # {"forget"/"reset": nom_base, ...}  cf. edges.BASIS_REGISTRY
    gate_kwargs: Optional[dict] = None
    hidden_size: int = 16
    W:           int = 10
    epochs:      int = 20
    lr:          float = 3e-3
    batch_size:  int = 64
    lam:         float = 1e-3
    mu1:         float = 1.0
    mu2:         float = 0.5
    seed:        int = 0
    max_accounts: Optional[int] = None   # sous-échantillonnage (smoke test)
    device:      str = "cpu"


def _get_device(name: str):
    if name == "directml":
        import torch_directml
        return torch_directml.device()
    return torch.device(name)


def run_config(cfg: RunConfig, train_path: str, val_path: str) -> dict:
    """Entraîne UNE configuration et rapporte les métriques (protocole §5)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _get_device(cfg.device)

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)

    df_train, feature_cols = build_regime_frame(df_train, cfg.regime)
    df_val, _              = build_regime_frame(df_val, cfg.regime)
    stats = standardize(df_train, feature_cols)
    standardize(df_val, feature_cols, stats=stats)

    X_train, y_train = build_windows(df_train, cfg.W, feature_cols, cfg.max_accounts)
    X_val,   y_val   = build_windows(df_val,   cfg.W, feature_cols, cfg.max_accounts)
    input_size = len(feature_cols)

    if cfg.cell_type == "lstm":
        model = ConfigurableMKANScorer(input_size, cfg.hidden_size,
                                        gate_bases=cfg.gate_bases, gate_kwargs=cfg.gate_kwargs)
    elif cfg.cell_type == "gru":
        h_gru = match_gru_hidden_size(input_size, cfg.hidden_size)
        model = GRUMKANScorer(input_size, h_gru,
                               gate_bases=cfg.gate_bases, gate_kwargs=cfg.gate_kwargs)
    else:
        raise ValueError(f"cell_type inconnu : {cfg.cell_type}")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    n = X_train_t.shape[0]

    t0 = time.time()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            opt.zero_grad()
            loss_total, loss_pred, reg_l1, reg_entropy = mkan_total_loss(
                model, xb, yb, lam=cfg.lam, mu1=cfg.mu1, mu2=cfg.mu2)
            loss_total.backward()
            opt.step()
    train_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).to(device)
        scores = model(X_val_t).cpu().numpy() if len(X_val) > 0 else np.array([])

    metrics = compute_metrics(y_val, scores) if len(X_val) > 0 else {}
    metrics.update({
        "n_params": n_params,
        "train_time_s": train_time,
        "n_train_windows": int(n),
        "n_val_windows": int(len(X_val)),
    })
    return metrics


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_recall_curve, brier_score_loss, f1_score)
    out = {}
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": float("nan"), "auc_pr": float("nan")}
    out["auc_roc"] = roc_auc_score(y_true, scores)
    out["auc_pr"]  = average_precision_score(y_true, scores)
    out["brier"]   = brier_score_loss(y_true, scores)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    for target_recall in (0.5, 0.8):
        mask = recall >= target_recall
        out[f"precision_at_recall_{target_recall}"] = float(precision[mask].max()) if mask.any() else 0.0

    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, scores >= t) for t in thresholds]
    out["f1_best"] = float(max(f1s))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plan de croisement (§2.3, §4 Niveau 1)
# ─────────────────────────────────────────────────────────────────────────────

GATE_CANDIDATES = {
    "forget":    ["hybrid", "fastkan", "fasterkan", "efficientkan", "relukan", "chebyshev"],
    "input":     ["hybrid", "fastkan", "fasterkan", "efficientkan", "relukan", "chebyshev"],
    "candidate": ["hybrid", "wavkan", "relukan", "chebyshev", "fourier", "fkan"],
    "output":    ["hybrid", "fasterkan", "relukan", "efficientkan", "linear"],
}
GRU_GATE_MAP = {"forget": "reset", "input": "update", "candidate": "candidate"}


def generate_grid():
    """
    Génère les 27 substitutions de porte (une porte à la fois, les autres en
    référence hybride) x {lstm, gru} x {brut, traite} = 108 configurations (§2.3).
    Retourne une liste de dicts prêts à instancier RunConfig (sans seed/chemins).
    """
    configs = []
    for cell_type in ("lstm", "gru"):
        gate_names = (["forget", "input", "candidate", "output"] if cell_type == "lstm"
                      else ["reset", "update", "candidate"])
        for gate_lstm_name, candidates in GATE_CANDIDATES.items():
            gate_name = (gate_lstm_name if cell_type == "lstm"
                        else GRU_GATE_MAP.get(gate_lstm_name))
            if gate_name is None:
                continue
            for basis_name in candidates:
                for regime in ("brut", "traite"):
                    gate_bases = {g: "hybrid" for g in gate_names}
                    gate_bases[gate_name] = basis_name
                    configs.append({
                        "cell_type": cell_type,
                        "regime": regime,
                        "gate_bases": gate_bases,
                        "substituted_gate": gate_lstm_name,
                        "substituted_basis": basis_name,
                    })
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON RunConfig (un point du plan)")
    parser.add_argument("--train_path", type=str,
                         default="../../MOMTSIM/config/featuresLog.parquet")
    parser.add_argument("--val_path", type=str, default="../../data/val_features.parquet")
    parser.add_argument("--list_grid", action="store_true",
                         help="affiche le plan de croisement complet (108 configs) en JSON")
    args = parser.parse_args()

    if args.list_grid:
        print(json.dumps(generate_grid(), indent=2, ensure_ascii=False))
    elif args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("substituted_gate", None)
        cfg_dict.pop("substituted_basis", None)
        cfg = RunConfig(**cfg_dict)
        result = run_config(cfg, args.train_path, args.val_path)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
