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
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lstm_cell import ConfigurableMKANScorer
from gru_cell import GRUMKANScorer, match_gru_hidden_size

import sys, os
# Dossier MKAN/ (parent de ce fichier), PAS os.path.abspath("..") : ce dernier
# depend du dossier courant au lancement et casse l'import de loss.py si le
# script n'est pas lance depuis MKAN/Etude_benchmarck.
_MKAN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MKAN_DIR not in sys.path:
    sys.path.insert(0, _MKAN_DIR)
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
    # None = comportement historique (pas de clipping). Etape 2
    # (extended_heuristic_search.compute_fitness) utilisait max_norm=1.0.
    clip_grad_norm: Optional[float] = None


def _get_device(name: str):
    if name == "directml":
        import torch_directml
        return torch_directml.device()
    return torch.device(name)


def build_windows_sampled(df: pd.DataFrame, W: int, feature_cols: list,
                          max_accounts: Optional[int] = None,
                          n_max: Optional[int] = None, seed: int = 42):
    """Version vectorisee de build_windows + tirage de n_max fenetres.

    Memes fenetres, meme ordre global (compte puis pas de temps) que
    build_windows, et meme tirage que extended_heuristic_search.compute_fitness
    (rng(seed).choice(total, n_max, replace=False), indices tries) -- mais
    seules les fenetres RETENUES sont materialisees : sur 16 M de lignes,
    build_windows construit d'abord ~16 M fenetres (boucle Python, ~10 Go de
    RAM pour W=10) pour n'en garder que 50 000."""
    df = df.sort_values([ACCOUNT_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    targ = df[TARGET_COL].to_numpy(dtype=np.float32)
    acct = df[ACCOUNT_COL].to_numpy()

    _, starts = np.unique(acct, return_index=True)
    ends = np.append(starts[1:], len(acct))
    if max_accounts is not None:
        starts, ends = starts[:max_accounts], ends[:max_accounts]
    n_win = np.maximum(ends - starts - W + 1, 0)
    total = int(n_win.sum())
    if total == 0:
        return (np.zeros((0, W, len(feature_cols)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32), 0)

    if n_max and total > n_max:
        g = np.sort(np.random.default_rng(seed).choice(total, n_max, replace=False))
    else:
        g = np.arange(total)
    cum = np.cumsum(n_win)                                   # fin (exclusive) par compte
    acc_idx = np.searchsorted(cum, g, side="right")          # compte de chaque fenetre
    win_start = starts[acc_idx] + (g - (cum[acc_idx] - n_win[acc_idx]))
    X = feat[win_start[:, None] + np.arange(W)]              # (n, W, n_features)
    y = targ[win_start + W - 1]                              # etiquette du dernier pas
    return X, y, total


def load_and_window(train_path: str, val_path: str, regime: str, W: int,
                     max_accounts: Optional[int] = None,
                     n_windows_train: Optional[int] = None,
                     n_windows_val: Optional[int] = None,
                     sample_seed: int = 42) -> dict:
    """Isole la partie COUTEUSE et INDEPENDANTE des choix de bases/graine de
    run_config (lecture parquet, construction du regime, standardisation,
    fenetrage par compte) : ne depend que de (regime, W). Un appelant qui
    balaie de nombreux (cell_type, gate_bases, seed) pour un meme (regime, W)
    fixe (ex. plan de croisement Niveau 1) doit appeler CETTE fonction UNE
    SEULE FOIS par (regime, W) distinct et reutiliser son resultat via
    run_config(..., precomputed=...) plutot que de laisser run_config
    relire/refenetrer les memes donnees a chaque run (365 runs pour
    seulement 4 combinaisons (regime, W) distinctes dans le plan Niveau 1 --
    fenetrage par compte en boucle Python, le poste le plus couteux hors
    entrainement lui-meme)."""
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)

    df_train, feature_cols = build_regime_frame(df_train, regime)
    df_val, _              = build_regime_frame(df_val, regime)
    stats = standardize(df_train, feature_cols)
    standardize(df_val, feature_cols, stats=stats)

    X_train, y_train, n_train_full = build_windows_sampled(
        df_train, W, feature_cols, max_accounts, n_windows_train, sample_seed)
    del df_train
    X_val, y_val, n_val_full = build_windows_sampled(
        df_val, W, feature_cols, max_accounts, n_windows_val, sample_seed + 1)
    del df_val
    print(f"[donnees] regime={regime} W={W} : train {len(X_train):,}/{n_train_full:,} "
          f"fenetres, val {len(X_val):,}/{n_val_full:,} fenetres", flush=True)

    return {
        "X_train_t": torch.from_numpy(X_train), "y_train_t": torch.from_numpy(y_train),
        "X_val": X_val, "y_val": y_val,
        "input_size": len(feature_cols), "feature_cols": feature_cols,
    }


def _cfg_fingerprint(cfg: RunConfig, n_train: int) -> str:
    """Identifie un run independamment du device (un checkpoint pris sur MPS
    reste valable sur CPU) : refuse de reprendre un checkpoint d'une autre
    configuration ou d'un autre jeu d'entrainement."""
    d = asdict(cfg)
    d.pop("device", None)
    d["n_train_windows"] = int(n_train)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_torch_save(obj, path: str) -> None:
    """Ecrit dans un fichier temporaire puis os.replace : un crash PENDANT la
    sauvegarde ne corrompt jamais le dernier checkpoint valide."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_config(cfg: RunConfig, train_path: str, val_path: str,
               return_model: bool = False, precomputed: Optional[dict] = None,
               checkpoint_path: Optional[str] = None,
               checkpoint_every_s: float = 600.0,
               eval_batch_size: int = 4096,
               return_scores: bool = False) -> dict:
    """Entraîne UNE configuration et rapporte les métriques (protocole §5).

    checkpoint_path : si fourni, l'etat complet (modele, optimiseur, RNG,
    epoque, position dans l'epoque, permutation) est sauvegarde toutes les
    checkpoint_every_s secondes ET a chaque fin d'epoque ; un nouvel appel
    avec le meme chemin reprend exactement la ou le run s'etait arrete.
    eval_batch_size : taille des lots de validation (evite l'OOM).
    return_scores : ajoute "_val_scores" (np.ndarray) aux metriques.

    precomputed : resultat de load_and_window(..., cfg.regime, cfg.W,
    cfg.max_accounts), reutilisable tel quel pour tout run PARTAGEANT le meme
    (regime, W, max_accounts) -- evite de relire/refenetrer les donnees a
    chaque appel (cf. docstring de load_and_window). Si absent, comportement
    inchange : chargement autonome comme avant (retro-compatible, protocole
    108-runs existant non affecte)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _get_device(cfg.device)

    if precomputed is None:
        precomputed = load_and_window(train_path, val_path, cfg.regime, cfg.W, cfg.max_accounts)

    X_train_t = precomputed["X_train_t"]
    y_train_t = precomputed["y_train_t"]
    X_val     = precomputed["X_val"]
    y_val     = precomputed["y_val"]
    input_size = precomputed["input_size"]

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
    n = X_train_t.shape[0]
    n_batches = math.ceil(n / cfg.batch_size)

    # ── Reprise depuis un checkpoint intra-run (crash, coupure, Ctrl+C) ──────
    start_epoch, start_batch, perm = 0, 0, None
    loss_history, train_time_before = [], 0.0
    resumed_loss = None   # (somme des pertes, nb de batches) de l'epoque en cours
    fingerprint = _cfg_fingerprint(cfg, n)
    ck = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:   # noqa: BLE001
            tqdm.write(f"  [checkpoint illisible, ignore] {checkpoint_path} : {exc}")
    if ck is not None:
        if ck.get("fingerprint") == fingerprint:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            torch.set_rng_state(ck["torch_rng"])
            np.random.set_state(ck["numpy_rng"])
            start_epoch, start_batch = ck["epoch"], ck["batch"]
            perm = ck["perm"]
            if start_batch > 0:
                resumed_loss = (ck.get("epoch_loss_sum", 0.0), ck.get("epoch_n_seen", 0))
            loss_history = ck["loss_history"]
            train_time_before = ck["train_time_s"]
            tqdm.write(f"  [reprise checkpoint] epoque {start_epoch + 1}/{cfg.epochs}, "
                       f"batch {start_batch}/{n_batches} ({checkpoint_path})")
        else:
            tqdm.write(f"  [checkpoint ignore] configuration differente : {checkpoint_path}")

    def _save_checkpoint(epoch, batch, perm_, loss_sum_=0.0, n_seen_=0):
        if not checkpoint_path:
            return
        _atomic_torch_save({
            "fingerprint": fingerprint,
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "epoch": epoch, "batch": batch, "perm": perm_,
            "epoch_loss_sum": float(loss_sum_), "epoch_n_seen": int(n_seen_),
            "loss_history": loss_history,
            "train_time_s": train_time_before + (time.time() - t0),
        }, checkpoint_path)

    t0 = time.time()
    last_ck = time.time()
    model.train()
    in_epoch = False   # True entre le debut et la fin d'une epoque (Ctrl+C)
    try:
        epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="epoques", unit="ep",
                          leave=False, initial=start_epoch, total=cfg.epochs)
        for epoch in epoch_pbar:
            if perm is None:
                perm = torch.randperm(n)
            b0 = start_batch if epoch == start_epoch else 0
            # Accumulation en tenseur : pas de .item() a chaque batch (synchro
            # GPU/MPS systematique = ralentissement majeur sur des millions de pas).
            if epoch == start_epoch and resumed_loss is not None:
                loss_sum = torch.tensor(resumed_loss[0], device=device)
                n_seen = resumed_loss[1]
            else:
                loss_sum = torch.zeros((), device=device)
                n_seen = 0
            next_b = b0
            in_epoch = True
            batch_pbar = tqdm(range(b0, n_batches), total=n_batches, initial=b0,
                              desc=f"epoque {epoch + 1}/{cfg.epochs}", unit="batch",
                              leave=False, mininterval=1.0)
            for b in batch_pbar:
                idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                xb = X_train_t[idx].to(device, non_blocking=True)
                yb = y_train_t[idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss_total, _, _, _ = mkan_total_loss(
                    model, xb, yb, lam=cfg.lam, mu1=cfg.mu1, mu2=cfg.mu2)
                loss_total.backward()
                if cfg.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)
                opt.step()
                loss_sum = loss_sum + loss_total.detach().reshape(())
                n_seen += 1
                next_b = b + 1
                if n_seen % 200 == 0:
                    avg = (loss_sum / n_seen).item()
                    if not math.isfinite(avg):
                        raise RuntimeError(f"loss NaN/Inf a l'epoque {epoch + 1}, batch {b}")
                    batch_pbar.set_postfix_str(f"loss={avg:.4f}")
                if checkpoint_path and time.time() - last_ck >= checkpoint_every_s:
                    ls = loss_sum.item()
                    # Jamais de checkpoint d'un etat NaN : sinon chaque relance
                    # repartirait de poids corrompus et echouerait indefiniment.
                    if not math.isfinite(ls):
                        raise RuntimeError(f"loss NaN/Inf a l'epoque {epoch + 1}, batch {b}")
                    _save_checkpoint(epoch, b + 1, perm, ls, n_seen)
                    last_ck = time.time()
            epoch_loss = (loss_sum / max(1, n_seen)).item()
            if not math.isfinite(epoch_loss):
                raise RuntimeError(f"loss NaN/Inf a la fin de l'epoque {epoch + 1}")
            loss_history.append(epoch_loss)
            epoch_pbar.set_postfix_str(f"loss={epoch_loss:.4f}")
            in_epoch = False
            perm = None
            _save_checkpoint(epoch + 1, 0, None)
            last_ck = time.time()
    except KeyboardInterrupt:
        # Ctrl+C : les poids correspondent a tous les batches deja faits, on
        # sauvegarde donc la position reelle (pas celle du dernier checkpoint).
        if checkpoint_path and in_epoch and math.isfinite(loss_sum.item()):
            _save_checkpoint(epoch, next_b, perm, loss_sum.item(), n_seen)
            tqdm.write(f"  [interruption] checkpoint sauvegarde : {checkpoint_path}")
        raise
    train_time = train_time_before + (time.time() - t0)

    n_params = sum(p.numel() for p in model.parameters())

    # ── Validation PAR LOTS : un seul forward sur des millions de fenetres
    #    sature la RAM/VRAM (cause du crash apres entrainement). ──────────────
    model.eval()
    scores = np.array([], dtype=np.float32)
    if len(X_val) > 0:
        parts = []
        with torch.no_grad():
            for s in tqdm(range(0, len(X_val), eval_batch_size), desc="validation",
                          unit="lot", leave=False):
                xb = torch.from_numpy(X_val[s:s + eval_batch_size]).to(device)
                parts.append(model(xb).float().cpu().numpy())
        scores = np.concatenate(parts)
        n_bad = int((~np.isfinite(scores)).sum())
        if n_bad:
            tqdm.write(f"  [avertissement] {n_bad}/{len(scores)} scores NaN/Inf en validation")
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    metrics = compute_metrics(y_val, scores) if len(X_val) > 0 else {}
    metrics.update({
        "n_params": n_params,
        "train_time_s": train_time,
        "n_train_windows": int(n),
        "n_val_windows": int(len(X_val)),
        "loss_history": loss_history,
    })
    if return_scores:
        metrics["_val_scores"] = scores
    if return_model:
        # Prefixe "_" : champs non serialisables (nn.Module, tenseurs) a retirer
        # par l'appelant avant tout json.dump -- reserves au calcul, dans le
        # script appelant, de metriques qui necessitent le modele/les donnees
        # brutes (latence, R2_symbolic) sans dupliquer la logique d'entrainement
        # ni introduire un import circulaire vers extended_heuristic_search.py
        # (qui importe deja CE module).
        metrics["_model"] = model
        metrics["_input_size"] = input_size
        metrics["_X_train_pool"] = X_train_t
    return metrics


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_recall_curve, brier_score_loss, f1_score,
                                  matthews_corrcoef)
    out = {}
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": float("nan"), "auc_pr": float("nan")}
    out["auc_roc"] = roc_auc_score(y_true, scores)
    out["auc_pr"]  = average_precision_score(y_true, scores)
    out["brier"]   = brier_score_loss(y_true, scores)
    # MCC au seuil standard 0.5 -- cf. compute_fitness (extended_heuristic_search.py),
    # meme convention (MCC_clipped = max(0, MCC) applique par l'appelant, pas ici).
    out["mcc"] = float(matthews_corrcoef(y_true, (scores >= 0.5).astype(int)))

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
