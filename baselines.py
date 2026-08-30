"""
baselines.py  Modèles de référence pour comparaison avec MKAN (section 5.x).

  make_xgb_features   agrégation de fenêtre W×F → features tabulaires pour XGBoost
  train_xgboost       entraînement XGBClassifier (xgboost library)
  LSTMBaseline        LSTM PyTorch 2 couches → sigmoid
  train_lstm          boucle d'entraînement avec early stopping

Tous les modèles exposent la même interface de prédiction :
    proba = predict_proba(model, X_windows, device, model_type)   → (N,) numpy
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── Features tabulaires pour XGBoost ─────────────────────────────────────────

def make_xgb_features(X_windows: np.ndarray) -> np.ndarray:
    """
    Transforme (N, W, F) en (N, 4·F) par agrégation temporelle.

    Les 4 agrégats par feature sont : dernière valeur, moyenne, écart-type,
    différence absolue début/fin (indicateur de tendance sur la fenêtre).
    """
    last  = X_windows[:, -1, :]
    mean  = X_windows.mean(axis=1)
    std   = X_windows.std(axis=1)
    delta = np.abs(X_windows[:, -1, :] - X_windows[:, 0, :])
    return np.concatenate([last, mean, std, delta], axis=1)


# ── XGBoost ──────────────────────────────────────────────────────────────────

class _XGBTqdmCallback(xgb.callback.TrainingCallback):
    """Callback tqdm pour XGBoost : barre de progression sur les estimateurs."""

    def __init__(self, n_estimators: int):
        super().__init__()
        self._n = n_estimators
        self._bar = None

    def before_training(self, model):
        self._bar = tqdm(total=self._n, desc="XGBoost", unit="tree", leave=True)
        return model

    def after_iteration(self, model, epoch, evals_log):
        self._bar.update(1)
        if evals_log:
            last_key    = list(evals_log.keys())[-1]
            last_metric = list(evals_log[last_key].keys())[-1]
            score = evals_log[last_key][last_metric][-1]
            self._bar.set_postfix({"val_logloss": f"{score:.4f}"})
        return False

    def after_training(self, model):
        self._bar.close()
        return model


def train_xgboost(
    X_train_w: np.ndarray, y_train: np.ndarray,
    X_val_w:   np.ndarray, y_val:   np.ndarray,
    n_estimators: int   = 400,
    max_depth:    int   = 6,
    lr:           float = 0.05,
    subsample:    float = 0.8,
    scale_pos_weight: float | None = None,
):
    """
    Entraîne un XGBClassifier sur les features agrégées (make_xgb_features).

    scale_pos_weight est calculé automatiquement à partir de y_train si None
    (ratio négatifs/positifs, recommandé sur données déséquilibrées).
    """
    assert HAS_XGB, "xgboost non installé : pip install xgboost"

    Xtr = make_xgb_features(X_train_w)
    Xva = make_xgb_features(X_val_w)

    if scale_pos_weight is None:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg * (1.0 / max(pos, 1))

    model = xgb.XGBClassifier(
        n_estimators          = n_estimators,
        max_depth             = max_depth,
        learning_rate         = lr,
        subsample             = subsample,
        colsample_bytree      = 0.8,
        scale_pos_weight      = scale_pos_weight,
        eval_metric           = "logloss",
        tree_method           = "hist",
        early_stopping_rounds = 30,
        random_state          = 42,
        verbosity             = 0,
        callbacks             = [_XGBTqdmCallback(n_estimators)],
    )
    model.fit(
        Xtr, y_train,
        eval_set = [(Xva, y_val)],
        verbose  = False,
    )
    return model


# ── LSTM Baseline ─────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    LSTM PyTorch 2 couches → projection linéaire → sigmoid.
    Architecture minimale pour comparaison équitable avec MKAN.
    Même interface d'entrée : (batch, W, input_size) → (batch,).
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return torch.sigmoid(self.head(h_n[-1])).squeeze(-1)


def train_lstm(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    input_size:   int,
    hidden_size:  int   = 32,
    n_epochs:     int   = 30,
    lr:           float = 1e-3,
    patience:     int   = 5,
    device:       str   = "cpu",
):
    """
    Entraîne LSTMBaseline avec BCE pondérée + early stopping sur la val loss.

    Returns:
        model   : LSTMBaseline avec les meilleurs poids val
        history : dict {'train_loss': [...], 'val_loss': [...]}
    """
    model = LSTMBaseline(input_size, hidden_size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    # Poids de classe positif estimé sur le jeu d'entraînement
    all_y     = torch.cat([yb for _, yb in train_loader])
    pos_w     = ((all_y == 0).sum().float() * (all_y == 1).sum().float().clamp(min=1).reciprocal()).to(device)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    history        = {"train_loss": [], "val_loss": []}

    epoch_bar = tqdm(range(n_epochs), desc="LSTM époques", unit="ep", leave=True)
    for epoch in epoch_bar:
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"  train ep{epoch+1}",
                            unit="batch", leave=False):
            xb, yb = xb.to(device), yb.float().to(device)
            opt.zero_grad()
            pred = model(xb)
            w    = torch.where(yb == 1, pos_w, torch.ones_like(yb))
            loss = (w * nn.functional.binary_cross_entropy(pred, yb, reduction="none")).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                pred = model(xb)
                w    = torch.where(yb == 1, pos_w, torch.ones_like(yb))
                loss = (w * nn.functional.binary_cross_entropy(pred, yb, reduction="none")).mean()
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        star = " ★" if val_loss < best_val_loss else ""
        epoch_bar.set_postfix({"train": f"{train_loss:.4f}",
                                "val":   f"{val_loss:.4f}{star}"})

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                tqdm.write(f"Early stopping à l'époque {epoch+1} (patience={patience})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ── Interface de prédiction unifiée ──────────────────────────────────────────

def predict_proba(model, X_windows: np.ndarray, device,
                  model_type: str = "mkan") -> np.ndarray:
    """
    Retourne les probabilités de fraude (N,) en numpy pour n'importe quel modèle.

    Pour les modèles PyTorch sur DirectML, l'inférence est automatiquement
    déportée sur CPU (bug version_counter sous no_grad) puis le modèle est
    remis sur son device d'origine.

    model_type : "mkan" | "lstm" | "xgb"
    """
    if model_type in ("mkan", "lstm"):
        model.eval()
        orig_dev = next(model.parameters()).device
        # DirectML (privateuseone) incompatible avec no_grad → inférer sur CPU
        use_cpu  = str(orig_dev.type) != "cpu"
        if use_cpu:
            model.cpu()
        infer_dev = "cpu"

        preds = []
        X_t   = torch.from_numpy(np.ascontiguousarray(X_windows, dtype=np.float32))
        with torch.no_grad():
            for i in range(0, len(X_t), 256):
                xb = X_t[i:i + 256].to(infer_dev)
                p  = model(xb)
                preds.append(p.cpu().numpy())

        if use_cpu:
            model.to(orig_dev)   # remettre sur DirectML après inférence
        return np.concatenate(preds)
    elif model_type == "xgb":
        return model.predict_proba(make_xgb_features(X_windows))[:, 1]
    else:
        raise ValueError(f"model_type inconnu : {model_type!r}")
