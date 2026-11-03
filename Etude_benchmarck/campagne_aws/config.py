"""
config.py -- Protocole de la campagne AWS (nouvelle Etape 4 puis entrainement final).

Chaque valeur renvoie a une decision de memoire/DECISIONS.md (entree du 2026-11-03,
« Reprise de l'Etape 4 sur AWS ») ou reprend telle quelle une valeur de l'Etape 4
du 21/10 (niveau1_benchmark_run.CELL_THETA_OPT). Les valeurs marquees
« A VALIDER » sont des propositions a confirmer par Miguel avant le lancement.
"""

import os
import sys

CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
ETUDE_DIR = os.path.dirname(CAMPAGNE_DIR)
MKAN_DIR = os.path.dirname(ETUDE_DIR)
ROOT_DIR = os.path.dirname(MKAN_DIR)
for _p in (ETUDE_DIR, MKAN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Donnees : jeux regeneres du 27/10 (decision : Etape 4 refaite sur ces jeux) ──
TRAIN_PATH = os.path.join(ROOT_DIR, "MOMTSIM", "config", "featuresLog.parquet")
VAL_PATH = os.path.join(ROOT_DIR, "data", "val_features.parquet")
# Le jeu de TEST n'est jamais lu par cette campagne (regle 4 du chapitre 4).

DATA_DIR = os.environ.get("MKAN_DATA_DIR", os.path.join(CAMPAGNE_DIR, "donnees"))
OUT_DIR = os.environ.get("MKAN_OUT_DIR", os.path.join(CAMPAGNE_DIR, "sorties"))

# ── Hyperparametres d'elite de l'Etape 2, inchanges (decision : lot 32, lr 1e-4) ──
CELL_THETA = {
    "lstm": dict(W=10, lr=1e-4, batch_size=32, lam=1e-4, mu1=0.1, mu2=1.6515820625005828),
    "gru":  dict(W=5,  lr=1e-4, batch_size=32, lam=1e-4, mu1=0.1, mu2=0.6989017708691363),
}
CLIP_GRAD_NORM = 1.0

# ── Taille : budget de parametres commun TKAN/GRU (decision du 03/11) ──
# hidden_size de la TKAN fixe par le banc d'essai (bench.py) ; la GRU recoit
# gru_cell.match_gru_hidden_size(input_size, HIDDEN_TKAN), qui egalise le nombre
# d'aretes des deux cellules (3 portes contre 4).
HIDDEN_TKAN = int(os.environ.get("MKAN_HIDDEN_TKAN", "64"))

# ── Domaine des bases a grille : recale sur les donnees (decision du 03/11) ──
GRID_INIT = "data"          # "data" (correctif) ou "legacy" ([-1, 1], Etape 4 du 21/10)
N_GRID_SAMPLES = 100_000    # valeurs x_t tirees des fenetres d'entrainement pour la grille

# ── Latence (decision du 03/11) : seuil 500 ms, penalite plafonnee ──
LATENCY_THRESHOLD_MS = 500.0
LATENCY_CAPPED = True

# ── Nouvelle Etape 4 ──
ETAPE4 = dict(
    n_windows_train=int(os.environ.get("MKAN_N_TRAIN", "300000")),  # fixe apres le banc d'essai
    max_epochs=15,          # A VALIDER (Etape 4 du 21/10 : 10 epoques fixes)
    patience=3,             # A VALIDER (arret anticipe sur la perte de suivi)
    min_delta=1e-4,
    n_monitor=50_000,       # fenetres de validation suivies pour l'arret anticipe
    n_seeds=5,
    top_n_engineered=3,     # strategie raw_first de l'Etape 4, inchangee
)

# ── Entrainement final des finalistes ──
FINAL = dict(
    n_windows_train=int(os.environ.get("MKAN_N_TRAIN_FINAL", "1000000")),  # 0 = toutes
    max_epochs=60,
    patience=6,
    min_delta=1e-4,
    n_monitor=50_000,
    seeds=[1, 2, 3, 4, 5],
)

# ── Sauvegarde et reprise ──
CHECKPOINT_EVERY_S = 300          # point de reprise intra-epoque (en plus de chaque fin d'epoque)
STATUS_EVERY_S = 2.0              # frequence d'ecriture de status.json (suivi en direct)
STOP_FILE = os.path.join(OUT_DIR, "STOP")   # cree par sentinel.py (preavis spot, budget)

# ── R2_symbolic : memes reglages que l'Etape 4 ──
R2_THRESHOLD = 0.99
R2_ALT = 0.80                     # seuil de travail principal depuis le 27/10
N_AUDIT_POOL = 256

# ── Profils d'activation des aretes sauvegardes avec chaque modele (decision du 03/11) ──
N_PROFILE_POINTS = 33

# ── AWS ──
S3_BUCKET = os.environ.get("MKAN_S3_BUCKET", "mkan-training-mko")
S3_PREFIX = os.environ.get("MKAN_S3_PREFIX", "campagne_v2")
BUDGET_USD = float(os.environ.get("MKAN_BUDGET_USD", "80"))
