#!/usr/bin/env bash
# lance_recherche.sh -- Recherche NSGA-II puis Sobol et Morris sur l'instance, en tmux (mkan-recherche).
# Recherche : 12 individus, 15 generations, 50 000 fenetres, 10 epoques (protocole du memoire).
# Sensibilite : Morris r = 10, Sobol N = 1024, 10 000 fenetres, 5 epoques ; run_search.py affiche le
# nombre d'entrainements au demarrage (r (d + 1) et N (d + 2)) : le verifier face au budget avant de laisser
# tourner. Reprise : evaluations relues depuis sorties/recherche/evaluations (NSGA-II) et
# sobol_evaluations.npy (Sobol). A ne lancer qu'une fois le dernier entrainement termine.
#     bash lance_recherche.sh [processus simultanes, defaut 30] [--skip-sensitivity ...]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$HOME/venv-mkan/bin/activate"
W="${1:-30}"
shift || true
OUT="$PWD/campagne_aws/sorties/recherche"
mkdir -p "$OUT"
CMD="OMP_NUM_THREADS=2 python run_search.py --device cpu --scratch_dir '$OUT' \
  --train_path ../../MOMTSIM/config/featuresLog.parquet --val_path ../../data/val_features.parquet \
  --eval_workers $W --sensitivity_max_workers $W $* 2>&1 | tee -a '$OUT/recherche.log'"
tmux new-session -d -s mkan-recherche "$CMD"
tmux ls
