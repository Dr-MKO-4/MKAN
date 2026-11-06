#!/usr/bin/env bash
# launch.sh -- Lance la campagne dans tmux (les sessions survivent a la deconnexion SSH).
#     bash launch.sh bench                     # banc d'essai seul (sentinelle + suivi actifs)
#     bash launch.sh etape4 32 2               # nouvelle Etape 4 : 32 processus x 2 threads
#     bash launch.sh final 20 3 finalistes.json
#     bash launch.sh preproc 64 auto finalistes.json       # experiences : preproc, multkan, baselines
#     bash launch.sh raccourci 64 auto finalistes.json emetteur   # ou : tous
#     bash launch.sh stop                      # arret propre : sauvegarde de tous les runs
# Suivre : tmux attach -t mkan-sched   (Ctrl+b puis d pour detacher)
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source "$HOME/venv-mkan/bin/activate"
OUT="${MKAN_OUT_DIR:-$PWD/sorties}"
mkdir -p "$OUT"
MODE="${1:-}"

start_bg() {   # nom, commande
  tmux has-session -t "$1" 2>/dev/null || tmux new-session -d -s "$1" "$2 2>&1 | tee -a $OUT/$1.log"
}

case "$MODE" in
  bench|etape4|final|preproc|multkan|raccourci|baselines)
    rm -f "$OUT/STOP"
    start_bg mkan-sentinel "python sentinel.py"
    start_bg mkan-monitor "python monitor/app.py"
    ;;
esac

case "$MODE" in
  bench)
    tmux new-session -d -s mkan-bench "python bench.py 2>&1 | tee -a $OUT/bench.log" ;;
  etape4)
    tmux new-session -d -s mkan-sched \
      "python scheduler.py --mode etape4 --workers ${2:-32} --threads ${3:-2} 2>&1 | tee -a $OUT/scheduler.log" ;;
  final)
    tmux new-session -d -s mkan-sched \
      "python scheduler.py --mode final --workers ${2:-20} --threads ${3:-3} --finalists ${4:?fichier des finalistes} 2>&1 | tee -a $OUT/scheduler.log" ;;
  preproc|multkan|raccourci|baselines)
    tmux new-session -d -s mkan-sched       "python scheduler.py --mode $MODE --workers ${2:-64} --threads ${3:-auto} --finalists ${4:?fichier des finalistes} ${5:+--variante $5} 2>&1 | tee -a $OUT/scheduler.log" ;;
  stop)
    echo "arret demande par l'utilisateur" > "$OUT/STOP"
    echo "STOP cree : chaque run sauvegarde et quitte ; la sentinelle copie vers S3." ;;
  *)
    echo "usage : bash launch.sh {bench|etape4 [workers threads]|final workers threads finalistes.json|preproc|multkan|raccourci|baselines workers threads finalistes.json [variante]|stop}"
    exit 1 ;;
esac
tmux ls
