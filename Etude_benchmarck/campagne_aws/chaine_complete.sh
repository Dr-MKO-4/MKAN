#!/usr/bin/env bash
# chaine_complete.sh -- Enchaine l'entrainement final et les experiences, sans intervention.
#     bash launch.sh chaine 64 auto sorties/finalistes.json
# Reprise : une etape finie laisse sorties/chaine/<etape>.ok et n'est pas refaite ; dans une etape, les runs
# finis (DONE) sont sautes. Une etape en echec est notee dans sorties/chaine/echecs.log, la suivante demarre,
# et l'etape en echec est retentee a la relance suivante.
# Budget : aucune etape ne demarre au-dela de MKAN_PLAFOND_CHAINE_USD (68 $ par defaut), et la recherche y est
# arretee, pour garder de quoi faire la derive et le test. Sentinel.py reste l'arret general (76 $).
# Derive : lancee des que ~/Modelisation/data/derive_features.parquet existe. Hors chaine : evaluation sur le
# test (une seule fois, en dernier).
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source "$HOME/venv-mkan/bin/activate"
# Coeurs reserves a la mesure de latence (13/11, accord de Miguel) : la chaine et tout ce qu'elle lance restent sur
# config.TRAIN_CORES ; la recherche et Sobol mesurent la latence seuls sur config.LATENCY_CORES.
taskset -a -p -c "$(python -c 'import config; print(config.TRAIN_CORES)')" $$ > /dev/null
OUT="${MKAN_OUT_DIR:-$PWD/sorties}"
W="${1:-64}"; T="${2:-auto}"; F="${3:?fichier des finalistes}"
PLAFOND="${MKAN_PLAFOND_CHAINE_USD:-68}"
M="$OUT/chaine"
mkdir -p "$M"
S=(python scheduler.py --workers "$W" --threads "$T" --finalists "$F")

depasse() {
  local c
  c=$(grep -o '"total_usd": [0-9.]*' "$OUT/cost_state.json" | head -1 | awk '{print $2}')
  awk -v c="${c:-0}" -v p="$PLAFOND" 'BEGIN { exit !(c + 0 >= p) }'
}

fin_etape() {    # nom, code de sortie
  if [ -f "$OUT/STOP" ] || [ "$2" = 3 ]; then
    echo "=== $(date -u) arret pendant $1" ; exit 3
  fi
  if [ "$2" = 0 ] && { [ "$1" != sobol_v2 ] || [ -f "$OUT/recherche_v2/sobol_results.json" ]; }; then   # 15/11 : run_search sort en 0 meme si Sobol echoue
    touch "$M/$1.ok"; echo "=== $(date -u) fin de $1"
  else
    echo "$(date -u) $1 : code $2" >> "$M/echecs.log"; echo "=== $(date -u) ECHEC de $1 (code $2), etape suivante"
  fi
}

debut_etape() {  # nom ; code 1 si l'etape ne doit pas tourner
  [ -f "$M/$1.ok" ] && { echo "=== $1 : deja finie"; return 1; }
  [ -f "$OUT/STOP" ] && { echo "=== $(date -u) STOP avant $1"; exit 3; }
  depasse && { echo "=== $(date -u) plafond de $PLAFOND \$ : $1 non lancee" | tee -a "$M/echecs.log"; return 1; }
  echo "=== $(date -u) debut de $1"
  return 0
}

etape() {        # nom, commande
  local nom="$1"; shift
  debut_etape "$nom" || return 0
  "$@"
  fin_etape "$nom" $?
}

etape_plafonnee() {   # nom, commande : arretee (groupe de processus entier) quand le plafond est atteint
  local nom="$1"; shift
  debut_etape "$nom" || return 0
  setsid "$@" &
  local pid=$! code
  while kill -0 "$pid" 2>/dev/null; do
    if depasse; then
      echo "=== $(date -u) plafond de $PLAFOND \$ atteint pendant $nom : arret, reprise possible" | tee -a "$M/echecs.log"
      kill -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 0
    fi
    sleep 60
  done
  wait "$pid"; code=$?
  fin_etape "$nom" "$code"
}

RECHERCHE=(env OMP_NUM_THREADS=2 python ../run_search.py --device cpu --scratch_dir "$OUT/recherche"
           --train_path ../../../MOMTSIM/config/featuresLog.parquet --val_path ../../../data/val_features.parquet
           --test_path ../../../data/val_features.parquet    # presence verifiee seulement : le test reste hors instance
           --eval_workers 30 --sensitivity_max_workers 30
           # Latence selon la decision du 03/11 (13/11 : la recherche du 12/11 avait garde 100 ms sans penalite) :
           # seuil lu dans config.py (source unique), penalite min(1,(L-T)/T) appliquee, contrainte au meme seuil.
           --enable-latency-penalty --latency-penalty-capped
           --latency-threshold-ms "$(python -c 'import config; print(config.LATENCY_THRESHOLD_MS)')"
           # Relecture de la session memoire (13/11) : R2 de la fitness a 0,80 et grille recalee, comme le plan et le
           # final (valeurs lues dans config.py).
           --r2-threshold "$(python -c 'import config; print(config.R2_ALT)')"
           --grid-init "$(python -c 'import config; print(config.GRID_INIT)')")
mkdir -p "$OUT/recherche"

DERIVE="${MKAN_DERIVE_PATH:-$HOME/Modelisation/data/derive_features.parquet}"
derive() {    # jeu derive a generer par Miguel (MOMTSIM/generer_derive.py) : reportee tant qu'il manque
  if [ -f "$DERIVE" ]; then etape derive "${S[@]}" --mode derive --drift "$DERIVE"
  else echo "=== $(date -u) jeu derive absent ($DERIVE) : derive reportee"; fi
}

# Ordre convenu : final et baselines, puis raccourci et attribution, le reste, NSGA-II et Sobol en dernier.
# La derive passe des que son jeu est la (apres les baselines, sinon en fin de chaine).
etape final               "${S[@]}" --mode final
etape baselines           "${S[@]}" --mode baselines
etape analyse_raccourci   python analyse_raccourci.py --out "$OUT/raccourci.json"
etape raccourci_emetteur  "${S[@]}" --mode raccourci --variante emetteur
etape raccourci_tous      "${S[@]}" --mode raccourci --variante tous

# Etapes independantes en files paralleles (demande de Miguel : charge maximale). 11/11 : trois files ; 12/11 :
# quatre (multkan n'attend plus la derive, qui n'occupait plus que 5 coeurs). Coeurs : derive 1 thread x runs
# restants ; preproc et multkan 2 threads x 20 runs ; attribution 8 ; recherche 10.
# La latence n'est pas mesuree pendant (autres calculs en cours) : une seule mesure a la fin, machine au repos.
export MKAN_LATENCE_DIFFEREE=1
P2=(python scheduler.py --workers "$W" --threads 2 --finalists "$F")
RECH=(env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKAN_SENSI_FILS=7 MKAN_LATENCY_THREADS="$(python -c 'import config; print(config.LATENCY_THREADS)')"
      MKAN_LATENCY_CORES="$(python -c 'import config; print(config.LATENCY_CORES)')"
      "${RECHERCHE[@]:2}")   # memes arguments, 1 thread par processus ; latence pre-mesuree a config.LATENCY_THREADS fils
RECH=("${RECH[@]/#30/8}")   # 15/11 : sensibilite a 8 processus x 7 fils (memoire : ~26 Go par processus) ; principal a 1 fil (--eval_workers 1 sur la ligne Sobol)
( derive ) &
( etape multkan "${P2[@]}" --mode multkan ) &
( etape preproc "${P2[@]}" --mode preproc ) &
( etape r2_attribution python r2_attribution.py --workers 8
  etape_plafonnee nsga2 "${RECH[@]}" --skip-sensitivity
  # Recherche refaite avec les reglages de la decision (13/11, decision de Miguel transmise par la session memoire) :
  # 500 ms, penalite plafonnee appliquee, R2 0,80, grille recalee, latence a 4 fils. Dossier separe : les sorties
  # du 12/11 (recherche/) restent intactes. Reprise automatique depuis recherche_v2/search_checkpoint.json.
  mkdir -p "$OUT/recherche_v2"
  etape_plafonnee nsga2_v2 "${RECH[@]/#$OUT\/recherche/$OUT/recherche_v2}" --skip-sensitivity
  # Morris puis Sobol N=32 sur la recherche refaite (13/11, accord de Miguel : ~2 100 evaluations, arret a 160 $).
  etape_plafonnee sobol_v2 "${RECH[@]/#$OUT\/recherche/$OUT/recherche_v2}" --sensitivity_N 16 --eval_workers 1   # 15/11 : N=16 (decision de Miguel, cout)
  # Sobol SUSPENDU (13/11, accord de Miguel) : N=1024 donnait 51 200 entrainements, hors budget. A relancer
  # seulement avec un N chiffre par verifier_protocole.py et accepte par Miguel (point de reprise conserve).
  # etape_plafonnee sobol "${RECH[@]}"
  ) &
wait
[ -f "$OUT/STOP" ] && { echo "=== $(date -u) arret pendant les files paralleles"; exit 3; }
unset MKAN_LATENCE_DIFFEREE
echo "=== $(date -u) latence (machine au repos) et attribution des nouveaux runs"
python -c "import scheduler; scheduler.latency_pass('$OUT')"
python r2_attribution.py --workers 32          # reprise : seuls les runs sans attribution sont traites
echo "=== $(date -u) chaine parcourue ; echecs eventuels : $M/echecs.log"
