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
  if [ "$2" = 0 ]; then
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
           --eval_workers 30 --sensitivity_max_workers 30)
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
derive
etape analyse_raccourci   python analyse_raccourci.py --out "$OUT/raccourci.json"
etape raccourci_emetteur  "${S[@]}" --mode raccourci --variante emetteur
etape raccourci_tous      "${S[@]}" --mode raccourci --variante tous
etape preproc             "${S[@]}" --mode preproc
etape multkan             "${S[@]}" --mode multkan
etape r2_attribution      python r2_attribution.py --workers 16
etape_plafonnee nsga2     "${RECHERCHE[@]}" --skip-sensitivity
etape_plafonnee sobol     "${RECHERCHE[@]}"
derive
echo "=== $(date -u) chaine parcourue ; echecs eventuels : $M/echecs.log"
