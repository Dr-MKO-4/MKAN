#!/usr/bin/env bash
# setup_instance.sh -- Installation sur une instance EC2 neuve (Ubuntu 24.04, x86_64).
# A lancer une fois, en tant qu'ubuntu, depuis le dossier personnel :
#     bash setup_instance.sh
# Reprend le code depuis GitHub, l'environnement Python (torch CPU) et les
# donnees depuis S3, puis restaure les sorties deja produites (reprise).
set -euo pipefail

BUCKET="${MKAN_S3_BUCKET:-mkan-training-mko}"
PREFIX="${MKAN_S3_PREFIX:-campagne_v2}"
REPO="${MKAN_REPO:-https://github.com/Dr-MKO-4/MKAN.git}"
BRANCH="${MKAN_BRANCH:-main}"
ROOT="$HOME/Modelisation"

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-dev git tmux htop unzip curl
if ! command -v aws >/dev/null; then
  curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install
fi

mkdir -p "$ROOT/MOMTSIM/config" "$ROOT/data"
if [ ! -d "$ROOT/MKAN/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$ROOT/MKAN"
else
  git -C "$ROOT/MKAN" pull --ff-only
fi

python3 -m venv "$HOME/venv-mkan"
# shellcheck disable=SC1091
source "$HOME/venv-mkan/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "$ROOT/MKAN/Etude_benchmarck/campagne_aws/requirements.txt"

# Donnees du 27/10 (envoyees une fois depuis le poste, cf. README)
aws s3 cp "s3://$BUCKET/$PREFIX/source/featuresLog.parquet" "$ROOT/MOMTSIM/config/featuresLog.parquet"
aws s3 cp "s3://$BUCKET/$PREFIX/source/val_features.parquet" "$ROOT/data/val_features.parquet"

cd "$ROOT/MKAN/Etude_benchmarck/campagne_aws"
python sentinel.py --restore
echo "Installation terminee. Etape suivante : bash launch.sh bench   (puis : bash launch.sh etape4)"
