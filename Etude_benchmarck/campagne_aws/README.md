# Campagne AWS : nouvelle Étape 4 et entraînement final

Décisions de référence : `memoire/DECISIONS.md`, entrée du 2026-11-03 « Reprise de l'Étape 4 sur
AWS ». Explications chiffrées : `contextes_sessions/01_aws_notes_techniques.md`.

## Ce que fait chaque fichier

| Fichier | Rôle |
|---|---|
| `config.py` | protocole (données, tailles, latence, époques, AWS) ; valeurs « A VALIDER » à confirmer |
| `prepare_data.py` | prépare une fois les fenêtres sur disque (lues sans copie par tous les processus) |
| `train_run.py` | entraîne un run, reprise exacte, état en direct, sauvegardes |
| `scheduler.py` | répartit les runs sur les cœurs et enchaîne les phases de l'Étape 4 ou l'entraînement final |
| `measure_latency.py` | latence de tous les modèles finis, machine au repos, puis fitness |
| `report.py` | `runs.csv` et `summary.json` (classement), régénérés après chaque run |
| `sentinel.py` | copie vers S3, préavis spot, garde-fou de coût |
| `monitor/` | suivi web en direct (progression, pertes, classement, charge, coût) |
| `bench.py` | banc d'essai : taille cachée, processus × threads, durée estimée |
| `setup_instance.sh`, `launch.sh` | installation et lancement sur l'instance |

## 1. Sur le poste (une fois)

```bash
# Code : commit et push de MKAN (Dr-MKO-4/MKAN, branche main), sur demande de Miguel.
# Données du 27/10 vers S3 (environ 0,7 Go) :
aws s3 cp MOMTSIM/config/featuresLog.parquet s3://mkan-training-mko/campagne_v2/source/featuresLog.parquet
aws s3 cp data/val_features.parquet           s3://mkan-training-mko/campagne_v2/source/val_features.parquet
```

## 2. Instance

- `c7i.16xlarge` (64 vCPU, 128 Go), spot, eu-north-1, Ubuntu 24.04 x86_64, disque gp3 100 Go.
- Étiquette `Project=mkan` sur l'instance et son volume (budget filtré sur cette étiquette).
- Rôle d'instance avec : `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` sur `mkan-training-mko`,
  et `ec2:DescribeSpotPriceHistory` (garde-fou de coût). Sans ce dernier droit :
  `export MKAN_PRICE_USD_H=<prix spot>` avant `launch.sh`.
- Groupe de sécurité : SSH (22) ouvert à votre seule adresse IP. Aucun autre port.
- Spot : comportement d'interruption « stop » (le disque est conservé) ; S3 garde de toute façon
  tout ce qu'il faut pour reprendre sur une autre instance.

## 3. Sur l'instance

```bash
bash setup_instance.sh                  # code, environnement, données, restauration S3
bash launch.sh bench                    # banc d'essai (30 à 60 min)
# lire sorties/bench/bench_results.json, fixer MKAN_HIDDEN_TKAN et MKAN_N_TRAIN, puis :
export MKAN_HIDDEN_TKAN=64 MKAN_N_TRAIN=300000
bash launch.sh etape4 32 2              # nombre de processus et de threads issus du banc
```

Suivi depuis le poste : `ssh -i cle.pem -N -L 8050:127.0.0.1:8050 ubuntu@<ip>` puis
http://localhost:8050.

## 4. Interruptions et reprise

- Préavis spot ou budget atteint : la sentinelle crée `sorties/STOP` ; chaque run sauvegarde son
  état et quitte ; tout est copié vers S3.
- Arrêt volontaire : `bash launch.sh stop`.
- Reprise (même instance ou nouvelle) : `bash setup_instance.sh` (restaure depuis S3) puis la même
  commande `launch.sh`. Les runs finis sont sautés, les autres repartent de leur dernier point de
  reprise (au plus 5 minutes perdues).

## 5. Rapatriement et fin

```bash
aws s3 sync s3://mkan-training-mko/campagne_v2/sorties MKAN/Etude_benchmarck/campagne_aws/sorties
```
En fin de campagne : terminer l'instance, supprimer son volume et ses instantanés.
