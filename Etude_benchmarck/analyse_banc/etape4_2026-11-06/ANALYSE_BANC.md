# Analyse du banc d'essai : etape4_2026-11-06

Provisoire : validation seule, le jeu de test n'est jamais lu. Tableau généré par `campagne_figures.py`.

## État des données

- Runs lus : 200 ; par phase : {'etape4_brut': 180, 'etape4_traite': 15, 'etape4_reference': 5} ; hors banc d'essai : aucun.
- Latences : complètes.
- Architectures sans toutes leurs graines : aucune.
- Threads par run (status.json) : [1, 4] ; 0 runs sans status.json copié.
- Fenêtres de validation par cellule : {'GRU': [1931357], 'TKAN': [1337759]} : les cellules ne sont pas évaluées sur les mêmes fenêtres.

## Classement à la fitness au seuil 0,80

| Rang | Architecture | n | MCC | perf | R² 0,80 | R² 0,99 | Latence (ms) | Param. | Fitness | Rang 0,99 | Rang perf | Rang V5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GRU DoG/DoG/DoG (traité) | 5 | 0,9725 ± 0,0016 | 0,6588 | 0,925 | 0,105 | 101,6 | 176 251 | 0,7398 ± 0,0019 | 22 | 3 | 2 |
| 2 | TKAN DoG/B-spl/DoG/Lin (traité) | 5 | 0,9631 ± 0,0008 | 0,6540 | 0,920 | 0,505 | 122,3 | 126 529 | 0,7346 ± 0,0025 | 5 | 4 | 4 |
| 3 | GRU DoG/ReLU/DoG (traité) | 5 | 0,9745 ± 0,0013 | 0,6598 | 0,820 | 0,095 | 122,2 | 274 126 | 0,7317 ± 0,0059 | 20 | 1 | 1 |
| 4 | TKAN DoG/B-spl/DoG/Lin | 5 | 0,9438 ± 0,0068 | 0,6445 | 0,920 | 0,750 | 113,4 | 126 529 | 0,7251 ± 0,0032 | 1 | 38 | 38 |
| 5 | GRU DoG/DoG/DoG | 5 | 0,9429 ± 0,0077 | 0,6441 | 0,915 | 0,295 | 92,4 | 176 251 | 0,7243 ± 0,0059 | 17 | 39 | 39 |
| 6 | GRU DoG/ReLU/DoG | 5 | 0,9489 ± 0,0042 | 0,6472 | 0,870 | 0,460 | 106,5 | 274 126 | 0,7234 ± 0,0085 | 6 | 29 | 13 |
| 7 | GRU DoG/B-spl/DoG | 5 | 0,9449 ± 0,0056 | 0,6451 | 0,875 | 0,275 | 70,8 | 163 201 | 0,7218 ± 0,0043 | 19 | 37 | 36 |
| 8 | TKAN DoG/DoG/DoG/Lin | 5 | 0,9421 ± 0,0062 | 0,6435 | 0,830 | 0,685 | 139,1 | 136 257 | 0,7162 ± 0,0036 | 2 | 40 | 40 |
| 9 | TKAN ReLU/B-spl/DoG/Lin | 5 | 0,9502 ± 0,0008 | 0,6478 | 0,770 | 0,635 | 134,6 | 199 489 | 0,7153 ± 0,0108 | 3 | 19 | 7 |
| 10 | TKAN ReLU/DoG/DoG/Lin | 5 | 0,9504 ± 0,0016 | 0,6479 | 0,750 | 0,580 | 164,4 | 209 217 | 0,7136 ± 0,0070 | 4 | 18 | 8 |
| 11 | GRU ReLU/B-spl/DoG | 5 | 0,9496 ± 0,0026 | 0,6475 | 0,655 | 0,270 | 86,5 | 261 076 | 0,7049 ± 0,0051 | 13 | 25 | 19 |
| 12 | Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,9727 ± 0,0008 | 0,6589 | 0,525 | 0,065 | 36,6 | 234 976 | 0,7049 ± 0,0090 | 28 | 2 | 3 |
| 13 | GRU ReLU/ReLU/DoG | 5 | 0,9503 ± 0,0027 | 0,6478 | 0,625 | 0,245 | 119,9 | 372 001 | 0,7025 ± 0,0061 | 18 | 21 | 16 |
| 14 | GRU ReLU/DoG/DoG | 5 | 0,9513 ± 0,0008 | 0,6483 | 0,585 | 0,275 | 108,5 | 274 126 | 0,6995 ± 0,0053 | 11 | 13 | 11 |
| 15 | TKAN DoG/DoG/Morlet/Lin | 5 | 0,9499 ± 0,0013 | 0,6477 | 0,590 | 0,260 | 176,4 | 150 849 | 0,6993 ± 0,0050 | 15 | 23 | 17 |
| 16 | TKAN DoG/ReLU/DoG/Lin | 5 | 0,9511 ± 0,0023 | 0,6483 | 0,455 | 0,375 | 160,5 | 209 217 | 0,6882 ± 0,0065 | 7 | 12 | 9 |
| 17 | TKAN DoG/DoG/Morlet/Morlet | 5 | 0,9472 ± 0,0030 | 0,6463 | 0,475 | 0,160 | 257,3 | 204 353 | 0,6879 ± 0,0047 | 31 | 36 | 35 |
| 18 | GRU DoG/DoG/Morlet | 5 | 0,9477 ± 0,0034 | 0,6467 | 0,460 | 0,330 | 118,4 | 195 826 | 0,6870 ± 0,0060 | 10 | 32 | 29 |
| 19 | TKAN DoG/B-spl/Morlet/Lin | 5 | 0,9476 ± 0,0017 | 0,6466 | 0,435 | 0,245 | 150,5 | 141 121 | 0,6847 ± 0,0069 | 21 | 33 | 32 |
| 20 | GRU DoG/ReLU/Morlet | 5 | 0,9485 ± 0,0052 | 0,6470 | 0,425 | 0,330 | 130,2 | 293 701 | 0,6843 ± 0,0067 | 9 | 30 | 26 |
| 21 | TKAN DoG/DoG/DoG/Morlet | 5 | 0,9493 ± 0,0022 | 0,6473 | 0,420 | 0,270 | 223,3 | 189 761 | 0,6841 ± 0,0063 | 14 | 27 | 27 |
| 22 | TKAN ReLU/B-spl/DoG/Morlet | 5 | 0,9510 ± 0,0009 | 0,6482 | 0,385 | 0,270 | 216,5 | 252 993 | 0,6819 ± 0,0037 | 12 | 15 | 15 |
| 23 | TKAN ReLU/ReLU/DoG/Lin | 5 | 0,9515 ± 0,0030 | 0,6485 | 0,380 | 0,355 | 186,6 | 282 177 | 0,6818 ± 0,0052 | 8 | 9 | 10 |
| 24 | TKAN DoG/B-spl/Morlet/Morlet | 5 | 0,9490 ± 0,0010 | 0,6472 | 0,390 | 0,200 | 231,5 | 194 625 | 0,6814 ± 0,0021 | 27 | 28 | 28 |
| 25 | TKAN ReLU/DoG/Morlet/Lin | 5 | 0,9494 ± 0,0021 | 0,6475 | 0,345 | 0,170 | 197,0 | 223 809 | 0,6778 ± 0,0078 | 30 | 24 | 23 |
| 26 | TKAN ReLU/DoG/DoG/Morlet | 5 | 0,9504 ± 0,0014 | 0,6479 | 0,340 | 0,215 | 246,8 | 262 721 | 0,6777 ± 0,0034 | 24 | 17 | 20 |
| 27 | GRU DoG/B-spl/Morlet | 5 | 0,9473 ± 0,0027 | 0,6465 | 0,340 | 0,270 | 95,3 | 182 776 | 0,6763 ± 0,0057 | 16 | 34 | 33 |
| 28 | TKAN DoG/B-spl/DoG/Morlet | 5 | 0,9486 ± 0,0022 | 0,6470 | 0,315 | 0,230 | 195,5 | 180 033 | 0,6746 ± 0,0050 | 23 | 31 | 31 |
| 29 | TKAN ReLU/B-spl/Morlet/Lin | 5 | 0,9520 ± 0,0013 | 0,6487 | 0,285 | 0,190 | 170,7 | 214 081 | 0,6737 ± 0,0111 | 26 | 6 | 6 |
| 30 | TKAN ReLU/DoG/Morlet/Morlet | 5 | 0,9515 ± 0,0019 | 0,6484 | 0,265 | 0,075 | 283,0 | 277 313 | 0,6717 ± 0,0076 | 33 | 10 | 14 |
| 31 | TKAN DoG/ReLU/DoG/Morlet | 5 | 0,9519 ± 0,0013 | 0,6487 | 0,240 | 0,205 | 246,8 | 262 721 | 0,6697 ± 0,0077 | 25 | 7 | 12 |
| 32 | TKAN DoG/ReLU/Morlet/Morlet | 5 | 0,9503 ± 0,0020 | 0,6478 | 0,195 | 0,070 | 278,0 | 277 313 | 0,6649 ± 0,0070 | 34 | 20 | 25 |
| 33 | TKAN DoG/ReLU/Morlet/Lin | 5 | 0,9510 ± 0,0018 | 0,6481 | 0,185 | 0,080 | 197,2 | 223 809 | 0,6643 ± 0,0053 | 32 | 16 | 24 |
| 34 | TKAN ReLU/ReLU/DoG/Morlet | 5 | 0,9527 ± 0,0007 | 0,6491 | 0,165 | 0,155 | 267,7 | 335 681 | 0,6635 ± 0,0018 | 29 | 5 | 5 |
| 35 | TKAN ReLU/B-spl/Morlet/Morlet | 5 | 0,9511 ± 0,0018 | 0,6483 | 0,105 | 0,060 | 256,0 | 267 585 | 0,6575 ± 0,0070 | 35 | 14 | 21 |
| 36 | GRU ReLU/DoG/Morlet | 5 | 0,9475 ± 0,0022 | 0,6463 | 0,070 | 0,040 | 130,2 | 293 701 | 0,6525 ± 0,0048 | 37 | 35 | 37 |
| 37 | TKAN ReLU/ReLU/Morlet/Lin | 5 | 0,9514 ± 0,0007 | 0,6485 | 0,040 | 0,040 | 217,0 | 296 769 | 0,6520 ± 0,0023 | 36 | 8 | 18 |
| 38 | TKAN ReLU/ReLU/Morlet/Morlet | 5 | 0,9514 ± 0,0025 | 0,6484 | 0,000 | 0,000 | 298,9 | 350 273 | 0,6484 ± 0,0012 | 38 | 11 | 22 |
| 39 | GRU ReLU/ReLU/Morlet | 5 | 0,9502 ± 0,0012 | 0,6478 | 0,000 | 0,000 | 144,8 | 391 576 | 0,6478 ± 0,0006 | 39 | 22 | 30 |
| 40 | GRU ReLU/B-spl/Morlet | 5 | 0,9495 ± 0,0011 | 0,6474 | 0,000 | 0,000 | 108,9 | 280 651 | 0,6474 ± 0,0006 | 40 | 26 | 34 |

## Règle préenregistrée

- Meilleure moyenne : GRU DoG/DoG/DoG (traité).
- Ex aequo : ['GRU DoG/DoG/DoG (traité)'].
- Retenue au nombre de paramètres puis à la latence : GRU DoG/DoG/DoG (traité).

## Application provisoire de la règle des finalistes

- GRU DoG/DoG/DoG (traité) : fitness, symbolisation.
- GRU DoG/ReLU/DoG (traité) : detection.
- Uniforme GRU Hyb/Hyb/Hyb (traité) : rapidite.
- TKAN ReLU/B-spl/DoG/Lin : B conservee.

## Influence effective

| Seuil | Critère | Poids (%) | Étendue | Influence (%) |
|---|---|---|---|---|
| 0,80 | MCC | 41,5 | 0,0324 | 13,8 |
| 0,80 | PR-AUC | 25,7 | 0,0065 | 1,7 |
| 0,80 | Brier | 15,3 | 0,0076 | 1,2 |
| 0,80 | R2 symbolique | 8,8 | 0,9250 | 83,3 |
| 0,80 | Pénalité de latence | 8,8 | 0,0000 | 0,0 |
| 0,99 | MCC | 41,5 | 0,0324 | 16,4 |
| 0,99 | PR-AUC | 25,7 | 0,0065 | 2,0 |
| 0,99 | Brier | 15,3 | 0,0076 | 1,4 |
| 0,99 | R2 symbolique | 8,8 | 0,7500 | 80,1 |
| 0,99 | Pénalité de latence | 8,8 | 0,0000 | 0,0 |

## Corrélations entre classements

- F80/perf : Spearman -0,121, Kendall -0,108.
- F80/r80 : Spearman 0,994, Kendall 0,961.
- F99/perf : Spearman -0,193, Kendall -0,115.
- F99/r99 : Spearman 0,966, Kendall 0,888.
- F80/F99 : Spearman 0,761, Kendall 0,587.
- F80/V5_80 : Spearman 0,202, Kendall 0,144.
- V5_80/perf : Spearman 0,889, Kendall 0,749.
- Paramètres et perf, TKAN : Spearman 0,805.
- Paramètres et perf, GRU : Spearman 0,618.

## Différences appariées (test t indicatif, n petit)

| Critère | Comparaison | n | Écart | IC 95 % | p | p Holm |
|---|---|---|---|---|---|---|
| MCC | GRU DoG/ReLU/DoG (traité) − GRU DoG/DoG/DoG (traité) | 5 | 0,0020 | ± 0,0022 | 0,066 | 0,197 |
| MCC | Uniforme GRU Hyb/Hyb/Hyb (traité) − GRU DoG/DoG/DoG (traité) | 5 | 0,0002 | ± 0,0021 | 0,808 | 0,808 |
| MCC | TKAN ReLU/B-spl/DoG/Lin − GRU DoG/DoG/DoG (traité) | 5 | -0,0223 | ± 0,0025 | < 0,001 | < 0,001 |
| MCC | TKAN DoG/B-spl/DoG/Lin (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,0094 | ± 0,0017 | < 0,001 | < 0,001 |
| MCC | GRU DoG/ReLU/DoG (traité) − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,0018 | ± 0,0023 | 0,099 | 0,198 |
| MCC | TKAN ReLU/B-spl/DoG/Lin − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | -0,0225 | ± 0,0013 | < 0,001 | < 0,001 |
| perf | GRU DoG/ReLU/DoG (traité) − GRU DoG/DoG/DoG (traité) | 5 | 0,0011 | ± 0,0010 | 0,048 | 0,145 |
| perf | Uniforme GRU Hyb/Hyb/Hyb (traité) − GRU DoG/DoG/DoG (traité) | 5 | 0,0001 | ± 0,0010 | 0,820 | 0,820 |
| perf | TKAN ReLU/B-spl/DoG/Lin − GRU DoG/DoG/DoG (traité) | 5 | -0,0109 | ± 0,0012 | < 0,001 | < 0,001 |
| perf | TKAN DoG/B-spl/DoG/Lin (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,0048 | ± 0,0009 | < 0,001 | < 0,001 |
| perf | GRU DoG/ReLU/DoG (traité) − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,0010 | ± 0,0012 | 0,082 | 0,164 |
| perf | TKAN ReLU/B-spl/DoG/Lin − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | -0,0110 | ± 0,0007 | < 0,001 | < 0,001 |
| R2 à 0,80 | GRU DoG/ReLU/DoG (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,1050 | ± 0,1015 | 0,045 | 0,142 |
| R2 à 0,80 | Uniforme GRU Hyb/Hyb/Hyb (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,4000 | ± 0,1536 | 0,002 | 0,010 |
| R2 à 0,80 | TKAN ReLU/B-spl/DoG/Lin − GRU DoG/DoG/DoG (traité) | 5 | -0,1550 | ± 0,1604 | 0,055 | 0,142 |
| R2 à 0,80 | TKAN DoG/B-spl/DoG/Lin (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,0050 | ± 0,0340 | 0,704 | 0,704 |
| R2 à 0,80 | GRU DoG/ReLU/DoG (traité) − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,2950 | ± 0,0773 | < 0,001 | 0,003 |
| R2 à 0,80 | TKAN ReLU/B-spl/DoG/Lin − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,2450 | ± 0,2177 | 0,035 | 0,142 |
| Fitness à 0,80 | GRU DoG/ReLU/DoG (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,0081 | ± 0,0082 | 0,051 | 0,103 |
| Fitness à 0,80 | Uniforme GRU Hyb/Hyb/Hyb (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,0349 | ± 0,0131 | 0,002 | 0,009 |
| Fitness à 0,80 | TKAN ReLU/B-spl/DoG/Lin − GRU DoG/DoG/DoG (traité) | 5 | -0,0245 | ± 0,0143 | 0,009 | 0,035 |
| Fitness à 0,80 | TKAN DoG/B-spl/DoG/Lin (traité) − GRU DoG/DoG/DoG (traité) | 5 | -0,0052 | ± 0,0032 | 0,011 | 0,035 |
| Fitness à 0,80 | GRU DoG/ReLU/DoG (traité) − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,0268 | ± 0,0074 | < 0,001 | 0,003 |
| Fitness à 0,80 | TKAN ReLU/B-spl/DoG/Lin − Uniforme GRU Hyb/Hyb/Hyb (traité) | 5 | 0,0104 | ± 0,0189 | 0,200 | 0,200 |

## Effets marginaux par porte (régime brut)

| Cellule | Porte | Base | n | MCC | perf | R² 0,80 | R² 0,99 | Latence |
|---|---|---|---|---|---|---|---|---|
| TKAN | Forget | ReLU-KAN | 60 | 0,9511 | 0,6483 | 0,319 | 0,229 | 219,9 |
| TKAN | Forget | DoG | 60 | 0,9485 | 0,6469 | 0,454 | 0,294 | 197,5 |
| TKAN | Input | B-spline | 40 | 0,9491 | 0,6473 | 0,451 | 0,322 | 183,6 |
| TKAN | Input | ReLU-KAN | 40 | 0,9514 | 0,6484 | 0,207 | 0,160 | 231,6 |
| TKAN | Input | DoG | 40 | 0,9488 | 0,6471 | 0,502 | 0,302 | 210,9 |
| TKAN | Candidate | DoG | 60 | 0,9494 | 0,6474 | 0,497 | 0,394 | 191,3 |
| TKAN | Candidate | Morlet | 60 | 0,9502 | 0,6478 | 0,276 | 0,129 | 226,1 |
| TKAN | Output | Linéaire | 60 | 0,9492 | 0,6473 | 0,499 | 0,364 | 167,3 |
| TKAN | Output | Morlet | 60 | 0,9504 | 0,6479 | 0,275 | 0,159 | 250,1 |
| GRU | Reset | ReLU-KAN | 30 | 0,9497 | 0,6475 | 0,323 | 0,138 | 116,5 |
| GRU | Reset | DoG | 30 | 0,9467 | 0,6461 | 0,648 | 0,327 | 102,3 |
| GRU | Update | B-spline | 20 | 0,9478 | 0,6466 | 0,467 | 0,204 | 90,4 |
| GRU | Update | ReLU-KAN | 20 | 0,9495 | 0,6475 | 0,480 | 0,259 | 125,4 |
| GRU | Update | DoG | 20 | 0,9473 | 0,6464 | 0,508 | 0,235 | 112,4 |
| GRU | Candidate | DoG | 30 | 0,9480 | 0,6467 | 0,754 | 0,303 | 97,4 |
| GRU | Candidate | Morlet | 30 | 0,9485 | 0,6470 | 0,216 | 0,162 | 121,3 |

## Stabilité et arrêt

- correlation_rangs_perf_lstm : 0,4477.
- correlation_rangs_perf_gru : 0,2210.
- ecart_type_perf_forget_candidate_dog : 0,0022.
- ecart_type_perf_autres : 0,0009.
- runs_arretes_par_patience : 176.
- runs_coupes_par_plafond : 24.
- runs_a_15_epoques : 34.

## Arêtes évaluées par le R² symbolique

- Arêtes survivantes à l'élagage par run : de 649 à 14153, médiane 2860 ; le R² n'en évalue que 40.
- TKAN à sortie linéaire, arêtes de la porte Output et R² à 0,99 : Spearman 0,866.
- TKAN à sortie linéaire, arêtes de la porte Output et R² à 0,80 : Spearman 0,893.
