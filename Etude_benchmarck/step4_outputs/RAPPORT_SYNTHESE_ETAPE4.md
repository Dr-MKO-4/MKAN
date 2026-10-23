# Rapport de synthese -- Etape 4 (Niveau 1)

Genere le 2026-10-21T18:45:55.463628+00:00 -- 195 runs termines, 39 architectures x regime, signature a641e9e5d9c65a87.
Parametres : {"epochs": 10, "n_seeds": 5, "device": "mps", "strategy": "raw_first", "n_windows_train": 50000, "n_windows_val": null, "run_signature": "a641e9e5d9c65a87", "top_n_engineered": 3}

## Meilleure architecture heterogene (a ce stade)

- TKAN [raw] wavkan_dog/efficientkan/wavkan_dog/linear
- Fitness moyenne : 0.7303 +/- 0.0059 (5 graine(s))
- MCC moyen : 0.9687 -- PR-AUC moyen : 0.9978 -- Brier moyen : 0.0114

## Comparaison au MKAN uniforme (hybrid sur toutes les portes)

- Fitness moyenne baseline : 0.6593 (5 graine(s))
- Gain heterogene - uniforme : +0.0709

## Classement (architecture x regime)

| Rang | Architecture | n | Fitness moy. | Ecart-type | MCC | PR-AUC | Brier | R2 symb. |
|---|---|---|---|---|---|---|---|---|
| 1 | TKAN [raw] wavkan_dog/efficientkan/wavkan_dog/linear | 5 | 0.7303 | 0.0059 | 0.9687 | 0.9978 | 0.0114 | 0.8400 |
| 2 | TKAN [engineered] wavkan_dog/efficientkan/wavkan_dog/linear | 5 | 0.7301 | 0.0097 | 0.9567 | 0.9953 | 0.0161 | 0.9100 |
| 3 | TKAN [engineered] relukan/efficientkan/wavkan_dog/linear | 5 | 0.7260 | 0.0050 | 0.9663 | 0.9974 | 0.0123 | 0.8050 |
| 4 | TKAN [raw] relukan/efficientkan/wavkan_dog/linear | 5 | 0.7258 | 0.0059 | 0.9735 | 0.9984 | 0.0093 | 0.7600 |
| 5 | TKAN [raw] wavkan_dog/relukan/wavkan_dog/linear | 5 | 0.7138 | 0.0097 | 0.9724 | 0.9984 | 0.0097 | 0.6300 |
| 6 | TKAN [raw] wavkan_dog/wavkan_dog/wavkan_dog/linear | 5 | 0.7117 | 0.0082 | 0.9699 | 0.9980 | 0.0108 | 0.6200 |
| 7 | TKAN [engineered] wavkan_dog/relukan/wavkan_dog/linear | 5 | 0.7079 | 0.0083 | 0.9660 | 0.9976 | 0.0127 | 0.6000 |
| 8 | TKAN [raw] relukan/wavkan_dog/wavkan_dog/linear | 5 | 0.7068 | 0.0029 | 0.9732 | 0.9984 | 0.0094 | 0.5450 |
| 9 | TKAN [raw] relukan/relukan/wavkan_dog/linear | 5 | 0.6993 | 0.0078 | 0.9740 | 0.9985 | 0.0091 | 0.4550 |
| 10 | TKAN [raw] wavkan_dog/wavkan_dog/wavkan_morlet/linear | 5 | 0.6928 | 0.0091 | 0.9618 | 0.9976 | 0.0137 | 0.4500 |
| 11 | TKAN [raw] relukan/efficientkan/wavkan_morlet/linear | 5 | 0.6874 | 0.0038 | 0.9738 | 0.9986 | 0.0093 | 0.3200 |
| 12 | TKAN [raw] wavkan_dog/efficientkan/wavkan_morlet/linear | 5 | 0.6856 | 0.0072 | 0.9644 | 0.9970 | 0.0125 | 0.3550 |
| 13 | TKAN [raw] relukan/wavkan_dog/wavkan_morlet/linear | 5 | 0.6754 | 0.0066 | 0.9713 | 0.9986 | 0.0094 | 0.1950 |
| 14 | TKAN [raw] wavkan_dog/wavkan_dog/wavkan_morlet/wavkan_morlet | 5 | 0.6740 | 0.0071 | 0.9728 | 0.9985 | 0.0094 | 0.2750 |
| 15 | TKAN [raw] wavkan_dog/efficientkan/wavkan_morlet/wavkan_morlet | 5 | 0.6600 | 0.0040 | 0.9724 | 0.9984 | 0.0099 | 0.1650 |
| 16 | TKAN [raw] relukan/efficientkan/wavkan_dog/wavkan_morlet | 5 | 0.6581 | 0.0052 | 0.9728 | 0.9987 | 0.0091 | 0.1350 |
| 17 | TKAN [raw] wavkan_dog/efficientkan/wavkan_dog/wavkan_morlet | 5 | 0.6570 | 0.0122 | 0.9748 | 0.9987 | 0.0092 | 0.1600 |
| 18 | TKAN [raw] relukan/wavkan_dog/wavkan_dog/wavkan_morlet | 5 | 0.6544 | 0.0073 | 0.9748 | 0.9987 | 0.0091 | 0.1250 |
| 19 | TKAN [raw] wavkan_dog/relukan/wavkan_morlet/linear | 5 | 0.6526 | 0.0127 | 0.9709 | 0.9985 | 0.0098 | 0.1900 |
| 20 | TKAN [raw] relukan/relukan/wavkan_morlet/linear | 5 | 0.6522 | 0.0095 | 0.9706 | 0.9985 | 0.0100 | 0.1300 |
| 21 | TKAN [raw] wavkan_dog/wavkan_dog/wavkan_dog/wavkan_morlet | 5 | 0.6491 | 0.0363 | 0.9745 | 0.9987 | 0.0095 | 0.1700 |
| 22 | TKAN [raw] relukan/efficientkan/wavkan_morlet/wavkan_morlet | 5 | 0.6208 | 0.0151 | 0.9712 | 0.9985 | 0.0099 | 0.0100 |
| 23 | TKAN [raw] relukan/wavkan_dog/wavkan_morlet/wavkan_morlet | 5 | 0.6014 | 0.0121 | 0.9689 | 0.9985 | 0.0104 | 0.0350 |
| 24 | TKAN [raw] relukan/relukan/wavkan_dog/wavkan_morlet | 5 | 0.5898 | 0.0104 | 0.9748 | 0.9987 | 0.0091 | 0.1350 |
| 25 | TKAN [raw] wavkan_dog/relukan/wavkan_dog/wavkan_morlet | 5 | 0.5827 | 0.0156 | 0.9751 | 0.9986 | 0.0091 | 0.1750 |
| 26 | TKAN [raw] relukan/relukan/wavkan_morlet/wavkan_morlet | 5 | 0.4919 | 0.0214 | 0.9740 | 0.9985 | 0.0092 | 0.0000 |
| 27 | TKAN [raw] wavkan_dog/relukan/wavkan_morlet/wavkan_morlet | 5 | 0.4613 | 0.0242 | 0.9730 | 0.9983 | 0.0100 | 0.1000 |
| 28 | GRU [raw] wavkan_dog/efficientkan/wavkan_dog | 5 | -7.1821 | 0.1016 | 0.9775 | 0.9989 | 0.0063 | 0.5150 |
| 29 | GRU [raw] wavkan_dog/efficientkan/wavkan_morlet | 5 | -11.4393 | 1.4890 | 0.9737 | 0.9990 | 0.0070 | 0.0750 |
| 30 | GRU [raw] wavkan_dog/wavkan_dog/wavkan_dog | 5 | -12.0440 | 0.2385 | 0.9767 | 0.9990 | 0.0066 | 0.5500 |
| 31 | GRU [raw] wavkan_dog/wavkan_dog/wavkan_morlet | 5 | -21.4945 | 0.4335 | 0.9720 | 0.9992 | 0.0077 | 0.0250 |
| 32 | GRU [raw] relukan/efficientkan/wavkan_dog | 5 | -28.0833 | 0.7427 | 0.7760 | 0.9493 | 0.0749 | 0.4650 |
| 33 | GRU [raw] relukan/efficientkan/wavkan_morlet | 5 | -36.9723 | 0.5781 | 0.0323 | 0.9057 | 0.4227 | 0.0000 |
| 34 | GRU [raw] relukan/wavkan_dog/wavkan_dog | 5 | -41.5941 | 0.7226 | 0.7255 | 0.9511 | 0.0974 | 0.4300 |
| 35 | GRU [raw] wavkan_dog/relukan/wavkan_dog | 5 | -42.2994 | 0.5870 | 0.4283 | 0.6670 | 0.2396 | 0.4850 |
| 36 | GRU [raw] relukan/wavkan_dog/wavkan_morlet | 5 | -55.1503 | 0.5919 | -0.0308 | 0.9130 | 0.4248 | 0.1100 |
| 37 | GRU [raw] wavkan_dog/relukan/wavkan_morlet | 5 | -55.4703 | 0.8192 | -0.0139 | 0.2466 | 0.4914 | 0.1850 |
| 38 | GRU [raw] relukan/relukan/wavkan_dog | 5 | -86.5333 | 1.4992 | 0.0613 | 0.2609 | 0.2472 | 0.4450 |
| 39 | GRU [raw] relukan/relukan/wavkan_morlet | 5 | -101.5811 | 1.1012 | 0.0119 | 0.2636 | 0.5708 | 0.0000 |

## Figures

- Figure fitness : `figures/fitness_par_architecture.html`
- Figure convergence : `figures/courbes_loss_top5.html`
- Figure precision-rappel : `figures/courbes_pr_meilleure_vs_baseline.html`
