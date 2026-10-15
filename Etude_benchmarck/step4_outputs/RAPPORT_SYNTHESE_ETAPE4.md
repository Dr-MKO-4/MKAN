# Rapport de synthese -- Etape 4 (Niveau 1)

Genere le 2026-10-11T07:45:07.975130+00:00 -- 65 runs termines, 14 architectures x regime, signature a641e9e5d9c65a87.
Parametres : {"epochs": 10, "n_seeds": 5, "device": "mps", "strategy": "raw_first", "n_windows_train": 50000, "n_windows_val": null, "run_signature": "a641e9e5d9c65a87", "top_n_engineered": 3}

## Meilleure architecture heterogene (a ce stade)

- TKAN [raw] relukan/efficientkan/wavkan_dog/linear
- Fitness moyenne : 0.7258 +/- 0.0059 (5 graine(s))
- MCC moyen : 0.9735 -- PR-AUC moyen : 0.9984 -- Brier moyen : 0.0093

## Comparaison au MKAN uniforme (hybrid sur toutes les portes)

- Baseline pas encore calculee.

## Classement (architecture x regime)

| Rang | Architecture | n | Fitness moy. | Ecart-type | MCC | PR-AUC | Brier | R2 symb. |
|---|---|---|---|---|---|---|---|---|
| 1 | TKAN [raw] relukan/efficientkan/wavkan_dog/linear | 5 | 0.7258 | 0.0059 | 0.9735 | 0.9984 | 0.0093 | 0.7600 |
| 2 | TKAN [raw] relukan/wavkan_dog/wavkan_dog/linear | 5 | 0.7068 | 0.0029 | 0.9732 | 0.9984 | 0.0094 | 0.5450 |
| 3 | TKAN [raw] relukan/relukan/wavkan_dog/linear | 5 | 0.6993 | 0.0078 | 0.9740 | 0.9985 | 0.0091 | 0.4550 |
| 4 | TKAN [raw] relukan/efficientkan/wavkan_morlet/linear | 5 | 0.6874 | 0.0038 | 0.9738 | 0.9986 | 0.0093 | 0.3200 |
| 5 | TKAN [raw] relukan/wavkan_dog/wavkan_morlet/linear | 5 | 0.6754 | 0.0066 | 0.9713 | 0.9986 | 0.0094 | 0.1950 |
| 6 | TKAN [raw] relukan/efficientkan/wavkan_dog/wavkan_morlet | 5 | 0.6581 | 0.0052 | 0.9728 | 0.9987 | 0.0091 | 0.1350 |
| 7 | TKAN [raw] relukan/wavkan_dog/wavkan_dog/wavkan_morlet | 5 | 0.6544 | 0.0073 | 0.9748 | 0.9987 | 0.0091 | 0.1250 |
| 8 | TKAN [raw] relukan/relukan/wavkan_morlet/linear | 5 | 0.6522 | 0.0095 | 0.9706 | 0.9985 | 0.0100 | 0.1300 |
| 9 | TKAN [raw] relukan/efficientkan/wavkan_morlet/wavkan_morlet | 5 | 0.6208 | 0.0151 | 0.9712 | 0.9985 | 0.0099 | 0.0100 |
| 10 | TKAN [raw] relukan/wavkan_dog/wavkan_morlet/wavkan_morlet | 5 | 0.6014 | 0.0121 | 0.9689 | 0.9985 | 0.0104 | 0.0350 |
| 11 | TKAN [raw] relukan/relukan/wavkan_dog/wavkan_morlet | 5 | 0.5898 | 0.0104 | 0.9748 | 0.9987 | 0.0091 | 0.1350 |
| 12 | TKAN [raw] relukan/relukan/wavkan_morlet/wavkan_morlet | 5 | 0.4919 | 0.0214 | 0.9740 | 0.9985 | 0.0092 | 0.0000 |
| 13 | GRU [raw] relukan/relukan/wavkan_dog | 3 | -87.1846 | 0.4702 | 0.0508 | 0.2609 | 0.2481 | 0.4417 |
| 14 | GRU [raw] relukan/relukan/wavkan_morlet | 2 | -102.5088 | 0.0263 | -0.0039 | 0.2673 | 0.6794 | 0.0000 |

## Figures

- Figure fitness : `figures/fitness_par_architecture.html`
- Figure convergence : `figures/courbes_loss_top5.html`
- Figure precision-rappel : `figures/courbes_pr_meilleure_vs_baseline.html`
