# Decision -- Etape 4 (Niveau 1)

Genere par niveau1_decision.py a partir de step4_level1_results.json.

## Regle pre-enregistree (chapitre 4)

- Meilleure fitness moyenne : TKAN·brut DoG/B-spl/DoG/Lin = 0.7303 +/- 0.0059
- Ex aequo (ecart < max des deux ecarts-types) : 4 ; (ecart < ecart-type de la meilleure) : 4

| Architecture | Fitness | Ecart-type | Ecart a la meilleure | Parametres | Latence (ms) | MCC | R2 symb |
|---|---|---|---|---|---|---|---|
| TKAN·brut DoG/B-spl/DoG/Lin | 0.7303 | 0.0059 | 0.0000 | 36,641 | 66.9 | 0.9687 | 0.840 |
| TKAN·traité DoG/B-spl/DoG/Lin | 0.7301 | 0.0097 | 0.0002 | 36,641 | 58.0 | 0.9567 | 0.910 |
| TKAN·traité ReLU/B-spl/DoG/Lin | 0.7260 | 0.0050 | 0.0043 | 57,761 | 80.1 | 0.9663 | 0.805 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 0.7258 | 0.0059 | 0.0045 | 57,761 | 79.0 | 0.9735 | 0.760 |

**Retenue par la regle (parametres puis latence) : TKAN·traité DoG/B-spl/DoG/Lin**

## Variantes de jugement

- Retenue par l'AHP normalise (V5, latence en contrainte) : TKAN·brut ReLU/B-spl/DoG/Lin
- Front de Pareto des faisables : 8 architectures

| Architecture | V1 | V2 | V3 | V4 | V5 | V6 | MCC | R2 symb | Latence (ms) | Parametres | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TKAN·brut DoG/B-spl/DoG/Lin | 1 | 1 | 9 | 27 | 9 | 3 | 0.9687 | 0.840 | 66.9 | 36,641 | oui |
| TKAN·traité DoG/B-spl/DoG/Lin | 2 | 2 | 14 | 32 | 14 | 5 | 0.9567 | 0.910 | 58.0 | 36,641 | oui |
| TKAN·traité ReLU/B-spl/DoG/Lin | 3 | 3 | 10 | 28 | 10 | 6 | 0.9663 | 0.805 | 80.1 | 57,761 |  |
| TKAN·brut ReLU/B-spl/DoG/Lin | 4 | 4 | 4 | 13 | 1 | 4 | 0.9735 | 0.760 | 79.0 | 57,761 | oui |
| TKAN·brut DoG/ReLU/DoG/Lin | 5 | 5 | 6 | 19 | 5 | 8 | 0.9724 | 0.630 | 95.3 | 60,577 | oui |
| TKAN·brut DoG/DoG/DoG/Lin | 6 | 6 | 8 | 25 | 8 | 1 | 0.9699 | 0.620 | 84.9 | 39,457 |  |
| TKAN·traité DoG/ReLU/DoG/Lin | 7 | 8 | 11 | 29 | 11 | 10 | 0.9660 | 0.600 | 89.0 | 60,577 |  |
| TKAN·brut ReLU/DoG/DoG/Lin | 8 | 9 | 5 | 14 | 4 | 2 | 0.9732 | 0.545 | 80.7 | 60,577 |  |
| TKAN·brut ReLU/ReLU/DoG/Lin | 9 | 11 | 1 | 10 | 2 | 12 | 0.9740 | 0.455 | 95.2 | 81,697 | oui |
| TKAN·brut DoG/DoG/Morlet/Lin | 10 | 12 | 13 | 31 | 13 | 9 | 0.9618 | 0.450 | 80.6 | 43,681 |  |
| TKAN·brut ReLU/B-spl/Morlet/Lin | 11 | 13 | 3 | 12 | 3 | 14 | 0.9738 | 0.320 | 93.3 | 61,985 | oui |
| TKAN·brut DoG/B-spl/Morlet/Lin | 12 | 14 | 12 | 30 | 12 | 13 | 0.9644 | 0.355 | 79.8 | 40,865 |  |
| TKAN·brut ReLU/DoG/Morlet/Lin | 13 | 16 | 7 | 21 | 7 | 11 | 0.9713 | 0.195 | 98.6 | 64,801 | oui |
| TKAN·brut DoG/DoG/Morlet/Morlet | 14 | 15 | 15 | 17 | - | 15 | 0.9728 | 0.275 | 109.8 | 59,169 |  |
| TKAN·brut DoG/B-spl/Morlet/Morlet | 15 | 21 | 17 | 20 | - | 18 | 0.9724 | 0.165 | 112.2 | 56,353 |  |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 16 | 31 | 2 | 11 | 6 | 7 | 0.9738 | 0.000 | 23.9 | 67,617 | oui |
| TKAN·brut ReLU/B-spl/DoG/Morlet | 17 | 23 | 16 | 16 | - | 20 | 0.9728 | 0.135 | 111.7 | 73,249 |  |
| TKAN·brut DoG/B-spl/DoG/Morlet | 18 | 20 | 19 | 6 | - | 19 | 0.9748 | 0.160 | 113.7 | 52,129 |  |
| TKAN·brut ReLU/DoG/DoG/Morlet | 19 | 24 | 18 | 5 | - | 17 | 0.9748 | 0.125 | 113.5 | 76,065 |  |
| TKAN·brut DoG/ReLU/Morlet/Lin | 20 | 18 | 21 | 23 | - | 21 | 0.9709 | 0.190 | 115.7 | 64,801 |  |
| TKAN·brut ReLU/ReLU/Morlet/Lin | 21 | 25 | 20 | 24 | - | 22 | 0.9706 | 0.130 | 113.9 | 85,921 |  |
| TKAN·brut DoG/DoG/DoG/Morlet | 22 | 19 | 22 | 8 | - | 16 | 0.9745 | 0.170 | 110.6 | 54,945 |  |
| TKAN·brut ReLU/B-spl/Morlet/Morlet | 23 | 32 | 23 | 22 | - | 23 | 0.9712 | 0.010 | 120.5 | 77,473 |  |
| TKAN·brut ReLU/DoG/Morlet/Morlet | 24 | 29 | 24 | 26 | - | 24 | 0.9689 | 0.035 | 125.7 | 80,289 |  |
| TKAN·brut ReLU/ReLU/DoG/Morlet | 25 | 22 | 25 | 4 | - | 25 | 0.9748 | 0.135 | 130.5 | 97,185 |  |
| TKAN·brut DoG/ReLU/DoG/Morlet | 26 | 17 | 26 | 3 | - | 26 | 0.9751 | 0.175 | 132.4 | 76,065 |  |
| TKAN·brut ReLU/ReLU/Morlet/Morlet | 27 | 30 | 27 | 9 | - | 27 | 0.9740 | 0.000 | 143.7 | 101,409 |  |
| TKAN·brut DoG/ReLU/Morlet/Morlet | 28 | 26 | 28 | 18 | - | 28 | 0.9730 | 0.100 | 148.5 | 80,289 |  |
| GRU·brut DoG/B-spl/DoG | 29 | 10 | 29 | 1 | - | 29 | 0.9775 | 0.515 | 400.1 | 599,875 |  |
| GRU·brut DoG/B-spl/Morlet | 30 | 27 | 30 | 7 | - | 30 | 0.9737 | 0.075 | 471.2 | 671,842 |  |
| GRU·brut DoG/DoG/DoG | 31 | 7 | 31 | 2 | - | 31 | 0.9767 | 0.550 | 481.5 | 647,853 |  |
| GRU·brut DoG/DoG/Morlet | 32 | 28 | 32 | 15 | - | 32 | 0.9720 | 0.025 | 602.9 | 719,820 |  |
| GRU·brut ReLU/B-spl/DoG | 33 | 33 | 33 | 33 | - | 33 | 0.7760 | 0.465 | 672.1 | 959,710 |  |
| GRU·brut ReLU/B-spl/Morlet | 34 | 36 | 34 | 36 | - | 34 | 0.0323 | 0.000 | 751.2 | 1,031,677 |  |
| GRU·brut ReLU/DoG/DoG | 35 | 34 | 35 | 34 | - | 35 | 0.7255 | 0.430 | 793.7 | 1,007,688 |  |
| GRU·brut DoG/ReLU/DoG | 36 | 35 | 36 | 35 | - | 36 | 0.4283 | 0.485 | 797.8 | 1,007,688 |  |
| GRU·brut ReLU/DoG/Morlet | 37 | 37 | 37 | 37 | - | 37 | -0.0308 | 0.110 | 894.7 | 1,079,655 |  |
| GRU·brut DoG/ReLU/Morlet | 38 | 39 | 38 | 39 | - | 38 | -0.0139 | 0.185 | 895.8 | 1,079,655 |  |
| GRU·brut ReLU/ReLU/DoG | 39 | 38 | 39 | 38 | - | 39 | 0.0613 | 0.445 | 1094.4 | 1,367,523 |  |
| GRU·brut ReLU/ReLU/Morlet | 40 | 40 | 40 | 40 | - | 40 | 0.0119 | 0.000 | 1176.8 | 1,439,490 |  |

## Poids AHP et influence reelle (TKANCell + reference)

| Critere | Poids AHP (%) | Influence reelle (%) |
|---|---|---|
| MCC | 41.5 | 2.6 |
| PR-AUC | 25.7 | 0.3 |
| Brier | 15.3 | 0.4 |
| R²_symb | 8.8 | 27.0 |
| Pénalité de latence | 8.8 | 69.8 |

## R2 symbolique par porte (recalcule a partir des poids)

| Architecture | Graine | Porte | Base | R2 median | Part R2>=0.99 | Arretes du top 40 | R2 estime (top 40) | R2 du JSON |
|---|---|---|---|---|---|---|---|---|
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 1 | Forget | hybrid | 0.818 | 0.00 | 0 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 1 | Input | hybrid | 0.799 | 0.00 | 3 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 1 | Candidate | hybrid | 0.825 | 0.00 | 18 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 1 | Output | hybrid | 0.812 | 0.00 | 19 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 2 | Forget | hybrid | 0.806 | 0.00 | 7 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 2 | Input | hybrid | 0.807 | 0.00 | 8 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 2 | Candidate | hybrid | 0.828 | 0.00 | 10 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 2 | Output | hybrid | 0.813 | 0.00 | 15 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 3 | Forget | hybrid | 0.813 | 0.00 | 2 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 3 | Input | hybrid | 0.808 | 0.00 | 6 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 3 | Candidate | hybrid | 0.837 | 0.00 | 18 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 3 | Output | hybrid | 0.811 | 0.00 | 14 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 4 | Forget | hybrid | 0.808 | 0.00 | 6 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 4 | Input | hybrid | 0.806 | 0.00 | 5 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 4 | Candidate | hybrid | 0.832 | 0.00 | 21 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 4 | Output | hybrid | 0.820 | 0.00 | 8 | 0.000 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 5 | Forget | hybrid | 0.807 | 0.00 | 3 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 5 | Input | hybrid | 0.809 | 0.00 | 4 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 5 | Candidate | hybrid | 0.832 | 0.00 | 27 | 0.025 | 0.000 |
| Uniforme TKAN·brut Hyb/Hyb/Hyb/Hyb | 5 | Output | hybrid | 0.804 | 0.00 | 6 | 0.025 | 0.000 |
| TKAN·traité DoG/B-spl/DoG/Lin | 1 | Forget | wavkan_dog | 0.985 | 0.41 | 2 | 1.000 | 0.975 |
| TKAN·traité DoG/B-spl/DoG/Lin | 1 | Input | efficientkan | 0.921 | 0.16 | 0 | 1.000 | 0.975 |
| TKAN·traité DoG/B-spl/DoG/Lin | 1 | Candidate | wavkan_dog | 0.992 | 0.68 | 2 | 1.000 | 0.975 |
| TKAN·traité DoG/B-spl/DoG/Lin | 1 | Output | linear | 1.000 | 1.00 | 36 | 1.000 | 0.975 |
| TKAN·traité DoG/B-spl/DoG/Lin | 2 | Forget | wavkan_dog | 0.985 | 0.41 | 3 | 1.000 | 0.875 |
| TKAN·traité DoG/B-spl/DoG/Lin | 2 | Input | efficientkan | 0.923 | 0.17 | 0 | 1.000 | 0.875 |
| TKAN·traité DoG/B-spl/DoG/Lin | 2 | Candidate | wavkan_dog | 0.992 | 0.67 | 2 | 1.000 | 0.875 |
| TKAN·traité DoG/B-spl/DoG/Lin | 2 | Output | linear | 1.000 | 1.00 | 35 | 1.000 | 0.875 |
| TKAN·traité DoG/B-spl/DoG/Lin | 3 | Forget | wavkan_dog | 0.981 | 0.37 | 4 | 0.975 | 0.800 |
| TKAN·traité DoG/B-spl/DoG/Lin | 3 | Input | efficientkan | 0.922 | 0.16 | 0 | 0.975 | 0.800 |
| TKAN·traité DoG/B-spl/DoG/Lin | 3 | Candidate | wavkan_dog | 0.992 | 0.68 | 1 | 0.975 | 0.800 |
| TKAN·traité DoG/B-spl/DoG/Lin | 3 | Output | linear | 1.000 | 1.00 | 35 | 0.975 | 0.800 |
| TKAN·traité DoG/B-spl/DoG/Lin | 4 | Forget | wavkan_dog | 0.981 | 0.38 | 5 | 1.000 | 1.000 |
| TKAN·traité DoG/B-spl/DoG/Lin | 4 | Input | efficientkan | 0.925 | 0.17 | 0 | 1.000 | 1.000 |
| TKAN·traité DoG/B-spl/DoG/Lin | 4 | Candidate | wavkan_dog | 0.992 | 0.68 | 2 | 1.000 | 1.000 |
| TKAN·traité DoG/B-spl/DoG/Lin | 4 | Output | linear | 1.000 | 1.00 | 33 | 1.000 | 1.000 |
| TKAN·traité DoG/B-spl/DoG/Lin | 5 | Forget | wavkan_dog | 0.975 | 0.35 | 5 | 1.000 | 0.900 |
| TKAN·traité DoG/B-spl/DoG/Lin | 5 | Input | efficientkan | 0.926 | 0.16 | 0 | 1.000 | 0.900 |
| TKAN·traité DoG/B-spl/DoG/Lin | 5 | Candidate | wavkan_dog | 0.992 | 0.66 | 1 | 1.000 | 0.900 |
| TKAN·traité DoG/B-spl/DoG/Lin | 5 | Output | linear | 1.000 | 1.00 | 34 | 1.000 | 0.900 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 1 | Forget | relukan | 0.362 | 0.00 | 0 | 0.950 | 0.800 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 1 | Input | efficientkan | 0.906 | 0.12 | 0 | 0.950 | 0.800 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 1 | Candidate | wavkan_dog | 0.992 | 0.63 | 3 | 0.950 | 0.800 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 1 | Output | linear | 1.000 | 1.00 | 37 | 0.950 | 0.800 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 2 | Forget | relukan | 0.366 | 0.00 | 0 | 1.000 | 0.675 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 2 | Input | efficientkan | 0.906 | 0.13 | 0 | 1.000 | 0.675 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 2 | Candidate | wavkan_dog | 0.991 | 0.59 | 0 | 1.000 | 0.675 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 2 | Output | linear | 1.000 | 1.00 | 40 | 1.000 | 0.675 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 3 | Forget | relukan | 0.358 | 0.00 | 0 | 1.000 | 0.725 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 3 | Input | efficientkan | 0.912 | 0.13 | 0 | 1.000 | 0.725 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 3 | Candidate | wavkan_dog | 0.992 | 0.64 | 2 | 1.000 | 0.725 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 3 | Output | linear | 1.000 | 1.00 | 38 | 1.000 | 0.725 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 4 | Forget | relukan | 0.367 | 0.00 | 0 | 0.975 | 0.750 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 4 | Input | efficientkan | 0.906 | 0.11 | 0 | 0.975 | 0.750 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 4 | Candidate | wavkan_dog | 0.992 | 0.65 | 3 | 0.975 | 0.750 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 4 | Output | linear | 1.000 | 1.00 | 37 | 0.975 | 0.750 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 5 | Forget | relukan | 0.356 | 0.00 | 0 | 0.975 | 0.850 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 5 | Input | efficientkan | 0.913 | 0.12 | 0 | 0.975 | 0.850 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 5 | Candidate | wavkan_dog | 0.992 | 0.61 | 1 | 0.975 | 0.850 |
| TKAN·brut ReLU/B-spl/DoG/Lin | 5 | Output | linear | 1.000 | 1.00 | 39 | 0.975 | 0.850 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 1 | Forget | wavkan_dog | 0.968 | 0.20 | 9 | 0.200 | 0.125 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 1 | Input | efficientkan | 0.912 | 0.12 | 2 | 0.200 | 0.125 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 1 | Candidate | wavkan_dog | 0.992 | 0.59 | 4 | 0.200 | 0.125 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 1 | Output | wavkan_morlet | 0.447 | 0.00 | 25 | 0.200 | 0.125 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 2 | Forget | wavkan_dog | 0.973 | 0.28 | 8 | 0.150 | 0.150 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 2 | Input | efficientkan | 0.914 | 0.12 | 1 | 0.150 | 0.150 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 2 | Candidate | wavkan_dog | 0.992 | 0.60 | 2 | 0.150 | 0.150 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 2 | Output | wavkan_morlet | 0.448 | 0.00 | 29 | 0.150 | 0.150 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 3 | Forget | wavkan_dog | 0.971 | 0.24 | 8 | 0.125 | 0.050 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 3 | Input | efficientkan | 0.914 | 0.12 | 0 | 0.125 | 0.050 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 3 | Candidate | wavkan_dog | 0.992 | 0.62 | 2 | 0.125 | 0.050 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 3 | Output | wavkan_morlet | 0.453 | 0.00 | 30 | 0.125 | 0.050 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 4 | Forget | wavkan_dog | 0.973 | 0.25 | 9 | 0.200 | 0.175 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 4 | Input | efficientkan | 0.910 | 0.11 | 1 | 0.200 | 0.175 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 4 | Candidate | wavkan_dog | 0.991 | 0.59 | 6 | 0.200 | 0.175 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 4 | Output | wavkan_morlet | 0.437 | 0.00 | 24 | 0.200 | 0.175 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 5 | Forget | wavkan_dog | 0.972 | 0.26 | 7 | 0.450 | 0.300 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 5 | Input | efficientkan | 0.905 | 0.10 | 4 | 0.450 | 0.300 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 5 | Candidate | wavkan_dog | 0.992 | 0.60 | 11 | 0.450 | 0.300 |
| TKAN·brut DoG/B-spl/DoG/Morlet | 5 | Output | wavkan_morlet | 0.438 | 0.00 | 18 | 0.450 | 0.300 |
