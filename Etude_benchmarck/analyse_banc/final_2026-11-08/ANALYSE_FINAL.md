# Analyse de l'entraînement final : final_2026-11-08

Validation seule, le jeu de test n'est jamais lu. Généré par `analyse_final.py`.

## Contrôles d'intégrité

- Dossiers de runs : 75 ; avec result.json : 75 ; par phase : {'final': 25, 'exp_baselines': 45, 'exp_raccourci_emetteur': 5}.
- Concordance runs.csv / result.json / latency.json : exacte.
- Runs arrêtés au plafond d'époques : aucun.
- Historique de perte de longueur différente du nombre d'époques (reprise incohérente) : aucun.
- Arrêt anticipé sans écart de « patience » époques entre fin et meilleure époque : aucun.
- Époques (min-max) : 11-38 ; meilleure époque 5-32.
- Threads d'entraînement (status.json) par phase : {'final': ['2'], 'exp_baselines': ['1'], 'exp_raccourci_emetteur': ['4']}.
- Fenêtres d'entraînement par phase : {'final': ['1000000'], 'exp_baselines': ['1000000'], 'exp_raccourci_emetteur': ['1000000']}.
- Threads de la mesure de latence par phase : {'final': ['4'], 'exp_baselines': ['4'], 'exp_raccourci_emetteur': ['4']}.
- Machine de la mesure par phase : {'final': ['aarch64'], 'exp_baselines': ['aarch64'], 'exp_raccourci_emetteur': ['aarch64']}.
- Dates des mesures de latence : {'final': ('2026-11-08T05:04:29', '2026-11-08T05:04:40'), 'exp_baselines': ('2026-11-08T08:29:51', '2026-11-08T08:29:52'), 'exp_raccourci_emetteur': ('2026-11-08T22:44:33', '2026-11-08T22:44:34')}.
- Fenêtres de validation par W : {5: [1931357], 10: [1337759]} : la population évaluée change avec W, on ne compare donc qu'à W égal.
- Scores invalides (n_bad_scores > 0) : aucun.
- scheduler.log : 0 mentions d'interruption, 0 de reprise.

## Finalistes (A traité exclu, décision du 09/11)

| Architecture | n | MCC | PR-AUC | Brier | perf (min-max) | R² 0,80 | R² 0,99 | Latence (ms) | Fitness | Époques | Param. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TKAN DoG/B-spl/DoG/Lin (traité), W=10 **(hors analyse)** | 5 | 0,9640 ± 0,0010 | 0,9960 | 0,0098 | 0,6545 ± 0,0004 (0,6540-0,6551) | 0,845 | 0,430 | 120,5 | 0,7286 ± 0,0036 | 19,6 | 126 529 |
| GRU DoG/DoG/DoG (traité), W=5 | 5 | 0,9745 ± 0,0024 | 0,9980 | 0,0071 | 0,6598 ± 0,0012 (0,6581-0,6610) | 0,755 | 0,055 | 99,1 | 0,7260 ± 0,0072 | 23,0 | 176 251 |
| GRU DoG/ReLU/DoG (traité), W=5 | 5 | 0,9758 ± 0,0024 | 0,9983 | 0,0067 | 0,6605 ± 0,0011 (0,6592-0,6616) | 0,670 | 0,060 | 119,4 | 0,7192 ± 0,0046 | 25,6 | 274 126 |
| TKAN ReLU/B-spl/DoG/Lin, W=10 | 5 | 0,9541 ± 0,0010 | 0,9947 | 0,0120 | 0,6498 ± 0,0005 (0,6489-0,6502) | 0,765 | 0,650 | 150,6 | 0,7168 ± 0,0061 | 18,6 | 199 489 |
| Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,9742 ± 0,0010 | 0,9979 | 0,0073 | 0,6596 ± 0,0005 (0,6589-0,6602) | 0,470 | 0,045 | 37,2 | 0,7008 ± 0,0066 | 20,8 | 234 976 |

## Décomposition de la fitness

| Architecture | perf | w·R² 0,80 | pénalité latence | Fitness |
|---|---|---|---|---|
| GRU DoG/DoG/DoG (traité), W=5 | 0,6598 | 0,0661 | -0,0000 | 0,7260 |
| GRU DoG/ReLU/DoG (traité), W=5 | 0,6605 | 0,0587 | -0,0000 | 0,7192 |
| TKAN ReLU/B-spl/DoG/Lin, W=10 | 0,6498 | 0,0670 | -0,0000 | 0,7168 |
| Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 0,6596 | 0,0412 | -0,0000 | 0,7008 |

Poids : w_R2 = 0,0876, w_lat = 0,0876.

## Front de Pareto (perf, R² 0,80, latence)

- Non dominés : ['GRU DoG/DoG/DoG (traité), W=5', 'GRU DoG/ReLU/DoG (traité), W=5', 'TKAN ReLU/B-spl/DoG/Lin, W=10', 'Uniforme GRU Hyb/Hyb/Hyb (traité), W=5'].
- Dominés : aucun.

## Modèles de référence

| Modèle | n | MCC | PR-AUC | Brier | perf | Latence (ms) |
|---|---|---|---|---|---|---|
| XGBoost (traité, W=5) | 5 | 0,9782 ± 0,0000 | 0,9986 | 0,0062 | 0,6617 ± 0,0000 | 1,53 |
| LSTM (traité, W=5) | 5 | 0,9755 ± 0,0006 | 0,9981 | 0,0071 | 0,6602 ± 0,0003 | 3,84 |
| MLP (traité, W=5) | 5 | 0,9732 ± 0,0011 | 0,9974 | 0,0078 | 0,6590 ± 0,0006 | 3,74 |
| XGBoost (traité, W=10) | 5 | 0,9697 ± 0,0000 | 0,9973 | 0,0083 | 0,6575 ± 0,0000 | 1,54 |
| LSTM (traité, W=10) | 5 | 0,9680 ± 0,0004 | 0,9968 | 0,0090 | 0,6565 ± 0,0002 | 7,38 |
| MLP (traité, W=10) | 5 | 0,9652 ± 0,0005 | 0,9957 | 0,0097 | 0,6550 ± 0,0002 | 3,82 |
| LSTM (brut, W=10) | 5 | 0,9566 ± 0,0004 | 0,9957 | 0,0115 | 0,6511 ± 0,0002 | 7,34 |
| XGBoost (brut, W=10) | 5 | 0,9496 ± 0,0012 | 0,9940 | 0,0131 | 0,6475 ± 0,0005 | 1,27 |
| MLP (brut, W=10) | 5 | 0,9480 ± 0,0018 | 0,9934 | 0,0137 | 0,6466 ± 0,0009 | 3,75 |

## Effets lisibles sur les références (appariés par graine)

| Modèle | Effet | Mesure | n | Écart | IC 95 % | p |
|---|---|---|---|---|---|---|
| LSTM | prétraitement (traité − brut, W=10) | mcc | 5 | 0,0114 | ± 0,0004 | < 0,001 |
| LSTM | prétraitement (traité − brut, W=10) | perf | 5 | 0,0054 | ± 0,0002 | < 0,001 |
| LSTM | fenêtre (W=5 − W=10, traité) | mcc | 5 | 0,0075 | ± 0,0013 | < 0,001 |
| LSTM | fenêtre (W=5 − W=10, traité) | perf | 5 | 0,0037 | ± 0,0007 | < 0,001 |
| MLP | prétraitement (traité − brut, W=10) | mcc | 5 | 0,0171 | ± 0,0020 | < 0,001 |
| MLP | prétraitement (traité − brut, W=10) | perf | 5 | 0,0083 | ± 0,0010 | < 0,001 |
| MLP | fenêtre (W=5 − W=10, traité) | mcc | 5 | 0,0081 | ± 0,0009 | < 0,001 |
| MLP | fenêtre (W=5 − W=10, traité) | perf | 5 | 0,0041 | ± 0,0005 | < 0,001 |
| XGBoost | prétraitement (traité − brut, W=10) | mcc | 5 | 0,0201 | ± 0,0015 | < 0,001 |
| XGBoost | prétraitement (traité − brut, W=10) | perf | 5 | 0,0099 | ± 0,0006 | < 0,001 |
| XGBoost | fenêtre (W=5 − W=10, traité) | mcc | 5 | 0,0085 | ± 0,0000 | < 0,001 |
| XGBoost | fenêtre (W=5 − W=10, traité) | perf | 5 | 0,0042 | ± 0,0000 | < 0,001 |

L'effet de fenêtre compare aussi deux populations de validation différentes.

## Différences appariées

| Groupe | Mesure | Comparaison | n | Écart | IC 95 % | p | p Holm (groupe, mesure) |
|---|---|---|---|---|---|---|---|
| references | perf | GRU DoG/DoG/DoG traitée, W=5 − XGBoost | 5 | -0,00183 | ± 0,00149 | 0,027 | 0,188 |
| references | perf | GRU DoG/DoG/DoG traitée, W=5 − LSTM | 5 | -0,00040 | ± 0,00186 | 0,584 | 1,000 |
| references | perf | GRU DoG/DoG/DoG traitée, W=5 − MLP | 5 | 0,00079 | ± 0,00127 | 0,159 | 0,636 |
| references | perf | GRU DoG/ReLU/DoG traitée, W=5 − XGBoost | 5 | -0,00119 | ± 0,00142 | 0,081 | 0,483 |
| references | perf | GRU DoG/ReLU/DoG traitée, W=5 − LSTM | 5 | 0,00025 | ± 0,00177 | 0,717 | 1,000 |
| references | perf | GRU DoG/ReLU/DoG traitée, W=5 − MLP | 5 | 0,00144 | ± 0,00103 | 0,018 | 0,145 |
| references | perf | TKAN ReLU/B-spl/DoG/Lin, W=10 − LSTM | 5 | -0,00135 | ± 0,00079 | 0,009 | 0,081 |
| references | perf | TKAN ReLU/B-spl/DoG/Lin, W=10 − XGBoost | 5 | 0,00224 | ± 0,00087 | 0,002 | 0,020 |
| references | perf | TKAN ReLU/B-spl/DoG/Lin, W=10 − MLP | 5 | 0,00312 | ± 0,00072 | < 0,001 | 0,003 |
| references | perf | GRU uniforme hybride traitée, W=5 − XGBoost | 5 | -0,00204 | ± 0,00059 | < 0,001 | 0,007 |
| references | perf | GRU uniforme hybride traitée, W=5 − LSTM | 5 | -0,00061 | ± 0,00100 | 0,166 | 0,636 |
| references | perf | GRU uniforme hybride traitée, W=5 − MLP | 5 | 0,00058 | ± 0,00083 | 0,124 | 0,618 |
| h2_final | perf | GRU DoG/DoG/DoG (traité), W=5 − GRU DoG/ReLU/DoG (traité), W=5 | 5 | -0,00065 | ± 0,00040 | 0,011 | 0,032 |
| h2_final | r80 | GRU DoG/DoG/DoG (traité), W=5 − GRU DoG/ReLU/DoG (traité), W=5 | 5 | 0,08500 | ± 0,09466 | 0,067 | 0,067 |
| h2_final | F80 | GRU DoG/DoG/DoG (traité), W=5 − GRU DoG/ReLU/DoG (traité), W=5 | 5 | 0,00680 | ± 0,00842 | 0,089 | 0,089 |
| h2_final | perf | GRU DoG/DoG/DoG (traité), W=5 − Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,00021 | ± 0,00107 | 0,614 | 0,614 |
| h2_final | r80 | GRU DoG/DoG/DoG (traité), W=5 − Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,28500 | ± 0,14157 | 0,005 | 0,010 |
| h2_final | F80 | GRU DoG/DoG/DoG (traité), W=5 − Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,02518 | ± 0,01156 | 0,004 | 0,008 |
| h2_final | perf | GRU DoG/ReLU/DoG (traité), W=5 − Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,00086 | ± 0,00111 | 0,099 | 0,197 |
| h2_final | r80 | GRU DoG/ReLU/DoG (traité), W=5 − Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,20000 | ± 0,05377 | < 0,001 | 0,001 |
| h2_final | F80 | GRU DoG/ReLU/DoG (traité), W=5 − Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 5 | 0,01838 | ± 0,00389 | < 0,001 | < 0,001 |
| raccourci | mcc | GRU DoG/DoG/DoG (traité), W=5 sans les soldes de l'émetteur − complet | 5 | 0,00014 | ± 0,00030 | 0,271 | 0,271 |
| raccourci | perf | GRU DoG/DoG/DoG (traité), W=5 sans les soldes de l'émetteur − complet | 5 | 0,00007 | ± 0,00015 | 0,272 | 0,272 |

## Raccourci dans les données (raccourci.json, entraînement)

- Tolérance : 0.01 ; rapprochement des soldes de l'émetteur : fraudes 31,7 %, légitimes 18,4 %.
- ATO : 63,1 % (n = 474 572).
- FAKE_CRED : 36,7 % (n = 218).
- REFUND : 32,9 % (n = 191 203).
- SMURFING : 13,6 % (n = 648 197).
- SPLIT_DEP : 22,7 % (n = 373 996).

## Étape 4 (500 000 fenêtres) contre final (1 000 000)

| Architecture | perf É4 | perf final | R² É4 | R² final | Fitness É4 | Fitness final | Lat. É4 (c8g) | Lat. final (m8g) |
|---|---|---|---|---|---|---|---|---|
| GRU DoG/DoG/DoG (traité), W=5 | 0,6588 | 0,6598 | 0,925 | 0,755 | 0,7398 | 0,7260 | 101,6 | 99,1 |
| GRU DoG/ReLU/DoG (traité), W=5 | 0,6598 | 0,6605 | 0,820 | 0,670 | 0,7317 | 0,7192 | 122,2 | 119,4 |
| TKAN ReLU/B-spl/DoG/Lin, W=10 | 0,6478 | 0,6498 | 0,770 | 0,765 | 0,7153 | 0,7168 | 134,6 | 150,6 |
| Uniforme GRU Hyb/Hyb/Hyb (traité), W=5 | 0,6589 | 0,6596 | 0,525 | 0,470 | 0,7049 | 0,7008 | 36,6 | 37,2 |

Spearman des rangs É4/final : fitness 1,000, perf 0,800, R² 0,400 (n = 4, indicatif).
Latences : machines différentes (c8g à l'Étape 4, m8g au final), non comparables entre phases.
