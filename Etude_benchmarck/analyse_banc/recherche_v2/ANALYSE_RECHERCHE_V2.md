# Recherche NSGA-II réexécutée (13-14/11/2026)

160 évaluations journalisées, 3 individus sur le front ; écart maximal de fitness recalculée : 0.0e+00. Cellules : {'TKANCell': 62, 'GRUKANCell': 98} ; régimes : {'engineered': 127, 'raw': 33} ; latence médiane 44,2 ms, 8 individus au-dessus de 500 ms.

| Cellule | Régime | Bases | h | W | MCC | PR-AUC | Brier | R² 0,80 | R² 0,99 | Latence (ms) | Fitness | Front |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRUKANCell | traité | FasterKAN/FasterKAN/DoG | 32 | 5 | 0,9603 | 0,9958 | 0,0115 | 1,000 | 0,050 | 28,7 | 0,7403 | oui |
| GRUKANCell | traité | hybride/FasterKAN/DoG | 32 | 5 | 0,9555 | 0,9927 | 0,0130 | 1,000 | 0,000 | 42,2 | 0,7373 |  |
| GRUKANCell | traité | hybride/FasterKAN/DoG | 64 | 5 | 0,9526 | 0,9920 | 0,0136 | 1,000 | 0,050 | 124,9 | 0,7358 |  |
| GRUKANCell | traité | FasterKAN/FasterKAN/DoG | 32 | 5 | 0,9524 | 0,9925 | 0,0143 | 1,000 | 0,075 | 29,5 | 0,7357 |  |
| GRUKANCell | traité | FasterKAN/FasterKAN/DoG | 32 | 5 | 0,9586 | 0,9941 | 0,0117 | 0,875 | 0,100 | 29,7 | 0,7282 |  |
| TKANCell | traité | ReLU-KAN/FasterKAN/Chebyshev/B-spline | 8 | 5 | 0,9680 | 0,9968 | 0,0094 | 0,700 | 0,100 | 10,4 | 0,7178 | oui |
| TKANCell | traité | ReLU-KAN/B-spline/Morlet/FasterKAN | 8 | 5 | 0,9686 | 0,9972 | 0,0092 | 0,300 | 0,100 | 9,1 | 0,6831 | oui |
