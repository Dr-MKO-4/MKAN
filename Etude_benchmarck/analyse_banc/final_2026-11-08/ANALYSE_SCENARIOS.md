# Rappel par scénario, précision à prévalence réaliste, base unique contre bases mixtes

Validation uniquement, seuil de décision 0,5. Le jeu de test n'a pas été lu.

Entraînements analysés : 65. Écartés par le contrôle d'intégrité : 0.

## 1. Rappel par scénario (moyenne sur les graines)

Effectifs de fenêtres par scénario (plus grande valeur entre W = 5 et W = 10) : Account Takeover 110707, Smurfing 183082, Split Deposit 82727, Refund Fraud 44514, Fake Credentials 212.

| Modèle | Régime | W | Account Takeover | Smurfing | Split Deposit | Refund Fraud | Fake Credentials | Rappel global |
|---|---|---|---|---|---|---|---|---|
| XGBoost, traité | traite | 5 | 1,0000 | 0,9996 | 1,0000 | 0,9969 | 0,4245 | 0,9992 |
| GRUKANCell DoG/ReLU-KAN/DoG, traité | traite | 5 | 0,9998 | 0,9983 | 0,9996 | 0,9977 | 0,0868 | 0,9985 |
| LSTM, traité | traite | 5 | 0,9999 | 0,9962 | 0,9995 | 0,9984 | 0,2368 | 0,9977 |
| GRUKANCell DoG/DoG/DoG, traité | traite | 5 | 0,9996 | 0,9972 | 0,9996 | 0,9983 | 0,1066 | 0,9979 |
| GRUKANCell hybride/hybride/hybride, traité | traite | 5 | 0,9996 | 0,9938 | 0,9994 | 0,9965 | 0,1113 | 0,9963 |
| MLP, traité | traite | 5 | 0,9995 | 0,9944 | 0,9993 | 0,9960 | 0,2736 | 0,9965 |
| XGBoost, traité | traite | 10 | 1,0000 | 0,9997 | 0,9998 | 0,9947 | 0,5520 | 0,9989 |
| LSTM, traité | traite | 10 | 0,9997 | 0,9973 | 0,9995 | 0,9992 | 0,4594 | 0,9979 |
| MLP, traité | traite | 10 | 0,9987 | 0,9941 | 0,9993 | 0,9978 | 0,3566 | 0,9957 |
| LSTM, brut | brut | 10 | 0,9996 | 0,9939 | 0,9955 | 0,9892 | 0,7623 | 0,9944 |
| TKANCell ReLU-KAN/B-spline/DoG/linéaire, brut | brut | 10 | 0,9981 | 0,9934 | 0,9982 | 0,9810 | 0,6457 | 0,9932 |
| XGBoost, brut | brut | 10 | 0,9994 | 0,9964 | 0,9984 | 0,9384 | 0,6434 | 0,9902 |
| MLP, brut | brut | 10 | 0,9985 | 0,9881 | 0,9869 | 0,9632 | 0,6354 | 0,9866 |

Lecture : un rappel faible sur un scénario désigne la typologie que le modèle laisse passer. Fake Credentials ne compte que quelques centaines de fenêtres : son rappel est très incertain.

## 2. Précision attendue selon la proportion de fraude

| Modèle | W | Rappel | Taux de faux positifs | Précision à 0,1 % | Précision à 1,0 % | Précision à 5,0 % | Fausses alertes par million à 0,1 % | Fausses alertes par million à 1,0 % | Fausses alertes par million à 5,0 % |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost, traité | 5 | 0,9992 | 0,00949 | 0,0953 | 0,5153 | 0,8471 | 9484 | 9398 | 9018 |
| GRUKANCell DoG/ReLU-KAN/DoG, traité | 5 | 0,9985 | 0,01038 | 0,0885 | 0,4940 | 0,8353 | 10370 | 10277 | 9861 |
| LSTM, traité | 5 | 0,9977 | 0,01029 | 0,0885 | 0,4948 | 0,8361 | 10280 | 10188 | 9776 |
| GRUKANCell DoG/DoG/DoG, traité | 5 | 0,9979 | 0,01081 | 0,0851 | 0,4834 | 0,8295 | 10797 | 10700 | 10267 |
| GRUKANCell hybride/hybride/hybride, traité | 5 | 0,9963 | 0,01047 | 0,0870 | 0,4902 | 0,8336 | 10459 | 10365 | 9946 |
| MLP, traité | 5 | 0,9965 | 0,01095 | 0,0837 | 0,4793 | 0,8273 | 10941 | 10842 | 10404 |
| XGBoost, traité | 10 | 0,9989 | 0,01250 | 0,0741 | 0,4466 | 0,8079 | 12488 | 12376 | 11876 |
| LSTM, traité | 10 | 0,9979 | 0,01297 | 0,0715 | 0,4374 | 0,8020 | 12954 | 12838 | 12319 |
| MLP, traité | 10 | 0,9957 | 0,01356 | 0,0685 | 0,4259 | 0,7945 | 13545 | 13423 | 12881 |
| LSTM, brut | 10 | 0,9944 | 0,01699 | 0,0554 | 0,3716 | 0,7549 | 16976 | 16823 | 16143 |
| TKANCell ReLU-KAN/B-spline/DoG/linéaire, brut | 10 | 0,9932 | 0,01776 | 0,0530 | 0,3610 | 0,7464 | 17743 | 17584 | 16873 |
| XGBoost, brut | 10 | 0,9902 | 0,01889 | 0,0498 | 0,3462 | 0,7339 | 18874 | 18704 | 17948 |
| MLP, brut | 10 | 0,9866 | 0,01854 | 0,0507 | 0,3500 | 0,7371 | 18517 | 18350 | 17609 |

Lecture : le rappel et le taux de faux positifs sont mesurés sur la validation (environ 20 % de fraude) et supposés inchangés ; seule la proportion de fraude varie. Cette hypothèse suppose que les fraudes et les transactions légitimes gardent la même distribution, ce que la simulation ne garantit pas.

## 3. Base unique contre bases mixtes (GRUKANCell, régime brut, plan de croisement)

Référence : GRU DoG/DoG/DoG (score perf 0,6441, R² symbolique 0,915). Écarts appariés par graine, p corrigés par Holm.

| Architecture mixte | Écart de score perf | p corrigé | Écart de R² symbolique | p corrigé |
|---|---|---|---|---|
| ReLU-KAN/DoG/DoG | 0,00414 ± 0,00397 | 0,719 | -0,330 ± 0,045 | < 0,001 |
| ReLU-KAN/ReLU-KAN/DoG | 0,00365 ± 0,00382 | 0,797 | -0,290 ± 0,091 | 0,006 |
| ReLU-KAN/ReLU-KAN/Morlet | 0,00363 ± 0,00429 | 0,856 | -0,915 ± 0,042 | < 0,001 |
| ReLU-KAN/B-spline/DoG | 0,00335 ± 0,00272 | 0,561 | -0,260 ± 0,068 | 0,004 |
| ReLU-KAN/B-spline/Morlet | 0,00327 ± 0,00374 | 0,856 | -0,915 ± 0,042 | < 0,001 |
| DoG/ReLU-KAN/DoG | 0,00311 ± 0,00254 | 0,561 | -0,045 ± 0,119 | 0,564 |
| DoG/ReLU-KAN/Morlet | 0,00291 ± 0,00474 | 0,856 | -0,490 ± 0,068 | < 0,001 |
| DoG/DoG/Morlet | 0,00255 ± 0,00301 | 0,856 | -0,455 ± 0,048 | < 0,001 |
| DoG/B-spline/Morlet | 0,00236 ± 0,00294 | 0,856 | -0,575 ± 0,109 | 0,001 |
| ReLU-KAN/DoG/Morlet | 0,00222 ± 0,00492 | 0,856 | -0,845 ± 0,094 | < 0,001 |
| DoG/B-spline/DoG | 0,00100 ± 0,00574 | 0,856 | -0,040 ± 0,072 | 0,564 |
