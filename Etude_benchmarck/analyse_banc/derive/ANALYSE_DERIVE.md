# Expérience de dérive

60 runs (4 modèles x 3 bras x graines). Régions en dérive détectées par run : 2 à 6.

| Modèle | Bras | n | perf D_eval | MCC D_eval | Rappel Split Deposit | perf D_ref | MCC D_ref |
|---|---|---|---|---|---|---|---|
| GRUKANCell DoG/DoG/DoG | affinage seul | 5 | 0,6591 | 0,9729 ± 0,0075 | 0,9995 | 0,6576 | 0,9698 ± 0,0066 |
| GRUKANCell DoG/DoG/DoG | extension et affinage | 5 | 0,6581 | 0,9708 ± 0,0117 | 0,9994 | 0,6569 | 0,9684 ± 0,0091 |
| GRUKANCell DoG/DoG/DoG | figé | 5 | 0,6619 | 0,9788 ± 0,0023 | 0,9990 | 0,6598 | 0,9745 ± 0,0024 |
| GRUKANCell DoG/ReLU-KAN/DoG | affinage seul | 5 | 0,6605 | 0,9757 ± 0,0060 | 0,9998 | 0,6591 | 0,9729 ± 0,0048 |
| GRUKANCell DoG/ReLU-KAN/DoG | extension et affinage | 5 | 0,6601 | 0,9749 ± 0,0043 | 0,9998 | 0,6588 | 0,9723 ± 0,0037 |
| GRUKANCell DoG/ReLU-KAN/DoG | figé | 5 | 0,6623 | 0,9795 ± 0,0031 | 0,9985 | 0,6605 | 0,9758 ± 0,0024 |
| GRUKANCell hybride | affinage seul | 5 | 0,6612 | 0,9774 ± 0,0012 | 0,9998 | 0,6593 | 0,9735 ± 0,0009 |
| GRUKANCell hybride | extension et affinage | 5 | 0,6609 | 0,9767 ± 0,0027 | 0,9999 | 0,6591 | 0,9730 ± 0,0019 |
| GRUKANCell hybride | figé | 5 | 0,6618 | 0,9785 ± 0,0011 | 0,9997 | 0,6596 | 0,9741 ± 0,0010 |
| TKANCell B (brut) | affinage seul | 5 | 0,6471 | 0,9484 ± 0,0045 | 0,9879 | 0,6443 | 0,9429 ± 0,0080 |
| TKANCell B (brut) | extension et affinage | 5 | 0,6466 | 0,9474 ± 0,0051 | 0,9880 | 0,6437 | 0,9417 ± 0,0084 |
| TKANCell B (brut) | figé | 5 | 0,6527 | 0,9599 ± 0,0007 | 0,9740 | 0,6497 | 0,9540 ± 0,0010 |

| Modèle | Comparaison | Écart perf D_eval | p (Holm) | Écart rappel Split | Écart perf D_ref | É.-t. D_ref entre graines |
|---|---|---|---|---|---|---|
| GRUKANCell DoG/DoG/DoG | extension - affinage | -0,00098 ± 0,00348 | 1,000 | -0,0000 | -0,00065 | 0,00318 |
| GRUKANCell DoG/DoG/DoG | affinage - fige | -0,00280 ± 0,00252 | 0,135 | 0,0004 | -0,00228 | 0,00121 |
| GRUKANCell DoG/DoG/DoG | extension - fige | -0,00378 ± 0,00451 | 0,269 | 0,0004 | -0,00293 | 0,00121 |
| GRUKANCell DoG/ReLU-KAN/DoG | extension - affinage | -0,00037 ± 0,00117 | 1,000 | 0,0000 | -0,00028 | 0,00227 |
| GRUKANCell DoG/ReLU-KAN/DoG | affinage - fige | -0,00182 ± 0,00140 | 0,131 | 0,0013 | -0,00142 | 0,00115 |
| GRUKANCell DoG/ReLU-KAN/DoG | extension - fige | -0,00219 ± 0,00067 | 0,008 | 0,0013 | -0,00169 | 0,00115 |
| GRUKANCell hybride | extension - affinage | -0,00032 ± 0,00076 | 1,000 | 0,0000 | -0,00022 | 0,00046 |
| GRUKANCell hybride | affinage - fige | -0,00056 ± 0,00096 | 0,260 | 0,0001 | -0,00031 | 0,00048 |
| GRUKANCell hybride | extension - fige | -0,00088 ± 0,00165 | 0,300 | 0,0001 | -0,00053 | 0,00048 |
| TKANCell B (brut) | extension - affinage | -0,00048 ± 0,00068 | 0,751 | 0,0001 | -0,00062 | 0,00389 |
| TKANCell B (brut) | affinage - fige | -0,00561 ± 0,00212 | 0,016 | 0,0138 | -0,00547 | 0,00050 |
| TKANCell B (brut) | extension - fige | -0,00609 ± 0,00244 | 0,015 | 0,0140 | -0,00609 | 0,00050 |

Règle de décision (extension utile) : GRUKANCell DoG/DoG/DoG : non, GRUKANCell DoG/ReLU-KAN/DoG : non, GRUKANCell hybride : non, TKANCell B (brut) : non
