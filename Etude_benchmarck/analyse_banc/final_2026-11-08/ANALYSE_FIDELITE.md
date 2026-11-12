# Fidélité des formules au réseau

Validation uniquement, 20000 fenêtres tirées avec une graine fixe, une graine par finaliste (celle de score perf médian). Accord : part des fenêtres où la décision au seuil 0,5 est celle du réseau complet.

## GRUKANCell DoG/DoG/DoG, traité

Arêtes : 19575 au total, 636 actives (L1 > 0,01). Part des arêtes actives symbolisables : 69,7 % au seuil 0,80 et 8,6 % au seuil 0,99 (sur les 40 plus fortes : 67,5 % et 5,0 %).

| Variante | Arêtes actives | Accord des décisions | R² des scores | MCC |
|---|---|---|---|---|
| réseau complet | 19575 | 1,0000 | 1,000 | 0,9753 |
| élagage, L1 > 0,01 | 636 | 0,9993 | 0,999 | 0,9751 |
| élagage, L1 > 0,05 | 357 | 0,9856 | 0,940 | 0,9409 |
| élagage, L1 > 0,10 | 226 | 0,9372 | 0,727 | 0,8119 |
| élagage, L1 > 0,20 | 133 | 0,7533 | 0,097 | 0,3763 |
| 1000 arêtes les plus fortes | 1000 | 0,9998 | 1,000 | 0,9755 |
| 300 arêtes les plus fortes | 300 | 0,9746 | 0,890 | 0,9086 |
| 100 arêtes les plus fortes | 100 | 0,7551 | 0,073 | 0,1062 |
| 40 arêtes les plus fortes | 40 | 0,2248 | -2,791 | -0,0134 |
| arêtes à R² ≥ 0,80 remplacées par leur formule, les autres apprises | 636 | 0,9448 | 0,771 | 0,8159 |
| arêtes à R² ≥ 0,80 remplacées sur [-2, 2] seulement, apprises au-delà | 636 | 0,9917 | 0,960 | 0,9595 |
| formule seule : arêtes à R² ≥ 0,80, les autres supprimées | 443 | 0,7812 | -0,138 | 0,1469 |
| arêtes à R² ≥ 0,99 remplacées par leur formule, les autres apprises | 636 | 0,9880 | 0,957 | 0,9436 |
| arêtes à R² ≥ 0,99 remplacées sur [-2, 2] seulement, apprises au-delà | 636 | 0,9991 | 0,998 | 0,9747 |
| formule seule : arêtes à R² ≥ 0,99, les autres supprimées | 55 | 0,7903 | -0,109 | 0,2315 |
| toutes les arêtes actives remplacées par leur meilleure formule | 636 | 0,8985 | 0,531 | 0,6848 |

Six arêtes les plus fortes partant d'une variable d'entrée :

| Porte | Variable | L1 | Meilleure formule | R² |
|---|---|---|---|---|
| reset | delta_commission | 2,126 | 0,177 · log(|x|+eps)(1,420·x + -0,100) + -0,005 | 0,187 |
| update | v1h | 1,154 | 0,767 · cos(x)(1,752·x + 1,248) + 0,046 | 0,605 |
| reset | v1h | 1,009 | 0,611 · tanh(x)(4,644·x + -0,074) + -0,018 | 0,799 |
| reset | delta_commission | 0,909 | 1,460 · tanh(x)(-1,343·x + -1,011) + 0,059 | 0,995 |
| reset | rho_nouveau | 0,851 | 1,631 · cos(x)(0,887·x + 2,054) + -0,013 | 1,000 |
| reset | delta_commission | 0,757 | 1,110 · sin(x)(1,801·x + -0,151) + -0,024 | 0,884 |

## GRUKANCell DoG/ReLU-KAN/DoG, traité

Arêtes : 19575 au total, 1122 actives (L1 > 0,01). Part des arêtes actives symbolisables : 48,6 % au seuil 0,80 et 4,7 % au seuil 0,99 (sur les 40 plus fortes : 77,5 % et 2,5 %).

| Variante | Arêtes actives | Accord des décisions | R² des scores | MCC |
|---|---|---|---|---|
| réseau complet | 19575 | 1,0000 | 1,000 | 0,9781 |
| élagage, L1 > 0,01 | 1122 | 0,9981 | 0,991 | 0,9747 |
| élagage, L1 > 0,05 | 526 | 0,9500 | 0,780 | 0,8690 |
| élagage, L1 > 0,10 | 319 | 0,8310 | 0,210 | 0,4582 |
| élagage, L1 > 0,20 | 173 | 0,7759 | -0,287 | 0,0134 |
| 1000 arêtes les plus fortes | 1000 | 0,9961 | 0,984 | 0,9689 |
| 300 arêtes les plus fortes | 300 | 0,8387 | 0,250 | 0,4914 |
| 100 arêtes les plus fortes | 100 | 0,7758 | -0,289 | 0,0000 |
| 40 arêtes les plus fortes | 40 | 0,7758 | -0,261 | 0,0000 |
| arêtes à R² ≥ 0,80 remplacées par leur formule, les autres apprises | 1122 | 0,9676 | 0,844 | 0,8845 |
| arêtes à R² ≥ 0,80 remplacées sur [-2, 2] seulement, apprises au-delà | 1122 | 0,9947 | 0,976 | 0,9651 |
| formule seule : arêtes à R² ≥ 0,80, les autres supprimées | 545 | 0,8603 | 0,304 | 0,5260 |
| arêtes à R² ≥ 0,99 remplacées par leur formule, les autres apprises | 1122 | 0,9785 | 0,899 | 0,9176 |
| arêtes à R² ≥ 0,99 remplacées sur [-2, 2] seulement, apprises au-delà | 1122 | 0,9980 | 0,991 | 0,9742 |
| formule seule : arêtes à R² ≥ 0,99, les autres supprimées | 53 | 0,7772 | -0,148 | 0,0886 |
| toutes les arêtes actives remplacées par leur meilleure formule | 1122 | 0,8905 | 0,508 | 0,6746 |

Six arêtes les plus fortes partant d'une variable d'entrée :

| Porte | Variable | L1 | Meilleure formule | R² |
|---|---|---|---|---|
| reset | rho_nouveau | 1,184 | 1,894 · tanh(x)(-1,482·x + -0,593) + 0,000 | 0,979 |
| reset | v1h | 1,160 | 0,892 · sin(x)(1,843·x + -0,077) + -0,012 | 0,781 |
| reset | v1h | 0,947 | 1,237 · sin(x)(1,537·x + -0,093) + -0,017 | 0,982 |
| reset | delta_commission | 0,920 | 0,268 · sin(x)(1,164·x + 0,408) + -0,007 | 0,489 |
| reset | delta_commission | 0,855 | 1,158 · sqrt(|x|)(1,163·x + 0,604) + -1,339 | 0,789 |
| reset | var_agent_split | 0,748 | 0,735 · sin(x)(1,686·x + -0,505) + -0,039 | 0,894 |

## GRUKANCell hybride, traité

Arêtes : 19575 au total, 2186 actives (L1 > 0,01). Part des arêtes actives symbolisables : 28,0 % au seuil 0,80 et 2,7 % au seuil 0,99 (sur les 40 plus fortes : 45,0 % et 2,5 %).

| Variante | Arêtes actives | Accord des décisions | R² des scores | MCC |
|---|---|---|---|---|
| réseau complet | 19575 | 1,0000 | 1,000 | 0,9759 |
| élagage, L1 > 0,01 | 2186 | 0,9981 | 0,993 | 0,9741 |
| élagage, L1 > 0,05 | 887 | 0,9795 | 0,927 | 0,9341 |
| élagage, L1 > 0,10 | 580 | 0,9540 | 0,803 | 0,8797 |
| élagage, L1 > 0,20 | 379 | 0,9603 | 0,836 | 0,8967 |
| 1000 arêtes les plus fortes | 1000 | 0,9944 | 0,978 | 0,9650 |
| 300 arêtes les plus fortes | 300 | 0,9465 | 0,482 | 0,8335 |
| 100 arêtes les plus fortes | 100 | 0,8502 | 0,268 | 0,5372 |
| 40 arêtes les plus fortes | 40 | 0,7762 | -0,147 | 0,0000 |
| arêtes à R² ≥ 0,80 remplacées par leur formule, les autres apprises | 2186 | 0,9378 | 0,671 | 0,7927 |
| arêtes à R² ≥ 0,80 remplacées sur [-2, 2] seulement, apprises au-delà | 2186 | 0,9960 | 0,985 | 0,9690 |
| formule seule : arêtes à R² ≥ 0,80, les autres supprimées | 612 | 0,2904 | -1,367 | -0,0630 |
| arêtes à R² ≥ 0,99 remplacées par leur formule, les autres apprises | 2186 | 0,9566 | 0,752 | 0,8504 |
| arêtes à R² ≥ 0,99 remplacées sur [-2, 2] seulement, apprises au-delà | 2186 | 0,9982 | 0,992 | 0,9736 |
| formule seule : arêtes à R² ≥ 0,99, les autres supprimées | 59 | 0,7804 | -0,420 | 0,1803 |
| toutes les arêtes actives remplacées par leur meilleure formule | 2186 | 0,2570 | -3,016 | 0,0901 |

Six arêtes les plus fortes partant d'une variable d'entrée :

| Porte | Variable | L1 | Meilleure formule | R² |
|---|---|---|---|---|
| reset | rho_nouveau | 1,084 | 1,304 · |x|(1,311·x + -0,580) + -1,933 | 0,973 |
| reset | delta_commission | 0,960 | 2,370 · sin(x)(1,423·x + -0,253) + 0,138 | 0,998 |
| reset | rho_nouveau | 0,910 | 1,661 · tanh(x)(1,951·x + 2,756) + -1,346 | 0,871 |
| update | v1h | 0,890 | 1,431 · cos(x)(1,539·x + 1,578) + -0,113 | 0,993 |
| reset | rho_nouveau | 0,887 | 1,235 · |x|(1,238·x + -0,448) + -1,902 | 0,971 |
| reset | v1h | 0,867 | 1,433 · sin(x)(1,488·x + 0,001) + 0,077 | 0,988 |

## TKANCell ReLU-KAN/B-spline/DoG/linéaire, brut

Arêtes : 19456 au total, 561 actives (L1 > 0,01). Part des arêtes actives symbolisables : 60,8 % au seuil 0,80 et 23,0 % au seuil 0,99 (sur les 40 plus fortes : 70,0 % et 55,0 %).

| Variante | Arêtes actives | Accord des décisions | R² des scores | MCC |
|---|---|---|---|---|
| réseau complet | 19456 | 1,0000 | 1,000 | 0,9520 |
| élagage, L1 > 0,01 | 561 | 0,9721 | 0,938 | 0,9161 |
| élagage, L1 > 0,05 | 329 | 0,9637 | 0,838 | 0,8983 |
| élagage, L1 > 0,10 | 210 | 0,9618 | 0,822 | 0,8941 |
| élagage, L1 > 0,20 | 102 | 0,9404 | 0,735 | 0,8314 |
| 1000 arêtes les plus fortes | 1000 | 0,9900 | 0,974 | 0,9386 |
| 300 arêtes les plus fortes | 300 | 0,9635 | 0,832 | 0,8984 |
| 100 arêtes les plus fortes | 100 | 0,9412 | 0,740 | 0,8336 |
| 40 arêtes les plus fortes | 40 | 0,9222 | 0,681 | 0,7669 |
| arêtes à R² ≥ 0,80 remplacées par leur formule, les autres apprises | 561 | 0,9623 | 0,883 | 0,8929 |
| arêtes à R² ≥ 0,80 remplacées sur [-2, 2] seulement, apprises au-delà | 561 | 0,9628 | 0,886 | 0,8950 |
| formule seule : arêtes à R² ≥ 0,80, les autres supprimées | 341 | 0,9240 | 0,636 | 0,7890 |
| arêtes à R² ≥ 0,99 remplacées par leur formule, les autres apprises | 561 | 0,9728 | 0,940 | 0,9171 |
| arêtes à R² ≥ 0,99 remplacées sur [-2, 2] seulement, apprises au-delà | 561 | 0,9725 | 0,940 | 0,9165 |
| formule seule : arêtes à R² ≥ 0,99, les autres supprimées | 129 | 0,9009 | 0,602 | 0,7058 |
| toutes les arêtes actives remplacées par leur meilleure formule | 561 | 0,9315 | 0,687 | 0,8082 |

Six arêtes les plus fortes partant d'une variable d'entrée :

| Porte | Variable | L1 | Meilleure formule | R² |
|---|---|---|---|---|
| Input | newBalanceOrig | 6,932 | 2,624 · x(0,143·x + 1,932) + 1,775 | 0,954 |
| Output | action_TRANSFER | 3,871 | 2,722 · |x|(1,219·x + -2,130) + -5,697 | 0,989 |
| Forget | amount | 2,215 | 2,245 · cos(x)(1,864·x + 1,412) + 0,226 | 0,438 |
| Output | amount | 1,267 | 1,304 · x(1,305·x + -0,002) + 0,002 | 1,000 |
| Output | action_TRANSFER | 1,219 | 1,044 · x(1,053·x + 0,001) + -0,001 | 1,000 |
| Output | action_TRANSFER | 1,121 | 1,346 · |x|(0,750·x + -1,462) + -1,968 | 1,000 |

