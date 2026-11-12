# Contrôle du raccourci par les soldes, sur XGBoost

Validation uniquement, 5 graines, 1000000 fenêtres d'entraînement. Écart de MCC apparié par graine à la version complète, p corrigés par Holm.

| Régime | W | Variante | Variables | MCC | PR-AUC | Rappel | Faux positifs | Écart de MCC | p corrigé |
|---|---|---|---|---|---|---|---|---|---|
| brut | 10 | complet | 12 | 0,9492 ± 0,0019 | 0,9939 | 0,9902 | 0,01906 |  |  |
| brut | 10 | sans les soldes de l'émetteur | 10 | 0,9363 ± 0,0003 | 0,9918 | 0,9817 | 0,02236 | -0,0129 ± 0,0021 | < 0,001 |
| brut | 10 | sans aucun solde | 8 | 0,9161 ± 0,0002 | 0,9864 | 0,9865 | 0,03323 | -0,0330 ± 0,0020 | < 0,001 |
| traité | 5 | complet | 12 | 0,9779 ± 0,0000 | 0,9986 | 0,9990 | 0,00958 |  |  |
| traité | 5 | sans les variables dérivées des soldes | 8 | 0,9559 ± 0,0013 | 0,9956 | 0,9960 | 0,01891 | -0,0220 ± 0,0013 | < 0,001 |

Lecture : une chute nette du MCC sans les soldes signifie que la détection s'appuie sur l'arithmétique des soldes du simulateur. Contrôle fait sur XGBoost, pas sur les cellules KAN.
