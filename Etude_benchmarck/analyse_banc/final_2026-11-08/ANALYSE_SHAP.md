# XGBoost expliqué par TreeSHAP contre fonctions d'arête des cellules KAN

Validation uniquement, 20000 fenêtres (W = 5, régime traité), XGBoost graine 1. Écart maximal d'additivité de TreeSHAP (somme des contributions moins marge du modèle) : 1.81e-05.
Contributions nécessaires pour couvrir 90 % de |SHAP| d'une décision : médiane 14, 9e décile 16 (sur 48).

| Variable | Part SHAP XGBoost | Part L1 GRUKANCell DoG/DoG/DoG | Part L1 GRUKANCell DoG/ReLU-KAN/DoG | Part L1 GRUKANCell hybride |
|---|---|---|---|---|
| delta_B_dest | 18,8 % | 12,2 % | 19,6 % | 25,2 % |
| v1h | 14,8 % | 18,7 % | 14,2 % | 12,2 % |
| r1 | 10,8 % | 0,5 % | 1,1 % | 0,6 % |
| rho_nouveau | 10,5 % | 18,2 % | 13,2 % | 23,8 % |
| r2 | 10,3 % | 1,8 % | 3,8 % | 4,9 % |
| delta_commission | 10,0 % | 26,3 % | 18,6 % | 8,6 % |
| delta_B_orig | 9,5 % | 10,1 % | 13,8 % | 15,0 % |
| rho_rupture | 7,3 % | 3,3 % | 6,4 % | 3,0 % |
| var_agent_split | 6,0 % | 8,0 % | 8,3 % | 4,1 % |
| rho_refund | 0,8 % | 0,3 % | 0,4 % | 1,0 % |
| flag_nuit | 0,8 % | 0,2 % | 0,3 % | 0,6 % |
| flag_anomalie | 0,3 % | 0,4 % | 0,4 % | 1,0 % |

| Cellule KAN | Spearman avec SHAP | p | Variables communes dans les 5 premières |
|---|---|---|---|
| GRUKANCell DoG/DoG/DoG | 0,650 | 0,022 | 3 |
| GRUKANCell DoG/ReLU-KAN/DoG | 0,706 | 0,010 | 3 |
| GRUKANCell hybride | 0,594 | 0,042 | 3 |
