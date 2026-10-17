# Analyse des résultats partiels  Étape 4 (Niveau 1)

*Analyse du 13/10/2026 sur les fichiers `step4_progress.jsonl`, `step4_runs.csv`,
`step4_failures.jsonl` et `val_scores/*.npy` (signature de run `a641e9e5d9c65a87`).
Toutes les valeurs ci-dessous sont recalculées à partir de ces fichiers ; rien n'est extrapolé.*

---

## 1. Avancement

| | Prévu (phase 1, régime `raw`) | Terminé | Échoué |
|---|---|---|---|
| TKAN (LSTM-KAN) | 24 architectures × 5 graines = 120 runs | **60 runs** (12 architectures) | 0 |
| GRU-KAN | 12 architectures × 5 graines = 60 runs | **5 runs** (2 architectures, incomplètes) | **31 tentatives (OOM)** |
| **Total phase 1** | 180 | **65 (36 %)** |  |

- Les 12 architectures TKAN terminées ont **toutes `Forget = relukan`**. Les 12 autres
  (`Forget = wavkan_dog`) n'ont pas encore tourné : **l'effet de la porte Forget n'est donc pas encore mesurable.**
- La phase 2 (régime `engineered`, top-3) et la **baseline MKAN uniforme** ne sont pas encore calculées.
  La conclusion « hétérogène vs uniforme », qui est l'objet de l'Étape 4, n'est donc pas encore établie.

---

## 2. Résultats TKAN (60 runs, 12 architectures)

Notation : `Input / Candidate / Output` (Forget = relukan partout).

| Rang | Architecture | Fitness | σ | MCC | Latence (ms) | R²_symb | Params |
|---|---|---|---|---|---|---|---|
| 1 | efficientkan / wavkan_dog / linear | **0.7258** | 0.0059 | 0.9735 | **79.0** | **0.760** | 57 761 |
| 2 | wavkan_dog / wavkan_dog / linear | 0.7068 | 0.0029 | 0.9732 | 80.7 | 0.545 | 60 577 |
| 3 | relukan / wavkan_dog / linear | 0.6993 | 0.0078 | 0.9740 | 95.2 | 0.455 | 81 697 |
| 4 | efficientkan / wavkan_morlet / linear | 0.6874 | 0.0038 | 0.9738 | 93.3 | 0.320 | 61 985 |
| 5 | wavkan_dog / wavkan_morlet / linear | 0.6754 | 0.0066 | 0.9713 | 98.6 | 0.195 | 64 801 |
| 6 | efficientkan / wavkan_dog / wavkan_morlet | 0.6581 | 0.0052 | 0.9728 | 111.7 | 0.135 | 73 249 |
| 7 | wavkan_dog / wavkan_dog / wavkan_morlet | 0.6544 | 0.0073 | 0.9748 | 113.5 | 0.125 | 76 065 |
| 8 | relukan / wavkan_morlet / linear | 0.6522 | 0.0095 | 0.9706 | 113.9 | 0.130 | 85 921 |
| 9 | efficientkan / wavkan_morlet / wavkan_morlet | 0.6208 | 0.0151 | 0.9712 | 120.5 | 0.010 | 77 473 |
| 10 | wavkan_dog / wavkan_morlet / wavkan_morlet | 0.6014 | 0.0121 | 0.9689 | 125.7 | 0.035 | 80 289 |
| 11 | relukan / wavkan_dog / wavkan_morlet | 0.5898 | 0.0104 | 0.9748 | 130.5 | 0.135 | 97 185 |
| 12 | relukan / wavkan_morlet / wavkan_morlet | 0.4919 | 0.0214 | 0.9740 | 143.7 | 0.000 | 101 409 |

### 2.1 Constat principal : le classement ne mesure pas la qualité de détection

Décomposition de la fitness, `0.4147·MCC + 0.2573·PR-AUC − 0.1529·Brier + 0.0876·R²_symb − 0.0876·pénalité_latence`,
en écart max−min entre architectures :

| Composante | Écart entre architectures |
|---|---|
| Partie détection (MCC + PR-AUC + Brier) | **0.0027** |
| Terme R²_symbolique | 0.0666 |
| Terme pénalité de latence | **0.1675** |
| **Fitness totale** | **0.2338** |

**99 % de l'écart de fitness vient de la latence et de l'interprétabilité symbolique, pas de la détection.**
Toutes les architectures TKAN détectent la fraude de façon quasi identique :
MCC entre 0.969 et 0.975, PR-AUC ≈ 0.998, F1 entre 0.982 et 0.984, précision à 80 % de rappel ≈ 1.000.
L'écart-type entre architectures (0.0018 sur le MCC moyen) est **du même ordre que l'écart-type entre graines (0.0021)**.

Interprétation : sur les données MOMTSIM en régime brut, la tâche de détection est **saturée** pour les
TKAN. Le benchmark départage donc les architectures sur leur coût (latence) et leur auditabilité (R²_symb).
C'est cohérent avec l'objectif du MKAN (modèle déployable et auditable), mais il faut le dire explicitement
dans le mémoire et ne pas présenter le rang 1 comme « le meilleur détecteur ».

### 2.2 Effets par porte (moyennes marginales sur les 12 architectures)

| Porte | Base | Fitness | Latence (ms) | R²_symb | Params |
|---|---|---|---|---|---|
| **Output** | linear | **0.691** | **93.4** | **0.401** | 68 790 |
| | wavkan_morlet | 0.603 | 124.3 | 0.073 | 84 278 |
| **Candidate** | wavkan_dog | **0.672** | **101.7** | **0.359** | 74 422 |
| | wavkan_morlet | 0.622 | 115.9 | 0.115 | 78 646 |
| **Input** | efficientkan | **0.673** | **101.1** | **0.306** | 67 617 |
| | wavkan_dog | 0.660 | 104.6 | 0.225 | 70 433 |
| | relukan | 0.608 | 120.8 | 0.180 | 91 553 |

- **Output = linear** est le choix le plus déterminant (+0.088 de fitness). Les 5 meilleures architectures l'ont toutes.
  Une porte de sortie linéaire n'enlève rien à la détection : son MCC moyen est identique à celui de wavkan_morlet (0.9727 vs 0.9728).
- **Morlet est pénalisant partout** : plus lent et quasi non symbolisable (R²_symb ≈ 0 quand il est sur 2 portes).
- **relukan sur Input** coûte cher en paramètres (+35 %) et en latence, sans gain de détection.
- Corrélations sur les 60 runs : params ↔ latence **r = 0.88** ; R²_symb ↔ latence **r = −0.86**.
  Les bases les plus lourdes sont aussi les moins traduisibles en formules.

### 2.3 Contrainte de latence (100 ms, USSD)

- **35 runs sur 60 dépassent 100 ms.** Seules les architectures avec `Output = linear` (sauf `relukan/morlet/linear`) respectent la contrainte.
- Les rangs 3 à 5 sont **juste sous le seuil** (93 à 99 ms). La latence est la meilleure de 3 répétitions sur un lot de 256 fenêtres aléatoires, mesurée sur le CPU du Mac.
  Leur conformité dépend donc de la machine : sur le serveur de production, elle doit être revérifiée.
- Le rang 1 (79 ms) et le rang 2 (81 ms) ont une marge confortable.

### 2.4 Significativité statistique (5 graines)

- **Test de Friedman** sur les 12 architectures : fitness χ² = 54.0, p = 1.2·10⁻⁷ ; MCC χ² = 39.2, p = 4.8·10⁻⁵.
  Les architectures ne sont donc pas toutes équivalentes.
- **Post-hoc de Nemenyi** (α = 0.05, k = 12, N = 5) : distance critique = **7.45 rangs**.
  Le rang 1 (rang moyen 1.0) n'est significativement meilleur **que les architectures 9 à 12**.
  **Il n'est pas statistiquement distinguable des rangs 2 à 8.**
- Pour le MCC seul, le rang 1 en fitness n'est que **7ᵉ** (rang moyen 7.4). Les meilleurs MCC sont obtenus avec `Output = wavkan_morlet`, mais par des marges de l'ordre de 0.001.
- À noter : avec 5 graines, un test de Wilcoxon apparié entre deux architectures **ne peut pas** descendre sous p = 0.0625 (bilatéral).
  Une comparaison deux à deux « significative à 5 % » est donc impossible avec ce plan. Il faut s'en tenir à Friedman + Nemenyi.

### 2.5 Stabilité

- Graines aberrantes (MCC < 0.968) : 5 runs sur 60, **tous avec wavkan_morlet** sur une porte, ou avec une perte isolée sur la graine 2.
  Exemple : `wavkan_dog/morlet/morlet` graine 2 a un MCC de 0.957.
- R²_symb est la mesure la plus bruitée (σ jusqu'à 0.09 entre graines). Sa résolution est de 1/40, car l'audit est plafonné à 40 arêtes.

### 2.6 Lecture provisoire (TKAN)

> Configuration recommandée à ce stade : **Forget = relukan, Input = efficientkan, Candidate = wavkan_dog, Output = linear**.
> C'est le modèle **le plus petit** (57 761 paramètres), **le plus rapide** (79 ms) et **le plus symbolisable** (R²_symb = 0.76),
> pour une détection équivalente aux autres (MCC 0.9735).
> Formulation honnête : *il domine sur le coût et l'auditabilité, à performance de détection égale*,
> et non *il détecte mieux*.

---

## 3. GRU-KAN : résultats inexploitables en l'état

### 3.1 Les 5 runs terminés ne détectent rien

| Candidate | Graine | MCC | AUC-ROC | PR-AUC | Brier | Latence |
|---|---|---|---|---|---|---|
| wavkan_morlet | 1 | −0.003 | 0.503 | 0.268 | 0.731 | 1182 ms |
| wavkan_morlet | 3 | −0.005 | 0.502 | 0.267 | 0.627 | 1182 ms |
| wavkan_dog | 1 | −0.002 | 0.504 | 0.261 | 0.252 | 1099 ms |
| wavkan_dog | 2 | 0.077 | 0.504 | 0.261 | 0.247 | 1100 ms |
| wavkan_dog | 5 | 0.077 | 0.504 | 0.261 | 0.245 | 1095 ms |

- AUC-ROC ≈ 0.50 et PR-AUC ≈ prévalence (0.255) : c'est **du hasard**.
- Inspection des scores de validation (`val_scores/*.npy`) : **les scores sont effondrés sur une constante**.
  - Pour `wavkan_dog`, 98 % des 993 245 fenêtres reçoivent le même score (0.5047, 0.495 ou 0.4913 selon la graine).
    Il y a seulement 861 à 1 273 valeurs distinctes, contre environ 350 000 pour un TKAN.
  - Pour `wavkan_morlet`, presque tout est à ≈ 0.99 (graine 1) ou ≈ 0.92 (graine 3). Brier 0.73 ≈ 1 − prévalence : le modèle prédit « fraude » pour tout.
- Pourtant, la **perte d'entraînement descend normalement** : 0.70 → 0.05 à 0.12 sur 10 époques.
  Le modèle apprend sur ses 50 000 fenêtres d'entraînement mais ne généralise pas du tout.
- J'ai vérifié le code : `forward` et `forward_with_reg` de `GenericKANLayer` sont identiques, et la correspondance des portes (`GRU_GATE_MAP`) est correcte.
  Ce n'est donc pas une incohérence évidente entre l'entraînement et l'évaluation.

### 3.2 Causes probables (à confirmer)

1. **Capacité démesurée** : le GRU utilise le θ d'élite de l'Étape 2 (`hidden_size = 128` → h_GRU ≈ 149, W = 5).
   Il a donc **1,37 à 1,44 million de paramètres**, soit **≈ 25 fois** le TKAN (32 unités, ≈ 58 000 à 101 000 paramètres), pour seulement 50 000 fenêtres d'entraînement.
   Une mémorisation, avec saturation des portes (sigmoïde/tanh) sur des fenêtres non vues, expliquerait le score constant.
2. **Comparaison non équitable par construction** : GRU et TKAN n'ont ni la même fenêtre (W = 5 vs 10), ni la même taille cachée, ni le même μ₂.
   Ils ne sont pas non plus évalués sur le même jeu de validation : 993 245 fenêtres à 25,5 % de fraude contre 450 955 fenêtres à 34,6 %.
   Même si le GRU fonctionnait, ses métriques ne seraient pas directement comparables à celles du TKAN.
3. **Latence rédhibitoire de toute façon** : 1 100 à 1 180 ms, soit 11 fois la contrainte.
   La pénalité (≈ 1 000) écrase la fitness à −87/−102. Le GRU avec ce θ est hors course quel que soit son MCC.

**Diagnostic suggéré, que tu peux lancer toi-même** : recharger un modèle GRU (`models/7081e7bc88a89dac.pt`) et calculer son AUC sur ses propres fenêtres d'entraînement.
- AUC ≈ 1 sur l'entraînement : sur-apprentissage ou saturation confirmé. Le problème vient de θ, pas du code.
- AUC ≈ 0.5 aussi sur l'entraînement : bug d'évaluation à chercher.

### 3.3 Les 31 échecs : OOM déterministe en validation

- Les 31 échecs sont tous `MPS backend out of memory`. Ils se produisent **au moment de la validation** (`niveau1_harness.py:463`), **après les ≈ 5 000 s d'entraînement**.
- Le lot de validation (`eval_batch_size = 4096`, non exposé en ligne de commande) est trop gros pour un GRU de 149 unités : il demande 1,5 à 2,9 Gio d'un coup, sur 18 Gio maximum.
- **Le problème se répète à chaque relance.** Le checkpoint de fin d'époque 10 est conservé, donc la relance repart directement en validation et plante de nouveau.
  C'est ce qui s'est passé ce matin (13/10, 08:43) pour `morlet/relukan/relukan` graines 2, 4 et 5.
- Correctif simple : passer un `eval_batch_size` plus petit pour le GRU (512, par exemple) ou faire la validation sur CPU.
  Les checkpoints présents dans `checkpoints/` permettraient alors de récupérer ces runs **sans réentraîner**.

---

## 4. Ce qu'on peut écrire dans le mémoire dès maintenant (et ce qu'on ne peut pas écrire)

**Peut être affirmé (sous réserve de la porte Forget et du régime engineered) :**
- Sur MOMTSIM, en régime brut, les 12 TKAN hétérogènes testés atteignent tous MCC ≈ 0.97 et PR-AUC ≈ 0.998. Le choix des bases par porte n'affecte pas significativement la détection en pratique (écarts ≤ 0.004 de MCC).
- Le choix des bases détermine en revanche la latence (79 à 144 ms) et l'auditabilité symbolique (R²_symb de 0 à 0.76).
- La porte de sortie linéaire et l'ondelette DoG sur Candidate sont les choix les plus favorables. Morlet est systématiquement défavorable.

**Ne peut pas encore être affirmé :**
- Que l'hétérogène bat l'uniforme : la baseline n'est pas calculée.
- Quoi que ce soit sur la porte Forget, le régime engineered ou la comparaison GRU vs TKAN.
- Que le rang 1 est « significativement le meilleur » : il ne l'est que face aux rangs 9 à 12.

---

## 5. Priorités suggérées pour la suite

1. **Corriger l'OOM GRU** (`eval_batch_size` réduit pour le GRU) avant de relancer, sinon chaque tentative GRU coûte environ 1h30 pour rien.
2. **Décider du sort du GRU** : soit garder le θ de l'Étape 2 et documenter qu'il est hors contrainte, soit le ramener à une capacité comparable au TKAN (h_LSTM = 32, même W).
   La seconde option donne une comparaison de cellules équitable.
3. Laisser finir les 12 architectures TKAN `Forget = wavkan_dog`.
4. Calculer la **baseline uniforme** : sans elle, la question centrale de l'Étape 4 reste ouverte.
5. Pour la discussion, envisager un graphique **latence vs R²_symb** (front de Pareto). Il rend mieux compte du résultat qu'un classement par fitness scalaire.
