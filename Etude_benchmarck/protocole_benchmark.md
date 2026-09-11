# Protocole de benchmark MKAN  étude des variantes KAN par porte

Document de référence pour l'étude comparative menée dans `MKAN/Etude_benchmarck/`.
Il opérationnalise les six problématiques de [probleme.md](probleme.md) à partir des
fiches techniques de [amélioration_MKAN.md](amélioration_MKAN.md).

---

## 0. Carte des sources primaires

Les six papiers sont disponibles en source LaTeX dans `latex/` :

| Dossier | Papier | Fichier principal | Rôle dans le benchmark |
|---|---|---|---|
| `New folder/` | fKAN  Fractional Jacobi | `main.tex` | Candidat Candidate (oscillant), critère interprétabilité |
| `New folder (2)/` | FastKAN  *KANs are RBF Networks* | `main.tex` | Baseline actuelle (composante gaussienne de MKAN) |
| `New folder (3)/` | BSRBF-KAN | `samplepaper.tex` | **Modèle méthodologique** (protocole + ablation) |
| `New folder (4)/` | Chebyshev-KAN | `main.tex` | Candidat Forget/Input/Candidate, ablation degré |
| `New folder (5)/` | ReLU-KAN | `sn-article.tex` | Candidat haute fréquence + Output léger |
| `Wav-KAN/` | Wav-KAN | `Wav-KAN.tex` | Candidat principal porte Candidate |

Le protocole expérimental de référence est celui de BSRBF-KAN (6 modèles, mêmes
hyperparamètres, 5 runs, ablation par retrait de composant). On le transpose sur les
données de fraude au lieu de MNIST/Fashion-MNIST.

---

## 1. Objet de l'étude

L'architecture actuelle ([cell.py](../cell.py), [hybrid_layer.py](../hybrid_layer.py))
applique **la même fonction d'arête hybride Gaussienne+Fourier** aux quatre portes de la
cellule T-KAN :

```
φ_ij(x) = Σ_m w_m exp(−(x−μ_m)²/2h²)  +  Σ_k [a_k cos(kx) + b_k sin(kx)]
```

avec `M=8`, `K=2` → **12 paramètres par arête**, centres μ_m fixes
([hybrid_edge.py:38-50](../hybrid_edge.py#L38-L50)).

L'hypothèse à tester est que cette homogénéité est sous-optimale : les quatre portes
approximent des familles de fonctions de nature différente (quasi-binaire pour
`f_t`/`i_t`, haute fréquence pour `c̃_t`, quasi-linéaire pour `o_t`), et une
**répartition hétérogène des bases KAN par porte** devrait dominer la configuration
uniforme à budget paramétrique égal.
---

## 2. Deux facteurs transversaux : cellule (LSTM/GRU) et régime d'entrée (brut/traité)

L'étude ne compare pas seulement les bases KAN par porte : **deux facteurs
supplémentaires traversent tous les niveaux**, dès le Niveau 1, et non en fin
d'étude comme extension isolée.

### 2.1 Cellule récurrente : LSTM-KAN vs GRU-KAN

Toute configuration de porte testée au Niveau 1 est instanciée à la fois dans
`TKANCell` (4 portes, LSTM) et dans une cellule GRU-KAN équivalente (3 portes :
reset, update, candidate  chacune une `HybridKANLayer` ou sa variante testée).
Comparaison à **budget de paramètres total égal**, pas à nombre de portes égal :
GRU ayant 3 portes contre 4, on augmente `hidden_size` du GRU jusqu'à égaliser le
compte total de paramètres avec le LSTM correspondant.

### 2.2 Régime d'entrée : données brutes vs données issues du feature engineering

**Brut** = les colonnes transactionnelles avant tout calcul dérivé : `amount`,
`oldBalanceOrig`, `newBalanceOrig`, `oldBalanceDest`, `newBalanceDest`, `step`,
`time_s`, `action` (encodée)  présentes telles quelles dans
`data/val_features.parquet` / `test_features.parquet`.

**Traité** = les features dérivées par le pipeline de feature engineering :
`delta_B_orig`, `delta_B_dest`, `r1`, `r2`, `r1_r2_product`, `flag_anomalie`,
`flag_nuit`, `v1h`, `delta_commission`, `delta_commission_ratio`,
`is_mule_candidate`, `var_agent_split`, `k_fragments`, `mean_historique_30j`,
`rho_rupture`, `rho_refund`, `rho_nouveau`  les 12 features MKAN mentionnées dans
[cell.py](../cell.py).

Chaque configuration de porte est entraînée séparément sur les deux régimes
(`input_size` diffère en conséquence). On ne concatène pas brut+traité par défaut :
c'est un régime supplémentaire à part (« combiné »), rapporté séparément s'il est
testé, pour ne pas masquer l'effet propre de chacun des deux régimes seuls.

### 2.3 Plan de croisement complet  Niveau 1

**Croisement complet dès le Niveau 1** : chaque configuration de porte (référence
hybride + chaque base candidate, cf. §1 Niveau 1 plus bas) est exécutée dans les
**4 cellules** {LSTM, GRU} × {brut, traité}, avec les 5 graines de §2.7. Aucune
présélection sur un seul régime avant croisement  l'objectif explicite est de
détecter les interactions cachées (une base qui gagne en LSTM+traité peut perdre en
GRU+brut, et c'est précisément ce genre d'interaction que l'étude doit mettre au
jour, pas escamoter).

Conséquence chiffrée : le GRU n'ayant pas de porte « output » distincte, son
plan de substitution (reset/update/candidate) est plus petit que celui du LSTM
(forget/input/candidate/output). Avec les listes de candidats effectivement
implémentées ([niveau1_harness.py](niveau1_harness.py) `GATE_CANDIDATES`,
6/6/6 candidats pour LSTM sur forget/input/candidate + 5 sur output, autant pour
les portes homologues du GRU), le plan de croisement complet compte
**82 configurations d'entraînement** (46 en LSTM, 36 en GRU) × {brut, traité} déjà
inclus dans ce compte, chacune sur 5 graines → **410 runs**. C'est le coût accepté
pour ne pas laisser d'interaction non mesurée. Le calendrier d'exécution (§8) et
le budget de calcul doivent être dimensionnés en conséquence ; si le volume
s'avère infaisable en pratique, la réduction se fait par **diminution du nombre
de bases candidates présélectionnées au Niveau 0** (§4), jamais par réduction
discrète du plan de croisement lui-même.

Note d'implémentation : BSRBF-KAN (combinaison B-spline+RBF citée en §0) n'est
pas instanciée comme base séparée dans `edges/bases.py`  la configuration
`hybrid` (Gaussienne+Fourier) déjà présente couvre la même idée de principe
(combiner deux familles de fonctions dans une arête) et sert de test de cette
hypothèse. Ajouter une base BSRBF dédiée reste possible sans changer
l'architecture du harness (`edges/bases.py` + entrée dans `BASIS_REGISTRY`).

---

## 3. Conditions d'équité expérimentale

Toute comparaison qui viole une de ces cinq conditions est rejetée.

### 3.1 Budget paramétrique iso  B = 12 paramètres/arête

C'est le budget de l'arête MKAN actuelle. Chaque variante est instanciée à ce budget :

| Variante | Paramètres/arête | Réglage à B=12 |
|---|---|---|
| MKAN hybride (référence) | M + 2K | M=8, K=2 |
| FastKAN (RBF gaussienne) | M | M=12 |
| FasterKAN (RSWAF `1−tanh²(r/h)`) | M | M=12 |
| EfficientKAN (B-spline ordre 3) | G + 3 | G=9 |
| Chebyshev-KAN | d + 1 | d=11 |
| Wav-KAN (ondelette unique) | 3 (w, τ, s) | mélange de 4 ondelettes |
| ReLU-KAN | 3 par base (s_i, e_i, w_i) | 4 bases |
| Fourier pur (KAN-AD) | 2K | K=6 |
| fKAN (Jacobi fractionnaire) | q + 3 (α, β, γ) | q=9 |
| Linéaire + σ (baseline non-KAN) | 1 |  |

**Second point de mesure obligatoire : le réglage natif optimal** publié par chaque
papier (Chebyshev d=3, Wav-KAN ondelette unique, ReLU-KAN selon le papier). Le budget
iso mesure l'expressivité à coût fixe ; le réglage natif mesure la performance
atteignable. Les deux sont rapportés  ne jamais n'en publier qu'un.

### 3.2 Données, découpage, fenêtrage

- Jeux : `data/val_features.parquet` (sélection de modèle), `data/test_features.parquet`
  (mesure finale, **un seul passage, à la toute fin de l'étude**).
- Découpage **temporel strict**, jamais aléatoire : la fraude dérive, un split
  stratifié aléatoire fuit de l'information du futur.
- Fenêtre glissante W identique pour tous les modèles (celle de `MKANScorer`).
- Normalisation identique. Contrainte forte : les bases de Chebyshev et Jacobi
  **exigent** une entrée dans [−1,1] ; les RBF et ondelettes exigent que l'entrée reste
  dans le domaine des centres. On applique donc à toutes les variantes la même
  normalisation d'entrée de couche (LayerNorm), y compris à celles qui n'en ont pas
  besoin, pour ne pas confondre l'effet de base avec l'effet de normalisation.

### 3.3 Optimisation

Repris de BSRBF-KAN pour comparabilité avec la littérature :
AdamW, `lr=1e-3`, `weight_decay=1e-4`, `gamma=0.8` (ExponentialLR), batch=64.
Même nombre d'époques, même *early stopping* (patience sur AUC-PR validation),
même loss (celle de [loss.py](../loss.py), régularisation λ_L1 et λ_entropie
inchangée entre variantes).

### 3.4 Répétitions et graines

**5 runs par configuration**, graines `{0,1,2,3,4}` fixées et journalisées.
Tout résultat est rapporté en **moyenne ± écart-type**. Un écart de moyenne inférieur
à l'écart-type inter-runs n'est pas une différence  il est déclaré non concluant.

### 3.5 Matériel et mesure de coût

Un seul type de GPU pour toute l'étude ; le modèle de GPU est journalisé.
Les latences sont mesurées après 10 itérations de chauffe, avec
`torch.cuda.synchronize()` avant et après, sur 100 itérations.

---

## 4. Structure en trois niveaux

L'étude procède du plus isolé au plus intégré. Un niveau ne démarre que si le
précédent a produit une présélection.

### Niveau 0  Approximation unitaire d'arête (hors architecture)

**But** : caractériser chaque base sur les familles de fonctions que chaque porte doit
réellement approximer, sans le bruit de l'entraînement récurrent complet. C'est ce
niveau qui rend l'étude *explicative* et pas seulement comparative.

Cibles synthétiques 1-D, régression sur 1000 points, MSE finale :

| Famille | Cible | Porte visée |
|---|---|---|
| Seuil dur | `1[x>0.3]` lissée (sigmoïde raide, pente 20) | Forget, Input |
| Seuil double | fonction créneau sur [−0.2, 0.5] | Forget, Input |
| Quasi-linéaire | `0.8x + 0.05` | Output |
| Haute fréquence | `sin(5πx) + x` (cible f₂ de ReLU-KAN) | Candidate |
| Multi-échelle | `sin(2πx) + 0.3 sin(20πx)` | Candidate |
| Périodicité + bruit | multi-échelle + bruit gaussien σ=0.1 | Candidate (robustesse) |

Sortie : matrice **base × famille**, MSE moyenne sur 5 graines, plus le nombre
d'époques jusqu'à convergence. La dernière ligne (bruit) départage spécifiquement
les bases qui surapprennent le bruit  argument central de Wav-KAN contre Spl-KAN.

### Niveau 1  Substitution par porte dans MKAN

**But** : répondre aux problématiques 1, 2 et 3, **croisé avec les deux facteurs
transversaux du §2** (LSTM/GRU, brut/traité  4 combinaisons par configuration).

Protocole : on remplace la base d'**une seule porte** à la fois, les trois autres
gardant l'arête hybride actuelle. Cela isole l'effet de la porte et évite l'explosion
combinatoire (4 portes × 10 variantes = 40 runs de substitution simple, contre 10⁴ en
grille complète)  avant application du croisement ×4 du §2.3.

- **Forget** : hybride (réf.), FastKAN, FasterKAN, EfficientKAN, ReLU-KAN,
  BSRBF-KAN, Chebyshev  7 configurations
- **Input** : mêmes 7 configurations
- **Candidate** : hybride (réf.), Wav-KAN (Morlet, Mexican hat, DOG  trois runs
  distincts, le papier montre que le choix de l'ondelette est critique et que Shannon
  sous-performe), ReLU-KAN, Chebyshev d=3, Fourier pur, fKAN  8 configurations
- **Output** : hybride (réf.), FasterKAN, ReLU-KAN, EfficientKAN G=2,
  **linéaire+σ**  5 configurations

La configuration `linéaire+σ` sur Output est la mesure directe du surcoût réel d'un KAN
sur une projection quasi-linéaire : si elle n'est pas battue de façon significative
(cf. §3.4), la conclusion est que la porte Output n'a pas besoin de KAN, et c'est un
résultat publiable en soi.

**Puis composition** : la meilleure variante de chaque porte est combinée en une
configuration hétérogène unique, comparée à MKAN uniforme. C'est le test de
l'hypothèse principale. La composition n'est pas garantie additive  il faut la
mesurer, pas la déduire.

### Niveau 2  Variations structurelles

**But** : problématiques 4, 5 et 6. Aucune de ces trois questions n'a de source
publiée (cf. amélioration_MKAN.md §« contribution originale »), ce qui en fait la
partie originale du mémoire  et impose une rigueur supérieure sur les baselines.

La problématique 4 (GRU vs LSTM) est déjà traitée transversalement dès le Niveau 1
(§2.1, §2.3) et n'est pas reprise ici comme étude séparée. Ce qui reste au Niveau 2
porte sur les deux questions structurelles restantes, mesurées sur la configuration
hétérogène retenue à l'issue du Niveau 1 (dans ses 4 variantes LSTM/GRU × brut/traité) :

**2.a  Étude parallèle des arêtes, au-delà de la comparaison de performance
(problématique 5)**
Le §2.2 mesure déjà l'effet du régime d'entrée sur la performance de bout en bout.
Ici on va plus loin : sur une même porte, comparer les fonctions d'arête *apprises*
elles-mêmes selon le régime (amplitude L1 par arête via `exact_l1_norm`, entropie de
couche via `layer_entropy`) pour déterminer si le prétraitement déplace la charge
d'approximation d'une arête à l'autre, et si une base donnée est plus robuste au
changement de distribution d'entrée. Croiser avec `drift.py`.

**2.b  MultKAN intra-Candidate (problématique 6)**
`node_types` accepte déjà des paires multiplicatives
([cell.py:54-59](../cell.py#L54-L59)). Tester l'activation de nœuds produits dans la
porte Candidate seule, avec sélection des paires (i₁,i₂) par les scores d'importance
existants plutôt qu'exhaustivement. Référence : section MultKAN de KAN 2.0.

---

## 5. Métriques

### 5.1 Performance (fraude, classes très déséquilibrées)

- **AUC-PR**  métrique primaire de décision. L'AUC-ROC seule est trompeuse à ce
  niveau de déséquilibre.
- AUC-ROC  reporté pour comparabilité externe.
- **Précision @ rappel fixé** (rappel = 0.5 et 0.8)  c'est la métrique
  opérationnelle : elle dit combien de faux positifs un analyste absorbe.
- F1 au seuil optimisé sur validation, appliqué tel quel au test.
- **Calibration** : Brier score + courbe de fiabilité. Un scoreur de risque mal calibré
  est inutilisable pour un seuillage réglementaire, même avec un bon AUC-PR.

### 5.2 Coût

Nombre de paramètres total ; latence forward par transaction (µs, conditions §3.5) ;
latence forward+backward ; temps d'entraînement par époque ; pic de mémoire GPU.

### 5.3 Interprétabilité  critère éliminatoire

Contrainte COBAC : une décision doit être explicable. On mesure :
- la **localité** de la base (une base globale  Chebyshev, Jacobi  rend l'attribution
  par région d'entrée impossible ; fKAN le reconnaît explicitement comme sa limitation) ;
- la sparsité effective après régularisation L1 (part des arêtes dont `exact_l1_norm`
  est sous le seuil d'élagage), via [audit.py](../audit.py) ;
- la faisabilité de l'extraction symbolique existante ([symbolic.py](../symbolic.py)).

Une variante qui gagne en AUC-PR mais rend l'extraction symbolique impossible est
**écartée**, avec le gain chiffré mentionné en discussion. Ce point doit être posé
avant les résultats, pas après, sous peine d'être un choix *post hoc*.

---

## 6. Règles de décision

1. Une variante remplace la référence sur une porte si elle améliore l'AUC-PR de test
   d'un écart supérieur à l'écart-type inter-runs **et** ne dégrade ni la latence de
   plus de 20 % ni le critère d'interprétabilité §5.3.
2. À performance statistiquement indiscernable, on retient la variante la moins
   coûteuse (paramètres, puis latence).
3. Le jeu de test n'est utilisé qu'une fois, en fin d'étude, sur les configurations
   présélectionnées en validation. Toute exploration se fait sur validation.
4. Les résultats non concluants sont publiés comme tels. Un « pas de différence
   significative entre X et Y » est un résultat, pas un échec.

---

## 7. Ablation finale

Sur la configuration hétérogène retenue (dans sa meilleure combinaison
cellule × régime issue du §2), répliquer la méthodologie d'ablation de BSRBF-KAN
(retrait d'un composant à la fois) :

- sans composante gaussienne ; sans composante Fourier (mesure la contribution réelle
  de l'hybridation, question laissée ouverte par le mémoire) ;
- sans layer normalization (BSRBF-KAN montre que c'est un composant critique) ;
- sans base output `b(x)` ;
- sans régularisation L1 ; sans régularisation entropique ;
- sans nœuds MultKAN ;
- retour à la configuration uniforme (contrôle négatif) ;
- retour à la cellule LSTM si la meilleure combinaison est en GRU, et inversement
  (contrôle négatif sur le facteur cellule) ;
- retour à l'autre régime d'entrée (contrôle négatif sur le facteur brut/traité).

---

## 8. Ordre d'exécution

| Étape | Contenu | Dépend de |
|---|---|---|
| 1 | Implémenter les bases dans une interface d'arête commune ; implémenter la cellule GRU-KAN |  |
| 2 | Niveau 0  matrice base × famille de fonctions | 1 |
| 3 | Présélection : 3–4 candidats par porte | 2 |
| 4 | Niveau 1  substitution par porte × {LSTM, GRU} × {brut, traité} (validation) | 3 |
| 5 | Composition hétérogène + comparaison à MKAN uniforme, sur les 4 combinaisons | 4 |
| 6 | Niveau 2  arêtes parallèles, MultKAN | 5 |
| 7 | Ablation finale | 6 |
| 8 | **Passage unique sur le jeu de test** | 7 |

L'étape 1 est le prérequis technique de tout le reste : toutes les variantes doivent
implémenter la même interface que `HybridKANLayer` (`forward`, `forward_with_reg`,
`exact_l1_norm`, `layer_entropy`, `edge_activations`) afin que la boucle
d'entraînement, la loss régularisée, l'audit et le visualiseur restent inchangés d'une
variante à l'autre. La cellule GRU-KAN doit exposer la même interface que `TKANCell`
(`forward`, `forward_with_reg`) pour que `MKANScorer` s'en serve sans modification.
Sans cela, aucune des conditions d'équité du §3 n'est vérifiable.
