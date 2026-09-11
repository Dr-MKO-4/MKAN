Parfait. Je lance une recherche ciblée sur les variantes KAN récentes pour chaque problématique.arxiv.org n'est pas dans les domaines autorisés de mon réseau. Je ne peux pas faire la recherche directement. Mais je peux faire mieux : te donner la liste structurée des articles à aller chercher toi-même, avec les identifiants arXiv exacts et ce qu'on cherche précisément dans chacun pour chaque problématique.

Voici le corpus cible, organisé par rôle dans ton benchmark. **Attention : vérifie chaque référence car je ne peux pas accéder à arXiv pour confirmer, certains identifiants peuvent être approximatifs.**

---

## Articles à chercher par problématique

### Problématique 1  Portes Forget & Input (décision quasi-binaire, seuillage local)

Ce qu'on cherche : variantes KAN performantes sur des fonctions monotones, sigmoïdes, à seuil. Les candidats sont ceux qui ont une bonne localisation scalaire sans coût paramétrique élevé.

**à chercher sur arXiv/Google Scholar :**

- **FastKAN**  Li, Z. (2024). *FastKAN: Very Fast Kolmogorov-Arnold Networks*. arXiv:2408.07785  déjà dans ton mémoire, c'est la baseline

- **BSRBF-KAN**  Ta, D.T. (2024). *BSRBF-KAN: A combination of B-spline and Radial Basis Functions in Kolmogorov-Arnold Networks*. arXiv:2406.11173  combine B-splines et RBF, intéressant pour les fonctions à seuil car plus flexible que FastKAN pur

- **ReLU-KAN**  Qiu, Y. et al. (2024). *ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU*. arXiv:2406.02075  extrêmement rapide, opérations matricielles pures, pertinent si la porte doit être légère

- **Chebyshev-KAN**  Sidharth, S.S. (2024). *Chebyshev Polynomial-Based Kolmogorov-Arnold Networks*. arXiv:2405.07200  polynômes de Chebyshev comme base, très bonne approximation des fonctions monotones et sigmoïdes

- **EfficientKAN**  Blealtan (2024). *EfficientKAN*  GitHub : https://github.com/Blealtan/efficient-kan  B-splines optimisées, faible empreinte mémoire

---

### Problématique 2  Porte Candidate (haute fréquence, localisation temps-fréquence)

Ce qu'on cherche : variantes avec bonne capacité de décomposition multi-échelle, localisation temps-fréquence, robustesse au bruit.

- **Wav-KAN**  Bozorgasl, A. & Chen, H. (2024). *Wav-KAN: Wavelet Kolmogorov-Arnold Networks*. arXiv:2405.12832  ondelettes de Morlet, Mexican Hat, Haar ; déjà cité dans ton mémoire, c'est le candidat principal

- **KAN-AD**  Zhou, X. et al. (2025). *KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks*. arXiv:2501.xxxxx  déjà dans ton architecture, c'est le compétiteur à battre

- **Jacobi-KAN**  Aghaei, A.A. (2024). *fKAN: Fractional Kolmogorov-Arnold Networks with Trainable Jacobi Basis Functions*. arXiv:2406.07456  polynômes de Jacobi paramétriques, très expressifs pour les fonctions oscillantes et les patterns à multi-échelles

- **FourierKAN**  Xu, J. et al. (2024). *FourierKAN-GCF: Fourier Kolmogorov-Arnold Network  An Effective and Efficient Feature Transformation for Graph Collaborative Filtering*. arXiv:2406.01034  variante Fourier pure, à comparer directement à KAN-AD sur la composante périodique

- **SincKAN**  à chercher sur Google Scholar : *SincKAN* ou *Sinc Kolmogorov-Arnold* (2024-2025)  fonctions Sinc qui ont une localisation fréquentielle naturelle très précise, pertinent pour les périodicités intra-journalières

---

### Problématique 3  Porte Output (quasi-linéaire, projection légère)

Ce qu'on cherche : le KAN le plus léger possible qui ne sur-paramétrise pas une projection quasi-linéaire.

- **EfficientKAN**  même référence que dessus  B-splines degré 3 grille minimale G=3, c'est le candidat naturel

- **ReLU-KAN**  même référence  encore plus léger, à tester comme alternative extrême

- **KAN original avec grille réduite**  Liu et al. (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756  déjà dans ton mémoire, à tester avec G=2 comme borne inférieure

- À tester aussi : **couche linéaire + σ pure** comme baseline non-KAN pour mesurer le surcoût réel d'un KAN sur cette porte

---

### Problématique 4  Structure récurrente GRU-KAN vs LSTM-KAN

Ce qu'on cherche : implémentations GRU avec activations KAN, études comparatives GRU vs LSTM dans les architectures KAN.

- **T-KAN / KAN-LSTM**  Zhu, X. et al. (2025). *KAN See In the Dark*.  chercher sur arXiv : *LSTM KAN recurrent* 2024-2025

- **GRU-KAN**  chercher sur Google Scholar : *GRU Kolmogorov-Arnold Network* 2024-2025. Pas de papier canonique unique à ma connaissance mais plusieurs implémentations expérimentales sont apparues en 2024-2025

- **RKAN**  chercher : *Recurrent Kolmogorov-Arnold Network* arXiv 2024  des papiers proposent des architectures récurrentes KAN génériques qui couvrent à la fois LSTM et GRU

- **TimesNet / PatchTST avec KAN**  pour voir comment la communauté séries temporelles intègre KAN dans les cellules récurrentes, utile pour le benchmark GRU vs LSTM

---

### Problématique 5  Niveau 2 : étude des arêtes (données brutes + traitées en parallèle)

Ce qu'on cherche : articles sur la sensibilité des fonctions d'arête KAN selon le prétraitement des entrées.

- **KAN 2.0**  Liu, Z. et al. (2024). *KAN 2.0: Kolmogorov-Arnold Networks Meet Science*. arXiv:2408.10205  introduit MultKAN et discute de la sensibilité des arêtes selon les transformations d'entrée

- **BSRBF-KAN**  même référence  compare explicitement l'apprentissage des arêtes sur différentes distributions d'entrée

---

### Problématique 6  MultKAN intra-Candidate

Ce qu'on cherche : articles sur les nœuds multiplicatifs dans KAN, intégration dans des cellules récurrentes.

- **KAN 2.0**  arXiv:2408.10205  section MultKAN, c'est la référence principale

- **Multiplicative KAN / Product KAN**  chercher sur Google Scholar : *product nodes KAN* ou *multiplicative Kolmogorov-Arnold* 2024-2025

---

## Stratégie de recherche recommandée

Pour chaque article, voici ce que tu dois extraire avant de concevoir le benchmark :
1. La formule de la fonction de base (ce qui remplace φ_ij)
2. Le nombre de paramètres par arête en fonction de la dimension d'entrée
3. La complexité computationnelle au passage avant
4. Les benchmarks déjà publiés (datasets, métriques)
5. Le code disponible (GitHub)

Les mots-clés Google Scholar à utiliser en complément : `"Kolmogorov-Arnold" site:arxiv.org 2024`, `KAN recurrent time series fraud`, `KAN wavelet anomaly detection`.

---

Une fois que tu as récupéré ces articles, on peut construire ensemble le protocole de benchmark formel pour chaque niveau  critères de comparaison, métriques, conditions d'équité expérimentale. Tu veux qu'on prépare ça maintenant en parallèle ?