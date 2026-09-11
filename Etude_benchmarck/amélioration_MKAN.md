Excellent, j'ai maintenant six papiers exploitables. Voici l'extraction technique et ce qu'ils permettent de construire.

---

## Fiches techniques des sources disponibles

**FastKAN  Li, Z. *Kolmogorov-Arnold Networks are Radial Basis Function Networks***
- Base : RBF gaussienne φ(r) = exp(−r²/2h²), r = ‖x − c‖, centres fixes
- Contribution théorique : les B-splines d'ordre 3 sont bien approximées par des RBF gaussiennes sous transformation linéaire → « KAN sont des réseaux RBF à centres fixes »
- Layer normalization pour maintenir les entrées dans le domaine des RBF
- Benchmark publié : 8 centres RBF vs efficient_kan (5 grilles, B-splines ordre 3, 8 paramètres/entrée). **Accélération forward ×3.33** (223 μs vs 742 μs), forward+backward ×1.25. GPU V100, couche 100×100.
- Précision MNIST : équivalente ou supérieure à efficient_kan, structure [784, 64, 10], 20 époques
- Code : github.com/ZiyaoLi/fast-kan

**Wav-KAN  Bozorgasl & Chen**
- Base : ondelettes mères ψ((t−τ)/s), avec poids w, translation τ, scaling s apprenables
- Ondelettes testées : Mexican hat, Morlet ψ(t)=cos(ω₀t)e^(−t²/2) avec ω₀=5, DOG, Shannon
- **Complexité paramétrique : O(3N²L)** vs Spl-KAN O(N²LG) vs MLP O(N²L). Le coefficient 3 = poids + translation + scaling
- Argument central pour ton cas : les ondelettes font un compromis entre représenter la structure sous-jacente et **ne pas surapprendre le bruit**. Citation directe : « la force de Spl-KAN est aussi sa faiblesse » (il capture le bruit d'entraînement)
- Pas besoin du terme b(x) additionnel de Spl-KAN grâce à la propriété de scaling intrinsèque → plus rapide
- Batch normalization améliore significativement Wav-KAN et Spl-KAN
- Benchmark : MNIST, [784,32,10], AdamW lr=0.001 weight_decay=1e-4, 5 essais × 50 époques. DOG et Mexican hat > Spl-KAN en test. **Shannon et Bump sous-performent**  le choix de l'ondelette est critique
- Code : github.com/zavareh1/Wav-KAN

**BSRBF-KAN  Ta, H.-T.**
- Base combinée : BSRBF(x) = w_b·b(x) + w_s·(BS(x) + RBF(x)), avec layer normalization
- **C'est le papier le plus directement utile pour toi** : il valide le principe de combinaison de deux bases dans une même arête, exactement ce que fait ton φ_ij hybride Gaussienne+Fourier
- Benchmark complet sur 6 modèles : BSRBF-KAN, FastKAN, FasterKAN, EfficientKAN, GottliebKAN, MLP. MNIST + Fashion-MNIST, 5 runs, structure (784,64,10), batch=64, lr=1e-3, weight_decay=1e-4, gamma=0.8, AdamW
- Résultats moyens (MNIST+FMNIST) : BSRBF-KAN 93.44% (221s), FastKAN 93.34% (131.5s), FasterKAN 93.27% (123.5s), EfficientKAN 93.15% (151.5s), GottliebKAN 92.18%, MLP 93.01%
- **Étude d'ablation directement transposable à ta méthodologie** : composants retirés un à un (No BS, No RBF, No BO, No LN). Résultat clé : layer normalization et base output sont critiques ; retirer les RBF impacte moins que retirer les B-splines
- Introduit FasterKAN (RSWAF) : φ(r) = 1 − tanh²(r/h)  variante RBF encore plus rapide
- Code : github.com/hoangthangta/BSRBF_KAN

**ReLU-KAN  Qiu, Zhu, Gong, Chen, Ning**
- Base : R_i(x) = [ReLU(e_i − x) × ReLU(x − s_i)]² × 16/(e_i − s_i)⁴
- **s_i et e_i sont entraînables** (position et forme adaptatives)  c'est la contribution clé vs B-splines statiques
- Uniquement additions matricielles, produits scalaires et ReLU → parallélisation GPU par convolution
- Benchmark : **5 à 20× plus rapide** que KAN, précision 1 à 3 ordres de grandeur supérieure. Exemple f₆ (4 variables) : KAN 3.47e-1, ReLU-KAN¹ (S,E figés) 8.83e-3, ReLU-KAN² (S,E entraînables) 6.37e-4
- **Point directement pertinent pour ta porte Candidate** : les auteurs notent explicitement que les B-splines « ne peuvent pas changer leur position et forme pendant l'entraînement, ce qui rend difficile l'ajustement des fonctions à haute fréquence de variation dans le domaine ». L'avantage est prononcé sur f₂ = sin(5πx) + x
- Préserve la résistance à l'oubli catastrophique de KAN (validé expérimentalement en 5 phases)
- Code : github.com/quiqi/relu_kan

**Chebyshev-KAN  Sidharth SS, Gokul R, Anas KP, Keerthana AR**
- Base : T_n(x) = cos(n·arccos(x)), récurrence T_n = 2x·T_{n−1} − T_{n−2}
- Normalisation obligatoire par tanh pour ramener dans [−1,1]
- Coefficients apprenables C ∈ R^(input_dim × output_dim × (degree+1)), calcul par einsum
- **Ablation sur le degré, très utile pour ton espace hyperparamétrique** : degré 2 → 96.97% (77 376 params), degré 3 → **97.18%** (103 136), degré 4 → 95.53% (128 896), degré 5 → 96.46% (154 656). Le degré 3 est optimal, le 4 s'effondre
- Ablation normalisation : tanh 96.80%, Min-Max 96.83%, **Standardisation 96.92%**
- Ablation initialisation : Xavier meilleur global (98.0% test), Normal le pire (97.0%)
- Code : github.com/sidhu2690/ChebyshevKAN

**fKAN  Aghaei, A.A.**
- Base : polynômes de Jacobi fractionnaires avec α, β, γ **entraînables**
- φ(ζ) = J_q^(ELU(α,1), ELU(β,1))(φ_(0,1;σ(γ))(σ(ζ)))
- ELU contraint α,β > −1 (convergence Jacobi) ; sigmoïde contraint γ ∈ (0,1)
- **Figure hiérarchique essentielle** : Chebyshev, Gegenbauer, Legendre sont des cas particuliers de Jacobi, lui-même sous-classe de l'hypergéométrique. Chebyshev 1ère espèce = Jacobi(α=β=−1/2)
- Choix architectural pertinent pour toi : les auteurs rejettent la batch normalization au profit d'une activation bornée (sigmoïde) pour cause d'incompatibilité avec petits batchs et apprentissage en ligne  **directement pertinent pour ton scoring temps réel**
- Coût : ~2× plus lent que sigmoïde en CPU, 3× plus lent que ReLU
- Limitation reconnue par les auteurs : **interprétabilité réduite vs KAN à cause de la nature globale des fonctions de base**  c'est un critère disqualifiant potentiel pour ta contrainte COBAC
- Code : github.com/alirezaafzalaghaei/fKAN

---

## Ce que ces sources permettent de construire

**Pour Forget & Input (décision quasi-binaire, seuillage local)**
Candidats ancrés : FastKAN (RBF), FasterKAN (RSWAF), EfficientKAN (B-splines), ReLU-KAN, BSRBF-KAN. Les cinq ont des benchmarks publiés comparables (MNIST/Fashion-MNIST, mêmes hyperparamètres) dans BSRBF-KAN, ce qui te donne un protocole de comparaison déjà validé à répliquer sur tes données.

**Pour Candidate (haute fréquence, localisation temps-fréquence)**
Trois compétiteurs légitimes et documentés : Wav-KAN (Morlet/Mexican hat/DOG), ReLU-KAN (basis à position entraînable  argument explicite sur les hautes fréquences), Chebyshev-KAN (degré 3). Ton KAN-AD actuel (Fourier) est le quatrième. C'est exactement le benchmark à 3-4 que tu voulais.

**Pour Output (quasi-linéaire, léger)**
FasterKAN (le plus rapide du benchmark BSRBF-KAN : 123.5s moyenne), ReLU-KAN, EfficientKAN, plus la baseline linéaire+σ. L'ablation BSRBF-KAN te donne le modèle méthodologique.

**Ce qui reste sans source et devient contribution originale**
GRU-KAN, MultKAN intra-porte, étude parallèle des arêtes, et la répartition hétérogène des types KAN par porte. Aucun des six papiers ne fait ça.

---

