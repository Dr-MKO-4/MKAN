"""
Recherche heuristique pour l'optimisation des hyperparamètres de MKANScorer.

Algorithme génétique amélioré :
  • Élitisme + sélection par roulette + tournoi
  • Croisement à un point + mutation adaptative avec refroidissement
  • Contrôle de la diversité (Hao & Solnon, 2024 — prévenir la convergence prématurée)
  • Composante EDA : modèle probabiliste sur les meilleures solutions (Hao & Solnon, 2024)
  • Taux de mutation décroissant (analogie recuit simulé, Hao & Solnon, 2024)

Stack : PyTorch / Plotly / pandas uniquement. Aucune dépendance sklearn / scipy / optuna / tqdm.

─── Connexion au workflow train.ipynb ────────────────────────────────────────────
    fitness_fn = build_mkan_fitness(
        df_train       = df_train,        # DataFrame normalisé (train)
        df_val         = df_val,          # DataFrame normalisé (val)
        make_windows   = make_windows,    # fonction définie dans train.ipynb
        device         = DEVICE,
        input_size     = INPUT_SIZE,      # 12
        n_epochs       = 10,              # epochs rapides pour le ranking
    )

    search = RechercheHeuristique(
        espace            = ESPACE_MKAN_DEFAULT,  # inclut W et batch_size
        fitness_fn        = fitness_fn,
        n_generations     = 20,
        taille_population = 12,
        elite_size        = 2,
        mutation_rate     = 0.35,
        refroidissement   = 0.97,
        min_diversite     = 0.40,
        patience          = 5,
        maximize          = True,
    )
    best = search.fit()

    search.plot_convergence().show()
    search.plot_parameter_space().show()
    search.plot_diversity().show()
─────────────────────────────────────────────────────────────────────────────────
"""

import math
import random
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Espace de recherche par défaut pour MKANScorer ────────────────────────────
# Basé sur l'analyse de sensibilité de Yang Lu & Felix Zhan (2024) :
# M, K et hidden_size sont les hyperparamètres les plus critiques pour les KAN
# appliqués à la détection de fraude.
ESPACE_MKAN_DEFAULT = {
    'hidden_size': [16, 32, 64, 128],  # dimension des états cachés h_t / c_t
    'M':           [4, 8, 16, 32],     # centres gaussiens par arête (eq. 4.13)
    'K':           [1, 2, 4],          # harmoniques Fourier par arête (eq. 4.13)
    'lam':         [1e-3, 5e-3, 1e-2], # force de régularisation globale (eq. 4.27)
    'mu1':         [0.5, 1.0, 2.0],    # poids norme L1  (eq. 4.27)
    'mu2':         [0.25, 0.5, 1.0],   # poids entropie  (eq. 4.27)
    'lr':          [5e-4, 1e-3, 3e-3], # taux d'apprentissage Adam
}

# ── Espace étendu : inclut W et batch_size (optionnel) ────────────────────────
# W et batch_size nécessitent de reconstruire les fenêtres / DataLoaders à
# chaque évaluation — coût supérieur, mais exploration plus complète.
# Utiliser ESPACE_MKAN_ETENDU à la place de ESPACE_MKAN_DEFAULT pour les activer.
ESPACE_MKAN_ETENDU = {
    **ESPACE_MKAN_DEFAULT,
    'W':          [5, 10, 15, 20],     # taille de la fenêtre glissante (pas de temps)
    'batch_size': [64, 128, 256, 512], # taille des mini-lots
}


def build_mkan_fitness(
    df_train,
    df_val,
    make_windows,
    device,
    input_size:         int   = 12,
    n_epochs:           int   = 10,
    default_W:          int   = 10,
    default_batch_size: int   = 256,
) -> "callable":
    """
    Fabrique une fonction fitness connectée au workflow MKAN (train.ipynb).

    Accepte les DataFrames normalisés et la fonction make_windows du notebook
    afin de pouvoir reconstruire les fenêtres glissantes et les DataLoaders
    pour chaque combinaison (W, batch_size) testée par la recherche.

    Chaque appel à la fonction retournée :
      1. Reconstruit les fenêtres train/val avec la valeur W du paramètre.
      2. Instancie MKANScorer avec les hyperparamètres proposés.
      3. Entraîne le modèle pendant n_epochs avec le batch_size proposé.
      4. Calcule le MCC sur val (métrique principale, eq. 3.1).
      5. Retourne ce MCC comme score de fitness.

    Args:
        df_train       : DataFrame pandas normalisé (split train).
        df_val         : DataFrame pandas normalisé (split val).
        make_windows   : callable(df, W) -> (X_np, y_np) — défini dans train.ipynb.
        device         : torch.device ('cpu' ou 'cuda').
        input_size     : nombre de features d'entrée (12 dans le pipeline).
        n_epochs       : epochs d'entraînement rapide pour le ranking.
        default_W      : fenêtre par défaut si 'W' absent du dict params.
        default_batch_size : batch_size par défaut si absent du dict params.

    Returns:
        fitness_fn(params: dict) -> float  (MCC ∈ [-1, 1]).
        params peut contenir : hidden_size, M, K, W, batch_size, lam, mu1, mu2, lr.
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from MKAN import MKANScorer, mkan_total_loss

    # État partagé : conserve les poids du meilleur modèle vu jusqu'ici.
    # Accessible après search.fit() via fitness_fn.meilleur['state_dict'].
    _meilleur = {'score': float('-inf'), 'state_dict': None, 'params': None}

    def fitness_fn(params: dict) -> float:
        hidden_size = int(params.get('hidden_size',  32))
        M           = int(params.get('M',             8))
        K           = int(params.get('K',             2))
        W           = int(params.get('W',            default_W))
        batch_size  = int(params.get('batch_size',   default_batch_size))
        lam         = float(params.get('lam',   1e-2))
        mu1         = float(params.get('mu1',   1.0))
        mu2         = float(params.get('mu2',   0.5))
        lr          = float(params.get('lr',    1e-3))

        # Reconstruction des fenêtres pour ce W (W peut varier à chaque évaluation)
        X_tr, y_tr = make_windows(df_train, W)
        X_vl, y_vl = make_windows(df_val,   W)

        if len(X_tr) == 0 or len(X_vl) == 0:
            return -1.0   # W trop grand pour les données disponibles

        def _loader(X, y, shuffle):
            ds = TensorDataset(torch.tensor(X), torch.tensor(y))
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        tr_loader = _loader(X_tr, y_tr, shuffle=True)
        vl_loader = _loader(X_vl, y_vl, shuffle=False)

        model = MKANScorer(
            input_size  = input_size,
            hidden_size = hidden_size,
            M           = M,
            K           = K,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(n_epochs):
            for X_batch, y_batch in tr_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss, *_ = mkan_total_loss(
                    model, X_batch, y_batch, lam=lam, mu1=mu1, mu2=mu2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # MCC sur val (eq. 3.1)
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in vl_loader:
                all_scores.append(model(X_batch.to(device)).cpu().numpy())
                all_labels.append(y_batch.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels).astype(int)
        preds  = (scores >= 0.5).astype(int)

        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        num = tp * tn - fp * fn
        den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = float(num / (den + 1e-12))

        # Sauvegarde des poids si ce modèle est le meilleur vu jusqu'ici
        if mcc > _meilleur['score']:
            _meilleur['score']      = mcc
            _meilleur['params']     = dict(params)
            _meilleur['state_dict'] = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        return mcc

    # Exposé pour récupérer les poids après search.fit()
    fitness_fn.meilleur = _meilleur
    return fitness_fn


class RechercheHeuristique:
    """
    Optimisation heuristique des hyperparamètres par algorithme génétique amélioré.

    Trois améliorations par rapport à l'algorithme génétique classique
    (Hao & Solnon, 2024 ; Yang Lu & Felix Zhan, 2024) :

    1. **Mutation adaptative** — le taux de mutation décroît d'un facteur
       `refroidissement` à chaque génération (analogie recuit simulé) :
       μ_{t+1} = max(min_mutation_rate, μ_t × refroidissement).
       Favorise l'exploration en début de recherche, l'exploitation en fin.

    2. **Contrôle de la diversité** — si la fraction d'individus uniques tombe
       sous `min_diversite`, des individus aléatoires remplacent les pires
       non-élites (Hao & Solnon §4.3 : prévenir la convergence prématurée).

    3. **Composante EDA** — un modèle probabiliste est construit sur les meilleurs
       individus (fréquences empiriques + lissage de Laplace). Un tiers des enfants
       est généré par tirage selon ce modèle, guidant vers les régions prometteuses
       (Hao & Solnon §5).

    Args:
        espace             : {str: list} — espace de recherche discret.
                             Utiliser `ESPACE_MKAN_DEFAULT` pour MKANScorer.
        fitness_fn         : callable(dict) -> float.
                             Utiliser `build_mkan_fitness(...)` pour le workflow MKAN.
        n_generations      : nombre max de générations.
        taille_population  : individus par génération (> elite_size + 1).
        elite_size         : individus préservés sans modification.
        tournament_size    : taille des groupes pour la sélection par tournoi.
        mutation_rate      : taux initial de mutation ∈ [0, 1].
        refroidissement    : facteur multiplicatif du taux de mutation par génération
                             (1.0 = aucun, 0.90 = fort, 0.97 = doux — recommandé).
        min_mutation_rate  : plancher du taux de mutation après refroidissement.
        min_diversite      : seuil de diversité (fraction d'individus uniques ∈ [0, 1]).
                             En dessous : injection d'individus aléatoires.
        patience           : arrêt anticipé si aucune amélioration sur N générations.
        maximize           : True → maximise (MCC, AUC) / False → minimise (loss).
    """

    def __init__(
        self,
        espace:             dict,
        fitness_fn,
        n_generations:      int   = 20,
        taille_population:  int   = 12,
        elite_size:         int   = 2,
        tournament_size:    int   = 3,
        mutation_rate:      float = 0.35,
        refroidissement:    float = 0.97,
        min_mutation_rate:  float = 0.05,
        min_diversite:      float = 0.40,
        patience:           int   = 5,
        maximize:           bool  = True,
    ) -> None:
        self._valider(espace, elite_size, taille_population, tournament_size, mutation_rate)

        self.espace             = {k: list(v) for k, v in espace.items()}
        self.fitness_fn         = fitness_fn
        self.n_generations      = n_generations
        self.taille_pop         = taille_population
        self.elite_size         = elite_size
        self.tournament_size    = tournament_size
        self.mutation_rate      = mutation_rate
        self.refroidissement    = refroidissement
        self.min_mutation_rate  = min_mutation_rate
        self.min_diversite      = min_diversite
        self.patience           = patience
        self.maximize           = maximize

        # État interne — réinitialisé à chaque appel à fit()
        self._population:             list  = []
        self._scores:                 list  = []
        self._journal:                list  = []
        self._best_params:            dict  = {}
        self._best_score:             float = float('-inf') if maximize else float('inf')
        self._mutation_rate_courant:  float = mutation_rate

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _valider(espace, elite_size, taille_pop, tournament_size, mutation_rate):
        if not espace:
            raise ValueError("espace ne peut pas être vide.")
        for k, v in espace.items():
            if not isinstance(v, (list, tuple)) or len(v) == 0:
                raise ValueError(f"espace['{k}'] doit être une liste non vide.")
        if elite_size < 1:
            raise ValueError("elite_size doit être ≥ 1.")
        if elite_size >= taille_pop:
            raise ValueError("elite_size doit être strictement inférieur à taille_population.")
        if tournament_size < 2:
            raise ValueError("tournament_size doit être ≥ 2.")
        if tournament_size > taille_pop:
            raise ValueError("tournament_size doit être ≤ taille_population.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate doit être dans [0.0, 1.0].")

    # ── Génération d'individus ────────────────────────────────────────────────

    def _individu_aleatoire(self) -> dict:
        return {k: random.choice(v) for k, v in self.espace.items()}

    def _initialiser(self) -> None:
        self._population             = [self._individu_aleatoire() for _ in range(self.taille_pop)]
        self._scores                 = [None] * self.taille_pop
        self._journal                = []
        self._best_params            = {}
        self._best_score             = float('-inf') if self.maximize else float('inf')
        self._mutation_rate_courant  = self.mutation_rate

    # ── Évaluation ───────────────────────────────────────────────────────────

    def _evaluer(self) -> None:
        """Appelle fitness_fn uniquement pour les individus non encore évalués."""
        for i, ind in enumerate(self._population):
            if self._scores[i] is None:
                score = self.fitness_fn(ind)
                if not isinstance(score, (int, float)):
                    raise TypeError(
                        f"fitness_fn doit retourner un float scalaire, reçu : {type(score)}"
                    )
                self._scores[i] = float(score)

    # ── Diversité (Hao & Solnon §4.3) ────────────────────────────────────────

    def _calculer_diversite(self) -> float:
        """Fraction d'individus uniques dans la population (∈ [0, 1])."""
        uniques = len({frozenset(ind.items()) for ind in self._population})
        return uniques / len(self._population)

    def _injecter_diversite(self, n_inject: int) -> None:
        """Remplace les n_inject pires non-élites par des individus aléatoires."""
        idx_tries = self._indices_tries()
        # Les élites occupent les premières positions — on touche uniquement les dernières
        pires = idx_tries[-n_inject:]
        for i in pires:
            self._population[i] = self._individu_aleatoire()
            self._scores[i]     = None

    # ── Modèle EDA (Hao & Solnon §5) ─────────────────────────────────────────

    def _eda_frequences(self, n_best: int) -> dict:
        """
        Fréquences empiriques des valeurs parmi les n_best meilleurs individus.
        Lissage de Laplace : évite les probabilités nulles (prior uniforme de 1).
        Retourne {param: {valeur: probabilité}}.
        """
        idx       = self._indices_tries()[:n_best]
        meilleurs = [self._population[i] for i in idx]
        freq = {}
        for k, valeurs in self.espace.items():
            counts = {v: 1 for v in valeurs}   # lissage de Laplace
            for ind in meilleurs:
                counts[ind[k]] = counts.get(ind[k], 1) + 1
            total  = sum(counts.values())
            freq[k] = {v: counts[v] / total for v in valeurs}
        return freq

    def _generer_depuis_eda(self, freq: dict) -> dict:
        """Génère un individu en tirant selon les fréquences EDA."""
        individu = {}
        for k, valeurs in self.espace.items():
            poids       = [freq[k][v] for v in valeurs]
            individu[k] = random.choices(valeurs, weights=poids, k=1)[0]
        return individu

    # ── Sélection ────────────────────────────────────────────────────────────

    def _indices_tries(self) -> list:
        """Indices de la population triés du meilleur au moins bon."""
        return sorted(
            range(len(self._scores)),
            key=lambda i: self._scores[i],
            reverse=self.maximize,
        )

    def _elites(self) -> tuple:
        """Retourne les elite_size meilleurs (individus, scores)."""
        idx       = self._indices_tries()[:self.elite_size]
        individus = [dict(self._population[i]) for i in idx]
        scores    = [self._scores[i]           for i in idx]
        return individus, scores

    def _roulette(self, n: int) -> list:
        """Sélection proportionnelle — décalage par min pour gérer les scores négatifs."""
        s_min = min(self._scores)
        poids = [s - s_min + 1e-9 for s in self._scores]
        total = sum(poids)
        poids = [p / total for p in poids]
        return [dict(ind) for ind in random.choices(self._population, weights=poids, k=n)]

    def _tournoi(self, n: int) -> list:
        """Sélection par tournoi : meilleur d'un groupe aléatoire de taille tournament_size."""
        sel     = []
        indices = list(range(len(self._population)))
        for _ in range(n):
            groupe  = random.sample(indices, self.tournament_size)
            gagnant = (max if self.maximize else min)(groupe, key=lambda i: self._scores[i])
            sel.append(dict(self._population[gagnant]))
        return sel

    def _selectionner_parents(self) -> list:
        """Pool de parents : élites + roulette + tournoi (50/50 sur les non-élites)."""
        reste      = self.taille_pop - self.elite_size
        n_roulette = reste // 2
        n_tournoi  = reste - n_roulette
        e_ind, _   = self._elites()
        return e_ind + self._roulette(n_roulette) + self._tournoi(n_tournoi)

    # ── Croisement ───────────────────────────────────────────────────────────

    def _croiser(self, parent1: dict, parent2: dict) -> tuple:
        """Croisement à un point sur les clés de paramètres."""
        keys = list(parent1.keys())
        if len(keys) < 2:
            return dict(parent1), dict(parent2)
        point   = random.randint(1, len(keys) - 1)
        enfant1 = {k: (parent1[k] if i < point else parent2[k]) for i, k in enumerate(keys)}
        enfant2 = {k: (parent2[k] if i < point else parent1[k]) for i, k in enumerate(keys)}
        return enfant1, enfant2

    # ── Mutation adaptative ───────────────────────────────────────────────────

    def _muter(self, individu: dict) -> dict:
        """
        Mutation par remplacement dans l'espace discret.
        Utilise _mutation_rate_courant (décroit avec le refroidissement à chaque génération).
        """
        mutant = dict(individu)
        for k, valeurs in self.espace.items():
            if random.random() < self._mutation_rate_courant:
                mutant[k] = random.choice(valeurs)
        return mutant

    # ── Génération des enfants ────────────────────────────────────────────────

    def _generer_enfants(self, parents: list, freq: dict = None) -> list:
        """
        Génère exactement (taille_pop - elite_size) enfants.
        Si freq fourni (modèle EDA) : 1/3 EDA, 2/3 croisement+mutation (Hao & Solnon §5).
        """
        n_enfants = self.taille_pop - self.elite_size
        n_eda     = (n_enfants // 3) if freq is not None else 0
        n_cross   = n_enfants - n_eda

        enfants = []
        while len(enfants) < n_cross:
            p1, p2 = random.sample(parents, 2)
            e1, e2 = self._croiser(p1, p2)
            enfants.append(self._muter(e1))
            if len(enfants) < n_cross:
                enfants.append(self._muter(e2))
        enfants = enfants[:n_cross]

        if freq is not None:
            for _ in range(n_eda):
                enfants.append(self._generer_depuis_eda(freq))

        return enfants[:n_enfants]

    # ── Tracking ─────────────────────────────────────────────────────────────

    def _mettre_a_jour_best(self) -> bool:
        """Actualise _best_params/_best_score. Retourne True si amélioré."""
        meilleur_score = (max if self.maximize else min)(self._scores)
        ameliore = (
            meilleur_score > self._best_score if self.maximize
            else meilleur_score < self._best_score
        )
        if ameliore:
            idx               = self._scores.index(meilleur_score)
            self._best_score  = meilleur_score
            self._best_params = dict(self._population[idx])
        return ameliore

    def _enregistrer_generation(self, generation: int, diversite: float, mutation_rate: float) -> None:
        for ind, score in zip(self._population, self._scores):
            record = {
                'generation':    generation,
                'score':         score,
                'diversite':     diversite,
                'mutation_rate': mutation_rate,
            }
            record.update(ind)
            self._journal.append(record)

    # ── Boucle principale ────────────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Lance la recherche heuristique.

        Par génération :
          1. Évaluation des individus non encore scorés (fitness_fn).
          2. Mise à jour du meilleur global.
          3. Calcul de la diversité (fraction d'individus uniques).
          4. Enregistrement dans le journal (→ .history).
          5. Rapport console + arrêt anticipé (patience).
          6. Injection aléatoire si diversité < min_diversite (Hao & Solnon §4.3).
          7. Refroidissement du taux de mutation (analogie recuit simulé).
          8. Modèle EDA sur les 2×elite_size meilleurs individus (Hao & Solnon §5).
          9. Nouvelle génération : élites + enfants (croisement/mutation + EDA).

        Returns:
            dict: 'params' → meilleur jeu d'hyperparamètres, 'score' → meilleur score.
        """
        self._initialiser()
        sans_amelioration = 0

        for gen in range(self.n_generations):

            # 1. Évaluation
            self._evaluer()

            # 2. Meilleur global
            ameliore = self._mettre_a_jour_best()

            # 3. Diversité
            diversite = self._calculer_diversite()

            # 4. Enregistrement
            self._enregistrer_generation(gen, diversite, self._mutation_rate_courant)

            # 5. Rapport
            scores_valides = [s for s in self._scores if s is not None]
            moy = sum(scores_valides) / len(scores_valides)
            print(
                f"[Gén {gen + 1:>3}/{self.n_generations}] "
                f"Meilleur={self._best_score:.4f}  "
                f"Moyen={moy:.4f}  "
                f"Div={diversite:.2f}  "
                f"μ={self._mutation_rate_courant:.3f}  "
                f"{'↑' if ameliore else ''}"
            )

            # Arrêt anticipé
            if ameliore:
                sans_amelioration = 0
            else:
                sans_amelioration += 1
            if sans_amelioration >= self.patience:
                print(f"  → Arrêt anticipé : {self.patience} générations sans amélioration.")
                break

            # 6. Injection de diversité si nécessaire (Hao & Solnon §4.3)
            if diversite < self.min_diversite:
                n_inject = max(1, self.taille_pop // 4)
                self._injecter_diversite(n_inject)

            # 7. Refroidissement du taux de mutation
            self._mutation_rate_courant = max(
                self.min_mutation_rate,
                self._mutation_rate_courant * self.refroidissement,
            )

            # 8. Modèle EDA sur les meilleurs individus (Hao & Solnon §5)
            n_eda_best = min(max(2, self.elite_size * 2), len(self._population))
            freq       = self._eda_frequences(n_eda_best)

            # 9. Nouvelle génération
            e_ind, e_scores = self._elites()
            parents         = self._selectionner_parents()
            enfants         = self._generer_enfants(parents, freq=freq)

            self._population = e_ind + enfants
            self._scores     = e_scores + [None] * len(enfants)

        return {'params': dict(self._best_params), 'score': self._best_score}

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def history(self) -> pd.DataFrame:
        """
        Historique complet de l'optimisation sous forme de DataFrame.
        Colonnes : generation | score | diversite | mutation_rate | <param_1> | ...
        """
        if not self._journal:
            raise RuntimeError("Appeler fit() avant d'accéder à .history.")
        return pd.DataFrame(self._journal)

    @property
    def best(self) -> dict:
        """Meilleur jeu de paramètres trouvé avec son score."""
        return {'params': dict(self._best_params), 'score': self._best_score}

    # ── Visualisations Plotly ─────────────────────────────────────────────────

    def plot_convergence(self) -> go.Figure:
        """Courbe de convergence : meilleur et moyen par génération avec bande ±1σ."""
        df    = self.history
        stats = (
            df.groupby('generation')['score']
            .agg(['max', 'min', 'mean', 'std'])
            .reset_index()
        )
        best_col = 'max' if self.maximize else 'min'

        x       = stats['generation'].tolist()
        y_best  = stats[best_col].tolist()
        y_mean  = stats['mean'].tolist()
        y_std   = stats['std'].fillna(0).tolist()
        y_upper = [m + s for m, s in zip(y_mean, y_std)]
        y_lower = [m - s for m, s in zip(y_mean, y_std)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x + list(reversed(x)),
            y=y_upper + list(reversed(y_lower)),
            fill='toself', fillcolor='rgba(255,152,0,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±1σ', hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode='lines',
            name='Score moyen',
            line=dict(color='#FF9800', width=2, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_best, mode='lines+markers',
            name='Meilleur de la génération',
            line=dict(color='#2196F3', width=2.5),
            marker=dict(size=7),
        ))
        fig.add_hline(
            y=self._best_score,
            line_dash='dash', line_color='#4CAF50', line_width=1.5,
            annotation_text=f'Best global = {self._best_score:.4f}',
            annotation_position='bottom right',
        )
        fig.update_layout(
            title='Convergence de la recherche heuristique',
            xaxis_title='Génération',
            yaxis_title='Score de fitness',
            template='plotly_white',
            legend=dict(orientation='h', y=-0.2),
        )
        return fig

    def plot_parameter_space(self) -> go.Figure:
        """Distribution des scores par valeur de chaque hyperparamètre (box plots)."""
        df     = self.history
        params = [p for p in self.espace.keys() if p in df.columns]
        n      = len(params)
        cols   = min(3, n)
        rows   = (n + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=params,
            vertical_spacing=0.12,
        )

        for idx, param in enumerate(params):
            r, c = divmod(idx, cols)
            for val in sorted(df[param].unique(), key=str):
                subset = df[df[param] == val]['score']
                fig.add_trace(go.Box(
                    y=subset.tolist(),
                    name=f'{param}={val}',
                    showlegend=False,
                    boxpoints='outliers',
                    marker_color='#2196F3',
                    line_color='#1565C0',
                ), row=r + 1, col=c + 1)

        fig.update_layout(
            title='Distribution des scores par hyperparamètre',
            height=350 * rows,
            template='plotly_white',
        )
        return fig

    def plot_best_evolution(self) -> go.Figure:
        """Meilleur score cumulatif (courbe monotone d'apprentissage)."""
        df  = self.history
        agg = (
            df.groupby('generation')['score'].max()
            if self.maximize
            else df.groupby('generation')['score'].min()
        )
        cumul = (agg.cummax() if self.maximize else agg.cummin()).reset_index()

        fig = go.Figure(go.Scatter(
            x=cumul['generation'].tolist(),
            y=cumul['score'].tolist(),
            mode='lines+markers',
            line=dict(color='#4CAF50', width=2.5),
            marker=dict(size=7, color='#4CAF50'),
            fill='tozeroy', fillcolor='rgba(76,175,80,0.1)',
            name='Meilleur cumulatif',
        ))
        fig.update_layout(
            title="Progression du meilleur score (courbe d'apprentissage)",
            xaxis_title='Génération',
            yaxis_title='Meilleur score cumulatif',
            template='plotly_white',
        )
        return fig

    def plot_diversity(self) -> go.Figure:
        """
        Diversité et taux de mutation par génération.
        Permet de vérifier que :
          - la diversité reste au-dessus du seuil min_diversite ;
          - le refroidissement converge bien vers min_mutation_rate.
        """
        df    = self.history
        stats = (
            df.groupby('generation')
            .agg(diversite=('diversite', 'first'), mutation_rate=('mutation_rate', 'first'))
            .reset_index()
        )
        x = stats['generation'].tolist()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Diversité de la population', 'Taux de mutation (refroidissement)'],
            vertical_spacing=0.18,
        )

        # Diversité
        fig.add_trace(go.Scatter(
            x=x, y=stats['diversite'].tolist(),
            mode='lines+markers',
            line=dict(color='#9C27B0', width=2.5),
            marker=dict(size=7),
            name='Diversité',
        ), row=1, col=1)
        fig.add_hline(
            y=self.min_diversite,
            line_dash='dash', line_color='#F44336', line_width=1.5,
            annotation_text=f'min_diversite = {self.min_diversite}',
            annotation_position='top right',
            row=1, col=1,
        )

        # Taux de mutation
        fig.add_trace(go.Scatter(
            x=x, y=stats['mutation_rate'].tolist(),
            mode='lines+markers',
            line=dict(color='#FF5722', width=2.5),
            marker=dict(size=7),
            name='Taux de mutation',
        ), row=2, col=1)
        fig.add_hline(
            y=self.min_mutation_rate,
            line_dash='dash', line_color='#FF9800', line_width=1.5,
            annotation_text=f'plancher = {self.min_mutation_rate}',
            annotation_position='bottom right',
            row=2, col=1,
        )

        fig.update_yaxes(range=[0.0, 1.05], row=1, col=1)
        max_mut = max(stats['mutation_rate'].tolist(), default=1.0)
        fig.update_yaxes(range=[0.0, max_mut * 1.15 + 0.01], row=2, col=1)
        fig.update_xaxes(title_text='Génération', row=2, col=1)
        fig.update_layout(
            title='Dynamique de la population — Diversité et refroidissement',
            height=560,
            template='plotly_white',
            showlegend=False,
        )
        return fig

    def resume(self) -> pd.DataFrame:
        """Meilleur individu de chaque génération sous forme de DataFrame."""
        df  = self.history
        idx = (
            df.groupby('generation')['score'].idxmax()
            if self.maximize
            else df.groupby('generation')['score'].idxmin()
        )
        return df.loc[idx].reset_index(drop=True)


# ── Exemple autonome ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    Fitness analytique pour tester sans données MoMTSim.
    Pour le vrai usage MKAN :
        fitness_fn = build_mkan_fitness(train_loader, val_loader, DEVICE, INPUT_SIZE, n_epochs=10)
        search = RechercheHeuristique(ESPACE_MKAN_DEFAULT, fitness_fn, maximize=True)
    """

    def fitness_demo(params: dict) -> float:
        score  = -abs(params['hidden_size'] - 64) / 64
        score += -abs(params['M'] - 8) / 32
        score += -abs(params['K'] - 2) / 4
        score += random.gauss(0, 0.02)
        return score

    search = RechercheHeuristique(
        espace={
            'hidden_size': [16, 32, 64, 128],
            'M':           [4, 8, 16, 32],
            'K':           [1, 2, 4],
        },
        fitness_fn       = fitness_demo,
        n_generations    = 15,
        taille_population= 12,
        elite_size       = 2,
        tournament_size  = 3,
        mutation_rate    = 0.35,
        refroidissement  = 0.95,
        min_diversite    = 0.40,
        patience         = 5,
        maximize         = True,
    )

    best = search.fit()
    print(f"\nMeilleurs hyperparamètres : {best['params']}")
    print(f"Score de fitness          : {best['score']:.4f}")
    print(f"\nRésumé par génération :\n{search.resume()}")

    search.plot_convergence().show()
    search.plot_parameter_space().show()
    search.plot_diversity().show()
