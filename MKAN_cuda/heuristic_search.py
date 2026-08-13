"""
heuristic_search.py (MKAN_cuda) — version GPU-CUDA de MKAN/heuristic_search.py.

Algorithme identique (genetique + EDA + mutation adaptative + diversite).
Trois ajouts specifiques GPU dans build_mkan_fitness :

  1. num_workers / pin_memory / persistent_workers / prefetch_factor dans DataLoader
     (l'original avait num_workers=0 code en dur pour eviter le freeze Jupyter/Windows).
  2. Automatic Mixed Precision (AMP) dans la boucle d'entrainement et d'evaluation.
     Controle par le parametre use_amp (defaut False ; mettre True si device=='cuda').
  3. torch.cuda.empty_cache() apres chaque evaluation pour liberer la VRAM
     entre les individus evalues sequentiellement.

RechercheHeuristique est inchangee (copie directe).

─── Usage dans train_colab.ipynb ────────────────────────────────────────────
    from MKAN_cuda import build_mkan_fitness, RechercheHeuristique, ESPACE_MKAN_DEFAULT

    fitness_fn = build_mkan_fitness(
        df_train, df_val, make_windows,
        device         = DEVICE,
        input_size     = INPUT_SIZE,
        n_epochs       = 10,
        num_workers    = NUM_WORKERS,   # 4 sur Colab
        pin_memory     = PIN_MEMORY,    # True si CUDA
        use_amp        = USE_AMP,       # True si CUDA
    )
    search = RechercheHeuristique(ESPACE_MKAN_DEFAULT, fitness_fn, ...)
    best   = search.fit()
─────────────────────────────────────────────────────────────────────────────
"""

import math
import random
import threading
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm


# ── Espace de recherche par defaut ────────────────────────────────────────────
ESPACE_MKAN_DEFAULT = {
    'hidden_size': [16, 32, 64, 128],
    'M':           [4, 8, 16, 32],
    'K':           [1, 2, 4],
    'lam':         [1e-3, 5e-3, 1e-2],
    'mu1':         [0.5, 1.0, 2.0],
    'mu2':         [0.25, 0.5, 1.0],
    'lr':          [5e-4, 1e-3, 3e-3],
}

# ── Espace etendu : inclut W et batch_size ────────────────────────────────────
# batch_size monte jusqu'a 1024 (GPU T4 dispose de 16 Go de VRAM)
ESPACE_MKAN_ETENDU = {
    **ESPACE_MKAN_DEFAULT,
    'W':          [5, 10, 15, 20],
    'batch_size': [128, 256, 512, 1024],
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
    max_train_samples:  int   = 30_000,
    max_val_samples:    int   = 10_000,
    num_workers:        int   = 0,
    pin_memory:         bool  = False,
    use_amp:            bool  = False,
) -> "callable":
    """
    Fabrique une fonction fitness connectee au workflow MKAN (train_colab.ipynb).

    Parametre supplementaires par rapport a MKAN/heuristic_search.py :

    num_workers        : workers du DataLoader (0 = sans fork, 2-4 sur Colab/Linux).
    pin_memory         : True si device est CUDA — active le transfert asynchrone
                         CPU->GPU via memoire epinglee.
    use_amp            : True pour activer torch.amp.autocast + GradScaler dans la
                         boucle d'entrainement et d'evaluation (CUDA uniquement).
                         Reduit l'utilisation VRAM et accelere de ~1.5x sur T4.

    Thread-safety : identique a l'original (verrou sur _meilleur et _window_cache).
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from .. import MKANScorer
    from .loss import mkan_total_loss

    _lock         = threading.Lock()
    _meilleur     = {'score': float('-inf'), 'state_dict': None, 'params': None}
    _window_cache = {}

    # AMP actif seulement si device est CUDA (GradScaler echoue sur CPU/DirectML)
    _amp_active = use_amp and device.type == 'cuda'

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

        # ── Cache des fenetres glissantes (double-checked locking) ────────────
        if W not in _window_cache:
            with _lock:
                if W not in _window_cache:
                    tqdm.write(f"  Mise en cache des fenetres W={W} (train + val)...")
                    try:
                        xtr, ytr = make_windows(df_train, W, leave=False)
                        xvl, yvl = make_windows(df_val,   W, leave=False)
                    except TypeError:
                        xtr, ytr = make_windows(df_train, W)
                        xvl, yvl = make_windows(df_val,   W)
                    _window_cache[W] = ((xtr, ytr), (xvl, yvl))
                    tqdm.write(
                        f"  Cache W={W} pret : {len(xtr):,} fenetres train  /  {len(xvl):,} val"
                    )

        with _lock:
            (X_tr, y_tr), (X_vl, y_vl) = _window_cache[W]

        if len(X_tr) == 0 or len(X_vl) == 0:
            return -1.0

        # ── Sous-echantillonnage reproductible ────────────────────────────────
        _rng = np.random.default_rng(42)
        if max_train_samples and len(X_tr) > max_train_samples:
            _idx = _rng.choice(len(X_tr), max_train_samples, replace=False)
            _idx.sort()
            X_tr, y_tr = X_tr[_idx], y_tr[_idx]
        if max_val_samples and len(X_vl) > max_val_samples:
            _idx = _rng.choice(len(X_vl), max_val_samples, replace=False)
            _idx.sort()
            X_vl, y_vl = X_vl[_idx], y_vl[_idx]

        # ── DataLoader GPU-optimise ───────────────────────────────────────────
        # pin_memory=True : tenseurs CPU epingles -> transfert asynchrone vers GPU.
        # persistent_workers : les workers survivent entre les epochs (evite le
        #   cout de fork a chaque epoch).
        # prefetch_factor=2 : prepare 2 batchs en avance pendant que le GPU calcule.
        _pin = pin_memory and device.type == 'cuda'

        def _loader(X, y, shuffle):
            X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            ds  = TensorDataset(X_t, y_t)
            return DataLoader(
                ds,
                batch_size         = batch_size,
                shuffle            = shuffle,
                num_workers        = num_workers,
                pin_memory         = _pin,
                persistent_workers = num_workers > 0,
                prefetch_factor    = 2 if num_workers > 0 else None,
            )

        tr_loader = _loader(X_tr, y_tr, shuffle=True)
        vl_loader = _loader(X_vl, y_vl, shuffle=False)

        model     = MKANScorer(input_size=input_size, hidden_size=hidden_size,
                               M=M, K=K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ── Entraainement avec AMP ─────────────────────────────────────────────
        # GradScaler gere la mise a l'echelle des gradients pour eviter le
        # underflow float16 ; enabled=False le rend un no-op sur CPU.
        scaler = torch.amp.GradScaler(device=device.type, enabled=_amp_active)

        model.train()
        for _ in range(n_epochs):
            for X_batch, y_batch in tr_loader:
                X_batch = X_batch.to(device, non_blocking=_pin)
                y_batch = y_batch.to(device, non_blocking=_pin)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=_amp_active):
                    loss, *_ = mkan_total_loss(
                        model, X_batch, y_batch, lam=lam, mu1=mu1, mu2=mu2)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        # ── Evaluation MCC sur val (eq. 3.1) ──────────────────────────────────
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in vl_loader:
                with torch.amp.autocast(device_type=device.type, enabled=_amp_active):
                    out = model(X_batch.to(device, non_blocking=_pin))
                all_scores.append(out.float().cpu().numpy())
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

        # ── Sauvegarde du meilleur modele (sous verrou) ────────────────────────
        with _lock:
            if mcc > _meilleur['score']:
                _meilleur['score']      = mcc
                _meilleur['params']     = dict(params)
                _meilleur['state_dict'] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

        # ── Liberation explicite de la VRAM ───────────────────────────────────
        # Sur GPU, le GC Python ne libere pas immediatement la VRAM apres del.
        # torch.cuda.empty_cache() rend la memoire disponible pour le prochain
        # individu evalue (12 modeles par generation -> risque OOM sans ca).
        import gc
        del model, optimizer, scaler
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return mcc

    def get_meilleur() -> dict:
        """Retourne une copie thread-safe de l'etat du meilleur modele trouve."""
        with _lock:
            return {
                'score':      _meilleur['score'],
                'params':     dict(_meilleur['params']) if _meilleur['params'] else None,
                'state_dict': dict(_meilleur['state_dict']) if _meilleur['state_dict'] else None,
            }

    fitness_fn.get_meilleur = get_meilleur
    return fitness_fn


# ── RechercheHeuristique ──────────────────────────────────────────────────────
# Identique a MKAN/heuristic_search.py — copiee ici pour que MKAN_cuda soit
# autonome (pas de besoin d'importer depuis MKAN pour cette classe).

class RechercheHeuristique:
    """
    Optimisation heuristique des hyperparametres par algorithme genetique ameliore.

    Identique a MKAN.RechercheHeuristique. Voir MKAN/heuristic_search.py
    pour la documentation complete.

    Utiliser build_mkan_fitness (MKAN_cuda) pour obtenir la fitness_fn GPU.
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
        n_workers:          int   = 1,
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
        self.n_workers          = max(1, n_workers)

        self._population:             list  = []
        self._scores:                 list  = []
        self._journal:                list  = []
        self._best_params:            dict  = {}
        self._best_score:             float = float('-inf') if maximize else float('inf')
        self._mutation_rate_courant:  float = mutation_rate

    @staticmethod
    def _valider(espace, elite_size, taille_pop, tournament_size, mutation_rate):
        if not espace:
            raise ValueError("espace ne peut pas etre vide.")
        for k, v in espace.items():
            if not isinstance(v, (list, tuple)) or len(v) == 0:
                raise ValueError(f"espace['{k}'] doit etre une liste non vide.")
        if elite_size < 1:
            raise ValueError("elite_size doit etre >= 1.")
        if elite_size >= taille_pop:
            raise ValueError("elite_size doit etre strictement inferieur a taille_population.")
        if tournament_size < 2:
            raise ValueError("tournament_size doit etre >= 2.")
        if tournament_size > taille_pop:
            raise ValueError("tournament_size doit etre <= taille_population.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate doit etre dans [0.0, 1.0].")

    def _individu_aleatoire(self) -> dict:
        return {k: random.choice(v) for k, v in self.espace.items()}

    def _initialiser(self) -> None:
        self._population             = [self._individu_aleatoire() for _ in range(self.taille_pop)]
        self._scores                 = [None] * self.taille_pop
        self._journal                = []
        self._best_params            = {}
        self._best_score             = float('-inf') if self.maximize else float('inf')
        self._mutation_rate_courant  = self.mutation_rate

    def _evaluer(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        a_evaluer = [(i, ind) for i, ind in enumerate(self._population)
                     if self._scores[i] is None]
        if not a_evaluer:
            return

        if self.n_workers == 1:
            pbar_ind = tqdm(a_evaluer, desc="  ├─ individus", leave=False, unit="ind")
            for i, ind in pbar_ind:
                score = self.fitness_fn(ind)
                if not isinstance(score, (int, float)):
                    raise TypeError(
                        f"fitness_fn doit retourner un float scalaire, recu : {type(score)}")
                self._scores[i] = float(score)
                pbar_ind.set_postfix({"MCC": f"{score:.4f}"})
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                futures  = {pool.submit(self.fitness_fn, ind): i for i, ind in a_evaluer}
                pbar_ind = tqdm(as_completed(futures), total=len(futures),
                                desc="  ├─ individus", leave=False, unit="ind")
                for fut in pbar_ind:
                    i     = futures[fut]
                    score = fut.result()
                    if not isinstance(score, (int, float)):
                        raise TypeError(
                            f"fitness_fn doit retourner un float scalaire, recu : {type(score)}")
                    self._scores[i] = float(score)
                    pbar_ind.set_postfix({"MCC": f"{score:.4f}"})

    def _calculer_diversite(self) -> float:
        uniques = len({frozenset(ind.items()) for ind in self._population})
        return uniques / len(self._population)

    def _injecter_diversite(self, n_inject: int) -> None:
        idx_tries = self._indices_tries()
        pires     = idx_tries[-n_inject:]
        for i in pires:
            self._population[i] = self._individu_aleatoire()
            self._scores[i]     = None

    def _eda_frequences(self, n_best: int) -> dict:
        idx       = self._indices_tries()[:n_best]
        meilleurs = [self._population[i] for i in idx]
        freq      = {}
        for k, valeurs in self.espace.items():
            counts = {v: 1 for v in valeurs}
            for ind in meilleurs:
                counts[ind[k]] = counts.get(ind[k], 1) + 1
            total   = sum(counts.values())
            freq[k] = {v: counts[v] / total for v in valeurs}
        return freq

    def _generer_depuis_eda(self, freq: dict) -> dict:
        individu = {}
        for k, valeurs in self.espace.items():
            poids       = [freq[k][v] for v in valeurs]
            individu[k] = random.choices(valeurs, weights=poids, k=1)[0]
        return individu

    def _indices_tries(self) -> list:
        return sorted(
            range(len(self._scores)),
            key=lambda i: self._scores[i],
            reverse=self.maximize,
        )

    def _elites(self) -> tuple:
        idx       = self._indices_tries()[:self.elite_size]
        individus = [dict(self._population[i]) for i in idx]
        scores    = [self._scores[i]           for i in idx]
        return individus, scores

    def _roulette(self, n: int) -> list:
        s_min = min(self._scores)
        poids = [s - s_min + 1e-9 for s in self._scores]
        total = sum(poids)
        poids = [p / total for p in poids]
        return [dict(ind) for ind in random.choices(self._population, weights=poids, k=n)]

    def _tournoi(self, n: int) -> list:
        sel     = []
        indices = list(range(len(self._population)))
        for _ in range(n):
            groupe  = random.sample(indices, self.tournament_size)
            gagnant = (max if self.maximize else min)(groupe, key=lambda i: self._scores[i])
            sel.append(dict(self._population[gagnant]))
        return sel

    def _selectionner_parents(self) -> list:
        reste      = self.taille_pop - self.elite_size
        n_roulette = reste // 2
        n_tournoi  = reste - n_roulette
        e_ind, _   = self._elites()
        return e_ind + self._roulette(n_roulette) + self._tournoi(n_tournoi)

    def _croiser(self, parent1: dict, parent2: dict) -> tuple:
        keys = list(parent1.keys())
        if len(keys) < 2:
            return dict(parent1), dict(parent2)
        point   = random.randint(1, len(keys) - 1)
        enfant1 = {k: (parent1[k] if i < point else parent2[k]) for i, k in enumerate(keys)}
        enfant2 = {k: (parent2[k] if i < point else parent1[k]) for i, k in enumerate(keys)}
        return enfant1, enfant2

    def _muter(self, individu: dict) -> dict:
        mutant = dict(individu)
        for k, valeurs in self.espace.items():
            if random.random() < self._mutation_rate_courant:
                mutant[k] = random.choice(valeurs)
        return mutant

    def _generer_enfants(self, parents: list, freq: dict = None) -> list:
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

    def _mettre_a_jour_best(self) -> bool:
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

    def fit(self) -> dict:
        """Lance la recherche heuristique. Identique a MKAN.RechercheHeuristique.fit()."""
        self._initialiser()
        sans_amelioration = 0

        pbar = tqdm(range(self.n_generations), desc="Generations", unit="gen", leave=True)

        for gen in pbar:

            self._evaluer()
            ameliore  = self._mettre_a_jour_best()
            diversite = self._calculer_diversite()
            self._enregistrer_generation(gen, diversite, self._mutation_rate_courant)

            scores_valides = [s for s in self._scores if s is not None]
            moy = sum(scores_valides) / len(scores_valides)
            print(
                f"Gen {gen + 1:02d}/{self.n_generations} | "
                f"best={self._best_score:.4f} | "
                f"moy={moy:.4f} | "
                f"div={diversite:.2f} | "
                f"mu={self._mutation_rate_courant:.3f} | "
                f"{'AMELIORATION' if ameliore else '='}"
            )
            pbar.set_postfix({
                "best": f"{self._best_score:.4f}",
                "moy":  f"{moy:.4f}",
                "div":  f"{diversite:.2f}",
            })

            if ameliore:
                sans_amelioration = 0
            else:
                sans_amelioration += 1
            if sans_amelioration >= self.patience:
                tqdm.write(f"  -> Arret anticipe : {self.patience} generations sans amelioration.")
                break

            if diversite < self.min_diversite:
                n_inject = max(1, self.taille_pop // 4)
                self._injecter_diversite(n_inject)

            self._mutation_rate_courant = max(
                self.min_mutation_rate,
                self._mutation_rate_courant * self.refroidissement,
            )

            n_eda_best = min(max(2, self.elite_size * 2), len(self._population))
            freq       = self._eda_frequences(n_eda_best)

            e_ind, e_scores = self._elites()
            parents         = self._selectionner_parents()
            enfants         = self._generer_enfants(parents, freq=freq)

            self._population = e_ind + enfants
            self._scores     = e_scores + [None] * len(enfants)

        return {'params': dict(self._best_params), 'score': self._best_score}

    @property
    def history(self) -> pd.DataFrame:
        if not self._journal:
            raise RuntimeError("Appeler fit() avant d'acceder a .history.")
        return pd.DataFrame(self._journal)

    @property
    def best(self) -> dict:
        return {'params': dict(self._best_params), 'score': self._best_score}

    def plot_convergence(self) -> go.Figure:
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
            x=x + list(reversed(x)), y=y_upper + list(reversed(y_lower)),
            fill='toself', fillcolor='rgba(255,152,0,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='+-1sigma', hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode='lines', name='Score moyen',
            line=dict(color='#FF9800', width=2, dash='dot')))
        fig.add_trace(go.Scatter(
            x=x, y=y_best, mode='lines+markers',
            name='Meilleur de la generation',
            line=dict(color='#2196F3', width=2.5), marker=dict(size=7)))
        fig.add_hline(
            y=self._best_score, line_dash='dash', line_color='#4CAF50', line_width=1.5,
            annotation_text=f'Best global = {self._best_score:.4f}',
            annotation_position='bottom right')
        fig.update_layout(
            title='Convergence de la recherche heuristique (MKAN_cuda)',
            xaxis_title='Generation', yaxis_title='Score de fitness',
            template='plotly_white', legend=dict(orientation='h', y=-0.2))
        return fig

    def plot_parameter_space(self) -> go.Figure:
        df     = self.history
        params = [p for p in self.espace.keys() if p in df.columns]
        n      = len(params)
        cols   = min(3, n)
        rows   = (n + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=params,
                            vertical_spacing=0.12)
        for idx, param in enumerate(params):
            r, c = divmod(idx, cols)
            for val in sorted(df[param].unique(), key=str):
                subset = df[df[param] == val]['score']
                fig.add_trace(go.Box(
                    y=subset.tolist(), name=f'{param}={val}', showlegend=False,
                    boxpoints='outliers', marker_color='#2196F3',
                    line_color='#1565C0'), row=r + 1, col=c + 1)
        fig.update_layout(title='Distribution des scores par hyperparametre',
                          height=350 * rows, template='plotly_white')
        return fig

    def plot_best_evolution(self) -> go.Figure:
        df  = self.history
        agg = (df.groupby('generation')['score'].max()
               if self.maximize
               else df.groupby('generation')['score'].min())
        cumul = (agg.cummax() if self.maximize else agg.cummin()).reset_index()
        fig = go.Figure(go.Scatter(
            x=cumul['generation'].tolist(), y=cumul['score'].tolist(),
            mode='lines+markers', line=dict(color='#4CAF50', width=2.5),
            marker=dict(size=7, color='#4CAF50'),
            fill='tozeroy', fillcolor='rgba(76,175,80,0.1)',
            name='Meilleur cumulatif'))
        fig.update_layout(
            title="Progression du meilleur score (courbe d'apprentissage)",
            xaxis_title='Generation', yaxis_title='Meilleur score cumulatif',
            template='plotly_white')
        return fig

    def plot_diversity(self) -> go.Figure:
        df    = self.history
        stats = (
            df.groupby('generation')
            .agg(diversite=('diversite', 'first'), mutation_rate=('mutation_rate', 'first'))
            .reset_index()
        )
        x = stats['generation'].tolist()
        fig = make_subplots(rows=2, cols=1,
            subplot_titles=['Diversite de la population',
                            'Taux de mutation (refroidissement)'],
            vertical_spacing=0.18)
        fig.add_trace(go.Scatter(x=x, y=stats['diversite'].tolist(),
            mode='lines+markers', line=dict(color='#9C27B0', width=2.5),
            marker=dict(size=7), name='Diversite'), row=1, col=1)
        fig.add_hline(y=self.min_diversite, line_dash='dash', line_color='#F44336',
            line_width=1.5,
            annotation_text=f'min_diversite = {self.min_diversite}',
            annotation_position='top right', row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=stats['mutation_rate'].tolist(),
            mode='lines+markers', line=dict(color='#FF5722', width=2.5),
            marker=dict(size=7), name='Taux de mutation'), row=2, col=1)
        fig.add_hline(y=self.min_mutation_rate, line_dash='dash', line_color='#FF9800',
            line_width=1.5,
            annotation_text=f'plancher = {self.min_mutation_rate}',
            annotation_position='bottom right', row=2, col=1)
        fig.update_yaxes(range=[0.0, 1.05], row=1, col=1)
        max_mut = max(stats['mutation_rate'].tolist(), default=1.0)
        fig.update_yaxes(range=[0.0, max_mut * 1.15 + 0.01], row=2, col=1)
        fig.update_xaxes(title_text='Generation', row=2, col=1)
        fig.update_layout(
            title='Dynamique de la population — Diversite et refroidissement (MKAN_cuda)',
            height=560, template='plotly_white', showlegend=False)
        return fig

    def resume(self) -> pd.DataFrame:
        df  = self.history
        idx = (df.groupby('generation')['score'].idxmax()
               if self.maximize
               else df.groupby('generation')['score'].idxmin())
        return df.loc[idx].reset_index(drop=True)
