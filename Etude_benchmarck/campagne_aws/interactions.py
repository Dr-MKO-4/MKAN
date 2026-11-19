"""
interactions.py -- Statistique d'interaction H de Friedman et Popescu (2008) entre variables,
calculee sur un finaliste entraine sans MultKAN, pour choisir les paires des noeuds produit.

    H2_ab = sum_i [PD_ab(x_a^i, x_b^i) - PD_a(x_a^i) - PD_b(x_b^i)]^2 / sum_i PD_ab(x_a^i, x_b^i)^2

PD : dependance partielle centree du score, en ne faisant varier que a (et b) au DERNIER pas de la
fenetre ; 500 fenetres de validation tirees avec la graine 0 ; variables continues de x seulement
(pas les colonnes h ni les indicatrices). Sortie : paires classees par H decroissant.

Usage (scheduler.py l'appelle une fois par finaliste) :
    python interactions.py --run_dir sorties/runs/<run_id> --data_dir donnees/<jeu> --out h.json
"""

import argparse
import json
import os
import sys
from itertools import combinations

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


@torch.no_grad()
def _pd(model, X: torch.Tensor, colonnes: tuple, batch: int = 4096) -> np.ndarray:
    """Dependance partielle centree aux points observes : pour chaque fenetre i, les variables de
    `colonnes` de toutes les fenetres prennent au dernier pas la valeur de la fenetre i."""
    n = X.shape[0]
    valeurs = X[:, -1, list(colonnes)].clone()
    pd = np.empty(n, dtype=np.float64)
    tampon = X.clone()
    for i in range(n):
        tampon[:, -1, list(colonnes)] = valeurs[i]
        total = 0.0
        for s in range(0, n, batch):
            total += float(model(tampon[s:s + batch]).sum())
        pd[i] = total / n
    return pd - pd.mean()


_TACHE = {}   # modele et fenetres de chaque processus (reconstruits a l'initialisation)


def _init_processus(construction, X, fils):
    """Processus lance a neuf (spawn) : « fork » apres usage d'OpenMP par le parent bloquait les fils de calcul
    des enfants (barriere gomp jamais franchie, constate le 13/11). Le modele est reconstruit a partir de la
    meme specification et des memes poids."""
    torch.set_num_threads(fils)
    if construction is None:
        return
    from train_run import build_model
    spec, input_size, etat = construction
    model = build_model(spec, input_size)
    model.load_state_dict(etat)
    model.eval()
    _TACHE.update(model=model, X=X)


def _pd_tache(colonnes: tuple) -> tuple:
    return colonnes, _pd(_TACHE["model"], _TACHE["X"], colonnes)


def _toutes_pd(model, X: torch.Tensor, taches: list, reprise: str, processus: int, fils: int,
               construction=None) -> dict:
    """Dependances partielles de toutes les taches (variables seules et paires), calculs independants repartis
    sur `processus` processus (13/11 : un seul processus en serie laissait la machine aux deux tiers vide).
    Chaque PD est gardee dans `reprise` (npz) au fur et a mesure : une interruption ne refait que les PD
    manquantes. Memes formules et memes lots qu'en serie : resultats identiques."""
    faites = {}
    if reprise and os.path.exists(reprise):
        with np.load(reprise) as z:
            faites = {tuple(int(c) for c in k.split("_")): z[k] for k in z.files}
    reste = [t for t in taches if t not in faites]

    def sauver():
        if reprise:
            tmp = reprise + ".tmp.npz"
            np.savez(tmp, **{"_".join(map(str, k)): v for k, v in faites.items()})
            os.replace(tmp, reprise)

    if reste and processus > 1 and construction is not None:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processus, initializer=_init_processus,
                                          initargs=(construction, X, fils)) as pool:
            for n, (cols, pd) in enumerate(pool.imap_unordered(_pd_tache, reste), 1):
                faites[cols] = pd
                if n % max(1, processus) == 0:
                    sauver()
    else:
        for cols in reste:
            faites[cols] = _pd(model, X, cols)
            sauver()
    sauver()
    return faites


def h_interactions(model, X: torch.Tensor, indices: list, reprise: str = None, processus: int = 1,
                   fils: int = 1, construction=None) -> list:
    """indices : positions des variables retenues dans x. Retourne [{i, j, H}] par H decroissant.
    construction = (spec, input_size, poids) pour reconstruire le modele dans les processus paralleles."""
    taches = [(a,) for a in indices] + list(combinations(indices, 2))
    pd = _toutes_pd(model, X, taches, reprise, processus, fils, construction)
    pd1 = {a: pd[(a,)] for a in indices}
    sortie = []
    for a, b in combinations(indices, 2):
        pd2 = pd[(a, b)]
        den = float((pd2 ** 2).sum())
        h2 = float(((pd2 - pd1[a] - pd1[b]) ** 2).sum() / den) if den > 0 else 0.0
        sortie.append({"i": int(a), "j": int(b), "H": float(np.sqrt(h2))})
    sortie.sort(key=lambda p: -p["H"])
    return sortie


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--threads", type=int, default=None,
                   help="fils torch par processus (defaut : 2 avec plusieurs processus, sinon coeurs - 2)")
    p.add_argument("--processus", type=int,
                   default=int(os.environ.get("MKAN_INTERACTIONS_PROCESSUS", "16")),
                   help="processus en parallele (MKAN_INTERACTIONS_PROCESSUS, 16 par defaut ; 1 = en serie)")
    a = p.parse_args()
    if a.threads is None:
        a.threads = 2 if a.processus > 1 else max(1, (os.cpu_count() or 2) - 2)
    from train_run import build_model
    from niveau1_harness import BINARY_COLS, ACTION_ONEHOT_COLS

    torch.set_num_threads(a.threads)
    ck = torch.load(os.path.join(a.run_dir, "model.pt"), map_location="cpu", weights_only=False)
    model = build_model(ck["spec"], ck["input_size"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cols = ck["preprocessing"]["feature_cols"]
    indices = [i for i, c in enumerate(cols) if c not in BINARY_COLS and c not in ACTION_ONEHOT_COLS]
    X_val = np.load(os.path.join(a.data_dir, "X_val.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(X_val), min(a.n, len(X_val)), replace=False))
    X = torch.from_numpy(np.ascontiguousarray(X_val[idx]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    reprise = a.out + ".partiel.npz"      # PD deja calculees (reprise apres interruption)
    paires = h_interactions(model, X, indices, reprise=reprise, processus=a.processus, fils=a.threads,
                            construction=(ck["spec"], ck["input_size"], ck["state_dict"]))
    for q in paires:
        q["variables"] = [cols[q["i"]], cols[q["j"]]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"run_id": ck["spec"]["run_id"], "n_fenetres": len(idx), "paires": paires}, f,
                  indent=1, ensure_ascii=False)
    os.replace(tmp, a.out)
    if os.path.exists(reprise):
        os.remove(reprise)
    print(f"[interactions] {len(paires)} paires -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
