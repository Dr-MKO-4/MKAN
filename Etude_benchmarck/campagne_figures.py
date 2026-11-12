"""
campagne_figures.py -- Analyse et graphiques du banc d'essai de la campagne AWS, reconstruits
UNIQUEMENT a partir d'une copie du dossier de resultats, sans reentrainement. Relancable tel
quel sur chaque nouvelle copie : une figure ou une ligne n'apparait que si ses donnees existent.

    python campagne_figures.py campagne_aws/resultats_instance/etape4_2026-11-06
    python campagne_figures.py <dossier> --no_png

Lit, en lecture seule : runs/<id>/{DONE, result.json, latency.json, status.json,
edge_profiles.npz, model.pt}. Le jeu de test n'est jamais lu (la campagne n'en produit aucune
metrique a ce stade). runs.csv et summary.json ne sont pas utilises : ils peuvent dater d'avant
la mesure des latences.

Definitions (moyennes sur les graines, memes poids que niveau1_decision.py) :
    perf     = w_MCC*max(0,MCC) + w_PR*PR-AUC - w_Brier*Brier     (V4, mesure de detection)
    F80, F99 = perf + w_R2*R2_symb - w_lat*P_lat, R2_symb au seuil 0,80 (seuil de travail,
               decision du 06/11) ou 0,99 (sensibilite)
    P_lat    = min(1, max(0, (L - T) / T)), T = 500 ms
    V5       = AHP normalise, min-max sur les architectures admissibles (L <= T)
Tant qu'une latence manque, TOUTES les fitness restent sans terme de latence et le rapport le
signale : aucune moyenne n'est calculee sur des graines partielles.

Sorties :
    <out>/ANALYSE_BANC.md, <out>/*.csv, <out>/finalistes_provisoires.json
    <out>/figures/*.html
    memoire/figure/banc_*.png                 (sans titre : la legende LaTeX le porte)
    memoire/tableaux_banc/*.tex
"""

import argparse
import copy
import csv
import glob
import json
import os
import traceback
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

from niveau1_decision import (ABBR, AHP_WEIGHT_BRIER, AHP_WEIGHT_LATENCY_PENALTY, AHP_WEIGHT_MCC,
                              AHP_WEIGHT_PR_AUC, AHP_WEIGHT_R2_SYMBOLIC, GATES, _minmax)

ETUDE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(ETUDE_DIR))

LATENCE_T = 500.0
GRAINES_ATTENDUES = 5
PHASES_BANC = ("etape4_brut", "etape4_traite", "etape4_reference")
N_ARETES_AUDIT, THETA_ELAGAGE = 40, 0.01    # regle de _compute_r2_symbolic

# Palette validee (validate_palette.js, 4 premiers rangs, paires adjacentes). Les finalistes
# portent une seule teinte d'accent, sauf dans les formes en lignes, ou ils sont aussi nommes.
ACCENT = "#2a78d6"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRIS, GRIS_CLAIR = "#898781", "#c3c2b7"
ENCRE, ENCRE_2 = "#0b0b0b", "#52514e"
GRILLE = "#e1e0d9"
BLEU_DIV, MILIEU_DIV, ROUGE_DIV = "#6da7ec", "#f0efec", "#e66767"
DIVERGENT = [[0.0, ROUGE_DIV], [0.5, MILIEU_DIV], [1.0, BLEU_DIV]]
DIVERGENT_INVERSE = [[0.0, BLEU_DIV], [0.5, MILIEU_DIV], [1.0, ROUGE_DIV]]
SYMBOLE = {"lstm": "circle", "gru": "diamond"}
NOM_CELLULE = {"lstm": "TKAN", "gru": "GRU"}
NOM_BASE = {"relukan": "ReLU-KAN", "wavkan_dog": "DoG", "efficientkan": "B-spline",
            "wavkan_morlet": "Morlet", "linear": "Linéaire", "hybrid": "Hybride"}
PORTES_AFFICHEES = {"lstm": {"Forget": "Forget", "Input": "Input", "Candidate": "Candidate",
                             "Output": "Output"},
                    "gru": {"Forget": "Reset", "Input": "Update", "Candidate": "Candidate"}}
PORTES_NPZ = {"forget": "Forget / Reset", "reset": "Forget / Reset", "input": "Input / Update",
              "update": "Input / Update", "candidate": "Candidate", "output": "Output"}
ORDRE_PORTES_NPZ = ["Forget / Reset", "Input / Update", "Candidate", "Output"]


# ─────────────────────────────────────────────────────────────────────────────
# Formats
# ─────────────────────────────────────────────────────────────────────────────

def fr(x, d: int = 4) -> str:
    return "n.d." if x is None or x != x else f"{x:.{d}f}".replace(".", ",")


def fp(p, latex: bool = False) -> str:
    """p-valeur a trois decimales ; sous 0,001, on ecrit < 0,001 plutot que 0,000."""
    if p == p and p < 0.001:
        return "$<$ 0,001" if latex else "< 0,001"
    return fr(p, 3)


def milliers(x) -> str:
    return f"{int(round(x)):,}".replace(",", " ")


def ecrire_csv(chemin: str, lignes: list) -> None:
    if not lignes:
        return
    cles = list(dict.fromkeys(k for l in lignes for k in l))
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cles)
        w.writeheader()
        w.writerows(lignes)
    print(chemin)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement et agregation
# ─────────────────────────────────────────────────────────────────────────────

def _flottant(x):
    try:
        x = float(x)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None


def label_arch(cell: str, regime: str, bases: dict) -> str:
    noms = "/".join(ABBR.get(bases[g], bases[g]) for g in GATES if g in bases)
    prefixe = "Uniforme " if bases.get("Forget") == "hybrid" else ""
    suffixe = " (traité)" if regime.startswith("trait") else ""
    return f"{prefixe}{NOM_CELLULE[cell]} {noms}{suffixe}"


def charger_runs(dossier: str):
    runs, hors_perimetre = [], defaultdict(int)
    # Un run est fini quand result.json existe (ecrit en fin d'entrainement) ; les copies legeres
    # des phases traite et reference n'ont ni DONE, ni status.json, ni model.pt.
    for res_json in sorted(glob.glob(os.path.join(dossier, "runs", "*", "result.json"))):
        rep = os.path.dirname(res_json)
        with open(os.path.join(rep, "result.json"), encoding="utf-8") as f:
            res = json.load(f)
        spec, m = res["spec"], res["metrics"]
        if spec["phase"] not in PHASES_BANC:
            hors_perimetre[spec["phase"]] += 1
            continue
        lat = None
        chemin_lat = os.path.join(rep, "latency.json")
        if os.path.exists(chemin_lat):
            with open(chemin_lat, encoding="utf-8") as f:
                lat = _flottant(json.load(f).get("latency_ms"))
        threads = None
        chemin_st = os.path.join(rep, "status.json")
        if os.path.exists(chemin_st):
            with open(chemin_st, encoding="utf-8") as f:
                threads = json.load(f).get("threads")
        mcc = _flottant(m.get("mcc")) or 0.0
        pr, brier = _flottant(m.get("auc_pr")) or 0.0, _flottant(m.get("brier"))
        brier = 1.0 if brier is None else brier
        perf = AHP_WEIGHT_MCC * max(0.0, mcc) + AHP_WEIGHT_PR_AUC * pr - AHP_WEIGHT_BRIER * brier
        r99 = _flottant(m.get("r2_symbolic")) or 0.0
        r80 = _flottant(m.get("r2_symbolic_0.80")) or 0.0
        runs.append(dict(
            run_id=spec["run_id"], rep=rep, phase=spec["phase"], cell=spec["cell_type"],
            regime=spec["regime"], bases=spec["base_by_gate"], seed=spec["seed"],
            hidden=spec["hidden_size"], W=spec.get("W"), label=label_arch(
                spec["cell_type"], spec["regime"], spec["base_by_gate"]),
            npar=m.get("n_params"), mcc=mcc, pr=pr, roc=_flottant(m.get("auc_roc")), brier=brier,
            f1=_flottant(m.get("f1_best")), r99=r99, r80=r80, perf=perf, lat=lat,
            pen=None if lat is None else min(1.0, max(0.0, (lat - LATENCE_T) / LATENCE_T)),
            epochs=m.get("epochs_run"), best_epoch=m.get("best_epoch"),
            stopped=bool(m.get("stopped_early")), n_val=m.get("n_val_windows"),
            n_bad=m.get("n_bad_scores"), threads=threads,
            train_hist=m.get("train_loss_history") or [],
            mon_hist=m.get("monitor_loss_history") or []))
    latence_complete = bool(runs) and all(r["lat"] is not None for r in runs)
    for r in runs:
        terme = AHP_WEIGHT_LATENCY_PENALTY * r["pen"] if latence_complete else 0.0
        r["F80"] = r["perf"] + AHP_WEIGHT_R2_SYMBOLIC * r["r80"] - terme
        r["F99"] = r["perf"] + AHP_WEIGHT_R2_SYMBOLIC * r["r99"] - terme
    return runs, latence_complete, dict(hors_perimetre)


def ic95(v) -> float:
    """Demi-largeur de l'intervalle de Student a 95 % ; 0 avec moins de deux valeurs."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def agreger(runs: list, latence_complete: bool) -> dict:
    groupes = defaultdict(list)
    for r in runs:
        groupes[r["label"]].append(r)
    agg = {}
    for label, rs in groupes.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        a = dict(arch=label, n=len(rs), cell=rs[0]["cell"], regime=rs[0]["regime"],
                 phase=rs[0]["phase"], bases=rs[0]["bases"], npar=rs[0]["npar"],
                 seeds=[r["seed"] for r in rs], runs=rs, lat_ok=all(r["lat"] is not None for r in rs))
        for k in ("mcc", "pr", "brier", "r99", "r80", "perf", "F80", "F99"):
            v = np.array([r[k] for r in rs], dtype=float)
            a[k], a[k + "_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            a[k + "_ic"] = ic95(v)
        if a["lat_ok"]:
            v = np.array([r["lat"] for r in rs], dtype=float)
            a["lat"], a["lat_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            a["pen"] = float(np.mean([r["pen"] for r in rs]))
        else:
            a["lat"], a["lat_sd"], a["pen"] = None, None, None
        agg[label] = a
    admissibles = [a for a in agg.values() if not latence_complete or a["lat"] <= LATENCE_T]
    for a in agg.values():
        a["admissible"] = a in admissibles
    poids = np.array([AHP_WEIGHT_MCC, AHP_WEIGHT_PR_AUC, AHP_WEIGHT_BRIER, AHP_WEIGHT_R2_SYMBOLIC])
    for cle, r2 in (("V5_80", "r80"), ("V5_99", "r99")):
        z = np.c_[_minmax(np.array([max(0.0, a["mcc"]) for a in admissibles])),
                  _minmax(np.array([a["pr"] for a in admissibles])),
                  1.0 - _minmax(np.array([a["brier"] for a in admissibles])),
                  _minmax(np.array([a[r2] for a in admissibles]))]
        scores = z @ poids / poids.sum()
        for a in agg.values():
            a[cle] = None
        for a, s in zip(admissibles, scores):
            a[cle] = float(s)
    for cle in ("F80", "F99", "perf", "r80", "r99", "V5_80", "V5_99"):
        presents = [a for a in agg.values() if a[cle] is not None]
        rangs = stats.rankdata([-a[cle] for a in presents], method="min")
        for a in agg.values():
            a["rang_" + cle] = None
        for a, rg in zip(presents, rangs):
            a["rang_" + cle] = int(rg)
    return agg


def regle_preenregistree(agg: dict) -> dict:
    """Un ecart inferieur au plus grand des deux ecarts-types n'est pas une difference ; parmi
    les ex aequo, la moins couteuse en parametres puis en latence (chapitre 4)."""
    ordre = sorted(agg.values(), key=lambda a: a["F80"], reverse=True)
    best = ordre[0]
    ex_aequo = [a for a in ordre if best["F80"] - a["F80"] < max(best["F80_sd"], a["F80_sd"])]
    retenue = min(ex_aequo, key=lambda a: (a["npar"], a["lat"] if a["lat"] is not None else 0.0))
    return dict(best=best, ex_aequo=ex_aequo, retenue=retenue)


def finalistes_provisoires(agg: dict, latence_complete: bool) -> list:
    """Regle du 06/11 : fitness, detection (perf), symbolisation (R2 a 0,80), rapidite, plus B.
    Une architecture qui gagne deux criteres n'est pas remplacee."""
    tous = list(agg.values())
    roles = [("fitness", max(tous, key=lambda a: a["F80"])),
             ("detection", max(tous, key=lambda a: a["perf"])),
             ("symbolisation", max(tous, key=lambda a: a["r80"]))]
    if latence_complete:
        roles.append(("rapidite", min(tous, key=lambda a: a["lat"])))
    b = next((a for a in tous if a["cell"] == "lstm" and a["regime"] == "brut"
              and a["bases"] == dict(Forget="relukan", Input="efficientkan",
                                     Candidate="wavkan_dog", Output="linear")), None)
    if b is not None:
        roles.append(("B conservee", b))
    # L'ajout du 07/11 (meilleure TKAN du regime traite) a ete retire de l'analyse par Miguel le 09/11
    # (DECISIONS.md) : seuls les finalistes de la regle sont retenus.
    finals = {}
    for role, a in roles:
        finals.setdefault(a["arch"], dict(arch=a["arch"], roles=[]))["roles"].append(role)
    return list(finals.values())


def front_pareto(points: list, sens: list) -> list:
    """Indices non domines ; sens[k] = +1 a maximiser, -1 a minimiser."""
    x = np.array(points, dtype=float) * np.array(sens, dtype=float)
    return [i for i in range(len(x))
            if not any((x[j] >= x[i]).all() and (x[j] > x[i]).any() for j in range(len(x)) if j != i)]


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────

def influence(agg: dict, latence_complete: bool) -> list:
    """Poids x etendue du critere entre architectures, en part du total, aux deux seuils."""
    tous = list(agg.values())
    lignes = []
    for seuil, r2 in (("0,80", "r80"), ("0,99", "r99")):
        crit = [("MCC", AHP_WEIGHT_MCC, [max(0.0, a["mcc"]) for a in tous]),
                ("PR-AUC", AHP_WEIGHT_PR_AUC, [a["pr"] for a in tous]),
                ("Brier", AHP_WEIGHT_BRIER, [a["brier"] for a in tous]),
                ("R2 symbolique", AHP_WEIGHT_R2_SYMBOLIC, [a[r2] for a in tous])]
        if latence_complete:
            crit.append(("Pénalité de latence", AHP_WEIGHT_LATENCY_PENALTY, [a["pen"] for a in tous]))
        eff = [w * (max(v) - min(v)) for _, w, v in crit]
        s_eff, s_w = sum(eff) or 1.0, sum(w for _, w, _ in crit)
        for (nom, w, v), e in zip(crit, eff):
            lignes.append(dict(seuil=seuil, critere=nom, poids_pct=100 * w / s_w,
                               etendue=max(v) - min(v), influence_pct=100 * e / s_eff))
    return lignes


def correlations_rangs(agg: dict) -> dict:
    tous = list(agg.values())
    cles = [("F80", "perf"), ("F80", "r80"), ("F99", "perf"), ("F99", "r99"), ("F80", "F99")]
    if all(a["V5_80"] is not None for a in tous):
        cles += [("F80", "V5_80"), ("V5_80", "perf")]
    out = {}
    for x, y in cles:
        u, v = [a[x] for a in tous], [a[y] for a in tous]
        out[f"{x}/{y}"] = dict(spearman=float(stats.spearmanr(u, v)[0]),
                               kendall=float(stats.kendalltau(u, v)[0]))
    return out


def holm(p: list) -> list:
    valides = [(v, i) for i, v in enumerate(p) if v == v]
    valides.sort()
    ajustees, courant = [float("nan")] * len(p), 0.0
    for rang, (v, i) in enumerate(valides):
        courant = max(courant, min(1.0, (len(valides) - rang) * v))
        ajustees[i] = courant
    return ajustees


def difference_appariee(a: dict, b: dict, cle: str) -> dict:
    va, vb = {r["seed"]: r[cle] for r in a["runs"]}, {r["seed"]: r[cle] for r in b["runs"]}
    graines = sorted(set(va) & set(vb))
    d = np.array([va[s] - vb[s] for s in graines], dtype=float)
    if len(d) < 2:
        return dict(n=len(d), graines=graines, d=d.tolist(), moyenne=float("nan"), ic=0.0, p=float("nan"))
    p = 1.0 if d.std(ddof=1) == 0 else float(stats.ttest_rel(
        [va[s] for s in graines], [vb[s] for s in graines]).pvalue)
    return dict(n=len(d), graines=graines, d=d.tolist(), moyenne=float(d.mean()), ic=ic95(d), p=p)


def comparaisons(agg: dict, finals: list) -> list:
    """Comparaisons fixees a l'avance : finalistes et trois premiers contre la meilleure fitness ;
    finalistes contre la reference uniforme des qu'elle existe. Holm par critere."""
    tous = sorted(agg.values(), key=lambda a: a["F80"], reverse=True)
    ref = tous[0]
    noms = list(dict.fromkeys([f["arch"] for f in finals] + [a["arch"] for a in tous[:3]]))
    couples = [(agg[n], ref) for n in noms if n != ref["arch"]]
    uniforme = [a for a in tous if a["bases"].get("Forget") == "hybrid"]
    if uniforme:
        u = uniforme[0]
        # pas de doublon : le couple (uniforme, meilleure) figure deja dans la premiere liste
        couples += [(agg[f["arch"]], u) for f in finals
                    if f["arch"] not in (u["arch"], ref["arch"])]
    lignes = []
    for cle, nom in (("mcc", "MCC"), ("perf", "perf"), ("r80", "R2 à 0,80"), ("F80", "Fitness à 0,80")):
        groupe = []
        for a, b in couples:
            res = difference_appariee(a, b, cle)
            groupe.append(dict(comparaison=f"{a['arch']} − {b['arch']}", a=a["arch"], b=b["arch"],
                               critere=nom, cle=cle, **res))
        for g, p in zip(groupe, holm([g["p"] for g in groupe])):
            g["p_holm"] = p
        lignes += groupe
    return lignes


def effets_bases(runs: list) -> list:
    """Effets marginaux par porte, sur le plan complet en regime brut, en ecart a la moyenne de
    la cellule. Plan equilibre seulement si toutes les graines sont la (colonne n)."""
    lignes = []
    brut = [r for r in runs if r["phase"] == "etape4_brut"]
    for cell in ("lstm", "gru"):
        rs = [r for r in brut if r["cell"] == cell]
        if not rs:
            continue
        cles = ["mcc", "perf", "r80", "r99"] + (["lat"] if all(r["lat"] is not None for r in rs) else [])
        moy = {k: float(np.mean([r[k] for r in rs])) for k in cles}
        for porte in PORTES_AFFICHEES[cell]:
            for base in sorted({r["bases"][porte] for r in rs if porte in r["bases"]}):
                sous = [r for r in rs if r["bases"].get(porte) == base]
                ligne = dict(cellule=cell, porte=porte, base=base, n=len(sous))
                for k in cles:
                    ligne[k], ligne["ecart_" + k] = (float(np.mean([r[k] for r in sous])),
                                                     float(np.mean([r[k] for r in sous])) - moy[k])
                lignes.append(ligne)
    return lignes


def interactions(runs: list, cle: str) -> list:
    lignes = []
    brut = [r for r in runs if r["phase"] == "etape4_brut"]
    for cell in ("lstm", "gru"):
        rs = [r for r in brut if r["cell"] == cell]
        if not rs:
            continue
        moy = float(np.mean([r[cle] for r in rs]))
        for p1, p2 in combinations(PORTES_AFFICHEES[cell], 2):
            cases = defaultdict(list)
            for r in rs:
                cases[(r["bases"][p1], r["bases"][p2])].append(r[cle])
            for (b1, b2), v in cases.items():
                lignes.append(dict(cellule=cell, porte_1=p1, base_1=b1, porte_2=p2, base_2=b2,
                                   n=len(v), critere=cle, valeur=float(np.mean(v)),
                                   ecart=float(np.mean(v)) - moy))
    return lignes


def aretes_par_porte(runs: list) -> list:
    """Repartition par porte des 40 aretes de plus forte norme L1 (matrice exacte sur le pool
    d'audit, edge_profiles.npz), meme selection que le calcul du R2 symbolique."""
    lignes = []
    for r in runs:
        chemin = os.path.join(r["rep"], "edge_profiles.npz")
        if not os.path.exists(chemin):
            continue
        with np.load(chemin) as z:
            v, g = [], []
            for k in z.files:
                if k.endswith("_gate_l1"):
                    x = z[k].ravel()
                    v.append(x)
                    g += [PORTES_NPZ[k[: -len("_gate_l1")]]] * len(x)
        v, g = np.concatenate(v), np.array(g)
        garde = v > THETA_ELAGAGE
        top = g[garde][np.argsort(-v[garde])[:N_ARETES_AUDIT]]
        ligne = dict(run_id=r["run_id"], arch=r["label"], cell=r["cell"], seed=r["seed"],
                     sortie=r["bases"].get("Output"), survivantes=int(garde.sum()),
                     aretes_totales=len(v), r99=r["r99"], r80=r["r80"])
        for nom in ORDRE_PORTES_NPZ:
            ligne[nom] = int((top == nom).sum())
        lignes.append(ligne)
    return lignes


def stabilite(runs: list, agg: dict) -> dict:
    out = {}
    brut = [r for r in runs if r["phase"] == "etape4_brut"]
    for cell in ("lstm", "gru"):
        par_arch = defaultdict(dict)
        for r in brut:
            if r["cell"] == cell:
                par_arch[r["label"]][r["seed"]] = r["perf"]
        complets = [d for d in par_arch.values() if len(d) == GRAINES_ATTENDUES]
        if len(complets) < 3:
            continue
        graines = sorted(complets[0])
        corr = [stats.spearmanr([d[s] for d in complets], [d[t] for d in complets])[0]
                for s, t in combinations(graines, 2)]
        out[f"correlation_rangs_perf_{cell}"] = float(np.mean(corr))
    dd = [a["perf_sd"] for a in agg.values() if a["phase"] == "etape4_brut"
          and a["bases"].get("Forget") == "wavkan_dog" and a["bases"].get("Candidate") == "wavkan_dog"]
    au = [a["perf_sd"] for a in agg.values() if a["phase"] == "etape4_brut"
          and not (a["bases"].get("Forget") == "wavkan_dog" and a["bases"].get("Candidate") == "wavkan_dog")]
    if dd and au:
        out["ecart_type_perf_forget_candidate_dog"] = float(np.mean(dd))
        out["ecart_type_perf_autres"] = float(np.mean(au))
    out["runs_arretes_par_patience"] = sum(r["stopped"] for r in runs)
    out["runs_coupes_par_plafond"] = sum(not r["stopped"] for r in runs)
    out["runs_a_15_epoques"] = sum(r["epochs"] == 15 for r in runs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures (Plotly : .html dans <out>/figures, .png sans titre dans memoire/figure)
# ─────────────────────────────────────────────────────────────────────────────

class Sortie:
    def __init__(self, html_dir: str, png_dir: str, png: bool):
        self.html_dir, self.png_dir, self.png = html_dir, png_dir, png
        os.makedirs(html_dir, exist_ok=True)
        if png:
            os.makedirs(png_dir, exist_ok=True)

    def enregistrer(self, fig, nom: str, titre: str, sous_titre: str) -> None:
        fig.update_layout(title=dict(text=f"{titre}<br><sup>{sous_titre}</sup>", x=0.01, xanchor="left"),
                          margin=dict(t=110))
        chemin = os.path.join(self.html_dir, nom + ".html")
        fig.write_html(chemin)
        print(chemin)
        if not self.png:
            return
        sans_titre = copy.deepcopy(fig)
        sans_titre.update_layout(title=None, margin=dict(t=44))
        try:
            sans_titre.write_image(os.path.join(self.png_dir, nom + ".png"), scale=2)
        except Exception as exc:   # kaleido peut manquer selon l'environnement
            print(f"[avertissement] export PNG impossible pour {nom} ({exc}) : HTML seul disponible.")


def _mise_en_page(fig, hauteur: int, largeur: int = 1000, **kw) -> None:
    fig.update_layout(template="plotly_white", height=hauteur, width=largeur, separators=", ",
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(color=ENCRE, size=12, family="Segoe UI, Arial, sans-serif"),
                      margin=dict(l=10, r=20, b=60), hoverlabel=dict(namelength=-1), **kw)
    fig.update_xaxes(gridcolor=GRILLE, linecolor=GRIS_CLAIR, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRILLE, linecolor=GRIS_CLAIR, zeroline=False, automargin=True)


def _etat(runs: list, latence_complete: bool) -> str:
    return (f"Provisoire : {len(runs)} runs, validation seule"
            + ("" if latence_complete else ", latence incomplète donc hors fitness") + ".")


def _noms_finalistes(finals: list) -> set:
    return {f["arch"] for f in finals}


def _ligne_y(a: dict, noms: set) -> str:
    return f"<b>{a['arch']}</b>" if a["arch"] in noms else a["arch"]


def fig_1_points_graines(agg, finals, sortie, etat):
    """Qui gagne, avec quelle incertitude : chaque graine, la moyenne, l'intervalle a 95 %."""
    from plotly.subplots import make_subplots
    noms = _noms_finalistes(finals)
    ordre = sorted(agg.values(), key=lambda a: a["F80"], reverse=True)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=("Détection (score perf)", "R² symbolique au seuil 0,80"))
    for col, (cle, borne) in enumerate((("perf", None), ("r80", (0.0, 1.0))), start=1):
        xg, yg = [], []
        for a in ordre:
            for r in a["runs"]:
                xg.append(r[cle])
                yg.append(_ligne_y(a, noms))
        fig.add_scatter(x=xg, y=yg, mode="markers", name="Une graine", legendgroup="graine",
                        showlegend=col == 1, marker=dict(color=GRIS_CLAIR, size=6),
                        row=1, col=col, hovertemplate="%{y}<br>graine : %{x:.4f}<extra></extra>")
        for cell in ("lstm", "gru"):
            for finaliste in (False, True):
                sel = [a for a in ordre if a["cell"] == cell and (a["arch"] in noms) == finaliste]
                if not sel:
                    continue
                moy = [a[cle] for a in sel]
                bas = [a[cle + "_ic"] if borne is None else a[cle] - max(borne[0], a[cle] - a[cle + "_ic"]) for a in sel]
                haut = [a[cle + "_ic"] if borne is None else min(borne[1], a[cle] + a[cle + "_ic"]) - a[cle] for a in sel]
                couleur = ACCENT if finaliste else ENCRE_2
                nom = NOM_CELLULE[cell] + (", finaliste provisoire" if finaliste else "")
                fig.add_scatter(x=moy, y=[_ligne_y(a, noms) for a in sel], mode="markers", name=nom,
                                legendgroup=nom, showlegend=col == 1,
                                error_x=dict(type="data", symmetric=False, array=haut, arrayminus=bas,
                                             color=couleur, thickness=1.5, width=0),
                                marker=dict(symbol=SYMBOLE[cell], color=couleur, size=9,
                                            line=dict(color="#ffffff", width=2)),
                                customdata=[a["n"] for a in sel], row=1, col=col,
                                hovertemplate="%{y}<br>moyenne : %{x:.4f}<br>graines : %{customdata}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=[_ligne_y(a, noms) for a in ordre],
                     autorange="reversed", showgrid=True)
    fig.update_xaxes(title_text="perf", row=1, col=1)
    fig.update_xaxes(title_text="R² à 0,80", range=[-0.03, 1.03], row=1, col=2)
    _mise_en_page(fig, 150 + 22 * len(ordre), 1100, legend=dict(orientation="h", y=-0.07))
    sortie.enregistrer(fig, "banc_01_points_graines", "Résultats par architecture et par graine",
                       f"Tri par fitness à 0,80 ; moyenne et intervalle de Student à 95 % ; en gras, les finalistes. {etat}")


def fig_2_differences(comp, sortie, etat):
    """Les ecarts sont-ils reels : differences appariees par graine, avec intervalle."""
    from plotly.subplots import make_subplots
    criteres = [("perf", "Détection (score perf)"), ("r80", "R² à 0,80"), ("F80", "Fitness à 0,80")]
    lignes = list(dict.fromkeys(c["comparaison"] for c in comp))
    if not lignes:
        return
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[n for _, n in criteres])
    for col, (cle, _) in enumerate(criteres, start=1):
        sel = [c for c in comp if c["cle"] == cle and c["n"] >= 2]
        xg, yg = [], []
        for c in sel:
            xg += c["d"]
            yg += [c["comparaison"]] * len(c["d"])
        fig.add_scatter(x=xg, y=yg, mode="markers", name="Écart sur une graine", legendgroup="g",
                        showlegend=col == 1, marker=dict(color=GRIS_CLAIR, size=7), row=1, col=col,
                        hovertemplate="%{y}<br>écart : %{x:.4f}<extra></extra>")
        fig.add_scatter(x=[c["moyenne"] for c in sel], y=[c["comparaison"] for c in sel], mode="markers",
                        name="Écart moyen et intervalle à 95 %", legendgroup="m", showlegend=col == 1,
                        error_x=dict(type="data", array=[c["ic"] for c in sel], color=ACCENT, thickness=1.5, width=0),
                        marker=dict(color=ACCENT, size=10, line=dict(color="#ffffff", width=2)),
                        customdata=[[c["p"], c["p_holm"], c["n"]] for c in sel], row=1, col=col,
                        hovertemplate="%{y}<br>écart : %{x:.4f}<br>p : %{customdata[0]:.3f}<br>"
                                      "p (Holm) : %{customdata[1]:.3f}<br>graines : %{customdata[2]}<extra></extra>")
        fig.add_vline(x=0, line=dict(color=ENCRE_2, width=1), row=1, col=col)
    fig.update_yaxes(categoryorder="array", categoryarray=lignes, autorange="reversed")
    _mise_en_page(fig, 170 + 46 * len(lignes), 1200, legend=dict(orientation="h", y=-0.12 if len(lignes) > 4 else -0.25))
    sortie.enregistrer(fig, "banc_02_differences_appariees", "Différences appariées sur les graines",
                       f"Test t apparié indicatif : p et p corrigée de Holm dans diff_appariees.csv. {etat}")


def fig_3_compromis(agg, finals, sortie, etat, latence_complete):
    """Quel compromis : nuage avec front de Pareto, plus un panneau de latence s'il est mesure."""
    from plotly.subplots import make_subplots
    noms = _noms_finalistes(finals)
    tous = list(agg.values())
    colonnes = [("r80", "R² symbolique à 0,80", +1, "hv")]
    if latence_complete:
        colonnes.append(("lat", "Latence moyenne, lot de 256 fenêtres (ms)", -1, "hv"))
    fig = make_subplots(rows=1, cols=len(colonnes), shared_yaxes=True, horizontal_spacing=0.04)
    # Decalage des etiquettes par panneau, dans l'ordre des finalistes (fitness, detection, rapidite, B, ajout)
    decalages = {"r80": [(-95, 28), (40, -22), (-90, -42), (45, -45), (40, 60)],
                 "lat": [(110, 40), (130, 0), (60, 55), (90, 62), (40, 60)]}
    for col, (cle, titre, sens, forme) in enumerate(colonnes, start=1):
        front = [tous[i] for i in front_pareto([[a["perf"], a[cle]] for a in tous], [1, sens])]
        front.sort(key=lambda a: a[cle])
        fig.add_scatter(x=[a[cle] for a in front], y=[a["perf"] for a in front], mode="lines",
                        name="Front de Pareto", legendgroup="front", showlegend=col == 1,
                        line=dict(color=ENCRE_2, width=2, shape=forme), row=1, col=col, hoverinfo="skip")
        for cell in ("lstm", "gru"):
            for finaliste in (False, True):
                sel = [a for a in tous if a["cell"] == cell and (a["arch"] in noms) == finaliste]
                if not sel:
                    continue
                nom = NOM_CELLULE[cell] + (", finaliste provisoire" if finaliste else "")
                couleur = ACCENT if finaliste else GRIS
                fig.add_scatter(x=[a[cle] for a in sel], y=[a["perf"] for a in sel], mode="markers", name=nom,
                                legendgroup=nom, showlegend=col == 1, text=[a["arch"] for a in sel],
                                marker=dict(symbol=SYMBOLE[cell], color=couleur, size=11 if finaliste else 9,
                                            line=dict(color="#ffffff", width=2)), row=1, col=col,
                                hovertemplate="%{text}<br>x : %{x:.3f}<br>perf : %{y:.4f}<extra></extra>")
        for i, f in enumerate(finals):
            a = agg[f["arch"]]
            ax, ay = decalages[cle][i % len(decalages[cle])]
            fig.add_annotation(x=a[cle], y=a["perf"], text=a["arch"], ax=ax, ay=ay, row=1, col=col,
                               font=dict(size=10, color=ENCRE), arrowcolor=GRIS, arrowwidth=1, arrowhead=0,
                               bgcolor="rgba(255,255,255,0.85)")
        fig.update_xaxes(title_text=titre, row=1, col=col)
    fig.update_yaxes(title_text="Détection (score perf)", row=1, col=1)
    _mise_en_page(fig, 560, 1200 if latence_complete else 800, legend=dict(orientation="h", y=-0.2))
    sortie.enregistrer(fig, "banc_03_compromis_pareto", "Compromis entre détection, symbolisation et latence",
                       f"Moyennes par architecture ; un point du front n'est dominé par aucun autre sur les deux axes du panneau. {etat}")


def fig_4_bases_par_porte(effets, sortie, etat, latence_complete):
    """Quelle base a quelle porte : ecart a la moyenne de la cellule, plan brut."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    criteres = [("ecart_perf", "Détection (perf)", 4, DIVERGENT), ("ecart_r80", "R² à 0,80", 2, DIVERGENT)]
    if latence_complete:
        criteres.append(("ecart_lat", "Latence (ms), bleu = plus rapide", 0, DIVERGENT_INVERSE))
    cellules = [c for c in ("lstm", "gru") if any(e["cellule"] == c for e in effets)]
    if not cellules:
        return
    fig = make_subplots(rows=len(cellules), cols=len(criteres), horizontal_spacing=0.09, vertical_spacing=0.14,
                        subplot_titles=[f"{NOM_CELLULE[c]} : {t}" for c in cellules for _, t, _, _ in criteres])
    bases = ["relukan", "wavkan_dog", "efficientkan", "wavkan_morlet", "linear"]
    for i, cell in enumerate(cellules, start=1):
        portes = list(PORTES_AFFICHEES[cell])
        for j, (cle, _, dec, _) in enumerate(criteres, start=1):
            z, txt = [], []
            for p in portes:
                ligne_z, ligne_t = [], []
                for b in bases:
                    e = next((e for e in effets if e["cellule"] == cell and e["porte"] == p and e["base"] == b), None)
                    v = None if e is None or cle not in e else e[cle]
                    ligne_z.append(v)
                    ligne_t.append("" if v is None else f"{v:+.{dec}f}".replace(".", ",").replace("-", "−"))
                z.append(ligne_z)
                txt.append(ligne_t)
            axe = "coloraxis" if j == 1 else f"coloraxis{j}"
            fig.add_trace(go.Heatmap(z=z, x=[NOM_BASE[b] for b in bases],
                                     y=[PORTES_AFFICHEES[cell][p] for p in portes], text=txt,
                                     texttemplate="%{text}", textfont=dict(size=11, color=ENCRE), xgap=2, ygap=2,
                                     coloraxis=axe, hoverongaps=False,
                                     hovertemplate="%{y}, %{x}<br>écart : %{text}<extra></extra>"), row=i, col=j)
    disposition = {}
    for j, (cle, titre, _, echelle) in enumerate(criteres, start=1):
        m = max((abs(e[cle]) for e in effets if cle in e), default=1.0) or 1.0
        nom = "coloraxis" if j == 1 else f"coloraxis{j}"
        disposition[nom] = dict(colorscale=echelle, cmin=-m, cmax=m, showscale=False)
    fig.update_layout(**disposition)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    _mise_en_page(fig, 240 * len(cellules) + 80, 1250 if latence_complete else 900)
    sortie.enregistrer(fig, "banc_04_bases_par_porte", "Effet de chaque base, porte par porte",
                       f"Écart à la moyenne de la cellule, régime brut, sans isoler l'effet du nombre de paramètres. {etat}")


def fig_5_rangs(agg, finals, sortie, etat, corr):
    """Le classement depend-il de la formule : pente des rangs."""
    import plotly.graph_objects as go
    colonnes = [("rang_F80", "Fitness<br>R² à 0,80"), ("rang_F99", "Fitness<br>R² à 0,99"),
                ("rang_perf", "Détection<br>perf (V4)"), ("rang_V5_80", "AHP normalisé<br>V5, R² à 0,80")]
    tous = list(agg.values())
    n = len(tous)
    fig = go.Figure()
    noms = _noms_finalistes(finals)
    for a in tous:
        if a["arch"] in noms:
            continue
        fig.add_scatter(x=list(range(4)), y=[a[c] for c, _ in colonnes], mode="lines", showlegend=False,
                        line=dict(color=GRILLE, width=1), text=[a["arch"]] * 4, connectgaps=False,
                        hovertemplate="%{text}<br>rang %{y}<extra></extra>")
    arrivees = sorted(finals, key=lambda f: agg[f["arch"]]["rang_V5_80"] or n + 1)
    y_label, dernier = {}, -10.0
    for f in arrivees:
        rang = agg[f["arch"]]["rang_V5_80"] or n
        dernier = max(rang, dernier + 1.3)
        y_label[f["arch"]] = dernier
    for i, f in enumerate(finals):
        a = agg[f["arch"]]
        couleur = SLOTS[i % len(SLOTS)]
        ys = [a[c] for c, _ in colonnes]
        fig.add_scatter(x=list(range(4)), y=ys, mode="lines+markers", name=a["arch"], connectgaps=False,
                        line=dict(color=couleur, width=2), text=[a["arch"]] * 4,
                        marker=dict(symbol=SYMBOLE[a["cell"]], color=couleur, size=10, line=dict(color="#ffffff", width=2)),
                        hovertemplate="%{text}<br>rang %{y}<extra></extra>")
        fig.add_shape(type="line", x0=3.02, x1=3.1, y0=ys[3] if ys[3] is not None else n, y1=y_label[a["arch"]],
                      line=dict(color=GRIS, width=1))
        fig.add_annotation(x=3.11, y=y_label[a["arch"]], text=f"{a['arch']} : {ys[3] if ys[3] is not None else 'n.d.'}",
                           showarrow=False, xanchor="left", font=dict(size=11, color=ENCRE))
    fig.update_xaxes(tickvals=list(range(4)), ticktext=[t for _, t in colonnes], range=[-0.1, 4.25], side="top", showgrid=False)
    fig.update_yaxes(title_text="Rang (1 = meilleur)", range=[n + 0.7, 0.3], tick0=1, dtick=5, showgrid=False)
    _mise_en_page(fig, 140 + 19 * n, 1050, showlegend=True, legend=dict(orientation="h", y=-0.05))
    tau = corr.get("F80/perf", {}).get("kendall")
    sortie.enregistrer(fig, "banc_05_rangs_selon_formule", "Rang des architectures selon la formule de classement",
                       f"Finalistes en couleur, autres en gris ; tau de Kendall fitness à 0,80 contre perf = {fr(tau, 2)}. {etat}")


def fig_6_seuil_aretes(agg, finals, aretes, sortie, etat):
    """Que mesure le R2 : effet du seuil, et ou se trouvent les 40 aretes evaluees."""
    from plotly.subplots import make_subplots
    noms = _noms_finalistes(finals)
    ordre = sorted(agg.values(), key=lambda a: (a["r80"], a["r99"]), reverse=True)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.54, 0.46], horizontal_spacing=0.2,
                        subplot_titles=("R² symbolique aux deux seuils", "40 arêtes évaluées, par porte (moyenne des graines)"))
    ys = [_ligne_y(a, noms) for a in ordre]
    xl, yl = [], []
    for a, y in zip(ordre, ys):
        xl += [a["r99"], a["r80"], None]
        yl += [y, y, None]
    fig.add_scatter(x=xl, y=yl, mode="lines", line=dict(color=GRIS_CLAIR, width=2), showlegend=False, hoverinfo="skip", row=1, col=1)
    symboles = [SYMBOLE[a["cell"]] for a in ordre]
    fig.add_scatter(x=[a["r99"] for a in ordre], y=ys, mode="markers", name="Seuil 0,99",
                    marker=dict(symbol=symboles, color=GRIS, size=8, line=dict(color="#ffffff", width=2)),
                    row=1, col=1, hovertemplate="%{y}<br>R² à 0,99 : %{x:.3f}<extra></extra>")
    fig.add_scatter(x=[a["r80"] for a in ordre], y=ys, mode="markers", name="Seuil 0,80",
                    marker=dict(symbol=symboles, color=ACCENT, size=9, line=dict(color="#ffffff", width=2)),
                    row=1, col=1, hovertemplate="%{y}<br>R² à 0,80 : %{x:.3f}<extra></extra>")
    fig.update_xaxes(title_text="Proportion d'arêtes symbolisables", range=[-0.03, 1.03], row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=ys, autorange="reversed", row=1, col=1)

    def groupe(l):
        if l["cell"] == "gru":
            return "GRU (toutes)"
        return "TKAN, sortie linéaire" if l["sortie"] == "linear" else "TKAN, sortie non linéaire"
    rangees = defaultdict(list)
    for l in aretes:
        rangees[groupe(l)].append(l)
        if l["arch"] in noms:
            rangees[l["arch"]].append(l)
    etiquettes = [e for e in ("TKAN, sortie linéaire", "TKAN, sortie non linéaire", "GRU (toutes)") if e in rangees]
    etiquettes += [f["arch"] for f in finals if f["arch"] in rangees]
    for porte, couleur, encre in zip(ORDRE_PORTES_NPZ, SLOTS, ("#ffffff", "#ffffff", "#ffffff", "#0b0b0b")):
        v = [float(np.mean([l[porte] for l in rangees[e]])) for e in etiquettes]
        fig.add_bar(y=etiquettes, x=v, orientation="h", name=porte, width=0.24,
                    marker=dict(color=couleur, line=dict(color="#ffffff", width=2)),
                    text=[fr(x, 0) if x >= 4 else "" for x in v], textposition="inside", insidetextanchor="start",
                    textfont=dict(color=encre, size=11), row=1, col=2,
                    hovertemplate="%{y}<br>" + porte + " : %{x:.1f} arêtes<extra></extra>")
    fig.update_xaxes(title_text="Nombre d'arêtes", range=[0, N_ARETES_AUDIT], row=1, col=2)
    fig.update_yaxes(categoryorder="array", categoryarray=etiquettes, autorange="reversed", row=1, col=2)
    _mise_en_page(fig, 150 + 21 * len(ordre), 1300, barmode="stack", legend=dict(orientation="h", y=-0.08))
    sortie.enregistrer(fig, "banc_06_seuil_et_aretes", "Ce que mesure le R² symbolique",
                       f"Le R² porte sur 40 arêtes parmi toutes les survivantes à l'élagage. {etat}")


def fig_a1_interactions(runs, cle, titre, dec, sortie, etat):
    """Annexe : interactions entre portes, deux a deux."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    lignes = interactions(runs, cle)
    if not lignes:
        return
    cases = [(cell, p1, p2) for cell in ("lstm", "gru") for p1, p2 in combinations(PORTES_AFFICHEES[cell], 2)]
    titres = [f"{NOM_CELLULE[c]} : {PORTES_AFFICHEES[c][p1]} × {PORTES_AFFICHEES[c][p2]}" for c, p1, p2 in cases]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titres, horizontal_spacing=0.09, vertical_spacing=0.1)
    m = max(abs(l["ecart"]) for l in lignes) or 1.0
    ordre_bases = list(NOM_BASE)
    for rang, (cell, p1, p2) in enumerate(cases):
        sel = [l for l in lignes if l["cellule"] == cell and l["porte_1"] == p1 and l["porte_2"] == p2]
        b1 = sorted({l["base_1"] for l in sel}, key=ordre_bases.index)
        b2 = sorted({l["base_2"] for l in sel}, key=ordre_bases.index)
        z = [[next((l["ecart"] for l in sel if l["base_1"] == u and l["base_2"] == v), None) for v in b2] for u in b1]
        txt = [["" if x is None else f"{x:+.{dec}f}".replace(".", ",").replace("-", "−") for x in ligne] for ligne in z]
        fig.add_trace(go.Heatmap(z=z, x=[NOM_BASE[v] for v in b2], y=[NOM_BASE[u] for u in b1], text=txt,
                                 texttemplate="%{text}", textfont=dict(size=11, color=ENCRE), xgap=2, ygap=2,
                                 colorscale=DIVERGENT, zmin=-m, zmax=m, showscale=False, hoverongaps=False,
                                 hovertemplate="%{y} × %{x}<br>écart : %{text}<extra></extra>"),
                      row=rang // 3 + 1, col=rang % 3 + 1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    _mise_en_page(fig, 860, 1150)
    sortie.enregistrer(fig, f"banc_a1_interactions_{cle}", f"Interactions entre portes : {titre}",
                       f"Écart à la moyenne de la cellule, régime brut. {etat}")


def fig_a2_taille(agg, finals, corr_taille, sortie, etat, latence_complete):
    """Annexe : le nombre de parametres confond l'effet des bases."""
    from plotly.subplots import make_subplots
    noms = _noms_finalistes(finals)
    colonnes = [("perf", "Détection (perf)")] + ([("lat", "Latence (ms)")] if latence_complete else [])
    fig = make_subplots(rows=1, cols=len(colonnes), horizontal_spacing=0.09)
    for col, (cle, titre) in enumerate(colonnes, start=1):
        for cell in ("lstm", "gru"):
            for finaliste in (False, True):
                sel = [a for a in agg.values() if a["cell"] == cell and (a["arch"] in noms) == finaliste]
                if not sel:
                    continue
                nom = NOM_CELLULE[cell] + (", finaliste provisoire" if finaliste else "")
                fig.add_scatter(x=[a["npar"] for a in sel], y=[a[cle] for a in sel], mode="markers", name=nom,
                                legendgroup=nom, showlegend=col == 1, text=[a["arch"] for a in sel],
                                marker=dict(symbol=SYMBOLE[cell], color=ACCENT if finaliste else GRIS, size=10,
                                            line=dict(color="#ffffff", width=2)), row=1, col=col,
                                hovertemplate="%{text}<br>paramètres : %{x}<br>%{y:.4f}<extra></extra>")
        fig.update_xaxes(title_text="Nombre de paramètres", row=1, col=col)
        fig.update_yaxes(title_text=titre, row=1, col=col)
    _mise_en_page(fig, 480, 1150 if latence_complete else 700, legend=dict(orientation="h", y=-0.2))
    texte = ", ".join(f"{NOM_CELLULE[c]} {fr(v, 2)}" for c, v in corr_taille.items())
    sortie.enregistrer(fig, "banc_a2_taille", "Taille du modèle, détection et latence",
                       f"Corrélation de rang paramètres et perf : {texte}. {etat}")


def fig_a3_arrets(runs, sortie, etat):
    """Annexe : epoque d'arret, par la patience ou par le plafond."""
    from plotly.subplots import make_subplots
    cellules = [c for c in ("lstm", "gru") if any(r["cell"] == c for r in runs)]
    fig = make_subplots(rows=1, cols=len(cellules), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[NOM_CELLULE[c] for c in cellules])
    epoques = list(range(1, max(r["epochs"] for r in runs) + 1))
    for col, cell in enumerate(cellules, start=1):
        for nom, couleur, test in (("Arrêt par la patience", SLOTS[0], True), ("Coupé par le plafond d'époques", SLOTS[1], False)):
            y = [sum(1 for r in runs if r["cell"] == cell and r["epochs"] == e and r["stopped"] == test) for e in epoques]
            fig.add_bar(x=epoques, y=y, name=nom, legendgroup=nom, showlegend=col == 1, width=0.7,
                        marker=dict(color=couleur, line=dict(color="#ffffff", width=2)), row=1, col=col,
                        hovertemplate="époque %{x}<br>%{y} runs<extra></extra>")
        fig.update_xaxes(title_text="Époques effectuées", dtick=1, row=1, col=col)
    fig.update_yaxes(title_text="Nombre de runs", row=1, col=1)
    _mise_en_page(fig, 420, 1000, barmode="stack", legend=dict(orientation="h", y=-0.22))
    sortie.enregistrer(fig, "banc_a3_arrets", "Époque d'arrêt des entraînements", etat)


def fig_a4_courbes(agg, finals, sortie, etat):
    """Annexe : courbes de perte des finalistes, toutes graines."""
    from plotly.subplots import make_subplots
    n = len(finals)
    titres = [f["arch"] for f in finals] + [""] * n
    fig = make_subplots(rows=2, cols=n, shared_yaxes=True, shared_xaxes=True, subplot_titles=titres,
                        horizontal_spacing=0.02, vertical_spacing=0.07)
    for col, f in enumerate(finals, start=1):
        couleur = SLOTS[(col - 1) % len(SLOTS)]
        for r in agg[f["arch"]]["runs"]:
            for ligne, cle in ((1, "train_hist"), (2, "mon_hist")):
                y = r[cle]
                fig.add_scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines", showlegend=False, opacity=0.8,
                                line=dict(color=couleur, width=2), row=ligne, col=col,
                                hovertemplate=f"graine {r['seed']}<br>époque %{{x}}<br>perte %{{y:.4f}}<extra></extra>")
        fig.update_xaxes(title_text="Époque", row=2, col=col)
    fig.update_yaxes(title_text="Perte d'entraînement", row=1, col=1)
    fig.update_yaxes(title_text="Perte de suivi (validation)", row=2, col=1)
    fig.update_annotations(font_size=11)
    _mise_en_page(fig, 560, 300 * n + 100)
    sortie.enregistrer(fig, "banc_a4_courbes_perte", "Courbes de perte des finalistes provisoires",
                       f"La perte d'entraînement inclut la régularisation : les deux lignes ne se comparent pas. {etat}")


def _noms_entrees(rep: str, regime: str = "") -> list:
    try:
        import torch
        ck = torch.load(os.path.join(rep, "model.pt"), map_location="cpu", weights_only=False)
        return list(ck["preprocessing"]["feature_cols"])
    except Exception as exc:   # model.pt absent des copies legeres : colonnes du regime (harness)
        try:
            import niveau1_harness as h
            return list(h.PROCESSED_FEATURE_COLS) if regime == "traite" else list(h.RAW_NUMERIC_COLS) + list(h.ACTION_ONEHOT_COLS)
        except Exception as exc2:
            print(f"[info] noms des variables indisponibles ({exc} ; {exc2}) : x[k] utilise.")
            return []


def fig_a5_profils(agg, finals, sortie, etat, n_aretes: int = 6):
    """Annexe : courbes apprises des aretes les plus fortes, graine de MCC median. Les unites
    cachees ne se correspondent pas d'une graine a l'autre : aucune moyenne entre graines."""
    from plotly.subplots import make_subplots
    panneaux = []
    for f in finals:
        rs = sorted(agg[f["arch"]]["runs"], key=lambda r: r["perf"])
        r = rs[len(rs) // 2]
        chemin = os.path.join(r["rep"], "edge_profiles.npz")
        if not os.path.exists(chemin):
            continue
        noms_x = _noms_entrees(r["rep"], r["regime"])
        with np.load(chemin) as z:
            cand = []
            for k in z.files:
                if k.endswith("_gate_l1"):
                    porte = k[: -len("_gate_l1")]
                    l1 = z[k]
                    for i, j in zip(*np.nonzero(l1 > THETA_ELAGAGE)):
                        cand.append((float(l1[i, j]), porte, int(i), int(j)))
            cand.sort(reverse=True)
            for l1, porte, i, j in cand[:n_aretes]:
                grille, phi = z[f"{porte}_gate_grid"][:, i], z[f"{porte}_gate_phi"][:, i, j]
                n_h = r["hidden"]
                if i < n_h:
                    entree = f"h[{i}]"
                else:
                    k = i - n_h
                    entree = noms_x[k] if k < len(noms_x) else f"x[{k}]"
                panneaux.append(dict(arch=f["arch"], seed=r["seed"], x=grille.tolist(), y=phi.tolist(),
                                     titre=f"{porte.capitalize()} : {entree} → {j}", l1=l1))
    if not panneaux:
        return
    archs = list(dict.fromkeys(p["arch"] for p in panneaux))
    fig = make_subplots(rows=len(archs), cols=n_aretes, horizontal_spacing=0.03, vertical_spacing=0.09)
    titres = []
    for ligne, a in enumerate(archs, start=1):
        sel = [p for p in panneaux if p["arch"] == a]
        for col, p in enumerate(sel, start=1):
            fig.add_scatter(x=p["x"], y=p["y"], mode="lines", showlegend=False, line=dict(color=SLOTS[(ligne - 1) % len(SLOTS)], width=2),
                            row=ligne, col=col, hovertemplate=f"{p['titre']}<br>L1 = {p['l1']:.3f}<br>entrée %{{x:.2f}}<br>φ %{{y:.3f}}<extra></extra>")
            titres.append(dict(text=p["titre"], ligne=ligne, col=col))
        fig.update_yaxes(title_text=f"{a}<br>graine {sel[0]['seed']}", title_font_size=10, row=ligne, col=1)
    for t in titres:
        sous_figure = fig.get_subplot(t["ligne"], t["col"])
        fig.add_annotation(xref=f"{sous_figure.yaxis.anchor} domain", yref=f"{sous_figure.xaxis.anchor} domain",
                           x=0.5, y=1.08, text=t["text"], showarrow=False, font=dict(size=9))
    _mise_en_page(fig, 240 * len(archs) + 90, 1300)
    sortie.enregistrer(fig, "banc_a5_profils_aretes", "Courbes apprises des arêtes les plus fortes",
                       f"Graine de MCC médian, sans moyenne entre graines : lecture descriptive, pas une formule extraite. {etat}")


# ─────────────────────────────────────────────────────────────────────────────
# Tableaux LaTeX et rapport
# ─────────────────────────────────────────────────────────────────────────────

def _tableau(chemin: str, legende: str, etiquette: str, colonnes: str, entete: str, lignes: list) -> None:
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                f"\\caption{{{legende}}}\n\\label{{{etiquette}}}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n"   # reduit seulement les tableaux trop larges
                f"\\begin{{tabular}}{{@{{}}{colonnes}@{{}}}}\n\\toprule\n{entete} \\\\\n\\midrule\n")
        f.write(" \\\\\n".join(lignes) + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    print(chemin)


def ecrire_tableaux(rep: str, agg: dict, finals: list, infl: list, comp: list, resume: str) -> None:
    os.makedirs(rep, exist_ok=True)
    noms = _noms_finalistes(finals)
    ordre = sorted(agg.values(), key=lambda a: a["F80"], reverse=True)
    sel = [a for i, a in enumerate(ordre) if i < 10 or a["arch"] in noms]
    _tableau(os.path.join(rep, "tab_banc_classement.tex"),
             f"Dix premières architectures à la fitness au seuil $0{{,}}80$ et finalistes provisoires ({resume})",
             "tab:banc-classement", "lrrrrrrrr",
             "\\textbf{Architecture} & $n$ & \\textbf{MCC} & \\textbf{perf} & $R^2_{0{,}80}$ & $R^2_{0{,}99}$ & "
             "\\textbf{Lat. (ms)} & \\textbf{Param.} & \\textbf{Fitness}",
             [f"{a['arch']} & {a['n']} & {fr(a['mcc'])} & {fr(a['perf'])} & {fr(a['r80'], 3)} & {fr(a['r99'], 3)} & "
              f"{fr(a['lat'], 1)} & {milliers(a['npar'])} & {fr(a['F80'])} $\\pm$ {fr(a['F80_sd'])}" for a in sel])
    _tableau(os.path.join(rep, "tab_banc_influence.tex"),
             f"Poids nominaux et influence effective des critères sur le classement ({resume})",
             "tab:banc-influence", "lrrr",
             "\\textbf{Critère} & \\textbf{Poids (\\%)} & \\textbf{Influence à $0{,}80$ (\\%)} & \\textbf{Influence à $0{,}99$ (\\%)}",
             [f"{l['critere'].replace('R2', '$R^2$')} & {fr(l['poids_pct'], 1)} & {fr(l['influence_pct'], 1)} & "
              f"{fr(next(m['influence_pct'] for m in infl if m['seuil'] == '0,99' and m['critere'] == l['critere']), 1)}"
              for l in infl if l["seuil"] == "0,80"])
    roles = {"fitness": "fitness", "detection": "détection", "symbolisation": "symbolisation",
             "rapidite": "rapidité", "B conservee": "conservée", "ajout TKAN traitee": "ajout (TKAN traitée)"}
    _tableau(os.path.join(rep, "tab_banc_finalistes.tex"),
             f"Application provisoire de la règle des finalistes ({resume})", "tab:banc-finalistes", "llrrrr",
             "\\textbf{Architecture} & \\textbf{Critère} & \\textbf{perf} & $R^2_{0{,}80}$ & \\textbf{Lat. (ms)} & \\textbf{Fitness}",
             [f"{f['arch']} & {', '.join(roles[r] for r in f['roles'])} & {fr(agg[f['arch']]['perf'])} & "
              f"{fr(agg[f['arch']]['r80'], 3)} & {fr(agg[f['arch']]['lat'], 1)} & {fr(agg[f['arch']]['F80'])}" for f in finals])
    det = [c for c in comp if c["cle"] == "perf" and c["n"] >= 2]
    if det:
        _tableau(os.path.join(rep, "tab_banc_apparies.tex"),
                 f"Différences appariées du score de détection perf (MCC, PR-AUC et Brier pondérés) sur les graines, test $t$ indicatif et correction de Holm ({resume})",
                 "tab:banc-apparies", "lrrrr",
                 "\\textbf{Comparaison} & $n$ & \\textbf{Écart} & $p$ & $p$ \\textbf{(Holm)}",
                 [f"{c['comparaison'].replace('−', '$-$')} & {c['n']} & {fr(c['moyenne']).replace('-', '$-$')} $\\pm$ {fr(c['ic'])} & "
                  f"{fp(c['p'], True)} & {fp(c['p_holm'], True)}" for c in det])


def ecrire_rapport(chemin, dossier, runs, agg, finals, regle, infl, corr, comp, effets, stab, aretes, corr_taille,
                   latence_complete, hors) -> None:
    ordre = sorted(agg.values(), key=lambda a: a["F80"], reverse=True)
    incomplets = [a for a in ordre if a["n"] != GRAINES_ATTENDUES]
    phases = defaultdict(int)
    for r in runs:
        phases[r["phase"]] += 1
    L = [f"# Analyse du banc d'essai : {os.path.basename(os.path.normpath(dossier))}", "",
         "Provisoire : validation seule, le jeu de test n'est jamais lu. Tableau généré par `campagne_figures.py`.", "",
         "## État des données", "",
         f"- Runs lus : {len(runs)} ; par phase : {dict(phases)} ; hors banc d'essai : {hors or 'aucun'}.",
         f"- Latences : {'complètes' if latence_complete else 'INCOMPLÈTES, fitness et V5 sans terme de latence'}.",
         f"- Architectures sans toutes leurs graines : {[(a['arch'], a['n']) for a in incomplets] or 'aucune'}.",
         f"- Threads par run (status.json) : {sorted({r['threads'] for r in runs if r['threads'] is not None})} ; "
         f"{sum(r['threads'] is None for r in runs)} runs sans status.json copié.",
         f"- Fenêtres de validation par cellule : "
         f"{ {NOM_CELLULE[c]: sorted({r['n_val'] for r in runs if r['cell'] == c}) for c in {r['cell'] for r in runs}} } : "
         "les cellules ne sont pas évaluées sur les mêmes fenêtres.", "",
         "## Classement à la fitness au seuil 0,80", "",
         "| Rang | Architecture | n | MCC | perf | R² 0,80 | R² 0,99 | Latence (ms) | Param. | Fitness | Rang 0,99 | Rang perf | Rang V5 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ordre:
        L.append(f"| {a['rang_F80']} | {a['arch']} | {a['n']} | {fr(a['mcc'])} ± {fr(a['mcc_sd'])} | {fr(a['perf'])} | "
                 f"{fr(a['r80'], 3)} | {fr(a['r99'], 3)} | {fr(a['lat'], 1)} | {milliers(a['npar'])} | "
                 f"{fr(a['F80'])} ± {fr(a['F80_sd'])} | {a['rang_F99']} | {a['rang_perf']} | {a['rang_V5_80']} |")
    L += ["", "## Règle préenregistrée", "",
          f"- Meilleure moyenne : {regle['best']['arch']}.",
          f"- Ex aequo : {[a['arch'] for a in regle['ex_aequo']]}.",
          f"- Retenue au nombre de paramètres puis à la latence : {regle['retenue']['arch']}.", "",
          "## Application provisoire de la règle des finalistes", ""]
    L += [f"- {f['arch']} : {', '.join(f['roles'])}." for f in finals]
    L += ["", "## Influence effective", "", "| Seuil | Critère | Poids (%) | Étendue | Influence (%) |", "|---|---|---|---|---|"]
    L += [f"| {l['seuil']} | {l['critere']} | {fr(l['poids_pct'], 1)} | {fr(l['etendue'])} | {fr(l['influence_pct'], 1)} |" for l in infl]
    L += ["", "## Corrélations entre classements", ""]
    L += [f"- {k} : Spearman {fr(v['spearman'], 3)}, Kendall {fr(v['kendall'], 3)}." for k, v in corr.items()]
    L += [f"- Paramètres et perf, {NOM_CELLULE[c]} : Spearman {fr(v, 3)}." for c, v in corr_taille.items()]
    L += ["", "## Différences appariées (test t indicatif, n petit)", "", "| Critère | Comparaison | n | Écart | IC 95 % | p | p Holm |", "|---|---|---|---|---|---|---|"]
    L += [f"| {c['critere']} | {c['comparaison']} | {c['n']} | {fr(c['moyenne'])} | ± {fr(c['ic'])} | {fp(c['p'])} | {fp(c['p_holm'])} |" for c in comp]
    L += ["", "## Effets marginaux par porte (régime brut)", "", "| Cellule | Porte | Base | n | MCC | perf | R² 0,80 | R² 0,99 | Latence |", "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {NOM_CELLULE[e['cellule']]} | {PORTES_AFFICHEES[e['cellule']][e['porte']]} | {NOM_BASE.get(e['base'], e['base'])} | {e['n']} | "
          f"{fr(e['mcc'])} | {fr(e['perf'])} | {fr(e['r80'], 3)} | {fr(e['r99'], 3)} | {fr(e.get('lat'), 1)} |" for e in effets]
    L += ["", "## Stabilité et arrêt", ""]
    L += [f"- {k} : {fr(v, 4) if isinstance(v, float) else v}." for k, v in stab.items()]
    if aretes:
        surv = [l["survivantes"] for l in aretes]
        L += ["", "## Arêtes évaluées par le R² symbolique", "",
              f"- Arêtes survivantes à l'élagage par run : de {min(surv)} à {max(surv)}, médiane {int(np.median(surv))} ; le R² n'en évalue que {N_ARETES_AUDIT}."]
        lin = [l for l in aretes if l["sortie"] == "linear"]
        if len(lin) > 2:
            for cle, nom in (("r99", "0,99"), ("r80", "0,80")):
                L.append(f"- TKAN à sortie linéaire, arêtes de la porte Output et R² à {nom} : Spearman "
                         f"{fr(float(stats.spearmanr([l['Output'] for l in lin], [l[cle] for l in lin])[0]), 3)}.")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(chemin)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("dossier", help="copie du dossier de resultats (lecture seule)")
    p.add_argument("--out", default=None, help="defaut : analyse_banc/<nom du dossier>, a cote du script")
    p.add_argument("--fig_dir", default=os.path.join(ROOT_DIR, "memoire", "figure"))
    p.add_argument("--tab_dir", default=os.path.join(ROOT_DIR, "memoire", "tableaux_banc"))
    p.add_argument("--no_png", action="store_true")
    args = p.parse_args()

    dossier = os.path.abspath(args.dossier)
    runs, latence_complete, hors = charger_runs(dossier)
    if not runs:
        raise SystemExit(f"aucun run fini du banc d'essai dans {dossier}")
    out = args.out or os.path.join(ETUDE_DIR, "analyse_banc", os.path.basename(dossier))
    os.makedirs(out, exist_ok=True)

    agg = agreger(runs, latence_complete)
    print(f"{len(runs)} runs, {len(agg)} architectures, latences "
          f"{'completes' if latence_complete else 'INCOMPLETES : fitness sans latence'}.")
    finals = finalistes_provisoires(agg, latence_complete)
    regle = regle_preenregistree(agg)
    infl = influence(agg, latence_complete)
    corr = correlations_rangs(agg)
    comp = comparaisons(agg, finals)
    effets = effets_bases(runs)
    aretes = aretes_par_porte(runs)
    stab = stabilite(runs, agg)
    corr_taille = {}
    for cell in ("lstm", "gru"):
        sel = [a for a in agg.values() if a["cell"] == cell and a["phase"] == "etape4_brut"]
        if len(sel) > 2:
            corr_taille[cell] = float(stats.spearmanr([a["npar"] for a in sel], [a["perf"] for a in sel])[0])

    ecrire_csv(os.path.join(out, "architectures.csv"),
               [{k: v for k, v in a.items() if k not in ("runs", "bases", "seeds")} | dict(seeds=a["seeds"], **a["bases"])
                for a in sorted(agg.values(), key=lambda a: a["F80"], reverse=True)])
    ecrire_csv(os.path.join(out, "runs_enrichis.csv"),
               [{k: v for k, v in r.items() if k not in ("rep", "bases", "train_hist", "mon_hist")} | r["bases"] for r in runs])
    ecrire_csv(os.path.join(out, "influence.csv"), infl)
    ecrire_csv(os.path.join(out, "effets_bases.csv"), effets)
    ecrire_csv(os.path.join(out, "interactions.csv"), interactions(runs, "perf") + interactions(runs, "r80"))
    ecrire_csv(os.path.join(out, "diff_appariees.csv"), [{k: v for k, v in c.items() if k != "d"} for c in comp])
    ecrire_csv(os.path.join(out, "aretes_par_porte.csv"), aretes)
    with open(os.path.join(out, "finalistes_provisoires.json"), "w", encoding="utf-8") as f:
        json.dump(dict(provisoire=True, latence_complete=latence_complete, finalistes=finals,
                       regle_preenregistree=dict(meilleure=regle["best"]["arch"],
                                                 ex_aequo=[a["arch"] for a in regle["ex_aequo"]],
                                                 retenue=regle["retenue"]["arch"])),
                  f, indent=1, ensure_ascii=False)
    ecrire_rapport(os.path.join(out, "ANALYSE_BANC.md"), dossier, runs, agg, finals, regle, infl, corr, comp, effets,
                   stab, aretes, corr_taille, latence_complete, hors)
    resume = (f"résultats partiels, {len(runs)} entraînements, validation"
              + ("" if latence_complete else ", latence non mesurée"))
    ecrire_tableaux(args.tab_dir, agg, finals, infl, comp, resume)

    sortie = Sortie(os.path.join(out, "figures"), args.fig_dir, not args.no_png)
    etat = _etat(runs, latence_complete)
    taches = [
        lambda: fig_1_points_graines(agg, finals, sortie, etat),
        lambda: fig_2_differences(comp, sortie, etat),
        lambda: fig_3_compromis(agg, finals, sortie, etat, latence_complete),
        lambda: fig_4_bases_par_porte(effets, sortie, etat, latence_complete),
        lambda: fig_5_rangs(agg, finals, sortie, etat, corr),
        lambda: fig_6_seuil_aretes(agg, finals, aretes, sortie, etat),
        lambda: fig_a1_interactions(runs, "perf", "détection (perf)", 4, sortie, etat),
        lambda: fig_a1_interactions(runs, "r80", "R² à 0,80", 2, sortie, etat),
        lambda: fig_a2_taille(agg, finals, corr_taille, sortie, etat, latence_complete),
        lambda: fig_a3_arrets(runs, sortie, etat),
        lambda: fig_a4_courbes(agg, finals, sortie, etat),
        lambda: fig_a5_profils(agg, finals, sortie, etat),
    ]
    echecs = 0
    for tache in taches:
        try:
            tache()
        except Exception:   # une figure en echec n'empeche pas les autres
            echecs += 1
            traceback.print_exc()
    print(f"Termine : {len(taches) - echecs} figures sur {len(taches)}. Rapport : {os.path.join(out, 'ANALYSE_BANC.md')}")


if __name__ == "__main__":
    main()
