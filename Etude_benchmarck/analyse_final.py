"""
analyse_final.py -- Analyse de l'entrainement final, des modeles de reference et des controles du
raccourci, reconstruite UNIQUEMENT a partir d'une copie du dossier de resultats (lecture seule).
Relancable tel quel sur chaque nouvelle copie : un tableau ou une figure n'apparait que si ses
donnees existent (raccourci_tous, par exemple, des qu'il est copie).

    python analyse_final.py campagne_aws/resultats_instance/final_2026-11-08
    python analyse_final.py <dossier> --etape4 campagne_aws/resultats_instance/etape4_2026-11-06
    python analyse_final.py <dossier> --no_png

Toutes les mesures portent sur la VALIDATION : le jeu de test n'est jamais lu.

Decisions appliquees (memoire/DECISIONS.md) :
  - perf = w_MCC*max(0,MCC) + w_PR*PR-AUC - w_Brier*Brier (mesure de detection, 06/11) ;
    fitness = perf + w_R2*R2_symb(0,80) - w_lat*P_lat (seuil de travail 0,80, 06/11) ;
  - A traite (TKAN traitee DoG/B-spl/DoG/Lin) retire de l'analyse principale (09/11) : ses runs
    sont lus, signales dans le rapport et places en annexe seulement ;
  - aucune architecture unique en tete : front de Pareto detection / R2 0,80 / latence (09/11) ;
  - modeles de reference XGBoost, MLP, LSTM (03/11), compares aux MKAN a regime ET fenetre W
    egaux seulement (la validation n'a pas le meme nombre de fenetres selon W) ;
  - pas de fitness pour les modeles de reference (pas de R2 symbolique) : detection et latence.

Sorties :
    analyse_banc/<nom du dossier>/ANALYSE_FINAL.md, *.csv, figures/*.html
    memoire/figure/final_*.png                  (sans titre : la legende LaTeX le porte)
    memoire/tableaux_banc/tab_final_*.tex
"""

import argparse
import glob
import json
import os
import traceback
from collections import defaultdict

import numpy as np
from scipy import stats

from campagne_figures import (ACCENT, ENCRE, ENCRE_2, GRAINES_ATTENDUES, GRIS, GRIS_CLAIR, LATENCE_T,
                              NOM_CELLULE, ROOT_DIR, SLOTS, ETUDE_DIR, Sortie, _flottant, _mise_en_page,
                              _tableau, charger_runs, difference_appariee, ecrire_csv, fp, fr, front_pareto,
                              holm, ic95, label_arch, milliers)
from niveau1_decision import (AHP_WEIGHT_BRIER, AHP_WEIGHT_LATENCY_PENALTY, AHP_WEIGHT_MCC,
                              AHP_WEIGHT_PR_AUC, AHP_WEIGHT_R2_SYMBOLIC)

PHASE_FINAL = "final"
PHASE_BASE = "exp_baselines"
PHASES_RACCOURCI = {"exp_raccourci_emetteur": "sans les soldes de l'émetteur",
                    "exp_raccourci_tous": "sans aucun solde"}
NOM_BASELINE = {"xgboost": "XGBoost", "mlp": "MLP", "lstm": "LSTM"}
# Decision du 09/11 : hors analyse principale
A_TRAITE = dict(cell="lstm", regime="traite",
                bases=dict(Forget="wavkan_dog", Input="efficientkan", Candidate="wavkan_dog", Output="linear"))


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def _lire_json(chemin):
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def est_a_traite(cell, regime, bases) -> bool:
    return cell == A_TRAITE["cell"] and regime == A_TRAITE["regime"] and bases == A_TRAITE["bases"]


def charger(dossier: str):
    runs = []
    for res_json in sorted(glob.glob(os.path.join(dossier, "runs", "*", "result.json"))):
        rep = os.path.dirname(res_json)
        res = _lire_json(res_json)
        spec, m = res["spec"], res["metrics"]
        lat_j = _lire_json(os.path.join(rep, "latency.json")) or {}
        st = _lire_json(os.path.join(rep, "status.json")) or {}
        mcc = _flottant(m.get("mcc")) or 0.0
        pr = _flottant(m.get("auc_pr")) or 0.0
        brier = _flottant(m.get("brier"))
        brier = 1.0 if brier is None else brier
        perf = AHP_WEIGHT_MCC * max(0.0, mcc) + AHP_WEIGHT_PR_AUC * pr - AHP_WEIGHT_BRIER * brier
        lat = _flottant(lat_j.get("latency_ms"))
        base = spec["cell_type"] == "baseline"
        if base:
            label = f"{NOM_BASELINE.get(spec['baseline'], spec['baseline'])} ({spec['regime'].replace('traite', 'traité')}, W={spec['W']})"
        else:
            label = label_arch(spec["cell_type"], spec["regime"], spec["base_by_gate"]) + f", W={spec['W']}"
        r80 = None if base else (_flottant(m.get("r2_symbolic_0.80")) or 0.0)
        r99 = None if base else (_flottant(m.get("r2_symbolic")) or 0.0)
        pen = None if lat is None else min(1.0, max(0.0, (lat - LATENCE_T) / LATENCE_T))
        F80 = None
        if not base and pen is not None:
            F80 = perf + AHP_WEIGHT_R2_SYMBOLIC * r80 - AHP_WEIGHT_LATENCY_PENALTY * pen
        hist = m.get("train_loss_history")
        runs.append(dict(
            run_id=spec["run_id"], phase=spec["phase"], base=base, modele=spec.get("baseline"),
            cell=spec["cell_type"], regime=spec["regime"], bases=spec["base_by_gate"], W=spec.get("W"),
            seed=spec["seed"], label=label, hidden=spec.get("hidden_size"), npar=m.get("n_params"),
            drop=tuple(spec.get("drop_cols") or ()),
            a_traite=(not base) and est_a_traite(spec["cell_type"], spec["regime"], spec["base_by_gate"]),
            mcc=mcc, pr=pr, brier=brier, perf=perf, r80=r80, r99=r99, lat=lat, pen=pen, F80=F80,
            lat_threads=lat_j.get("threads"), lat_machine=lat_j.get("machine"), lat_date=lat_j.get("measured"),
            threads=st.get("threads"), n_train=m.get("n_train_windows"), n_val=m.get("n_val_windows"),
            epochs=m.get("epochs_run"), best_epoch=m.get("best_epoch"), stopped=m.get("stopped_early"),
            max_epochs=spec.get("max_epochs"), patience=spec.get("patience"),
            n_hist=None if hist is None else len(hist), n_bad=m.get("n_bad_scores")))
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Controles d'integrite (directives du 09/11, point 1)
# ─────────────────────────────────────────────────────────────────────────────

def integrite(dossier: str, runs: list) -> list:
    L = []
    par_phase = defaultdict(int)
    for r in runs:
        par_phase[r["phase"]] += 1
    n_dirs = len(glob.glob(os.path.join(dossier, "runs", "*")))
    L.append(f"- Dossiers de runs : {n_dirs} ; avec result.json : {len(runs)} ; par phase : {dict(par_phase)}.")
    # concordance avec runs.csv
    csv_path = os.path.join(dossier, "runs.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, encoding="utf-8") as f:
            lignes = {l["run_id"]: l for l in csv.DictReader(f)}
        ecarts = []
        for r in runs:
            l = lignes.get(r["run_id"])
            if l is None:
                ecarts.append((r["run_id"], "absent de runs.csv"))
                continue
            for cle, col in (("mcc", "mcc"), ("pr", "auc_pr"), ("brier", "brier"), ("r80", "r2_symbolic_0.80"),
                             ("lat", "latency_ms")):
                v = _flottant(l.get(col))
                if r[cle] is None and v is None:
                    continue
                if r[cle] is None or v is None or abs(r[cle] - v) > 1e-9:
                    ecarts.append((r["run_id"], col))
        L.append(f"- Concordance runs.csv / result.json / latency.json : "
                 f"{'exacte' if not ecarts else f'{len(ecarts)} écarts : {ecarts[:10]}'}.")
    # arret et reprises
    entraines = [r for r in runs if r["epochs"] is not None]
    plafond = [r["label"] + f" s{r['seed']}" for r in entraines if r["epochs"] >= (r["max_epochs"] or 60)]
    incoherents = [r["run_id"] for r in entraines
                   if r["n_hist"] is not None and r["n_hist"] != r["epochs"]]
    patience = [r["run_id"] for r in entraines
                if r["stopped"] and r["patience"] and r["epochs"] - r["best_epoch"] != r["patience"]]
    L.append(f"- Runs arrêtés au plafond d'époques : {plafond or 'aucun'}.")
    L.append(f"- Historique de perte de longueur différente du nombre d'époques (reprise incohérente) : {incoherents or 'aucun'}.")
    L.append(f"- Arrêt anticipé sans écart de « patience » époques entre fin et meilleure époque : {patience or 'aucun'}.")
    L.append(f"- Époques (min-max) : {min(r['epochs'] for r in entraines)}-{max(r['epochs'] for r in entraines)} ; "
             f"meilleure époque {min(r['best_epoch'] for r in entraines)}-{max(r['best_epoch'] for r in entraines)}.")
    # conditions d'execution
    for cle, nom in (("threads", "Threads d'entraînement (status.json)"), ("n_train", "Fenêtres d'entraînement"),
                     ("lat_threads", "Threads de la mesure de latence"), ("lat_machine", "Machine de la mesure")):
        val = defaultdict(set)
        for r in runs:
            val[r["phase"]].add(r[cle])
        L.append(f"- {nom} par phase : { {k: sorted(map(str, v)) for k, v in val.items()} }.")
    dates = defaultdict(list)
    for r in runs:
        if r["lat_date"]:
            dates[r["phase"]].append(r["lat_date"])
    L.append(f"- Dates des mesures de latence : { {k: (min(v), max(v)) for k, v in dates.items()} }.")
    val = defaultdict(set)
    for r in runs:
        val[r["W"]].add(r["n_val"])
    L.append(f"- Fenêtres de validation par W : { {k: sorted(v) for k, v in val.items()} } : "
             "la population évaluée change avec W, on ne compare donc qu'à W égal.")
    L.append(f"- Scores invalides (n_bad_scores > 0) : {[r['run_id'] for r in runs if r['n_bad']] or 'aucun'}.")
    sch = os.path.join(dossier, "scheduler.log")
    if os.path.exists(sch):
        with open(sch, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        L.append(f"- scheduler.log : {txt.count('interrupt') + txt.count('Interrupt')} mentions d'interruption, "
                 f"{txt.count('reprise') + txt.count('resume')} de reprise.")
    return L


# ─────────────────────────────────────────────────────────────────────────────
# Agregation et comparaisons
# ─────────────────────────────────────────────────────────────────────────────

CLES = ("mcc", "pr", "brier", "perf", "r80", "r99", "lat", "F80", "epochs")


def agreger(runs: list) -> dict:
    g = defaultdict(list)
    for r in runs:
        g[(r["phase"], r["label"], r["drop"])].append(r)
    agg = {}
    for (phase, label, drop), rs in g.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        a = dict(phase=phase, arch=label, drop=drop, n=len(rs), runs=rs, base=rs[0]["base"], cell=rs[0]["cell"],
                 regime=rs[0]["regime"], W=rs[0]["W"], bases=rs[0]["bases"], modele=rs[0]["modele"],
                 npar=rs[0]["npar"], a_traite=rs[0]["a_traite"], seeds=[r["seed"] for r in rs])
        for k in CLES:
            v = [r[k] for r in rs]
            if any(x is None for x in v):
                a[k] = a[k + "_sd"] = a[k + "_min"] = a[k + "_max"] = a[k + "_ic"] = None
                continue
            v = np.array(v, dtype=float)
            a[k], a[k + "_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            a[k + "_min"], a[k + "_max"], a[k + "_ic"] = float(v.min()), float(v.max()), ic95(v)
        if a["perf"] is not None and a["r80"] is not None:
            a["terme_r2"] = AHP_WEIGHT_R2_SYMBOLIC * a["r80"]
            a["terme_lat"] = None if a["lat"] is None else -AHP_WEIGHT_LATENCY_PENALTY * float(
                np.mean([r["pen"] for r in rs]))
        agg[(phase, label, drop)] = a
    return agg


def finalistes(agg: dict, avec_a: bool = False) -> list:
    sel = [a for a in agg.values() if a["phase"] == PHASE_FINAL and (avec_a or not a["a_traite"])]
    return sorted(sel, key=lambda a: -(a["F80"] or 0))


def references(agg: dict) -> list:
    return sorted([a for a in agg.values() if a["phase"] == PHASE_BASE],
                  key=lambda a: (a["regime"] != "traite", a["W"], -a["perf"]))


def _paire(a, b, cle, groupe, nom):
    d = difference_appariee(a, b, cle)
    return dict(groupe=groupe, comparaison=nom, cle=cle, **{k: v for k, v in d.items() if k != "graines"},
                graines=d["graines"])


def comparaisons(agg: dict) -> list:
    """(1) MKAN contre references, a regime et W egaux ; (2) H2 dans le final (GRU, memes donnees) ;
    (3) controles du raccourci contre le meme finaliste complet. Holm par groupe."""
    fin, ref = finalistes(agg), references(agg)
    comp = []
    for m in fin:
        for b in ref:
            if b["regime"] == m["regime"] and b["W"] == m["W"]:
                comp.append(_paire(m, b, "perf", "references", f"{_court(m)} − {b['arch'].split(' (')[0]}"))
    gru = [m for m in fin if m["cell"] == "gru"]
    for i in range(len(gru)):
        for j in range(i + 1, len(gru)):
            for cle in ("perf", "r80", "F80"):
                comp.append(_paire(gru[i], gru[j], cle, "h2_final", f"{gru[i]['arch']} − {gru[j]['arch']}"))
    for a in agg.values():
        if a["phase"] in PHASES_RACCOURCI:
            pleins = [m for m in fin if m["arch"] == a["arch"]]
            if pleins:
                for cle in ("mcc", "perf"):
                    comp.append(_paire(a, pleins[0], cle, "raccourci",
                                       f"{a['arch']} {PHASES_RACCOURCI[a['phase']]} − complet"))
    for groupe in {c["groupe"] for c in comp}:
        for cle in {c["cle"] for c in comp if c["groupe"] == groupe}:
            sel = [c for c in comp if c["groupe"] == groupe and c["cle"] == cle]
            for c, ph in zip(sel, holm([c["p"] for c in sel])):
                c["p_holm"] = ph
    return comp


def effets_references(agg: dict) -> list:
    """Effet du pretraitement (traite - brut, W=10) et de la fenetre (W=5 - W=10, traite), appariees
    par graine. Le second change aussi la population de validation : a lire comme tel."""
    ref = {(a["modele"], a["regime"], a["W"]): a for a in references(agg)}
    out = []
    for mod in sorted({k[0] for k in ref}):
        for nom, x, y in (("prétraitement (traité − brut, W=10)", (mod, "traite", 10), (mod, "brut", 10)),
                          ("fenêtre (W=5 − W=10, traité)", (mod, "traite", 5), (mod, "traite", 10))):
            if x in ref and y in ref:
                for cle in ("mcc", "perf"):
                    d = difference_appariee(ref[x], ref[y], cle)
                    out.append(dict(modele=NOM_BASELINE.get(mod, mod), effet=nom, cle=cle, n=d["n"],
                                    moyenne=d["moyenne"], ic=d["ic"], p=d["p"]))
    return out


def pareto(fin: list) -> list:
    pts = [[a["perf"], a["r80"], a["lat"]] for a in fin]
    idx = set(front_pareto(pts, [1, 1, -1]))
    return [a["arch"] for i, a in enumerate(fin) if i in idx]


def etape4_contre_final(dossier_e4: str, fin: list) -> list:
    if not dossier_e4 or not os.path.isdir(dossier_e4):
        return []
    runs4, _, _ = charger_runs(dossier_e4)
    g = defaultdict(list)
    for r in runs4:
        g[(r["cell"], r["regime"], json.dumps(r["bases"], sort_keys=True))].append(r)
    out = []
    for a in fin:
        rs = g.get((a["cell"], a["regime"], json.dumps(a["bases"], sort_keys=True)))
        if not rs:
            continue
        m = lambda k: float(np.mean([r[k] for r in rs]))
        out.append(dict(arch=a["arch"], n4=len(rs), perf4=m("perf"), r80_4=m("r80"), F80_4=m("F80"),
                        lat4=m("lat") if all(r["lat"] is not None for r in rs) else None,
                        perf=a["perf"], r80=a["r80"], F80=a["F80"], lat=a["lat"]))
    if len(out) > 2:
        for cle in ("F80", "perf", "r80"):
            rho = float(stats.spearmanr([o[cle + ("4" if cle == "perf" else "_4")] for o in out],
                                        [o[cle] for o in out])[0])
            for o in out:
                o["spearman_" + cle] = rho
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _court(a) -> str:
    return a["arch"].replace(" (traité)", " traitée").replace("Uniforme GRU Hyb/Hyb/Hyb", "GRU uniforme hybride")


def fig_final_graines(fin, sortie, etat):
    """Chaque finaliste, chaque graine : detection, R2 a 0,80, fitness."""
    from plotly.subplots import make_subplots
    crit = [("perf", "Détection (score perf)"), ("r80", "R² symbolique à 0,80"), ("F80", "Fitness à 0,80")]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[c[1] for c in crit])
    y = [_court(a) for a in fin]
    for col, (cle, _) in enumerate(crit, start=1):
        xs, ys = [], []
        for a in fin:
            xs += [r[cle] for r in a["runs"]]
            ys += [_court(a)] * a["n"]
        fig.add_scatter(x=xs, y=ys, mode="markers", name="Une graine", legendgroup="g", showlegend=col == 1,
                        marker=dict(color=GRIS_CLAIR, size=8), row=1, col=col,
                        hovertemplate="%{y}<br>graine : %{x:.4f}<extra></extra>")
        fig.add_scatter(x=[a[cle] for a in fin], y=y, mode="markers", name="Moyenne et intervalle à 95 %",
                        legendgroup="m", showlegend=col == 1,
                        error_x=dict(type="data", array=[a[cle + "_ic"] for a in fin], color=ACCENT, thickness=1.5, width=0),
                        marker=dict(color=ACCENT, size=11, line=dict(color="#ffffff", width=2)), row=1, col=col,
                        hovertemplate="%{y}<br>moyenne : %{x:.4f}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=y, autorange="reversed")
    _mise_en_page(fig, 190 + 48 * len(fin), 1200, legend=dict(orientation="h", y=-0.22))
    sortie.enregistrer(fig, "final_01_graines", "Entraînement final : résultats par finaliste et par graine",
                       f"Tri par fitness à 0,80. {etat}")


def fig_final_decomposition(fin, sortie, etat):
    """D'ou vient l'ecart de fitness : chaque terme, en ecart a la moyenne des finalistes."""
    import plotly.graph_objects as go
    termes = [("perf", "Détection (perf)", SLOTS[0]), ("terme_r2", "Symbolisation (w × R² à 0,80)", SLOTS[1]),
              ("terme_lat", "Pénalité de latence", SLOTS[2])]
    fig = go.Figure()
    y = [_court(a) for a in fin]
    for cle, nom, c in termes:
        v = [a.get(cle) or 0.0 for a in fin]
        if not any(abs(x) > 1e-12 for x in v):   # terme nul partout (latences sous le seuil) : signale dans la legende LaTeX
            continue
        moy = float(np.mean(v))
        fig.add_bar(y=y, x=[x - moy for x in v], orientation="h", name=nom, marker=dict(color=c),
                    customdata=v, hovertemplate="%{y}<br>" + nom + " : %{customdata:.4f}<br>écart : %{x:+.4f}<extra></extra>")
    moyF = float(np.mean([a["F80"] for a in fin]))
    fig.add_scatter(y=y, x=[a["F80"] - moyF for a in fin], mode="markers", name="Écart de fitness (somme)",
                    marker=dict(symbol="line-ns", size=22, color=ENCRE, line=dict(width=3, color=ENCRE)),
                    hovertemplate="%{y}<br>écart de fitness : %{x:+.4f}<extra></extra>")
    fig.add_vline(x=0, line=dict(color=ENCRE_2, width=1))
    fig.update_yaxes(categoryorder="array", categoryarray=y, autorange="reversed")
    fig.update_xaxes(title_text="Écart à la moyenne des finalistes")
    _mise_en_page(fig, 200 + 55 * len(fin), 1000, barmode="group", bargap=0.25, bargroupgap=0.08,
                  legend=dict(orientation="h", y=-0.25))
    sortie.enregistrer(fig, "final_02_decomposition", "Décomposition de la fitness des finalistes",
                       f"Chaque terme en écart à la moyenne ; leur somme donne l'écart de fitness. {etat}")


def fig_final_pareto(fin, ref, front, sortie, etat):
    """Compromis : detection contre R2 (finalistes) ; detection contre latence (finalistes et references)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("Détection et symbolisation", "Détection et latence (échelle log)"))
    # Etiquettes en annotations decalees (les noms sont longs et les GRU proches les unes des autres)
    decal = {1: [(0, -34), (60, -30), (-60, 34), (-40, -34), (0, 34)],
             2: [(-30, -70), (20, 48), (-110, 34), (-80, -34), (0, 34)]}
    for i, a in enumerate(fin):
        sur = a["arch"] in front
        style = dict(symbol="diamond" if a["cell"] == "gru" else "circle", size=13,
                     color=ACCENT if sur else GRIS, line=dict(color="#ffffff", width=2))
        for col, x in ((1, a["r80"]), (2, a["lat"])):
            fig.add_scatter(x=[x], y=[a["perf"]], mode="markers", marker=style, showlegend=False, row=1, col=col,
                            hovertemplate=_court(a) + "<br>x : %{x:.3f}<br>perf : %{y:.4f}<extra></extra>")
            ax, ay = decal[col][i % len(decal[col])]
            fig.add_annotation(x=np.log10(x) if col == 2 else x, y=a["perf"], text=_court(a), ax=ax, ay=ay,
                               row=1, col=col, font=dict(size=10, color=ENCRE), arrowcolor=GRIS, arrowwidth=1,
                               arrowhead=0, bgcolor="rgba(255,255,255,0.85)")
    symb = {5: "square", 10: "square-open"}
    for b in ref:
        if b["regime"] != "traite":
            continue
        fig.add_scatter(x=[b["lat"]], y=[b["perf"]], mode="markers+text", text=[b["arch"]],
                        textposition="middle right", textfont=dict(size=9, color=ENCRE_2),
                        marker=dict(symbol=symb.get(b["W"], "square"), size=10, color=ENCRE_2), showlegend=False,
                        row=1, col=2, hovertemplate=b["arch"] + "<br>latence : %{x:.2f} ms<br>perf : %{y:.4f}<extra></extra>")
    legende = [("Finaliste sur le front de Pareto (3 critères)", "diamond", ACCENT)]
    if any(a["arch"] not in front for a in fin):
        legende.append(("Finaliste dominé", "diamond", GRIS))
    legende += [("Référence traitée, W=5", "square", ENCRE_2), ("Référence traitée, W=10", "square-open", ENCRE_2)]
    for nom, s, c in legende:
        fig.add_scatter(x=[None], y=[None], mode="markers", name=nom, marker=dict(symbol=s, color=c, size=10))
    r2 = [a["r80"] for a in fin]
    lats = [a["lat"] for a in fin] + [b["lat"] for b in ref if b["regime"] == "traite"]
    fig.update_xaxes(title_text="R² symbolique à 0,80", range=[min(r2) - 0.08, max(r2) + 0.08], row=1, col=1)
    fig.update_xaxes(title_text="Latence, lot de 256 fenêtres (ms)", type="log",
                     range=[np.log10(min(lats)) - 0.15, np.log10(max(lats)) + 0.25], row=1, col=2)
    fig.update_yaxes(title_text="Détection (score perf)", row=1, col=1)
    _mise_en_page(fig, 560, 1250, legend=dict(orientation="h", y=-0.2))
    sortie.enregistrer(fig, "final_03_pareto", "Compromis entre détection, symbolisation et latence",
                       f"Les références n'ont pas de R² symbolique ; comparer à W égal. {etat}")


def fig_final_references(comp, sortie, etat):
    """Ecarts apparies MKAN - reference, a regime et W egaux."""
    import plotly.graph_objects as go
    sel = [c for c in comp if c["groupe"] == "references" and c["n"] >= 2]
    if not sel:
        return
    y = [c["comparaison"] for c in sel]
    fig = go.Figure()
    xs, ys = [], []
    for c in sel:
        xs += c["d"]
        ys += [c["comparaison"]] * len(c["d"])
    fig.add_scatter(x=xs, y=ys, mode="markers", name="Écart sur une graine", marker=dict(color=GRIS_CLAIR, size=7),
                    hovertemplate="%{y}<br>écart : %{x:.4f}<extra></extra>")
    fig.add_scatter(x=[c["moyenne"] for c in sel], y=y, mode="markers", name="Écart moyen et intervalle à 95 %",
                    error_x=dict(type="data", array=[c["ic"] for c in sel], color=ACCENT, thickness=1.5, width=0),
                    marker=dict(color=ACCENT, size=10, line=dict(color="#ffffff", width=2)),
                    customdata=[[c["p"], c["p_holm"]] for c in sel],
                    hovertemplate="%{y}<br>écart : %{x:.4f}<br>p : %{customdata[0]:.3f}<br>p (Holm) : %{customdata[1]:.3f}<extra></extra>")
    fig.add_vline(x=0, line=dict(color=ENCRE_2, width=1))
    fig.update_yaxes(categoryorder="array", categoryarray=y, autorange="reversed")
    fig.update_xaxes(title_text="Écart de perf (MKAN − référence) ; positif : la MKAN détecte mieux")
    _mise_en_page(fig, 160 + 34 * len(sel), 1100, legend=dict(orientation="h", y=-0.12))
    sortie.enregistrer(fig, "final_04_references", "MKAN finalistes contre modèles de référence",
                       f"Différences appariées par graine, même régime et même fenêtre. {etat}")


def fig_final_raccourci(agg, fin, sortie, etat):
    """Retirer les soldes change-t-il la detection ? Graines du finaliste complet et des variantes."""
    import plotly.graph_objects as go
    variantes = [a for a in agg.values() if a["phase"] in PHASES_RACCOURCI]
    if not variantes:
        return
    fig = go.Figure()
    lignes = []
    for v in variantes:
        plein = next((m for m in fin if m["arch"] == v["arch"]), None)
        for a, nom in ((plein, "complet"), (v, PHASES_RACCOURCI[v["phase"]])):
            if a is None:
                continue
            lab = f"{_court(a)}, {nom}"
            if lab in lignes:
                continue
            lignes.append(lab)
            fig.add_scatter(x=[r["mcc"] for r in a["runs"]], y=[lab] * a["n"], mode="markers", showlegend=False,
                            marker=dict(color=GRIS_CLAIR, size=8), hovertemplate="%{y}<br>MCC : %{x:.4f}<extra></extra>")
            fig.add_scatter(x=[a["mcc"]], y=[lab], mode="markers", showlegend=False,
                            error_x=dict(type="data", array=[a["mcc_ic"]], color=ACCENT, thickness=1.5, width=0),
                            marker=dict(color=ACCENT, size=11, line=dict(color="#ffffff", width=2)),
                            hovertemplate="%{y}<br>MCC moyen : %{x:.4f}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=lignes, autorange="reversed")
    fig.update_xaxes(title_text="MCC sur la validation")
    _mise_en_page(fig, 160 + 50 * len(lignes), 950)
    sortie.enregistrer(fig, "final_05_raccourci", "Contrôle du raccourci par les soldes",
                       f"Points gris : graines ; bleu : moyenne et intervalle à 95 %. {etat}")


# ─────────────────────────────────────────────────────────────────────────────
# Tableaux et rapport
# ─────────────────────────────────────────────────────────────────────────────

def _m(x, d=4):
    return fr(x, d).replace("-", "$-$")


def ecrire_tableaux(rep, fin, ref, comp, eff, resume):
    os.makedirs(rep, exist_ok=True)
    _tableau(os.path.join(rep, "tab_final_finalistes.tex"),
             f"Entraînement final des finalistes : moyenne $\\pm$ écart-type sur les graines ({resume})",
             "tab:final-finalistes", "lrrrrrrrr",
             "\\textbf{Architecture} & $W$ & \\textbf{MCC} & \\textbf{PR-AUC} & \\textbf{Brier} & \\textbf{perf} & "
             "$R^2_{0{,}80}$ & \\textbf{Lat. (ms)} & \\textbf{Fitness}",
             [f"{_court(a)} & {a['W']} & {fr(a['mcc'])} $\\pm$ {fr(a['mcc_sd'])} & {fr(a['pr'])} & {fr(a['brier'])} & "
              f"{fr(a['perf'])} $\\pm$ {fr(a['perf_sd'])} & {fr(a['r80'], 3)} & {fr(a['lat'], 1)} & {fr(a['F80'])}"
              for a in fin])
    _tableau(os.path.join(rep, "tab_final_decomposition.tex"),
             f"Décomposition de la fitness des finalistes ({resume})", "tab:final-decomposition", "lrrrr",
             "\\textbf{Architecture} & \\textbf{perf} & $w_{R^2}\\,R^2_{0{,}80}$ & \\textbf{Pénalité de latence} & \\textbf{Fitness}",
             [f"{_court(a)} & {fr(a['perf'])} & {fr(a['terme_r2'])} & {_m(a['terme_lat'])} & {fr(a['F80'])}" for a in fin])
    _tableau(os.path.join(rep, "tab_final_references.tex"),
             f"Modèles de référence : moyenne sur les graines ({resume}) ; sans $R^2$ symbolique, donc sans fitness",
             "tab:final-references", "lllrrrrr",
             "\\textbf{Modèle} & \\textbf{Régime} & $W$ & \\textbf{MCC} & \\textbf{PR-AUC} & \\textbf{Brier} & \\textbf{perf} & \\textbf{Lat. (ms)}",
             [f"{NOM_BASELINE[b['modele']]} & {b['regime'].replace('traite', 'traité')} & {b['W']} & {fr(b['mcc'])} & "
              f"{fr(b['pr'])} & {fr(b['brier'])} & {fr(b['perf'])} $\\pm$ {fr(b['perf_sd'])} & {fr(b['lat'], 2)}" for b in ref])
    sel = [c for c in comp if c["groupe"] == "references" and c["n"] >= 2]
    if sel:
        _tableau(os.path.join(rep, "tab_final_comparaisons.tex"),
                 f"Différences appariées du score perf entre finalistes et modèles de référence, à régime et fenêtre égaux, test $t$ et correction de Holm ({resume})",
                 "tab:final-comparaisons", "lrrrr",
                 "\\textbf{Comparaison} & $n$ & \\textbf{Écart} & $p$ & $p$ \\textbf{(Holm)}",
                 [f"{c['comparaison'].replace('−', '$-$').replace(' (traité)', ' traitée')} & {c['n']} & "
                  f"{_m(c['moyenne'])} $\\pm$ {fr(c['ic'])} & {fp(c['p'], True)} & {fp(c['p_holm'], True)}" for c in sel])
    rac = [c for c in comp if c["groupe"] == "raccourci"]
    if rac:
        _tableau(os.path.join(rep, "tab_final_raccourci.tex"),
                 f"Contrôle du raccourci par les soldes : écart apparié à la version complète ({resume})",
                 "tab:final-raccourci", "llrrr",
                 "\\textbf{Variante} & \\textbf{Mesure} & $n$ & \\textbf{Écart} & $p$",
                 [f"{c['comparaison'].replace('−', '$-$').replace(' (traité)', ' traitée')} & "
                  f"{'MCC' if c['cle'] == 'mcc' else 'perf'} & {c['n']} & {_m(c['moyenne'], 5)} $\\pm$ {fr(c['ic'], 5)} & {fp(c['p'], True)}"
                  for c in rac])


def ecrire_rapport(chemin, dossier, runs, agg, fin, fin_a, ref, front, comp, eff, e4, integ, raccourci_donnees):
    L = [f"# Analyse de l'entraînement final : {os.path.basename(os.path.normpath(dossier))}", "",
         "Validation seule, le jeu de test n'est jamais lu. Généré par `analyse_final.py`.", "",
         "## Contrôles d'intégrité", ""] + integ
    L += ["", "## Finalistes (A traité exclu, décision du 09/11)", "",
          "| Architecture | n | MCC | PR-AUC | Brier | perf (min-max) | R² 0,80 | R² 0,99 | Latence (ms) | Fitness | Époques | Param. |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in fin_a:
        nom = a["arch"] + (" **(hors analyse)**" if a["a_traite"] else "")
        L.append(f"| {nom} | {a['n']} | {fr(a['mcc'])} ± {fr(a['mcc_sd'])} | {fr(a['pr'])} | {fr(a['brier'])} | "
                 f"{fr(a['perf'])} ± {fr(a['perf_sd'])} ({fr(a['perf_min'])}-{fr(a['perf_max'])}) | {fr(a['r80'], 3)} | "
                 f"{fr(a['r99'], 3)} | {fr(a['lat'], 1)} | {fr(a['F80'])} ± {fr(a['F80_sd'])} | {fr(a['epochs'], 1)} | {milliers(a['npar'])} |")
    L += ["", "## Décomposition de la fitness", "", "| Architecture | perf | w·R² 0,80 | pénalité latence | Fitness |", "|---|---|---|---|---|"]
    L += [f"| {a['arch']} | {fr(a['perf'])} | {fr(a['terme_r2'])} | {fr(a['terme_lat'])} | {fr(a['F80'])} |" for a in fin]
    L += ["", f"Poids : w_R2 = {fr(AHP_WEIGHT_R2_SYMBOLIC)}, w_lat = {fr(AHP_WEIGHT_LATENCY_PENALTY)}.", "",
          "## Front de Pareto (perf, R² 0,80, latence)", "", f"- Non dominés : {front}.",
          f"- Dominés : {[a['arch'] for a in fin if a['arch'] not in front] or 'aucun'}."]
    L += ["", "## Modèles de référence", "", "| Modèle | n | MCC | PR-AUC | Brier | perf | Latence (ms) |", "|---|---|---|---|---|---|---|"]
    L += [f"| {b['arch']} | {b['n']} | {fr(b['mcc'])} ± {fr(b['mcc_sd'])} | {fr(b['pr'])} | {fr(b['brier'])} | "
          f"{fr(b['perf'])} ± {fr(b['perf_sd'])} | {fr(b['lat'], 2)} |" for b in ref]
    L += ["", "## Effets lisibles sur les références (appariés par graine)", "",
          "| Modèle | Effet | Mesure | n | Écart | IC 95 % | p |", "|---|---|---|---|---|---|---|"]
    L += [f"| {e['modele']} | {e['effet']} | {e['cle']} | {e['n']} | {fr(e['moyenne'])} | ± {fr(e['ic'])} | {fp(e['p'])} |" for e in eff]
    L += ["", "L'effet de fenêtre compare aussi deux populations de validation différentes.", "",
          "## Différences appariées", "", "| Groupe | Mesure | Comparaison | n | Écart | IC 95 % | p | p Holm (groupe, mesure) |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {c['groupe']} | {c['cle']} | {c['comparaison']} | {c['n']} | {fr(c['moyenne'], 5)} | ± {fr(c['ic'], 5)} | "
          f"{fp(c['p'])} | {fp(c.get('p_holm', float('nan')))} |" for c in comp]
    if raccourci_donnees:
        t = raccourci_donnees.get("train", {})
        L += ["", "## Raccourci dans les données (raccourci.json, entraînement)", "",
              f"- Tolérance : {t.get('tolerance')} ; rapprochement des soldes de l'émetteur : fraudes "
              f"{fr(100 * t['fraude']['taux'], 1)} %, légitimes {fr(100 * t['legitime']['taux'], 1)} %."]
        L += [f"- {k} : {fr(100 * v['taux'], 1)} % (n = {milliers(v['n'])})."
              for k, v in t.get("par_scenario", {}).items() if v.get("taux") is not None]
    if e4:
        L += ["", "## Étape 4 (500 000 fenêtres) contre final (1 000 000)", "",
              "| Architecture | perf É4 | perf final | R² É4 | R² final | Fitness É4 | Fitness final | Lat. É4 (c8g) | Lat. final (m8g) |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {o['arch']} | {fr(o['perf4'])} | {fr(o['perf'])} | {fr(o['r80_4'], 3)} | {fr(o['r80'], 3)} | "
              f"{fr(o['F80_4'])} | {fr(o['F80'])} | {fr(o['lat4'], 1)} | {fr(o['lat'], 1)} |" for o in e4]
        if "spearman_F80" in e4[0]:
            L.append(f"\nSpearman des rangs É4/final : fitness {fr(e4[0]['spearman_F80'], 3)}, perf "
                     f"{fr(e4[0]['spearman_perf'], 3)}, R² {fr(e4[0]['spearman_r80'], 3)} (n = {len(e4)}, indicatif).")
        L.append("Latences : machines différentes (c8g à l'Étape 4, m8g au final), non comparables entre phases.")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(chemin)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("dossier", help="copie du dossier de resultats du final (lecture seule)")
    p.add_argument("--etape4", default=os.path.join(ETUDE_DIR, "campagne_aws", "resultats_instance", "etape4_2026-11-06"))
    p.add_argument("--out", default=None, help="defaut : analyse_banc/<nom du dossier>")
    p.add_argument("--fig_dir", default=os.path.join(ROOT_DIR, "memoire", "figure"))
    p.add_argument("--tab_dir", default=os.path.join(ROOT_DIR, "memoire", "tableaux_banc"))
    p.add_argument("--no_png", action="store_true")
    args = p.parse_args()

    dossier = os.path.abspath(args.dossier)
    runs = charger(dossier)
    if not runs:
        raise SystemExit(f"aucun run fini dans {dossier}")
    out = args.out or os.path.join(ETUDE_DIR, "analyse_banc", os.path.basename(dossier))
    os.makedirs(out, exist_ok=True)

    agg = agreger(runs)
    fin, fin_a, ref = finalistes(agg), finalistes(agg, avec_a=True), references(agg)
    incomplets = [(a["arch"], a["phase"], a["n"]) for a in agg.values() if a["n"] != GRAINES_ATTENDUES]
    front = pareto(fin)
    comp = comparaisons(agg)
    eff = effets_references(agg)
    e4 = etape4_contre_final(args.etape4, fin)
    integ = integrite(dossier, runs)
    if incomplets:
        integ.append(f"- Groupes sans toutes leurs graines : {incomplets}.")

    ecrire_csv(os.path.join(out, "final_groupes.csv"),
               [{k: v for k, v in a.items() if k not in ("runs", "bases")} | {"bases": json.dumps(a["bases"])}
                for a in agg.values()])
    ecrire_csv(os.path.join(out, "final_runs.csv"), [{k: v for k, v in r.items() if k != "bases"} for r in runs])
    ecrire_csv(os.path.join(out, "final_comparaisons.csv"), [{k: v for k, v in c.items() if k != "d"} for c in comp])
    ecrire_csv(os.path.join(out, "final_effets_references.csv"), eff)
    ecrire_csv(os.path.join(out, "final_etape4_contre_final.csv"), e4)
    ecrire_rapport(os.path.join(out, "ANALYSE_FINAL.md"), dossier, runs, agg, fin, fin_a, ref, front, comp, eff, e4,
                   integ, _lire_json(os.path.join(dossier, "raccourci.json")))
    resume = f"résultats partiels, validation, {len(runs)} entraînements"
    ecrire_tableaux(args.tab_dir, fin, ref, comp, eff, resume)

    sortie = Sortie(os.path.join(out, "figures"), args.fig_dir, not args.no_png)
    etat = f"Validation seule, {len(runs)} entraînements."
    taches = [lambda: fig_final_graines(fin, sortie, etat),
              lambda: fig_final_decomposition(fin, sortie, etat),
              lambda: fig_final_pareto(fin, ref, front, sortie, etat),
              lambda: fig_final_references(comp, sortie, etat),
              lambda: fig_final_raccourci(agg, fin, sortie, etat)]
    echecs = 0
    for t in taches:
        try:
            t()
        except Exception:
            echecs += 1
            traceback.print_exc()
    print(f"Termine : {len(taches) - echecs} figures sur {len(taches)}. Rapport : {os.path.join(out, 'ANALYSE_FINAL.md')}")


if __name__ == "__main__":
    main()
