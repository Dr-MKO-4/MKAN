"""
fidelite_formule.py -- Fidelite des formules extraites au reseau, sans reentrainement (validation uniquement).

Audit du 12/11/2026 (AUDIT_TRAVAUX_2026-11-12.md, point 2.2). Le R2 symbolique de la campagne ne juge que les 40 aretes
de plus forte norme L1 et ne dit pas si le modele symbolise decide comme le reseau. Ce script mesure, pour un modele
finaliste entraine :

  A. Elagage : nombre d'aretes actives et detection quand on ne garde que les aretes de norme L1 superieure a un
     seuil, ou seulement les K aretes les plus fortes. Repond a « combien d'aretes faut-il pour decider pareil ? ».
  B. Symbolisation de TOUTES les aretes actives (L1 > 0,01), pas de 40 : part des aretes dont le meilleur ajustement
     c*f(a*x+b)+d atteint R2 >= 0,80 et 0,99 (meme catalogue et meme ajustement que _compute_r2_symbolic).
  C. Fidelite : les aretes sont REMPLACEES par leur formule dans le reseau, puis le modele est reevalue.
       - « symbolisables remplacees » : aretes a R2 >= seuil remplacees par leur formule, les autres gardees apprises ;
       - « formule seule »            : aretes a R2 >= seuil remplacees, les autres supprimees ;
       - « tout remplace »            : toutes les aretes actives remplacees par leur meilleure formule.
     Mesures : accord des decisions avec le reseau complet (seuil 0,5), R2 entre les scores, MCC face aux etiquettes.

La formule est ajustee sur [-2, 2] (convention de _compute_r2_symbolic) et appliquee telle quelle au-dela.

Usage (depuis MKAN/Etude_benchmarck) :  python fidelite_formule.py [--n-eval 20000] [--threads 8]
Sorties : analyse_banc/final_2026-11-08/fidelite_formule.csv, fidelite_aretes.csv, ANALYSE_FIDELITE.md,
          memoire/tableaux_banc/tab_fidelite.tex, tab_fidelite_aretes.tex
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.abspath(os.path.join(ICI, "..", ".."))
for d in (ICI, os.path.join(ICI, ".."), os.path.join(ICI, "campagne_aws")):
    if d not in sys.path:
        sys.path.insert(0, d)
import extended_heuristic_search as ehs  # noqa: E402
from niveau1_harness import (read_momtsim_parquet, build_regime_frame, standardize, prepare_window_arrays,  # noqa: E402
                             sample_windows_from_arrays, compute_metrics)
from niveau1_benchmark_run import _build_x_pool  # noqa: E402
from train_run import build_model  # noqa: E402

VAL_PATH = os.path.join(RACINE, "data", "val_features.parquet")
PORTES = ("forget_gate", "input_gate", "candidate_gate", "output_gate", "reset_gate", "update_gate")
NOMS_PORTES = {"forget_gate": "Forget", "input_gate": "Input", "candidate_gate": "Candidate", "output_gate": "Output",
               "reset_gate": "reset", "update_gate": "update"}
THETA = 0.01
DOMAINE = 2.0
# graine de score perf median de chaque finaliste (catalogue du tableau de bord)
DEFAUT = [("GRU traite F=dog I=dog C=dog", 1), ("GRU traite F=dog I=relu C=dog", 1),
          ("GRU traite F=hyb I=hyb C=hyb", 4), ("TKAN brut F=relu I=bspl C=dog O=lin", 5)]
JOLI = {"GRU traite F=dog I=dog C=dog": "GRUKANCell DoG/DoG/DoG, traité",
        "GRU traite F=dog I=relu C=dog": "GRUKANCell DoG/ReLU-KAN/DoG, traité",
        "GRU traite F=hyb I=hyb C=hyb": "GRUKANCell hybride, traité",
        "TKAN brut F=relu I=bspl C=dog O=lin": "TKANCell ReLU-KAN/B-spline/DoG/linéaire, brut"}


def fr(x, nd=4):
    return "n.d." if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}".replace(".", ",")


class Intervention:
    """Remplace edge_activations de chaque porte : masque d'elagage puis substitution par les formules."""

    def __init__(self, model):
        self.couches = {n: getattr(model.cell, n) for n in PORTES if getattr(model.cell, n, None) is not None}
        self.orig = {n: c.edge_activations for n, c in self.couches.items()}
        self.masque, self.subst, self.dans_domaine = {}, {}, False
        for n, c in self.couches.items():
            c.edge_activations = self._fabrique(n)

    def _fabrique(self, nom):
        def f(x):
            e = self.orig[nom](x)
            m = self.masque.get(nom)
            if m is not None:
                e = e * m
            for fn, I, J, a, b, c, d in self.subst.get(nom, []):
                v = c * fn(a * x[:, I] + b) + d
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                if self.dans_domaine:          # formule sur son domaine d'ajustement, arete apprise au-dela
                    v = torch.where(x[:, I].abs() <= DOMAINE, v, e[:, I, J])
                e = e.clone()
                e[:, I, J] = v
            return e
        return f

    def regler(self, masque=None, subst=None, dans_domaine=False):
        self.masque, self.subst, self.dans_domaine = masque or {}, subst or {}, dans_domaine

    def brut(self, nom, x):
        return self.orig[nom](x)


@torch.no_grad()
def scores(model, X, lot=2048):
    out = [model(torch.from_numpy(np.ascontiguousarray(X[s:s + lot]))).numpy() for s in range(0, len(X), lot)]
    return np.nan_to_num(np.concatenate(out), nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def ajuster(courbes):
    """Meilleure famille c*f(a*x+b)+d par arete : meme catalogue, 150 pas, lr 0,05 que _compute_r2_symbolic."""
    n = courbes.shape[0]
    x = torch.linspace(-DOMAINE, DOMAINE, courbes.shape[1]).unsqueeze(0)
    r2_best = torch.full((n,), -float("inf"))
    fam = np.array([""] * n, dtype=object)
    P = torch.zeros(n, 4)
    Adam = getattr(ehs, "DMLAdam", torch.optim.Adam)
    with torch.enable_grad():
        for nom, fn in ehs.SYMBOLIC_LIBRARY.items():
            a = torch.nn.Parameter(torch.ones(n, 1)); b = torch.nn.Parameter(torch.zeros(n, 1))
            c = torch.nn.Parameter(torch.ones(n, 1)); d = torch.nn.Parameter(torch.zeros(n, 1))
            opt = Adam([a, b, c, d], lr=0.05)
            div = torch.zeros(n, dtype=torch.bool)
            for _ in range(150):
                opt.zero_grad(set_to_none=True)
                pred = c * fn(a * x + b) + d
                div = div | torch.isnan(pred).any(dim=1) | torch.isinf(pred).any(dim=1)
                perte = (pred - courbes).square().mean(dim=1)
                torch.where(div, torch.zeros_like(perte), perte).sum().backward()
                opt.step()
            with torch.no_grad():
                pred = c * fn(a * x + b) + d
                ss_res = (courbes - pred).square().sum(dim=1)
                ss_tot = (courbes - courbes.mean(dim=1, keepdim=True)).square().sum(dim=1)
                r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                r2 = torch.where(div | ~torch.isfinite(r2), torch.full_like(r2, -float("inf")), r2)
                mieux = r2 > r2_best
                r2_best = torch.where(mieux, r2, r2_best)
                fam[mieux.numpy()] = nom
                P[mieux] = torch.cat([a, b, c, d], dim=1).detach()[mieux]
    return fam, P, r2_best


def mesurer(nom, n_aretes, s, s_ref, y):
    p, p_ref = s >= 0.5, s_ref >= 0.5
    m = compute_metrics(y, s)
    ss = float(((s - s_ref) ** 2).sum()); st = float(((s_ref - s_ref.mean()) ** 2).sum())
    return {"variante": nom, "aretes_actives": int(n_aretes), "accord_decisions": float((p == p_ref).mean()),
            "r2_scores": 1.0 - ss / (st + 1e-12), "mcc": m["mcc"], "auc_pr": m["auc_pr"]}


def traiter(run_dir, n_eval, val_cache):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    spec, pre = ck["spec"], ck["preprocessing"]
    cols, W, regime = pre["feature_cols"], int(spec["W"]), spec["regime"]
    model = build_model(spec, ck["input_size"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cle = (regime, W)
    if cle not in val_cache:
        df, cols_h = build_regime_frame(read_momtsim_parquet(VAL_PATH), regime, "B", {})
        assert list(cols_h) == list(cols), (cols_h, cols)
        standardize(df, cols, stats={c: (v["mean"], v["std"]) for c, v in pre["standardize_stats"].items()})
        arr = prepare_window_arrays(df, cols)
        del df
        val_cache.clear()
        val_cache[cle] = sample_windows_from_arrays(arr, W, None, n_eval, 43)[:2]
    X, y = val_cache[cle]
    y = np.asarray(y).astype(int)
    inter = Intervention(model)
    s_ref = scores(model, X)
    with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
        mcc_ref = json.load(f)["metrics"]["mcc"]
    lignes = [mesurer("réseau complet", 0, s_ref, s_ref, y)]
    print(f"  MCC sur {len(y)} fenetres : {lignes[0]['mcc']:.4f} (toute la validation : {mcc_ref:.4f})", flush=True)

    # normes L1 sur les entrees reelles des portes
    x_pool = _build_x_pool(model, spec["cell_type"], torch.from_numpy(np.ascontiguousarray(X[:2048])), W, n_audit=2048)
    with torch.no_grad():
        l1 = {n: inter.brut(n, x_pool).abs().mean(dim=0) for n in inter.couches}
    total = sum(int(v.numel()) for v in l1.values())
    lignes[0]["aretes_actives"] = total
    toutes = torch.cat([v.reshape(-1) for v in l1.values()])
    hors = float((x_pool[:, spec["hidden_size"]:].abs() > DOMAINE).float().mean())
    lignes[0]["part_entrees_hors_domaine"] = hors

    # A. elagage
    for theta in (0.01, 0.05, 0.1, 0.2):
        masque = {n: (v > theta).float() for n, v in l1.items()}
        inter.regler(masque=masque)
        lignes.append(mesurer(f"élagage, L1 > {fr(theta, 2)}", sum(int(m.sum()) for m in masque.values()),
                              scores(model, X), s_ref, y))
    for K in (1000, 300, 100, 40):
        if K >= total:
            continue
        seuil = float(torch.topk(toutes, K).values[-1])
        masque = {n: (v >= seuil).float() for n, v in l1.items()}
        inter.regler(masque=masque)
        lignes.append(mesurer(f"{K} arêtes les plus fortes", sum(int(m.sum()) for m in masque.values()),
                              scores(model, X), s_ref, y))

    # B. symbolisation de toutes les aretes actives
    grille = torch.linspace(-DOMAINE, DOMAINE, 100)
    fits, aretes = {}, []
    h = spec["hidden_size"]
    for n, couche in inter.couches.items():
        ij = (l1[n] > THETA).nonzero(as_tuple=False)
        if len(ij) == 0:
            continue
        courbes = torch.empty(len(ij), 100)
        with torch.no_grad():
            for i in ij[:, 0].unique().tolist():
                x = torch.zeros(100, couche.in_features)
                x[:, i] = grille
                sortie = inter.brut(n, x)[:, i, :]
                sel = ij[:, 0] == i
                courbes[sel] = sortie[:, ij[sel, 1]].T
        fam, P, r2 = [], [], []
        for s in range(0, len(ij), 4000):
            f_, p_, r_ = ajuster(courbes[s:s + 4000])
            fam.append(f_); P.append(p_); r2.append(r_)
        fam, P, r2 = np.concatenate(fam), torch.cat(P), torch.cat(r2)
        fits[n] = (ij, fam, P, r2)
        for k in range(len(ij)):
            i, j = int(ij[k, 0]), int(ij[k, 1])
            aretes.append({"porte": NOMS_PORTES[n], "entree": cols[i - h] if i >= h else f"h[{i}]", "noeud": j,
                           "l1": float(l1[n][i, j]), "famille": fam[k], "r2": float(r2[k]),
                           "a": float(P[k, 0]), "b": float(P[k, 1]), "c": float(P[k, 2]), "d": float(P[k, 3])})
        print(f"  {NOMS_PORTES[n]} : {len(ij)} aretes ajustees", flush=True)
    A = pd.DataFrame(aretes)

    # C. fidelite
    def substitutions(seuil):
        sub, n_sub = {}, 0
        for n, (ij, fam, P, r2) in fits.items():
            garde = (r2 >= seuil).numpy()
            n_sub += int(garde.sum())
            sub[n] = []
            for nom_f, fn in ehs.SYMBOLIC_LIBRARY.items():
                sel = torch.from_numpy(garde & (fam == nom_f))
                if sel.any():
                    sub[n].append((fn, ij[sel, 0], ij[sel, 1], P[sel, 0], P[sel, 1], P[sel, 2], P[sel, 3]))
        return sub, n_sub

    actif = {n: (v > THETA).float() for n, v in l1.items()}
    n_actif = sum(int(m.sum()) for m in actif.values())
    for seuil in (0.80, 0.99):
        sub, n_sub = substitutions(seuil)
        inter.regler(masque=actif, subst=sub)
        l = mesurer(f"arêtes à R² ≥ {fr(seuil, 2)} remplacées par leur formule, les autres apprises", n_actif,
                    scores(model, X), s_ref, y)
        l["aretes_symboliques"] = n_sub
        lignes.append(l)
        inter.regler(masque=actif, subst=sub, dans_domaine=True)
        l = mesurer(f"arêtes à R² ≥ {fr(seuil, 2)} remplacées sur [-2, 2] seulement, apprises au-delà", n_actif,
                    scores(model, X), s_ref, y)
        l["aretes_symboliques"] = n_sub
        lignes.append(l)
        seul = {}
        for n, (ij, fam, P, r2) in fits.items():
            m = torch.zeros_like(l1[n])
            g = r2 >= seuil
            m[ij[g, 0], ij[g, 1]] = 1.0
            seul[n] = m
        for n in actif:
            seul.setdefault(n, torch.zeros_like(l1[n]))
        inter.regler(masque=seul, subst=sub)
        l = mesurer(f"formule seule : arêtes à R² ≥ {fr(seuil, 2)}, les autres supprimées", n_sub,
                    scores(model, X), s_ref, y)
        l["aretes_symboliques"] = n_sub
        lignes.append(l)
    sub, n_sub = substitutions(-float("inf"))
    inter.regler(masque=actif, subst=sub)
    l = mesurer("toutes les arêtes actives remplacées par leur meilleure formule", n_actif, scores(model, X), s_ref, y)
    l["aretes_symboliques"] = n_sub
    lignes.append(l)
    inter.regler()
    return pd.DataFrame(lignes), A, total


def main():
    ap = argparse.ArgumentParser()
    base = os.path.join(ICI, "campagne_aws", "resultats_instance", "final_2026-11-08")
    ap.add_argument("--final", default=base)
    ap.add_argument("--n-eval", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(ICI, "analyse_banc", "final_2026-11-08"))
    ap.add_argument("--tex", default=os.path.join(RACINE, "memoire", "tableaux_banc"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    runs = pd.read_csv(os.path.join(a.final, "runs.csv"))
    runs = runs[runs.phase == "final"]
    chemin, chemin_a = os.path.join(a.out, "fidelite_formule.csv"), os.path.join(a.out, "fidelite_aretes.csv")
    F = pd.read_csv(chemin) if os.path.exists(chemin) else pd.DataFrame()
    AR = pd.read_csv(chemin_a) if os.path.exists(chemin_a) else pd.DataFrame()
    cache = {}
    # regime traite d'abord puis brut : une seule preparation des fenetres par (regime, W)
    for label, seed in DEFAUT:
        if len(F) and (F.modele == label).any():
            continue
        r = runs[(runs.label == f"{label} s{seed}")]
        if r.empty:
            print("absent :", label, seed)
            continue
        print(f"[{label} s{seed}]", flush=True)
        f, ar, total = traiter(os.path.join(a.final, "runs", r.run_id.iloc[0]), a.n_eval, cache)
        f.insert(0, "modele", label); f.insert(1, "seed", seed); f["aretes_total"] = total
        ar.insert(0, "modele", label)
        F, AR = pd.concat([F, f], ignore_index=True), pd.concat([AR, ar], ignore_index=True)
        F.to_csv(chemin, index=False); AR.to_csv(chemin_a, index=False)

    md = ["# Fidélité des formules au réseau", "",
          f"Validation uniquement, {a.n_eval} fenêtres tirées avec une graine fixe, une graine par finaliste "
          "(celle de score perf médian). Accord : part des fenêtres où la décision au seuil 0,5 est celle du réseau complet.", ""]
    tex, tex_a = [], []
    for label, g in F.groupby("modele", sort=False):
        ar = AR[AR.modele == label]
        n_act = len(ar)
        p80, p99 = float((ar.r2 >= 0.80).mean()), float((ar.r2 >= 0.99).mean())
        top40 = ar.nlargest(40, "l1")
        md += [f"## {JOLI.get(label, label)}", "",
               f"Arêtes : {int(g.aretes_total.iloc[0])} au total, {n_act} actives (L1 > 0,01). Part des arêtes actives "
               f"symbolisables : {fr(100 * p80, 1)} % au seuil 0,80 et {fr(100 * p99, 1)} % au seuil 0,99 "
               f"(sur les 40 plus fortes : {fr(100 * float((top40.r2 >= 0.80).mean()), 1)} % et "
               f"{fr(100 * float((top40.r2 >= 0.99).mean()), 1)} %).", "",
               "| Variante | Arêtes actives | Accord des décisions | R² des scores | MCC |", "|---|---|---|---|---|"]
        for _, l in g.iterrows():
            md.append(f"| {l.variante} | {int(l.aretes_actives)} | {fr(l.accord_decisions)} | {fr(l.r2_scores, 3)} | {fr(l.mcc)} |")
            tex.append(" & ".join([JOLI.get(label, label) if l.variante == "réseau complet" else "",
                                   l.variante.replace("≥", "$\\geq$").replace("²", "$^2$"), str(int(l.aretes_actives)),
                                   fr(l.accord_decisions), fr(l.r2_scores, 3), fr(l.mcc)]) + " \\\\")
        tex.append("\\midrule")
        tex_a.append(" & ".join([JOLI.get(label, label), str(int(g.aretes_total.iloc[0])), str(n_act),
                                 fr(100 * p80, 1), fr(100 * p99, 1),
                                 fr(100 * float((top40.r2 >= 0.80).mean()), 1)]) + " \\\\")
        ex = ar[~ar.entree.str.startswith("h[")].nlargest(6, "l1")
        md += ["", "Six arêtes les plus fortes partant d'une variable d'entrée :", "",
               "| Porte | Variable | L1 | Meilleure formule | R² |", "|---|---|---|---|---|"]
        for _, e in ex.iterrows():
            md.append(f"| {e.porte} | {e.entree} | {fr(e.l1, 3)} | {fr(e.c, 3)} · {e.famille}({fr(e.a, 3)}·x + {fr(e.b, 3)}) + {fr(e.d, 3)} | {fr(e.r2, 3)} |")
        md.append("")
    with open(os.path.join(a.out, "ANALYSE_FIDELITE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    if tex and tex[-1] == "\\midrule":
        tex.pop()
    with open(os.path.join(a.tex, "tab_fidelite.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\scriptsize\n"
                f"\\caption{{Fidélité au réseau complet après élagage et après remplacement des arêtes par leur formule : "
                f"accord des décisions au seuil $0{{,}}5$, $R^2$ entre les scores et MCC, sur {a.n_eval} fenêtres de "
                "validation, une graine par finaliste}\n\\label{tab:fidelite}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
                "\\textbf{Modèle} & \\textbf{Variante} & \\textbf{Arêtes} & \\textbf{Accord} & $R^2$ \\textbf{scores} & "
                "\\textbf{MCC} \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    with open(os.path.join(a.tex, "tab_fidelite_aretes.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
                "\\caption{Part des arêtes actives (norme L1 supérieure à $0{,}01$) dont le meilleur ajustement symbolique "
                "atteint le seuil, sur toutes les arêtes actives et sur les $40$ plus fortes (validation, une graine par "
                "finaliste)}\n\\label{tab:fidelite-aretes}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
                "\\textbf{Modèle} & \\textbf{Arêtes} & \\textbf{Actives} & $R^2 \\geq 0{,}80$ (\\%) & $R^2 \\geq 0{,}99$ (\\%) & "
                "\\textbf{40 plus fortes,} $0{,}80$ (\\%) \\\\\n\\midrule\n" + "\n".join(tex_a)
                + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
