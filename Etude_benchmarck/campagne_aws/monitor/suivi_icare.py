"""
monitor/suivi_icare.py -- Etat detaille du projet Icare pour la page de suivi (tourne sur l'instance Icare).

Toutes les --every secondes, ecrit <icare>/resultats/statut.json puis le copie directement sur S3
(s3://<bucket>/<prefixe>/sorties/statut.json), ou monitor/app.py le relit pour la carte « Icare » :
  * charge de l'instance : CPU par coeur, memoire, disque, processus de calcul actifs ;
  * commande en cours de la chaine (essai reduit ou execution complete) et barre de progression de l'etape
    en cours (derniere ligne de progression.Barre dans chaine.log) ;
  * avancement de chaque etape (python run.py --statut, toutes les --statut_every secondes) ;
  * dernieres unites terminees et debit sur la derniere heure (journaux du magasin).
Lecture seule : ne touche ni au calcul ni au magasin.

Usage (sur l'instance Icare) :
    python suivi_icare.py --icare ~/Icare --s3 s3://mkan-training-mko/icare/sorties/statut.json
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import psutil

# Deux formats de progression.Barre : « ... 12.3 %  5 min écoulées, reste ~1 h  info » (code du 22/10)
# et « ... 12.3 % | 5 min, reste ~1 h | info » (code du 04/11)
BARRE = re.compile(r"^(?P<etape>\S+)\s+\[[#.]+\]\s+(?P<faites>\d+)/(?P<total>\d+)\s+(?P<pct>[\d.]+) %\s*(?:\|\s*)?"
                   r"(?P<temps>.*?)(?:\s*\|\s*|\s{2,})(?P<info>\S.*?)?\s*$")
STATUT = re.compile(r"^(?P<etape>\S+)\s+:\s+(?P<faites>\d+)\s+/\s+(?P<total>\d+) unit")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def systeme(icare):
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(icare)
    calc = []
    for p in psutil.process_iter(["pid", "cmdline", "cpu_percent"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if "multiprocessing" in cmd and "spawn_main" in cmd:
            calc.append(p.info["cpu_percent"] or 0.0)
    return {"cpu_per_core": psutil.cpu_percent(percpu=True), "cpu_total": psutil.cpu_percent(),
            "n_cores": psutil.cpu_count(), "load_avg": os.getloadavg(),
            "mem_used_gb": round((vm.total - vm.available) / 2**30, 1), "mem_total_gb": round(vm.total / 2**30, 1),
            "disk_used_gb": round(du.used / 2**30, 1), "disk_total_gb": round(du.total / 2**30, 1),
            "n_processus": len(calc), "n_processus_actifs": sum(c > 50 for c in calc)}


def commandes():
    """Arguments des commandes run.py en cours (essai reduit et execution complete peuvent tourner ensemble)."""
    out = []
    for p in psutil.process_iter(["cmdline"]):
        c = p.info["cmdline"] or []
        if len(c) >= 2 and "python" in os.path.basename(c[0]) and c[1].endswith("run.py"):
            out.append(" ".join(c[2:]) or "toutes les étapes")
    return out


def barre(log):
    """Derniere ligne de progression et dernieres lignes de controle (« === ... ») de chaine.log."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 20000))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return None, []
    lignes = [l.strip() for l in re.split(r"[\r\n]", txt) if l.strip()]
    b = None
    for l in reversed(lignes):
        m = BARRE.match(l)
        if m:
            b = {k: v for k, v in m.groupdict().items()}
            b["faites"], b["total"], b["pct"] = int(b["faites"]), int(b["total"]), float(b["pct"])
            break
    controle = [l for l in lignes if l.startswith(("===", "ERREUR", "  ERREUR")) or " ERREUR " in l][-6:]
    return b, controle


def statut(icare, rapide):
    cmd = [sys.executable, "run.py", "--statut"] + (["--rapide"] if rapide else [])
    r = subprocess.run(cmd, cwd=icare, capture_output=True, text=True, timeout=600,
                       env={**os.environ, "PYTHONUTF8": "1"})
    out = []
    for l in r.stdout.splitlines():
        m = STATUT.match(l.strip())
        if m:
            out.append({"etape": m["etape"], "faites": int(m["faites"]), "total": int(m["total"])})
        elif " : non évaluable" in l:
            out.append({"etape": l.split(":")[0].strip(), "faites": None, "total": None})
    nc = [l for l in r.stdout.splitlines() if l.startswith("paramètres non calibrés")]
    return {"etapes": out, "non_calibres": nc[0].split(":", 1)[1].strip() if nc else None}


def recentes(icare, n=12):
    lignes = []
    for f in glob.glob(os.path.join(icare, "resultats", "magasin*", "_journal", "*.jsonl")):
        rapide = "magasin_rapide" in f
        try:
            with open(f, encoding="utf-8") as fh:
                for l in fh:
                    try:
                        d = json.loads(l)
                    except json.JSONDecodeError:
                        continue
                    d["rapide"] = rapide
                    lignes.append(d)
        except OSError:
            pass
    lignes.sort(key=lambda d: d.get("t", ""))
    limite = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    return lignes[-n:][::-1], sum(d.get("t", "") >= limite for d in lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icare", default=os.path.expanduser("~/Icare"))
    ap.add_argument("--s3", default="s3://mkan-training-mko/icare/sorties/statut.json")
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--statut_every", type=float, default=300.0)
    a = ap.parse_args()
    # journaux suivis : (source, fichier) ; le calibrage complet lance a part ecrit dans calibrage.log
    logs = [("chaîne", os.path.join(a.icare, "resultats", "chaine.log")),
            ("calibrage complet", os.path.join(a.icare, "resultats", "calibrage.log"))]
    dest = os.path.join(a.icare, "resultats", "statut.json")
    psutil.cpu_percent(percpu=True)
    for p in psutil.process_iter():
        try:
            p.cpu_percent()
        except psutil.Error:
            pass
    st_complet, st_rapide, t_statut = None, None, 0.0
    while True:
        cmds = commandes()
        if time.time() - t_statut > a.statut_every:
            try:
                st_complet, st_rapide = statut(a.icare, False), statut(a.icare, True)
            except (subprocess.SubprocessError, OSError):
                pass
            t_statut = time.time()
        barres, controle = [], []
        for source, log in logs:
            b, c = barre(log)
            if b:
                barres.append({"source": source, "maj": os.path.getmtime(log), **b})
            controle += c
        barres.sort(key=lambda b: -b["maj"])
        complet = [b for b in barres if b["source"] == "calibrage complet" or "--rapide" not in " ".join(cmds)]
        rapide = bool(cmds) and all("--rapide" in c for c in cmds)
        rec, debit = recentes(a.icare)
        etat = {"updated": _now(), "commande": " | ".join(cmds) or None, "commandes": cmds, "rapide": rapide,
                "systeme": systeme(a.icare), "barres": barres,
                "barre": (complet or barres or [None])[0], "controle": controle[-8:],
                "etapes_complet": st_complet, "etapes_rapide": st_rapide,
                "recentes": rec, "unites_derniere_heure": debit,
                # champs lus par l'ancienne carte (texte brut)
                "statut": "\n".join(f"{e['etape']:15s} : {e['faites']} / {e['total']}"
                                    for e in (st_complet or {}).get("etapes", []) if e["total"] is not None),
                "journal": "\n".join(controle)}
        tmp = dest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(etat, f, ensure_ascii=False)
        os.replace(tmp, dest)
        subprocess.run(["aws", "s3", "cp", dest, a.s3, "--only-show-errors"], timeout=60)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
