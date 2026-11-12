"""
monitor/suivi_nera.py -- Etat du projet NERA pour la page de suivi (tourne sur l'instance NERA).

Toutes les --every secondes, ecrit statut.json, cost_state.json et sentinel_state.json sous le prefixe S3
--s3 (par defaut s3://mkan-training-mko/nera/sorties/), ou monitor/app.py les relit pour la carte « NERA » :
  * charge de l'instance : CPU par coeur, memoire, disque, processus de calcul NERA actifs ;
  * avancement de la chaine des lits, lu dans la sortie de lancer_tout.sh / lancer_suite.sh (--journal) :
    chaque etape s'annonce par « ── HH:MM <nom> » et finit par « terminé (resultats/journaux/<nom>.log) » ou
    « ÉCHEC : voir resultats/journaux/<nom>.log » ;
  * lignes de resume du triage (--triage : sortie de lancer_triage.py, lignes « ══════ », bancs, selection) ;
  * cout : prix horaire (--prix, $/h) multiplie par le temps depuis le demarrage de l'instance.
Ne publie AUCUNE donnee de patient : seulement des noms d'etapes, des compteurs et des lignes de resume.
Lecture seule : ne touche ni au calcul ni aux resultats.

Usage (sur l'instance NERA) :
    python suivi_nera.py --journal ~/prod_lit.log --triage ~/prod_triage.log --prix 0.31 --attendus 62
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime, timezone

import psutil

DEBUT = re.compile(r"^── (?P<h>\d\d:\d\d) (?P<nom>[^\s.]\S*)$")            # « ── 05:31 niveaux_PED_minuit »
FIN = re.compile(r"terminé \(resultats/journaux/(?P<nom>\S+)\.log\)")
ECHEC = re.compile(r"ÉCHEC : voir resultats/journaux/(?P<nom>\S+)\.log")
FINI = re.compile(r"^── \d\d:\d\d fini\.")
RESUME_TRIAGE = re.compile(r"^\s*(══════|T\d+\b|modèle principal|►|Raccordement)")
SCRIPTS = ("lancer_niveaux", "benchmark_", "sensibilite", "visualisations", "modele_survie", "exporter_resultats",
           "transfert", "lien_triage_lits", "selection", "lancer_triage", "traitement.", "verifier_calculs")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lignes(chemin, octets=400_000):
    """Dernieres lignes d'un fichier texte (fin de fichier seulement : les journaux peuvent etre longs)."""
    try:
        with open(chemin, "rb") as f:
            f.seek(max(0, os.path.getsize(chemin) - octets))
            return [l.rstrip() for l in f.read().decode("utf-8", "replace").splitlines()]
    except OSError:
        return []


def systeme(racine):
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(racine if os.path.exists(racine) else "/")
    calc = []
    for p in psutil.process_iter(["cmdline", "cpu_percent"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if "python" in cmd and any(s in cmd for s in SCRIPTS):
            calc.append(p.info["cpu_percent"] or 0.0)
    return {"cpu_per_core": psutil.cpu_percent(percpu=True), "cpu_total": psutil.cpu_percent(),
            "n_cores": psutil.cpu_count(), "load_avg": os.getloadavg(),
            "mem_used_gb": round((vm.total - vm.available) / 2**30, 1), "mem_total_gb": round(vm.total / 2**30, 1),
            "disk_used_gb": round(du.used / 2**30, 1), "disk_total_gb": round(du.total / 2**30, 1),
            "n_processus": len(calc), "n_processus_actifs": sum(c > 50 for c in calc)}


def etapes(journaux):
    """Etapes lancees, finies, en echec et en cours, d'apres la sortie des lanceurs (plusieurs fichiers possibles)."""
    debut, fin, echec, evenements, fini = {}, set(), set(), [], False
    for chemin in journaux:
        for l in _lignes(chemin):
            s = l.strip()
            if m := DEBUT.match(s):
                debut[m["nom"]] = m["h"]
                evenements.append({"h": m["h"], "nom": m["nom"], "etat": "lancée"})
            elif m := FIN.search(s):
                fin.add(m["nom"]); echec.discard(m["nom"])
                evenements.append({"h": "", "nom": m["nom"], "etat": "terminée"})
            elif m := ECHEC.search(s):
                echec.add(m["nom"]); fin.discard(m["nom"])
                evenements.append({"h": "", "nom": m["nom"], "etat": "échec"})
            elif FINI.match(s):
                fini = True
    en_cours = [n for n in debut if n not in fin and n not in echec]
    return {"lancees": len(debut), "faites": len(fin), "echecs": sorted(echec), "en_cours": en_cours,
            "fini": fini and not en_cours, "evenements": evenements[-12:][::-1]}


def metadonnees():
    """Type et cycle de vie de l'instance (IMDSv2) ; '?' hors d'AWS."""
    def lire(chemin, jeton):
        req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{chemin}",
                                     headers={"X-aws-ec2-metadata-token": jeton})
        return urllib.request.urlopen(req, timeout=2).read().decode()
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
        jeton = urllib.request.urlopen(req, timeout=2).read().decode()
        return {"instance_type": lire("instance-type", jeton), "lifecycle": lire("instance-life-cycle", jeton),
                "az": lire("placement/availability-zone", jeton)}
    except OSError:
        return {"instance_type": "?", "lifecycle": "?", "az": "?"}


def publier(dossier_s3, nom, contenu):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(contenu, f, ensure_ascii=False)
    try:
        subprocess.run(["aws", "s3", "cp", f.name, f"{dossier_s3.rstrip('/')}/{nom}", "--only-show-errors"], timeout=60)
    finally:
        os.unlink(f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nera", default=os.path.expanduser("~/nera/nera-modele"), help="dossier du depot nera-modele")
    ap.add_argument("--journal", nargs="+", default=[os.path.expanduser("~/prod_lit.log")],
                    help="sortie(s) de lancer_tout.sh ou lancer_suite.sh")
    ap.add_argument("--triage", default=os.path.expanduser("~/prod_triage.log"), help="sortie de lancer_triage.py")
    ap.add_argument("--s3", default="s3://mkan-training-mko/nera/sorties/")
    ap.add_argument("--prix", type=float, default=None, help="prix de l'instance en $/h (cout = prix x heures)")
    ap.add_argument("--attendus", type=int, default=None, help="nombre d'etapes attendues (barre d'avancement)")
    ap.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args()
    meta = metadonnees()
    psutil.cpu_percent(percpu=True)
    for p in psutil.process_iter():
        try:
            p.cpu_percent()
        except psutil.Error:
            pass
    while True:
        e = etapes(a.journal)
        triage = [l.strip() for l in _lignes(a.triage, 60_000) if RESUME_TRIAGE.match(l)][-8:]
        heures = (time.time() - psutil.boot_time()) / 3600
        pct = round(100 * e["faites"] / a.attendus, 2) if a.attendus else None
        etat = {"type": "nera", "updated": _now(), "systeme": systeme(a.nera), "etapes": e, "attendus": a.attendus,
                "avancement_pct": pct, "triage": triage,
                "statut": f"lits : {e['faites']} étapes finies, {len(e['en_cours'])} en cours, {len(e['echecs'])} en échec"}
        publier(a.s3, "statut.json", etat)
        publier(a.s3, "cost_state.json", {"price_usd_h": a.prix, "hours": round(heures, 3),
                                          "total_usd": round(a.prix * heures, 2) if a.prix else None})
        publier(a.s3, "sentinel_state.json", {**meta, "updated": _now()})
        time.sleep(a.every)


if __name__ == "__main__":
    main()
