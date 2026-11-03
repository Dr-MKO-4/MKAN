"""
sentinel.py -- Garde de l'instance pendant la campagne, a lancer a cote de
scheduler.py (launch.sh le fait dans tmux). Trois boucles :

1. Copie vers S3 toutes les --sync_every secondes de tout OUT_DIR (resultats,
   rapports, points de reprise, etats) : rien n'est perdu si l'instance
   disparait, et une nouvelle instance reprend la ou l'ancienne s'est arretee
   (`python sentinel.py --restore` avant de relancer scheduler.py).
2. Preavis d'interruption spot (metadonnees de l'instance, IMDSv2, 2 minutes de
   preavis) : cree le fichier STOP (chaque run sauvegarde et quitte), attend la
   fin des runs, puis copie immediatement vers S3.
3. Garde-fou de cout : cumule heures x prix (prix spot reel de la zone, ou
   MKAN_PRICE_USD_H) plus le disque, dans cost_state.json (copie sur S3, donc
   cumule d'une instance a l'autre). A --stop_fraction du budget : STOP,
   attente des sauvegardes, copie S3, puis arret de la machine.

Droits IAM requis sur l'instance (role d'instance) : s3:ListBucket, GetObject,
PutObject sur le bucket ; ec2:DescribeSpotPriceHistory (sinon, prix fixe
MKAN_PRICE_USD_H).

Usage :
    python sentinel.py --restore          # une fois, sur une instance neuve
    python sentinel.py                    # en continu pendant la campagne
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CAMPAGNE_DIR not in sys.path:
    sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

IMDS = "http://169.254.169.254/latest"
S3_URI = f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/sorties"
STATE_PATH = os.path.join(config.OUT_DIR, "sentinel_state.json")
COST_PATH = os.path.join(config.OUT_DIR, "cost_state.json")
EBS_USD_PER_GB_MONTH = 0.0836      # gp3 eu-north-1 (a verifier sur la page de prix AWS)

_lock = threading.Lock()
_state = {"last_sync": None, "last_sync_ok": None, "last_sync_s": None, "spot_notice": None,
          "lifecycle": None, "instance_type": None, "az": None, "events": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[sentinelle {_now()[11:19]}] {msg}", flush=True)
    with _lock:
        _state["events"] = (_state["events"] + [f"{_now()} {msg}"])[-50:]


def _save_state() -> None:
    with _lock:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_state, f, indent=1)
        os.replace(tmp, STATE_PATH)


def _imds(path: str, timeout: float = 2.0):
    """Lecture des metadonnees de l'instance (IMDSv2). None hors EC2 ou si absent (404)."""
    try:
        req = urllib.request.Request(f"{IMDS}/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
        token = urllib.request.urlopen(req, timeout=timeout).read().decode()
        req = urllib.request.Request(f"{IMDS}/{path}",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()
    except Exception:   # noqa: BLE001  (404 = pas d'evenement, ou machine hors EC2)
        return None


def sync_up() -> bool:
    t0 = time.time()
    r = subprocess.run(["aws", "s3", "sync", config.OUT_DIR, S3_URI, "--exclude", "*.tmp",
                        "--exclude", "*.tmp.npy", "--only-show-errors"],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    with _lock:
        _state.update(last_sync=_now(), last_sync_ok=ok, last_sync_s=round(time.time() - t0, 1))
    if not ok:
        _log(f"copie S3 en echec : {r.stderr.strip()[:300]}")
    _save_state()
    return ok


def restore() -> None:
    os.makedirs(config.OUT_DIR, exist_ok=True)
    r = subprocess.run(["aws", "s3", "sync", S3_URI, config.OUT_DIR, "--only-show-errors"])
    if os.path.exists(config.STOP_FILE):
        os.remove(config.STOP_FILE)     # un STOP de l'ancienne instance ne doit pas bloquer
    print(f"Restauration depuis {S3_URI} : {'ok' if r.returncode == 0 else 'ECHEC'}")


def _scheduler_alive() -> bool:
    try:
        with open(os.path.join(config.OUT_DIR, "scheduler_state.json"), encoding="utf-8") as f:
            pid = json.load(f)["pid"]
        os.kill(pid, 0)
        return True
    except Exception:   # noqa: BLE001
        return False


def request_stop(reason: str, wait_s: float) -> None:
    """Cree STOP, attend que le scheduler et ses runs aient sauvegarde, copie S3."""
    if not os.path.exists(config.STOP_FILE):
        with open(config.STOP_FILE, "w") as f:
            f.write(f"{_now()} {reason}\n")
        _log(f"STOP demande : {reason}")
    t0 = time.time()
    while _scheduler_alive() and time.time() - t0 < wait_s:
        time.sleep(2)
    sync_up()


def spot_loop(stop_evt: threading.Event) -> None:
    while not stop_evt.is_set():
        action = _imds("meta-data/spot/instance-action")
        if action:
            with _lock:
                _state["spot_notice"] = action
            _log(f"preavis d'interruption spot : {action}")
            request_stop("preavis d'interruption spot", wait_s=90)
            stop_evt.set()
            return
        stop_evt.wait(5)


def _spot_price(instance_type: str, az: str):
    r = subprocess.run(["aws", "ec2", "describe-spot-price-history", "--instance-types",
                        instance_type, "--availability-zone", az, "--product-descriptions",
                        "Linux/UNIX", "--max-items", "1", "--query",
                        "SpotPriceHistory[0].SpotPrice", "--output", "text"],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def cost_loop(stop_evt: threading.Event, stop_fraction: float, ebs_gb: float) -> None:
    cost = {"total_usd": 0.0, "hours": 0.0, "price_usd_h": None, "budget_usd": config.BUDGET_USD,
            "stop_at_usd": config.BUDGET_USD * stop_fraction, "ebs_gb": ebs_gb, "segments": []}
    if os.path.exists(COST_PATH):
        with open(COST_PATH, encoding="utf-8") as f:
            cost.update(json.load(f))
        cost["stop_at_usd"] = config.BUDGET_USD * stop_fraction
    itype = _imds("meta-data/instance-type") or "inconnu"
    az = _imds("meta-data/placement/availability-zone") or "inconnu"
    life = _imds("meta-data/instance-life-cycle") or "inconnu"
    with _lock:
        _state.update(instance_type=itype, az=az, lifecycle=life)
    cost["segments"].append({"start": _now(), "instance_type": itype, "az": az, "lifecycle": life})
    # Le temps facture est celui de la MACHINE (temps depuis son demarrage, par
    # demarrage), pas celui de la sentinelle : si la sentinelle s'arrete puis
    # repart, l'intervalle est compte au tick suivant (correctif du 03/11 : 3 h
    # non comptees apres l'arret de tmux).
    with open("/proc/sys/kernel/random/boot_id") as f:
        boot_id = f.read().strip()
    boots = cost.setdefault("boots", {})
    boots.setdefault(boot_id, {"uptime_h": 0.0, "usd": 0.0, "first_seen": _now()})
    fixed = os.environ.get("MKAN_PRICE_USD_H")
    price, last_price_t = None, 0.0
    while not stop_evt.is_set():
        if fixed:
            price = float(fixed)
        elif life == "spot" and time.time() - last_price_t > 3600:
            price = _spot_price(itype, az) or price
            last_price_t = time.time()
        with open("/proc/uptime") as f:
            uptime_h = float(f.read().split()[0]) / 3600.0
        b = boots[boot_id]
        dt_h = max(0.0, uptime_h - b["uptime_h"])
        ebs = ebs_gb * EBS_USD_PER_GB_MONTH / 730.0
        if price is None:
            _log("prix inconnu : definir MKAN_PRICE_USD_H (role sans ec2:DescribeSpotPriceHistory ?)")
        else:
            b["usd"] += (price + ebs) * dt_h
            b["uptime_h"] = uptime_h
        cost["total_usd"] = sum(x["usd"] for x in boots.values())
        cost["hours"] = sum(x["uptime_h"] for x in boots.values())
        cost["price_usd_h"] = price
        cost["updated"] = _now()
        tmp = COST_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cost, f, indent=1)
        os.replace(tmp, COST_PATH)
        if cost["total_usd"] >= cost["stop_at_usd"]:
            _log(f"budget atteint ({cost['total_usd']:.2f} $ >= {cost['stop_at_usd']:.2f} $)")
            request_stop("garde-fou de cout", wait_s=600)
            subprocess.run(["sudo", "shutdown", "-h", "+1"])
            stop_evt.set()
            return
        stop_evt.wait(60)


def main():
    ap = argparse.ArgumentParser(description="Garde de l'instance (S3, spot, cout)")
    ap.add_argument("--restore", action="store_true", help="recupere OUT_DIR depuis S3 et quitte")
    ap.add_argument("--sync_every", type=float, default=180.0)
    ap.add_argument("--stop_fraction", type=float, default=0.95,
                    help="fraction du budget a laquelle l'instance s'arrete")
    ap.add_argument("--ebs_gb", type=float, default=float(os.environ.get("MKAN_EBS_GB", "100")))
    args = ap.parse_args()
    os.makedirs(config.OUT_DIR, exist_ok=True)
    if args.restore:
        restore()
        return
    stop_evt = threading.Event()
    threading.Thread(target=spot_loop, args=(stop_evt,), daemon=True).start()
    threading.Thread(target=cost_loop, args=(stop_evt, args.stop_fraction, args.ebs_gb),
                     daemon=True).start()
    _log(f"demarree : copie vers {S3_URI} toutes les {args.sync_every:.0f} s")
    while not stop_evt.is_set():
        sync_up()
        stop_evt.wait(args.sync_every)
    sync_up()
    _log("arretee")


if __name__ == "__main__":
    main()
