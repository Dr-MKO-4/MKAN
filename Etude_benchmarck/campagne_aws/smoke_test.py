"""
smoke_test.py -- Essai de bout en bout de la campagne sur de PETITES donnees
fictives (quelques minutes, CPU), dans un dossier temporaire : entrainement,
arret par fichier STOP puis reprise, evaluation, latence, rapport.
A lancer sur le poste et sur l'instance avant la vraie campagne.

Usage : python smoke_test.py
"""

import json
import os
import shutil
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="mkan_smoke_")
os.environ["MKAN_OUT_DIR"] = os.path.join(TMP, "sorties")
os.environ["MKAN_DATA_DIR"] = os.path.join(TMP, "donnees")
_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402

import numpy as np  # noqa: E402


def fake_dataset(W: int) -> str:
    path = os.path.join(config.DATA_DIR, f"brut_W{W}_nsmoke")
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    n_tr, n_va, d = 640, 300, 12
    y_tr = (rng.random(n_tr) < 0.2).astype(np.float32)
    y_va = (rng.random(n_va) < 0.2).astype(np.float32)
    X_tr = rng.normal(size=(n_tr, W, d)).astype(np.float32) + y_tr[:, None, None]
    X_va = rng.normal(size=(n_va, W, d)).astype(np.float32) + y_va[:, None, None]
    arrays = {"X_train": X_tr.clip(-4, 4), "y_train": y_tr, "X_val": X_va.clip(-4, 4), "y_val": y_va,
              "monitor_idx": np.arange(100), "audit_idx": np.arange(64),
              "grid_samples": X_tr.reshape(-1, d)[:2000].clip(-4, 4)}
    for k, v in arrays.items():
        np.save(os.path.join(path, k + ".npy"), v)
    with open(os.path.join(path, "preprocessing.json"), "w") as f:
        json.dump({"regime": "brut", "feature_cols": [f"f{i}" for i in range(d)]}, f)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"regime": "brut", "W": W, "input_size": d, "n_train": n_tr}, f)
    return path


def main():
    from scheduler import make_spec
    from train_run import train
    from measure_latency import measure_pending
    from report import write_reports
    os.makedirs(os.path.join(config.OUT_DIR, "runs"), exist_ok=True)
    proto = dict(max_epochs=2, patience=2, min_delta=1e-4, n_windows_train=640)
    archs = [("lstm", {"Forget": "wavkan_dog", "Input": "efficientkan",
                       "Candidate": "wavkan_dog", "Output": "linear"}),
             ("lstm", {g: "hybrid" for g in ("Forget", "Input", "Candidate", "Output")}),
             ("gru", {"Forget": "wavkan_dog", "Input": "efficientkan", "Candidate": "wavkan_dog"})]
    ok = True
    for i, (cell, bbg) in enumerate(archs):
        W = config.CELL_THETA[cell]["W"]
        spec = make_spec("smoke", cell, "brut", bbg, 1, proto, fake_dataset(W), 12)
        run_dir = os.path.join(config.OUT_DIR, "runs", spec["run_id"])
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "spec.json"), "w") as f:
            json.dump(spec, f)
        if i == 0:   # arret puis reprise sur le premier run
            open(config.STOP_FILE, "w").close()
            code = train(spec, run_dir, threads=2)
            os.remove(config.STOP_FILE)
            resumed = os.path.exists(os.path.join(run_dir, "checkpoint.pt"))
            print(f"[{'OK ' if code == 3 and resumed else 'ECHEC'}] arret par STOP (code {code}), "
                  f"point de reprise {'present' if resumed else 'absent'}")
            ok &= code == 3 and resumed
        code = train(spec, run_dir, threads=2)
        files = ["DONE", "result.json", "model.pt", "val_scores.npy", "edge_profiles.npz"]
        missing = [f for f in files if not os.path.exists(os.path.join(run_dir, f))]
        print(f"[{'OK ' if code == 0 and not missing else 'ECHEC'}] {spec['label']} "
              f"(h={spec['hidden_size']}) {'' if not missing else 'manque ' + str(missing)}")
        ok &= code == 0 and not missing
    n = measure_pending(config.OUT_DIR, threads=2)
    s = write_reports(config.OUT_DIR)
    print(f"[{'OK ' if n == 3 and s['n_runs_done'] == 3 else 'ECHEC'}] latence ({n}) et rapport "
          f"({s['n_runs_done']} runs, fitness de tete {s['ranking'][0]['fitness']:.4f})")
    ok &= n == 3 and s["n_runs_done"] == 3
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nESSAI COMPLET CONFORME" if ok else "\nESSAI EN ECHEC")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
