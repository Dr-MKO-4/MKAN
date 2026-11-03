"""
verifier_correctifs.py -- Verifie les correctifs du 03/11 avant le lancement AWS
(quelques secondes, CPU, aucune donnee lue sauf un modele de l'Etape 4 s'il existe).

  1. weighted_bce sans synchronisation == ancienne formule (ecart <= 1e-6)
  2. bases a grille sans fit_grid == ancienne construction (sorties identiques)
  3. anciens modeles de l'Etape 4 (step4_outputs/models/*.pt) rechargeables
  4. apres fit_grid, la B-spline n'est plus constante au-dela de |x| = 1
  5. penalite de latence : ancienne forme inchangee, nouvelle forme plafonnee

Usage : python verifier_correctifs.py
"""

import glob
import math
import os
import sys

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402,F401

import torch  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK ' if cond else 'ECHEC'}] {name} {detail}")


# 1. weighted_bce
from loss import weighted_bce  # noqa: E402
torch.manual_seed(0)
s, y = torch.rand(32).clamp(0.01, 0.99), (torch.rand(32) < 0.2).float()
n_f = max(y.sum().item(), 1.0); n_l = max(32 - n_f, 1.0)
w = y * (32 * 0.5 / n_f - 32 * 0.5 / n_l) + 32 * 0.5 / n_l
ref = -(w * (y * s.log() + (1 - y) * (1 - s).log())).mean()
check("weighted_bce", abs(float(weighted_bce(s, y)) - float(ref)) <= 1e-6,
      f"(ecart {abs(float(weighted_bce(s, y)) - float(ref)):.2e})")

# 2. bases sans fit_grid : comparaison a l'ancienne formule (centres partages)
from edges import BASIS_REGISTRY  # noqa: E402
x = torch.randn(64, 20) * 2
hyb = BASIS_REGISTRY["hybrid"](20, 8)
c = torch.linspace(-1, 1, 8); h = 2 / 7
g = torch.exp(-0.5 * h ** -2 * (x.unsqueeze(-1) - c) ** 2)
old = (torch.einsum("bim,iom->bio", g, hyb.w_gauss)
       + torch.einsum("bik,iok->bio", torch.cos(x.unsqueeze(-1) * hyb.k_idx), hyb.a_fourier)
       + torch.einsum("bik,iok->bio", torch.sin(x.unsqueeze(-1) * hyb.k_idx), hyb.b_fourier))
check("hybrid inchangee sans fit_grid", torch.allclose(hyb(x), old, atol=1e-5))
bs = BASIS_REGISTRY["efficientkan"](20, 8, G=3, k=4)
xc = x.clamp(float(bs.grid[0, bs.k]), float(bs.grid[0, -bs.k - 1]))
old_bs = torch.einsum("bin,ion->bio", bs._b_splines(xc), bs.weight)
check("B-spline inchangee sans fit_grid", torch.allclose(bs(x), old_bs, atol=1e-5))

# 3. anciens modeles
models = sorted(glob.glob(os.path.join(config.ETUDE_DIR, "step4_outputs", "models", "*.pt")))[:5]
if models:
    from niveau1_r2_seuils import build_model  # noqa: E402
    import json
    rows = [json.loads(l) for l in open(os.path.join(config.ETUDE_DIR, "step4_outputs",
                                                     "step4_progress.jsonl"), encoding="utf-8")]
    by_hash = {r["result"]["key_hash"]: r["result"] for r in rows}
    n_ok = 0
    for m in models:
        res = by_hash.get(os.path.basename(m)[:-3])
        if res is None:
            continue
        model = build_model(res["cell_type"], res["base_by_gate"], 12)
        model.load_state_dict(torch.load(m, map_location="cpu"))
        n_ok += 1
    check("anciens modeles de l'Etape 4 rechargeables", n_ok > 0, f"({n_ok} testes)")
else:
    print("[--] aucun modele de l'Etape 4 trouve : test 3 saute")

# 4. fit_grid
from edges import fit_grids_to_data  # noqa: E402
from lstm_cell import ConfigurableMKANScorer  # noqa: E402
model = ConfigurableMKANScorer(12, 16, gate_bases={"forget": "efficientkan", "input": "hybrid",
                                                   "candidate": "wavkan", "output": "linear"},
                               gate_kwargs={"forget": {"G": 3, "k": 4}})
samples = (torch.randn(5000, 12) * 1.5).clamp(-4, 4)
n = fit_grids_to_data(model, samples)
layer = model.cell.forget_gate
probe = torch.zeros(3, 28)
probe[:, 16] = torch.tensor([1.5, 2.5, 3.5])
vals = layer.edge_activations(probe)[:, 16, 0]
check("fit_grid : B-spline non constante au-dela de 1", n == 2 and float(vals.std()) > 1e-6,
      f"({n} portes recalees ; grille x : {layer.basis.grid[16, layer.basis.k]:.2f} a "
      f"{layer.basis.grid[16, -layer.basis.k - 1]:.2f})")
check("fit_grid : colonnes h inchangees", torch.allclose(layer.basis.grid[0], layer.basis.grid[15]) and
      abs(float(layer.basis.grid[0, layer.basis.k]) + 1) < 1e-6)

# 5. penalite
from extended_heuristic_search import _penalty_latency  # noqa: E402
check("penalite ancienne inchangee", math.isclose(_penalty_latency(150.0), 10 * 0.25))
check("penalite plafonnee", _penalty_latency(400, threshold_ms=500, capped=True) == 0.0
      and math.isclose(_penalty_latency(750, threshold_ms=500, capped=True), 0.5)
      and _penalty_latency(5000, threshold_ms=500, capped=True) == 1.0)

print("\nTOUT EST CONFORME" if ok else "\nAU MOINS UNE VERIFICATION A ECHOUE")
sys.exit(0 if ok else 1)
