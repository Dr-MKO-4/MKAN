"""
compile_test.py -- Compare l'entrainement normal (eager) et compile (torch.compile)
sur les memes poids initiaux et les memes lots : vitesse, et ecart numerique sur la
perte, les poids et les gradients apres N pas.

Usage : python compile_test.py TKAN_A 64 4 [n_steps]
"""

import copy
import os
import sys
import time

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402,F401

import torch  # noqa: E402

from bench import INPUT_SIZE, _spec  # noqa: E402
from train_run import build_model  # noqa: E402
from loss import mkan_total_loss  # noqa: E402

arch, h, t = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 60
torch.set_num_threads(t)
spec = _spec(arch, h)
torch.manual_seed(0)
base = build_model(spec, INPUT_SIZE)
g = torch.Generator().manual_seed(1)
xs = torch.randn(n_steps, 32, spec["W"], INPUT_SIZE, generator=g).clamp(-4, 4)
ys = (torch.rand(n_steps, 32, generator=g) < 0.2).float()


def run(model, compiled):
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    fwd = torch.compile(model, dynamic=False) if compiled else model

    def loss_fn(x, y):
        scores, l1, ent = fwd(x, return_reg=True)
        from loss import weighted_bce
        return weighted_bce(scores, y) + spec["lam"] * (spec["mu1"] * l1 + spec["mu2"] * ent)

    losses, t_first, t0 = [], None, None
    for i in range(n_steps):
        if i == 1:
            t_first = time.perf_counter() - t_start
        if i == 10:
            t0 = time.perf_counter()
        if i == 0:
            t_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(xs[i], ys[i])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    dt = (time.perf_counter() - t0) / (n_steps - 10)
    return losses, dt, t_first


m_eager = copy.deepcopy(base)
m_comp = copy.deepcopy(base)
le, dte, _ = run(m_eager, False)
lc, dtc, tfirst = run(m_comp, True)
dl = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(le, lc))
dp = max(float((p1 - p2).abs().max()) for p1, p2 in zip(m_eager.parameters(), m_comp.parameters()))
pscale = max(float(p.abs().max()) for p in m_eager.parameters())
print(f"{arch} h={h} T={t} : eager {1000 * dte:.1f} ms/pas, compile {1000 * dtc:.1f} ms/pas "
      f"(x{dte / dtc:.2f}), compilation {tfirst:.0f} s ; ecart relatif max perte {dl:.2e}, "
      f"ecart max poids {dp:.2e} (echelle {pscale:.2f}) apres {n_steps} pas")
