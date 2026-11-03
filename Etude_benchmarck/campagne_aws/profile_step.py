"""
profile_step.py -- Profil d'un pas d'entrainement (passage avant + arriere +
Adam) pour trouver ou passe le temps : python profile_step.py TKAN_A 64 4
"""

import os
import sys

_CAMPAGNE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CAMPAGNE_DIR)
import config  # noqa: E402,F401

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

from bench import ARCHS, INPUT_SIZE, _spec  # noqa: E402
from train_run import build_model  # noqa: E402
from loss import mkan_total_loss  # noqa: E402

arch, h, t = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(t)
spec = _spec(arch, h)
torch.manual_seed(0)
model = build_model(spec, INPUT_SIZE)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
x = torch.randn(32, spec["W"], INPUT_SIZE)
y = (torch.rand(32) < 0.2).float()


def step():
    opt.zero_grad(set_to_none=True)
    loss, _, _, _ = mkan_total_loss(model, x, y, lam=spec["lam"], mu1=spec["mu1"], mu2=spec["mu2"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


for _ in range(5):
    step()
with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
    for _ in range(20):
        step()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
