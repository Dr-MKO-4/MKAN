"""Tests des experiences sur les finalistes : extension sans changement de la fonction, pretraitements,
statistique H, noeuds produit, generateur de la recherche. A lancer par Miguel : python -m pytest tests/test_experiences.py"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ICI = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.dirname(ICI), os.path.join(os.path.dirname(ICI), "campagne_aws")):
    sys.path.insert(0, d)

from edges import GenericKANLayer, BASIS_REGISTRY, etendre_modele  # noqa: E402
import niveau1_harness as H  # noqa: E402


def _couche(nom, in_f=6, out_f=4, **kw):
    torch.manual_seed(0)
    cls = BASIS_REGISTRY[nom]
    return GenericKANLayer(cls(in_f, out_f, **(kw or cls.budget_kwargs)), in_f, out_f)


@pytest.mark.parametrize("nom,kw", [("wavkan", {"wavelet": "dog", "n_wavelets": 3}),
                                    ("wavkan", {"wavelet": "morlet", "n_wavelets": 4}),
                                    ("relukan", {"n_bases": 8}), ("hybrid", {})])
def test_poids_nul_conserve_la_fonction(nom, kw):
    c = _couche(nom, **kw)
    x = torch.linspace(-3, 3, 200)[:, None].repeat(1, 6)
    avant = c(x).detach().clone()
    c.basis.extend(3, 0.5, 2.0, 4)
    assert torch.allclose(c(x), avant, atol=1e-6)


def test_bspline_extension_approche_la_fonction():
    c = _couche("efficientkan", G=3, k=4)
    g = torch.Generator().manual_seed(1)
    x = torch.rand(4000, 6, generator=g) * 2 - 1
    avant = c(x).detach().clone()
    n_avant = c.basis.weight.shape[-1]
    c.basis.extend(3, -0.5, 0.5, 4, x)
    assert c.basis.weight.shape[-1] == n_avant + 4
    assert (c(x) - avant).abs().max() < 1e-2           # espace de splines raffine, moindres carres


def test_seules_les_nouvelles_fonctions_de_la_colonne_apprennent():
    c = _couche("wavkan", wavelet="dog", n_wavelets=3)
    c.basis.extend(2, 0.0, 1.0, 4)
    x = torch.rand(64, 6)
    c(x).sum().backward()
    g = c.basis.weight.grad
    assert g[2, :, 3:].abs().sum() > 0
    assert g[0, :, 3:].abs().sum() == 0 and g[5, :, 3:].abs().sum() == 0


def test_deux_extensions_gardent_les_fonctions_figees():
    c = _couche("wavkan", wavelet="dog", n_wavelets=3)
    c.basis.extend(2, 0.0, 1.0, 4)
    c.basis.extend(4, 0.0, 1.0, 4)
    c(torch.rand(64, 6)).sum().backward()
    g = c.basis.weight.grad
    assert g[2, :, 3:7].abs().sum() > 0 and g[4, :, 7:].abs().sum() > 0
    assert g[4, :, 3:7].abs().sum() == 0 and g[2, :, 7:].abs().sum() == 0


def test_preproc_a_ne_regarde_que_le_train():
    g = np.random.default_rng(0)
    n = 6000
    base = {H.ACCOUNT_COL: ["c"] * n, H.TIME_COL: np.arange(n), H.TARGET_COL: 0.0, "action": "PAYMENT"}
    for c in H.RAW_NUMERIC_COLS:
        base[c] = g.lognormal(3, 2, n)
    train = pd.DataFrame(base)
    val = train.copy()
    val["amount"] = train["amount"].min() - 1.0          # sous le min du train
    etat = {}
    H.build_regime_frame(train, "brut", "A", etat)
    assert "amount" in etat["m_j"] and etat["d_ks"]["amount"] >= 0.15
    copie = json.loads(json.dumps(etat))
    v, _ = H.build_regime_frame(val, "brut", "A", etat)
    assert etat == copie                                  # le train seul fixe les parametres
    assert (v["amount"] == 0.0).all()                     # log(1 + max(x - m, 0))


def test_preproc_b_inchange_et_c_sans_transformation():
    train = pd.DataFrame({H.ACCOUNT_COL: ["c"], H.TIME_COL: [0], H.TARGET_COL: [0.0], "action": ["PAYMENT"],
                          **{c: [10.0] for c in H.RAW_NUMERIC_COLS}})
    b, _ = H.build_regime_frame(train, "brut")
    assert b["amount"].iloc[0] == pytest.approx(np.log1p(10.0))
    c, _ = H.build_regime_frame(train, "brut", "C")
    assert c["amount"].iloc[0] == 10.0


def test_d_ks_distingue_normale_et_lognormale():
    g = np.random.default_rng(0)
    assert H.d_ks_normale(g.normal(size=20000), 42) < 0.05
    assert H.d_ks_normale(g.lognormal(0, 2, 20000), 42) >= 0.15


def test_h_trouve_l_interaction():
    from interactions import h_interactions

    class F(torch.nn.Module):
        def forward(self, x):
            d = x[:, -1, :]
            return d[:, 0] * d[:, 1] + d[:, 2]             # 0 et 1 interagissent, pas 2

    X = torch.randn(80, 3, 3, generator=torch.Generator().manual_seed(0))
    p = h_interactions(F(), X, [0, 1, 2])
    assert (p[0]["i"], p[0]["j"]) == (0, 1) and p[0]["H"] > 0.8
    assert p[-1]["H"] < 0.1


def test_noeud_produit_dans_le_modele():
    from train_run import build_model
    base = {"Forget": "wavkan_dog", "Input": "efficientkan", "Candidate": "wavkan_dog", "Output": "linear"}
    spec = {"cell_type": "lstm", "base_by_gate": base, "hidden_size": 8,
            "mult_pairs": {"candidate": {"7": [8, 9]}}}
    m = build_model(spec, 5)
    assert m.cell.candidate_gate.node_types[7] == (8, 9)
    sans = build_model({**spec, "mult_pairs": None}, 5)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in sans.parameters())
    assert m(torch.rand(4, 3, 5)).shape == (4,)


def test_etendre_modele_conserve_le_score_avec_bases_a_poids_nul():
    from train_run import build_model
    base = {"Forget": "wavkan_dog", "Input": "relukan", "Candidate": "wavkan_dog", "Output": "linear"}
    torch.manual_seed(0)
    m = build_model({"cell_type": "lstm", "base_by_gate": base, "hidden_size": 8}, 5).eval()
    x = torch.rand(16, 4, 5)
    avant = m(x).detach().clone()
    n = etendre_modele(m, [(2, 0.0, 1.0)], 5, 4, torch.rand(500, 5))
    assert n == 3                                         # la porte Output lineaire n'a pas d'extension
    assert torch.allclose(m(x), avant, atol=1e-5)


def test_identifiants_du_plan_de_base_inchanges():
    from scheduler import make_spec, threads_par_run
    p = dict(max_epochs=15, patience=3, min_delta=1e-4, n_windows_train=500000)
    base = {"Forget": "wavkan_dog", "Input": "efficientkan", "Candidate": "wavkan_dog", "Output": "linear"}
    a = make_spec("etape4_brut", "lstm", "brut", base, 1, p, "x", 10)
    b = make_spec("etape4_brut", "lstm", "brut", base, 1, p, "y", 10, None)
    c = make_spec("etape4_brut", "lstm", "brut", base, 1, p, "x", 10, {"preproc": "A"})
    assert a["run_id"] == b["run_id"] != c["run_id"]
    assert threads_par_run(25, 64) == 2


def test_derive_regions():
    from derive import regions_de_derive
    g = np.random.default_rng(0)
    ref = g.normal(size=(20000, 2))
    cur = np.stack([g.normal(2.0, 0.3, 20000), g.normal(size=20000)], axis=1)
    r = regions_de_derive(ref, cur, ["a", "b"])
    assert r[0]["derive"] and r[0]["region"][0] < 2.0 < r[0]["region"][1]
    assert not r[1]["derive"]
