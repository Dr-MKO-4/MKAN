"""
edges  Interface d'arête commune du benchmark (protocole_benchmark.md §8, étape 1).

    GenericKANLayer  couche in_features x out_features, base interchangeable
    BASIS_REGISTRY   dict {nom: classe de base}, cf. bases.py pour le détail
                      et les kwargs budget_kwargs (B=12/arête) / native_kwargs

Usage :
    from edges import GenericKANLayer, BASIS_REGISTRY

    cls = BASIS_REGISTRY["wavkan"]
    basis = cls(in_features=22, out_features=16, **cls.budget_kwargs)
    layer = GenericKANLayer(basis, in_features=22, out_features=16)
"""

from .bases import (
    BASIS_REGISTRY,
    HybridBasis, GaussianBasis, RSWAFBasis, BSplineBasis, ChebyshevBasis,
    WaveletBasis, ReLUKANBasis, FourierBasis, JacobiBasis, LinearBasis,
)
from .layer import GenericKANLayer

__all__ = [
    "BASIS_REGISTRY", "GenericKANLayer",
    "HybridBasis", "GaussianBasis", "RSWAFBasis", "BSplineBasis", "ChebyshevBasis",
    "WaveletBasis", "ReLUKANBasis", "FourierBasis", "JacobiBasis", "LinearBasis",
]
