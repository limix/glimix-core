"""Product of likelihood functions.

BernoulliProdLik  Bernoulli likelihood.
BinomialProdLik   Binomial likelihood.
DeltaProdLik      Delta likelihood.
PoissonProdLik    Poisson likelihood.
"""
from ._prod import BernoulliProdLik, BinomialProdLik, DeltaProdLik, PoissonProdLik

__all__ = ["DeltaProdLik", "BernoulliProdLik", "BinomialProdLik", "PoissonProdLik"]
