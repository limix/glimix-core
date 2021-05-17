"""
Generalised linear mixed models.

Fast inference for generalised linear mixed models.
"""
from . import cov, example, ggp, glmm, gp, lik, link, lmm, mean, random
from ._testit import test

__version__ = "3.1.12"

__all__ = [
    "__version__",
    "example",
    "test",
    "ggp",
    "gp",
    "lmm",
    "glmm",
    "cov",
    "lik",
    "mean",
    "link",
    "random",
]
