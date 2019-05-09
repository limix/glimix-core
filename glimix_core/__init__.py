"""
*******************
glimix_core package
*******************

Fast inference for Generalised Linear Mixed Models.

"""

from __future__ import absolute_import as _

from . import cov, example, ggp, glmm, gp, lik, link, lmm, mean, random
from ._testit import test

__version__ = "2.0.7"

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
