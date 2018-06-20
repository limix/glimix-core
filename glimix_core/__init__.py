"""
*******************
glimix_core package
*******************

Fast inference for Generalised Linear Mixed Models.

"""

from __future__ import absolute_import as _

from . import cov, ggp, glmm, gp, lik, link, lmm, mean, random, util
from ._testit import test

__version__ = "1.5.0"

__all__ = [
    "__version__", "test", "ggp", "gp", "lmm", "glmm", "cov", "lik", "mean",
    "link", "random", "util"
]
