"""
*******************
glimix_core package
*******************

Fast inference for Generalised Linear Mixed Models.

"""

from __future__ import absolute_import as _

from . import cov, ggp, glmm, gp, lik, link, lmm, mean, random, util
from .__about__ import (__name__, __version__, __author__, __author_email__,
                        __maintainer__, __maintainer_email__, __description__)
from .testit import test

__all__ = [
    "__name__", "__version__", "__author__", "__author_email__",
    "__maintainer__", "__maintainer_email__", "__description__", "test", "ggp",
    "gp", "lmm", "glmm", "cov", "lik", "mean", "link", "random", "util"
]
