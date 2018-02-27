"""
*******************
glimix_core package
*******************

Fast inference for Generalized Linear Mixed Models.

"""

from __future__ import absolute_import as _

from . import cov, ggp, glmm, gp, lik, link, lmm, mean, random, util
from ._test import test

__name__ = "glimix-core"
__version__ = "1.2.25"
__author__ = "Danilo Horta"
__author_email__ = "horta@ebi.ac.uk"

__all__ = [
    "__name__", "__version__", "__author__", "__author_email__", "test", 'ggp',
    'gp', 'lmm', 'glmm', 'cov', 'lik', 'mean', 'link', 'random', 'util'
]
