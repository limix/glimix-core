from . import classes, numbers
from ._array import vec
from .check import check_covariates, check_economic_qs, check_outcome
from .eigen import economic_qs_zeros
from .hsolve import hsolve
from .normalise import normalise_covariance, normalise_covariates, normalise_outcome
from .format import format_function, format_named_arr

log2pi = 1.837877066409345339081937709124758839607238769531250

__all__ = [
    "check_covariates",
    "check_economic_qs",
    "check_outcome",
    "classes",
    "economic_qs_zeros",
    "hsolve",
    "normalise_covariance",
    "normalise_covariates",
    "normalise_outcome",
    "numbers",
    "vec",
    "format_function",
    "format_named_arr"
]
