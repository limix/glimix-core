from . import numbers
from ._array import unvec, vec
from .check import check_covariates, check_economic_qs, check_outcome
from .eigen import economic_qs_zeros
from .format import format_function, format_named_arr
from .hsolve import hsolve

log2pi = 1.837877066409345339081937709124758839607238769531250

__all__ = [
    "check_covariates",
    "check_economic_qs",
    "check_outcome",
    "economic_qs_zeros",
    "hsolve",
    "numbers",
    "vec",
    "unvec",
    "format_function",
    "format_named_arr",
]