from numpy import all as npall
from numpy import isfinite, clip

from .io import wprint


def check_economic_qs(QS):
    if not isinstance(QS, tuple):
        raise ValueError("QS must be a tuple.")

    if not isinstance(QS[0], tuple):
        raise ValueError("QS[0] must be a tuple.")

    fmsg = "QS has non-finite values."

    if not all(npall(isfinite(Q)) for Q in QS[0]):
        raise ValueError(fmsg)

    if not npall(isfinite(QS[1])):
        raise ValueError(fmsg)

    return QS


def check_covariates(X):
    if not X.ndim == 2:
        raise ValueError("Covariates must be a bidimensional array.")

    if not npall(isfinite(X)):
        raise ValueError("Covariates must have finite values only.")

    return X


def check_outcome(y, lik_name):

    if not all(npall(isfinite(yi)) for yi in y):
        raise ValueError("Outcome must be finite.")

    if lik_name.lower() == 'poisson':
        return _check_poisson_outcome(y)

    if len(set(len(yi) for yi in y)) != 1:
        raise ValueError("Outcome must be a tuple of arrays of the same size.")

    if lik_name.lower() == 'normal':
        if len(y) != 2:
            msg = "Outcome must be a tuple of two arrays"
            msg += " for normal likelihood."
            raise ValueError(msg)

    return y


def _check_poisson_outcome(y):
    poisson_lim = 25000

    if y[0].max() > poisson_lim:
        msg = "Output values of Poisson likelihood greater"
        msg += " than {lim} is set to {lim} before applying GLMM."
        wprint(msg.format(lim=poisson_lim))
        y = (clip(y[0], 0, poisson_lim), )

    return y
