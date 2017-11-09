from numpy import all as npall
from numpy import isfinite, clip, asarray

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
    lik_name = lik_name.lower()
    y = asarray(y, float)

    if not npall(isfinite(y)):
        raise ValueError("Outcome must be finite.")

    if lik_name == 'poisson':
        return _check_poisson_outcome(y)

    if lik_name == 'binomial' or lik_name == 'normal':
        if y.ndim != 2 or y.shape[1] != 2:
            msg = "Outcome must be a matrix of two columns"
            msg += " for {} likelihood.".format(lik_name)
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
