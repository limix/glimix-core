import warnings

from numpy import all as npall, ascontiguousarray, clip, isfinite


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


def check_outcome(y, lik):
    if not isinstance(lik, (list, tuple)):
        lik = (lik,)

    str_err = "The first item of ``lik`` has to be a string."
    if not isinstance(lik[0], str):
        raise ValueError(str_err)

    lik_name = lik[0].lower()

    y = ascontiguousarray(y, float)
    lik = lik[:1] + tuple(ascontiguousarray(i, float) for i in lik[1:])

    if not npall(isfinite(y)):
        raise ValueError("Outcome must be finite.")

    if lik_name == "poisson":
        return _check_poisson_outcome(y)

    if lik_name in ("binomial", "normal"):
        if len(lik) != 2:
            msg = "``lik`` must be a tuple of two elements for"
            msg += " {} likelihood.".format(lik_name[0].upper() + lik_name[1:])
            raise ValueError(msg)

    return y


def _check_poisson_outcome(y):
    poisson_lim = 25000

    if y.max() > poisson_lim:
        msg = "Output values of Poisson likelihood greater"
        msg += " than {lim} is set to {lim} before applying GLMM."
        warnings.warn(msg.format(lim=poisson_lim))
        y = clip(y, 0, poisson_lim)

    return y
