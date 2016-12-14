from __future__ import division

from numpy import ndarray

from . import _liknorm_ffi


def ptr(a):
    if isinstance(a, ndarray):
        return _liknorm_ffi.ffi.cast("double *", a.ctypes.data)
    return a


def create_liknorm(likelihood_name, nintervals):
    _liknorm_ffi.lib.initialize(nintervals)
    if likelihood_name.lower() == 'bernoulli':
        return BernoulliLikNorm()
    elif likelihood_name.lower() == 'binomial':
        return BinomialLikNorm()
    elif likelihood_name.lower() == 'poisson':
        return PoissonLikNorm()
    elif likelihood_name.lower() == 'exponential':
        return ExponentialLikNorm()
    raise ValueError


class BernoulliLikNorm(object):
    def __init__(self):
        self._moments = _liknorm_ffi.lib.binomial_moments

    def moments(self, phenotype, eta, tau, log_zeroth, mean, variance):
        outcome = phenotype.outcome
        from numpy import ones
        ntrials = ones((len(outcome), ))
        size = len(outcome)
        self._moments(size,
                      ptr(outcome),
                      ptr(ntrials),
                      ptr(eta),
                      ptr(tau), ptr(log_zeroth), ptr(mean), ptr(variance))

    def destroy(self):
        _liknorm_ffi.lib.destroy()


class BinomialLikNorm(object):
    def __init__(self):
        self._moments = _liknorm_ffi.lib.binomial_moments

    def moments(self, phenotype, eta, tau, log_zeroth, mean, variance):
        nsuccesses = phenotype.nsuccesses
        ntrials = phenotype.ntrials
        size = len(nsuccesses)
        _liknorm_ffi.lib.binomial_moments(size,
                                          ptr(nsuccesses),
                                          ptr(ntrials),
                                          ptr(eta),
                                          ptr(tau),
                                          ptr(log_zeroth),
                                          ptr(mean), ptr(variance))

    def destroy(self):
        _liknorm_ffi.lib.destroy()


class PoissonLikNorm(object):
    def moments(self, phenotype, eta, tau, log_zeroth, mean, variance):
        noccurrences = phenotype.noccurrences
        size = len(noccurrences)
        _liknorm_ffi.lib.poisson_moments(size,
                                         ptr(noccurrences),
                                         ptr(eta),
                                         ptr(tau),
                                         ptr(log_zeroth),
                                         ptr(mean), ptr(variance))

    def destroy(self):
        _liknorm_ffi.lib.destroy()


class ExponentialLikNorm(object):
    def moments(self, x, eta, tau, log_zeroth, mean, variance):
        size = len(x)
        _liknorm_ffi.lib.exponential_moments(size,
                                             ptr(x),
                                             ptr(eta),
                                             ptr(tau),
                                             ptr(log_zeroth),
                                             ptr(mean), ptr(variance))

    def destroy(self):
        _liknorm_ffi.lib.destroy()
