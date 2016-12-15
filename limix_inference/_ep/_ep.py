from __future__ import absolute_import, division, unicode_literals

from tqdm import tqdm

import logging
from math import fsum
from time import time

from cachetools import LRUCache
from operator import attrgetter
from cachetools import cachedmethod

from numpy import var as variance
from numpy import (abs, all, any, asarray, diagonal, dot, empty, empty_like,
                   errstate, inf, isfinite, log, maximum, sqrt, sum, zeros,
                   zeros_like)
from scipy.linalg import cho_factor
from scipy.optimize import fmin_tnc
from numpy.linalg import LinAlgError

from numpy_sugar import is_all_finite
from numpy_sugar.linalg import (cho_solve, ddot, dotd, economic_svd, solve, sum2diag,
                        trace2)

from ._conditioning import make_sure_reasonable_conditioning
from numpy_sugar import epsilon

from ._fixed import FixedEP

MAX_EP_ITER = 10
EP_EPS = 1e-5

_magic_numbers = dict(xtol=1e-5, rescale=10, pgtol=1e-5, ftol=1e-5, disp=0)


class EP(object):
    r"""Generic EP implementation.

    Let :math:`\mathrm Q \mathrm S \mathrm Q^{\intercal}` be the economic
    eigendecomposition of the genetic covariance matrix.
    Let :math:`\mathrm U\mathrm S\mathrm V^{\intercal}` be the singular value
    decomposition of the user-provided covariates :math:`\mathrm M`. We define

    .. math::

        \mathrm K = v ((1-\delta)\mathrm Q \mathrm S \mathrm Q^{\intercal} +
                    \delta \mathrm I)

    as the covariance of the prior distribution. As such,
    :math:`v` and :math:`\delta` refer to :py:attr:`_v` and :py:attr:`_delta`
    class attributes, respectively. We also use the following variables for
    convenience:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \sigma_b^2          & = & v (1-\delta) \\
            \sigma_{\epsilon}^2 & = & v \delta
        \end{eqnarray}

    The covariate effect-sizes is given by :math:`\boldsymbol\beta`, which
    implies

    .. math::

        \mathbf m = \mathrm M \boldsymbol\beta

    The prior is thus defined as

    .. math::

        \mathcal N(\mathbf z ~|~ \mathbf m; \mathrm K)

    and the marginal likelihood is given by

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(\mathrm E[y_i | z_i])=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    However, the singular value decomposition of the covariates allows us to
    automatically remove dependence between covariates, which would create
    infinitly number of :math:`\boldsymbol\beta` that lead to global optima.
    Let us define

    .. math::

        \tilde{\boldsymbol\beta} = \mathrm S^{1/2} \mathrm V^{\intercal}
                                    \boldsymbol\beta

    as the covariate effect-sizes we will effectively work with during the
    optimization process. Let us also define the

    .. math::

        \tilde{\mathrm M} = \mathrm U \mathrm S^{1/2}

    as the redundance-free covariates. Naturally,

    .. math::

        \mathbf m = \tilde{\mathrm M} \tilde{\boldsymbol\beta}

    In summary, we will optimize :math:`\tilde{\boldsymbol{\beta}}`, even
    though the user will be able to retrieve the corresponding
    :math:`\boldsymbol{\beta}`.


    Let

    .. math::

        \mathrm{KL}[p(y_i|z_i) p_{-}(z_i|y_i)_{\text{EP}} ~|~
            p(y_i|z_i)_{\text{EP}} p_{-}(z_i|y_i)_{\text{EP}}]

    be the KL divergence we want to minimize at each EP iteration.
    The left-hand side can be described as
    :math:`\hat c_i \mathcal N(z_i | \hat \mu_i; \hat \sigma_i^2)`


    Args:
        M (array_like): :math:`\mathrm M` covariates.
        Q (array_like): :math:`\mathrm Q` of the economic
                        eigendecomposition.
        S (array_like): :math:`\mathrm S` of the economic
                        eigendecomposition.
        overdispersion (bool): `True` for :math:`\sigma_{\epsilon}^2 \ge 0`,
                `False` for :math:`\sigma_{\epsilon}^2=0`.
        QSQt (array_like): :math:`\mathrm Q \mathrm S
                        \mathrm Q^{\intercal}` in case this has already
                        been computed. Defaults to `None`.


    Attributes:
        _v (float): Total variance :math:`v` from the prior distribution.
        _delta (float): Fraction of the total variance due to the identity
                        matrix :math:`\mathrm I`.
        _loghz (array_like): This is :math:`\log(\hat c)` for each site.
        _hmu (array_like): This is :math:`\hat \mu` for each site.
        _hvar (array_like): This is :math:`\hat \sigma^2` for each site.

    """

    def __init__(self, M, Q, S, overdispersion):
        self._cache_SQt = LRUCache(maxsize=1)
        self._cache_m = LRUCache(maxsize=1)
        self._cache_K = LRUCache(maxsize=1)
        self._cache_diagK = LRUCache(maxsize=1)
        self._cache_update = LRUCache(maxsize=1)
        self._cache_lml_components = LRUCache(maxsize=1)
        self._cache_L = LRUCache(maxsize=1)
        self._cache_A = LRUCache(maxsize=1)
        self._cache_C = LRUCache(maxsize=1)
        self._cache_BiQt = LRUCache(maxsize=1)
        self._cache_QBiQtAm = LRUCache(maxsize=1)
        self._cache_QBiQtCteta = LRUCache(maxsize=1)

        self._logger = logging.getLogger(__name__)

        if not is_all_finite(Q) or not is_all_finite(isfinite(S)):
            raise ValueError("There are non-finite numbers in the provided" +
                             " eigen decomposition.")

        if S.min() <= 0:
            raise ValueError("The provided covariance matrix is not" +
                             " positive-definite because the minimum" +
                             " eigvalue is %f." % S.min())

        make_sure_reasonable_conditioning(S)

        self._S = S
        self._Q = Q
        self.__QSQt = None

        nsamples = M.shape[0]
        self._previous_sitelik_tau = zeros(nsamples)
        self._previous_sitelik_eta = zeros(nsamples)

        self._sitelik_tau = zeros(nsamples)
        self._sitelik_eta = zeros(nsamples)

        self._cav_tau = zeros(nsamples)
        self._cav_eta = zeros(nsamples)

        self._joint_tau = zeros(nsamples)
        self._joint_eta = zeros(nsamples)

        self._v = None
        self._delta = 0
        self._overdispersion = overdispersion
        self._tM = None
        self.__tbeta = None
        self._covariate_setup(M)

        self._loghz = empty(nsamples)
        self._hmu = empty(nsamples)
        self._hvar = empty(nsamples)
        self._ep_params_initialized = False

    def _covariate_setup(self, M):
        self._M = M
        SVD = economic_svd(M)
        self._svd_U = SVD[0]
        self._svd_S12 = sqrt(SVD[1])
        self._svd_V = SVD[2]
        self._tM = ddot(self._svd_U, self._svd_S12, left=False)

    def _init_ep_params(self):
        self._logger.info("EP parameters initialization.")

        if self._ep_params_initialized:
            self._joint_update()
        else:
            self._joint_initialize()
            self._sitelik_initialize()
            self._ep_params_initialized = True

    def fixed_ep(self):
        w1, w2, w3, _, _, w6, w7 = self._lml_components()

        lml_const = w1 + w2 + w3 + w6 + w7

        beta_nom = self._optimal_beta_nom()

        return FixedEP(lml_const,
                       self._A(),
                       self._C(),
                       self._L(), self._Q,
                       self._QBiQtCteta(), self._sitelik_eta, beta_nom)

    def _joint_initialize(self):
        r"""Initialize the mean and covariance of the posterior.

        Given that :math:`\tilde{\mathrm T}` is a matrix of zeros before the
        first EP iteration, we have

        .. math::
            :nowrap:

            \begin{eqnarray}
                \Sigma         & = & \mathrm K \\
                \boldsymbol\mu & = & \mathrm K^{-1} \mathbf m
            \end{eqnarray}
        """
        self._joint_tau[:] = 1 / self._diagK()
        self._joint_eta[:] = self.m()
        self._joint_eta[:] *= self._joint_tau

    def _sitelik_initialize(self):
        self._sitelik_tau[:] = 0.
        self._sitelik_eta[:] = 0.

    @cachedmethod(attrgetter('_cache_K'))
    def K(self):
        r"""Covariance matrix of the prior.

        Returns:
            :math:`\sigma_b^2 \mathrm Q_0 \mathrm S_0 \mathrm Q_0^{\intercal} + \sigma_{\epsilon}^2 \mathrm I`.
        """
        return sum2diag(self.sigma2_b * self._QSQt(), self.sigma2_epsilon)

    def _Kdot(self, x):
        Q = self._Q
        S = self._S
        out = dot(Q.T, x)
        out *= S
        out = dot(Q, out)
        out *= (1 - self.delta)
        out += self.delta * x
        out *= self.v
        return out

    @cachedmethod(attrgetter('_cache_diagK'))
    def _diagK(self):
        r"""Returns the diagonal of :math:`\mathrm K`."""
        return self.sigma2_b * self._diagQSQt() + self.sigma2_epsilon

    def _diagQSQt(self):
        return self._QSQt().diagonal()

    @cachedmethod(attrgetter('_cache_m'))
    def m(self):
        r"""Mean vector of the prior.

        Returns:
            :math:`\mathrm M \boldsymbol\beta`.
        """
        return dot(self._tM, self._tbeta)

    @property
    def covariates_variance(self):
        r"""Variance explained by the covariates.

        It is defined as

        .. math::

            \sigma_a^2 = \sum_{s=1}^p \left\{ \sum_{i=1}^n \left(
                \mathrm M_{i,s}\beta_s - \sum_{j=1}^n
                \frac{\mathrm M_{j,s}\beta_s}{n} \right)^2 \Big/ n
            \right\}

        where :math:`p` is the number of covariates and :math:`n` is the number
        of individuals. One can show that it amounts to
        :math:`\sum_s \beta_s^2` whenever the columns of :math:`\mathrm M`
        are normalized to have mean and standard deviation equal to zero and
        one, respectively.
        """
        return fsum(variance(self.M * self.beta, axis=0))

    @property
    def sigma2_b(self):
        r"""Returns :math:`v (1-\delta)`."""
        return self.v * (1 - self.delta)

    @property
    def sigma2_epsilon(self):
        r"""Returns :math:`v \delta`."""
        return self.v * self.delta

    @property
    def delta(self):
        r"""Returns :math:`\delta`."""
        return self._delta

    @delta.setter
    def delta(self, v):
        r"""Set :math:`\delta`."""
        self._cache_K.clear()
        self._cache_diagK.clear()
        self._cache_update.clear()
        self._cache_lml_components.clear()
        self._cache_L.clear()
        self._cache_A.clear()
        self._cache_C.clear()
        self._cache_BiQt.clear()
        self._cache_QBiQtAm.clear()
        self._cache_QBiQtCteta.clear()
        assert 0 <= v <= 1
        self._delta = v

    @property
    def v(self):
        r"""Returns :math:`v`."""
        return self._v

    @v.setter
    def v(self, v):
        r"""Set :math:`v`."""
        self._cache_K.clear()
        self._cache_diagK.clear()
        self._cache_update.clear()
        self._cache_lml_components.clear()
        self._cache_L.clear()
        self._cache_A.clear()
        self._cache_C.clear()
        self._cache_BiQt.clear()
        self._cache_QBiQtAm.clear()
        self._cache_QBiQtCteta.clear()
        assert 0 <= v
        self._v = max(v, epsilon.small)

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        self._cache_lml_components.clear()
        self._cache_QBiQtAm.clear()
        self._cache_m.clear()
        self._cache_update.clear()
        if self.__tbeta is None:
            self.__tbeta = asarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        r"""Returns :math:`\boldsymbol\beta`."""
        return solve(self._svd_V.T, self._tbeta / self._svd_S12)

    @beta.setter
    def beta(self, value):
        self._tbeta = self._svd_S12 * dot(self._svd_V.T, value)

    @property
    def M(self):
        r"""Returns :math:`\mathrm M`."""
        return self._M

    @M.setter
    def M(self, value):
        self._covariate_setup(value)
        self._cache_m.clear()
        self._cache_QBiQtAm.clear()
        self._cache_update.clear()
        self._cache_lml_components.clear()

    @cachedmethod(attrgetter('_cache_lml_components'))
    def _lml_components(self):
        self._update()

        S = self._S
        m = self.m()
        ttau = self._sitelik_tau
        teta = self._sitelik_eta
        ctau = self._cav_tau
        ceta = self._cav_eta
        tctau = ttau + ctau
        A = self._A()
        C = self._C()
        L = self._L()
        Am = A * m

        QBiQtCteta = self._QBiQtCteta()
        QBiQtAm = self._QBiQtAm()

        gS = self.sigma2_b * S
        eC = self.sigma2_epsilon * C

        w1 = -sum(log(diagonal(L))) + (-sum(log(gS)) / 2 + log(A).sum() / 2)

        w2 = eC * teta
        w2 += C * QBiQtCteta
        w2 -= teta / tctau
        w2 = dot(teta, w2) / 2

        w3 = dot(ceta, (ttau * ceta - 2 * teta * ctau) / (ctau * tctau)) / 2

        w4 = dot(m * C, teta) - dot(Am, QBiQtCteta)

        w5 = -dot(Am, m) / 2 + dot(Am, QBiQtAm) / 2

        w6 = -sum(log(ttau)) + sum(log(tctau)) - sum(log(ctau))
        w6 /= 2

        w7 = sum(self._loghz)

        return (w1, w2, w3, w4, w5, w6, w7)

    def lml(self):
        return fsum(self._lml_components())

    def _gradient_over_v(self):
        self._update()

        A = self._A()
        Q = self._Q
        S = self._S
        C = self._C()
        m = self.m()
        delta = self.delta
        v = self.v
        teta = self._sitelik_eta

        AQ = ddot(A, Q, left=True)
        SQt = ddot(S, Q.T, left=True)

        Am = A * m
        Em = Am - A * self._QBiQtAm()

        Cteta = C * teta
        Eu = Cteta - A * self._QBiQtCteta()

        u = Em - Eu

        uBiQtAK0, uBiQtAK1 = self._uBiQtAK()

        out = dot(u, self._Kdot(u))
        out /= v
        out -= (1 - delta) * trace2(AQ, SQt)
        out -= delta * A.sum()
        out += (1 - delta) * trace2(AQ, uBiQtAK0)
        out += delta * trace2(AQ, uBiQtAK1)
        out /= 2
        return out

    def _gradient_over_delta(self):
        self._update()

        v = self.v
        delta = self.delta
        Q = self._Q
        S = self._S

        A = self._A()
        C = self._C()
        m = self.m()
        teta = self._sitelik_eta

        Am = A * m
        Em = Am - A * self._QBiQtAm()

        Cteta = C * teta
        Eu = Cteta - A * self._QBiQtCteta()

        u = Em - Eu

        AQ = ddot(A, Q, left=True)
        SQt = ddot(S, Q.T, left=True)

        BiQt = self._BiQt()

        uBiQtAK0, uBiQtAK1 = self._uBiQtAK()

        out = -trace2(AQ, uBiQtAK0)
        out -= (delta / (1 - delta)) * trace2(AQ, uBiQtAK1)
        out += trace2(AQ, ddot(BiQt, A, left=False)) * \
            ((delta / (1 - delta)) + 1)
        out += (1 + delta / (1 - delta)) * dot(u, u)
        out += trace2(AQ, SQt) + (delta / (1 - delta)) * A.sum()
        out -= (1 + delta / (1 - delta)) * A.sum()

        out *= v

        out -= dot(u, self._Kdot(u)) / (1 - delta)

        return out / 2

    def _gradient_over_both(self):
        self._update()

        v = self.v
        delta = self.delta
        Q = self._Q
        S = self._S
        A = self._A()
        AQ = ddot(A, Q, left=True)
        SQt = ddot(S, Q.T, left=True)
        BiQt = self._BiQt()
        uBiQtAK0, uBiQtAK1 = self._uBiQtAK()

        C = self._C()
        m = self.m()
        teta = self._sitelik_eta
        Q = self._Q
        As = A.sum()

        Am = A * m
        Em = Am - A * self._QBiQtAm()

        Cteta = C * teta
        Eu = Cteta - A * self._QBiQtCteta()

        u = Em - Eu
        uKu = dot(u, self._Kdot(u))
        tr1 = trace2(AQ, uBiQtAK0)
        tr2 = trace2(AQ, uBiQtAK1)

        dv = uKu / v
        dv -= (1 - delta) * trace2(AQ, SQt)
        dv -= delta * As
        dv += (1 - delta) * tr1
        dv += delta * tr2
        dv /= 2

        dd = delta / (1 - delta)
        ddelta = -tr1
        ddelta -= dd * tr2
        ddelta += trace2(AQ, ddot(BiQt, A, left=False)) * (dd + 1)
        ddelta += (dd + 1) * dot(u, u)
        ddelta += trace2(AQ, SQt)
        ddelta -= As
        ddelta *= v
        ddelta -= uKu / (1 - delta)
        ddelta /= 2

        return asarray([dv, ddelta])

    @cachedmethod(attrgetter('_cache_update'))
    def _update(self):
        self._init_ep_params()

        self._logger.info('EP loop has started.')

        pttau = self._previous_sitelik_tau
        pteta = self._previous_sitelik_eta

        ttau = self._sitelik_tau
        teta = self._sitelik_eta

        jtau = self._joint_tau
        jeta = self._joint_eta

        ctau = self._cav_tau
        ceta = self._cav_eta

        i = 0
        while i < MAX_EP_ITER:
            pttau[:] = ttau
            pteta[:] = teta

            ctau[:] = jtau - ttau
            ceta[:] = jeta - teta
            self._tilted_params()

            if not all(isfinite(self._hvar)) or any(self._hvar == 0.):
                raise Exception('Error: not all(isfinite(hsig2))' +
                                ' or any(hsig2 == 0.).')

            self._sitelik_update()
            self._cache_lml_components.clear()
            self._cache_L.clear()
            self._cache_A.clear()
            self._cache_C.clear()
            self._cache_BiQt.clear()
            self._cache_QBiQtAm.clear()
            self._cache_QBiQtCteta.clear()

            self._joint_update()

            tdiff = abs(pttau - ttau)
            ediff = abs(pteta - teta)
            aerr = tdiff.max() + ediff.max()

            if pttau.min() <= 0. or (0. in pteta):
                rerr = inf
            else:
                rtdiff = tdiff / abs(pttau)
                rediff = ediff / abs(pteta)
                rerr = rtdiff.max() + rediff.max()

            i += 1
            if aerr < 2 * EP_EPS or rerr < 2 * EP_EPS:
                break

        if i + 1 == MAX_EP_ITER:
            self._logger.warning('Maximum number of EP iterations has' +
                                 ' been attained.')

        self._logger.info('EP loop has performed %d iterations.', i)

    def _joint_update(self):
        A = self._A()
        C = self._C()
        m = self.m()
        Q = self._Q
        v = self.v
        delta = self.delta
        teta = self._sitelik_eta
        jtau = self._joint_tau
        jeta = self._joint_eta
        Kteta = self._Kdot(teta)

        BiQt = self._BiQt()
        uBiQtAK0, uBiQtAK1 = self._uBiQtAK()

        jtau[:] = -dotd(Q, uBiQtAK0)
        jtau *= 1 - delta
        jtau -= delta * dotd(Q, uBiQtAK1)
        jtau *= v
        jtau += self._diagK()

        jtau[:] = 1 / jtau

        dot(Q, dot(BiQt, -A * Kteta), out=jeta)
        jeta += Kteta
        jeta += m
        jeta -= self._QBiQtAm()
        jeta *= jtau
        jtau /= C

    def _sitelik_update(self):
        hmu = self._hmu
        hvar = self._hvar
        tau = self._cav_tau
        eta = self._cav_eta
        self._sitelik_tau[:] = maximum(1.0 / hvar - tau, 1e-16)
        self._sitelik_eta[:] = hmu / hvar - eta

    def _optimal_beta_nom(self):
        A = self._A()
        C = self._C()
        teta = self._sitelik_eta
        Cteta = C * teta
        return Cteta - A * self._QBiQtCteta()

    def _optimal_tbeta_denom(self):
        L = self._L()
        Q = self._Q
        AM = ddot(self._A(), self._tM, left=True)
        QBiQtAM = dot(Q, cho_solve(L, dot(Q.T, AM)))
        return dot(self._tM.T, AM) - dot(AM.T, QBiQtAM)

    def _optimal_tbeta(self):
        self._update()

        if all(abs(self._M) < 1e-15):
            return zeros_like(self._tbeta)

        u = dot(self._tM.T, self._optimal_beta_nom())
        Z = self._optimal_tbeta_denom()

        try:
            with errstate(all='raise'):
                self._tbeta = solve(Z, u)

        except (LinAlgError, FloatingPointError):
            self._logger.warning('Failed to compute the optimal beta.' +
                                 ' Zeroing it.')
            self.__tbeta[:] = 0.

        return self.__tbeta

    def _optimize_beta(self):
        ptbeta = empty_like(self._tbeta)

        step = inf
        i = 0
        while step > 1e-7 and i < 5:
            ptbeta[:] = self._tbeta
            self._optimal_tbeta()
            step = sum((self._tbeta - ptbeta)**2)
            i += 1

    def _start_optimizer(self):
        x0 = [self.v]
        bounds = [(epsilon.small, inf)]

        if self._overdispersion:
            klass = FunCostOverdispersion
            x0 += [self.delta]
            bounds += [(0, 1 - 1e-5)]
        else:
            klass = FunCost

        return (klass, x0, bounds)

    def _finish_optimizer(self, x):
        self.v = x[0]
        if self._overdispersion:
            self.delta = x[1]

        self._optimize_beta()

    def optimize(self, progress=None):
        self._logger.info("Start of optimization.")
        progress = tqdm() if progress is None else progress

        (klass, x0, bounds) = self._start_optimizer()

        start = time()
        with progress as pbar:
            func = klass(self, pbar)
            x = fmin_tnc(func, x0, bounds=bounds, **_magic_numbers)[0]

        self._finish_optimizer(x)

        msg = "End of optimization (%.3f seconds, %d function calls)."
        self._logger.info(msg, time() - start, func.nfev)

    @cachedmethod(attrgetter('_cache_A'))
    def _A(self):
        r"""Returns :math:`\mathcal A = \tilde{\mathrm T} \mathcal C^{-1}`."""
        ttau = self._sitelik_tau
        s2 = self.sigma2_epsilon
        return ttau / (ttau * s2 + 1)

    @cachedmethod(attrgetter('_cache_C'))
    def _C(self):
        r"""Returns :math:`\mathcal C = \sigma_{\epsilon}^2 \tilde{\mathrm T} +
            \mathrm I`."""
        ttau = self._sitelik_tau
        s2 = self.sigma2_epsilon
        return 1 / (ttau * s2 + 1)

    @cachedmethod(attrgetter('_cache_SQt'))
    def _SQt(self):
        r"""Returns :math:`\mathrm S \mathrm Q^\intercal`."""
        return ddot(self._S, self._Q.T, left=True)

    def _QSQt(self):
        r"""Returns :math:`\mathrm Q \mathrm S \mathrm Q^\intercal`."""
        if self.__QSQt is None:
            Q = self._Q
            self.__QSQt = dot(Q, self._SQt())
        return self.__QSQt

    @cachedmethod(attrgetter('_cache_BiQt'))
    def _BiQt(self):
        Q = self._Q
        return cho_solve(self._L(), Q.T)

    @cachedmethod(attrgetter('_cache_L'))
    def _L(self):
        r"""Returns the Cholesky factorization of :math:`\mathcal B`.

        .. math::

            \mathcal B = \mathrm Q^{\intercal}\mathcal A\mathrm Q
                (\sigma_b^2 \mathrm S)^{-1}
        """
        Q = self._Q
        A = self._A()
        B = dot(Q.T, ddot(A, Q, left=True))
        sum2diag(B, 1. / (self.sigma2_b * self._S), out=B)
        return cho_factor(B, lower=True)[0]

    @cachedmethod(attrgetter('_cache_QBiQtCteta'))
    def _QBiQtCteta(self):
        Q = self._Q
        L = self._L()
        C = self._C()
        teta = self._sitelik_eta
        return dot(Q, cho_solve(L, dot(Q.T, C * teta)))

    @cachedmethod(attrgetter('_cache_QBiQtAm'))
    def _QBiQtAm(self):
        Q = self._Q
        L = self._L()
        A = self._A()
        m = self.m()

        return dot(Q, cho_solve(L, dot(Q.T, A * m)))

    def _uBiQtAK(self):
        BiQt = self._BiQt()
        S = self._S
        Q = self._Q
        BiQtA = ddot(BiQt, self._A(), left=False)
        BiQtAQS = dot(BiQtA, Q)
        ddot(BiQtAQS, S, left=False, out=BiQtAQS)

        return dot(BiQtAQS, Q.T), BiQtA


class FunCostOverdispersion(object):
    def __init__(self, ep, pbar):
        super(FunCostOverdispersion, self).__init__()
        self._ep = ep
        self._pbar = pbar
        self.nfev = 0

    def __call__(self, x):
        self._ep.v = x[0]
        self._ep.delta = x[1]
        self._ep._optimize_beta()
        # self._pbar.update(self.nfev)
        self._pbar.update()
        self.nfev += 1
        return (-self._ep.lml(), -self._ep._gradient_over_both())


class FunCost(object):
    def __init__(self, ep, pbar):
        super(FunCost, self).__init__()
        self._ep = ep
        self._pbar = pbar
        self.nfev = 0

    def __call__(self, x):
        self._ep.v = x[0]
        self._ep._optimize_beta()
        # self._pbar.update(self.nfev)
        self._pbar.update()
        self.nfev += 1
        return (-self._ep.lml(), -self._ep._gradient_over_v())
