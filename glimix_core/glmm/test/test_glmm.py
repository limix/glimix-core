import pytest
from numpy import asarray, ascontiguousarray, dot, ones, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs, economic_qs_linear

from glimix_core.example import linear_eye_cov, nsamples
from glimix_core.glmm import GLMMExpFam, GLMMNormal
from glimix_core.random import bernoulli_sample
from optimix import check_grad

ATOL = 1e-6
RTOL = 1e-6


def test_glmm_glmmnormal():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    eta = random.randn(nsamples())
    tau = 10 * random.rand(nsamples())

    glmm = GLMMNormal(eta, tau, X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    assert_allclose(glmm.feed().value(), -50.69748417680114)

    assert_allclose(check_grad(glmm), 0, atol=1e-3, rtol=RTOL)


def test_glmmexpfam_precise():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1.0
    assert_allclose(glmm.lml(), -83.60701102862096, atol=ATOL, rtol=RTOL)
    glmm.scale = 2.0
    assert_allclose(glmm.lml(), -80.27206896319716, atol=ATOL, rtol=RTOL)
    glmm.scale = 3.0
    assert_allclose(glmm.lml(), -80.258491082353, atol=ATOL, rtol=RTOL)
    glmm.scale = 4.0
    assert_allclose(glmm.lml(), -80.98308047220573, atol=ATOL, rtol=RTOL)
    glmm.scale = 5.0
    assert_allclose(glmm.lml(), -81.90719565722284, atol=ATOL, rtol=RTOL)
    glmm.scale = 6.0
    assert_allclose(glmm.lml(), -82.86902861753809, atol=ATOL, rtol=RTOL)
    glmm.delta = 0.1
    assert_allclose(glmm.lml(), -84.33056430633508, atol=ATOL, rtol=RTOL)

    assert_allclose(check_grad(glmm), 0, atol=1e-3, rtol=RTOL)


def test_glmm_delta0():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0

    assert_allclose(glmm.lml(), -82.56509596644209, atol=ATOL, rtol=RTOL)
    assert_allclose(check_grad(glmm, step=2e-5), 0, atol=1e-2)


def test_glmm_delta1():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 1

    assert_allclose(glmm.lml(), -90.22937890822317, atol=ATOL, rtol=RTOL)
    assert_allclose(check_grad(glmm), 0, atol=1e-4)


def test_glmm_wrong_qs():
    random = RandomState(0)
    X = random.randn(10, 15)
    linear_eye_cov().feed().value()
    QS = [0, 1]

    ntri = random.randint(1, 30, 10)
    nsuc = [random.randint(0, i) for i in ntri]

    with pytest.raises(ValueError):
        print(GLMMExpFam((nsuc, ntri), 'binomial', X, QS))


def test_glmm_optimize():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    z = random.multivariate_normal(0.2 * ones(nsamples()), K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = zeros(nsamples(), dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.lml(), -99.33404651904951, atol=ATOL, rtol=RTOL)
    glmm.fix('beta')
    glmm.fix('scale')

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -89.42929673951151, atol=ATOL, rtol=RTOL)

    glmm.unfix('beta')
    glmm.unfix('scale')

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -35.22754172936816, atol=ATOL, rtol=RTOL)


def test_glmm_optimize_low_rank():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = dot(X, X.T)
    z = dot(X, 0.2 * random.randn(5))
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = zeros(nsamples(), dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.lml(), -54.05478637740203, atol=ATOL, rtol=RTOL)
    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -37.82906612336332, atol=ATOL, rtol=RTOL)


def test_glmm_bernoulli_problematic():
    random = RandomState(1)
    N = 500
    G = random.randn(N, N + 50)
    y = bernoulli_sample(0.0, G, random_state=random)
    y = (y, )

    G = ascontiguousarray(G, dtype=float)
    _stdnorm(G, 0, out=G)
    G /= sqrt(G.shape[1])

    QS = economic_qs_linear(G)
    S0 = QS[1]
    S0 /= S0.mean()

    X = ones((len(y[0]), 1))
    model = GLMMExpFam(y, 'bernoulli', X, QS=(QS[0], QS[1]))
    model.delta = 0
    model.fix('delta')
    model.fit(verbose=False)
    assert_allclose(model.lml(), -344.86474884323525, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0, atol=1e-3)
    assert_allclose(model.scale, 0.6026005889095781, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [-0.01806123661347892], atol=ATOL, rtol=RTOL)


def _stdnorm(X, axis=None, out=None):
    X = ascontiguousarray(X)
    if out is None:
        out = X.copy()

    m = out.mean(axis)
    s = out.std(axis)
    ok = s > 0

    out -= m

    if out.ndim == 1:
        if s > 0:
            out /= s
    else:
        out[..., ok] /= s[ok]

    return out


def test_glmm_binomial_pheno_list():
    random = RandomState(0)
    nsamples = 50

    X = random.randn(50, 2)
    G = random.randn(50, 100)
    K = dot(G, G.T)
    ntrials = random.randint(1, 100, nsamples)
    z = dot(G, random.randn(100)) / sqrt(100)

    successes = zeros(len(ntrials), int)
    for i in range(len(ntrials)):
        for _ in range(ntrials[i]):
            successes[i] += int(z[i] + 0.1 * random.randn() > 0)

    y = [successes, ntrials]

    QS = economic_qs(K)
    glmm = GLMMExpFam(y, 'binomial', X, QS)
    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -64.8433300480514)


def test_glmm_scale_very_low():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1e-3
    assert_allclose(glmm.lml(), -151.19262511895698, atol=ATOL, rtol=RTOL)

    assert_allclose(check_grad(glmm), 0, atol=1e-2)


def test_glmm_scale_very_high():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 30.
    assert_allclose(glmm.lml(), -96.75175059098383, atol=ATOL, rtol=RTOL)

    assert_allclose(check_grad(glmm), 0, atol=1e-3)


def test_glmm_delta_zero():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0
    assert_allclose(glmm.lml(), -82.3831297143636)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -76.20092968002656, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0.00012207624715307688, atol=ATOL, rtol=RTOL)


def test_glmm_delta_one():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    K = dot(X, X.T)

    z = dot(X, random.randn(5))

    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = zeros(nsamples())
    for i in range(nsamples()):
        nsuc[i] = sum(z[i] + 0.001 * random.randn(ntri[i]) > 0)

    glmm = GLMMExpFam((nsuc, ntri), 'binomial', ones((nsamples(), 1)), QS)

    glmm.delta = 1
    assert_allclose(glmm.lml(), -126.71338343726902, atol=ATOL, rtol=RTOL)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -9.308984106518762, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 9.930610996068862e-08, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_copy():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    z = random.multivariate_normal(0.2 * ones(nsamples()), K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = zeros(nsamples(), dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm0 = GLMMExpFam((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm0.lml(), -99.33404651904951, atol=ATOL, rtol=RTOL)
    glmm0.fit(verbose=False)

    v = -35.22754141429125
    assert_allclose(glmm0.lml(), v)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), -35.227541384298654, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -219.44355209729233, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    v = -35.227541384298654
    assert_allclose(glmm0.lml(), v)
    assert_allclose(glmm1.lml(), v)


def test_glmmnormal_copy():
    random = RandomState(0)

    X = random.randn(nsamples(), 5)
    QS = economic_qs(linear_eye_cov().feed().value())

    eta = random.randn(nsamples())
    tau = random.rand(nsamples()) * 10

    glmm0 = GLMMNormal(eta, tau, X, QS)

    assert_allclose(glmm0.lml(), -38.29931140952595, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)

    v = -13.00173479849569
    assert_allclose(glmm0.lml(), v)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -33.17274486229496, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), v, atol=ATOL, rtol=RTOL)
