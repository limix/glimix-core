import pytest
from numpy import asarray, ascontiguousarray, dot, ones, sqrt, zeros
from numpy.random import RandomState
from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs, economic_qs_linear

from glimix_core.example import linear_eye_cov, nsamples
from glimix_core.glmm import GLMM
from glimix_core.random import bernoulli_sample
from optimix import check_grad

ATOL = 1e-6
RTOL = 1e-6


def assert_allclose(*args, **kwargs):
    from numpy.testing import assert_allclose as aa
    if 'atol' not in kwargs:
        kwargs['atol'] = ATOL
    if 'rtol' not in kwargs:
        kwargs['rtol'] = RTOL
    return aa(*args, **kwargs)


def test_glmm_precise():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1.0
    assert_allclose(glmm.value(), -84.45744885880542)
    glmm.scale = 2.0
    assert_allclose(glmm.value(), -80.72774727035623)
    glmm.scale = 3.0
    assert_allclose(glmm.value(), -80.88552880499182)
    glmm.scale = 4.0
    assert_allclose(glmm.value(), -81.85135542384927)
    glmm.scale = 5.0
    assert_allclose(glmm.value(), -83.01324393832967)
    glmm.scale = 6.0
    assert_allclose(glmm.value(), -84.19516223411439)
    glmm.delta = 0.1
    assert_allclose(glmm.value(), -85.71734789051598)

    assert_allclose(check_grad(glmm), 0, atol=1e-3)


def test_glmm_delta0():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0

    assert_allclose(glmm.value(), -82.56509596644209)
    assert_allclose(check_grad(glmm, step=2e-5), 0, atol=1e-2)


def test_glmm_delta1():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 1

    assert_allclose(glmm.value(), -90.22937890822317)
    assert_allclose(check_grad(glmm), 0, atol=1e-4)


def test_glmm_wrong_qs():
    random = RandomState(0)
    X = random.randn(10, 15)
    linear_eye_cov().feed().value()
    QS = [0, 1]

    ntri = random.randint(1, 30, 10)
    nsuc = [random.randint(0, i) for i in ntri]

    with pytest.raises(ValueError):
        print(GLMM((nsuc, ntri), 'binomial', X, QS))


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
    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.value(), -99.33404651904951)
    glmm.fix('beta')
    glmm.fix('scale')

    glmm.feed().maximize(verbose=False)

    assert_allclose(glmm.value(), -89.42929673951151)

    glmm.unfix('beta')
    glmm.unfix('scale')

    glmm.feed().maximize(verbose=False)

    assert_allclose(glmm.value(), -35.22754172936816, rtol=1e-06)


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
    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.value(), -54.05478637740203)
    glmm.feed().maximize(verbose=False)
    assert_allclose(glmm.value(), -37.82906612336332, rtol=1e-06)


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
    model = GLMM(y, 'bernoulli', X, QS=(QS[0], QS[1]))
    model.delta = 0
    model.fix('delta')
    model.feed().maximize(verbose=False)
    assert_allclose(model.feed().value(), -344.86474884323525)
    assert_allclose(model.delta, 0, atol=1e-3)
    assert_allclose(model.scale, 0.6026005889095781, rtol=1e-5)
    assert_allclose(model.beta, [-0.01806123661347892], rtol=1e-5)


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
    glmm = GLMM(y, 'binomial', X, QS)
    glmm.feed().maximize(verbose=False)

    assert_allclose(glmm.value(), -64.8433300480514)


def test_glmm_scale_very_low():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1e-3
    assert_allclose(glmm.value(), -151.19262511895698)

    assert_allclose(check_grad(glmm), 0, atol=1e-2)


def test_glmm_scale_very_high():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 30.
    assert_allclose(glmm.value(), -96.75175059098383)

    assert_allclose(check_grad(glmm), 0, atol=1e-3)


def test_glmm_delta_zero():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0
    assert_allclose(glmm.value(), -82.3831297143636)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.feed().maximize(verbose=False)
    assert_allclose(glmm.value(), -76.20092968002656)
    assert_allclose(glmm.delta, 0.00012207624715307688)


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

    glmm = GLMM((nsuc, ntri), 'binomial', ones((nsamples(), 1)), QS)

    glmm.delta = 1
    assert_allclose(glmm.value(), -126.71338343726902)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.feed().maximize(verbose=False)
    assert_allclose(glmm.value(), -9.308984106518762)
    assert_allclose(glmm.delta, 9.930610996068862e-08, rtol=1e-4)


def test_glmm_copy():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    K = linear_eye_cov().feed().value()
    z = random.multivariate_normal(0.2 * ones(nsamples()), K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples())
    nsuc = zeros(100, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm0 = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm0.value(), -99.33404651904951)
    glmm0.feed().maximize(verbose=False)

    v = -35.22754141429125
    assert_allclose(glmm0.value(), v)
    glmm1 = glmm0.copy()
    assert_allclose(glmm1.value(), v)
    glmm1.scale = 0.92
    assert_allclose(glmm0.value(), -35.227541384298654)
    assert_allclose(glmm1.value(), -219.4745424133988)

    glmm0.feed().maximize(verbose=False)
    glmm1.feed().maximize(verbose=False)

    v = -35.227541384298654
    assert_allclose(glmm0.value(), v)
    assert_allclose(glmm1.value(), v)
