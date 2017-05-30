import pytest

from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import ascontiguousarray, sqrt, ones, dot, zeros, asarray
from numpy_sugar.linalg import economic_qs, economic_qs_linear

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMM
from glimix_core.random import bernoulli_sample

from optimix import check_grad

def test_glmm_precise():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    assert_allclose(glmm.value(), -301.131178219417)
    # assert_allclose(check_grad(glmm), 0, atol=1e-4)

def test_glmm_wrong_qs():
    random = RandomState(0)
    X = random.randn(10, 15)
    K = linear_eye_cov().feed().value()
    QS = [0, 1]

    ntri = random.randint(1, 30, 10)
    nsuc = [random.randint(0, i) for i in ntri]

    with pytest.raises(ValueError):
        print(GLMM((nsuc, ntri), 'binomial', X, QS))

def test_glmm_optimize():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    z = random.multivariate_normal(0.2 * ones(100), K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = zeros(100, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.value(), -323.53924104721864)
    glmm.fix('beta')
    glmm.fix('scale')

    glmm.feed().maximize(progress=False)

    assert_allclose(glmm.value(), -299.47042725069565)

    glmm.unfix('beta')
    glmm.unfix('scale')

    glmm.feed().maximize(progress=False)

    assert_allclose(glmm.value(), -159.1688201218538, rtol=1e-06)

def test_glmm_optimize_low_rank():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = dot(X, X.T)
    z = dot(X, 0.2 * random.randn(5))
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = zeros(100, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.value(), 22.994881066533782)
    #glmm.feed().maximize(progress=False)
    #assert_allclose(glmm.value(), -159.1688201218538, rtol=1e-06)

def test_glmm_bernoulli_problematic():
    random = RandomState(1)
    N = 500
    G = random.randn(N, N+50)
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
    model.feed().maximize(progress=False)
    assert_allclose(model.feed().value(), -344.86474884323525)
    assert_allclose(model.delta, 0, atol=1e-6)
    assert_allclose(model.scale, 0.6027972365996683)
    assert_allclose(model.beta, [-0.018073717897607946])

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
        for j in range(ntrials[i]):
            successes[i] += int(z[i] + 0.1 * random.randn() > 0)

    y = [successes, ntrials]

    QS = economic_qs(K)
    glmm = GLMM(y, 'binomial', X, QS)
    glmm.feed().maximize(progress=False)

    assert_allclose(glmm.value(), -64.84605684476243)
