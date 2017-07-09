import pytest
from numpy import asarray, ascontiguousarray, dot, ones, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose
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

    glmm.scale = 1.0
    assert_allclose(glmm.value(), -301.131178219417)
    glmm.scale = 2.0
    assert_allclose(glmm.value(), -282.0168292483553)
    glmm.scale = 3.0
    assert_allclose(glmm.value(), -278.9578864177154)
    glmm.scale = 4.0
    assert_allclose(glmm.value(), -279.7798134817152)
    glmm.scale = 5.0
    assert_allclose(glmm.value(), -281.8759791133573)
    glmm.scale = 6.0
    assert_allclose(glmm.value(), -284.41654060633704)
    glmm.delta = 0.1
    assert_allclose(glmm.value(), -288.52736106924954)

    assert_allclose(check_grad(glmm), 0, atol=1e-4)


def test_glmm_delta0():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0

    assert_allclose(glmm.value(), -294.3289786264443)
    assert_allclose(check_grad(glmm, step=1e-5), 0, atol=1e-2)


def test_glmm_delta1():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 1

    assert_allclose(glmm.value(), -317.9043148331947)
    assert_allclose(check_grad(glmm), 0, atol=1e-4)


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

    glmm.feed().maximize(verbose=False)

    assert_allclose(glmm.value(), -299.47042725069565)

    glmm.unfix('beta')
    glmm.unfix('scale')

    glmm.feed().maximize(verbose=False)

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

    assert_allclose(glmm.value(), -179.73542932110485)
    glmm.feed().maximize(verbose=False)
    assert_allclose(glmm.value(), -155.4794212740998, rtol=1e-06)


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
    assert_allclose(model.beta, [-0.01806123661347892])


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
    glmm.feed().maximize(verbose=False)

    assert_allclose(glmm.value(), -64.84586890596634)


def test_glmm_scale_very_low():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1e-3
    assert_allclose(glmm.value(), -660.0166421862141)

    assert_allclose(check_grad(glmm), 0, atol=1e-2)


def test_glmm_scale_very_high():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 30.
    assert_allclose(glmm.value(), -328.25708726581706)

    assert_allclose(check_grad(glmm), 0, atol=1e-3)


def test_glmm_delta_zero():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0
    assert_allclose(glmm.value(), -294.3289786264443)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.feed().maximize(verbose=False)
    assert_allclose(glmm.value(), -263.56884343483136)
    assert_allclose(glmm.delta, 1)


def test_glmm_delta_one():
    random = RandomState(0)
    X = random.randn(100, 5)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    K = dot(X, X.T)

    z = dot(X, random.randn(5))

    QS = economic_qs(K)

    ntri = random.randint(1, 30, 100)
    nsuc = zeros(100)
    for i in range(100):
        nsuc[i] = sum(z[i] + 0.001 * random.randn(ntri[i]) > 0)

    glmm = GLMM((nsuc, ntri), 'binomial', ones((100, 1)), QS)

    glmm.delta = 1
    assert_allclose(glmm.value(), -426.18257638533225)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.feed().maximize(verbose=False)
    assert_allclose(glmm.value(), -20.657040329898603)
    assert_allclose(glmm.delta, 0.01458391103525475, rtol=1e-4)
