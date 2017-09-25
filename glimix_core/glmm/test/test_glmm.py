import pytest
from numpy import asarray, ascontiguousarray, dot, ones, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from optimix import check_grad

from glimix_core.example import linear_eye_cov, nsamples
from glimix_core.glmm import GLMMExpFam, GLMMNormal
from glimix_core.random import bernoulli_sample

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

    assert_allclose(glmm.lml(), -18.950752841710603)

    assert_allclose(check_grad(glmm), 0, atol=1e-3, rtol=RTOL)


# def test_glmm_glmmnormal_get_fast_scanner():
#     random = RandomState(0)
#     X = random.randn(nsamples(), 5)
#     K = linear_eye_cov().feed().value()
#     QS = economic_qs(K)
#
#     eta = random.randn(nsamples())
#     tau = 10 * random.rand(nsamples())
#
#     glmm = GLMMNormal(eta, tau, X, QS)
#     glmm.fit(verbose=False)
#
#     scanner = glmm.get_fast_scanner()
#     scanner.set_scale(1.0)
#     lmls, effect_sizes = scanner.fast_scan(X, verbose=False)
#
#     assert_allclose(lmls, [
#         0.65993260598994397, 3.6666642592705188, 3.6666642592705188,
#         3.4687842095112131, 3.6666642592705188
#     ])
#     assert_allclose(effect_sizes, [
#         -831173461326518.12, -0.0015024987823475178, 0.085827590202214823,
#         42866244922529.086, 0.024047703652107007
#     ])


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
    assert_allclose(glmm.lml(), -44.74191041468836, atol=ATOL, rtol=RTOL)
    glmm.scale = 2.0
    assert_allclose(glmm.lml(), -36.19907331929086, atol=ATOL, rtol=RTOL)
    glmm.scale = 3.0
    assert_allclose(glmm.lml(), -33.02139830387104, atol=ATOL, rtol=RTOL)
    glmm.scale = 4.0
    assert_allclose(glmm.lml(), -31.42553401678996, atol=ATOL, rtol=RTOL)
    glmm.scale = 5.0
    assert_allclose(glmm.lml(), -30.507029479473243, atol=ATOL, rtol=RTOL)
    glmm.scale = 6.0
    assert_allclose(glmm.lml(), -29.937569702301232, atol=ATOL, rtol=RTOL)
    glmm.delta = 0.1
    assert_allclose(glmm.lml(), -30.09977907145003, atol=ATOL, rtol=RTOL)

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

    assert_allclose(glmm.lml(), -27.441700236574512, atol=ATOL, rtol=RTOL)
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

    assert_allclose(glmm.lml(), -33.67996838167506, atol=ATOL, rtol=RTOL)
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

    assert_allclose(glmm.lml(), -29.102168129099287, atol=ATOL, rtol=RTOL)
    glmm.fix('beta')
    glmm.fix('scale')

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -27.635788105778012, atol=ATOL, rtol=RTOL)

    glmm.unfix('beta')
    glmm.unfix('scale')

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -19.68486269551159, atol=ATOL, rtol=RTOL)


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

    assert_allclose(glmm.lml(), -18.60476792256323, atol=ATOL, rtol=RTOL)
    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -7.800621320491801, atol=ATOL, rtol=RTOL)


def test_glmm_bernoulli_problematic():
    random = RandomState(1)
    N = 30
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
    assert_allclose(model.lml(), -20.727007958026853, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0, atol=1e-3)
    assert_allclose(model.scale, 0.879915823030081, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [-0.00247856564728], atol=ATOL, rtol=RTOL)


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
    nsamples = 10

    X = random.randn(nsamples, 2)
    G = random.randn(nsamples, 100)
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

    assert_allclose(glmm.lml(), -11.43920790567486)


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
    assert_allclose(glmm.lml(), -145.01170823743104, atol=ATOL, rtol=RTOL)

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
    assert_allclose(glmm.lml(), -29.632791380478736, atol=ATOL, rtol=RTOL)

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
    assert_allclose(glmm.lml(), -43.154282363439364)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -23.55477374056832, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0.9999999996205406, atol=ATOL, rtol=RTOL)


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
    assert_allclose(glmm.lml(), -39.86006066093746, atol=ATOL, rtol=RTOL)
    assert_allclose(check_grad(glmm, step=1e-4), 0, atol=1e-2)

    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -5.851337541533554, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0.001599583006485038, atol=ATOL, rtol=RTOL)


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

    assert_allclose(glmm0.lml(), -29.10216812909928, atol=ATOL, rtol=RTOL)
    glmm0.fit(verbose=False)

    v = -19.575736562427252
    assert_allclose(glmm0.lml(), v)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -30.832831740038056, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    v = -19.575736562378573
    assert_allclose(glmm0.lml(), v)
    assert_allclose(glmm1.lml(), v)


def test_glmmnormal_copy():
    random = RandomState(0)

    X = random.randn(nsamples(), 5)
    QS = economic_qs(linear_eye_cov().feed().value())

    eta = random.randn(nsamples())
    tau = random.rand(nsamples()) * 10

    glmm0 = GLMMNormal(eta, tau, X, QS)

    assert_allclose(glmm0.lml(), -12.646439806030257, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)

    v = -4.758450057194982
    assert_allclose(glmm0.lml(), v)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -10.986014936977927, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), v, atol=ATOL, rtol=RTOL)
