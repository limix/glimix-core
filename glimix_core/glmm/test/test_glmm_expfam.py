import pytest
from numpy import (
    arange,
    asarray,
    ascontiguousarray,
    corrcoef,
    dot,
    exp,
    eye,
    ones,
    sqrt,
    stack,
    zeros,
)
from numpy.random import default_rng
from numpy.testing import assert_, assert_allclose
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from glimix_core._util import multivariate_normal

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMMExpFam, GLMMNormal
from glimix_core.random import bernoulli_sample

ATOL = 1e-3
RTOL = 1e-3


def test_glmmexpfam_layout():
    y = asarray([1.0, 0.5])
    X = asarray([[0.5, 1.0]])
    K = asarray([[1.0, 0.0], [0.0, 1.0]])
    QS = economic_qs(K)

    with pytest.raises(ValueError):
        GLMMExpFam(y, "poisson", X, QS=QS)

    y = asarray([1.0])
    with pytest.raises(ValueError):
        GLMMExpFam(y, "poisson", X, QS=QS)


def test_glmmexpfam_copy():
    nsamples = 10

    random = default_rng(2)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    z = multivariate_normal(random, 0.2 * ones(nsamples), K)
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for i, ni in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.random(size=ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm0 = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)

    assert_allclose(glmm0.lml(), -28.7031933618, atol=ATOL, rtol=RTOL)
    glmm0.fit(verbose=False)

    v = -7.0665573503
    assert_allclose(glmm0.lml(), v)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v, rtol=1e-3)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -195.013480839, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    v = -7.0665573503
    assert_allclose(glmm0.lml(), v)
    assert_allclose(glmm1.lml(), v, atol=ATOL)


def test_glmmexpfam_precise():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = [random.integers(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ["binomial", ntri], X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1.0
    assert_allclose(glmm.lml(), -29.6908906579, atol=ATOL, rtol=RTOL)
    glmm.scale = 2.0
    assert_allclose(glmm.lml(), -27.6417735686, atol=ATOL, rtol=RTOL)
    glmm.scale = 3.0
    assert_allclose(glmm.lml(), -27.2379649267, atol=ATOL, rtol=RTOL)
    glmm.scale = 4.0
    assert_allclose(glmm.lml(), -27.243363099, atol=ATOL, rtol=RTOL)
    glmm.scale = 5.0
    assert_allclose(glmm.lml(), -27.3896549751, atol=ATOL, rtol=RTOL)
    glmm.scale = 6.0
    assert_allclose(glmm.lml(), -27.5897920258, atol=ATOL, rtol=RTOL)
    glmm.delta = 0.1
    assert_allclose(glmm.lml(), -28.032630672, atol=ATOL, rtol=RTOL)

    assert_allclose(glmm._check_grad(), 0, atol=1e-3, rtol=RTOL)


def test_glmmexpfam_glmmnormal_get_fast_scanner():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    eta = random.random(size=nsamples)
    tau = 10 * random.uniform(size=nsamples)

    glmm = GLMMNormal(eta, tau, X, QS)
    glmm.fit(verbose=False)
    want = [
        0.176704059082,
        -0.248669792939,
        0.165562054326,
        -0.068362796775,
        0.160376987127,
    ]
    assert_allclose(glmm.beta, want, atol=1e-3, rtol=1e-3)
    assert_allclose(0.001, glmm.scale, atol=1e-3, rtol=1e-3)
    assert_allclose(0.999999994119, glmm.delta, atol=1e-3, rtol=1e-3)

    scanner = glmm.get_fast_scanner()
    r = scanner.fast_scan(X, verbose=False)

    assert_allclose(
        r["lml"],
        [
            6.251971978275,
            6.251971978275,
            6.251971978275,
            6.251971978275,
            6.251971978275,
        ],
        rtol=1e-6,
    )
    assert_allclose(
        r["effsizes0"],
        [
            [0.0883515605, -0.2486713745, 0.1655632824, -0.0683615838, 0.1603763469],
            [0.176703121, -0.1243356873, 0.1655632824, -0.0683615838, 0.1603763469],
            [0.176703121, -0.2486713745, 0.0827816412, -0.0683615838, 0.1603763469],
            [0.176703121, -0.2486713745, 0.1655632824, -0.0341807919, 0.1603763469],
            [0.176703121, -0.2486713745, 0.1655632824, -0.0683615838, 0.0801881735],
        ],
        rtol=1e-6,
    )
    assert_allclose(
        r["effsizes1"],
        [0.0883515605, -0.1243356873, 0.0827816412, -0.0341807919, 0.0801881735],
        rtol=1e-6,
    )
    assert_allclose(
        r["scale"],
        [0.0503128779, 0.0503128779, 0.0503128779, 0.0503128779, 0.0503128779],
        rtol=1e-6,
    )


def test_glmmexpfam_delta0():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = [random.integers(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0

    assert_allclose(glmm.lml(), -29.2156146315, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm._check_grad(step=2e-5), 0, atol=1e-2)


def test_glmmexpfam_delta1():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = [random.integers(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 1

    assert_allclose(glmm.lml(), -31.014766648, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm._check_grad(), 0, atol=1e-4)


def test_glmmexpfam_wrong_qs():
    random = default_rng(0)
    X = random.random(size=(10, 15))
    linear_eye_cov().value()
    QS = [0, 1]

    ntri = random.integers(1, 30, 10)
    nsuc = [random.integers(0, i) for i in ntri]

    with pytest.raises(ValueError):
        GLMMExpFam((nsuc, ntri), "binomial", X, QS)


def test_glmmexpfam_optimize():
    nsamples = 10

    random = default_rng(1)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    z = multivariate_normal(random, 0.2 * ones(nsamples), K)
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for i, ni in enumerate(ntri):
        x = random.random(size=ni)
        nsuc[i] += sum(z[i] + 0.2 * x > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)

    assert_allclose(glmm.lml(), -35.9777836472, atol=ATOL, rtol=RTOL)
    glmm.fix("beta")
    glmm.fix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -32.6840270428, atol=ATOL, rtol=RTOL)

    glmm.unfix("beta")
    glmm.unfix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -1.0447594303e-05, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_optimize_low_rank():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = dot(X, X.T)
    z = dot(X, 0.2 * random.random(size=5))
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for i, ni in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.random(size=ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)

    assert_allclose(glmm.lml(), -16.1748642332, atol=ATOL, rtol=RTOL)
    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -7.7868635894e-08, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_bernoulli_problematic():
    random = default_rng(1)
    N = 30
    G = random.random(size=(N, N + 50))
    y = bernoulli_sample(0.0, G, random_state=random)

    G = ascontiguousarray(G, dtype=float)
    _stdnorm(G, 0, out=G)
    G /= sqrt(G.shape[1])

    QS = economic_qs_linear(G)
    S0 = QS[1]
    S0 /= S0.mean()

    X = ones((len(y), 1))
    model = GLMMExpFam(y, "bernoulli", X, QS=(QS[0], QS[1]))
    model.delta = 0
    model.fix("delta")
    model.fit(verbose=False)
    assert_allclose(model.lml(), -18.9974167099, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0, atol=1e-3)
    assert_allclose(model.scale, 7.6792483084, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [1.0464641056], atol=ATOL, rtol=RTOL)


def test_glmmexpfam_bernoulli_probit_problematic():
    random = default_rng(1)
    N = 30
    G = random.random(size=(N, N + 50))
    y = bernoulli_sample(0.0, G, random_state=random)

    G = ascontiguousarray(G, dtype=float)
    _stdnorm(G, 0, out=G)
    G /= sqrt(G.shape[1])

    QS = economic_qs_linear(G)
    S0 = QS[1]
    S0 /= S0.mean()

    X = ones((len(y), 1))
    model = GLMMExpFam(y, "probit", X, QS=(QS[0], QS[1]))
    model.delta = 0
    model.fix("delta")
    model.fit(verbose=False)
    assert_allclose(model.lml(), -19.0071728127, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 1.4901161194e-08, atol=1e-3)
    assert_allclose(model.scale, 2.6268716097, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [0.6039762584], atol=ATOL, rtol=RTOL)

    h20 = model.scale * (1 - model.delta) / (model.scale + 1)

    model.unfix("delta")
    model.delta = 0.5
    model.scale = 1.0
    model.fit(verbose=False)

    assert_allclose(model.lml(), -19.0071728127, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0.1310042942, atol=1e-3)
    assert_allclose(model.scale, 5.0048839203, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [0.7771521303], atol=ATOL, rtol=RTOL)

    h21 = model.scale * (1 - model.delta) / (model.scale + 1)

    assert_allclose(h20, h21, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_bernoulli_probit_assure_delta_fixed():
    random = default_rng(1)
    N = 10
    G = random.random(size=(N, N + 50))
    y = bernoulli_sample(0.0, G, random_state=random)

    G = ascontiguousarray(G, dtype=float)
    _stdnorm(G, 0, out=G)
    G /= sqrt(G.shape[1])

    QS = economic_qs_linear(G)
    S0 = QS[1]
    S0 /= S0.mean()

    X = ones((len(y), 1))
    model = GLMMExpFam(y, "probit", X, QS=(QS[0], QS[1]))
    model.fit(verbose=False)

    assert_allclose(model.lml(), -6.7303089368, rtol=RTOL)
    assert_allclose(model.delta, -1.7828204051e-06, atol=1e-5)
    assert_(model._isfixed("logitdelta"))


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


def test_glmmexpfam_binomial_pheno_list():
    random = default_rng(0)
    n = 10

    X = random.random(size=(n, 2))
    G = random.random(size=(n, 100))
    K = dot(G, G.T)
    ntrials = random.integers(1, 100, n)
    z = dot(G, random.random(size=100)) / sqrt(100)

    successes = zeros(len(ntrials), int)
    for i in range(len(ntrials)):
        for _ in range(ntrials[i]):
            successes[i] += int(z[i] + 0.1 * random.random() > 0)

    QS = economic_qs(K)
    glmm = GLMMExpFam(successes, ("binomial", ntrials), X, QS)
    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -1.7828204051e-06, rtol=RTOL, atol=ATOL)


def test_glmmexpfam_binomial_large_ntrials():
    random = default_rng(0)
    n = 10

    X = random.random(size=(n, 2))
    G = random.random(size=(n, 100))
    K = dot(G, G.T)
    ntrials = random.integers(1, 100000, n)
    z = dot(G, random.random(size=100)) / sqrt(100)

    successes = zeros(len(ntrials), int)
    for i in range(len(ntrials)):
        for _ in range(ntrials[i]):
            successes[i] += int(z[i] + 0.1 * random.random() > 0)

    QS = economic_qs(K)
    glmm = GLMMExpFam(successes, ("binomial", ntrials), X, QS)
    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -8.2741657696e-05, rtol=RTOL, atol=ATOL)


def test_glmmexpfam_scale_very_low():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = [random.integers(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1e-3
    assert_allclose(glmm.lml(), -64.4994083456, atol=ATOL, rtol=RTOL)

    assert_allclose(glmm._check_grad(), 0, atol=1e-2)


def test_glmmexpfam_scale_very_high():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = [random.integers(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 30.0
    assert_allclose(glmm.lml(), -31.4476266883, atol=ATOL, rtol=RTOL)

    assert_allclose(glmm._check_grad(), 0, atol=1e-3)


def test_glmmexpfam_delta_one_zero():
    random = default_rng(1)
    n = 30
    X = random.random(size=(n, 6))
    K = dot(X, X.T)
    K /= K.diagonal().mean()
    QS = economic_qs(K)

    ntri = random.integers(1, 30, n)
    nsuc = [random.integers(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4, -0.2])

    glmm.delta = 0
    assert_allclose(glmm.lml(), -104.5701519999)
    assert_allclose(glmm._check_grad(step=1e-4), 0, atol=1e-2)

    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -91.7868283938, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0, atol=ATOL, rtol=RTOL)

    glmm.delta = 1
    assert_allclose(glmm.lml(), -91.5788328052, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm._check_grad(step=1e-4), 0, atol=1e-1)

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -71.1968560204, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0.9999999850988439, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_predict():
    random = default_rng(1)
    n = 100
    p = n + 1

    X = ones((n, 2))
    X[:, 1] = random.random(size=n)

    G = random.random(size=(n, p))
    G /= G.std(0)
    G -= G.mean(0)
    G /= sqrt(p)
    K = dot(G, G.T)

    i = asarray(arange(0, n), int)
    si = random.choice(i, n, replace=False)
    ntest = int(n // 5)
    itrain = si[:-ntest]
    itest = si[-ntest:]

    Xtrain = X[itrain, :]
    Ktrain = K[itrain, :][:, itrain]

    Xtest = X[itest, :]

    beta = random.random(size=2)
    z = multivariate_normal(random, dot(X, beta), 0.9 * K + 0.1 * eye(n))

    ntri = random.integers(1, 100, n)
    nsuc = zeros(n, dtype=int)
    for i, ni in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.random(size=ni) > 0)

    ntri = ascontiguousarray(ntri)

    QStrain = economic_qs(Ktrain)
    nsuc_train = ascontiguousarray(nsuc[itrain])
    ntri_train = ascontiguousarray(ntri[itrain])

    nsuc_test = ascontiguousarray(nsuc[itest])
    ntri_test = ascontiguousarray(ntri[itest])

    glmm = GLMMExpFam(nsuc_train, ("binomial", ntri_train), Xtrain, QStrain)
    glmm.fit(verbose=False)
    ks = K[itest, :][:, itrain]
    kss = asarray([K[i, i] for i in itest])
    pm = glmm.predictive_mean(Xtest, ks, kss)
    pk = glmm.predictive_covariance(Xtest, ks, kss)
    r = nsuc_test / ntri_test
    assert_(corrcoef([pm, r])[0, 1] > 0.5)
    assert_allclose(pk[0], 320.4925540377, rtol=1e-6)


def test_glmmexpfam_qs_none():
    nsamples = 10

    random = default_rng(0)
    X = random.random(size=(nsamples, 5))
    K = linear_eye_cov().value()
    z = multivariate_normal(random, 0.2 * ones(nsamples), K)

    ntri = random.integers(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for i, ni in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.random(size=ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, None)

    assert_allclose(glmm.lml(), -56.5137400209, atol=ATOL, rtol=RTOL)
    glmm.fix("beta")
    glmm.fix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -43.4232487385, atol=ATOL, rtol=RTOL)

    glmm.unfix("beta")
    glmm.unfix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -6.745656177, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_poisson():
    random = default_rng(1)

    # sample size
    n = 30

    # covariates
    offset = ones(n) * random.random()
    age = random.integers(16, 75, n)
    M = stack((offset, age), axis=1)

    # genetic variants
    G = random.random(size=(n, 4))

    # sampling the phenotype
    alpha = random.random(size=2)
    beta = random.random(size=4)
    eps = random.random(size=n)
    y = M @ alpha + G @ beta + eps

    # Whole genotype of each sample.
    X = random.random(size=(n, 50))
    # Estimate a kinship relationship between samples.
    X_ = (X - X.mean(0)) / X.std(0) / sqrt(X.shape[1])
    K = X_ @ X_.T + eye(n) * 0.1
    # Update the phenotype
    y += multivariate_normal(random, zeros(n), K)
    y = (y - y.mean()) / y.std()

    z = y.copy()
    y = random.poisson(exp(z))

    M = M - M.mean(0)
    QS = economic_qs(K)
    glmm = GLMMExpFam(y, "poisson", M, QS)
    assert_allclose(glmm.lml(), -48.8750807107)
    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -38.7780148893)
