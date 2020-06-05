import pytest
from numpy import (
    arange,
    array,
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
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from pandas import DataFrame

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

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    z = random.multivariate_normal(0.2 * ones(nsamples), K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm0 = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)

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


def test_glmmexpfam_precise():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ["binomial", ntri], X, QS)
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

    assert_allclose(glmm._check_grad(), 0, atol=1e-3, rtol=RTOL)


def test_glmmexpfam_glmmnormal_get_fast_scanner():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    eta = random.randn(nsamples)
    tau = 10 * random.rand(nsamples)

    glmm = GLMMNormal(eta, tau, X, QS)
    glmm.fit(verbose=False)
    want = [-0.08228058, -0.03910674, 0.04226152, -0.05893827, 0.01718722]
    assert_allclose(glmm.beta, want, atol=1e-3, rtol=1e-3)
    assert_allclose(0.001, glmm.scale, atol=1e-3, rtol=1e-3)
    assert_allclose(0.999999994119, glmm.delta, atol=1e-3, rtol=1e-3)

    scanner = glmm.get_fast_scanner()
    r = scanner.fast_scan(X, verbose=False)

    assert_allclose(
        r["lml"],
        [
            3.666664259270515,
            3.6666642592705188,
            3.666664259270515,
            3.666664259270515,
            3.6666642592705188,
        ],
        rtol=1e-6,
    )
    assert_allclose(
        r["effsizes0"],
        [
            array([-0.04114011, -0.03910684, 0.04226118, -0.05893773, 0.01718666]),
            array([-0.08228023, -0.01955342, 0.04226118, -0.05893773, 0.01718666]),
            array([-0.08228023, -0.03910684, 0.02113059, -0.05893773, 0.01718666]),
            array([-0.08228023, -0.03910684, 0.04226118, -0.02946886, 0.01718666]),
            array([-0.08228023, -0.03910684, 0.04226118, -0.05893773, 0.00859333]),
        ],
        rtol=1e-6,
    )
    assert_allclose(
        r["effsizes1"],
        [
            -0.04114011396191331,
            -0.01955341813308458,
            0.02113058926220273,
            -0.029468864872716646,
            0.008593328546792367,
        ],
        rtol=1e-6,
    )
    assert_allclose(
        r["scale"],
        [
            0.07341651661479422,
            0.07341651661479419,
            0.07341651661479422,
            0.07341651661479422,
            0.0734165166147942,
        ],
        rtol=1e-6,
    )


def test_glmmexpfam_delta0():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 0

    assert_allclose(glmm.lml(), -43.154282363439364, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm._check_grad(step=2e-5), 0, atol=1e-2)


def test_glmmexpfam_delta1():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.delta = 1

    assert_allclose(glmm.lml(), -47.09677870648636, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm._check_grad(), 0, atol=1e-4)


def test_glmmexpfam_wrong_qs():
    random = RandomState(0)
    X = random.randn(10, 15)
    linear_eye_cov().value()
    QS = [0, 1]

    ntri = random.randint(1, 30, 10)
    nsuc = [random.randint(0, i) for i in ntri]

    with pytest.raises(ValueError):
        GLMMExpFam((nsuc, ntri), "binomial", X, QS)


def test_glmmexpfam_optimize():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    z = random.multivariate_normal(0.2 * ones(nsamples), K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)

    assert_allclose(glmm.lml(), -29.102168129099287, atol=ATOL, rtol=RTOL)
    glmm.fix("beta")
    glmm.fix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -27.635788105778012, atol=ATOL, rtol=RTOL)

    glmm.unfix("beta")
    glmm.unfix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -19.68486269551159, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_optimize_low_rank():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = dot(X, X.T)
    z = dot(X, 0.2 * random.randn(5))
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)

    assert_allclose(glmm.lml(), -18.60476792256323, atol=ATOL, rtol=RTOL)
    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -7.800621320491801, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_bernoulli_problematic():
    random = RandomState(1)
    N = 30
    G = random.randn(N, N + 50)
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
    assert_allclose(model.lml(), -20.727007958026853, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0, atol=1e-3)
    assert_allclose(model.scale, 0.879915823030081, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [-0.00247856564728], atol=ATOL, rtol=RTOL)


def test_glmmexpfam_bernoulli_probit_problematic():
    random = RandomState(1)
    N = 30
    G = random.randn(N, N + 50)
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
    assert_allclose(model.lml(), -20.725623168378615, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0.0001220703125, atol=1e-3)
    assert_allclose(model.scale, 0.33022865011938707, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [-0.002617161564786044], atol=ATOL, rtol=RTOL)

    h20 = model.scale * (1 - model.delta) / (model.scale + 1)

    model.unfix("delta")
    model.delta = 0.5
    model.scale = 1.0
    model.fit(verbose=False)

    assert_allclose(model.lml(), -20.725623168378522, atol=ATOL, rtol=RTOL)
    assert_allclose(model.delta, 0.5017852859580029, atol=1e-3)
    assert_allclose(model.scale, 0.9928931515372, atol=ATOL, rtol=RTOL)
    assert_allclose(model.beta, [-0.003203427206253548], atol=ATOL, rtol=RTOL)

    h21 = model.scale * (1 - model.delta) / (model.scale + 1)

    assert_allclose(h20, h21, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_bernoulli_probit_assure_delta_fixed():
    random = RandomState(1)
    N = 10
    G = random.randn(N, N + 50)
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

    assert_allclose(model.lml(), -6.108751595773174, rtol=RTOL)
    assert_allclose(model.delta, 1.4901161193847673e-08, atol=1e-5)
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
    random = RandomState(0)
    n = 10

    X = random.randn(n, 2)
    G = random.randn(n, 100)
    K = dot(G, G.T)
    ntrials = random.randint(1, 100, n)
    z = dot(G, random.randn(100)) / sqrt(100)

    successes = zeros(len(ntrials), int)
    for i in range(len(ntrials)):
        for _ in range(ntrials[i]):
            successes[i] += int(z[i] + 0.1 * random.randn() > 0)

    QS = economic_qs(K)
    glmm = GLMMExpFam(successes, ("binomial", ntrials), X, QS)
    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -11.438821585341866)


def test_glmmexpfam_binomial_large_ntrials():
    random = RandomState(0)
    n = 10

    X = random.randn(n, 2)
    G = random.randn(n, 100)
    K = dot(G, G.T)
    ntrials = random.randint(1, 100000, n)
    z = dot(G, random.randn(100)) / sqrt(100)

    successes = zeros(len(ntrials), int)
    for i in range(len(ntrials)):
        for _ in range(ntrials[i]):
            successes[i] += int(z[i] + 0.1 * random.randn() > 0)

    QS = economic_qs(K)
    glmm = GLMMExpFam(successes, ("binomial", ntrials), X, QS)
    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -43.067433588125446)


def test_glmmexpfam_scale_very_low():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 1e-3
    assert_allclose(glmm.lml(), -145.01170823743104, atol=ATOL, rtol=RTOL)

    assert_allclose(glmm._check_grad(), 0, atol=1e-2)


def test_glmmexpfam_scale_very_high():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    glmm.scale = 30.0
    assert_allclose(glmm.lml(), -29.632791380478736, atol=ATOL, rtol=RTOL)

    assert_allclose(glmm._check_grad(), 0, atol=1e-3)


def test_glmmexpfam_delta_one_zero():
    random = RandomState(1)
    n = 30
    X = random.randn(n, 6)
    K = dot(X, X.T)
    K /= K.diagonal().mean()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, n)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4, -0.2])

    glmm.delta = 0
    assert_allclose(glmm.lml(), -113.24570457063275)
    assert_allclose(glmm._check_grad(step=1e-4), 0, atol=1e-2)

    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -98.21144899310399, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0, atol=ATOL, rtol=RTOL)

    glmm.delta = 1
    assert_allclose(glmm.lml(), -98.00058169240869, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm._check_grad(step=1e-4), 0, atol=1e-1)

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -72.82680948264196, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm.delta, 0.9999999850988439, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_predict():

    random = RandomState(4)
    n = 100
    p = n + 1

    X = ones((n, 2))
    X[:, 1] = random.randn(n)

    G = random.randn(n, p)
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

    beta = random.randn(2)
    z = random.multivariate_normal(dot(X, beta), 0.9 * K + 0.1 * eye(n))

    ntri = random.randint(1, 100, n)
    nsuc = zeros(n, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

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
    assert_(corrcoef([pm, r])[0, 1] > 0.8)
    assert_allclose(pk[0], 54.263705682514846)


def test_glmmexpfam_qs_none():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    K = linear_eye_cov().value()
    z = random.multivariate_normal(0.2 * ones(nsamples), K)

    ntri = random.randint(1, 30, nsamples)
    nsuc = zeros(nsamples, dtype=int)
    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = ascontiguousarray(ntri)
    glmm = GLMMExpFam(nsuc, ("binomial", ntri), X, None)

    assert_allclose(glmm.lml(), -38.30173374439622, atol=ATOL, rtol=RTOL)
    glmm.fix("beta")
    glmm.fix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -32.03927471370041, atol=ATOL, rtol=RTOL)

    glmm.unfix("beta")
    glmm.unfix("scale")

    glmm.fit(verbose=False)

    assert_allclose(glmm.lml(), -19.575736561760586, atol=ATOL, rtol=RTOL)


def test_glmmexpfam_poisson():
    random = RandomState(1)

    # sample size
    n = 30

    # covariates
    offset = ones(n) * random.randn()
    age = random.randint(16, 75, n)
    M = stack((offset, age), axis=1)
    M = DataFrame(stack([offset, age], axis=1), columns=["offset", "age"])
    M["sample"] = [f"sample{i}" for i in range(n)]
    M = M.set_index("sample")

    # genetic variants
    G = random.randn(n, 4)

    # sampling the phenotype
    alpha = random.randn(2)
    beta = random.randn(4)
    eps = random.randn(n)
    y = M @ alpha + G @ beta + eps

    # Whole genotype of each sample.
    X = random.randn(n, 50)
    # Estimate a kinship relationship between samples.
    X_ = (X - X.mean(0)) / X.std(0) / sqrt(X.shape[1])
    K = X_ @ X_.T + eye(n) * 0.1
    # Update the phenotype
    y += random.multivariate_normal(zeros(n), K)
    y = (y - y.mean()) / y.std()

    z = y.copy()
    y = random.poisson(exp(z))

    M = M - M.mean(0)
    QS = economic_qs(K)
    glmm = GLMMExpFam(y, "poisson", M, QS)
    assert_allclose(glmm.lml(), -52.479557279193585)
    glmm.fit(verbose=False)
    assert_allclose(glmm.lml(), -34.09720756737648)
