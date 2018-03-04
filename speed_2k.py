import numpy as np

import numpy_sugar as ns
from glimix_core.glmm import GLMMExpFam


def logit(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    nsamples = 2000

    random = np.random.RandomState(11)
    X = random.randn(nsamples, nsamples + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])

    K = X.dot(X.T)

    z = random.multivariate_normal(0.2 * np.ones(nsamples), 0.3 * K)
    z += 0.7 * random.randn(nsamples)
    QS = ns.linalg.economic_qs(K)

    ntri = random.randint(5, 30, nsamples)
    nsuc = np.zeros(nsamples, dtype=int)

    for (i, ni) in enumerate(ntri):
        nsuc[i] = random.binomial(ni, logit(z[i]))

    nsuc = np.ascontiguousarray(nsuc, float)
    ntri = np.ascontiguousarray(ntri, float)

    # glmm = GLMMExpFam(
    #     (nsuc, ntri),
    #     'binomial',
    #     np.ones((nsamples, 1)),
    #     QS,
    #     n_int=200,
    #     rtol=ns.epsilon.small * 1000 * 100,
    #     atol=ns.epsilon.small * 100)

    # 156.23s
    # glmm = GLMMExpFam((nsuc, ntri), 'binomial', np.ones((nsamples, 1)), QS)
    # glmm.fit(verbose=True)

    #
    # glmm = GLMMExpFam((nsuc, ntri), 'binomial', np.ones((nsamples, 1)), QS)
    # glmm.fit(verbose=True, factr=1e6, pgtol=1e-3)

    #
    # glmm = GLMMExpFam((nsuc, ntri), 'binomial', np.ones((nsamples, 1)), QS)
    # glmm.fit(verbose=True, factr=1e7, pgtol=1e-3)

    #
    # glmm = GLMMExpFam((nsuc, ntri), 'binomial', np.ones((nsamples, 1)), QS)
    # glmm.fit(verbose=True, factr=1e7, pgtol=1e-2)

    #
    # glmm = GLMMExpFam(
    #     (nsuc, ntri), 'binomial', np.ones((nsamples, 1)), QS, n_int=500)
    # glmm.fit(verbose=True, factr=1e7, pgtol=1e-2)

    #
    # glmm = GLMMExpFam(
    #     (nsuc, ntri), 'binomial', np.ones((nsamples, 1)), QS, n_int=100)
    # glmm.fit(verbose=True, factr=1e7, pgtol=1e-2)

    # 144.87s
    # glmm = GLMMExpFam(
    #     (nsuc, ntri),
    #     'binomial',
    #     np.ones((nsamples, 1)),
    #     QS,
    #     n_int=100,
    #     rtol=ns.epsilon.small * 1000 * 100,
    #     atol=ns.epsilon.small * 100)
    # glmm.fit(verbose=True, factr=1e7, pgtol=1e-2)

    # 106.61s
    glmm = GLMMExpFam(
        (nsuc, ntri),
        'binomial',
        np.ones((nsamples, 1)),
        QS,
        n_int=200,
        rtol=ns.epsilon.small * 1000 * 100,
        atol=ns.epsilon.small * 100)
    glmm.fit(verbose=True, factr=1e7, pgtol=1e-2)

    print(glmm.lml() + 5263.092957685137)
    print(glmm.scale - 0.8443040399536208)
    print(glmm.delta - 0.6250301143942105)
