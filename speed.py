import numpy as np

import numpy_sugar as ns
from glimix_core.glmm import GLMMExpFam

if __name__ == '__main__':

    nsamples = 2000

    random = np.random.RandomState(1)
    X = random.randn(nsamples, nsamples + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])

    K = X.dot(X.T)

    z = random.multivariate_normal(0.2 * np.ones(nsamples), K)
    QS = ns.linalg.economic_qs(K)

    ntri = random.randint(5, 30, nsamples)
    nsuc = np.zeros(nsamples, dtype=int)

    for (i, ni) in enumerate(ntri):
        nsuc[i] += sum(z[i] + 0.2 * random.randn(ni) > 0)

    ntri = np.ascontiguousarray(ntri)

    glmm = GLMMExpFam(
        (nsuc, ntri),
        'binomial',
        X,
        QS,
        n_int=200,
        rtol=ns.epsilon.small * 1000 * 100,
        atol=ns.epsilon.small * 100)
    glmm.fit(verbose=True, factr=1e6, pgtol=1e-3)
