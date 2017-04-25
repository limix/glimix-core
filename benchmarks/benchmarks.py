from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import ascontiguousarray, sqrt, ones, dot, zeros
from numpy_sugar.linalg import economic_qs, economic_qs_linear, sum2diag

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMM
from glimix_core.random import bernoulli_sample

def get_data(n):
    random = RandomState(0)

    X = ones((n, 1))
    G = random.randn(n, n + 5)
    K = dot(G, G.T)
    K /= K.diagonal().mean()
    sum2diag(K, 1e-3, out=K)
    QS = economic_qs(K)

    ntri = random.randint(1, 30, n)

    z = dot(G, random.randn(n + 5)) / sqrt(n + 5)

    nsuc = zeros(len(ntri), int)
    for i in range(len(ntri)):
        for j in range(ntri[i]):
            nsuc[i] += int(z[i] + 0.5 * random.randn() > 0)

    return (ntri, nsuc, QS, X)

class TimeSuite(object):
    def setup(self):
        self._data = {10: get_data(10), 100: get_data(100)}

    def time_qep_binomial_10_learn(self):
        (ntri, nsuc, QS, X) = self._data[10]

        glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
        glmm.feed().maximize(progress=False)
        assert_allclose(glmm.feed().value(), -19.74077399053363)

    def time_qep_binomial_100_learn(self):
        (ntri, nsuc, QS, X) = self._data[100]

        glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
        glmm.feed().maximize(progress=False)
        assert_allclose(glmm.feed().value(), -218.9503853656612)
