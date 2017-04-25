from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import ascontiguousarray, sqrt, ones, dot, zeros
from numpy_sugar.linalg import economic_qs, economic_qs_linear, sum2diag

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMM
from glimix_core.random import bernoulli_sample

class TimeSuite(object):
    def setup(self):
        random = RandomState(0)
        n = 100

        self._X = ones((n, 1))
        self._G = random.randn(n, n + 5)
        K = dot(self._G, self._G.T)
        K /= K.diagonal().mean()
        sum2diag(K, 1e-3, out=K)
        self._QS = economic_qs(K)

        self._ntri = random.randint(1, 30, n)

        z = dot(self._G, random.randn(n + 5)) / sqrt(n + 5)

        self._nsuc = zeros(len(self._ntri), int)
        for i in range(len(self._ntri)):
            for j in range(self._ntri[i]):
                self._nsuc[i] += int(z[i] + 0.5 * random.randn() > 0)

    def time_qep_binomial_1k_learn(self):
        glmm = GLMM((self._nsuc, self._ntri), 'binomial', self._X,
                    self._QS)
        glmm.feed().maximize(progress=False)
        print(glmm.feed().value())
        assert_allclose(glmm.feed().value(), -218.95038533366554)

ts = TimeSuite()
ts.setup()
ts.time_qep_binomial_1k_learn()
