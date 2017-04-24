from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import ascontiguousarray, sqrt, ones
from numpy_sugar.linalg import economic_qs, economic_qs_linear, sum2diag

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMM
from glimix_core.random import bernoulli_sample

class TimeSuite:
    def setup(self):
        random = RandomState(0)
        n = 100
        n1k = 1000
        self._X = random.randn(n, 5)
        self._K = linear_eye_cov().feed().value()
        self._QS = economic_qs(self._K)

        self._ntri = random.randint(1, 30, n)
        self._nsuc = [random.randint(0, i) for i in self._ntri]

        self._X1k = random.randn(n1k, 5)
        self._K1k = self._X1k.dot(self._X1k.T)
        sum2diag(self._K1k, 1e-3, out=self._K1k)
        self._QS1k = economic_qs(self._K1k)

        self._ntri1k = random.randint(1, 30, n1k)
        self._nsuc1k = [random.randint(0, i) for i in self._ntri1k]

    def time_qep_binomial_1k_learn(self):
        glmm = GLMM((self._nsuc1k, self._ntri1k), 'binomial', self._X1k,
                    self._QS1k)
        glmm.feed().maximize(progress=False)

        glmm = GLMM((nsuc1k, ntri1k), 'binomial', X1k,
                    QS1k)
        glmm.feed().maximize(progress=False)
        print(".12f" % glmm.feed().value())
        assert_allclose(glmm.feed().value(), -2611.6160207784023)
