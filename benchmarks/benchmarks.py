from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import ascontiguousarray, sqrt, ones
from numpy_sugar.linalg import economic_qs, economic_qs_linear

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMM
from glimix_core.random import bernoulli_sample

class TimeSuite:
    def setup(self):
        random = RandomState(0)
        self._X = random.randn(100, 5)
        self._K = linear_eye_cov().feed().value()
        self._QS = economic_qs(K)

        self._ntri = random.randint(1, 30, 500)
        self._nsuc = [random.randint(0, i) for i in self._ntri]

    def time_qep_binomial_lml_no_learn(self):
        glmm = GLMM((self._nsuc, self._ntri), 'binomial', self._X, self._QS)
        assert_allclose(glmm.value(), -272.1213895386019)

    def time_qep_binomial_lml_learn(self):
        glmm = GLMM((self._nsuc, self._ntri), 'binomial', self._X, self._QS)

        assert_allclose(glmm.value(), -272.1213895386019)
        glmm.fix('beta')
        glmm.fix('scale')

        glmm.feed().maximize(progress=False)
        assert_allclose(glmm.value(), -271.367864630782)

        glmm.unfix('beta')
        glmm.unfix('scale')

        glmm.feed().maximize(progress=False)
        assert_allclose(glmm.value(), -266.9517518211878)
