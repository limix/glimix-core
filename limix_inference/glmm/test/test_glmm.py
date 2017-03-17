import numpy as np
from numpy.testing import assert_allclose

from numpy_sugar.linalg import economic_qs

from limix_inference.glmm import GLMM
from limix_inference.example import offset_mean, linear_eye_cov



def test_glmm():
    random = np.random.RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)
    QS = (QS[0][0], QS[1])

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    step = 1e-7

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    f0 = glmm.value()
    glmm.set('logitdelta', glmm.get('logitdelta') + step)
    f1 = glmm.value()
    assert_allclose(glmm.gradient()[0], (f1 - f0)/step, rtol=1e-4)

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    f0 = glmm.value()
    glmm.set('logscale', glmm.get('logscale') + step)
    f1 = glmm.value()
    assert_allclose(glmm.gradient()[1], (f1 - f0)/step, rtol=1e-4)

    for i in range(5):
        glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
        f0 = glmm.value()
        beta = glmm.get('beta')
        beta[i] += step
        glmm.set('beta', beta)
        f1 = glmm.value()
        assert_allclose(glmm.gradient()[2][i], (f1 - f0)/step, rtol=1e-4)
