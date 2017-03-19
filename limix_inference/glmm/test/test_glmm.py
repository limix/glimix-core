from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs

from limix_inference.example import linear_eye_cov
from limix_inference.glmm import GLMM


def test_glmm():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)
    QS = (QS[0][0], QS[1])

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    step = 1e-7

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    f0 = glmm.value()
    g0 = glmm.gradient()[0]
    glmm.set('logitdelta', glmm.get('logitdelta') + step)
    f1 = glmm.value()
    assert_allclose(g0, (f1 - f0) / step, rtol=1e-4)

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
    f0 = glmm.value()
    g0 = glmm.gradient()[1]
    glmm.set('logscale', glmm.get('logscale') + step)
    f1 = glmm.value()
    assert_allclose(g0, (f1 - f0) / step, rtol=1e-4)

    for i in range(5):
        glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
        f0 = glmm.value()
        g0 = glmm.gradient()[2][i]
        beta = glmm.get('beta')
        beta[i] += step
        glmm.set('beta', beta)
        f1 = glmm.value()
        assert_allclose(g0, (f1 - f0) / step, rtol=1e-4)

# def test_glmm_optimize():
#     random = RandomState(0)
#     X = random.randn(100, 5)
#     K = linear_eye_cov().feed().value()
#     QS = economic_qs(K)
#     QS = (QS[0][0], QS[1])
#
#     ntri = random.randint(1, 30, 100)
#     nsuc = [random.randint(0, i) for i in ntri]
#
#     glmm = GLMM((nsuc, ntri), 'binomial', X, QS)
#     # f0 = glmm.value()
#     glmm.fix('beta')
#     glmm.fix('logscale')
#     from numpy import linspace
#     glmm.feed().maximize()
#     # for s in linspace(0.1, 10):
#     #     glmm.scale = s
#     #     print("%f %f %f" % (s, glmm.feed().value(), glmm.feed().gradient()[0]))
