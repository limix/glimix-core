import pytest

from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs

from limix_inference.example import linear_eye_cov
from limix_inference.glmm import GLMM

from optimix import check_grad

def test_glmm():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)
    QS = (QS[0][0], QS[1])

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.value(), -272.1213895386019)
    assert_allclose(check_grad(glmm), 0, atol=1e-4)

def test_glmm_wrong_qs():
    random = RandomState(0)
    X = random.randn(10, 15)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    ntri = random.randint(1, 30, 10)
    nsuc = [random.randint(0, i) for i in ntri]

    with pytest.raises(ValueError):
        glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

def test_glmm_optimize():
    random = RandomState(0)
    X = random.randn(100, 5)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)
    QS = (QS[0][0], QS[1])

    ntri = random.randint(1, 30, 100)
    nsuc = [random.randint(0, i) for i in ntri]

    glmm = GLMM((nsuc, ntri), 'binomial', X, QS)

    assert_allclose(glmm.value(), -272.1213895386019)
    glmm.fix('beta')
    glmm.fix('logscale')

    glmm.feed().maximize(progress=False)
    assert_allclose(glmm.value(), -271.367864630782)

    glmm.unfix('beta')
    glmm.unfix('logscale')

    glmm.feed().maximize(progress=False)
    assert_allclose(glmm.value(), -266.9517518211878)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
