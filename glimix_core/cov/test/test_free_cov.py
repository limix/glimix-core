from __future__ import division

from numpy import ones, zeros

from glimix_core.cov import FreeFormCov
from optimix import Assertion


def test_freeform_optimix():
    item0 = 0
    item1 = 1

    cov = FreeFormCov(2)
    cov.L = [[1, 0], [2, 1]]

    a = Assertion(lambda: cov, item0, item1, 0.0, Llow=ones(2), Llogd=zeros(1))

    a.assert_layout()
    # a.assert_gradient()

    # cov.variables().get("Llow").value = [2]
    #  TODO: activate it back
    # cov.variables().get("Llogd").value = [log(0.5), log(1.5)]
    # assert_allclose(cov.L, [[0.5, 0], [2, 1.5]])

    # cov.L = [[3, 0], [2, 1]]
    # assert_allclose(cov.L, [[3, 0], [2, 1]])
