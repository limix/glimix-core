from time import time

from numpy import array, concatenate, eye, kron
from numpy.linalg import slogdet
from numpy.random import RandomState
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

from glimix_core.cov import Kron2SumCov
from glimix_core.lmm import RKron2Sum

random = RandomState(0)
n = 150
# G = random.randn(n, 5)

# cov = Kron2SumCov(G, 2, 1)
# cov.C0.Lu = abs(random.randn(2))

# start = time()
# cov.gradient()
# print("Elapsed: {}s".format(time() - start))

Y = random.randn(n, 2)
A = random.randn(2, 2)
A = A @ A.T
F = random.randn(n, 4)
G = random.randn(n, 5)
lmm = RKron2Sum(Y, A, F, G)

start = time()
# breakpoint()
# lmm.cov.logdet()
lmm.cov.logdet_gradient()
# lmm.lml_gradient()
print("Elapsed: {}s".format(time() - start))

# I = eye(G.shape[0])
# assert_allclose(cov._check_grad(), 0, atol=1e-5)
# assert_allclose(cov.solve(cov.value()), eye(2 * G.shape[0]), atol=1e-7)
# assert_allclose(cov.logdet(), slogdet(cov.value())[1], atol=1e-7)
# assert_allclose(
#     [cov.L[0, 0], cov.L[2, 3], cov.L[2, 1]],
#     [0.23093921294934955, -5.2536114062217535e-17, 0.2828416166629259],
# )


# def func(x):
#     cov.C0.Lu = x[:2]
#     cov.Cn.L0 = x[2:3]
#     cov.Cn.L1 = x[3:]
#     return cov.logdet()


# def grad(x):
#     cov.C0.Lu = x[:2]
#     cov.Cn.L0 = x[2:3]
#     cov.Cn.L1 = x[3:]
#     D = cov.logdet_gradient()
#     return concatenate((D["C0.Lu"], D["Cn.L0"], D["Cn.L1"]))


# random = RandomState(0)
# assert_allclose(check_grad(func, grad, random.randn(5)), 0, atol=1e-5)
