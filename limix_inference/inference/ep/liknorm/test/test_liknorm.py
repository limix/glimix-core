from __future__ import division

from numpy import array, empty, set_printoptions
from numpy.testing import assert_allclose

from lim.inference.ep.liknorm import create_liknorm
from lim.genetics.phenotype import BinomialPhenotype, PoissonPhenotype


def test_liknorm_binomial():
    set_printoptions(precision=16)

    ln = create_liknorm('Binomial', 500)

    N = array([3, 7, 1, 98], float)
    K = array([2, 0, 1, 66], float)

    tau = array([0.1, 0.9, 1.3, 1.1])
    eta = array([-1.2, 0.0, +0.9, -0.1]) * tau

    log_zeroth = empty(4)
    mean = empty(4)
    variance = empty(4)

    ln.moments(BinomialPhenotype(K, N), eta, tau, log_zeroth, mean, variance)

    assert_allclose([
        -1.9523944026844868, -2.8039404963019825, -0.3788565926515431,
        -4.3286813960067958
    ], log_zeroth)
    assert_allclose([
        0.5808253224545352, -1.5435498588501912, 1.1109257189367012,
        0.6910369705990104
    ], mean)
    assert_allclose([
        1.5854044107023924, 0.5286947014880381, 0.6775522066257242,
        0.0440818199239704
    ], variance)

    ln.destroy()


def test_liknorm_poisson():
    set_printoptions(precision=16)

    ln = create_liknorm('Poisson', 500)

    k = array([2, 0, 1, 66], float)

    tau = array([0.1, 0.9, 1.3, 1.1])
    eta = array([-1.2, 0.0, +0.9, -0.1]) * tau

    log_zeroth = empty(4)
    mean = empty(4)
    variance = empty(4)

    ln.moments(PoissonPhenotype(k), eta, tau, log_zeroth, mean, variance)

    assert_allclose([
        -2.9206093278791672, -0.9585584570898347, -1.649214737245454,
        -14.9841482839478264
    ], log_zeroth)
    assert_allclose([
        0.3401321321980286, -0.7306833253132138, 0.3637351533499351,
        4.108900854827759
    ], mean)
    assert_allclose([
        0.6425143694408278, 0.6735068928338729, 0.3500115006759242,
        0.0161318528782672
    ], variance)

    ln.destroy()


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
