from numpy.testing import assert_allclose

from limix_inference.link import ProbitLink


def test_probit_link():
    lik = ProbitLink()
    assert_allclose(lik.value(lik.inv(3.2)), 3.2)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
