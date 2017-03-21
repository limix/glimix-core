from limix_inference.mean import LinearMean
from optimix.testing import Assertion


def test_offsetmean_optimix():
    item0 = [5.1, 1.0]
    item1 = [2.1, -0.2]

    a = Assertion(
        lambda: LinearMean(2), item0, item1, 0.0, effsizes=[0.5, 1.0])
    a.assert_layout()
    a.assert_gradient()


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
