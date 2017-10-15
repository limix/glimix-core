import pytest
from numpy.testing import assert_allclose

from glimix_core.link import LogitLink, LogLink, ProbitLink
from glimix_core.link.link import Link


def test_probit_link():
    link = ProbitLink()
    assert_allclose(link.value(link.inv(3.2)), 3.2)
    assert_allclose(link.latent_variance, 1.0)


def test_logit_link():
    link = LogitLink()
    assert_allclose(link.value(link.inv(3.2)), 3.2)
    assert_allclose(link.latent_variance, 3.289868133696453)


def test_loglink_link():
    link = LogLink()
    assert_allclose(link.value(link.inv(3.2)), 3.2)
    with pytest.raises(NotImplementedError):
        link.latent_variance


def test_link_interface():
    link = Link()

    with pytest.raises(NotImplementedError):
        link.value(None)

    with pytest.raises(NotImplementedError):
        link.inv(None)

    with pytest.raises(NotImplementedError):
        link.latent_variance
