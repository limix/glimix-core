from numpy.testing import assert_allclose

from glimix_core.link import LogitLink, LogLink, ProbitLink


def test_probit_link():
    link = ProbitLink()
    assert_allclose(link.value(link.inv(3.2)), 3.2)


def test_logit_link():
    link = LogitLink()
    assert_allclose(link.value(link.inv(3.2)), 3.2)


def test_loglink_link():
    link = LogLink()
    assert_allclose(link.value(link.inv(3.2)), 3.2)
