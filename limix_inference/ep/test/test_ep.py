import pytest

from limix_inference.ep import EP

def test_ep():
    ep = EP()
    with pytest.raises(NotImplementedError):
        print(ep._compute_moments()) # pylint: disable=W0212
