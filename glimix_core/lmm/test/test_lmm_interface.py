from numpy import inf, nan, newaxis
from numpy.random import RandomState
from numpy.testing import assert_allclose

import pytest
from glimix_core.lmm import LMM
from numpy_sugar.linalg import economic_qs_linear

