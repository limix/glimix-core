import pytest
import scipy.stats as st
from numpy import concatenate
from numpy.random import RandomState
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

from glimix_core.lmm import Kron2Sum
