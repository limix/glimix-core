from numpy import dot
from numpy.random import RandomState
import logging

logging.basicConfig(level=logging.DEBUG)

from numpy_sugar.linalg import economic_qs

from glimix_core.glmm import GLMM

random = RandomState(0)
nsamples = 50

X = random.randn(50, 2)
G = random.randn(50, 100)
K = dot(G, G.T)
ntrials = random.randint(1, 100, nsamples)
successes = [random.randint(0, i + 1) for i in ntrials]

y = (successes, ntrials)

QS = economic_qs(K)
glmm = GLMM(y, 'binomial', X, QS)
print('Before: %.4f' % glmm.feed().value())
glmm.feed().maximize(progress=False)
print('After: %.2f' % glmm.feed().value())
