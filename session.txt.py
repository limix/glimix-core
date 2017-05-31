from numpy.random import RandomState
from numpy import dot
from numpy import sqrt

import numpy as np

X0 = np.random.randn(5000, 5000)

X1 = np.random.randn(5000, 5)

K0 = X0.dot(X0.T)

K1 = X1.dot(X1.T)

from numpy import dot, sqrt, zeros

from numpy.random import RandomState

from numpy_sugar.linalg import economic_qs

from glimix_core.glmm import GLMM

random = RandomState(0)
ntrials = random.randint(1, 100, 5000)

z0 = dot(X0, random.randn(5000)) / sqrt(5000)
z1 = dot(X1, random.randn(5)) / sqrt(5)

successes0 = zeros(len(ntrials), int)
successes1 = zeros(len(ntrials), int)
for i in range(len(ntrials)):
    successes0[i] = sum(z0[i] + 0.2 * random.randn(ntrials[i]) > 0)
    
for i in range(len(ntrials)):
    successes1[i] = sum(z1[i] + 0.2 * random.randn(ntrials[i]) > 0)
    
y0 = (successes0, ntrials)
y1 = (successes1, ntrials)
QS0 = economic_qs(K0)
QS1 = economic_qs(K1)
from numpy import ones

def func0():
    glmm0 = GLMM(y0, 'binomial', ones((5000, 1)), QS0)
    glmm0.feed().maximize(progress=False)

def func1():
    glmm1 = GLMM(y1, 'binomial', ones((5000, 1)), QS1)
    glmm1.feed().maximize(progress=False)

get_ipython().magic('timeit func0()')
get_ipython().magic('timeit func1()')
