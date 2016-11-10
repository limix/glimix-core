from numpy import exp
from numpy import log

from optimix import Scalar
from optimix import Function


class LinearCov(Function):
    def __init__(self):
        Function.__init__(self, logscale=Scalar(0.0))

    @property
    def scale(self):
        return exp(self.get('logscale'))

    @scale.setter
    def scale(self, scale):
        self.set('logscale', log(scale))

    def value(self, x0, x1):
        return self.scale * x0.dot(x1.T)

    def derivative_logscale(self, x0, x1):
        return self.scale * x0.dot(x1.T)
