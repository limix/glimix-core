from numpy import exp
from numpy import log

from optimix import Function
from optimix import Scalar


class EyeCov(Function):
    def __init__(self):
        Function.__init__(self, logscale=Scalar(0.0))

    @property
    def scale(self):
        return exp(self.get('logscale'))

    @scale.setter
    def scale(self, scale):
        self.set('logscale', log(scale))

    def value(self, x0, x1):
        return self.scale * (x0 == x1)

    def derivative_logscale(self, x0, x1):
        return self.value(x0, x1)
