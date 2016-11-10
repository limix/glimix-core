from numpy import zeros
from numpy import ascontiguousarray

from optimix import Function
from optimix import Vector


class LinearMean(Function):
    def __init__(self, size):
        Function.__init__(self, effsizes=Vector(zeros(size)))

    def value(self, x):
        return x.dot(self.get('effsizes'))

    def derivative_effsizes(self, x):
        return x

    @property
    def effsizes(self):
        return self.get('effsizes')

    @effsizes.setter
    def effsizes(self, v):
        self.set('effsizes', ascontiguousarray(v))
