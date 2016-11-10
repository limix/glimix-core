from numpy import full
from numpy import ones

from optimix import Function
from optimix import Scalar


class OffsetMean(Function):
    def __init__(self):
        Function.__init__(self, offset=Scalar(1.0))

    def value(self, size):
        return full(size, self.get('offset'))

    def derivative_offset(self, size):
        return ones(size)

    @property
    def offset(self):
        return self.get('offset')

    @offset.setter
    def offset(self, v):
        self.set('offset', v)
