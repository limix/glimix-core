from numpy import eye
from collections import Sequence


class Fruits(object):
    __slots__ = ['shape']

    def __init__(self, shape=None):
        if shape is None or isinstance(shape, Sequence):
            self.shape = shape
        else:
            self.shape = (shape, )

    def __eq__(self, that):
        assert self.ndim == that.ndim
        c = 1.0 * (isinstance(self, type(that)))
        if self.ndim == 0:
            return c
        elif self.ndim == 1:
            return c * eye(self.shape[0], that.shape[0])
        assert False

    @property
    def ndim(self):
        if self.shape is None:
            return 0
        return len(self.shape)


class Apples(Fruits):
    def __init__(self, shape=None):
        super(Apples, self).__init__(shape)


class Oranges(Fruits):
    def __init__(self, shape=None):
        super(Oranges, self).__init__(shape)
