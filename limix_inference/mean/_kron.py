from numpy import add


class KronSumMean(object):
    def __init__(self, A, B):
        self._A = A
        self._B = B

    def value_reduce(self, values):
        return None

    def derivative_reduce(self, derivatives):
        return None
