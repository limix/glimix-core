from numpy import add

from optimix import FunctionReduce


class SumCov(FunctionReduce):
    def __init__(self, covariances):
        self._covariances = [c for c in covariances]
        FunctionReduce.__init__(self, self._covariances, 'sum')

    def value_reduce(self, values):
        return add.reduce(values)

    def derivative_reduce(self, derivatives):
        return add.reduce(derivatives)
