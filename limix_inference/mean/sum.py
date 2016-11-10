from numpy import add

from optimix import FunctionReduce


class SumMean(FunctionReduce):
    def __init__(self, means):
        self._means = [c for c in means]
        FunctionReduce.__init__(self, self._means, 'sum')

    def value_reduce(self, values):
        return add.reduce(values)

    def derivative_reduce(self, derivatives):
        return add.reduce(derivatives)
