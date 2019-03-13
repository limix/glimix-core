from numpy import maximum, zeros


class Site(object):
    """EP parameters."""

    def __init__(self, n):
        self.tau = zeros(n)
        self.eta = zeros(n)

    def update(self, mean, variance, cav):
        self.tau[:] = maximum(1.0 / variance - cav["tau"], 0)
        self.eta[:] = mean / variance - cav["eta"]
