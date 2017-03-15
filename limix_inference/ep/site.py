from __future__ import division

from numpy import zeros, maximum

class Site(object):
    def __init__(self, n):
        self.tau = zeros(n)
        self.eta = zeros(n)

    def update(self, mean, variance, eta, tau):
        self.tau[:] = maximum(1.0 / variance - tau, 0)
        self.eta[:] = mean / variance - eta
