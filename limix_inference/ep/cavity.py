from __future__ import division

from numpy import zeros

class Cavity(object):

    def __init__(self, n):
        self.tau = zeros(n)
        self.eta = zeros(n)

    def update(self, jtau, jeta, ttau, teta):
        self.tau[:] = jtau - ttau
        self.eta[:] = jeta - teta
