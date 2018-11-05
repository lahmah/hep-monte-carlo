import numpy as np
from scipy.special import logit, expit
#from .mapping import PhaseSpaceMapping
from ..density import Density

#class Logit(PhaseSpaceMapping):
#    def __init__(self, ndim):
#        super().__init__(ndim)
#
#    def pdf(self, xs):
#        return 1.
#
#    def pdf_gradient(self, xs):
#        return 0.
#
#    def map(self, xs):
#        return expit(xs)
#
#    def map_inverse(self, xs):
#        return logit(xs)

class LogitDensity(Density):
    def __init__(self, density):
        super().__init__(density.ndim)
        self.density = density

    def pdf(self, xs):
        ps = expit(xs)
        pdf = self.density.pdf(ps)
        return pdf

    def pot(self, xs):
        ps = expit(xs)
        pot = self.density.pot(ps)
        return pot

    #def pot_gradient(self, xs):
    #    ps = expit(xs)
    #    grad = self.density.pot_gradient(ps)
    #    return grad
