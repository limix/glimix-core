r"""
****************
Gaussian Process
****************

Introduction
^^^^^^^^^^^^

A GP is a statistical distribution :math:`Y_t`, :math:`t\in\mathrm T`, for
which any finite linear combination of samples has a joint Gaussian
distribution.
An instance of such class of processes is defined via a mean function
:math:`m(\cdot)` and a covariance function :math:`k(\cdot, \cdot)` whose
domains are :math:`\mathrm T` and :math:`\mathrm T\times\mathrm T`,
respectively.

Class :class:`.GP` performs inference over the mean and covariance parameters
by maximum likelihood.


Public interface
^^^^^^^^^^^^^^^^
"""
from .gp import GP

__all__ = ['GP']
