"""Script for re-generating the notes figures

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# Standard python packages
import sys
import os
import pdb
import argparse
import time
# External python packages
import numpy as np
import matplotlib.pyplot as plt

# F_UNLCE packages
sys.path.append(os.path.abspath('./../../../fit_9501'))    
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Experiments.GunModel import Gun, GunExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Experiments.Sphere import Sphere
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from fit_9501.Models.equation_of_state import AugmentedIsentrope
from F_UNCLE.Opt.Bayesian import Bayesian
from F_UNCLE.Models.Ptw import Ptw

#################
#    Get Data   #
#################

# 1. Generate a functional form for the prior
# Prior is the JWL isentrope. Values are from [TarverManganin2001]

vol_model = AugmentedIsentrope(
    lambda v: 2.56E9 / v**3,
    rho_0=1.844,
    pres_0=101325,
    spline_sigma=0.05,
    basis='vol',
    spacing='lin'
    )

eos_model = AugmentedIsentrope(
    lambda r: 2.56E9 * r**3,
    spline_min=1.0,
    spline_max=4.0,
    cj_vol_range=(3.5**-1, 1.7**-1),
    rho_0=1.844,
    pres_0=101325,
    spline_sigma=0.05,
    basis='dens',
    spacing='lin'
)

n_end = eos_model.get_option('spline_end')
rho_unique=eos_model.get_t()[n_end - 1 : 1 - n_end]

tmp_g = np.empty((len(rho_unique), eos_model.shape()))

fig = plt.figure()
ax1 = fig.gca()
for i in xrange(eos_model.shape()):
    tmp_coeff = np.zeros(eos_model.shape())
    tmp_coeff[i] = 1.0
    new_model = eos_model.set_c(tmp_coeff)
    # response = new_model.derivative(2)(rho_unique)
    response = -1 * rho_unique**3 * (new_model.derivative(2)(rho_unique) * rho_unique**1
                  + 2 * rho_unique**0 * new_model.derivative(1)(rho_unique))
    tmp_g[:,i] = response
    ax1.plot(rho_unique, response, label="%d"%i)    
# end

h = np.dot(tmp_g, eos_model.get_dof())
scalemat = np.diag(np.fabs(h)**-1)

ax1.plot(rho_unique, np.zeros(rho_unique.shape[0]), 'xk')
ax1.legend(loc='best', title="Knot ID")
ax1.set_xlabel('Density / g cm-3')
ax1.set_xlabel('Convexity Basis Function per unit pressure /')
fig2 = plt.figure()
ax1 = fig2.gca()
response = -1 * rho_unique**3 * (eos_model.derivative(2)(rho_unique) * rho_unique**1
              + 2 * rho_unique**0 * eos_model.derivative(1)(rho_unique))

ax1.plot(rho_unique, response, '--',
         label =r"$\frac{\partial P}{\partial v}$ computed by density based spline")
ax1.plot(rho_unique, -vol_model.derivative(2)(rho_unique**-1), ':',
         label=r"$\frac{\partial P}{\partial v}$ computed by volume based spline")
ax1.plot(rho_unique, np.dot(tmp_g, eos_model.get_dof()), '-.',
         label=r'$G \cdot \Theta$')
print(np.dot(scalemat, tmp_g))
ax1.set_xlabel('Density / g cm-3')
ax1.set_xlabel('Convexity / Pa cm6 g-2')
ax1.legend(loc='best')

fig3 = plt.figure()
ax1=fig3.gca()
ax1.plot(rho_unique, np.dot(np.dot(scalemat, tmp_g), eos_model.get_dof()))
ax1.set_xlabel('Density / g cm-3')
ax1.set_xlabel('Convexity / Pa cm6 g-2')
ax1.legend(loc='best')

plt.show()
    
