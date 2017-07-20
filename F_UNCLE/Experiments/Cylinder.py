"""Toy problem of a cylinder tests. The cylinder is modelled as an
expanding ring

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (06-06-2016)

ToDo
----

None

"""

# =========================
# Python Standard Libraries
# =========================

import sys
import os
import pdb
import copy
import unittest

# =========================
# Python Packages
# =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline
import scipy.integrate as spint
# =========================
# Custom Packages
# =========================
from ..Utils.Simulation import Simulation
from ..Utils.Experiment import GaussianExperiment
from ..Utils.NoiseModel import NoiseModel
from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
from ..Models.SimpleStr import SimpleStr
from .Sandwich import ToySandwich
# =========================
# Main Code
# =========================


class ToyCylinder(ToySandwich):
    """A toy physics model representing a Cylinder experiment
    
    Based on the ToySandwich class 

    Attributes:
        const(dict): A dictionary of conversion factors

    """
    def __init__(self, name='Gun Toy Computational Experiment', *args, **kwargs):
        """Instantiate the Experiment object

        Args:
            eos(Isentrope): The equation of state model used in the toy computational
                            experiment

        Keyword Args:
            name(str): A name. (Default = 'Gun Toy Computational Experiment')

        """

        def_opts = {
            'r_i': [float, 2.54, 0.0, None, 'cm',
                    'Initial HE radius'],
            'r_o': [float, 5.0, 0.0, None, 'cm',
                    'Maximum expansion of the cylinder'],
            'case_t': [float, 0.254, 0.0, None, 'cm',
                    'Maximum expansion of the cylinder'],
            'rho_cu': [float, 8.9, 0.0, None, 'g cm-3',
                       'Ring density'],
            'rho_cj': [float, 2.8, 0.0, None, 'g cm-3',
                       'The CJ density of the reactants'],                 
            't_min': [float, 1.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            't_max': [float, 25e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            'n_t': [int, 250, 0, None, '',
                    'Number of times for t2v spline']
        }

        Simulation.__init__(self, {'eos': Isentrope, 'strength': SimpleStr},
                            name=name, def_opts=def_opts, *args, **kwargs)

    def _on_check_models(self, models):
        """Checks that the model is valid

        Args:
            model(Isentrope): A new EOS model

        Return:
            (GunModel): A copy of self with the new eos model
        """
        return (models['eos'], models['strength'],)
        
    def _on_call(self, eos, strength):
        """Solves the simulation 
        """

        def dfunc(x_in, tho, ro, rho_cj, rho_cu):
            """Solves the ODE representing the cylinder case velocity

            Args:
                x(list): The state vector

                   -x[0] is the position
                   -x[1] is the velocity
                   -x[2] is the thickness

            Keyword Args:
                p(dict): Parameter dictionary

            Return:
               (list): The derrivative of the state vector

                  -[0] is the velocity
                  -[1] is the acceleration
                  -[2] is the rate of change of thickness

            """
            rad = x_in[0]  # Case radius
            drad = x_in[1]  # Rate of change of case radius
            th = x_in[2]  # Case thickness

            v_spec = rho_cj**-1 * (rad/ro)**2
            epsilon = (rad - ro) / ro

            stress = strength(epsilon, drad / ro)

            ddrad = 0.1 * (rho_cu * th)**-1 * eos(v_spec)\
                - 0.1 * (rad * rho_cu)**-1 * stress  # cm s**-2

            dth = - drad * ( th / rad)  # cm s**-1

            return [drad, ddrad, dth]


        tho = self.get_option('case_t')
        ro = self.get_option('r_i')
        rho_cj = self.get_option('rho_cj')
        rho_cu = self.get_option('rho_cu')

        t_list = np.linspace(1E-7, self.get_option('t_max'),
                             self.get_option('n_t'))
        data, infodict = spint.odeint(
            dfunc,
            [self.get_option('r_i'), 1E-9, tho],
            t_list,
            args = (ro, rho_cj, rho_cu),                                
            full_output=True
        )

        return t_list,\
            [data[:, 1], data[:, 0], data[:, 2]],\
            {'mean_fn': IUSpline(t_list, data[:,1])}
        

    def _on_str(self, *args, **kwargs):
        """Print method of the gun model

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (str): A string representing the object
        """

        out_str = '\n'
        out_str += 'Equation of State Model\n'
        out_str += '-----------------------\n'
        out_str += str(self.eos)

        return out_str

class ToyCylinderExperiment(GaussianExperiment):
    """A class representing pseudo experimental data for a gun show
    """
    def _get_data(self, models=None, rstate=None, *args, **kwargs):
        """Creates a simulated set of experimental data from a user provided
        model

        Args:
           models(dict): A dictionary of models with all the keys that
                         ToyCylinder needs
           rstate(np.random.RandomState): A random state, None generates a
                                          new one
        """

        sim = ToyCylinder(**kwargs)

        simdata = sim(models)

        noise_model = NoiseModel()
        noisedata = noise_model(simdata[0], simdata[1][0], rstate=rstate)
        
        return simdata[0], simdata[1][0] + noisedata,\
            np.zeros(simdata[0].shape)
    # end
        
    def get_sigma(self):
        """Returns the co-variance matrix

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`
        """

        return np.diag(np.ones(self.shape())
                       * self.data[1] * 0.05)
    
