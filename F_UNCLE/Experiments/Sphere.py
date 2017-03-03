"""

Stick: A simplified model of a rate stick

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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# =========================
# Python Standard Libraries
# =========================

import sys
import os
import pdb
import copy
import unittest
from math import pi
import math
# =========================
# Python Packages
# =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =========================
# Custom Packages
# =========================
from ..Utils.Simulation import Simulation
from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
from ..Models.Ptw import Ptw
# if __name__ == '__main__':
#     sys.path.append(os.path.abspath('./../../'))
#     from F_UNCLE.Utils.Experiment import Simulation
#     from F_UNCLE.Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
#     from F_UNCLE.Models.Ptw import Ptw
# else:
#     from ..Utils.Experiment import Simulation
#     from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
#     from ..Models.Ptw import Ptw
# # end


# =========================
# Main Code
# =========================


class Sphere(Simulation):
    """A toy physics model representing an expanding plastic sphere driven by HE


    Attributes:
        const(dict): A dictionary of conversion factors

    """
    def __init__(self, name='Sphere compuational experiment', *args, **kwargs):
        """Instantiate the Experiment object

        Keyword Args:
            name(str): A name. (Default = 'Sphere computational experiment')

        """

        def_opts = {
            'ri': [float, 1, 0.0, None, 'cm',
                   'Initial inner radius of spherical case'],
            'case_t': [float, 0.1, 0.0, None, 'cm',
                       'Case thickness'],
            'case_mat': [str, 'Cu', None, None, '-',
                         'Valid material specifier for case material'],
            'mass_he': [float, 16.5, 0.0, None, 'g',
                        'The initial mass of high explosives used to drive'
                        'the projectile'],
            'temp': [float, 1000, 0.0, 1E4, 'K',
                     'The temperatire of the case during the expansion'],
            't_sim': [float, 1.0E-6, 0.0, None, 's',
                      'Time to simulate explosion'],
            'sigma': [float, 1.0, 0.0, None, '??',
                      'Variance attributed to v measurements'],
            'n_t': [int, 250, 0, None, '',
                    'Number of time steps']
        }

        GausianExperiment.__init__(self, {'eos': Isentrope, 'strength': Ptw},
                                   name=name, def_opts=def_opts,
                                   *args, **kwargs)

    def _on_check_models(self, models):
        """Checks that the model is valid

        Args:
            model(Isentrope): A new EOS model

        Return:
            (GunModel): A copy of self with the new eos model
        """
        return models['eos'], models['strength']

    def _on_call(self, eos, strength):
        """Solves the sphere problem

        Args:
            eos(Isentrope): The Equation of strength model
            strength(Ptw): The material strength model
        Return:
            (tuple):

               [0] - times
               [1](list): Data at each timestep

                  [0] - velocity of the case
                  [1] - radius of the case
                  [2] - case thickness
                  [3] - specific volume of gasses within case
                  [4] - pressure of gasses within case
                  [5] - stress in the case
                  [6] - strain in the case
        """

        def dfunc(x_in, param=None):
            """Solves the ODE representing the case velocity

            Args:
                x(list): The state vector

                   -x[0] is the position
                   -x[1] is the velocity
                   -x[2] is the thickness

            Keyword Args:
                p(dict): Parameter dictionart

            Return:
               (list): The derrivative of the state vector

                  -[0] is the velocity
                  -[1] is the acceleration
                  -[2] is the rate of change of thickness

            """
            tho = self.get_option('case_t')  # Initial case thickness
            temp = self.get_option('temp')  # Temperature of case
            ro = self.get_option('ri')  # Initial case radius
            mass_he = self.get_option('mass_he')  # Mass of High Explosive
            mat = self.get_option('case_mat')

            rad = x_in[0]  # Case radius
            drad = x_in[1]  # Rate of change of case radius
            th = x_in[2]  # Case thickness

            v_spec = (4.0 / 3.0) * np.pi * rad**3 / mass_he
            epsilon = (rad - ro) / ro

            stress = strength.get_stress(epsilon, drad / ro, temp, mat)

            rho = strength.get_option('rho') * 1E-6  # kg cm**-3

            m_o = rho * 4 * np.pi * ro**2 * tho  # kg

            ddrad = (rho * th)**-1 * 1E-2 * eos(v_spec)\
                - 2E-2 / (rad * rho) * stress  # cm s**-2

            dth = -m_o * drad / (rho * 2.0 * np.pi * rad**3)  # cm s**-1

            return [drad, ddrad, dth]

        temp = self.get_option('temp')
        tho = self.get_option('case_t')
        ro = self.get_option('ri')
        mass_he = self.get_option('mass_he')
        mat = self.get_option('case_mat')

        t_list = np.linspace(0, self.get_option('t_sim'),
                             self.get_option('n_t'))
        data, infodict = odeint(dfunc,
                                [self.get_option('ri'), 0, tho],
                                t_list,
                                full_output=True)

        pres = []
        stress = []
        strain = []
        v_spec = []
        for i in range(len(data)):
            rad = data[i, 0]
            drad = data[i, 1]
            v_spec.append((4.0 / 3.0) * np.pi * rad**3 / mass_he)
            pres.append(eos(v_spec[-1]))
            strain.append((rad - ro) / ro)
            stress.append(strength.get_stress(strain[-1],
                                              drad / ro,
                                              temp,
                                              mat))
        # end

        return t_list,\
            (data[:, 1], data[:, 0], data[:, 2], v_spec, pres, stress, strain),\
            None

    def compare(self, indep, dep, model_data):
        """Compares a set of experimental data to the model

        Error is `dep` less the `model_data`

        See :py:meth:`F_UNCLE.Utils.Experiment.Experiment.compare`

        """

        spline = Spline(model_data[0], model_data[1][0])
        return dep - spline(indep)
    # end

    def get_sigma(self, eos, *args, **kwargs):
        """Gets the co-variance matrix of the experiment

        Args:
            eos(GaussianModel): The model under investigation

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`

        """

        return np.diag(np.ones(self.shape()) * self.get_option('sigma')**2)

    def shape(self, *args, **kwargs):
        """Gives the length of the independent variable vector

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.shape`

        """

        return self.get_option('n_t')

    def plot(self, data, axes=None, fig=None, linestyles=['-k'], labels=[]):
        """Plots the object

        Args:
            data(tuple): The output from a call to a Sphere object

        Keyword Args:
            axes(plt.Axes): The axes on which to plot *Ignored*
            fig(plt.Figure): The figure on which to plot
            linestyles(list): Strings for the linestyles
            labels(list): Strings for the labels

        Return:
            (plt.Figure): A reference to the figure containing the plot

        """

        if fig is None:
            fig = plt.figure()
        else:
            pass
        # end

        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

        ax1.plot(data[0], data[1][1], linestyles[0])
        ax1.set_xlabel('Time from detonation / s')
        ax1.set_ylabel('Radius of sphere / cm')

        ax2.plot(data[0], data[1][0])
        ax2.set_ylabel(r'Velocity of sphere / cm s$^{-1}$')
        ax3.plot(data[0], data[1][3])
        ax3.set_ylabel(r'Specific volume / cm$^{3}$ g$^{-1}$')
        ax4.plot(data[0], data[1][4])
        ax4.plot(data[0], np.array(data[1][5]) * 1E3, '-k')
        ax4.set_ylabel(r'Pressure / Pa')
        ax5.plot(data[0], data[1][2])
        ax5.set_ylabel(r'Thickness / cm')
        ax6.plot(data[0], data[1][6])
        ax6.set_ylabel(r'Strain in material / cm cm$^{-1}$')

        return fig

    def update(self, model=None, strength=None):
        """Update the analysis with a new EOS model

        Args:
            model(Isentrope): A new EOS model

        Return:
            (GunModel): A copy of self with the new eos model
        """
        if model is None:
            model = self.eos
        elif isinstance(model, Isentrope):
            model = model
        else:
            raise TypeError('{:} Model must be an isentrope for update'
                            .format(self.get_inform(1)))
        # end

        if strength is None:
            strength = self.strength
        elif isinstacne(strength, Ptw):
            pass
        else:
            raise TypeError('{:} Strength model must be a Ptw model instance'
                            .format(self.get_inform(1)))
        # end

        new_sim = copy.deepcopy(self)
        new_sim.eos = model
        new_sim.strength = strength

        return new_sim

    def _on_str(self, *args, **kwargs):
        """Print method of the gun model

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (str): A string representing the object
        """

        out_str = ''

        return out_str
