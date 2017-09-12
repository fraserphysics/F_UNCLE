"""

Sandwich

Toy computational experiment of a simplified sandwich test

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (03-16-2016)

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
import copy
import unittest
import sys
import os
# =========================
# Python Packages
# =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =========================
# Custom Packages
# =========================
from ..Utils.Experiment import GaussianExperiment
from ..Utils.Simulation import Simulation
from ..Utils.NoiseModel import NoiseModel
from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline

# =========================
# Main Code
# =========================


class ToySandwich(Simulation):
    """A toy physics model representing a sandwich test

    The problem integrates the differential equation for a horixontal
    slicke through an assembly of a HE block sandwich between two thin
    HE plates. The expanding products- of-detonation of a high
    explosive accelerate these plates, the effects of material strengh
    are negligable.

    **Units**

    This model is based on the CGS units system


    **Diagram**

    The problem is symmetric::

        :-------------+----+
        :   HE        | Cu |
        :-------------+----+
        CL


    **Attributes**
       Inherited from ::pyclass::`pyStruc`

    **Methods**
    """
    def __init__(self, name=u'Gun Toy Computational Experiment',
                 *args, **kwargs):
        """Instantiate the Experiment object

        Args:
            None

        Keyword Args:
            name(str): A name. (Default = 'Gun Toy Computational Experiment')

        """

        # 'Name': [Type, Default, Min, Max, Units, Description]
        def_opts = {
            'x_i': [float, 1.003, 0.0, None, 'cm',
                    'Initial thickness of HE'],
            'x_f': [float, 1E21, 0.0, None, 'cm',
                    'Final thickness of HE'],
            't_cu': [float, 0.102, 0.0, None, 'cm',
                     'Initial thickness of Cu'],
            'rho_cu': [float, 8.924, 0.0, None, 'g cm-3',
                       'Mass of projectile'],
            'rho_he': [float, 2.75, 0.0, None, 'g cm-3',
                       'The density of the HE reactants'
                       'the projectile'],
            'sigma': [float, 1.0e0, 0.0, None, 'cm s-1',
                      'Variance attributed to v measurements'],
            't_min': [float, 0.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            't_max': [float, 10.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            'n_t': [int, 1 * 512, 0, None, '',
                    'Number of times for t2v spline']
        }

        Simulation.__init__(self, {'eos': Isentrope}, name=name,
                            def_opts=def_opts, *args, **kwargs)

    def _on_check_models(self, models):
        """Checks that the model is valid

        Args:
            model(Isentrope): A new EOS model

        Return:
            (GunModel): A copy of self with the new eos model
        """
        return (models['eos'],)

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

    def _get_accel(self, eos, posn):
        """Calculates the acceleration of the projectile

        The force is the pressure of the HE gas acting on the projectile.
        The pressure is given by the EOS model

        Args:
            eos(Isentrope): The equation of state model
            posn(float): The scalar position

        Return:
            (float): The force in dyes

        """
        rho_cu = self.get_option('rho_cu')
        t_cu = self.get_option('t_cu')
        vol_he = 1E2 * posn / (self.get_option('rho_he')
                               * self.get_option('x_i'))
        # print('vol {:} g cm-3'.format(vol_he))
        # 0.1 converts the resulting acceleration into m s**-2
        if eos.get_option('basis') == 'dens':
            return 0.1 * eos(vol_he**-1) / (rho_cu * t_cu)
        else:
            return 0.1 * eos(vol_he) / (rho_cu * t_cu)
        # end

    def _shoot(self, eos):
        """ Run a simulation and return the results: t, [x,v]

        Solves the ODE

        .. math::

           F(x,v,t) = \\frac{d}{dt} (x, v)

        Args:
           eos(Isentrope): The equation of state model

        Return:
            (tuple): Length 2 elements are:

                 0. (np.ndarray): time vector
                 1. (list): elements are:

                     0. (np.ndarray): position
                     1. (np.ndarray): velocity
        """
        t_min = self.get_option('t_min')
        t_max = self.get_option('t_max')
        n_t = self.get_option('n_t')
        x_i = 1E-2 * self.get_option('x_i')  # cm to m
        x_f = 1E-2 * self.get_option('x_f')  # cm to m

        def diffeq(state, time):
            """vector field for integration

            Args:
               state (list): state variable [position, velocity]
               time (np.ndarray): time

            Return:
               (float): velocity
               (float): acceleration

            .. math::

               F((position,velocity),t) = \frac{d}{dt} (position,velocity)

            """
            if time < 0:
                return np.zeros(2)
            if state[0] > x_f:  # beyond end of gun barrel, muzzle
                accel = 0.0
            else:
                accel = self._get_accel(eos, state[0])
            return np.array([state[1], accel])

        time_list = np.linspace(t_min, t_max, n_t)
        xv_states = odeint(
            diffeq,
            [x_i, 0],
            time_list,
            atol=1e-11,  # Default 1.49012e-8
            rtol=1e-11,  # Default 1.49012e-8
        )
        if not xv_states.shape == (len(time_list), 2):
            raise ValueError('{} did not solve the differential equation'
                             'correctly'.format(self.get_inform(1)))
        # end

        # xv is array of calculated positions and velocities at times in t
        return time_list, xv_states

    def _fit_t2v(self, vel, time):
        """Fits a cubic spline to the velocity-time history

        This allows simulations and experiments to be compared at the
        experimental time-stamps

        Args:
           vel(np.ndarray): Velocity history
           time(np.ndarray): Time history

        Return:
           (Spline): Spline of vel = f(time)

        """

        return Spline(time, vel)

    def get_sigma(self, models):
        """Returns the co-variance matrix

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`
        """

        return np.diag(np.ones(self.shape()) * self.get_option('sigma'))

    def shape(self):
        """Returns the degrees of freedom of the model

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.shape`
        """

        return self.get_option('n_t')

    def compare(self, simdata1, simdata2):
        """Compares a set of experimental data to the model

        Error is `dep` less the `model_data`

        See :py:meth:`F_UNCLE.Utils.Experiment.Experiment.compare`
        """

        return simdata2[1][0] - simdata1[2]['mean_fn'](simdata2[0])

    def _on_call(self, eos, **kwargs):
        """Performs the simulation / experiment using the internal EOS

        Args:
            eos(Isentrope): The equation of state model
            **kwargs: Arbitrary keyword arguments.

        Return:
            (tuple): Length 3, elements are:

                0. (np.ndarray): Time, the independent variable
                1. (tuple): length 2 for the two dependent variables

                    0. (np.ndarray): Velocity history of the simulation
                    1. (np.ndarray): Position history of the simulation

                2. (Spline): A spline representing the velocity-time history

        """

        time, states = self._shoot(eos)

        vt_spline = self._fit_t2v(states[:, 1], time)

        return time, [states[:, 1], states[:, 0], ['Velocity', 'Position']],\
            {'mean_fn': vt_spline}

    def plot(self, axes=None, fig=None, data=None,
             linestyles=['-k', '-r'], labels=['Model', 'Error'],
             *args, **kwargs):
        """Plots the sandwich experiment

        Overloads :py:meth:`F_UNCLE.Utils.Struc.Struc.plot`


        Args:
            axes(plt.Axes): Axes object on which to plot, if None, creates
                new figure
            fig(str): A Figure object on which to plot, used for level3 plot
            linestyles(list): A list of strings for linestyles
                0. Data '-k'
                1. Error '-r'
            labels(list): A list of strings for labels
                0. 'Model'
                1. 'Error'

            level(int): A tag for which kind of plot should be generated
            data(dict): A list of other data to be plotted for comparison

        Return:
            (plt.figure): A figure

        """

        if axes is None:
            if fig is None:
                fig = plt.figure()
            # end
            ax1 = fig.gca()
        elif isinstance(axes, plt.Axes):
            fig = None
            ax1 = axes
        else:
            raise TypeError("{} axis must be a matplotlib Axis object"
                            .format(self.get_inform(1)))
        # end

        # Plot the velocity time history
        lineiter = iter(linestyles)
        labeliter = iter(labels)
        for dati in data:
            ax1.plot(1E6 * dati[0], 1E-4 * dati[1][0], next(lineiter),
                     label=next(labeliter))
        ax1.set_xlabel(r'Time / $\mu$s')
        ax1.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')

        if len(data) == 2:
            ax2 = ax1.twinx()
            err_style = next(lineiter)
            ax2.plot(1E6 * data[0][0],
                     1E-4 * (data[0][1][0] - data[1][1][0]),
                     err_style)
            # ax1.plot(None, None, err_style, label='Error')

            ax2.set_ylabel(r'Error / cm $\mu$s$^{-1}$')
        # end

        ax1.legend(loc='best')

        return fig


class ToySandwichExperiment(GaussianExperiment):
    """A class representing pseudo experimental data for a gun show
    """
    def _get_data(self, model=None, rstate=None, *args, **kwargs):
        """Creates a simulated set of experimental data from a user provided
        model
        """

        sim = ToySandwich(**kwargs)

        simdata = sim({'eos': model})

        noise_model = NoiseModel()
        noisedata = noise_model(simdata[0], 0.1 * simdata[1][0], rstate=rstate)

        return simdata[0], simdata[1][0] + noisedata,\
            np.zeros(simdata[0].shape)
    # end

    def get_sigma(self):
        """Returns the co-variance matrix

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`
        """
        corr = np.eye(self.shape())
        var = 1E-6 * self.get_option('exp_corr')
        times = self.data[0][self.window]
        for i in range(corr.shape[0]):
            x_list = (times[i] - times)
            corr[i, :] = np.exp(-0.5 * (x_list/var)**2) 
        eig, vect = np.linalg.eig(corr)
        eig = np.real(eig)
        eig = np.where(eig>0, eig, 0.0)

        assert eig.shape[0] == corr.shape[0], "Too few eigenvalues"
        assert np.all(np.isreal(eig)), "Imaginary eigenvalues"
        assert np.all(eig >= 0.0), "Negative eigenvalues {:}".format(eig)
        corr = np.dot(vect.T, np.dot(np.diag(eig), vect))
        
        sigma = np.diag(np.ones(self.shape())
                        * (np.fabs(self.data[1][self.window])
                           * self.get_option('exp_var'))**2)
        assert np.all(np.isfinite(sigma))

        retval = np.dot(np.sqrt(sigma), np.dot(corr, np.sqrt(sigma)))
        assert np.all(np.isfinite(retval))
        return retval
