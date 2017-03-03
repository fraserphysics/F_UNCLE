"""

pyGunModel

Toy computational experiment to

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
from ..Utils.Experiment import Experiment
from ..Utils.Simulation import Simulation
from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline

# if __name__ == '__main__':
#     sys.path.append(os.path.abspath('./../../'))
#     from F_UNCLE.Utils.DataExperiment import DataExperiment
#     from F_UNCLE.Utils.Experiment import Simulation
#     from F_UNCLE.Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
# else:
#     from ..Utils.DataExperiment import DataExperiment
#     from ..Utils.Experiment import Simulation
#     from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
# # end

# =========================
# Main Code
# =========================


class Gun(Simulation):
    """A toy physics model representing a gun type experiment

    The problem integrates the differential equation for a mass being
    accelerated down the barrel of a gun by an the expanding products-
    of-detonation of a high explosive. The gun has finite dimensions
    and the integration lasts beyond when the projectile exits the gun.

    **Units**

    This model is based on the CGS units system

    **Diagram**

    .. figure:: /_static/gun.png

       variables defining the gun experiment

    **Options**

   +---------+-------+------+-----+-----+-----+--------------------------------+
   |Name     |Type   |Def   |Min  |Max  |Units|Description                     |
   +=========+=======+======+=====+=====+=====+================================+
   |`X_i`    |(float)|0.4   |0.0  |None |cm   |Initial position of projectile  |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`x_f`    |(float)|3.0   |0.0  |None |cm   |Final/muzzle position of        |
   |         |       |      |     |     |     |projectile                      |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`m`      |(float)|500.0 |0.0  |None |g    |Mass of projectile              |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`mass_he`|(float)|4     |0.0  |None |g    |The initial mass of high        |
   |         |       |      |     |     |     |explosives used to drive the    |
   |         |       |      |     |     |     |projectile                      |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`area`   |(float)|1.0   |0.0  |None |cm**2|Projectile cross section        |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`sigma`  |(float)|1.0e0 |0.0  |None |??   |Variance attributed to v        |
   |         |       |      |     |     |     |measurements                    |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`t_min`  |(float)|1.0e-6|0.0  |None |sec  |Range of times for t2v spline   |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`t_max`  |(float)|1.0e-2|0.0  |None |sec  |Range of times for t2v spline   |
   +---------+-------+------+-----+-----+-----+--------------------------------+
   |`n_t`    |(int)  |250   |0    |None |''   |Number of times for t2v spline  |
   +---------+-------+------+-----+-----+-----+--------------------------------+

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
            'x_i': [float, 0.4, 0.0, None, 'cm',
                    'Initial position of projectile'],
            'x_f': [float, 3.0, 0.0, None, 'cm',
                    'Final/muzzle position of projectile'],
            'm': [float, 500.0, 0.0, None, 'g',
                  'Mass of projectile'],
            'mass_he': [float, 4, 0.0, None, 'g',
                        'The initial mass of high explosives used to drive'
                        'the projectile'],
            'area': [float, 1.0, 0.0, None, 'cm**2',
                     'Projectile cross section'],
            'sigma': [float, 1.0e0, 0.0, None, '??',
                      'Variance attributed to v measurements'],
            't_min': [float, 1.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            't_max': [float, 1.0e-2, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            'n_t': [int, 250, 0, None, '',
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

    def _get_force(self, eos, posn):
        """Calculates the force on the projectile

        The force is the pressure of the HE gas acting on the projectile.
        The pressure is given by the EOS model

        Args:
            eos(Isentrope): The equation of state model
            posn(float): The scalar position

        Return:
            (float): The force in dyes

        """
        area = self.get_option('area')
        mass_he = self.get_option('mass_he')

        # 1E-4 converts from area in cm**2 to m**2 so result is N
        if eos.get_option('basis') == 'dens':
            return eos(mass_he / posn * area) * area * 1E-4
        else:
            return eos(posn * area / mass_he) * area * 1E-4
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
        x_i = self.get_option('x_i')
        x_f = self.get_option('x_f')
        proj_mass = self.get_option('m')

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
                accel = self._get_force(eos, state[0]) / proj_mass  # F = MA
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

    def plot(self, axes=None, fig=None, level=0, data=None,
             linestyles=['-k', '-r'], labels=['Model', 'Error'],
             *args, **kwargs):
        """Plots the gun experiment

        Overloads :py:meth:`F_UNCLE.Utils.Struc.Struc.plot`

        **Plot Levels**

        0. The velocity-time history
        1. The position-time history
        2. The velocity-position history
        3. A 4 subplot figure with levels 1-3 as well as the EOS plotted

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
            fig = plt.figure()
        elif isinstance(axes, plt.Axes):
            ax1 = axes
        else:
            raise TypeError("{} axis must be a matplotlib Axis object"
                            .format(self.get_inform(1)))
        # end

        if level == 0:
            # Plot the velocity time history
            ax1.plot(data[0][0], data[0][1][0], linestyles[0],
                     label=labels[0])
            ax1.set_xlabel('Time / s')
            ax1.set_ylabel(r'Velocity / cm s$^{-1}$')

            if len(data) == 2:
                ax2 = ax1.twinx()
                ax2.plot(data[0][0], data[0][1][0] - data[1][1][0],
                         linestyles[1], label=labels[1])
                ax2.set_ylabel('Error / cm s$^{-1}$')
            # end

        elif level == 1:
            # Plot the position time history
            pass
        elif level == 3:
            # Plot everything
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            v_spec = data[0][1][0] * self.get_option('area')\
                / self.get_option('mass_he')
            ax1.plot(data[0][0], data[0][1][0], 'k')
            ax2.plot(data[0][0], data[0][1][1], 'k')
            ax3.plot(v_spec, data[0][1][1], 'k')
            ax3.axhline(self.get_option('x_f'))

            if len(data) == 2:
                ax1.plot(data[0][0], 10 + (data[0][1][1] - data[1][1][1]), 'r')
                ax2.plot(data[0][0], 1 + (data[0][1][0] - data[1][1][0]), 'r')
            # end

            ax1.set_xlabel("Simulation time / s")
            ax1.set_ylabel("Projectile velocity / cm s**-1")
            ax2.set_xlabel("Simulation time / s")
            ax2.set_ylabel("Projectile position / cm")
            ax3.set_xlabel("HE Specific volume / cm**3 g**-1")
            ax3.set_ylabel("Projectile position / cm s**-1")


class GunExperiment(Experiment):
    """A class representing pseudo experimental data for a gun show
    """
    def _get_data(self, model=None, *args, **kwargs):
        """Creates a simulated set of experimental data from a user provided
        model
        """

        sim = Gun(**kwargs)

        simdata = sim({'eos': model})

        return simdata[0], simdata[1][0],\
            np.zeros(simdata[0].shape)
    # end
        
    def get_sigma(self):
        """Returns the co-variance matrix

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`
        """

        return np.diag(np.ones(self.shape())
                       * self.get_option('exp_var'))
