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
from scipy.integrate import odeint

# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Utils.Experiment import Experiment
from F_UNCLE.Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline


# =========================
# Main Code
# =========================
class Gun(Experiment):
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

    Attributes:
         eos(Isentrope): A model of the products-of-detonation equation of
             state  
    """
    def __init__(self, eos, name='Gun Toy Computational Experiment', *args, **kwargs):
        """Instantiate the Experiment object

        Args:
            eos(Isentrope): The equation of state model used in the toy computational experiment

        Keyword Args:
            name(str): A name. (Default = 'Gun Toy Computational Experiment')

        """

        if isinstance(eos, Isentrope):
            self.eos = eos
        else:
            raise TypeError('{:} Equation of state model must be an Isentrope object'\
                            .format(self.get_inform(2)))
        # end

        def_opts = {
            'x_i': [float, 0.4, 0.0, None, 'cm',
                    'Initial position of projectile'],
            'x_f': [float, 3.0, 0.0, None, 'cm',
                    'Final/muzzle position of projectile'],
            'm': [float, 500.0, 0.0, None, 'g',
                  'Mass of projectile'],
            'mass_he': [float, 4, 0.0, None, 'g',
                        'The initial mass of high explosives used to drive\
                        the projectile'],
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

        Experiment.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)

    def update(self, model=None):
        """Update the analysis with a new model
        """
        if model is None:
            pass
        elif isinstance(model, Isentrope):
            self.eos = copy.deepcopy(model)
        else:
            raise TypeError('{}: Model must be an isentrope for update'\
                            .format(self.get_inform(1)))
        #end
        
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

    def _get_force(self, posn):
        """Calculates the force on the prjectile

        The force is the pressure of the HE gas acting on the projectile.
        The pressure is given by the EOS model

        Args:
            posn(float): The scalar position

        Retun:
            (float): The force in dynes

        """
        area = self.get_option('area')
        mass_he = self.get_option('mass_he')

        return self.eos(posn * area / mass_he) * area  * 1E-4

    def _shoot(self):
        """ Run a simulation and return the results: t, [x,v]

        Solves the ODE

        .. math::

           F(x,v,t) = \\frac{d}{dt} (x, v)

        Args:
           None

        Return:
            (np.ndarray): time vector
            (list): elements are
                - [0] -> np.ndarray: position
                - [1] -> np.ndarray: velocity
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
               time (numpy.ndarray): time

            Return:
               (float): velocity
               (float): acceleration

            .. math::

               F((position,velocity),t) = \frac{d}{dt} (position,velocity)

            """

            if time < 0:
                return np.zeros(2)
            if state[0] > x_f: # beyond end of gun barrel, muzzle
                accel = 0.0
            else:
                accel = self._get_force(state[0])/proj_mass # F = MA
            return np.array([state[1], accel])

        time_list = np.linspace(t_min, t_max, n_t)
        xv_states = odeint(
            diffeq,            #
            [x_i, 0],
            time_list,
            atol=1e-11, # Default 1.49012e-8
            rtol=1e-11, # Default 1.49012e-8
            )
        if not  xv_states.shape == (len(time_list), 2):
            raise ValueError('{} did not solve the differential equation correctly'\
                             .format(self.get_inform(1)))
        #end

        # xv is array of calculated positions and velocities at times in t
        return time_list, xv_states

    def _fit_t2v(self, vel, time):
        """Fits a cubic spline to the velocity-time history

        This allows simulations and experiments to be compared at the
        experimental timestamps

        Args:
           vel(np.ndarray): Velocity history
           time(np.ndarray): Time history

        Return
           (Spline): Spline of vel = f(time)

        """

        return Spline(time,vel)

    def get_sigma(self):
        """Returns the covariance matrix
        
        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`
        """

        return np.diag(np.ones(self.shape())* self.get_option('sigma'))

    def shape(self):
        """Returns the degrees of freedom of the model

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.shape`
        """

        return  self.get_option('n_t')

    def compare(self, indep, dep, model_data):
        """Compares a set of experimental data to the model

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.compare`
        """

#        return dep - model_data[1][1]

        return dep - model_data[2](indep)



    def __call__(self, *args, **kwargs):
        """Performs the simulation / experiment using the internal EOS

        Args:

        Returns:
           (np.ndarray): Time, the independent variable
           (tuple): length 2 for the two depdendent variables
                    [0] (np.ndarray): Velocity history of the simulation
                    [1] (np.ndarray): Position history of the simulation
           (Spline): A spline representing the velocity-time history

        """

        time, states = self._shoot()

        vt_spline = self._fit_t2v(states[:, 1], time)

        return time, (states[:, 1], states[:, 0]), vt_spline

    def plot(self, axis = None, level = 0, data = None, *args, **kwargs):
        """Plots the gun experiment
        """

        if axis == None:
            fig = plt.figure()

        elif isinstance(ax1,plt.Axes):
            fig = None
            ax1 = axis
        else:
            raise TypeError("{} axis must be a matplotlib Axis obect".\
                            format(self.get_inform(1)))
        #end
        
        if level == 0:
            """Plot the velocity time history
            """
            pass
        elif level == 1:
            """Plot the position time history
            """
            pass
        elif level == 2:
            """Plot everything
            """

            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            v_spec = data[0][1][0] * self.get_option('area')/self.get_option('mass_he')
            ax1.plot(data[0][0],data[0][1][0], 'k')
            ax2.plot(data[0][0],data[0][1][1], 'k')
            ax3.plot(v_spec, data[0][1][1], 'k')
            ax3.axhline(self.get_option('x_f'))
            self.eos.plot(axis = ax4)

            if len(data) == 2:
                ax1.plot(data[0][0],10+(data[0][1][1] - data[1][1][1]), 'r')
                ax2.plot(data[0][0],1 +(data[0][1][0] - data[1][1][0]), 'r')
#                ax3.plot(v_spec, 10+(data[0][1][1] - data[1][1][1]), 'r')
            #end
            
            ax1.set_xlabel("Simulation time / s")
            ax1.set_ylabel("Projectile velocity / cm s**-1")
            ax2.set_xlabel("Simulation time / s")
            ax2.set_ylabel("Projectile position / cm")
            ax3.set_xlabel("HE Specific volume / cm**3 g**-1")
            ax3.set_ylabel("Projectile position / cm s**-1")
        

class TestGun(unittest.TestCase):
    """Tets of the Gun experiment
    """

    def test_instantiation(self):
        """Tests that the model is properly instantiated
        """
        eos = EOSBump()
        gun = Gun(eos)

        self.assertIsInstance(gun, Gun)
        print gun
    # end

    def test_shoot_exp_eos(self):
        """Performs a test shot using default settings
        """

        eos = EOSBump()
        gun = Gun(eos)

        time, (vel, pos), spline = gun()

        n_time = gun.get_option('n_t')

        self.assertEqual(len(time), n_time)
        self.assertEqual(len(pos), n_time)
        self.assertEqual(len(vel), n_time)

    def test_shoot_model_eos(self):
        """Performs a test shot using default settings a model eos    
        """
        p_fun = lambda v: 2.56e9 / v**3
        
        eos = EOSModel(p_fun)
        gun = Gun(eos)

        print gun
        
        time, (vel, pos), spline = gun()

        n_time = gun.get_option('n_t')

        self.assertEqual(len(time), n_time)
        self.assertEqual(len(pos), n_time)
        self.assertEqual(len(vel), n_time)

    
    # @unittest.skip('skipped plotting routine')
    def test_shot_plot(self):
        """
        """
        import matplotlib.pyplot as plt

        init_prior = np.vectorize(lambda v: 2.56e9 / v**3)
        
        # Create the model and *true* EOS
        eos = EOSModel(init_prior)
        

        gun = Gun(eos)

        data0  = gun()        
        old_dof = eos.get_c()
        old_dof[0] *= 1.02
        eos.set_dof(old_dof)
        gun.update(model = eos)
        data1 = gun()

        gun.plot(level = 2, data = [data0, data1] )
        # n_time = gun.get_option('n_t')

        # fig = plt.figure()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(223)
        # ax3 = fig.add_subplot(222)
                              
        
        
        # ax1.plot(time, vel1)
        # ax1.plot(time, spline1(time))
        # ax1.set_xlabel('Time')
        # ax1.set_ylabel('Velocity')        

        # ax3.plot(pos1, vel1)
        # ax3.set_xlabel('Position')
        # ax3.set_ylabel('Velocity')        
        
        # v_spec_list = np.linspace(0.01, 10.0, 30)
        # ax2.plot(v_spec_list, eos(v_spec_list))
        # ax2.set_xlabel('Specific volume')
        # ax2.set_ylabel('Pressure')
        # ax3 = ax2.twinx()
        # ax3.plot(pos1*gun.get_option('area')/gun.get_option('mass_he'), vel1)
        
        plt.show()
# end

if __name__ == '__main__':
    unittest.main(verbosity=4)
