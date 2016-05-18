#/usr/bin/pyton
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

from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.
from scipy.integrate import quad
from scipy.integrate import odeint

# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../'))
from FUNCLE.pyExperiment import Experiment
from FUNCLE.pyIsentrope import EOSBump, EOSModel, Isentrope


# =========================
# Main Code
# =========================
class Gun(Experiment):
    """A toy physics model representing a gun type experiment

    Attributes:
        const(dict): A dictionary of conversion factors

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
            'x_f': [float, 4.0, 0.0, None, 'cm',
                    'Final/muzzle position of projectile'],
            'm': [float, 100.0, 0.0, None, 'g',
                  'Mass of projectile'],
            'area': [float, 1e-4, 0.0, None, 'm**2',
                     'Projectile cross section'],
            'var': [float, 1.0e-0, 0.0, None, '??',
                    'Variance attributed to v measurements'],
            't_min': [float, 5.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            't_max': [float, 110.0e-6, 0.0, None, 'sec',
                      'Range of times for t2v spline'],
            'n_t': [int, 500, 0, None, '',
                    'Number of times for t2v spline']
        }

        self.const = {'newton2dyne':1e5,
                      'cm2km':1.0e5}

        Experiment.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)


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

    def _f(self, x):
        """Calculates the force on the prjectile

        Args:
            x(float): The scalar position

        Retun
            (float): The force in dynes

        """

        return self.eos(x) * self.get_option('area') * self.const['newton2dyne']

    def _e(self, x):
        """Integrates the force up to position x

        Args:
            x(float): Scalar position

        Return:
            (float): The intergral of the foce over the distance to x

        """

        x_i = self.get_option('x_i')
        x_f = self.get_option('x_f')

        rv, err = quad(self._f, x_i, min(x,x_f))

        if np.isnan(rv):
            raise Exception("{:} NaN encountered when intergating energy"\
                            .format(self.get_inform(1)))
        # end

        return rv

    def _x_dot(self, x):
        """Calculate the projectile velocity

        Calculates at a single position x, or
        if x is an array, calculate the velocity for each element of x

        Args:
           x(float or np.ndarray): scalar position

        Return
           (np.ndarray): velocity
        """
        x_i = self.get_option('x_i')
        x_f = self.get_option('x_f')
        m = self.get_option('m')

        if isinstance(x, np.ndarray):
            return np.array([self._x_dot(x_) for x_ in x])
        elif isinstance(x, (float,np.float)):
            if x <= x_i:
                return 0.0
            # assert(self._E(x) > =0.0),'E(%s)=%s'%(x,self._E(x))
            return np.sqrt(2*self._E(x)/m) # Invert E = (mv^2)/2
        else:
            raise TypeError('x has type %s'%(type(x),))
        # end
        
    def _shoot(self, t_min, t_max, n_t):
        """ Run a simulation and return the results: t, [x,v]

        Solves the ODE

        .. math::

           F(x,v,t) = \\frac{d}{dt} (x, v)

        Args:
            t_min(float): start time of the solution
            t_max(float): end time of the solution

        **Returns**
            (list): elements are
                - [0] -> np.ndarray: position
                - [1] -> np.ndarray: velocity
        """
        x_i = self.get_option('x_i')
        x_f = self.get_option('x_f')
        m = self.get_option('m')

        def F(x,t):
            """vector field for integration

            Args:
             x (numpy.ndarray): position

            .. math::

               F((position,velocity),t) = \frac{d}{dt} (position,velocity)

            """
            if t < 0:
                return np.zeros(2)
            if x[0] > x_f: # beyond end of gun barrel, muzzle
                acceleration = 0.0
            else:
                acceleration = self._f(x[0])/m # F = MA
            return np.array([x[1], acceleration])
        t = np.linspace(t_min, t_max, n_t)
        xv = odeint(
            F,            #
            [x_i, 0],
            t,
            atol=1e-11, # Default 1.49012e-8
            rtol=1e-11, # Default 1.49012e-8
            )
        assert xv.shape == (len(t), 2)
        # xv is array of calculated positions and velocities at times in t
        return t, xv

    def _fit_t2v(self, x):
        pass

class TestGun(unittest.TestCase):
    """Tets of the Gun experiment
    """

    def test_instantiation(self):
        """Tests that the model is properly instantiated
        """
        eos = EOSBump()
        gun = Gun(eos)

        print gun
    # end
# end

if __name__ == '__main__':
    unittest.main(verbosity=4)
