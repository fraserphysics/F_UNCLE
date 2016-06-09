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
from scipy.optimize import brentq

# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Utils.Experiment import Experiment
from F_UNCLE.Models.Isentrope import EOSBump, EOSModel, Isentrope


# =========================
# Main Code
# =========================
class Stick(Experiment):
    """A toy physics model representing a rate stick
    
    **Units**
    
    Units are based on CGS system

    **Diagram**

    .. figure:: /_static/stick.png
       
       The assumed geometry of the rate stick

    Attributes:
        const(dict): A dictionary of conversion factors

    """
    def __init__(self, eos, name='Rate Stick  Computational Experiment',\
                 *args, **kwargs):
        """Instantiate the Experiment object

        Args:
            eos(Isentrope): The equation of state model used in the toy
                            computational experiment

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
            'sigma_t': [float, 1.0e0, 0.0, None, 's',
                      'Variance attributed to t measurements'],
            'sigma_x': [float, 1.0e0, 0.0, None, 'cm',
                        'Variance attributed to x positions'],
            'd_min': [float, 1.0e5, 0.0, None, 'cm sec**-1',
                      'Lower search range for detonation velocity'],
            'd_max': [float, 1.0e6, 0.0, None, 'cm sec**-1',
                      'Uper search range for detonation velocity'],
            'vol_0': [float, 1.835**-1, 0.0, None, 'cm**3 g**-1',
                      'The pre detonation specific volume of HE'],
            'x_min': [float, 0.0, None, 'cm',
                      'The position of the first sensor on the rate stick'],
            'x_max': [float, 0.4, None, 'cm',
                      'The position of the last sensor on the rate stick'],
            'n_x': [int, 10, 0, None, '',
                    'Number of sensor positions']
        }

        self.const = {'newton2dyne':1e5,
                      'cm2km':1.0e5}

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

    def get_sigma(self):
        """Returns the variance matrix
        """
        vol_0 = self.get_option('vol_0')
        eos = self.eos

        vel_cj, vol_cj, p_cj, rayl_line = self._get_cj_point(eos, vol_0)

        var = np.ones(self.get_option('n_x'))*(self.get_option('sigma_t')**2)
        var += (self.get_option('sigma_x')/vel_cj)**2
        return np.diag(var)

    def shape(self):
        """Returns the shape of the object
        """

        return self.get_option('n_x')

    def __call__(self):
        """Performs the rate stick experiment

        Return:
            (np.ndarray): The independent variabe, the `n` sensor positions
            (tuple): The dependent varaibles
                [0] (None)
                [1] (np.ndarray): The arrival `n` times at each sensor
            (tuple): The other solution data
                [0] the detonation velocity
                [1] the specific volume at the CJ point
                [2] the pressure at the CJ point
                [3] a rayleight ling function

                    p = ray(v, vel, vol0, eos)
                    Args:
                        v(np.ndarray): The specific volume
                        vel(float): Detonation velocity
                        vol_0(float): Specific volume ahead of the shock
                        eos(Isentrope): An equation of state model
                    Return:
                        p(float): The pressure along the Rayleight line at v
        """
        x_min = self.get_option('x_min')
        x_max = self.get_option('x_max')
        n_x = self.get_option('n_x')
        vol_0 = self.get_option('vol_0')
        eos = self.eos

        x_list = np.linspace(x_min, x_max, n_x)

        cj_vel, cj_vol, cj_p, ray_fun = self._get_cj_point(eos, vol_0)

        t_list = x_list / cj_vel

        return x_list, [t_list], (cj_vel, cj_vol, cj_p, ray_fun)

    def compare(self, indep, dep, data):
        """Compares the model instance to other data

        Args:
           indep(np.ndarray): The sensor positions of the other model
           dep(np.ndarray): The data from the other model
           data(tuple): Summary of data for comparisson

        Return:

        """

        det_vel = data[2][0]


        return dep - indep/det_vel

    def _get_cj_point(self, eos, vol_0):
        '''Find CJ conditions using two nested line searches.
        '''
        # Search for det velocity between 1 and 10 km/sec
        d_min = 2.0e5 # cm s**-1
        d_max = 7.0e5 # cm s**-1
        v_min = eos.get_option('spline_min')
        v_max = eos.get_option('spline_max')

        # R is Rayleigh line
        rayl_line = lambda vel, vol, eos, vol_0: (vel**2)*(vol_0 - vol)/(vol_0**2)

        # F is self - R
        rayl_err = lambda vel, vol, eos, vol_0: eos(vol)  - rayl_line(vel, vol, eos, vol_0)

        # d_F is derivative of F wrt vol
        derr_dvol = lambda vol, vel, eos, vol_0: eos.derivative(1)(vol) + (vel/vol_0)**2

        # arg_min(vel, self) finds volume that minimizes self(v) - R(v)
        arg_min = lambda vel, eos, vol_0: brentq(derr_dvol, v_min, v_max,
                                          args=(vel, eos, vol_0))

        error = lambda vel, eos, vol_0: rayl_err(vel, arg_min(vel, eos, vol_0), eos, vol_0)

        # print rayl_err(d_min, v_min, eos)
        # print rayl_err(d_min, v_max, eos)
        # print rayl_err(d_max, v_min, eos)
        # print rayl_err(d_max, v_max, eos)

        # print derr_dvol(v_min, d_min, eos)
        # print derr_dvol(v_max, d_min, eos)
        # print derr_dvol(v_min, d_max, eos)
        # print derr_dvol(v_max, d_max, eos)

        # print "err dmax ", error(d_max, eos)
        # print "err dmin ", error(d_min, eos)

        vel_cj = brentq(error, d_min, d_max, args=(eos,vol_0))
        vol_cj = arg_min(vel_cj,eos, vol_0)
        p_cj = eos(vol_cj)

        # plt.figure()
        # v = np.logspace(np.log10(v_min), np.log10(v_max), 20)

        # plt.plot(v, eos(v))
        # plt.plot(v, rayl_line(d_min, v, eos))
        # plt.plot(v, rayl_line(d_max, v, eos))
        # plt.plot(vol_0, 0.0, 'sk', ms = 10, mfc = 'None')
        # plt.plot(v, rayl_line(vel_cj, v, eos))
        # plt.plot(vol_cj, p_cj, '*k', ms = 10, mfc = 'None')
        # plt.show()

        return vel_cj, vol_cj, p_cj, rayl_line

    def plot(self, axis = None):

        v_min = self.eos.get_option('spline_min')
        v_max = self.eos.get_option('spline_max')
        v_0 = self.get_option('vol_0')
        if axis is None:
            fig = plt.figure()
            ax1 = fig.gca()
        # end

        self.eos.plot(axis = ax1)
        vel_cj, vol_cj, p_cj, rayl_line = self._get_cj_point(self.eos, 1.835**-1)

        v_eos = np.logspace(np.log10(v_min), np.log10(v_max), 30)

        ax1.plot(v_eos, rayl_line(vel_cj,v_eos, self.eos, v_0), '-k')
        ax1.plot(vol_cj,p_cj,'sk')

        plt.show()


class TestStick(unittest.TestCase):
    """Test of the Stick experiment
    """
    def setUp(self):
        """Common setup options for the models
        """
        init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

        self.true_eos = EOSBump()
        self.model_eos =  EOSModel(init_prior)
    def test_instatntiation(self):
        """Tests basic instantiation
        """

        stick = Stick(self.true_eos)

    def test_call(self):
        """Tests that the default settings run and the outputs are structured
           correctly
        """
        stick = Stick(self.true_eos)

        n_data = stick.get_option('n_x')

        data = stick()

        # Check data was output correctly
        self.assertEqual(len(data), 3)

        indep = data[0]
        dep = data[1]
        smry = data[2]

        # Check the independent data
        self.assertEqual(len(indep), n_data)

        # Check the depdendent data
        self.assertEqual(len(dep),1)
        self.assertEqual(len(dep[0]), n_data)

        # Check the summary data
        self.assertEqual(len(smry),4)
        self.assertIsInstance(smry[0], float)
        self.assertIsInstance(smry[1], float)
        self.assertIsInstance(smry[2], float)
        self.assertTrue(hasattr(smry[3], '__call__'))

    def test_eos_step(self):
        """Tets that the sitck model is sensitive to changes in eos
        """
        stick = Stick(self.model_eos)

        n_data = stick.get_option('n_x')
        
        data1 = stick()
        stick.plot()
        i = np.argmin(np.fabs(self.model_eos.get_t() - data1[2][1]))
        i -= 2

        plt.figure()
        ax1 = plt.gca()
        for i in xrange(self.model_eos.shape()[0]):
            initial_dof = self.model_eos.get_dof()

            delta = 0.02
            delta *= initial_dof[i]

            new_dof = copy.deepcopy(initial_dof)
            new_dof[i] += delta

            self.model_eos.set_dof(new_dof)

            stick.update(model = self.model_eos)

            data2 = stick()
            # stick.plot()        
            plt.plot((data1[1][0] - data2[1][0])/delta)

        plt.show()
        
        
        
    def test_compare(self):
        """Tests the comparisson function
        """

        true_stick = Stick(self.true_eos)
        sim_stick = Stick(self.model_eos)
        n_true = true_stick.get_option('n_x')
        n_sim = 10
        sim_stick.set_option('n_x', n_sim)

        true_data = true_stick()
        sim_data = sim_stick()

        self.assertEqual(len(true_data[0]), n_true)
        self.assertEqual(len(sim_data[0]), n_sim)

        epsilon = sim_stick.compare(true_data[0], true_data[1][0], sim_data)

        self.assertEqual(len(epsilon), n_true)


    def test_shape(self):
        """Tests the shape function
        """

        stick = Stick(self.true_eos)

        self.assertEqual(stick.shape(), stick.get_option('n_x'))

    def test_sigma(self):
        """Tests the variance function
        """

        stick = Stick(self.true_eos)

        dim = stick.shape()
        var = stick.get_sigma()
        self.assertEqual(var.shape, (dim, dim))

if __name__ == '__main__':
    try:
        unittest.main(verbosity=4)
    except Exception as inst:
        print inst
        pdb.post_moretem()
