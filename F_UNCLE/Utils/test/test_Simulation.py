#!/usr/bin/pyton
"""

test_Simulation

Tests of the simulation Class

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

import unittest
import sys
import os
import copy
import warnings
import time

# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import numpy.testing as npt
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline

# =========================
# Custom Packages
# =========================
from ..Struc import Struc
from ..PhysicsModel import PhysicsModel
from .test_PhysicsModel import SimpleModel
from ..Simulation import Simulation

class SimpleSimulation(Simulation):
    """A simplified simulation to test how simulation objects work
    """
    def __init__(self, *args, **kwargs):

        def_opts = {'sigma': [float, 1.0, None, None, '', 'Variance'],
                    'nDOF': [int, 10, None, None, '', 'Number of DOFs']}

        req_models = {'simp': SimpleModel}
        Simulation.__init__(self, req_models,
                            name="Simplified simulation for testing",
                            def_opts=def_opts, *args, **kwargs)

    def get_sigma(self, model):
        """Dummy get sigma
        """
        return np.diag(np.ones(
            self.get_option('nDOF')) * self.get_option('sigma'))
    # end

    def _on_check_models(self, models):
        return (models['simp'],)

    def _on_call(self, models):
        """Dummy simeriment
        """
        sim_model = models
        x_list = np.arange(self.get_option('nDOF'))
        mean_fn = IUSpline(x_list, np.array(sim_model(x_list)))
        return x_list, [np.array(sim_model(x_list))],\
            {'mean_fn': mean_fn, 'tau': 0}

    def compare(self, data1, data2):
        """Comparison of data
        """
        return data2[1][0] - data1[2]['mean_fn'](data2[0] - data2[2]['tau'])

    def shape(self):
        """Shape
        """
        return self.get_option('nDOF')


class TestSimulation(unittest.TestCase):
    """Test of the simeriment class
    """

    def test_instantiation(self):
        """Tests that the class can instantiate correctly
        """

        req_models = {'def': PhysicsModel}
        sim = Simulation(req_models)

        self.assertIsInstance(sim, Simulation)
    # end


class TestModelDictionary(unittest.TestCase):
    """Tests of the model dictionary
    """
    def test_instantiation(self):
        """Tests proper and improper instantiation of models
        """

        model_dict = {'def': PhysicsModel}

        # Tests good useage
        tmp = Simulation(model_dict)

        # Tests passing no dict
        with self.assertRaises(TypeError) as inst:
            tmp = Simulation()

        # Tests passing not a dict
        with self.assertRaises(IOError) as inst:
            tmp = Simulation(PhysicsModel)

        # Tests passing a dict with wrong values
        with self.assertRaises(IOError) as inst:
            tmp = Simulation({'def': 'not a model'})

        # Tests passing a dict with wrong values
        with self.assertRaises(IOError) as inst:
            tmp = Simulation({'def': Struc})

    def test_check_model(self):
        """Tests proper and improper calling using model dict
        """

        # Create simple Simulation, dependent on SimpleModel
        sim = SimpleSimulation()

        sim_model = SimpleModel([2, 1])

        # Test proper useage when calling
        data = sim({'simp': sim_model})

        # Call with no models
        with self.assertRaises(TypeError):
            data = sim()

        # Call with not a dict
        with self.assertRaises(IOError):
            data = sim(sim_model)

        # Call with missing key
        with self.assertRaises(KeyError):
            data = sim({'wrongkey': sim_model})

        # Call with wrong values
        with self.assertRaises(IOError):
            data = sim({'simp': Struc})

        with self.assertRaises(IOError):
            data = sim({'simp': 'not a model'})


class TestSimpleSimulation(unittest.TestCase):
    """Tests a simple simulation
    """

    def setUp(self):
        self.simSimp = SimpleSimulation()

    def test_init(self):
        """Test that the object instantiated correctly
        """
        self.assertEqual(self.simSimp.name, "Simplified simulation for testing")
        self.assertEqual(self.simSimp.get_option('nDOF'), 10)
        self.assertEqual(self.simSimp.get_option('sigma'), 1.0)

    def test_shape(self):
        """Tests the shape function
        """
        self.assertEqual(self.simSimp.shape(), 10)

    def test_sigma(self):
        """Test variance matrix
        """
        npt.assert_equal(self.simSimp.get_sigma({'simp': SimpleModel([2, 1])}),
                         np.diag(np.ones(10)))

    def test_call(self):
        """Test of the function call
        """
        models = {'simp': SimpleModel([2, 1])}
        data = self.simSimp(models)

        self.assertEqual(len(data), 3)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertEqual(len(data[1]), 1)
        self.assertIsInstance(data[1][0], np.ndarray)
        self.assertIsInstance(data[2], dict)
        self.assertTrue('mean_fn' in data[2])
        self.assertIsInstance(data[2]['mean_fn'], IUSpline)

        xx = np.arange(10)
        npt.assert_equal(data[0], xx)
        npt.assert_equal(data[1][0], (2 * xx)**2 + 1 * xx)

    def test_compare(self):
        """Test the compare function
        """
        xx = np.arange(10)
        yy = (3 * xx)**2 + xx

        models = {'simp': SimpleModel([2, 1])}
        model_data = self.simSimp(models)
        print(model_data)
        print(yy)

        res = self.simSimp.compare(model_data, (xx, (yy,), {'tau': 0}))

        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), 10)
        npt.assert_array_almost_equal(5 * xx**2, res)

    def test_sens(self):
        """Test sensitivity calculation
        """
        models = {'simp': SimpleModel([2, 1])}

        t0 = time.time()
        sens = self.simSimp.get_sens(models, 'simp')
        t0 = time.time() - t0
        print('Serial sense took {:f}'.format(t0))

        self.assertIsInstance(sens, np.ndarray)
        self.assertEqual(sens.shape, (10, 2))

        indep = np.arange(10)
        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1 * indep -
                             (2 * indep)**2 - indep,
                             (2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - 1 * indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02]])

        true_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T
        npt.assert_array_almost_equal(sens, true_sens, decimal=8)

    @unittest.expectedFailure
    def test_pll_sens(self):
        """Test sensitivity calculation
        """
        models = {'simp': SimpleModel([2, 1])}

        t0 = time.time()
        sens = self.simSimp.get_sens_pll(models, 'simp')
        t0 = time.time() - t0
        print('Parallel sense took {:f}'.format(t0))

        self.assertIsInstance(sens, np.ndarray)
        self.assertEqual(sens.shape, (10, 2))

        indep = np.arange(10)
        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1 * indep -
                             (2 * indep)**2 - indep,
                             (2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - 1 * indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02]])
        true_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        npt.assert_array_almost_equal(sens, true_sens, decimal=8)

if __name__ == '__main__':
    unittest.main(verbosity=4)
