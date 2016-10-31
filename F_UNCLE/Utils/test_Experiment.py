#!/usr/bin/pyton
"""

test_Experiment

Tests of the experiment Class

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
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import numpy.testing as npt
# =========================
# Custom Packages
# =========================
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Struc import Struc
    from F_UNCLE.Utils.PhysicsModel import PhysicsModel
    from F_UNCLE.Utils.test_PhysicsModel import SimpleModel
    from F_UNCLE.Utils.Experiment import Experiment, GausianExperiment

else:
    from .Struc import Struc
    from .PhysicsModel import PhysicsModel
    from .test_PhysicsModel import SimpleModel
    from .Experiment import Experiment, GausianExperiment
# end


class SimpleExperiment(GausianExperiment):
    """A simplified experiment to test how experiment objects work
    """
    def __init__(self, *args, **kwargs):

        def_opts = {'sigma': [float, 1.0, None, None, '', 'Variance'],
                    'nDOF': [int, 10, None, None, '', 'Number of DOFs']}

        req_models = {'simp': SimpleModel}
        Experiment.__init__(self, req_models,
                            name="Simplified experiment for testing",
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
        """Dummy experiment
        """
        sim_model = models
        x_list = np.arange(self.get_option('nDOF'))
        return x_list, [np.array(sim_model(x_list))], None

    def compare(self, indep, dep, model_data):
        """Comparison of data
        """
        return dep - model_data[1][0]

    def shape(self):
        """Shape
        """
        return self.get_option('nDOF')


class TestExperiment(unittest.TestCase):
    """Test of the experiment class
    """

    def test_instantiation(self):
        """Tests that the class can instantiate correctly
        """

        req_models = {'def': PhysicsModel}
        exp = Experiment(req_models)

        self.assertIsInstance(exp, Experiment)
    # end


class TestModelDictionary(unittest.TestCase):
    """Tests of the model dictionary
    """
    def test_instantiation(self):
        """Tests proper and improper instantiation of models
        """

        model_dict = {'def': PhysicsModel}

        # Tests good useage
        tmp = Experiment(model_dict)

        # Tests passing no dict
        with self.assertRaises(TypeError) as inst:
            tmp = Experiment()

        # Tests passing not a dict
        with self.assertRaises(IOError) as inst:
            tmp = Experiment(PhysicsModel)

        # Tests passing a dict with wrong values
        with self.assertRaises(IOError) as inst:
            tmp = Experiment({'def': 'not a model'})

        # Tests passing a dict with wrong values
        with self.assertRaises(IOError) as inst:
            tmp = Experiment({'def': Struc})

    def test_model_attribute(self):
        """Tests the use of a model attribute
        """

        # Instatntiate a SimpleExperiment with a model_attribute
        exp_model = SimpleModel([2, 1])
        sim_model = SimpleModel([4, 2])

        exp = SimpleExperiment(model_attribute=exp_model)

        # Call without passing a model dict
        data = exp()
        xx = np.arange(10)
        # Check that the output corresponds to the model attribute
        npt.assert_array_equal(data[1][0], (2 * xx)**2 + 1 * xx)

        # Call with a model dict
        data = exp({'simp': sim_model})

        # Check that the output corresponds to the model attribute,
        # not the dict
        npt.assert_array_equal(data[1][0], (2 * xx)**2 + 1 * xx)

    def test_check_model(self):
        """Tests proper and improper calling using model dict
        """

        # Create simple Experiment, dependent on SimpleModel
        exp = SimpleExperiment()

        exp_model = SimpleModel([2, 1])

        # Test proper useage when calling
        data = exp({'simp': exp_model})

        # Call with no models
        with self.assertRaises(TypeError):
            data = exp()

        # Call with not a dict
        with self.assertRaises(IOError):
            data = exp(exp_model)

        # Call with missing key
        with self.assertRaises(KeyError):
            data = exp({'wrongkey': exp_model})

        # Call with wrong values
        with self.assertRaises(IOError):
            data = exp({'simp': Struc})

        with self.assertRaises(IOError):
            data = exp({'simp': 'not a model'})


class TestSimpleExperiment(unittest.TestCase):
    """Tests a simple experiment
    """

    def setUp(self):
        self.expSimp = SimpleExperiment()

    def test_init(self):
        """Test that the object instantiated correctly
        """
        self.assertEqual(self.expSimp.name, "Simplified experiment for testing")
        self.assertEqual(self.expSimp.get_option('nDOF'), 10)
        self.assertEqual(self.expSimp.get_option('sigma'), 1.0)

    def test_shape(self):
        """Tests the shape function
        """
        self.assertEqual(self.expSimp.shape(), 10)

    def test_sigma(self):
        """Test variance matrix
        """
        npt.assert_equal(self.expSimp.get_sigma({'simp': SimpleModel([2, 1])}),
                         np.diag(np.ones(10)))

    def test_call(self):
        """Test of the function call
        """
        models = {'simp': SimpleModel([2, 1])}
        data = self.expSimp(models)

        self.assertEqual(len(data), 3)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertEqual(len(data[1]), 1)
        self.assertIsInstance(data[1][0], np.ndarray)
        self.assertTrue(data[2] is None)

        xx = np.arange(10)
        npt.assert_equal(data[0], xx)
        npt.assert_equal(data[1][0], (2 * xx)**2 + 1 * xx)

    def test_compare(self):
        """Test the compare function
        """
        xx = np.arange(10)
        yy = (2 * xx)**2

        models = {'simp': SimpleModel([2, 1])}
        model_data = self.expSimp(models)

        res = self.expSimp.compare(xx, yy, model_data)

        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), 10)
        npt.assert_equal(-xx, res)

if __name__ == '__main__':
    unittest.main(verbosity=4)