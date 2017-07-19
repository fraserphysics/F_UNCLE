# !/usr/bin/pthon2
"""Test of PhysicsModels

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
import unittest
from math import pi, erf, log
import pdb
# =========================
# Python Installed Packages
# =========================

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Module Packages
# =========================

from ...Utils.Struc import Struc
from ...Utils.PhysicsModel import PhysicsModel
from ..Ptw import Ptw
from ..SimpleStr import SimpleStr

class TestPtw(unittest.TestCase):
    """Test of the Ptw class
    """

    def setUp(self):
        """Initial setup
        """
        pass

    def test_basic_operation(self):
        """Tests normal functioning of the class

        """

        flow_stress_model = Ptw()

        flow_stress, yield_stress = flow_stress_model(300.0, 1E2, 'Cu')

        print(flow_stress_model)
        self.assertIsInstance(flow_stress, float)
        self.assertIsInstance(yield_stress, float)

    @unittest.expectedFailure
    def test_all_materials(self):
        """Tests that all valid materials will run
        """
        flow_stress_model = Ptw()

        for mat in ['Cu', 'U', 'Ta', 'V', 'Mo', 'Be', 'SS_304', 'SS_21-6-9']:
            flow_stress, yield_stress = flow_stress_model(300.0, 1E2, mat)
            self.assertIsInstance(flow_stress, float)
            self.assertIsInstance(yield_stress, float)
            self.assertFalse(np.isnan(flow_stress),
                             msg="Flow stress was nan for {}".format(mat))
            self.assertFalse(np.isnan(yield_stress),
                             msg="Flow stress was nan for {}".format(mat))
            # end

    def test_temp_bounds(self):
        """Tests that the preprocessor can deal with out of bound temperatures
        """
        flow_stress_model = Ptw()

        # Tests low temperatures
        with self.assertRaises(ValueError):
            flow_stress_model(-1.0, 1E2, 'Cu')
        # end

        # Tests high temperatures
        with self.assertRaises(ValueError):
            flow_stress_model(2001.0, 1E2, 'Cu')
        # end

    def test_override(self):
        """Tests that overrides work properly
        """

        flow_stress_model = Ptw()

        # run with no override
        data = flow_stress_model(300.0, 1E2, 'Cu')

        self.assertEqual(flow_stress_model.get_option('gamma'), 0.00001)
        self.assertEqual(flow_stress_model.get_option('beta'), 0.25)

        # run with the override
        data = flow_stress_model(300.0, 1E2, 'Cu', gamma=0.00002, beta=0.30)

        self.assertEqual(flow_stress_model.get_option('gamma'), 0.00002)
        self.assertEqual(flow_stress_model.get_option('beta'), 0.30)

    def test_bad_override(self):
        """Tests that improper overrides do not work
        """

        flow_stress_model = Ptw()

        with self.assertRaises(KeyError):
            data = flow_stress_model(300.0, 1E2, 'Cu', potato=2.0)
        # end

    def test_bad_material(self):
        """Test that bad material specifications are caught
        """

        flow_stress_model = Ptw()

        # Tests low rates
        with self.assertRaises(IOError):
            flow_stress_model(300.0, 1E2, 'potato')
        # end

    def test_strain_bounds(self):
        """Tests that the preprocessor can deal with out of bounds strain rates
        """
        flow_stress_model = Ptw()

        # Tests low rates
        with self.assertRaises(ValueError):
            flow_stress_model(300.0, -0.1, 'Cu')
        # end

        # Tests high rates
        with self.assertRaises(ValueError):
            flow_stress_model(300.0, 1.1E12, 'Cu')
        # end

class TestSimpleStr(unittest.TestCase):
    """Test of the simplified strength model
    """

    def setUp(self):
        """
        """

        self.Cu_coeff = [101E9, 0.1]
        
    def test_instantiation(self):
        """Test that the object can be printed 
        """

        strmod = SimpleStr(self.Cu_coeff)

    def test_alt_instantiation(self):
        """Alternative instantiation methods
        """

        strmod = SimpleStr((90E9, 1.0))
        strmod = SimpleStr(np.array((90E9, 1.0)))        

    def test_bad_instantiation(self):
        """Object raises the correct error when incorret parameters are given 
        """

        with self.assertRaises(ValueError):
            strmod = SimpleStr([90E9])

        with self.assertRaises(ValueError):
            strmod = SimpleStr([90E9, 0.0, 0.0])
            
    def test_print(self):

        strmod = SimpleStr(self.Cu_coeff)
        print(strmod)

    def test_get_sigma(self):

        strmod = SimpleStr(self.Cu_coeff)
        strmod.get_sigma()
        
    def test_set_dof(self):

        strmod = SimpleStr(self.Cu_coeff)
        newmod = strmod.update_dof([90E9, 2.0])

        self.assertFalse(strmod is newmod)

        dof = strmod.get_dof()

        self.assertEqual(dof[0], self.Cu_coeff[0])
        self.assertEqual(dof[1], self.Cu_coeff[1])

        dof = newmod.get_dof()

        self.assertEqual(dof[0], 90E9)
        self.assertEqual(dof[1], 2.0)
        

    def test_get_dof(self):

        strmod = SimpleStr(self.Cu_coeff)

        dof = strmod.get_dof()

        self.assertEqual(dof[0], self.Cu_coeff[0])
        self.assertEqual(dof[1], self.Cu_coeff[1])
        
    def test_call(self):

        strmod = SimpleStr(self.Cu_coeff)

        epsilon = np.array([1E-3, 2E-4])
        epsilon_dot = np.array([1E-2, 2E-2])
        sigma = strmod(epsilon, epsilon_dot)

        np.testing.assert_array_equal(
            sigma,
            self.Cu_coeff[0] * epsilon + self.Cu_coeff[1] * epsilon_dot
        )
    
    def test_shape(self):
        strmod = SimpleStr(self.Cu_coeff)

        shape = strmod.shape()

        self.assertEqual(shape, 2)
if __name__ == '__main__':
    unittest.main(verbosity=4)
