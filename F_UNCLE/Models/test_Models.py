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

if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Struc import Struc
    from F_UNCLE.Utils.PhysicsModel import PhysicsModel
    from F_UNCLE.Models.Ptw import Ptw

else:
    from ..Utils.Struc import Struc
    from ..Utils.PhysicsModel import PhysicsModel
    from .Ptw import Ptw

# end
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

if __name__ == '__main__':
    unittest.main(verbosity=4)