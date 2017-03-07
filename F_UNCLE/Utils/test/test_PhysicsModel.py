# /usr/bin/pyton
"""

test_PhysicsModel

Test class for the PhysicsModel object

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
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import numpy.testing as npt

# =========================
# Custom Packages
# =========================
from ..Struc import Struc
from ..PhysicsModel import PhysicsModel, GaussianModel


class SimpleModel(GaussianModel):
    """Simplified physics model for testing
    """
    def __init__(self, prior, name="Simplified physics model"):
        """Create dummy physics model
        """
        def_opts = {'nDOF': [int, 2, None, None, '', 'Number of DOF'],
                    'sigma': [float, 1.0, None, None, '', 'Variance']
                    }

        PhysicsModel.__init__(self, None, name=name, def_opts=def_opts)
        self.dof = np.array(prior, dtype=np.float64)
        self.prior = copy.deepcopy(self)

    def _on_str(self):
        """Prints the dof
        """

        out_str = "dof\n"
        for i, value in enumerate(self.dof):
            out_str += '{:d} {:f}'.format(i, value)
        # end

        return out_str
    def get_sigma(self):
        """Get variance
        """

        return np.diag(np.ones(self.get_option('nDOF')) *
                       self.get_option('sigma'))

    def get_dof(self):
        """Get dofs
        """

        return np.array(self.dof)

    def get_scaling(self):
        return np.diag(np.ones(2))

    def update_dof(self, new_dof):
        """Gives a new instance with updated dof
        """

        new_model = copy.deepcopy(self)
        new_model.dof = copy.deepcopy(np.array(new_dof, np.float64))
        return new_model

    def shape(self):
        """Return the shape
        """
        return self.get_option('nDOF')

    def __call__(self, x):
        """Run the model
        """

        return (self.dof[0] * x)**2 + self.dof[1] * x


class TestPhysModel(unittest.TestCase):
    """Test of PhysicsModel object
    """
    def test_standard_instantiation(self):
        """Tests that the model can be instantiated
        """
        model = PhysicsModel(prior=3.5)

        self.assertIsInstance(model, PhysicsModel)

    def test_update_prior(self):
        """Tests setting and updating the prior
        """
        model = PhysicsModel(prior=[3.5])

        npt.assert_equal(model.prior, [3.5])

        new_prior = [2.5]
        new_model = model.update_prior(new_prior)

        npt.assert_equal(new_model.prior, new_prior,
                         err_msg="Test that new prior set correctly")

        self.assertFalse(model is new_model,
                         msg="Test that update_prior gives a new instance")
        self.assertFalse(new_model.prior is new_prior,
                         msg="Test that prior is not linked to passed value")

    def test_bad_update_prior(self):
        """Tests bad use of prior setting
        """
        model = PhysicsModel(prior=3.5)

        with self.assertRaises(TypeError):
            new_model = model.update_prior('two point five')
        # end


class TestSimpleModel(unittest.TestCase):
    """Test of SimpleModel
    """
    def setUp(self):
        self.simple = SimpleModel([2, 1])

    def test_init(self):
        """Test the object was instantiated correctly
        """

        self.assertEqual(self.simple.name, "Simplified physics model")
        self.assertEqual(self.simple.get_option('nDOF'), 2)
        self.assertEqual(self.simple.get_option('sigma'), 1.0)
        npt.assert_equal(self.simple.dof, [2, 1])

    def test_get_dof(self):
        """Test the object can return its dof
        """

        npt.assert_equal(self.simple.get_dof(), [2, 1])

    def test_update_dof(self):
        """Test the object returns a distinct copy when updated
        """
        new_dof = [4, 2]
        new_model = self.simple.update_dof(new_dof)
        npt.assert_equal(new_model.get_dof(), [4, 2],
                         err_msg='Test that the new model was set correctly')

        npt.assert_equal(self.simple.get_dof(), [2, 1],
                         err_msg='Test that the original model was not changed')

        self.assertFalse(new_model is self.simple)
        self.assertFalse(new_model.get_dof() is new_dof)

    def test_shape(self):
        """Test shape function
        """

        self.assertEqual(self.simple.shape(), 2)

    def test_sigma(self):
        """Tests the variance matric
        """
        npt.assert_equal(self.simple.get_sigma(),
                         np.diag(np.ones(2)))

if __name__ == '__main__':

    unittest.main(verbosity=4)
