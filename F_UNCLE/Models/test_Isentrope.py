#!/usr/bin/pyton
"""

test_Isentrope

Collection of tests for Isentrope objects

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraser (AMF)

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
import math
import sys
import os

# =========================
# Python Packages
# =========================
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.PhysicsModel import GausianModel
    from F_UNCLE.Models.Isentrope import EOSModel, EOSBump, Isentrope
else:
    from ..Utils.PhysicsModel import GausianModel
    from .Isentrope import EOSModel, EOSBump, Isentrope
# end


class TestIsentrope(unittest.TestCase):
    """Test of the isentrope object
    """
    def test_standard_instantiation(self):
        """Test basic use of isentrope
        """
        model = Isentrope()

        self.assertEqual(model.get_option('spline_N'), 50)
        self.assertEqual(model.get_option('spline_min'), 0.1)
        self.assertEqual(model.get_option('spline_max'), 1.0)
        self.assertEqual(model.get_option('spline_end'), 4)
    # end

    def test_custom_instantiation(self):
        """Test instantiated with non default values
        """
        model = Isentrope(name="Francis", spline_N=55,
                          spline_sigma=0.0)

        self.assertEqual(model.name, "Francis")
        self.assertEqual(model.get_option('spline_N'), 55)


# end

class TestEosModel(unittest.TestCase):
    """Test the spline EOS functions
    """
    def setUp(self):
        """Create test EOS function
        """

        p_fun = lambda v: 2.56e9 / v**3

        self.p_fun = p_fun

    def test_standard_instantiation(self):
        """Test normal instantiation of the EOS
        """

        eos = EOSModel(self.p_fun)

        print(eos)

        self.assertEqual(eos.get_option('spline_sigma'), 5e-3)
        self.assertEqual(eos.get_option('precondition'), False)
    # end

    def test_bad_instantiation(self):
        """Test improper instantiation
        """

        with self.assertRaises(TypeError):
            EOSModel(2.5)
        # end

    # end

    def test_custom_instantiation(self):
        """Tests instantiation with non default values
        """
        eos = EOSModel(self.p_fun, name="Ajax", spline_N=65,
                       spline_sigma=2.5e-3)

        self.assertEqual(eos.name, "Ajax")
        self.assertEqual(eos.get_option('spline_N'), 65)
        self.assertEqual(eos.get_option('spline_sigma'), 2.5e-3)

    def test_spline_get_t(self):
        """Test spline interaction method, get knots
        """

        # generate an EOS spline
        eos = EOSModel(self.p_fun)

        # get t
        t_list = eos.get_t()

        # check it is a numpy array

        self.assertIsInstance(t_list, np.ndarray)

        # check it is long enough
        self.assertEqual(
            eos.get_option('spline_N') + eos.get_option('spline_end'),
            len(t_list)
        )

        # check is spans the v min and v max limits
        self.assertEqual(eos.get_option('spline_min'), t_list.min())
        self.assertEqual(eos.get_option('spline_max'), t_list.max())

    def test_spline_get_c(self):
        """Test spline interaction method, get coefficients
        """
        # generate an EOS spline
        eos = EOSModel(self.p_fun)

        # get c
        c_list = eos.get_c()

        self.assertEqual(eos.get_option('spline_N'), len(c_list))

    def test_spline_new_c(self):
        """Test spline interaction method, set new coefficients
        """

        eos = EOSModel(self.p_fun)

        # get c
        c_list = eos.get_dof()

        # add 0.1 to all c
        # get a new spline with updated c
        new_eos = eos.update_dof(c_list + 0.1)

        # check it matches
        npt.assert_array_equal(new_eos.get_dof(), c_list + 0.1)
        npt.assert_array_equal(c_list, eos.get_dof())

    def test_eos_get_sigma(self):
        """ Tests that the co-variance matrix is generated correctly
        """

        eos = EOSModel(self.p_fun)

        sigma_eos = eos.get_sigma()

        n_spline = eos.get_option('spline_N')
        spline_var = eos.get_option('spline_sigma')

        self.assertEqual(sigma_eos.shape, (n_spline, n_spline))
        npt.assert_array_equal(np.diag(sigma_eos),
                               (eos.get_c() * spline_var)**2)


class TestBumpEOS(unittest.TestCase):
    """Test of the bump EOS
    """
    def test_instantiation(self):
        """Tests that the object is properly instantiated
        """
        bump_eos = EOSBump()

        self.assertEqual(bump_eos.get_option('const_C'), 2.56e9)
        self.assertEqual(bump_eos.get_option('bumps')[0][0], 0.4)
        self.assertEqual(bump_eos.get_option('bumps')[0][1], 0.1)
        self.assertEqual(bump_eos.get_option('bumps')[0][2], 0.25)

    def test_custom_instatntiation(self):
        """Test non default instantiation
        """
        bump_eos = EOSBump(const_C=6e-3,
                           bumps=[(0.25, 0.07, 0.3),
                                  (0.35, 0.08, 0.4),
                                  ])

        bump_eos(0.25)

        self.assertEqual(bump_eos.get_option('const_C'), 6e-3)
        self.assertEqual(bump_eos.get_option('bumps')[0][0], 0.25)
        self.assertEqual(bump_eos.get_option('bumps')[0][1], 0.07)
        self.assertEqual(bump_eos.get_option('bumps')[0][2], 0.3)
        self.assertEqual(bump_eos.get_option('bumps')[1][0], 0.35)
        self.assertEqual(bump_eos.get_option('bumps')[1][1], 0.08)
        self.assertEqual(bump_eos.get_option('bumps')[1][2], 0.4)

    def test_bad_instatntiation(self):
        """Test improper instantiation
        """
        pass

    def test_call(self):
        """Test that the bump EOS can be called
        """
        bump_eos = EOSBump()

        vol = np.logspace(np.log10(.1), np.log10(1), 50)

        pressure = bump_eos(vol)

        self.assertIsInstance(pressure, np.ndarray)
        self.assertEqual(50, len(pressure))

    def test_derivative(self):
        """Test the derivative function
        """
        bump_eos = EOSBump()

        vol = np.logspace(np.log10(.1), np.log10(1), 50)

        pressure = bump_eos.derivative()(vol)

        self.assertIsInstance(pressure, np.ndarray)
        self.assertEqual(50, len(pressure))

    def test_bad_derivative(self):
        """Tests that derivative errors are caught
        """
        bump_eos = EOSBump()

        vol = np.logspace(np.log10(.1), np.log10(1), 50)

        with self.assertRaises(IOError):
            pressure = bump_eos.derivative(order=2)(vol)
        # end

        p_fun = lambda v: 2.56e9 / v**3


class TestPlot(unittest.TestCase):
    def setUp(self):
        p_fun = lambda v: 2.56e9 / v**3

        self.indep = np.linspace(0.2, 1.0, 200)
        self.model1 = EOSModel(p_fun)

        dof = self.model1.get_dof()
        self.model2 = self.model1.update_dof(dof * 0.95)
    # end

    def test_plot(self):
        """Test of basic plotting
        """

        self.model1.plot()

        plt.figure()
        ax1 = plt.gca()
        self.model1.plot(axes=ax1, style='-r')

        with self.assertRaises(TypeError):
            self.model1.plot(axes='invalid axis object')
        # end

    # @unittest.skip('no')
    def test_diff(self):
        """Test diff plotting
        """

        self.model1.plot_diff([self.model1, self.model2])

        ax1 = plt.figure().gca()
        self.model1.plot_diff([self.model1, self.model2], axes=ax1)

    def test_basis(self):
        self.model1.plot_basis()

if __name__ == '__main__':
    unittest.main(verbosity=4)
