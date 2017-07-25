#!/usr/bin/pyton
"""

test_Experiment

Tests of the Experiment abstract class

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
import matplotlib.pyplot as plt 
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline
# =========================
# Custom Packages
# =========================
from ..Struc import Struc
from ..PhysicsModel import PhysicsModel
from .test_PhysicsModel import SimpleModel
from .test_Simulation import SimpleSimulation
from ..Simulation import Simulation
from ..Experiment import Experiment, GaussianExperiment


class SimpleExperiment(GaussianExperiment):
    """This creates a set of synthetic experimental data.

    It follows the same functional form as the SimpleModel and 
    SimpleSimulation but introduces a time shift

    """
    def _get_data(self, *args, **kwargs):
        """Generates some synthetic data

        Independent variable spans form 1 to 11
        Dependent variable is :math:`3 (xx-1)^2 + xx -1`

        This allows us to test the trigger as well
        """
        xx = np.linspace(0, 10, 5)
        yy = (4 * (xx))**2 + 2 * (xx) 

        return xx + 1, yy, None

    def simple_trigger(self, x, y):
        """Rather than test the trigger funciton, this returns the known offset
        """

        return x[0]

    def get_splines(self, x_data, y_data, var_data=0):
        """Overloads the base spline generateion which cannot deal with the
        smooth experiment
        """

        return IUSpline(x_data, y_data), None
    
class TestSimpleExperiment(unittest.TestCase):
    """Exercises the functionality of Expeiment using simple classes
    """
    
    def setUp(self):
        """Creates a SimpleSimulation
        """
        self.models = {'simp': SimpleModel([2, 1])}
        self.simSimp = SimpleSimulation()
        

    def test_init(self):
        """Tests that the base experiment can be instantiated
        """

        exp = SimpleExperiment()
        self.assertIsInstance(exp, GaussianExperiment)
        
    def test_shape(self):
        """Tests that the correct shape of the data is returned
        """

        exp = SimpleExperiment()
        self.assertEqual(5, exp.shape())
        
    def test_sigma(self):
        """Tests that the correct covariance matrix is generated
        """

        exp = SimpleExperiment(exp_var=0.01)

        true_sigma = np.diag((exp.data[1] * 0.01)**2)

        npt.assert_array_almost_equal(true_sigma, exp.get_sigma())

    def test_align(self):
        """Test of the align function
        """
        exp = SimpleExperiment()

        # Generate some simulation data
        sim_data = self.simSimp(self.models)
        stored_data = copy.deepcopy(sim_data)
        # Align the simulation data with the experiment
        aligned_data = exp.align(sim_data)

        # self.plot('aligned', exp, aligned_data)
        self.assertEqual(aligned_data[2]['tau'], 1)
        npt.assert_array_equal(aligned_data[0], exp.data[0])
        
    def test_compare(self):
        """Tests that the experiment can be compared to aligned simulation data
        """

        exp = SimpleExperiment()

        # Generate some simulation data
        sim_data = self.simSimp(self.models)
        stored_data = copy.deepcopy(sim_data)
        sim_data = exp.align(sim_data)
        epsilon = exp.compare(sim_data)

        npt.assert_array_almost_equal((4**2 - 2**2) * (exp.data[0]-1)**2
                                      + (2-1) * (exp.data[0]-1), epsilon)
        

    def test_pq(self):
        """Tests that the experiment can generate the p and q matrix
        """
        exp = SimpleExperiment()

        # Generate some simulation data
        sim_data = self.simSimp(self.models)

        # Align the data so that it is evaluated at the same time steps as
        # the experiment
        aligned_data = exp.align(sim_data)

        # Get the sensitivity matrix
        sens_matrix = self.simSimp.get_sens(self.models, ['simp'], aligned_data)

        # Get the error between the sim and experiment
        epsilon = exp.compare(sim_data)
        
        p_mat, q_vec = exp.get_pq(self.models, ['simp'], sim_data, sens_matrix,
                                  scale=False)
        sigma_mat = exp.get_sigma()
        p_mat_hat = np.dot(np.dot(sens_matrix.T, np.linalg.inv(sigma_mat)),
                           sens_matrix)

        npt.assert_array_equal(p_mat, p_mat_hat,
                               err_msg="P matrix calculated incorrectly")

        q_vec_hat = -np.dot(np.dot(epsilon, np.linalg.inv(sigma_mat)),
                           sens_matrix)
        npt.assert_array_equal(q_vec, q_vec_hat,
                               err_msg="q vector calculated incorrectly")
        
    def test_log_like(self):
        """Tests the log likelyhood of the experiment given simulation data
        """

        
        exp = SimpleExperiment()

        # Generate some simulation data
        sim_data = self.simSimp(self.models)

        # Align the data so that it is evaluated at the same time steps as
        # the experiment
        aligned_data = exp.align(sim_data)
        
        epsilon = exp.compare(sim_data)
        sigma_mat = exp.get_sigma()

        log_like_hat = -0.5 * np.dot(epsilon, np.dot(np.linalg.inv(sigma_mat),
                                                     epsilon))

        log_like = exp.get_log_like(sim_data)

        self.assertEqual(log_like_hat, log_like)
        
    def plot(self, name, exp, simdata):
        """Plots a comparisson
        """

        fig = plt.figure()
        ax1 = fig.gca()

        ax1.plot(exp.data[0], exp.data[1], 'o', label='Experiment')
        ax1.plot(simdata[0], simdata[1][0], '-', label='Simulation')

        fig.savefig(name + '.pdf')
