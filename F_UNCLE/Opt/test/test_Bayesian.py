# /usr/bin/pyton
"""

test_Bayesian

Test class for the Bayesian analysis

To run all tests, issue "nosetests" in the root directory, ie ../..

To run this file, issue "python test_Bayesian.py" in this directory.

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

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
import copy
import math
import pdb
import pytest
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import numpy.testing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# =========================
# Custon Packages
# =========================
from ...Utils.Experiment import Experiment
from ...Utils.Simulation import Simulation
from ...Utils.PhysicsModel import PhysicsModel
from ...Utils.Struc import Struc
from ..Bayesian import Bayesian

# ================
# Testing Packages
# ================
from ...Experiments.GunModel import Gun, GunExperiment
from ...Experiments.Stick import Stick, StickExperiment
from ...Models.Isentrope import EOSModel, EOSBump
from ...Utils.test.test_PhysicsModel import SimpleModel
from ...Utils.test.test_Experiment import SimpleExperiment
from ...Utils.test.test_Simulation import SimpleSimulation


#@pytest.mark.skip('Skipping long tests')
class TestBayesian(unittest.TestCase):
    """Test class for the bayesian object
    """
    def setUp(self):
        """Setup script for each test
        """
        # Initial estimate of prior functional form
        init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

        # Create the model and *true* EOS
        self.eos_model = EOSModel(init_prior, name="Default EOS Model")
        self.eos_true = EOSBump()

        # Create the objects to generate simulations and
        # pseudo experimental data
        self.exp1 = GunExperiment(model=self.eos_true)
        self.sim1 = Gun(name="Default Gun Simulation")

        self.exp2 = StickExperiment(model=self.eos_true)
        self.sim2 = Stick()

    # end

    def test_instantiation(self):
        """Test that the object can instantiate correctly
        """

        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1]},
                         models={'eos': self.eos_model},
                         opt_key='eos')

        self.assertIsInstance(bayes, Bayesian)
        self.assertIsInstance(bayes.models['eos'], EOSModel)
        self.assertEqual(bayes.models['eos'].name, "Default EOS Model")
        self.assertIsInstance(bayes.simulations['Gun']['sim'], Gun)
        self.assertEqual(bayes.simulations['Gun']['sim'].name,
                         "Default Gun Simulation")
        self.assertIsInstance(bayes.simulations['Gun']['exp'], GunExperiment)
        self.assertEqual(bayes.simulations['Gun']['exp'].name,
                         "Data Experiment")

    # end

    def test_check_inputs(self):
        """Checks that the method accepts all valid input and
        rejects all invalid input

        """
        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1]},
                         models={'eos': self.eos_model},
                         opt_key='eos')

        # Test passing a valid eos model in a dict
        models, sim = bayes._check_inputs({'eos': self.eos_model},
                                          {'Gun': [self.sim1, self.exp1]})
        self.assertIsInstance(models, dict,
                              msg="Passing dict did not yeild dict")
        self.assertEqual(len(models), 1,
                         msg="Did not create 1 element dict")
        self.assertFalse('def' in models,
                         msg="def should not be a valid model key")
        self.assertTrue('eos' in models,
                        msg="eos should be a valid model key")
        self.assertIsInstance(models['eos'], type(self.eos_model))
        self.assertFalse(self.eos_model is models['eos'],
                         msg='the model is still linked to the orignal'
                             'instance')

        # # Test passing a valid eos model but not in a dict
        # model, sim = bayes._check_inputs(self.eos_model,
        #                                  {'Gun': [self.sim1, self.exp1]})

        # self.assertIsInstance(model, dict,
        #                       msg="Passing PhysicsModel did not yeild dict")
        # self.assertEqual(len(model), 1,
        #                  msg="Did not create 1 element dict")
        # self.assertTrue('def' in model,
        #                 msg="The default name of a PhysicsModel is def")
        # self.assertFalse('eos' in model,
        #                  msg="eos should no longer be a valid eos model key")

        # Test passing an invalid model
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs('Not a PhysicsModel instance',
                                             {'Gun': [self.sim1, self.exp1]})
        self.assertEqual(cm.exception.args[0][:3], '001')

        # Test passing an invalid model in a dict
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'def':
                                              'Not a PhysicsModel instance'},
                                             {'Gun': [self.sim1, self.exp1]})
        self.assertEqual(cm.exception.args[0][:3], '002')

        # Test passing not a dict to sims
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'eos': self.eos_model},
                                             'Not a proper dict')
        self.assertEqual(cm.exception.args[0][:3], '003')

        # Test passing a dict with invalid valies to sim
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'eos': self.eos_model},
                                             {'Gun': 'not a list'})
        self.assertEqual(cm.exception.args[0][:3], '004')

        # Test passing a dict with too many values
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'eos': self.eos_model},
                                             {'Gun': [self.sim1,
                                                      self.sim1,
                                                      self.sim1]})
        self.assertEqual(cm.exception.args[0][:3], '005')

        # Test passing a dict with not experiments
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'eos': self.eos_model},
                                             {'Gun': [self.sim1, 'not as sim']})
        self.assertEqual(cm.exception.args[0][:3], '006')

        # Test passing a dict of dicts
        model, sim = bayes._check_inputs({'eos': self.eos_model},
                                         {'Gun': {'sim': self.sim1,
                                                  'exp': self.exp1}})

        # Test passing a dict of dicts with wrong keys
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'eos': self.eos_model},
                                             {'Gun': {'not sim': self.sim1,
                                                      'exp': self.exp1}})
        self.assertEqual(cm.exception.args[0][:3], '007')

        # Test passing a dict of dicts with not experiments
        with self.assertRaises(TypeError) as cm:
            model, sim = bayes._check_inputs({'eos': self.eos_model},
                                             {'Gun': {'sim': self.sim1,
                                                      'exp': 'Not experiment'}})
        self.assertEqual(cm.exception.args[0][:3], '008')

    def test_bad_instantiaion(self):
        """Tets impropper instantiation raises the correct errors
        """

        pass
    # end

    def test_singel_case_pq_mod(self):
        """Tests the P and Q matrix generation for a single case
        """
        bayes = Bayesian(models={'eos': self.eos_model},
                         simulations={'Gun': [self.sim1, self.exp1]},
                         opt_key='eos')

        shape = bayes.shape()

        exp_shape = self.exp1.shape()
        sim_shape = self.sim1.shape()
        mod_shape = self.eos_model.shape()

        initial_data = bayes.get_data()
        sens_matrix = bayes._get_sens(initial_data=initial_data)

        P, q = bayes._get_sim_pq(initial_data, sens_matrix)

        self.assertEqual(P.shape, (mod_shape, mod_shape))
        self.assertEqual(q.shape, (mod_shape, ))

    def test_mlt_case_pq_mod(self):
        """Tests the P and Q matrix generation for a multiple case
        """

        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1],
                                      'Stick': [self.sim2, self.exp2]},
                         models={'eos': self.eos_model},
                         opt_key='eos')

        shape = bayes.shape()

        exp_shape = self.exp1.shape() + self.exp2.shape()
        sim_shape = self.sim1.shape() + self.sim2.shape()
        mod_shape = self.eos_model.shape()

        initial_data = bayes.get_data()
        sens_matrix = bayes._get_sens(initial_data=initial_data)
        P, q = bayes._get_sim_pq(initial_data, sens_matrix)

        self.assertEqual(P.shape, (mod_shape, mod_shape))
        self.assertEqual(q.shape, (mod_shape, ))

    def test_gun_case_sens(self):
        """
        """

        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1]},
                         models={'eos': self.eos_model},
                         opt_key='eos')

        shape = bayes.shape()

        exp_shape = self.exp1.shape()
        sim_shape = self.sim1.shape()
        mod_shape = self.eos_model.shape()

        initial_data = bayes.get_data()
        sens_matrix = bayes._get_sens(initial_data=initial_data)
        self.assertEqual(sens_matrix['Gun'].shape, (exp_shape, mod_shape))
        # bayes.plot_sens_matrix(initial_data)

    def test_stick_case_sens(self):
        """Test of the sensitivity of the stick model to the eos
        """

        bayes = Bayesian(simulations={'Stick': [self.sim2, self.exp2]},
                         models={'eos': self.eos_model},
                         opt_key='eos')

        shape = bayes.shape()

        exp_shape = self.exp2.shape()
        sim_shape = self.sim2.shape()
        mod_shape = self.eos_model.shape()

        initial_data = bayes.get_data()

        sens_matrix = bayes._get_sens()
        self.assertEqual(sens_matrix['Stick'].shape, (exp_shape, mod_shape))

    def test_mult_case_sens(self):
        """Test of sens matrix generation for mult models
        """
        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1],
                                      'Stick': [self.sim2, self.exp2]},
                         models={'eos': self.eos_model})

        shape = bayes.shape()

        mod_shape = self.eos_model.shape()

        initial_data = bayes.get_data()

        sens_matrix = bayes._get_sens(initial_data=initial_data)

        self.assertEqual(sens_matrix['Gun'].shape,
                         (self.exp1.shape(), mod_shape))
        self.assertEqual(sens_matrix['Stick'].shape,
                         (self.exp2.shape(), mod_shape))

    def test_local_opt(self):
        """Test the local optimization scheme
        """

        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1],
                                      'Stick': [self.sim2, self.exp2]},
                         models={'eos': self.eos_model},
                         opt_key='eos')
        initial_data = bayes.get_data()

        sens_matrix = bayes._get_sens()
        local_sol = bayes._local_opt(initial_data,
                                     sens_matrix)
        d_hat = np.array(local_sol['x']).reshape(-1)
        self.assertIsInstance(d_hat, (list, np.ndarray))
        self.assertEqual(len(d_hat), self.eos_model.shape())

    def test_single_iteration(self):
        """Test a single iteration of the test
        """
        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1],
                                      'Stick': [self.sim2, self.exp2]},
                         models={'eos': self.eos_model},
                         opt_key='eos')
        bayes.set_option('maxiter', 1)
        out = bayes()

        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], Bayesian)
        self.assertIsInstance(out[1], tuple)
        self.assertEqual(len(out[1]), 2)

    def test_multiple_calls(self):
        """Tests that the results are the same after calling multiple times
        """
        bayes = Bayesian(simulations={'Gun': [self.sim1, self.exp1]},
                         models={'eos': self.eos_model},
                         opt_key='eos')

        sol1, hist1, sens1 = bayes()

        sol2, hist2, sens2 = bayes()

        npt.assert_almost_equal(hist1[0], hist2[0], decimal=4,
                                err_msg='Histories not equal for subsequent'
                                        'runs')

        npt.assert_almost_equal(sol1.models['eos'].get_dof() /
                                sol2.models['eos'].get_dof(),
                                np.ones(bayes.shape()[1]),
                                decimal=10,
                                err_msg='DOF not equal for subsequent runs')
        npt.assert_almost_equal(np.fabs(sens1['Gun'] - sens2['Gun']),
                                np.zeros(sens1['Gun'].shape),
                                decimal=10)

    @unittest.skip('skipped plotting routine')
    def test_fisher_matrix(self):
        """Tests if the fisher information matrix can be generated correctly
        """

        bayes = Bayesian(simulations=[(self.sim1, self.exp1),
                                      (self.sim2, self.exp2)],
                         model=self.eos_model)

        fisher = bayes.get_fisher_matrix(simid=1)

        n_model = bayes.shape()[1]

        self.assertIsInstance(fisher, np.ndarray)
        self.assertEqual(fisher.shape, (n_model, n_model))

        data = bayes.fisher_decomposition(fisher)
        bayes.plot_fisher_data(data)
        #plt.show()


class TestSimpleModels(unittest.TestCase):
    """Test the numeric values using simplified model and sim

    SimpleModel returns dof[0]*x**2 + dof[1]*x

    SimpleExperiment returns the value at SimpleModel, evaluated at
    x in np.arange(10)

    The simulation has dof [1,2]
    The experiment has dof [4,2]
    """
    def setUp(self):
        """Generates the simplified model
        """
        self.model = SimpleModel([2, 1])


        self.sim1 = SimpleSimulation()
        self.exp1 = SimpleExperiment()

        self.bayes = Bayesian(models={'simp': self.model},
                              simulations={'simple': [self.sim1, self.exp1]},
                              precondition=True,
                              opt_key='simp')

    def test_model_dict(self):
        """Tests of the model dictonary and setting the opt_key
        """

        # Tests that the Bayesian obect assumed the correct key when there is
        # just one model
        bayes = Bayesian(models={'simp': self.model},
                         simulations={'basic': [self.sim1, self.exp1]})
        self.assertEqual(bayes.opt_key, 'simp')

        # Test that the object uses the correct key when passed
        bayes = Bayesian(models={'simp': self.model},
                         simulations={'basic': [self.sim1, self.exp1]},
                         opt_key='simp')
        self.assertEqual(bayes.opt_key, 'simp')

        # Test that the object uses the correct key when there are
        # multiple models
        bayes = Bayesian(models={'simp': self.model, 'simp2': self.model},
                         simulations={'basic': [self.sim1, self.exp1]},
                         opt_key='simp2')
        self.assertEqual(bayes.opt_key, 'simp2')

        # Test that the object will fail if two models are used and no key
        # is given
        with self.assertRaises(IOError) as inst:
            bayes = Bayesian(models={'simp': self.model,
                                     'simp2': self.model},
                             simulations={'basic': [self.sim1, self.exp1]})

        # Test that the object will fail if an invalid key is given
        with self.assertRaises(KeyError) as inst:
            bayes = Bayesian(models={'simp': self.model,
                                     'simp2': self.model},
                             simulations={'basic': [self.sim1, self.exp1]},
                             opt_key='simp3')


    def test_sens(self):
        """Test the sign and magnitude of the sensitivity
        """
        initial_data = {'simple': self.sim1({'simp': self.model})}
        sens_matrix = self.bayes._get_sens(initial_data=initial_data)

        indep = np.arange(10)
        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1 * indep -
                             (2 * indep)**2 - indep,
                             (2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - 1 * indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02]])
        true_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        npt.assert_array_almost_equal(sens_matrix['simple'], true_sens,
                                      decimal=8,
                                      err_msg='Sens matrix  not as predicted')

        resp_mat2 = resp_mat
        resp_mat2[0, :] /= 0.04
        resp_mat2[1, :] /= 0.02

        npt.assert_array_almost_equal(sens_matrix['simple'], resp_mat2.T,
                                      decimal=8,
                                      err_msg='Sens matrix  not as predicted')
    @pytest.mark.xfail
    def test_hessian(self):
        """Tests hessian calculation
        """

        indep = np.arange(10)
        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1 * indep -
                             (2 * indep)**2 - indep,
                             (2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02]])
        init_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        resp_mat = np.array([(1.02**2 * 2 * indep)**2 + indep -
                             (1.02 * 2 * indep)**2 - indep,
                             (1.02 * 2 * indep)**2 + 1.02 * indep -
                             (1.02 * 2 * indep)**2 - indep])
        inp_mat = np.array([[0.02 * 1.02 * 2, 0], [0, 0.02]])
        step1_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - 1.02 * indep,
                             (2 * indep)**2 + 1.02**2 * indep -
                             (2 * indep)**2 - 1.02 * indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02 * 1.02]])
        step2_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        true_hess = np.zeros((2, 10, 2))
        true_hess[0, :, :] = (step1_sens - init_sens) / (0.02 * 2.0)

        true_hess[1, :, :] = (step2_sens - init_sens) / (0.02 * 1.0)

        hessian = self.bayes.get_hessian()
        npt.assert_array_almost_equal(hessian['simple'], true_hess,
                                      decimal=8,
                                      err_msg='Hessian not as predicted')

    def test_sens_passed_inp_data(self):
        """Test the sensitivity using provided initial data
        """
        data = self.bayes.get_data()
        sens_matrix = self.bayes._get_sens(initial_data=data)

        indep = np.linspace(0, 10, 100) - data['simple'][2]['tau']
        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1 * indep -
                             (2 * indep)**2 - indep,
                             (2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - 1 * indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02]])
        true_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        npt.assert_array_almost_equal(sens_matrix['simple'], true_sens,
                                      decimal=8,
                                      err_msg='Sens matrix  not as predicted')

    def test_model_pq(self):
        """Test the model PQ matrix generation
        """

        new_model = self.bayes.update(models={
            'simp': self.model.update_dof([4, 2])})

        P, q = new_model._get_model_pq()

        epsilon = np.array([2, 1])
        sigma = inv(np.diag(np.ones(2)))
        P_true = sigma
        q_true = -np.dot(epsilon, sigma)

        npt.assert_array_almost_equal(P, P_true, decimal=8)

        npt.assert_array_almost_equal(q, q_true, decimal=8)

    def test_sim_pq(self):
        """Test the simulation PQ matrix generation
        """

        initial_data = self.bayes.get_data()
        sens_matrix = self.bayes._get_sens(initial_data=initial_data)

        P, q = self.bayes._get_sim_pq(initial_data, sens_matrix)

        sigma = self.exp1.get_sigma()

        P_true = np.dot(np.dot(sens_matrix['simple'].T,
                               inv(sigma)),
                        sens_matrix['simple'])

        npt.assert_array_almost_equal(P, P_true, decimal=8)

        epsilon = self.bayes.simulations['simple']['exp']\
                            .compare(initial_data['simple'])

        q_true = -np.dot(np.dot(epsilon, inv(sigma)), sens_matrix['simple'])

        npt.assert_array_almost_equal(q, q_true, decimal=8)

    def test_mult_sens(self):
        """Test the sensitivity calculation for multiple experiments
        """

        new = self.bayes.update(simulations={'simple1': [self.sim1, self.exp1],
                                             'simple2': [self.sim1, self.exp1]})

        initial_data = new.get_data()
        sens_matrix = new._get_sens(initial_data=initial_data)

        indep = np.linspace(0,10,100) + 1
        resp_mat = np.array([(1.02 * 2 * indep)**2 + 1 * indep -
                             (2 * indep)**2 - indep,
                             (2 * indep)**2 + 1.02 * indep -
                             (2 * indep)**2 - 1 * indep])
        inp_mat = np.array([[0.02 * 2, 0], [0, 0.02]])
        true_sens = np.linalg.lstsq(inp_mat, resp_mat)[0].T

        npt.assert_array_almost_equal(sens_matrix['simple1'], true_sens,
                                      decimal=8)
        npt.assert_array_almost_equal(sens_matrix['simple2'], true_sens,
                                      decimal=8)

    def test_mult_simPQ(self):
        """Test the sensitivity calculation for multiple experiments
        """

        new = self.bayes.update(simulations={'simple1': [self.sim1, self.exp1],
                                             'simple2': [self.sim1, self.exp1]})

        initial_data = new.get_data()
        sens_matrix = new._get_sens(initial_data=initial_data)

        P, q = new._get_sim_pq(initial_data, sens_matrix)

        sigma = self.exp1.get_sigma()

        P_true = np.dot(np.dot(sens_matrix['simple1'].T,
                               inv(sigma)),
                        sens_matrix['simple1'])
        P_true += P_true

        npt.assert_array_almost_equal(P, P_true, decimal=8)

        epsilon = new.simulations['simple1']['exp'].\
                  compare(initial_data['simple1'])

        q_true = -np.dot(np.dot(epsilon, inv(sigma)), sens_matrix['simple1'])
        q_true += q_true
        npt.assert_array_almost_equal(q, q_true, decimal=8)

    def test_log_like(self):
        """Tests log likelyhood calculation
        """
        model = self.model.update_dof([1, 0.5])
        new = self.bayes.update(simulations={'simple1': [self.sim1, self.exp1],
                                             'simple2': [self.sim1, self.exp1]},
                                models={'simp': model})

        initial_data = new.get_data()

        model_log_like = new.model_log_like()

        sim_log_like = new.sim_log_like(initial_data)

        true_model_ll = -0.5 * np.dot(np.dot(np.array([-1, -0.5]),
                                             inv(self.model.get_sigma())),
                                      np.array([-1, -0.5]))

        epsilon = self.exp1.compare(initial_data['simple1'])
        true_sim_ll = -0.5 * np.dot(np.dot(epsilon,
                                           inv(self.exp1.get_sigma())),
                                    epsilon)
        true_sim_ll *= 2

        self.assertEqual(model_log_like, true_model_ll)
        self.assertEqual(sim_log_like, true_sim_ll)

    @pytest.mark.xfail
    def test_fisher_matrix_with_hessian(self):
        """
        """
        new = self.bayes.update(simulations={'simple1': [self.sim1, self.exp1],
                                             'simple2': [self.sim1, self.exp1]},
                                models={'simp':
                                        self.model.update_dof([1, 0.5])})

        initial_data = new.get_data()
        sens_matrix = new._get_sens(initial_data=initial_data)

        fisher = new.simulations['simple1']['exp'].get_fisher_matrix(
            new.models,
            use_hessian=True,
            sens_matrix=sens_matrix['simple1'])

    def test_fisher_matrix(self):
        """
        """
        new = self.bayes.update(simulations={'simple1': [self.sim1, self.exp1],
                                             'simple2': [self.sim1, self.exp1]},
                                models={'simp':
                                        self.model.update_dof([1, 0.5])})

        initial_data = new.get_data()
        sens_matrix = new._get_sens(initial_data=initial_data)

        sigma = inv(new.simulations['simple1']['exp'].get_sigma())

        fisher_1 = np.dot(np.dot(sens_matrix['simple1'].T,
                                 sigma),
                          sens_matrix['simple1'])

        fisher_2 = np.dot(np.dot(sens_matrix['simple2'].T,
                                 sigma),
                          sens_matrix['simple2'])

        npt.assert_array_almost_equal(self.exp1.get_fisher_matrix(
                                      new.models,
                                      sens_matrix=sens_matrix['simple1']),
                                      fisher_1)
        npt.assert_array_almost_equal(self.exp1.get_fisher_matrix(
                                      new.models,
                                      sens_matrix=sens_matrix['simple2']),
                                      fisher_2)


@unittest.skip('skipped empty')
class TestPlottingMethods(unittest.TestCase):
    """Test of the plotting methods
    """

    def setUp():
        pass


if __name__ == '__main__':

    unittest.main(verbosity=4)
