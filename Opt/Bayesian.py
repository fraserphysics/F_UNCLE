#/usr/bin/pyton
"""

pyBayesian

An object to extract properties of the bayesian analysis of experiments

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (03-16-2016)


"""
from __future__ import print_function

# =========================
# Python Standard Libraries
# =========================

import sys
import os
# import pdb
import unittest
import copy
import pdb


# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv 
from cvxopt import matrix, solvers

# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Experiments.Experiment import Experiment
from F_UNCLE.Models.PhysicsModel import PhysicsModel
from F_UNCLE.Utils.Struc import Struc

# =========================
# Main Code
# =========================

class Bayesian(Struc):
    """A calss for performing bayesian inference on a model given data

    Attributes:
       simulations(tuple): [0] Simulation object
                           [1] Experimental data
       model(PhysicsModel): The model under consideration
       sens_matrix(nump.ndarray): The (nxm) sensitivity matrix
                                  n - model degrees of freedom
                                  m - total experiment DOF
                                  [i,j] - sensitivity of model DOF i
                                          to experiment DOF j

    """

    def __init__(self, simulations, model, name='Bayesian', *args, **kwargs):
        """Instantiates the Bayesian analysis

        Args:
           sim_exp(Experiment): The simulated experimental data
           true_exp(Experiment): The true experimental data
           prior(Struc): The prior for the physics model

        Keyword Args:
            name(str): Name for the analysis.('Bayesian')

        Return:
           None

        """

        def_opts = {
            'outer_atol' : [float, 1E-6, 0.0, 1.0, '-',
                            'Absolute tolerance on change in likelyhood for\
                            outer loop convergence'],
            'outer_rtol' : [float, 1E-6, 0.0, 1.0, '-',
                            'Relative tolerance on change in likelyhood for\
                            outer loop convergence'],
            'maxiter' : [int, 20, 1, 100, '-', 'Maximum iterations for convergence\
                         of the likelyhood'],
            'constrain': [bool, True, None, None, '-', 'Flag to constrain the\
                          optimization']
        }

        Struc.__init__(self, name=name, def_opts=def_opts)

        self.simulations = None
        self.model = None
        self.update(simulations=simulations,
                    model=model)

        self.sens_matrix = np.nan * np.ones(self.shape())
    #end

    def _on_str(self):
        """Print method for bayesian model
        """

        out_str = ''
        out_str += 'Model\n'
        out_str += '=====\n'
        out_str += str(self.model)

        out_str += 'Experiments\n'
        out_str += '===========\n'
        for sim, exp in self.simulations:
            out_str += str(exp)
        #end

        out_str += 'Simulations\n'
        out_str += '===========\n'
        for sim, exp in self.simulations:
            out_str += str(sim)
        #end

    def update(self, simulations=None, model=None):
        """Updates the properties of the bayesian analtsis

        Keyword Args:
           simulations(Experiment): The tupples of simulations and experiments (Default None)
           model(PhysicsModel): The physics model used in the simulaitons (Default None)

        Return:
            None

        """
        if simulations is None:
            self.simulations = None
        else:
            if not isinstance(simulations, list):
                simulations = [simulations]
            #end

            for i in xrange(len(simulations)):
                if not isinstance(simulations[i], tuple) and\
                   not len(simulations[i]) == 2:
                    raise TypeError('{:} simulations must be provided with a\
                                     2-tuple with the first element the\
                                     simulation, the second, the experiment'.\
                                     format(self.get_inform(1)))
                elif not isinstance(simulations[i][0], Experiment) and\
                     not isinstance(simulations[i][1], Experiment):
                    raise TypeError('{:} Both the simulation and experiment must\
                                     be Experiment types'.\
                                     format(self.get_inform(1)))
                else:
                    pass
                #end
            #end
            self.simulations = simulations
        #end

        if model is None:
            self.model = None
        elif isinstance(model, PhysicsModel):
            self.model = model
        else:
            raise TypeError("{:} the model must be a PhysicsModel type"\
                            .format(self.get_inform(1)))
        # end

    def shape(self):
        """Gets the dimenstions of the problem

        Return:
           (tuple): The n x m dimensions of the problem
        """

        n = self.model.shape()[0]
        m = 0
        for sim, exp in self.simulations:
            m += exp.shape()
        #end

        return (n, m)

    def model_log_like(self):
        r"""Gets the log likelyhood of the model given the prior

        Args:
           None

        Return:
           (float): Log likelyhood of the model

        .. math::

           \log(p(f|y))_{model} = -\frac{1}{2}(f - \mu_f)\Sigma_f^{-1}(f-\mu_f)

        """
        model = self.model

        epsilon = model.get_dof() - model.prior.get_dof()

        return -0.5 * np.dot(epsilon, np.dot(inv(model.get_sigma()), epsilon))

    def sim_log_like(self, initial_data):
        r"""Gets the log likelyhood of the simulations given the data

        Args:
           initial_data(list): A list of the initial data for the simulations

        Return:
           (float): Log likelyhood of the prior

        .. math::

           \log(p(f|y))_{model} = -\frac{1}{2}
             (y_k - \mu_k(f))\Sigma_k^{-1}(y_k-\mu_k(f))

        """

        log_like = 0
        for (sim, exp), sim_data in zip(self.simulations, initial_data):
            exp_indep, (exp_dep1, exp_dep2), spline = exp()
            epsilon = sim.compare(exp_indep, exp_dep2, sim_data)
            log_like += -0.5 * np.dot(epsilon,
                                      np.dot(inv(sim.get_sigma()), epsilon))
        #end

        return log_like

    def __call__(self):
        """Determines the best candidate EOS function for the models

        Return:
           (Isentrope): The isentrope which gives best agreement over the space
           (list): The history of candiate prior DOF's
        """

        atol = self.get_option('outer_atol')
        reltol = self.get_option('outer_rtol')
        maxiter = self.get_option('maxiter')

        model = self.model
        history = []
        dof_hist = []
        dhat_hist = []
        conv = False
        log_like = 0.0
        
        initial_data = []
        for sim, exp in self.simulations:
            sim.update(model=self.model)
            initial_data.append(sim())
        #end

        for i in xrange(maxiter):
            dof_hist.append(self.model.get_dof())
            print('Iter {} of {}'.format(i, maxiter))
            # Solve all simulations with the curent model

            print ('prior log like', self.model_log_like())
            print ('sim log like', self.sim_log_like(initial_data))            
            new_log_like = self.model_log_like()\
                           + self.sim_log_like(initial_data)

            history.append(new_log_like)#, self.model.get_dof()))

            
            # if new_log_like > log_like:
            #     self.model.set_dof(dof_hist[-2])
            #     continue
            if np.fabs(log_like - new_log_like) < atol\
               and np.fabs((log_like - new_log_like) / new_log_like) < reltol:
                conv = True
                break
            else:
                log_like = new_log_like
            #end

            self._get_sens(initial_data)

            local_sol = self._local_opt(initial_data)
                
            #end

            # Perfirm basic line search along direction of best improvement
            d_hat = np.array(local_sol['x']).reshape(-1)

            dhat_hist.append(d_hat)
            
            n_steps = 4
            xs = np.linspace(0,1, n_steps)
            costs = np.zeros(n_steps)
            tmp_data = [[]]*n_steps
            for i,x in enumerate(xs):
                model.set_dof(self.model.get_dof() + x * d_hat)
                costs[i] += self.model_log_like()
                
                for sim, exp in self.simulations:
                    sim.update(model = self.model)
                    tmp_data[i].append(sim())
                #end

                costs[i] += self.sim_log_like(tmp_data[i])
                
            i = len(xs)-1

            print("Step size ", xs[i])
            self.model.set_dof(self.model.get_dof()\
                                + d_hat * xs[i])
            initial_data = tmp_data[i]

            print(self.model.get_dof())
                               
        if not conv:
            print("{}: Outer loop could not converge to the given\
                             tolerance in the maximum number of iterations"\
                            .format(self.get_inform(1)))
        #end
        
        dof_hist = np.array(dof_hist)
        dhat_hist = np.array(dhat_hist)
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for i in xrange(dof_hist.shape[1]):
            ax1.plot(dof_hist[:, i])
            ax2.plot(dhat_hist[:, i])
        #end
        fig.suptitle('Convergence of iterative process')
        ax1.set_ylabel('Spline knot value')
        ax1.set_xlabel('Iteration number')
        ax2.set_ylabel('Step direction vector component')
        ax2.set_xlabel('Iteration number')
        fig.savefig('EOS_convergence.pdf')

        return self.model, history

    def _local_opt(self, initial_data):
        """
        """
        constrain = self.get_option('constrain')

        # Get constraints
        G,h = self._get_constraints()

        P_mat, q_mat = self._get_model_PQ()
        tmp = self._get_sim_PQ(initial_data)

        P_mat += tmp[0]
        q_mat += tmp[1]

        P_mat *= 0.5
        
        solvers.options['show_progress'] = True
        solvers.options['maxiters'] = 100  # 100 default
        solvers.options['reltol'] = 1e-6   # 1e-6 default
        solvers.options['abstol'] = 1e-7   # 1e-7 default
        solvers.options['feastol'] = 1e-7  # 1e-7 default


        if constrain:
            sol = solvers.qp(matrix(P_mat), matrix(q_mat), matrix(G), matrix(h))
        else:
            sol = solvers.qp(matrix(P_mat), matrix(q_mat))
        
        if sol['status'] != 'optimal':
            for key, value in sol.items():
                print(key, value)
            raise RuntimeError('{} The optimization algorithm could not locate\
                               an optimal point'.format(self.get_inform(1)))
        return sol
    
    def _get_constraints(self):
        r"""EOS MODEL - get the constraints on the model

        Return:
           ():
           ():

        Method

        Calculate constraint matrix G and vector h.  The
        constraint enforced by `cvxopt.solvers.qp` is

        .. math::

           G*x \leq_component_wise h

        Equivalent to :math:`max(G*x - h) \leq 0`

        Since

        .. math::

           c_{f_{new}} = c_f+x,

        .. math::

           G(c_f+x) \leq_component_wise 0

        is the same as

        .. math::

           G*x \leq_component_wise -G*c_f,

        and :math:`h = -G*c_f`

        Here are the constraints for :math:`p(v)`:

        p'' positive for all v
        p' negative for v_max
        p positive for v_max

        For cubic splines between knots, f'' is constant and f' is
        affine.  Consequently, f''*rho + 2*f' is affine between knots
        and it is sufficient to check eq:star at the knots.

        """

        c_model = copy.deepcopy(self.model)
        spline_end = c_model.get_option('spline_end')
        dim = c_model.shape()[0]

        v_all = c_model.get_t()
        v_unique = v_all[spline_end-1:1-spline_end]
        n_v = len(v_unique)
        n_constraints = n_v + 2


        G = np.zeros((n_constraints, dim))
        c = np.zeros(dim)
        c_init = c_model.get_dof()
        for i in range(dim):
            c[i] = 1.0
            c_model.set_dof(c)
            G[:-2,i] = -c_model.derivative(2)(v_unique)
            G[-2,i] = c_model.derivative(1)(v_unique[-1])
            G[-1,i] = -c_model(v_unique[-1])
            c[i] = 0.0
        # end

        h = -np.dot(G,c_init)

        scale = np.abs(h)
        scale = np.maximum(scale, scale.max()*1e-15)
        # Scale to make |h[i]| = 1 for all i
        HI = np.diag(1.0/scale)
        h = np.dot(HI,h)
        G = np.dot(HI,G)

        return G,h

    def _get_model_PQ(self):
        """Gets the quadratic optimizaiton matrix contributions from the prior
        """

        prior_var = inv(self.model.get_sigma())

        prior_delta = self.model.get_dof() - self.model.prior.get_dof()

        return prior_var, np.dot(prior_delta, prior_var)

    def _get_sim_PQ(self, initial_data):
        """
        """

        spline_end = 4
        
        D_mat = self.sens_matrix
        P_mat = np.zeros((self.shape()[0], self.shape()[0]))
        q_mat = np.zeros(self.shape()[0])
        i = 0
        for (sim, exp), sim_data in zip(self.simulations, initial_data):
            dim_k = exp.shape()
            sens_k = D_mat[:,i:i+dim_k]
            exp_indep, (exp_dep1, exp_dep2), exp_spline = exp()
            basis_k = sim_data[2].get_basis(exp_indep,
                                            spline_end = spline_end)
            sens_k = np.dot(sens_k, basis_k)
            epsilon = sim.compare(exp_indep, exp_dep2, sim_data)
            P_mat += np.dot(sens_k,np.dot(inv(sim.get_sigma()),sens_k.T))
            q_mat += np.dot(epsilon,np.dot(inv(sim.get_sigma()),sens_k.T))
            i += dim_k
        #end

        return P_mat, q_mat
    
    def _get_sens(self, initial_data):
        """Gets the sensitivity of the simulated experiment to the EOS

        Args:
           initial_data(list): The results of each simulation with the
                               curent best model
        """
        simulations = self.simulations
        model = self.model
        step_frac = 2E-2
        spline_end = 4
        
        original_dofs = model.get_dof()
        sens_matrix = np.zeros(self.shape())
        
        for i in xrange(len(original_dofs)):
            new_dofs = copy.deepcopy(original_dofs)
            step = new_dofs[i] * step_frac
            new_dofs[i] += step

            model.set_dof(new_dofs)

            j = 0 # counter for the starting column of this sim's data
            k = 0 # counter for simulation number
            # pdb.set_trace()
            for (sim, exp), sim_data in zip(simulations,initial_data):
                original_response = sim_data[2].get_c(spline_end=spline_end)
                sim.update(model = model)
                new_data = sim()
                dim = sim.shape()                
                delta = original_response - new_data[2].get_c(spline_end=spline_end)
                sens_matrix[i,j:j+dim] = delta / step
                j += dim
            # end
            
            k += 1
        #end

        # Return all simulations to original state
        model.set_dof(original_dofs)
        for sim, exp in simulations:
            sim.update(model = model)
        #end

        self.sens_matrix = sens_matrix
    # end

class TestBayesian(unittest.TestCase):
    """Test class for the bayesian object
    """
    def setUp(self):
        """Setup script for each test
        """

        self.exp = Experiment(name="Dummy experiment")
        self.mod = Experiment(name="Dummy model")
        self.model = Experiment(name="Dummy prior")

    # end

    def test_instantiation(self):
        """Test that the object can instantiate correctly
        """

        bayes = Bayesian(self.exp, self.mod, self.prior)

        print(bayes)

        self.assertIsInstance(bayes, Bayesian)
    # end

    def test_bad_instantiaion(self):
        """Tets impropper instantiation raises the correct errors
        """

        pass
    # end

if __name__ == '__main__':

    unittest.main(verbosity=4)
