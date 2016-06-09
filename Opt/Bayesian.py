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
import matplotlib.pyplot as plt
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
            'outer_rtol' : [float, 1E-4, 0.0, 1.0, '-',
                            'Relative tolerance on change in likelyhood for\
                            outer loop convergence'],
            'maxiter' : [int, 6, 1, 100, '-',
                         'Maximum iterations for convergence\
                         of the likelyhood'],
            'constrain': [bool, True, None, None, '-',
                          'Flag to constrain the optimization'],
            'precondition':[bool, True, None, None, '-',
                            'Flag to scale the problem']
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

        return out_str

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

        return (m, n)

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

        return -0.5 * np.dot(np.dot(epsilon,inv(model.get_sigma())), epsilon)

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
            exp_indep, exp_dep, spline = exp()
            epsilon = sim.compare(exp_indep, exp_dep[0], sim_data)
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
        import matplotlib.pyplot as plt
        
        precondition = self.get_option('precondition')
        atol = self.get_option('outer_atol')
        reltol = self.get_option('outer_rtol')
        maxiter = self.get_option('maxiter')

        sims = self.simulations
        model = self.model

        history = []
        dof_hist = []
        dhat_hist = []
        conv = False
        log_like = 0.0

        initial_data = []
        for sim, exp in sims:
            sim.update(model=model)
            initial_data.append(sim())
        #end

        for i in xrange(maxiter):
            dof_hist.append(model.get_dof())
            print('Iter {} of {}'.format(i, maxiter))
            self._get_sens(sims, model, initial_data)
            
            # Solve all simulations with the curent model

            print ('prior log like', -self.model_log_like())
            print ('sim log like', -self.sim_log_like(initial_data))
            new_log_like = -self.model_log_like()\
                           - self.sim_log_like(initial_data)
            print ('total log like', new_log_like)

            history.append(new_log_like)#, self.model.get_dof()))


            if np.fabs(log_like - new_log_like) < atol\
               and np.fabs((log_like - new_log_like) / new_log_like) < reltol:
                conv = True
                break
            else:
                log_like = new_log_like
            #end

            
            #self.plot_sens_matrix(initial_data)

            
            local_sol = self._local_opt(sims, model, initial_data)
            

            # Perform basic line search along direction of best improvement
            d_hat = np.array(local_sol['x']).reshape(-1)
            if precondition:
                d_hat = np.dot(model.get_scaling(), d_hat)
            # end
            
            dhat_hist.append(d_hat)            
            n_steps = 5

            costs = np.zeros(n_steps)
            iter_data = []
            initial_dof = model.get_dof()
            besti = 0
            max_step = 10
            while besti == 0:
                max_step /= 10.0
                if max_step < 1E-10:
                    besti = 1
                    break
                x_list = np.linspace(0, max_step, n_steps)
                for i, x_i in enumerate(x_list):
                    model.set_dof(initial_dof + x_i * d_hat)
                    costs[i] = -self.model_log_like()
                    tmp = []
                    for sim, exp in sims:
                        sim.update(model=model)
                        tmp.append(sim())
                    #end
                    iter_data.append(tmp)
                    costs[i] -= self.sim_log_like(tmp)
                #end
                besti = np.argmin(costs)
                print('Zooming in to max step {:f}'.format(max_step/10.0))
            #end


            print(costs)
            print("{:d} {:f}".format(i, costs[besti]))
            print("Step size ", x_list[besti])
           
            model.set_dof(initial_dof + d_hat * x_list[besti])

            initial_data = iter_data[besti]
            for sim, exp in sims:
                sim.update(model = model)
            # end
        #end
        if not conv:
            print("{}: Outer loop could not converge to the given\
                             tolerance in the maximum number of iterations"\
                            .format(self.get_inform(1)))
        #end

        dof_hist = np.array(dof_hist)
        dhat_hist = np.array(dhat_hist)
        self.model = model
        self.simulations = sims
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for i in xrange(dof_hist.shape[1]):
            ax1.plot(dof_hist[:, i]/dof_hist[0,i])
            ax2.plot(dhat_hist[:, i])
        #end
        fig.suptitle('Convergence of iterative process')
        ax1.set_ylabel('Spline knot value')
        ax1.set_xlabel('Iteration number')
        ax2.set_ylabel('Step direction vector component')
        ax2.set_xlabel('Iteration number')
        fig.savefig('EOS_convergence.pdf')


        return model, history

    def get_fisher_matrix(self, simid = 0, sens_calc = True):
        """Returns the fisher information matrix of the simulation
        
        Keyword Args:
            simid(int): The index of the simulation to be investigated
                        *Default 0*
            sens_calc(bool): Flag to recalcualte sensitivities
                             *Default True*

        Return:
            (np.ndarray): The fisher information matrix, a nxn matrix where 
                          `n` is the degrees of freedom of the model.
        """
        if not isinstance(simid, int):
            raise TypeError("{:} the simid must be an integer"\
                            .format(self.get_inform(1)))
        elif simid >= len(self.simulations):
            raise IndexError("{:} simulation index out of range"\
                             .format(self.get_inform(1)))
        else:
            sim, exp = self.simulations[simid]
        #end

        if sens_calc:
            initial_data = [sim()]
            self._get_sens([(sim, exp)], self.model, initial_data)
        #end
        
        for i in xrange(len(self.simulations)):
            dim_k = self.simulations[i][1].shape()
            if i == simid:
                sens_k = self.sens_matrix[i:i+dim_k,:]
            #end
            i += dim_k

        sigma = inv(sim.get_sigma())
        
        return np.dot(sens_k.T, np.dot(sigma, sens_k))

    def fisher_decomposition(self, fisher, tol = 1E-3):
        """
        
        Args:
            fisher(np.ndarray): A nxn array where n is model dof
        
        Keyword Args:
            tol(float): Eigen values less than tol are ignored
        
        Return:
            (list): Eigenvalues greater than tol
            (np.ndarray): nxm array. 
                          n is number of eigenvalues greater than tol
                          m is model dof
            (np.ndarray): nxm array:
                          n is the number of eigenvalues greater than tol
                          m is an arbutary dimension of independent variable
            (np.ndarray): vector of independent varible
        
        """
        eos = self.model
        
        # Spectral decomposition of info matrix and sort by eigenvalues
        eig_vals, eig_vecs = np.linalg.eigh(fisher)
        eig_vals = np.maximum(eig_vals, 0)        # info is positive definite
        i = np.argsort(eig_vals)[-1::-1]
        vals = eig_vals[i]
        vecs = eig_vecs.T[i]
        
        n_vals = max(len(np.where(vals > vals[0]*1e-3)[0]), 3)
        n_vecs = len(np.where(vals > vals[0]*1e-2)[0])

        # Find range of v that includes support of eigenfunctions
        knots = eos.get_t()
        v_min = knots[0]
        v_max = knots[-1]
        v = np.logspace(np.log10(v_min), np.log10(v_max), len(knots)*10)
        max_k = 0
        min_k = len(v)-1
        funcs = []
        for vec in vecs[:n_vecs]:
            eos.set_dof(vec)
            func = eos(v)  # An eigenfunction of the Fisher Information
            a = np.abs(func)
            argmax = np.argmax(a)
            if func[argmax] < 0:
                func *= -1
            funcs.append(func)
            big = np.where(a > a[argmax]*1e-4)[0]
            min_k = min(big[0], min_k)
            max_k = max(big[-1], max_k)

        funcs = np.array(funcs)

        vals = vals[np.where(vals>tol*vals[0])]
        
        k_range = np.arange(min_k, max_k+1)

        return vals, vecs, funcs, v

    def plot_fisher_data(self, fisher_data, filename = None):
        """
        
        Args:
            fisher_dat(tuple): Data from the fisher_decomposition function
                               *see docscring for definition*

        Keyword Args:
            filename(str or None): If none, do not make a hardcopy, otherwise 
                                   save to the file specified
        
        """

        fig = plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        eigs = fisher_data[0]
        eig_vects = fisher_data[1]
        eig_func = fisher_data[2]
        indep = fisher_data[3]

        ax1.bar(np.arange(eigs.shape[0]), eigs , width = 0.9 , color = 'black',
                edgecolor='none', orientation = 'vertical')

        ax1.set_xlabel("Eigenvalue number /")
        ax1.set_ylabel("Eigenvalue /")
        
        for i in xrange(eig_func.shape[0]):
            ax2.plot(indep, eig_func[i], label = "eig {:d}".format(i))
        #end

        ax2.set_xlabel("Specific volume / cm**3 g**-1")
        ax2.set_ylabel("Eigenfunction response / Pa")

        fig.tight_layout()
                          
    def _local_opt(self, sims, model,  initial_data):
        """
        """
        constrain = self.get_option('constrain')
        
        # Get constraints
        G,h = self._get_constraints(model)

        P_mat, q_mat = self._get_model_PQ(model)
        tmp = self._get_sim_PQ(sims, model, initial_data)

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

    def _get_constraints(self, model):
        r"""EOS MODEL - get the constraints on the model
        
        Args:
           model(PhysicsModel): The physics model subject to 
                                physical constraints
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
        precondition = self.get_option('precondition')
        c_model = copy.deepcopy(model)
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

        if precondition:
            G = np.dot(G, c_model.get_scaling())
        #end
        
        return G,h

    def _get_model_PQ(self, model):
        """Gets the quadratic optimizaiton matrix contributions from the prior
        
        Args:
           model(PhysicsModel): A physics model with degrees of freedom
        
        Retrun:
            (np.ndarray): `P`, a nxn matrix where n is the model DOF
            (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """

        precondition = self.get_option('precondition')
        prior_scale = model.get_scaling()
        prior_var = inv(model.get_sigma())
        
        prior_delta = model.get_dof() - model.prior.get_dof()

        if precondition:
            return np.dot(prior_scale, np.dot(prior_var, prior_scale)),\
                np.dot(prior_delta, prior_var)
        else:
            return prior_var,  np.dot(prior_scale,\
                                      np.dot(prior_delta, prior_var))

    def _get_sim_PQ(self, sims, model, initial_data):
        """Gets the QP contribytions from the model

        Args:
           sims(list): A list of tuples of experiments each tuple contains
                       [0] the simulation
                       [1] the corresponding experiment
           model(PhysicsModel): A physics model with degrees of freedom
           initial_data(list): A list of the inital results from the simulations
                               in the same order as in the `sim` list
        
        Retrun:
            (np.ndarray): `P`, a nxn matrix where n is the model DOF
            (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """

        precondition = self.get_option('precondition')
        prior_sigma = model.get_sigma()
        prior_scale = model.get_scaling()        
        D_mat = self.sens_matrix
        P_mat = np.zeros((self.shape()[1], self.shape()[1]))
        q_mat = np.zeros(self.shape()[1])

        i = 0
        
        for (sim, exp), sim_data in zip(sims, initial_data):
            dim_k = exp.shape()
            sens_k = D_mat[i:i+dim_k,:]
            exp_indep, exp_dep, exp_spline = exp()
            # basis_k = sim_data[2].get_basis(exp_indep,
            #                                 spline_end = spline_end)
            # sens_k = np.dot(sens_k, basis_k)
            epsilon = sim.compare(exp_indep, exp_dep[0], sim_data)

            P_mat = np.dot(np.dot(sens_k.T, inv(sim.get_sigma())), sens_k)
            q_mat = np.dot(np.dot(epsilon, inv(sim.get_sigma())), sens_k)
            i += dim_k
        #end

        if precondition:
            P_mat = np.dot(prior_scale, np.dot(P_mat, prior_scale))
            q_mat = np.dot(prior_scale, q_mat)
        #end
        
        return P_mat, -q_mat

    def _get_sens(self, sims, model, initial_data):
        """Gets the sensitivity of the simulated experiment to the EOS

        Args:
           initial_data(list): The results of each simulation with the
                               curent best model
        """

        model = copy.deepcopy(model)
        sens_tol = 1E-12
        step_frac = 2E-2

        original_dofs = model.get_dof()
        sens_matrix = np.empty(self.shape())
        new_dofs = copy.deepcopy(original_dofs)

        # initial_data = []
        # for sim, exp in sims:
        #      initial_data.append(sim())
        # #end

        for i in xrange(len(original_dofs)):
            step = float(new_dofs[i] * step_frac)
            new_dofs[i] += step

            model.set_dof(new_dofs)

            j = 0 # counter for the starting column of this sim's data

            for (sim, exp), sim_data in zip(sims,initial_data):
                sim.update(model = model)
                new_data = sim()
                dim = sim.shape()
                delta = sim.compare(sim_data[0],
                                    sim_data[1][0],
                                    new_data)
                delta /= -step
                # If the sensitivity is less than the tolerance, make it
                # zero
                sens_matrix[j:j+dim,i] = np.where(np.fabs(delta) > sens_tol,\
                                                  delta,\
                                                  np.zeros(len(delta)))
                j += dim
            #end
            new_dofs[i] -= step
        #end

        # Return all simulations to original state
        model.set_dof(original_dofs)
        for sim, exp in sims:
            sim.update(model = model)
        #end
        
        self.sens_matrix = sens_matrix
    # end

    def plot_sens_matrix(self, initial_data):
        """Prints the sensitivity matrix
        """

        import matplotlib.pyplot as plt
        sens_matrix = self.sens_matrix

        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)        

        knot_post = self.model.get_t()
        
        for i in xrange(10):
            ax1.plot( sens_matrix[:,i], label = "{:4.3f}".format(knot_post[i]))
        ax1.legend(loc = 'best')

        for i in xrange(10,20):
            ax2.plot( sens_matrix[:,i], label = "{:4.3f}".format(knot_post[i]))
        ax2.legend(loc = 'best')

        for i in xrange(20,30):
            ax3.plot( sens_matrix[:,i], label = "{:4.3f}".format(knot_post[i]))
        ax3.legend(loc = 'best')

        for i in xrange(30,40):
            ax4.plot( sens_matrix[:,i], label = "{:4.3f}".format(knot_post[i]))
        ax4.legend(loc = 'best')

        for i in xrange(40,50):
            ax5.plot( sens_matrix[:,i], label = "{:4.3f}".format(knot_post[i]))
        ax5.legend(loc = 'best')


        
        # for i in xrange(0,basis_k.shape[0],5):
        #     ax2.plot(exp_indep, basis_k[i,:])

        # for i in xrange(-8,0):
        #     ax4.plot(exp_indep, basis_k[i,:])
        #     # ax4.plot(basis_f[i,:])
        # #end

        # BC = np.dot(sens_matrix,basis_k)
        # for i in xrange(BC.shape[0]):
        #     ax3.plot(exp_indep, BC[i,:])
        # # end

        plt.show()
class TestBayesian(unittest.TestCase):
    """Test class for the bayesian object
    """
    def setUp(self):
        """Setup script for each test
        """

        from F_UNCLE.Experiments.GunModel import Gun
        from F_UNCLE.Experiments.Stick import Stick
        from F_UNCLE.Models.Isentrope import EOSModel, EOSBump

        # Initial estimate of prior functional form
        init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

        # Create the model and *true* EOS
        self.eos_model = EOSModel(init_prior)
        self.eos_true = EOSBump()

        # Create the objects to generate simulations and pseudo experimental data
        self.exp1 = Gun(self.eos_true)
        self.sim1 = Gun(self.eos_model)

        self.exp2 = Stick(self.eos_true)
        self.sim2 = Stick(self.eos_model)

    # end

    def test_instantiation(self):
        """Test that the object can instantiate correctly
        """

        bayes = Bayesian(simulations = [(self.sim1, self.exp1)],
                         model = self.eos_model)

        # print(bayes)

        self.assertIsInstance(bayes, Bayesian)
    # end

    def test_bad_instantiaion(self):
        """Tets impropper instantiation raises the correct errors
        """

        pass
    # end

    def test_singel_case_PQ_mod(self):
        """Tests the P and Q matrix generation for a single case
        """

        bayes = Bayesian(simulations = [(self.sim1, self.exp1)],
                         model = self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape()
        sim_shape = self.sim1.shape()
        mod_shape = self.eos_model.shape()[0]
        
        initial_data = [self.sim1()]
        P, q = bayes._get_sim_PQ([(self.sim1, self.exp1)], self.eos_model, initial_data)

        self.assertEqual(P.shape, (mod_shape, mod_shape))
        self.assertEqual(q.shape, (mod_shape, ))

    def test_mlt_case_PQ_mod(self):
        """Tests the P and Q matrix generation for a multiple case
        """

        bayes = Bayesian(simulations = [(self.sim1, self.exp1),
                                        (self.sim2, self.exp2)],
                         model = self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape() + self.exp2.shape()
        sim_shape = self.sim1.shape() + self.sim2.shape()
        mod_shape = self.eos_model.shape()[0]
        
        initial_data = [self.sim1(), self.sim2()]
        P, q = bayes._get_sim_PQ([(self.sim1, self.exp1),(self.sim2, self.exp2)],
                                  self.eos_model, initial_data)

        self.assertEqual(P.shape, (mod_shape, mod_shape))
        self.assertEqual(q.shape, (mod_shape, ))

    def test_gun_case_sens(self):
        """
        """

        bayes = Bayesian(simulations = [(self.sim1, self.exp1)],
                         model = self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape()
        sim_shape = self.sim1.shape()
        mod_shape = self.eos_model.shape()[0]
        
        initial_data = [self.sim1()]

        bayes._get_sens([(self.sim1, self.exp1)], self.eos_model, initial_data)
        self.assertEqual(bayes.sens_matrix.shape, (exp_shape, mod_shape))
        # bayes.plot_sens_matrix(initial_data)

    def test_stick_case_sens(self):
        """Test of the sensitivity of the stick model to the eos
        """

        bayes = Bayesian(simulations = [(self.sim2, self.exp2)],
                         model = self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp2.shape()
        sim_shape = self.sim2.shape()
        mod_shape = self.eos_model.shape()[0]
        
        initial_data = [self.sim2()]

        bayes._get_sens([(self.sim2, self.exp2)],
                        self.eos_model, initial_data)
        self.assertEqual(bayes.sens_matrix.shape, (exp_shape, mod_shape))
        #bayes.plot_sens_matrix(initial_data)
        
    def test_mult_case_sens(self):
        """Test of sens matrix generation for mult models
        """
        bayes = Bayesian(simulations = [(self.sim1, self.exp1),
                                        (self.sim2, self.exp2)],
                         model = self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape() + self.exp2.shape()
        sim_shape = self.sim1.shape() + self.sim1.shape()
        mod_shape = self.eos_model.shape()[0]
        
        initial_data = [self.sim1(), self.sim2()]

        bayes._get_sens([(self.sim1, self.exp1),
                         (self.sim2, self.exp2)],
                        self.eos_model, initial_data)

        self.assertEqual(bayes.sens_matrix.shape, (exp_shape, mod_shape))
        
        # bayes.plot_sens_matrix(initial_data)

    def test_model_pq(self):
        """Tests the pq matrix generation by the model
        """

        pass

    def test_fisher_matrix(self):
        """Tests if the fisher information matrix can be generated correctly
        """

        bayes = Bayesian(simulations = [(self.sim1, self.exp1),
                                        (self.sim2, self.exp2)],
                         model = self.eos_model)
        
        fisher = bayes.get_fisher_matrix(simid = 1)

        n_model = bayes.shape()[1]

        self.assertIsInstance(fisher, np.ndarray)
        self.assertEqual(fisher.shape, (n_model, n_model))

        data = bayes.fisher_decomposition(fisher)
        bayes.plot_fisher_data(data)
        plt.show()
        
        
        
    
if __name__ == '__main__':

    unittest.main(verbosity=4)
