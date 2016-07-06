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

TODO
----

- Examine impact of prior weight **Prior weight currently zero**
- Examine effect of choice of true eos

"""
from __future__ import print_function
# =========================
# Python Standard Libraries
# =========================
import sys
import os
import unittest
import copy
import math
import pdb
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

try:
    from cvxopt import matrix, solvers
except:
    print('cvxopt package not found. Install by running\n'+\
          '`$ pip install cvxopt`')
#end

# =========================
# Custom Packages
# =========================

if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Experiment import Experiment
    from F_UNCLE.Utils.PhysicsModel import PhysicsModel
    from F_UNCLE.Utils.Struc import Struc
else:
    from ..Utils.Experiment import Experiment
    from ..Utils.PhysicsModel import PhysicsModel
    from ..Utils.Struc import Struc
# end

# =========================
# Main Code
# =========================

class Bayesian(Struc):
    """A calss for performing bayesian inference on a model given data

    **Attributes**

    Attributes:
       simulations(list):  Each element is a tupple with the following elemnts

           0. A simulation
           1. Experimental results

       model(PhysicsModel): The model under consideration
       sens_matrix(nump.ndarray): The (nxm) sensitivity matrix

           - n model degrees of freedom
           - m total experiment DOF
           - [i,j] sensitivity of model response i to experiment DOF j

    **Options**

    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |Name          |Type   |Def  |Min  |Max  |Units|Description                 |
    +==============+=======+=====+=====+=====+=====+============================+
    |`outer_atol`  |(float)|1E-6 |0.0  |1.0  |-    |Absolute tolerance on change|
    |              |       |     |     |     |     |in likelyhood for outer loop|
    |              |       |     |     |     |     |convergence                 |
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`outer_rtol`  |(float)|1E-4 |0.0  |1.0  |-    |Relative tolerance on change|
    |              |       |     |     |     |     |in likelyhood for outer loop|
    |              |       |     |     |     |     |convergence                 |
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`maxiter`     |(int)  |6    |1    |100  |-    |Maximum iterations for      |
    |              |       |     |     |     |     |convergence of the          |
    |              |       |     |     |     |     |likelyhood                  |
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`constrain`   |(bool) |True |None |None |-    |Flag to constrain the       |
    |              |       |     |     |     |     |optimization                |
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`precondition`|(bool) |True |None |None |-    |Flag to scale the problem   |
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`prior_weight`|(float)|1.0  |0.0  |0.0  |-    |Weight of the prior when    |
    |              |       |     |     |     |     |calculating the log         |
    |              |       |     |     |     |     |likelihood                  |
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`debug`       |(bool) |False|None |None |-    |Flag to print debug         |         
    |              |       |     |     |     |     |information                 |               
    +--------------+-------+-----+-----+-----+-----+----------------------------+
    |`verb`        |(bool) |True |None |None |-    |Flag to print stats during  |
    |              |       |     |     |     |     |optimization                |
    +--------------+-------+-----+-----+-----+-----+----------------------------+

    .. note::

       The options `outer_atol` and `prior_weight` are depricated and should be
       used for debugging purposes only

    **Methods**
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
                            'Flag to scale the problem'],
            'debug' : [bool, False, None, None, '-',
                       'Flag to print debug information'],
        
            'verb' : [bool, True, None, None, '-',
                     'Flag to print stats during optimization'],
            'prior_weight': [float, 1.0, 0.0, 1.0, '-',
                             'Weighting of the prior when calculating log\
                             likelihood']
        }

        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)

        self.simulations = None
        self.model = None
        self.update(simulations=simulations,
                    model=model)

        self.sens_matrix = np.nan * np.ones(self.shape())
    #end

    def _on_str(self):
        """Print method for bayesian model

        Args:
            None

        Return:
           (str): String describing the Bayesian object
        """
        out_str = ''
        out_str += "=======================================================\n"
        out_str += "=======================================================\n"
        out_str += "           ____                        _               \n"
        out_str += "          |  _ \                      (_)              \n"
        out_str += "          | |_) | __ _ _   _  ___  ___ _  __ _ _ __    \n"
        out_str += "          |  _ < / _` | | | |/ _ \/ __| |/ _` | '_ \   \n"
        out_str += "          | |_) | (_| | |_| |  __/\__ \ | (_| | | | |  \n"
        out_str += "          |____/ \__,_|\__, |\___||___/_|\__,_|_| |_|  \n"
        out_str += "                        __/ |                          \n"
        out_str += "                       |___/                           \n"
        out_str += "=======================================================\n"
        out_str += "=======================================================\n"
        out_str += "  __  __           _      _ \n"
        out_str += " |  \/  |         | |    | |\n"
        out_str += " | \  / | ___   __| | ___| |\n"
        out_str += " | |\/| |/ _ \ / _` |/ _ \ |\n"
        out_str += " | |  | | (_) | (_| |  __/ |\n"
        out_str += " |_|  |_|\___/ \__,_|\___|_|\n"
        out_str += str(self.model)
        out_str += " _____      _             \n"
        out_str += "|  __ \    (_)            \n"
        out_str += "| |__) | __ _  ___  _ __  \n"
        out_str += "|  ___/ '__| |/ _ \| '__| \n"
        out_str += "| |   | |  | | (_) | |    \n"
        out_str += "|_|   |_|  |_|\___/|_|    \n"
        out_str += str(self.model.prior)
        out_str += " ______                      _                      _        \n"
        out_str += "|  ____|                    (_)                    | |       \n"
        out_str += "| |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___  \n"
        out_str += "|  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __| \n"
        out_str += "| |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \ \n"
        out_str += "|______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/ \n"
        out_str += "            | |                                              \n"
        out_str += "            |_|                                              \n"
        for sim, exp in self.simulations:
            out_str += str(exp)
        #end
        out_str += "  _____ _                 _       _   _                  \n"
        out_str += " / ____(_)               | |     | | (_)                 \n"
        out_str += "| (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __  ___  \n"
        out_str += " \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \/ __| \n"
        out_str += " ____) | | | | | | | |_| | | (_| | |_| | (_) | | | \__ \ \n"
        out_str += "|_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|___/ \n"
        for sim, exp in self.simulations:
            out_str += str(sim)
        #end

        return out_str

    def update(self, simulations=None, model=None):
        """Updates the properties of the bayesian analtsis

        Keyword Args:
           simulations(Experiment): The tupples of simulations and experiments
                                    (Default None)
           model(PhysicsModel): The physics model used in the simulaitons
                                (Default None)

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
           (tuple): The (n, m) dimensions of the problem

               - n is the total degrees of freedom of all the model responses
               - m is the degrees of freedom of the model
        """

        dof_model = self.model.shape()[0]
        dof_sim = 0
        for sim, exp in self.simulations:
            dof_sim += exp.shape()
        #end

        return (dof_sim, dof_model)

    def model_log_like(self):
        r"""Gets the log likelyhood of the `self.model` given that model's prior

        Return:
           (float): Log likelyhood of the model

        .. math::

           \log(p(f|y))_{model} = -\frac{1}{2}(f - \mu_f)\Sigma_f^{-1}(f-\mu_f)

        """
        model = self.model
        epsilon = model.get_dof() - model.prior.get_dof()

        return -0.5 * np.dot(np.dot(epsilon, inv(model.get_sigma())), epsilon)

    def sim_log_like(self, initial_data):
        r"""Gets the log likelyhood of the simulations given the data

        Args:
            initial_data(list): A list of the initial data for the simulations
                Each element in the list is the output from a __call__ to the
                corresponding element in the `self.simulations` list

        Return:
           (float): Log likelyhood of the simulation given the data

        .. math::

           \log(p(f|y))_{model} = -\frac{1}{2}
             (y_k - \mu_k(f))\Sigma_k^{-1}(y_k-\mu_k(f))

        """
        log_like = 0
        for (sim, exp), sim_data in zip(self.simulations, initial_data):
            exp_dep = exp()[1]
            epsilon = exp_dep[0] - sim_data[1]
            log_like -= 0.5 * np.dot(epsilon,
                                      np.dot(inv(sim.get_sigma()), epsilon))
        #end

        return log_like

    def __call__(self):
        """Determines the best candidate EOS function for the models

        Return:
           (tuple): length 2, elements are:

               0. (PhysicsModel): The model which gives best agreement over the
                  space
               1. (list): is of solution history elements are:

                   0. (np.ndarray) Log likelyhood, (nx1) where n is number of
                      iterations
                   1. (np.ndarray) model dof history (nxm) where n is iterations
                      and m is the model dofs
                   2. (np.ndarray) model step history (nxm) where n is iterations
                      and m is the model dofs
        """
        precondition = self.get_option('precondition')
        atol = self.get_option('outer_atol')
        reltol = self.get_option('outer_rtol')
        maxiter = self.get_option('maxiter')
        verb = self.get_option('verb')

        prior_weight = self.get_option('prior_weight')
        sims = self.simulations
        model = self.model

        history = []
        dof_hist = []
        dhat_hist = []
        conv = False
        log_like = 0.0

        initial_data = self.compare(sims, model)
        
        for i in xrange(maxiter):
            dof_hist.append(model.get_dof())
            if verb: print('Iter {} of {}'.format(i, maxiter))
            self._get_sens(sims, model, initial_data)
            #self.plot_sens_matrix()
            # Solve all simulations with the curent model
            if verb: print('prior log like', -self.model_log_like())
            if verb: print('sim log like', -self.sim_log_like(initial_data))
            new_log_like = prior_weight*self.model_log_like()\
                           +self.sim_log_like(initial_data)
            if verb: print('total log like', new_log_like)

            history.append(new_log_like)#, self.model.get_dof()))

            if np.fabs((log_like - new_log_like) / new_log_like) < reltol:
            #np.fabs(log_like - new_log_like) < atol\             
                log_like = new_log_like
                conv = True
                break
            else:
                log_like = new_log_like
            #end

            local_sol = self._local_opt(sims, model, initial_data)

            # model.set_dof(local_sol)
            
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
            max_step = 1.0
            x_list = np.linspace(0, max_step, n_steps)
            for i, x_i in enumerate(x_list):
                model.set_dof(initial_dof + x_i * d_hat)
                costs[i] = self.model_log_like()
                iter_data.append(self.compare(sims,model))
                costs[i] += self.sim_log_like(iter_data[-1])           
            #end
            besti = np.argmax(costs)

            ## Depricated zooming line search
            # besti = 0
            # max_step = 0.5
            # while besti == 0:
            #     x_list = np.linspace(0, max_step, n_steps)
            #     for i, x_i in enumerate(x_list):
            #         model.set_dof(initial_dof + x_i * d_hat)
            #         costs[i] = prior_weight*self.model_log_like()
            #         iter_data.append(self.compare(sims,model))
            #         costs[i] += self.sim_log_like(iter_data[-1])           
            #     #end

            #     besti = np.argmax(costs)

            #     break
            #     max_step /= 2.0
            #     if max_step < 1E-4:
            #         besti = 1
            #         break

            #     if verb and besti==0:
            #         print('Zooming in to max step {:f}'.format(max_step/10.0))
            #     #end
            # #end

            model.set_dof(initial_dof + d_hat * x_list[besti])

            initial_data = iter_data[besti]
            for sim, exp in sims:
                sim.update(model=model)
            # end
            # conv= True
            # break
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
        return model, history

    def get_fisher_matrix(self, simid=0, sens_calc=True):
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
            self._get_sens(self.simulations, self.model)
        #end
        
        for i in xrange(len(self.simulations)):
            dim_k = self.simulations[i][1].shape()
            if i == simid:
                sens_k = self.sens_matrix[i:i+dim_k, :]
            #end
            i += dim_k
        #end
        
        sigma = inv(sim.get_sigma())

        return np.dot(sens_k.T, np.dot(sigma, sens_k))

    def fisher_decomposition(self, fisher, tol=1E-3):
        """Performs a spectral decomposition on the fisher information matrix

        Args:
            fisher(np.ndarray): A nxn array where n is model dof

        Keyword Args:
            tol(float): Eigen values less than tol are ignored

        Return:
            (tuple): Elements are:

                0. (list): Eigenvalues greater than tol
                1. (np.ndarray): nxm array.

                      - n is number of eigenvalues greater than tol
                      - m is model dof

                2. (np.ndarray): nxm array

                      - n is the number of eigenvalues greater than tol
                      - m is an arbutary dimension of independent variable

                3. (np.ndarray): vector of independent varible

        """
        eos = copy.deepcopy(self.model)

        # Spectral decomposition of info matrix and sort by eigenvalues
        eig_vals, eig_vecs = np.linalg.eigh(fisher)
        eig_vals = np.maximum(eig_vals, 0)        # info is positive definite

        i = np.argsort(eig_vals)[-1::-1]
        vals = eig_vals[i]
        vecs = eig_vecs.T[i]

        n_vals = max(len(np.where(vals > vals[0]*1e-2)[0]), 3)
        n_vecs = max(len(np.where(vals > vals[0]*1e-2)[0]), 3)

        # Find range of v that includes support of eigenfunctions
        knots = eos.get_t()
        v_min = knots[0]
        v_max = knots[-1]
        vol = np.logspace(np.log10(v_min), np.log10(v_max), len(knots)*10)
        funcs = []
        for vec in vecs[:n_vecs]:
            eos.set_dof(vec)
            funcs.append(eos(vol))
            if funcs[-1][np.argmax(np.fabs(funcs[-1]))] < 0:
                funcs[-1] *= -1
            #end
        funcs = np.array(funcs)

        return vals[:n_vals], vecs, funcs, vol

    def _local_opt(self, sims, model, initial_data):
        """Soves the quadratic problem for minimization of the log likelyhood

        Args:
           sims(list): The simulation/experiment pairs
           model(PhysicsModel): The model being examined
           initial_data(list): The initial data corresonding to simulations
               from sims

        Return:
            (np.ndarray):
                The step direction for greates improvement in log lieklyhood
        """
        constrain = self.get_option('constrain')
        debug = self.get_option('debug')
        
        # Get constraints
        g_mat, h_vec = self._get_constraints(model)

        p_mat, q_mat = self._get_model_pq(model)
        if debug: print(q_mat)
        tmp = self._get_sim_pq(sims, model, initial_data)

        p_mat += tmp[0]
        q_mat += tmp[1]

        p_mat *= 0.5

        if debug: print(q_mat)

        solvers.options['show_progress'] = False
        solvers.options['debug'] = False
        solvers.options['maxiters'] = 100  # 100 default
        solvers.options['reltol'] = 1e-6   # 1e-6 default
        solvers.options['abstol'] = 1e-7   # 1e-7 default
        solvers.options['feastol'] = 1e-7  # 1e-7 default

        try:
            if constrain:
                sol = solvers.qp(matrix(p_mat), matrix(q_mat),
                                 matrix(g_mat), matrix(h_vec))
            else:
                sol = solvers.qp(matrix(p_mat), matrix(q_mat))
        except ValueError as inst:
            print(inst)
            print("G "+str(g_mat.shape))
            print("P "+str(p_mat.shape))
            print("h "+str(h_vec.shape))
            print("q "+str(q_mat.shape))            
            pdb.post_mortem()
        if sol['status'] != 'optimal':
            for key, value in sol.items():
                print(key, value)
            raise RuntimeError('{} The optimization algorithm could not locate\
                               an optimal point'.format(self.get_inform(1)))
        return sol

    def _get_constraints(self, model):
        r"""Get the constraints on the model

        .. note::

             This method is specific to the model type under consideration.
             This implementation is onlt for spline models of EOS

        Args:
           model(PhysicsModel): The physics model subject to
                                physical constraints
        Return:
           ():
           ():

        Method

        Calculate constraint matrix G and vector h.  The
        constraint enforced by :py:class:`cvxopt.solvers.qp` is

        .. math::

           G*x \leq  h

        Equivalent to :math:`max(G*x - h) \leq 0`

        Since

        .. math::

           c_{f_{new}} = c_f+x,

        .. math::

           G(c_f+x) \leq 0

        is the same as

        .. math::

           G*x \leq -G*c_f,

        and :math:`h = -G*c_f`

        Here are the constraints for :math:`p(v)`:

        p'' positive for all v
        p' negative for v_max
        p positive for v_max

        For cubic splines between knots, f'' is constant and f' is
        affine.  Consequently, :math:`f''rho + 2f'` is affine between knots
        and it is sufficient to check eq:star at the knots.

        """
        precondition = self.get_option('precondition')
        c_model = copy.deepcopy(model)
        spline_end = c_model.get_option('spline_end')
        dim = c_model.shape()[0]

        v_unique = c_model.get_t()[spline_end-1:1-spline_end]
        n_constraints = len(v_unique) + 2


        G_mat = np.zeros((n_constraints, dim))
        c_tmp = np.zeros(dim)
        c_init = c_model.get_dof()
        scaling = c_model.get_scaling()
        for i in range(dim):
            c_tmp[i] = 1.0
            c_model.set_dof(c_tmp)
            G_mat[:-2, i] = -c_model.derivative(2)(v_unique)
            G_mat[-2, i] = c_model.derivative(1)(v_unique[-1])
            G_mat[-1, i] = -c_model(v_unique[-1])
            c_tmp[i] = 0.0
        # end

        h_vec = -np.dot(G_mat, c_init)

        scale = np.abs(h_vec)
        scale = np.maximum(scale, scale.max()*1e-15)
        # Scale to make |h[i]| = 1 for all i
        HI = np.diag(1.0/scale)
        h_vec = np.dot(HI, h_vec)
        G_mat = np.dot(HI, G_mat)

        if precondition:
            G_mat = np.dot(G_mat, scaling)
        #end
        
        return G_mat, h_vec

    def _get_model_pq(self, model):
        """Gets the quadratic optimizaiton matrix contributions from the prior

        Args:
           model(PhysicsModel): A physics model with degrees of freedom

        Retrun:
            (tuple): elements are

                0. (np.ndarray): `p`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF
        """

        precondition = self.get_option('precondition')
        prior_scale = model.get_scaling()
        prior_var = inv(model.get_sigma())

        prior_delta = model.get_dof() - model.prior.get_dof()
        
        if precondition:
            return np.dot(prior_scale, np.dot(prior_var, prior_scale)),\
                -np.dot(prior_scale, np.dot(prior_delta, prior_var))
        else:
            return prior_var, -np.dot(prior_delta, prior_var)

    def _get_sim_pq(self, sims, model, initial_data):
        """Gets the QP contribytions from the model

        .. note::

             This method is specific to the model type under consideration.
             This implementation is onlt for spline models of EOS

        Args:
           sims(list): A list of tuples of experiments each tuple contains
                       [0] the simulation
                       [1] the corresponding experiment
           model(PhysicsModel): A physics model with degrees of freedom
           initial_data(list): A list of the inital results from the simulations
                               in the same order as in the `sim` list

        Return:
            (tuple): Elements are:

                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """

        precondition = self.get_option('precondition')
        prior_scale = model.get_scaling()
        debug = self.get_option('debug')
        d_mat = self.sens_matrix
        p_mat = np.zeros((self.shape()[1], self.shape()[1]))
        q_mat = np.zeros(self.shape()[1])

        i = 0

        for (sim, exp), sim_data in zip(sims, initial_data):
            dim_k = exp.shape()
            sens_k = d_mat[i:i+dim_k, :]
            exp_dep  = exp()[1]
            # basis_k = sim_data[2].get_basis(exp_indep,
            #                                 spline_end = spline_end)
            # sens_k = np.dot(sens_k, basis_k)
            epsilon = exp_dep[0] - sim_data[1]
            if debug: print(epsilon)
            p_mat += np.dot(np.dot(sens_k.T, inv(sim.get_sigma())), sens_k)
            q_mat -= np.dot(np.dot(epsilon, inv(sim.get_sigma())), sens_k)
            i += dim_k
        #end

        if precondition:
            p_mat = np.dot(prior_scale, np.dot(p_mat, prior_scale))
            q_mat = np.dot(prior_scale, q_mat)
        #end

        return p_mat, q_mat

    def compare(self, sims, model):
        """
        
        Args:
            sims(list): List of tuples of simulation, experiment pairs
            model(PhysicsModel): A valid physics model instance

        Return:
            (list):
                List of lists for experiment comparison data
             
                0. independent value
                1. dependent value of interest
        """

        data = []
        for sim, exp in sims:
            sim.update(model = model)
            exp_indep = exp()[0]
            sim_data = sim()
            data.append([exp_indep,
                         -1*sim.compare(exp_indep,
                                        np.zeros(len(exp_indep)),
                                        sim_data)])
        #end

        return data
        #end
    def _get_sens(self, sims, model, initial_data = None):
        """Gets the sensitivity of the simulated experiment to the EOS
        
        The sensitivity matrix is the attribute `self.sens_matrix` which is set
        by this method

        .. note::

             This method is specific to the model type under consideration.
             This implementation is onlt for spline models of EOS

        Args:
            sims(list): List of tuples of simulation, experiment pairs
            model(PhysicsModel): A valid physics model instance

        Keyword Args:
            initial_data(list): The results of each simulation with the current
                best model. Each element in the list corresponds tho the output
                from a `__call__` to each element in the `self.simulations` list
   
        Return:
            None
        """

        model = copy.deepcopy(model)
        sims = copy.deepcopy(sims)
        sens_tol = 1E-12
        step_frac = 2E-2

        original_dofs = model.get_dof()

        resp_mat = np.zeros(self.shape())
        inp_mat = np.zeros((self.shape()[1],self.shape()[1]))
        new_dofs = copy.deepcopy(original_dofs)

        if initial_data is None:
            initial_data = self.compare(sims, model)
        #end
        
        for i in xrange(len(original_dofs)):
            step = float(new_dofs[i] * step_frac)
            new_dofs[i] += step

            model.set_dof(new_dofs)
            inp_mat[:, i] = (model.get_dof() - original_dofs) 
            j = 0 # counter for the starting column of this sim's data
            for (sim, exp), sim_data in zip(sims, initial_data):
                sim.update(model=model)
                new_data = sim()
                dim = sim.shape()
                delta = -sim.compare(sim_data[0],
                                     sim_data[1],
                                     new_data)
                # If the sensitivity is less than the tolerance, make it
                # zero
                # delta = np.where(np.fabs(delta) > sens_tol,\
                #                                   delta,\
                #                                   np.zeros(len(delta)))
                resp_mat[j:j+dim, i] = delta
                j += dim
            #end
            new_dofs[i] -= step
        #end

        # Use a better algorithm!
        # sens_matrix = np.dot(resp_mat, inv(inp_mat))
        

        sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T
        sens_matrix = np.where(np.fabs(sens_matrix) > 1E-20,\
                                          sens_matrix,\
                                          np.zeros(self.shape()))

        # Return all simulations to original state
        model.set_dof(original_dofs)
        for sim, exp in sims:
            sim.update(model=model)
        #end

        self.sens_matrix = sens_matrix

    def plot_fisher_data(self, fisher_data, filename=None):
        """

        Args:
            fisher_dat(tuple): Data from the fisher_decomposition function
                               *see docscring for definition*

        Keyword Args:
            filename(str or None): If none, do not make a hardcopy, otherwise
                                   save to the file specified

        """

        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        eigs = fisher_data[0]
        eig_vects = fisher_data[1]
        eig_func = fisher_data[2]
        indep = fisher_data[3]

#        ax1.bar(np.arange(eigs.shape[0]), eigs, width=0.9, color='black',
#                edgecolor='none', orientation='vertical')
        ax1.semilogy(eigs, 'sk')
        ax1.set_xlabel("Eigenvalue number")
        ax1.set_ylabel(r"Eigenvalue / Pa$^{-2}$")
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        styles = ['-g', '-.b', '--m', ':k', '-c', '-.y', '--r']*\
                 int(math.ceil(eig_func.shape[0]/4.0))

        for i in xrange(eig_func.shape[0]):
            ax2.plot(indep, eig_func[i], styles[i],
                     label="{:d}".format(i))
        #end
        
        ax2.legend(loc = 'best')
        ax2.get_legend().set_title("Eigen-\nfunctions", prop = {'size': 7})
        ax2.set_xlabel(r"Specific volume / cm$^3$ g$^{-1}$")
        ax2.set_ylabel("Eigenfunction response / Pa")
        
        fig.tight_layout()

        return fig

    def plot_convergence(self, hist, dof_hist=None, axis = None, hardcopy = None):
        """
        
        Args:
            axis(plt.Axis): A valid :py:class:`plt.Axis` object on which to plot.
                if none, generates a new figure
            hist(list): Convergence history of log likelyhood
            dof_hist(list): List of model DOFs at each iteration
        
        """

        if axis is None:
            fig = plt.figure()
            ax1 = fig.gca()
        else:
            fig = None
            ax1 = axis
        #end

        ax1.semilogy(-np.array(hist), '-k')

        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        ax1.set_xlabel('Iteration number')
        ax1.set_ylabel('Negative a posteori log likelihood')
        
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # for i in xrange(dof_hist.shape[1]):
        #     ax1.plot(dof_hist[:, i]/dof_hist[0, i])
        # #end
        # fig.suptitle('Convergence of iterative process')
        # ax1.set_ylabel('Spline knot value')
        # ax1.set_xlabel('Iteration number')
        # fig.savefig('EOS_convergence.pdf')

    def plot_sens_matrix(self):
        """Prints the sensitivity matrix
        """
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
            ax1.plot(sens_matrix[:, i], label="{:4.3f}".format(knot_post[i]))
        ax1.legend(loc='best')

        for i in xrange(10, 20):
            ax2.plot(sens_matrix[:, i], label="{:4.3f}".format(knot_post[i]))
        ax2.legend(loc='best')

        for i in xrange(20, 30):
            ax3.plot(sens_matrix[:, i], label="{:4.3f}".format(knot_post[i]))
        ax3.legend(loc='best')

        for i in xrange(30, 40):
            ax4.plot(sens_matrix[:, i], label="{:4.3f}".format(knot_post[i]))
        ax4.legend(loc='best')

        for i in xrange(40, 50):
            ax5.plot(sens_matrix[:, i], label="{:4.3f}".format(knot_post[i]))
        ax5.legend(loc='best')



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
        if __name__ == '__main__':
            from FUNCLE.Experiments.GunModel import Gun
            from FUNCLE.Experiments.Stick import Stick
            from FUNCLE.Models.Isentrope import EOSModel, EOSBump
        else:
            from ..Experiments.GunModel import Gun
            from ..Experiments.Stick import Stick
            from ..Models.Isentrope import EOSModel, EOSBump
        # end
        
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

    def test_singel_case_pq_mod(self):
        """Tests the P and Q matrix generation for a single case
        """

        bayes = Bayesian(simulations=[(self.sim1, self.exp1)],
                         model=self.eos_model)

        #print(bayes)
        shape = bayes.shape()

        exp_shape = self.exp1.shape()
        sim_shape = self.sim1.shape()
        mod_shape = self.eos_model.shape()[0]

        initial_data = bayes.compare(bayes.simulations, bayes.model)
        P, q = bayes._get_sim_pq([(self.sim1, self.exp1)], self.eos_model, initial_data)

        self.assertEqual(P.shape, (mod_shape, mod_shape))
        self.assertEqual(q.shape, (mod_shape, ))

    def test_mlt_case_pq_mod(self):
        """Tests the P and Q matrix generation for a multiple case
        """

        bayes = Bayesian(simulations=[(self.sim1, self.exp1),
                                        (self.sim2, self.exp2)],
                         model=self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape() + self.exp2.shape()
        sim_shape = self.sim1.shape() + self.sim2.shape()
        mod_shape = self.eos_model.shape()[0]

        initial_data = bayes.compare(bayes.simulations, bayes.model)
        
        P, q = bayes._get_sim_pq([(self.sim1, self.exp1), (self.sim2, self.exp2)],
                                 self.eos_model, initial_data)

        self.assertEqual(P.shape, (mod_shape, mod_shape))
        self.assertEqual(q.shape, (mod_shape, ))

    def test_gun_case_sens(self):
        """
        """

        bayes = Bayesian(simulations=[(self.sim1, self.exp1)],
                         model=self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape()
        sim_shape = self.sim1.shape()
        mod_shape = self.eos_model.shape()[0]

        initial_data = bayes.compare(bayes.simulations, bayes.model)

        bayes._get_sens([(self.sim1, self.exp1)], self.eos_model, initial_data)
        self.assertEqual(bayes.sens_matrix.shape, (exp_shape, mod_shape))
        # bayes.plot_sens_matrix(initial_data)

    def test_stick_case_sens(self):
        """Test of the sensitivity of the stick model to the eos
        """

        bayes = Bayesian(simulations=[(self.sim2, self.exp2)],
                         model=self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp2.shape()
        sim_shape = self.sim2.shape()
        mod_shape = self.eos_model.shape()[0]

        initial_data = bayes.compare(bayes.simulations, bayes.model)

        bayes._get_sens([(self.sim2, self.exp2)],
                        self.eos_model, initial_data)
        self.assertEqual(bayes.sens_matrix.shape, (exp_shape, mod_shape))
        #bayes.plot_sens_matrix(initial_data)

    def test_mult_case_sens(self):
        """Test of sens matrix generation for mult models
        """
        bayes = Bayesian(simulations=[(self.sim1, self.exp1),
                                      (self.sim2, self.exp2)],
                         model = self.eos_model)

        shape = bayes.shape()

        exp_shape = self.exp1.shape() + self.exp2.shape()
        sim_shape = self.sim1.shape() + self.sim1.shape()
        mod_shape = self.eos_model.shape()[0]

        initial_data = bayes.compare(bayes.simulations, bayes.model)

        bayes._get_sens([(self.sim1, self.exp1),
                         (self.sim2, self.exp2)],
                        self.eos_model, initial_data)

        self.assertEqual(bayes.sens_matrix.shape, (exp_shape, mod_shape))

        # bayes.plot_sens_matrix(initial_data)

    def test_model_pq(self):
        """Tests the pq matrix generation by the model
        """

        pass
    
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
        plt.show()

if __name__ == '__main__':

    unittest.main(verbosity=4)
