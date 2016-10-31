# /usr/bin/pyton
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
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

try:
    from cvxopt import matrix, solvers
except:
    print('cvxopt package not found. Install by running\n'
          '`$ pip install cvxopt`')
# end

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

    def __init__(self, simulations, models, opt_key=None, name='Bayesian',
                 *args, **kwargs):
        """Instantiates the Bayesian analysis

        Args:
           simulations(dict): A dictionary of simulation, experiment pairs
           models(dict): A dictionary of models
           opt_key(str): The model from models which is being optimized
        Keyword Args:
            name(str): Name for the analysis.('Bayesian')

        Return:
           None

        """

        # Name: [type, default, min, max, units, note]
        def_opts = {
            'outer_atol': [float, 1E-6, 0.0, 1.0, '-',
                           'Absolute tolerance on change in likelyhood for'
                           'outer loop convergence'],
            'outer_rtol': [float, 1E-4, 0.0, 1.0, '-',
                           'Relative tolerance on change in likelyhood for'
                           'outer loop convergence'],
            'maxiter': [int, 6, 1, 100, '-',
                        'Maximum iterations for convergence'
                        'of the likelyhood'],
            'constrain': [bool, True, None, None, '-',
                          'Flag to constrain the optimization'],
            'precondition': [bool, True, None, None, '-',
                             'Flag to scale the problem'],
            'debug': [bool, False, None, None, '-',
                      'Flag to print debug information'],

            'verb': [bool, True, None, None, '-',
                     'Flag to print stats during optimization'],
            'prior_weight': [float, 1.0, 0.0, 1.0, '-',
                             'Weighting of the prior when calculating log'
                             'likelihood']
        }

        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)

        self.models, self.simulations = self._check_inputs(models, simulations)

        if opt_key is not None:
            if opt_key not in self.models:
                raise KeyError("{:} opt key does not refer to a valid model"
                               .format(self.get_inform(1)))
            else:
                self.opt_key = opt_key
            # end
        elif len(self.models.keys()) == 1:
            self.opt_key = self.models.keys()[0]
        else:
            raise IOError("{:} Must define opt key if more than one model"
                          "used".format(self.get_inform(1)))
        # end
    # end

    def _check_inputs(self, models, simulations):
        """Checks that the values for model and simulaion are valid
        """
        if not isinstance(models, dict):
            raise TypeError('001 {:}, model must be provided as a dictonary'
                            'with the key being the name of the model'
                            .format(self.get_inform(1)))
        elif not np.all([isinstance(models[key], PhysicsModel)
                         for key in models]):
            raise TypeError('002 {:}, all models must be a PhysicsModel type'
                            .format(self.get_inform(1)))
        elif not isinstance(simulations, dict):
            raise TypeError('003 {:}, sumulations must be provided as a'
                            'dictonary with the key being the name of the'
                            'simulation'
                            .format(self.get_inform(1)))
        # end

        if np.all([isinstance(simulations[key], (list, tuple))
                   for key in simulations]):
            if not np.all([len(simulations[key]) == 2 for key in simulations]):
                raise TypeError('005 {:}, each list for simulation must be'
                                'length 2'
                                .format(self.get_inform(1)))
            elif not np.all([[isinstance(sim, Experiment)
                              for sim in simulations[key]]
                             for key in simulations]):
                raise TypeError('006 {:}, each element in the simulation list'
                                'must be an experiment type'
                                .format(self.get_inform(1)))
            else:
                sim_out = {}
                for key in simulations:
                    sim_out[key] = {'sim': copy.deepcopy(simulations[key][0]),
                                    'exp': copy.deepcopy(simulations[key][1])}
                # end
            # end
        elif np.all([isinstance(simulations[key], dict)
                     for key in simulations]):
            if not np.all(['sim' in simulations[key] and
                           'exp' in simulations[key]
                          for key in simulations]):
                raise TypeError('007 {:}Each dictionary must contain the keys'
                                'sim and exp'.format(self.get_inform(1)))
            elif not np.all([[isinstance(sim, Experiment)
                              for sim in [simulations[key]['sim'],
                                          simulations[key]['exp']]]
                             for key in simulations]):
                raise TypeError('008 {:}, each element in the simulation dict'
                                'must be an experiment type'
                                .format(self.get_inform(1)))
            else:
                sim_out = copy.deepcopy(simulations)
        else:
            raise TypeError('004 {:}, each simulation must be a list or tuple'
                            'of experiment objects or a dict with keys sim and'
                            'exp'.format(self.get_inform(1)))
        # end

        return copy.deepcopy(models), sim_out

    def _on_str(self):
        """Print method for bayesian model

        Args:
            None

        Return:
           (str): String describing the Bayesian object
        """
        out_str = '''
=======================================================
=======================================================
           ____                        _
          |  _ \                      (_)
          | |_) | __ _ _   _  ___  ___ _  __ _ _ __
          |  _ < / _` | | | |/ _ \/ __| |/ _` | '_ \
          | |_) | (_| | |_| |  __/\__ \ | (_| | | | |
          |____/ \__,_|\__, |\___||___/_|\__,_|_| |_|
                        __/ |
                       |___/
=======================================================
=======================================================
'''

        for key in self.models:
            out_str += 'Model {:}\n'.format(key)
            out_str += str(self.models[key])
            out_str += 'Model {:} Prior\n'.format(key)
            out_str += str(self.models[key].prior)
        # end

        for key in self.simulations:
            out_str += 'Model {:} simulation'.format(key)
            out_str += str(self.simulations[key]['sim'])
            out_str += 'Model {:} experiment'.format(key)
            out_str += str(self.simulations[key]['exp'])
        # end

        return out_str

    def update(self, simulations=None, models=None):
        """Updates the properties of the bayesian analtsis

        Keyword Args:
           simulations(dict): Dictonary of simulation experiment pairs
                                    (Default None)
           models(dict): Dictonary of models
                                (Default None)

        Return:
            None
        """

        new_object = copy.deepcopy(self)

        if simulations is None:
            simulations = new_object.simulations
        if models is None:
            models = new_object.models
        # end

        new_object.models, new_object.simulations\
            = self._check_inputs(models, simulations)

        return new_object

    def shape(self):
        """Gets the dimenstions of the problem

        Return:
           (tuple): The (n, m) dimensions of the problem

               - n is the total degrees of freedom of all the model responses
               - m is the degrees of freedom of the model
        """
        dof_model = self.models[self.opt_key].shape()

        dof_sim = 0
        for key in self.simulations:
            dof_sim += self.simulations[key]['exp'].shape()
        # end

        return (dof_sim, dof_model)

    def model_log_like(self):
        r"""Gets the log likelihood of the `self.model` given that model's prior

        Return:
           (float): Log likelihood of the model

        .. math::

           \log(p(f|y))_{model} = -\frac{1}{2}(f - \mu_f)\Sigma_f^{-1}(f-\mu_f)

        """

        return self.models[self.opt_key].get_log_like()

    def sim_log_like(self, initial_data):
        r"""Gets the log likelihood of the simulations given the data

        Args:
            model(PhysicsModel): The model under investigation
            initial_data(list): A list of the initial data for the simulations
                Each element in the list is the output from a __call__ to the
                corresponding element in the `self.simulations` list

        Return:
           (float): Log likelihood of the simulation given the data

        .. math::

           \log(p(f|y))_{model} = -\frac{1}{2}
             (y_k - \mu_k(f))\Sigma_k^{-1}(y_k-\mu_k(f))

        """
        sims = self.simulations
        log_like = 0
        for key in sims:
            log_like += sims[key]['sim'].\
                get_log_like(self.models, initial_data[key], sims[key]['exp'])
        # end

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
                2. (np.ndarray) sensitity matrix of all experiments to
                   the model
        """
        def outer_loop_iteration(analysis, log_like, intitial_data):
            """Performs a single step of the outer loop optimization

            Args:
                analysis(Bayesian): The collection of simulations and
                                    experiments to be tested
                log_like(float): The log likelyhood of the last iteration
                initial_data(list): The results of the simulations at the
                                    current model state
            Return:
                (Bayesian): A copy of the analysis object with the new model
                            estimate
                (float): The log likelyhood of the updated model
                (list): The results of the simulations at the new model state
                (list): A list of the model DOFs
                (bool): A flag, true of the log likelyhood has converged
            """
            precondition = self.get_option('precondition')
            atol = self.get_option('outer_atol')
            reltol = self.get_option('outer_rtol')
            verb = self.get_option('verb')
            prior_weight = self.get_option('prior_weight')

            sens_matrix = analysis._get_sens(initial_data)
            models = analysis.models
            opt_key = analysis.opt_key
            opt_model = models[opt_key]
            model_dict = analysis.models

            # Solve all simulations with the curent model
            if verb:
                print('prior log like', -analysis.model_log_like())
            if verb:
                print('sim log like', -analysis.sim_log_like(initial_data))
            new_log_like = prior_weight * analysis.model_log_like()\
                + analysis.sim_log_like(initial_data)

            if verb:
                print('total log like', new_log_like)

            if np.fabs((log_like - new_log_like) / new_log_like) < reltol:
                return copy.deepcopy(analysis),\
                    new_log_like,\
                    initial_data,\
                    opt_model.get_dof(),\
                    True

            # end

            local_sol = analysis._local_opt(initial_data,
                                            sens_matrix)

            d_hat = np.array(local_sol['x']).reshape(-1)
            if precondition:
                d_hat = np.dot(opt_model.get_scaling(), d_hat)
            # end

            n_steps = 5
            costs = np.zeros(n_steps)
            iter_data = []
            initial_dof = copy.deepcopy(opt_model.get_dof())
            max_step = 1.0
            x_list = np.linspace(0, max_step, n_steps)
            for i, x_i in enumerate(x_list):
                model_dict[opt_key] = opt_model.update_dof(
                    initial_dof + x_i * d_hat)
                new_analysis = analysis.update(models=model_dict)
                costs[i] = prior_weight * new_analysis.model_log_like()
                iter_data.append(new_analysis.get_data())
                costs[i] += new_analysis.sim_log_like(iter_data[-1])
            # end

            besti = np.argmax(costs)

            model_dict[opt_key] = opt_model.update_dof(
                initial_dof + d_hat * x_list[besti])

            return (analysis.update(models=model_dict),
                    new_log_like,
                    iter_data[besti],
                    model_dict[opt_key].get_dof(),
                    False)

        # end

        maxiter = self.get_option('maxiter')
        verb = self.get_option('verb')
        history = []
        dof_hist = []
        log_like = 0.0

        initial_data = self.get_data()

        conv = False
        analysis = copy.deepcopy(self)
        log_like = 0
        i = 0
        while not conv and i < maxiter:
            if verb:
                print('Iter {} of {}'.format(i, maxiter))
            analysis, log_like, initial_data, model_dof, conv =\
                outer_loop_iteration(analysis, log_like, initial_data)
            history.append(log_like)
            dof_hist.append(model_dof)
            i += 1
        # end

        sens_matrix = analysis._get_sens(initial_data)

        if not conv:
            print("{}: Outer loop could not converge to the given"
                  "tolerance in the maximum number of iterations"
                  .format(self.get_inform(1)))
        # end

        dof_hist = np.array(dof_hist)
        history = np.array(history)

        return analysis, (history, dof_hist), sens_matrix

    def fisher_decomposition(self, fisher, tol=1E-3):
        """Calculate a spectral decomposition of the fisher information matrix

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

        # Spectral decomposition of info matrix and sort by eigenvalues
        eig_vals, eig_vecs = np.linalg.eigh(fisher)
        eig_vals = np.maximum(eig_vals, 0)        # info is positive definite

        i = np.argsort(eig_vals)[-1::-1]
        vals = eig_vals[i]
        vecs = eig_vecs.T[i]

        n_vals = max(len(np.where(vals > vals[0] * 1e-2)[0]), 3)
        n_vecs = max(len(np.where(vals > vals[0] * 1e-2)[0]), 1)

        # Find range of v that includes support of eigenfunctions
        knots = self.models[self.opt_key].get_t()
        v_min = knots[0]
        v_max = knots[-1]
        vol = np.logspace(np.log10(v_min), np.log10(v_max), len(knots) * 10)
        funcs = []
        for vec in vecs[:n_vecs]:
            new_model = self.models[self.opt_key].\
                update_dof(vec * self.models[self.opt_key].get_dof())
            funcs.append(new_model(vol))
            if funcs[-1][np.argmax(np.fabs(funcs[-1]))] < 0:
                funcs[-1] *= -1
            # end
        funcs = np.array(funcs)

        return vals[:n_vals], vecs, funcs, vol

    def _local_opt(self, initial_data, sens_matrix):
        """Soves the quadratic problem for maximization of the log likelihood

        Args:
           initial_data(list): The initial data corresonding to simulations
               from sims
           sens_matrix(np.array): The sensitivity matrix about the model

        Return:
            (np.ndarray):
                The step direction for greates improvement in log lieklyhood
        """
        constrain = self.get_option('constrain')
        debug = self.get_option('debug')

        # Get constraints
        g_mat, h_vec = self._get_constraints()

        p_mat, q_vec = self._get_model_pq()

        tmp = self._get_sim_pq(initial_data, sens_matrix)

        p_mat += tmp[0]
        q_vec += tmp[1]

        p_mat *= 0.5

        solvers.options['show_progress'] = False
        solvers.options['debug'] = False
        solvers.options['maxiters'] = 100  # 100 default
        solvers.options['reltol'] = 1e-6   # 1e-6 default
        solvers.options['abstol'] = 1e-7   # 1e-7 default
        solvers.options['feastol'] = 1e-7  # 1e-7 default

        try:
            if constrain:
                sol = solvers.qp(matrix(p_mat), matrix(q_vec),
                                 matrix(g_mat), matrix(h_vec))
            else:
                sol = solvers.qp(matrix(p_mat), matrix(q_vec))
        except ValueError as inst:
            print(inst)
            print("G " + str(g_mat.shape))
            print("P " + str(p_mat.shape))
            print("h " + str(h_vec.shape))
            print("q " + str(q_vec.shape))
            pdb.post_mortem()
        if sol['status'] != 'optimal':
            for key, value in sol.items():
                print(key, value)
            raise RuntimeError('{} The optimization algorithm could not locate'
                               'an optimal point'.format(self.get_inform(1)))
        return sol

    def _get_constraints(self):
        r"""Get the constraints on the model
        """
        return self.models[self.opt_key].get_constraints(
            scale=self.get_option('precondition'))

    def _get_model_pq(self):
        """Gets the quadratic optimizaiton matrix contributions from the prior

        Args:
            None
        Retrun:
            (tuple): elements are

                0. (np.ndarray): `p`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF
        """
        return self.models[self.opt_key].get_pq(
            scale=self.get_option('precondition'))

    def _get_sim_pq(self, initial_data, sens_matrix):
        """Gets the QP contributions from the model

        .. note::

             This method is specific to the model type under consideration.
             This implementation is onlt for spline models of EOS

        Args:
           initial_data(list): A list of the inital results from the simulations
                               in the same order as in the `sim` list
           sens_matric(np.ndarray): The sensitivity matrix

        Return:
            (tuple): Elements are:

                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """
        sims = self.simulations

        p_mat = np.zeros((self.shape()[1], self.shape()[1]))
        q_mat = np.zeros(self.shape()[1])

        i = 0
        for key in sims:
            p_tmp, q_tmp = sims[key]['sim'].get_pq(
                self.models,
                self.opt_key,
                initial_data[key],
                sims[key]['exp'],
                sens_matrix[key],
                scale=self.get_option('precondition'))
            p_mat += p_tmp
            q_mat += q_tmp
        # end

        return p_mat, q_mat

    def get_data(self):
        """Generates a set of data at each expeimental data-point

        For each pair of simulations and experiments, generates the
        the simulation response.

        Args:
            None
        Return:
            (list):
                List of lists for experiment comparison data

                0. independent value
                1. dependent value of interest

        """
        sims = self.simulations

        data = {}
        for key in sims:
            expdata = sims[key]['exp']()
            simdata = sims[key]['sim'](self.models)
            tmp = -1 * sims[key]['sim'].compare(
                expdata[0], np.zeros(expdata[0].shape), simdata)
            data[key] = (expdata[0], [tmp], simdata[2])
        # end

        return data

    def _get_sens(self, initial_data=None):
        """Gets the sensitivity of the simulated experiment to the EOS

        The sensitivity matrix is the attribute `self.sens_matrix` which is set
        by this method

        .. note::

             This method is specific to the model type under consideration.
             This implementation is onlt for spline models of EOS

        Args:
            None

        Keyword Args:
            initial_data(list): The results of each simulation with the current
                best model. Each element in the list corresponds tho the output
                from a `__call__` to each element in the `self.simulations` list

        Return:
            None
        """
        sims = self.simulations
        models = self.models
        opt_key = self.opt_key

        if initial_data is None:
            initial_data = self.get_data()
        # end

        sens_matrix = {}
        for key in self.simulations:
            sens_matrix[key] = sims[key]['sim'].\
                get_sens(models, opt_key, initial_data[key])
        # end

        return sens_matrix

    def get_hessian(self, initial_data=None, simid=None):
        """Gets the Hessian (matrix of second derrivatives) of the simulated
        experiments to the EOS

        .. math::

          H(f_i) = \begin{smallmatrix}
            \frac{\partial^2 f_i}{\partial \mu_1^2} & \frac{\partial^2 f_i}{\partial \mu_1 \partial \mu_2} & \ldots & \frac{\partial^2 f_i}{\partial \mu_1 \partial \mu_n}\\
            \frac{\partial^2 f_i}{\partial \mu_2 \partial \mu_1} & \frac{\partial^2 f_i}{\partial \mu_2^2} & \ldots & \frac{\partial^2 f_i}{\partial \mu_2 \partial \mu_n}\\
            \frac{\partial^2 f_i}{\partial \mu_n \partial \mu_1} & \frac{\partial^2 f_i}{\partial \mu_n \partial \mu_2} & \ldots & \frac{\partial^2 f_i}{\partial \mu_n^2}
            \end{smallmatrix}

        .. math::

          H(f) = (H(f_1), H(f_2), \ldots , H(f_n))

        where

        .. math::

          f \in \mathcal{R}^m \\
          \mu \in \mathcal{R}^m
        """

        sims = self.simulations
        models = self.models
        opt_key = self.opt_key

        if initial_data is None:
            initial_data = self.get_data()
        # end

        hessian = {}
        for key in self.simulations:
            hessian[key] = sims[key]['sim'].\
                _get_hessian(models, opt_key, initial_data[key])
        # end

        return hessian

    def plot_fisher_data(self, fisher_data, axes=None, fig=None,
                         linestyles=[], labels=[]):
        """

        Args:
            fisher_dat(tuple): Data from the fisher_decomposition function
                               *see docscring for definition*

        Keyword Args:
            axes(plt.Axes): *Ignored*
            fig(plt.Figure): A valid figure to plot on
            linestyles(list): A list of valid linestyles *Ignored*
            labels(list): A list of labels *Ignored*
        """

        if fig is None:
            fig = plt.Figuree()
        else:
            pass
        # end

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
        ax1.set_xlim(-0.5, len(eigs) - 0.5)
        ax1.set_ylim([0.1 * min(eigs[np.nonzero(eigs)]), 10 * max(eigs)])
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        styles = ['-g', '-.b', '--m', ':k', '-c', '-.y', '--r'] *\
            int(math.ceil(eig_func.shape[0] / 7.0))

        for i in range(eig_func.shape[0]):
            ax2.plot(indep, eig_func[i], styles[i],
                     label="{:d}".format(i))
        # end

        ax2.legend(loc='best')
        ax2.get_legend().set_title("Eigen-\nfunctions", prop={'size': 7})
        ax2.set_xlabel(r"Specific volume / cm$^3$ g$^{-1}$")
        ax2.set_ylabel("Eigenfunction response / Pa")

        fig.tight_layout()

        return fig

    def plot_convergence(self, hist, axes=None, linestyles=['-k'], labels=[]):
        """

        Args:
            hist(tuple): Convergence history, elements
                0. (list): MAP history
                1. (list): DOF history

        Keyword Args:
            axes(plt.Axes): The axes on which to plot the figure, if None,
                creates a new figure object on which to plot.
            linestyles(list): Strings for the linestyles
            labels(list): Strings for the labels

        """

        if axes is None:
            fig = plt.figure()
            ax1 = fig.gca()
        else:
            fig = None
            ax1 = axes
        # end

        ax1.semilogy(-np.array(hist[0]), linestyles[0])

        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.set_xlabel('Iteration number')
        ax1.set_ylabel('Negative a posteori log likelihood')

        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # for i in range(dof_hist.shape[1]):
        #     ax1.plot(dof_hist[:, i]/dof_hist[0, i])
        # # end
        # fig.suptitle('Convergence of iterative process')
        # ax1.set_ylabel('Spline knot value')
        # ax1.set_xlabel('Iteration number')
        # fig.savefig('EOS_convergence.pdf')

    def plot_sens_matrix(self, simid, axes=None, fig=None,
                         sens_matrix=None, labels=[], linestyles=[]):
        """Prints the sensitivity matrix

        Args:
            simid(str): The key for the simulation to plot
            axes(plt.Axes): The axes object *Ignored*
            fig(plt.Figure): A valid matplotlib figure on which to plot.
                             If `None`, creates a new figure
            sens_matrix(dict): A dict of the total sensitivity
            labels(list): Strings for labels *Ignored*
            linestyles(list): Strings for linestyles *Ignored*

        Retrun:
            (plt.Figure): The figure
        """
        if sens_matrix is None:
            sens_matrix = self._get_sens()
        elif isinstance(sens_matrix, dict):
            if simid not in sens_matrix:
                raise IndexError('{:} simid not in the sensitity'
                                 'matrix'.format(self.get_inform(1)))
        else:
            raise TypeError('{:} sens matrix must be none or an dict'.
                            format(self.get_inform(1)))
        # end

        if fig is None:
            fig = plt.figure()
        else:
            fig = fig
        # end

        model = self.models[self.opt_key]

        gs = gridspec.GridSpec(3, 4,
                               width_ratios=[6, 1, 6, 1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[4])
        ax4 = fig.add_subplot(gs[6])
        ax5 = fig.add_subplot(gs[8])

        knot_post = model.get_t()

        resp_val = self.simulations[simid]['exp']()[0]

        style = ['-r', '-g', '-b', ':r', ':g', ':b',
                 '--r', '--g', '--b', '--k']
        for i in range(10):
            ax1.plot(sens_matrix[simid][:, i],
                     style[i], label="{:4.3f}".format(knot_post[i]))
        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # ax1.get_legend().set_title("knots",
        #   prop = {'size':rcParams['legend.fontsize']})
        for i in range(10, 20):
            ax2.plot(sens_matrix[simid][:, i],
                     style[i - 10], label="{:4.3f}".format(knot_post[i]))
        ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # ax2.get_legend().set_title("knots",
        #   prop = {'size':rcParams['legend.fontsize']})
        for i in range(20, 30):
            ax3.plot(sens_matrix[simid][:, i],
                     style[i - 20], label="{:4.3f}".format(knot_post[i]))
        ax3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # ax3.get_legend().set_title("knots",
        #   prop = {'size':rcParams['legend.fontsize']})

        for i in range(30, 40):
            ax4.plot(sens_matrix[simid][:, i],
                     style[i - 30], label="{:4.3f}".format(knot_post[i]))
        ax4.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # ax4.get_legend().set_title("knots",
        #   prop = {'size':rcParams['legend.fontsize']})

        for i in range(40, 50):
            ax5.plot(sens_matrix[simid][:, i],
                     style[i - 40], label="{:4.3f}".format(knot_post[i]))
        ax5.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # ax5.get_legend().set_title("knots",
        #   prop = {'size':rcParams['legend.fontsize']})

        ax1.set_ylabel('Sensitivity')
        ax3.set_ylabel('Sensitivity')
        ax5.set_ylabel('Sensitivity')
        ax5.set_xlabel('Model resp. indep. var.')
        ax4.set_xlabel('Model resp. indep. var.')

        # xlocator = (max(resp_val) - min(resp_val)) / 4
        # ax1.xaxis.set_major_locator(MultipleLocator(xlocator))
        # ax2.xaxis.set_major_locator(MultipleLocator(xlocator))
        # ax3.xaxis.set_major_locator(MultipleLocator(xlocator))
        # ax4.xaxis.set_major_locator(MultipleLocator(xlocator))
        # ax5.xaxis.set_major_locator(MultipleLocator(xlocator))

        fig.tight_layout()

        return fig
