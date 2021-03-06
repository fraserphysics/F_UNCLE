# /usr/bin/pyton
"""

pyBayesian

An object to extract properties of the Bayesian analysis of experiments

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraser (AMF)

Revisions
---------

0 -> Initial class creation (03-16-2016)

TODO
----

- Remove "prior-weight" from code and documentation
- Examine effect of choice of true EOS
- Change name from EOS to more general uncertain function
- Change name "simulations" attribute of Bayesian to "experiment".  Present
  stucture is designed for toy data from simulations.  Future structure should
  be designed for real experimental measurements and simulated toy data is
  special case.
- Change names from Bayesian to indicate limitations, namely constrained MAP
  with quadratic approximation at the maximum
- Enable simultaneous optimization of more than one model

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
import time
import copy
import math
import pdb
import pickle
from collections import OrderedDict
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

from cvxopt import matrix, solvers

# =========================
# Custom Packages
# =========================

from ..Utils.Simulation import Simulation
from ..Utils.Experiment import Experiment
from ..Utils.PhysicsModel import PhysicsModel
from ..Utils.Struc import Struc

try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    if mpi_comm.Get_rank() == 0:
        mpi_print = True
    # end
    else:
        mpi_print = False
    # end
except ImportError as inst:
    mpi_print = True
# end

# Allows the unicode type when running in python3
try:
    unicode
except NameError:
    unicode = str
# end

# =========================
# Main Code
# =========================


class Bayesian(Struc):
    """A class for performing Bayesian inference on a model given data

    **Attributes**

    Attributes:
       simulations(list):  Each element is a tuple with the following elements

           0. A simulation
           1. Experimental results

       model(PhysicsModel): The model under consideration
       sens_matrix(nump.ndarray): The (nxm) sensitivity matrix

           - n model degrees of freedom
           - m total experiment DOF
           - [i,j] sensitivity of model response i to experiment DOF j

    **Options**
    These can be set as keyword arguments in a call to self.__init__

  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |Name          |Type   |Def  |Min  |Max  |Units|Description                 |
  +==============+=======+=====+=====+=====+=====+============================+
  |`outer_atol`  |(float)|1E-6 |0.0  |1.0  |-    |Absolute tolerance on change|
  |              |       |     |     |     |     |in likelihood for outer loop|
  |              |       |     |     |     |     |convergence                 |
  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |`outer_rtol`  |(float)|1E-4 |0.0  |1.0  |-    |Relative tolerance on change|
  |              |       |     |     |     |     |in likelihood for outer loop|
  |              |       |     |     |     |     |convergence                 |
  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |`maxiter`     |(int)  |6    |1    |100  |-    |Maximum iterations for      |
  |              |       |     |     |     |     |convergence of the          |
  |              |       |     |     |     |     |likelihood                  |
  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |`constrain`   |(bool) |True |None |None |-    |Flag to constrain the       |
  |              |       |     |     |     |     |optimization                |
  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |`precondition`|(bool) |True |None |None |-    |Flag to scale the problem   |
  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |`debug`       |(bool) |False|None |None |-    |Flag to print debug         |
  |              |       |     |     |     |     |information                 |
  +--------------+-------+-----+-----+-----+-----+----------------------------+
  |`verb`        |(bool) |True |None |None |-    |Flag to print stats during  |
  |              |       |     |     |     |     |optimization                |
  +--------------+-------+-----+-----+-----+-----+----------------------------+

    .. note::

       The options `outer_atol` and `prior_weight` are deprecated and should be
       used for debugging purposes only

    **Methods**
    """

    def __init__(self, simulations, models, opt_keys=None, name='Bayesian',
                 *args, **kwargs):
        """Instantiates the Bayesian analysis

        Args:
           simulations(dict): A dictionary of simulation, experiment pairs
           models(dict): A dictionary of models
           opt_keys(list): The list of models to be optimized
        Keyword Args:
           name(str): Name for the analysis.('Bayesian')

        Return:
           None

        """

        # Name: [type, default, min, max, units, note]
        def_opts = {
            'outer_atol': [float, 1E-6, 0.0, 1.0, '-',
                           'Absolute tolerance on change in likelihood for'
                           'outer loop convergence'],
            'outer_rtol': [float, 1E-4, 0.0, 1.0, '-',
                           'Relative tolerance on change in likelihood for'
                           'outer loop convergence'],
            'maxiter': [int, 6, 1, 100, '-',
                        'Maximum iterations for convergence'
                        'of the likelihood'],
            'constrain': [bool, True, None, None, '-',
                          'Flag to constrain the optimization'],
            'precondition': [bool, True, None, None, '-',
                             'Flag to scale the problem'],
            'pickle_sens': [bool, False, None, None, '-',
                            'Flag to save sens matrix each iteration'],
            'debug': [bool, False, None, None, '-',
                      'Flag to print debug information'],
            'verb': [bool, True, None, None, '-',
                     'Flag to print stats during optimization'],
            # 'mpi_comm': [type(mpi_comm), None, None, None, '-',
            #               'MPI commuicator for sensitivities']
        }

        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)

        self.models, self.simulations = self._check_inputs(models, simulations)

        if opt_keys is None and len(self.models) > 1:
            raise IOError("{:} Must define opt key if more than one model"
                          "used".format(self.get_inform(1)))
        elif isinstance(opt_keys, (unicode, str)):
            opt_keys = [opt_keys]
        elif opt_keys is None:
            opt_keys = [list(self.models.keys())[0]]
        # end

        for key in opt_keys:
            if key not in self.models:
                raise KeyError("{:} Opt key {:} is not in models dict"
                               .format(self.get_inform(1), key))
            # end
        # end
        self.opt_keys = opt_keys
    # end

    def _check_inputs(self, models, simulations):
        """Checks that the values for model and simulation are valid
        """
        if not isinstance(models, (dict, OrderedDict)):
            raise TypeError('001 {:}, model must be provided as a dictionary'
                            'with the key being the name of the model'
                            .format(self.get_inform(1)))
        elif len(models) > 1 and not isinstance(models, OrderedDict):
            raise TypeError('001b {:}, model must be provided as an ordered'
                            ' dictionary when there is more than 1 model'
                            .format(self.get_inform(1)))
        elif not np.all([isinstance(models[key], PhysicsModel)
                         for key in models]):
            raise TypeError('002 {:}, all models must be a PhysicsModel type'
                            .format(self.get_inform(1)))
        elif not isinstance(simulations, dict):
            raise TypeError('003 {:}, simulations must be provided as a'
                            'dictionary with the key being the name of the'
                            'simulation'
                            .format(self.get_inform(1)))
        # end

        if np.all([isinstance(simulations[key], (list, tuple))
                   for key in simulations]):
            if not np.all([len(simulations[key]) == 2 for key in simulations]):
                raise TypeError('005 {:}, each list for simulation must be'
                                'length 2'
                                .format(self.get_inform(1)))
            elif not np.all([isinstance(simulations[key][0],
                                        (Simulation))
                             for key in simulations]):
                raise TypeError('006A {:}, each sim in the simulation list'
                                'must be an Simulation type'
                                .format(self.get_inform(1)))
            elif not np.all([isinstance(simulations[key][1],
                                        (Experiment))
                             for key in simulations]):
                raise TypeError('006B {:}, each experiemnt in the simulation'
                                ' list must be a Experiment type'
                                .format(self.get_inform(1)))
            else:
                for key in simulations:
                    simulations[key] = {
                        'sim': copy.deepcopy(simulations[key][0]),
                        'exp': copy.deepcopy(simulations[key][1])
                    }
                # end
                sim_out = copy.deepcopy(simulations)
            # end
        elif np.all([isinstance(simulations[key], dict)
                     for key in simulations]):
            if not np.all(['sim' in simulations[key] and
                           'exp' in simulations[key]
                          for key in simulations]):
                raise TypeError('007 {:}Each dictionary must contain the keys'
                                'sim and exp'.format(self.get_inform(1)))
            elif not np.all([isinstance(simulations[key]['sim'],
                                        Simulation)
                             for key in simulations]):
                raise TypeError('008A {:}, each sim in the simulation dict'
                                'must be an experiment type'
                                .format(self.get_inform(1)))
            elif not np.all([isinstance(simulations[key]['exp'],
                                        Experiment)
                             for key in simulations]):
                raise TypeError('008B {:}, each exp in the simulation dict'
                                'must be a Experiment type'
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
        """Print method for Bayesian model called by Struct.__str__

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
        """Updates the properties of the Bayesian analysis

        Keyword Args:
           simulations(dict): Dictionary of simulation experiment pairs
                                    (Default None)
           models(dict): Dictionary of models
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
        """Gets the dimensions of the problem

        Return:
           (tuple): The (n, m) dimensions of the problem

               - n is the total degrees of freedom of all the model responses
               - m is the degrees of freedom of the model
        """
        dof_model = 0

        for key in self.opt_keys:
            dof_model += self.models[key].shape()
        # end

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
        lk = 0.0

        for key in self.opt_keys:
            lki = self.models[key].get_log_like()
            lk += lki
            if mpi_print and self.get_option('verb'):
                print("Log Like for {:s} is {:f}".format(key, lki))
        # end
        return lk

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
            lk = sims[key]['exp'].\
                get_log_like(initial_data[key])
            if mpi_print and self.get_option('verb'):
                print("Log Like for {:s} is {:f}".format(key, lk))
            log_like += lk
        # end

        return log_like

    def get_opt_dof(self):
        """Returns a vector of the model DOF for the optimization

        """

        dof_vec = np.empty((self.shape()[1],))

        idx = 0
        for key in self.opt_keys:
            shape = self.models[key].shape()
            dof_vec[idx: idx + shape] = self.models[key].get_dof()
            idx += shape
        # end

        return dof_vec

    def __call__(self, comm=None):
        """Determines the best candidate EOS function for the models

        Args:
            comm(MPI.intracom): An MPI communicator for sensitities

        Return:
           (tuple): length 3, elements are:

               0. (Bayesian): A copy of self with the optimal models
               1. (list): is of solution history elements are:

                   0. (np.ndarray) Log likelihood, (nx1) where n is number of
                      iterations
                   1. (np.ndarray) model dof history (nxm) where n is iterations
                      and m is the model dofs

               2. (dict) sensitivity matrix of all experiments to
                   the model, keys are simulation keys
               3. (dict) fisher information matrix for all experiments
                   keys are simulation keys

        """

        def outer_loop_iteration(analysis, log_like, intitial_data, itn):
            """Performs a single step of the outer loop optimization

            Args:
                analysis(Bayesian): The collection of simulations and
                                    experiments to be tested
                log_like(float): The log likelihood of the last iteration
                initial_data(list): The results of the simulations at the
                                    current model state
                i(int): The iteration number
            Return:
                (Bayesian): A copy of the analysis object with the new model
                            estimate
                (float): The log likelihood of the updated model
                (list): The results of the simulations at the new model state
                (list): A list of the model DOFs
                (bool): A flag, true of the log likelihood has converged
            """
            precondition = analysis.get_option('precondition')
            atol = analysis.get_option('outer_atol')
            reltol = analysis.get_option('outer_rtol')
            verb = analysis.get_option('verb')
            opt_keys = analysis.opt_keys
            model_dict = analysis.models

            # Solve all simulations with the current model
            if verb and mpi_print:
                print('model log like', analysis.model_log_like())
            if verb and mpi_print:
                print('sim log like', analysis.sim_log_like(initial_data))

            new_log_like = analysis.model_log_like()\
                + analysis.sim_log_like(initial_data)

            # Calculate the current likelyhood and determine if the loop
            # should exit
            if verb and mpi_print:
                print('total log like', new_log_like)
                print('error', np.fabs((log_like - new_log_like)
                                       / new_log_like))
            if np.fabs((log_like - new_log_like) / new_log_like) < reltol:
                return copy.deepcopy(analysis),\
                    new_log_like,\
                    initial_data,\
                    analysis.get_opt_dof(),\
                    True

            if verb and mpi_print:
                print('Begining sensitivities calculation')
            sens_matrix = analysis._get_sens(initial_data, comm=comm)
            if verb and mpi_print:
                print('End of sensitivities calculation')

            if analysis.get_option('pickle_sens'):
                with open("./sens_iter{:02d}.pkl".format(itn), 'wb') as fid:
                    pickle.dump(sens_matrix, fid)
            # end

            if verb and mpi_print:
                print('Begining local optimization')

            local_sol = analysis._local_opt(initial_data,
                                            sens_matrix)
            d_hat = np.array(local_sol['x']).reshape(-1)

            if precondition:
                scale_vect = np.zeros(2 * (analysis.shape()[1],))
                idx = 0
                for key in self.opt_keys:
                    shape = model_dict[key].shape()
                    scale_vect[idx: idx + shape, idx: idx + shape] =\
                        model_dict[key].get_scaling()
                    idx += shape
                # end
                d_hat = np.dot(scale_vect, d_hat)
            # end

            if verb and mpi_print:
                print('End of local optimization')
                print('Start of line search')

            # Finds the optimal step in the d_hat direction
            n_steps = 7
            costs = np.zeros(n_steps)
            iter_data = []
            initial_dof = copy.deepcopy(analysis.get_opt_dof())
            max_step = 1.0
            x_list = np.linspace(0, max_step, n_steps)
            dof_list = []
            analysis_list = []

            # Builds the list of dof values to test and gets the prior cost
            plot_lsearch = self.get_option('debug') * mpi_print
            if plot_lsearch:
                fig = plt.figure()
                search_ax = {}
                for i, key in enumerate(opt_keys):
                    search_ax[key] = fig.add_subplot(1, len(opt_keys), i + 1)
                    model_dict[key].prior.plot(
                        axes=search_ax[key],
                        labels=['Prior'],
                        linestyles=['-'],
                        vrange=(0.1, 0.8),
                        log=False)
            # end

            for i, x_i in enumerate(x_list):
                dof_list.append(initial_dof + x_i * d_hat)
                idx = 0

                for key in opt_keys:
                    shape = model_dict[key].shape()
                    model_dict[key] = model_dict[key].update_dof(
                        dof_list[-1][idx: idx + shape])
                    # print(model_dict[key])
                    idx += shape
                # end

                if plot_lsearch:
                    for key in opt_keys:
                        model_dict[key].plot(
                            axes=search_ax[key],
                            labels=['Step {:02d}'.format(i)],
                            linestyles=['-'],
                            vrange=(0.1, 0.8),
                            log=False
                        )
                    fig.savefig('lineSearch_itn{:02d}.pdf'.format(itn))
                    plt.close(fig)
                analysis_list.append(analysis.update(models=model_dict))
                costs[i] = analysis_list[-1].model_log_like()
            # end

            if plot_lsearch:
                for ax in search_ax:
                    search_ax[ax].legend(loc='best')
                # end
                fig.savefig('lineSearch_itn{:02d}.pdf'.format(itn))
            # end

            # Evaluates each simulation at each dof step
            # Note: This approach makes use of the multi_solve method
            #       which helps accelerate the solution when using a
            #       mpi or run_job.
            iter_data = {}
            for key in analysis.simulations:
                sim_i = analysis.simulations[key]['sim']
                if sim_i.get_option('sens_mode') == 'mpi':
                    sim_data = sim_i.multi_solve_mpi(
                        model_dict, opt_keys, dof_list)
                elif sim_i.get_option('sens_mode') == 'runjob':
                    sim_data = sim_i.multi_solve_runjob(
                        model_dict, opt_keys, dof_list)
                else:
                    sim_data = sim_i.multi_solve(
                        model_dict, opt_keys, dof_list)
                # end
                # Evaluate the simulation data to the same times as the exp
                for i, data in enumerate(sim_data):
                    sim_data[i] = analysis.simulations[key]['exp'].align(
                        data
                    )
                # end
                iter_data[key] = sim_data
            # end

            # Calculates the cost of the data for each step
            sorted_data = []
            for i in range(len(x_list)):
                if verb and mpi_print:
                    print('--Step {:d}--'.format(i))
                    print("Log Like for models is {:f}".format(costs[i]))
                sorted_data.append({})
                for key in analysis.simulations:
                    sorted_data[-1][key] = iter_data[key][i]
                # end
                costs[i] += analysis_list[i].sim_log_like(sorted_data[-1])
            # end

            if self.get_option('debug'):
                for key in analysis.simulations:
                    try:
                        fig = plt.figure()
                        ax9 = fig.add_subplot(121)
                        ax10 = fig.add_subplot(122)
                        exp_data = analysis.simulations[key]['exp']()
                        ax9.plot(exp_data[0], exp_data[1][0], '*', ms=3,
                                 label='Experiment')
                        sim_times = np.linspace(
                            exp_data[0].min(),
                            exp_data[0].max(),
                            200)
                        for i, data in enumerate(iter_data[key]):
                            epsilon = analysis.simulations[key]['exp']\
                                .compare(data)
                            sigma = analysis.simulations[key]['exp']\
                                .get_sigma()
                            ax9.plot(sim_times,
                                     data[2]['mean_fn'](
                                         sim_times - data[2]['tau']),
                                     label="step %02d" % i)

                            ax10.plot(exp_data[0],
                                      0.5 * np.dot(epsilon**2,
                                                   np.linalg.inv(sigma)),
                                      '*', ms=3,
                                      label="sigma %02d" % i)
                        # end
                        ax9.legend(loc="best")
                        ax10.legend(loc="best")
                        fig.savefig("{:}-itn{:02d}_search_res.pdf"
                                    .format(key, itn))
                        plt.close(fig)
                    except Exception:
                        pass
                    # end
                # end

            # end

            # Updates the model dof to the optimal value
            besti = np.argmax(costs)

            if verb and mpi_print:
                print('End of line search\n'
                      'Costs {:s}\n'
                      'Besti {:d}\n'
                      'End of iteration {:02d}'.format(str(costs), besti, itn))

            return (analysis_list[besti],
                    new_log_like,  # costs[besti],
                    sorted_data[besti],
                    analysis.get_opt_dof(),
                    False)

        # end

        maxiter = self.get_option('maxiter')
        verb = self.get_option('verb')
        history = []
        dof_hist = []
        data_hist = []
        initial_data = self.get_data()

        conv = False
        analysis = copy.deepcopy(self)
        log_like = 0.0
        i = 0

        if self.get_option('pickle_sens'):
            exp_data = {}
            for key in self.simulations:
                exp_data[key] = self.simulations[key]['exp']()
            # end

            with open('exp_data.pkl', 'wb') as fid:
                pickle.dump(exp_data, fid)
            # end

            with open('prior_data.pkl', 'wb') as fid:
                pickle.dump(initial_data, fid)
            # end

            with open('prior_model.pkl', 'wb') as fid:
                pickle.dump(analysis.models, fid)
            # end

        while not conv and i < maxiter:
            if verb and mpi_print:
                print('Iter {:d} of {:d}'.format(i, maxiter))
            # end

            analysis, log_like, initial_data, model_dof, conv =\
                outer_loop_iteration(analysis, log_like, initial_data, i)
            history.append(log_like)
            dof_hist.append(model_dof)
            data_hist.append(initial_data)
            if self.get_option('pickle_sens'):
                with open('models_iter{:02d}.pkl'.format(i), 'wb') as fid:
                    pickle.dump(analysis.models, fid)
                # end

                with open('sim_data_iter{:02d}.pkl'.format(i), 'wb') as fid:
                    pickle.dump(initial_data, fid)
                # end
            # end
            i += 1
        # end

        sens_matrix = analysis._get_sens(initial_data, comm=comm)

        if not conv and mpi_print:
            print("{}: Outer loop could not converge to the given"
                  "tolerance in the maximum number of iterations"
                  .format(self.get_inform(1)))
        # end

        # Make the fisher information matrix
        fisher_data = {}
        # Get fisher info for everyone
        fisher_all = np.empty((analysis.shape()[1], analysis.shape()[1]))
        for key in analysis.simulations:
            fisher = analysis.simulations[key]['exp']\
                             .get_fisher_matrix(sens_matrix[key])
            fisher_all += fisher
            fisher_data[key] = fisher
        # end
        fisher_data['All'] = fisher_all
        if self.get_option('pickle_sens'):
            with open('fisher_matrix.pkl', 'wb') as fid:
                pickle.dump(fisher_data, fid)
            # end
        # end

        dof_hist = np.array(dof_hist)
        history = np.array(history)

        return analysis, (history, dof_hist, data_hist), sens_matrix,\
            fisher_data

    @staticmethod
    def fisher_decomposition(fisher_dct, simid, models, mkey, tol=1E-3):
        """Calculate a spectral decomposition of the fisher information matrix

        Args:
            fisher_dct(dict): A dictionary nxn array where n is sum of all
                              model dof
            simid(str): Key for the simulation you want the Fisher info for
            models(dict): A dictionary of all models
            mkey(str): The key for the model on which the analysis is being
                       performed

        Keyword Args:
            tol(float): Eigenvalues less than tol are ignored

        Return:
            (tuple): Elements are:

                0. (list): Eigenvalues greater than tol
                1. (np.ndarray): nxm array.

                      - n is number of eigenvalues greater than tol
                      - m is model dof

                2. (np.ndarray): nxm array

                      - n is the number of eigenvalues greater than tol
                      - m is an arbitrary dimension of independent variable

                3. (np.ndarray): vector of independent variable

        """
        if simid not in fisher_dct:
            raise IndexError('simid not in the fisher information'
                             ' matrix dictionary')

        if mkey not in models:
            raise IndexError('mkey not in the models'
                             ' dictionary')

        fisher = fisher_dct[simid]
        idx = 0
        for key in models:
            shape = models[key].shape()
            if key == mkey:
                break
            else:
                idx += shape
            # end
        # end
        model = models[mkey]

        # Spectral decomposition of info matrix and sort by eigenvalues
        eig_vals, eig_vecs = np.linalg.eigh(
            fisher[idx: idx + shape, idx: idx + shape]
        )
        eig_vals = np.maximum(eig_vals, 0)        # info is positive definite

        i = np.argsort(eig_vals)[-1::-1]
        vals = eig_vals[i]
        vecs = eig_vecs.T[i]

        n_vals = max(len(np.where(vals > vals[0] * tol)[0]), 3)
        n_vecs = max(len(np.where(vals > vals[0] * tol)[0]), 3)

        # Find range of v that includes support of eigenfunctions
        # knots = self.models[self.opt_key].get_t()
        knots = model.get_t()
        v_min = knots[0]
        v_max = knots[-1]
        vol = np.logspace(np.log10(v_min), np.log10(v_max), len(knots) * 100)
        funcs = []
        for eig, vec in zip(vals[:n_vecs], vecs[:n_vecs]):
            new_model = model.\
                set_c(vec * model.get_dof())
            funcs.append(new_model(vol))
            if funcs[-1][np.argmax(np.fabs(funcs[-1]))] < 0:
                funcs[-1] *= -1
            # end
            funcs[-1] = np.array(funcs[-1]) * 1 / max(funcs[-1])
        funcs = np.array(funcs)

        return vals[:n_vals], vecs, funcs, vol

    def _local_opt(self, initial_data, sens_matrix):
        """Solves the quadratic problem for maximization of the log likelihood

        Args:
           initial_data(list): The initial data corresponding to simulations
               from sims
           sens_matrix(np.array): The sensitivity matrix about the model

        Return:
            (np.ndarray):
                The step direction for greatest improvement in log likelihood
        """
        constrain = self.get_option('constrain')
        debug = self.get_option('debug')
        precondition = self.get_option('precondition')

        # Get constraints
        g_mat, h_vec = self._get_constraints()

        p_mat, q_vec = self._get_model_pq()

        tmp = self._get_sim_pq(initial_data, sens_matrix)

        p_mat += tmp[0]
        q_vec += tmp[1]

        n_dof = self.shape()[1]

        # If the convex approximateion is too smooth, you have reached an
        # optimum
        # if np.linalg.matrix_rank(p_mat,
        #                          tol=np.finfo(np.float64).eps) < n_dof\
        #    and np.linalg.matrix_rank(p_mat, tol=1E-21) == n_dof:
        #     return {'x':np.zeros((n_dof,))}

        # >>>>IMPORTANT<<<<
        # The formulation shows a factor of 1/2 applied to the P matrix but
        # CVXOPT *already* applies this factor to the P matrix, so the
        # following line, if uncommneted, does it twice. DO NOT DO THIS
        # p_mat *= 0.5 # DO NOT UNCOMMENT THIS LINE
        # >>>>IMPORTANT<<<<

        solvers.options['show_progress'] = (self.get_option('verb')
                                            and mpi_print)
        solvers.options['debug'] = False
        solvers.options['maxiters'] = 100  # 100 default
        solvers.options['reltol'] = 1e-6   # 1e-6 default
        solvers.options['abstol'] = 1e-7   # 1e-7 default
        solvers.options['feastol'] = 1e-7  # 1e-7 default

        # >>Supressed plotting routine to show the constraint matrix
        # fig = plt.figure()
        # ax9 = fig.add_subplot(121)
        # ax8 = fig.add_subplot(122)
        # opt_model = self.models[self.opt_key]
        # n_end = opt_model.get_option('spline_end')
        # rho_unique = opt_model.get_t()[n_end -1 : 1 - n_end]
        # n_rho = rho_unique.shape[0]
        # for i in xrange(opt_model.shape()):
        #     ax9.plot(rho_unique, g_mat[:n_rho, i])
        #     ax8.plot(rho_unique, g_mat[n_rho:2*n_rho, i])
        #     ax9.plot(rho_unique, g_mat[2 * n_rho:, i])
        # fig.savefig('gmat{:f}.pdf'.format(time.time()))
        # >>Supressed plotting routine to show the constraint matrix

        try:
            # return{'x':np.dot(np.linalg.inv(p_mat),-q_vec)}
            if constrain:
                sol = solvers.qp(matrix(p_mat), matrix(q_vec),
                                 matrix(g_mat), matrix(h_vec),
                                 intervals={'x': np.zeros(g_mat.shape[1])})
            else:
                sol = solvers.qp(matrix(p_mat), matrix(q_vec))
            # end
        except ValueError as inst:
            print(inst)
            print("G " + str(g_mat.shape))
            print("P " + str(p_mat.shape))
            print("h " + str(h_vec.shape))
            print("q " + str(q_vec.shape))
            pdb.post_mortem()

        # >>Supressed plotting routine to show the constraints and their values
        #   at the optimal point
#         fig = plt.figure()
#         ax2 = fig.add_subplot(111)
#         opt_model = self.models[self.opt_key]
#         rho_unique = opt_model.get_t()[3:-3]
#         n_unique = rho_unique.shape[0]
#         d_hat = np.array(sol['x']).reshape(-1)
# #        d_hat = np.ones(opt_model.shape())
#         ax2.plot(rho_unique,
#                  np.dot(g_mat, d_hat)[:n_unique],
#                  '-o',
#                  label='Volume Convexity value')
#         ax2.plot(rho_unique,
#                  np.dot(g_mat, d_hat)[n_unique: 2 * n_unique],
#                  '-o',
#                  label='Positivity value')
#         ax2.plot(rho_unique,
#                  np.dot(g_mat, d_hat)[2 * n_unique:],
#                  '-o',
#                  label='Density Convecity value')
#         ax2.plot(rho_unique,
#                  h_vec[:n_unique],
#                  '--o',
#                  label='Convexity limit')
#         ax2.plot(rho_unique,
#                  h_vec[n_unique:2*n_unique],
#                  '--o',
#                  label='Positivity limit')
#         ax2.legend(loc='best')
#         fig.savefig('function_constraints{:f}.pdf'.format(time.time()))

        if sol['status'] != 'optimal':
            for key, value in sol.items():
                print(key, value)
            raise RuntimeError('{} The optimization algorithm could not locate'
                               ' an optimal point'.format(self.get_inform(1)))
        return sol

    def _get_constraints(self):
        r"""Get the constraints on the model
        """
        gmat_list = []
        hvec_list = []
        ncon = 0
        for key in self.opt_keys:
            g_tmp, h_tmp = self.models[key].get_constraints(
                scale=self.get_option('precondition'))
            ncon += h_tmp.shape[0]
            gmat_list.append(g_tmp)
            hvec_list.append(h_tmp)
        # end

        gmat = np.zeros((ncon, self.shape()[1]))
        hvec = np.zeros((ncon,))

        idx_dof = 0
        idx_con = 0
        for i in range(len(hvec_list)):
            shape_con, shape_dof = gmat_list[i].shape
            gmat[idx_con: idx_con + shape_con,
                 idx_dof: idx_dof + shape_dof] = gmat_list[i]
            hvec[idx_con: idx_con + shape_con] = hvec_list[i]
            idx_dof += shape_dof
            idx_con += shape_con
        # end

        return gmat, hvec

    def _get_model_pq(self):
        """Gets the quadratic optimization matrix contributions from the prior

        Args:
            None
        Return:
            (tuple): elements are

                0. (np.ndarray): `p`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF
        """

        pmat = np.zeros(2 * (self.shape()[1],), np.float64)
        qvec = np.zeros((self.shape()[1],), np.float64)
        idx = 0
        for key in self.opt_keys:
            p_tmp, q_tmp = self.models[key].get_pq(
                scale=self.get_option('precondition'))
            shape = self.models[key].shape()
            pmat[idx: idx + shape, idx: idx + shape] = p_tmp
            qvec[idx: idx + shape] = q_tmp
            idx += shape
        # end

        return pmat, qvec

    def _get_sim_pq(self, initial_data, sens_matrix):
        """Gets the QP contributions from the model

        .. note::

             This method is specific to the model type under consideration.
             This implementation is only for spline models of EOS

        Args:
           initial_data(list): A list of the initial results from the
                               simulations in the same order as in the
                               `sim` list
           sens_matrix(np.ndarray): The sensitivity matrix

        Return:
            (tuple): Elements are:

                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """
        debug = False  # self.get_option('debug')
        sims = self.simulations

        p_mat = np.zeros((self.shape()[1], self.shape()[1]))
        q_mat = np.zeros(self.shape()[1])

        i = 0

        # Plot commands for q matrix debug

        if debug:
            # Debugging creates plots of the P and Q matricies
            fig = plt.figure()
            ax1 = fig.gca()
            fig2 = plt.figure()
            ax2 = fig2.gca()

            knots = self.models[self.opt_key].get_t()[
                :-self.models[self.opt_key].get_option('spline_end')]
        for key in sims:
            p_tmp, q_tmp = sims[key]['exp'].get_pq(
                self.models,
                self.opt_keys,
                initial_data[key],
                sens_matrix[key],
                scale=self.get_option('precondition'))
            p_mat += p_tmp
            q_mat += q_tmp

            if debug:
                ax1.plot(knots, q_tmp, label=key)
                for i in range(p_tmp.shape[0]):
                    ax2.plot(knots, p_tmp[i, :])
        # end

        if debug:
            ax1.legend(loc='best')
            fig.savefig('q_vec.pdf')
            fig2.savefig('P_mat.pdf')
            plt.close(fig)
            plt.close(fig2)
        return p_mat, q_mat

    def get_data(self):
        """Generates a set of data at each experimental data-point

        For each pair of simulations and experiments, generates the
        the simulation response, aligning it with the experimental
        data

        Args:
            None
        Return:
            (list):
                List of lists for experiment comparison data

                0. independent value
                1. dependent value of interest
                2. the summary data (3rd element) of sim

        """
        sims = self.simulations

        data = {}
        for key in sims:
            data[key] = sims[key]['exp'].align(
                sims[key]['sim'](copy.deepcopy(self.models))
            )
        # end

        return data

    def _get_sens(self, initial_data=None, comm=None):
        """Gets the sensitivity of the simulated experiment to the EOS

        The sensitivity matrix is the attribute `self.sens_matrix` which is set
        by this method

        .. note::

             This method is specific to the model type under consideration.
             This implementation is only for spline models of EOS

        Args:
            None

        Keyword Args:
            initial_data(list): The results of each simulation with the current
                best model. Each element in the list corresponds tho the output
                from a `__call__` to each element in the `self.simulations` list
           comm(MPI.Intracom): An MPI communicator for mpi sensitivities

        Return:
            None
        """
        sims = self.simulations
        models = self.models
        opt_keys = self.opt_keys

        if initial_data is None:
            initial_data = self.get_data()
        # end

        t0 = time.time()
        sens_matrix = {}
        for key in self.simulations:
            sim_i = sims[key]['sim']
            t1 = time.time()
            if self.get_option('verb') and mpi_print:
                print('Getting sens for {:} using {:}'.format(
                    key,
                    sim_i.get_option('sens_mode')))
            # end

            if sim_i.get_option('sens_mode') == 'pll':
                raise NotImplementedError('Parallel sesnitivity not yet'
                                          ' support multiple models')
                sens_matrix[key] = sim_i.get_sens_pll(
                    models, opt_keys, initial_data[key])
            elif sim_i.get_option('sens_mode') == 'mpi':
                sens_matrix[key] = sim_i.get_sens_mpi(
                    models, opt_keys, initial_data[key],
                    comm=comm)
            elif sim_i.get_option('sens_mode') == 'runjob':
                # raise NotImplementedError('runjob sesnitivity not yet support'
                #                           ' multiple models')
                sens_matrix[key] = sim_i.get_sens_runjob(
                    models, opt_keys, initial_data[key])
            else:
                sens_matrix[key] = sim_i.get_sens(
                    models, opt_keys, initial_data[key])

            if self.get_option('verb') and mpi_print:
                print('Sensitivity calculation for {:} using {:} took {:4.3f}s'
                      .format(
                          key,
                          sim_i.get_option('sens_mode'),
                          time.time() - t1))

        if self.get_option('verb') and mpi_print:
            print('Total sensitivity calculation took {:4.3f}s'
                  .format(
                      time.time() - t0))
        # end

        return sens_matrix

    def get_hessian(self, initial_data=None, simid=None):
        r"""Gets the Hessian (matrix of second derivatives) of the simulated
        experiments to the EOS

        .. math::

            H(f_i) = \begin{smallmatrix}
            \frac{\partial^2 f_i}{\partial \mu_1^2}
                & \frac{\partial^2 f_i}{\partial \mu_1 \partial \mu_2}
                & \ldots
                & \frac{\partial^2 f_i}{\partial \mu_1 \partial \mu_n}\\
            \frac{\partial^2 f_i}{\partial \mu_2 \partial \mu_1}
                & \frac{\partial^2 f_i}{\partial \mu_2^2}
                & \ldots
                & \frac{\partial^2 f_i}{\partial \mu_2 \partial \mu_n}\\
            \frac{\partial^2 f_i}{\partial \mu_n \partial \mu_1}
                & \frac{\partial^2 f_i}{\partial \mu_n \partial \mu_2}
                & \ldots
                & \frac{\partial^2 f_i}{\partial \mu_n^2}
            \end{smallmatrix}

        .. math::

          H(f) = (H(f_1), H(f_2), \ldots , H(f_n))

        where

        .. math::

          f \in \mathcal{R}^m \\
          \mu \in \mathcal{R}^m
        """

        raise NotImplementedError('The Get Hessian method does not yet support'
                                  ' multiple models')
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
            fig = plt.Figure()
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
        ax1.set_ylabel('Negative a posteriori log likelihood')

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
