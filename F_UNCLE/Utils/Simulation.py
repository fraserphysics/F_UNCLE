#!/usr/bin/pyton
"""

Simulation

Abstract class for simulations

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

import sys
import os
import copy
import warnings

# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv

# =========================
# Custom Packages
# =========================
from .Struc import Struc
from .PhysicsModel import PhysicsModel

try:
    from .mpi_loop import pll_loop
except Exception:
    print("Could not load mpi_loop, Ensure that mpi is"
          " installed on your system")
# end

# =========================
# main Code
# =========================


class Simulation(Struc):
    """Abstract class for experiments

    A child of the Struc class. This abstract class contains methods common to
    all Simulations.

    **Definitions**

    Simulation
        A analysis based on one or more physics
        models which attempts to predict experimental data

    In order for an Experiment to work with the F_UNCLE framework, it must
    implement **all** the inherited methods from `Experiment`.

    **Attributes**

    None

    **Methods**

    """
    def __init__(self, req_models, name='Simulation', *args, **kwargs):
        """Instantiates the object.

        Args:
            req_models(dict): A dictionary in the form {key: type} of the names
                              and types of the mandatory physics models this
                              Experiment uses

        Options can be set by passing them as keyword arguments

        """
        # Name: [type, default, min, max, units, note]
        def_opts = {'sens_mode': [str, 'ser', None, None, '',
                                  'Method of sensitivity calculation, can be'
                                  ' "ser" for serial "mpi" for mpi and'
                                  ' "runjob" for runjob'],
                    'fd_tol': [float, 1E-21, 0.0, 1.0, '',
                               'Minimum threshold for sensitivities'],
                    'fd_step': [float, 2E-2, 0.0, 1.0, '-',
                                'Fraction of DOF value for finite difference'
                                'sensitivity step size']}

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)

        if not isinstance(req_models, dict):
            raise IOError("{:} `req_models` must be a dict".
                          format(self.get_inform(1)))
        elif not all([isinstance(req_models[key], type)
                      for key in req_models]):
            raise IOError("{:} each value in req_models muse be a PhysicsModel")
        elif not all([issubclass(req_models[key], PhysicsModel)
                      for key in req_models]):
            raise IOError("{:} each value in req_models muse be a PhysicsModel")
        else:
            self.req_models = req_models
        # end

    def get_sigma(self, models, *args, **kwargs):
        """Gets the co-variance matrix of the experiment

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (np.ndarray):
                A nxn array of the co-variance matrix of the simulation.
                Where n is the length of the independent variable vector, given
                by :py:meth:`PhysicsModel.shape`
        """

        raise NotImplementedError('{} has not defined a co-variance matrix'
                                  .format(self.get_inform(1)))

    def shape(self, *args, **kwargs):
        """Gives the length of the independent variable vector

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Return:
            (int): The number of independent variables in the experiment
        """

        raise NotImplementedError('{} has no shape specified'
                                  .format(self.get_inform(1)))

    def __call__(self, models=None, *args, **kwargs):
        """Runs the simulation.

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        The simulation instance should be structured so that the necessary
        attributes, options and initial conditions are instantiated before
        calling the object.

        Args:
            *args: Variable length list of models.
            **kwargs: Arbitrary keyword arguments.

        Returns:
           (tuple): length 3 tuple with components

              0. (np.ndarray): The single vector of the simulation
                 independent variable
              1. (list): A list of np.ndarray objects representing the various
                 dependent variables for the problem. Element zero is the
                 most important quantity. By default, comparisons to other
                 data-sets are made to element zero. The length of each
                 element of this list must be equal to the length of the
                 independent variable vector. The last element of this list is
                 the labels for the previous elements
              2. (dict): A dictionary of other attributes of the simulation
                         result. The composition of this dictionary is
                         problem dependent
        """

        return self._on_call(*self.check_models(models), **kwargs)

    def _on_call(self, *models):
        """The overloaded method where the Experiment does its work
        """

        return NotImplemented

    def check_models(self, models):
        """Checks that the models passed to the Experiment are valid
        """

        # Checks that the dictionary models contains all needed models
        if models is None:
            raise TypeError("{:} must provide a dictionary of models"
                            .format(self.get_inform(1)))
        elif not isinstance(models, dict):
            raise IOError("{:} Models must be a dictionary".
                          format(self.get_inform(1)))
        elif not all([key in models for key in self.req_models]):
            raise KeyError("{:} models dict missing a required model".
                           format(self.get_inform(1)))
        elif not all([isinstance(models[key], self.req_models[key])
                      for key in self.req_models]):
            raise IOError("{:} Incorrect model types passed".
                          format(self.get_inform(1)))
        else:
            return self._on_check_models(models)
        # end

    def _on_check_models(self, models):
        return models

    def compare(self, simdata1, simdata2):
        """Compares a the results of two Experiments

        The comparison is made by comparing the results of simdata1
        and simdata2 at each independent variable in simdata2.

        The difference is evaluated as simdata2 **less** simdata1

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
           simdata1(tuple): Complete output of a `__call__` to an `Experiment`
                            object.
           simdata2(tuple): Complete output of a `__call__` to an `Experiment`
                            object.
        Returns:
            (np.ndarray):
                The error between the dependent variables
                and the model for each value of independent variable

        """

        raise NotImplementedError('{} has not compare method instantiated'
                                  .format(self.get_inform(1)))

    def get_pq(self, model, sim_data, experiment, sens_matrix, scale=False):
        """Generates the P and q matrix for the Bayesian analysis

        Args:
           model(PhysicsModel): The model being analyzed
           sim_data(list): Length three list corresponding to the `__call__`
                           from an Experiment object
           experiment(Experiment): A valid Experiment object
           sens_matrix(np.ndarray): The sensitivity matrix

        Return:
            (tuple):
                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """

        return NotImplemented

    def get_log_like(self, model, sim_data, experiment):
        """Gets the log likelihood of the current simulation

        Args:
           model(PhysicsModel): The model under investigation
           sim_data(list): Length three list corresponding to the `__call__`
                           from an Experiment object
           experiment(Experiment): A valid Experiment object

        Return:
            (float): The log of the likelihood of the simulation
        """

        return NotImplemented

    def multi_solve(self, models, opt_keys, dof_list):
        """Returns the results from calling the Experiment object for each
        element of dof_list

        Args:
            models(dict): Dictionary of models
            model_key(list): A list of the model keys used
            dof_list(list): A list of numpy array's defining the model DOFs

        Returns:
            (list): A list of outputs to __call___ for the experiment
                    corresponding to the elements of dof_list
        """

        models = copy.deepcopy(models)

        ret_list = []
        for dof in dof_list:
            idx = 0
            for key in opt_keys:
                shape = models[key].shape()
                models[key] = models[key].update_dof(
                    dof[idx: idx + shape])
                idx += shape
            # end
            ret_list.append(self(models))
        # end

        return ret_list

    def multi_solve_mpi(self, models, model_keys, dof_list, comm=None,
                        verb=False):
        """Returns the results from calling the Experiment object for each
        element of dof_list using mpi

        Args:
            models(dict): Dictionary of models
            model_keys(str): The list of key for the models in the dof list
            dof_list(list): A list of numpy array's defining the model DOFs

        Returns:
            (list): A list of outputs to __call___ for the experiment
                    corresponding to the elements of dof_list
        """

        def eval_fn(x, models, model_keys):
            """The function which is distributed via mpi
            """
            idx = 0
            model_dct = copy.deepcopy(models)
            for key in model_keys:
                shape = models[key].shape()
                try:
                    model_dct[key] = models[key].update_dof(
                        x[idx: idx + shape]
                    )
                except Exception as inst:
                    print("Eror in DOF {:d} for model {:s}"
                          .format(i, key))
                # end

                idx += shape
            # end

            return self(model_dct)
        # end

        ret_dct = pll_loop(dof_list, eval_fn,
                           shape=self.shape(),
                           comm=comm,
                           verb=verb,
                           models=models,
                           model_keys=model_keys)

        out_list = []
        for i in range(len(ret_dct)):
            out_list.append(ret_dct[i])
        # end

        return out_list

    def multi_solve_runjob(self, models, model_key, dof_list):
        """Returns the results from calling the Experiment object for each
        element of dof_list using the runjob script

        .. note::

             runjob is an internal LANL code

        Args:
            models(dict): Dictionary of models
            model_key(str): The key for the model in the models to be used
            dof_list(list): A list of numpy array's defining the model DOFs

        Returns:
            (list): A list of outputs to __call___ for the experiment
                    corresponding to the elements of dof_list
        """

        models = copy.deepcopy(models)

        ret_list = []

        raise NotImplementedError('{:} multi solve using runjob is not'
                                  ' available'
                                  .format(self.get_inform(0)))
        return ret_list

    def get_sens_runjob(self, models, model_key, initial_data=None):
        """Uses runjob for  evaluation of the model gradients

        .. note::

            runjob is an internal LANL code

        Args:
            models(dict): The dictionary of models
            model_key(str): The key of the model for which the sensitivity is
                            desired
        Keyword Args:
            initial_data(np.ndarray): The response for the nominal model DOF, if
                it is `None`, it is calculated when this method is called
            comm(): A MPI communicator

        """

        raise NotImplementedError('{:} multi solve using runjob is not'
                                  ' available'
                                  .format(self.get_inform(0)))

    def get_sens_mpi(self, models, model_keys, initial_data=None, comm=None):
        """MPI evaluation of the model gradients

        Args:
            models(dict): The dictionary of model dof to run
            model_key(str): The list of models for which the sensitivity is
                            desired
        Keyword Args:
            initial_data(np.ndarray): The response for the nominal model DOF,
                if it is `None`, it is calculated when this method is called
            comm(): A MPI communicator

        """
        models = copy.deepcopy(models)

        step_frac = self.get_option('fd_step')

        if initial_data is None:
            initial_data = self(models)
        # end

        ndof = 0
        for key in model_keys:
            ndof += models[key].shape()
        # end

        resp_mat = np.zeros((initial_data[0].shape[0], ndof))
        inp_mat = np.zeros((ndof, ndof))
        new_dof_mat = []

        all_dof = np.zeros((ndof,))
        idx = 0
        for key in model_keys:
            modeli = models[key]
            shape = modeli.shape()
            all_dof[idx: idx + shape] = modeli.get_dof()
            idx += shape
        # end

        idx = 0
        for key in model_keys:
            modeli = models[key]
            new_dofs = np.array(copy.deepcopy(modeli.get_dof()))
            shape = modeli.shape()

            # Get the steps needed for the finite difference sensitivity
            # calculation
            if key in self.req_models:
                for i, coeff in enumerate(new_dofs):
                    step_size = float(coeff * step_frac)
                    new_dofs[i] += step_size
                    inp_mat[idx: idx + shape, idx + i] =\
                        (new_dofs - copy.deepcopy(modeli.get_dof()))
                    all_dof[idx + i] += step_size
                    new_dof_mat.append(copy.deepcopy(all_dof))
                    all_dof[idx + i] -= step_size
                    new_dofs[i] -= step_size
                # end
            else:
                pass
            # end
            idx += shape
        # end

        resp_list = self.multi_solve_mpi(models, model_keys, new_dof_mat,
                                         comm=comm)

        for i, resp in enumerate(resp_list):
            resp_mat[:, i] = -self.compare(resp, initial_data)
        # end

        sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T
        return np.where(np.fabs(sens_matrix) > self.get_option('fd_tol'),
                        sens_matrix,
                        np.zeros(sens_matrix.shape))

    def get_sens(self, models, model_keys, initial_data=None, verb=False):
        """Gets the sensitivity of the experiment response to the model DOF

        Args:
            models(dict): The dictionary of models
            model_keys(list): The list of model keys for which the sensitivity
                              is desired
        Keyword Args:
            initial_data(np.ndarray): The response for the nominal model DOF,
                if it is `None`, it is calculated when this method is called

        """

        models = copy.deepcopy(models)
        step_frac = self.get_option('fd_step')

        if initial_data is None:
            initial_data = self(models)
        # end

        ndof = 0
        for key in model_keys:
            ndof += models[key].shape()
        # end

        resp_mat = np.zeros((initial_data[0].shape[0], ndof))
        inp_mat = np.zeros((ndof, ndof))

        idx = 0
        if verb: print("Getting sens for ", self.name)
        for key in model_keys:
            if verb: print("\tSolving sens for model, ", key)
            modeli = models[key]
            new_dofs = np.array(copy.deepcopy(modeli.get_dof()),
                                dtype=np.float64)
            shape = models[key].shape()
            if key in self.req_models:
                for i, coeff in enumerate(modeli.get_dof()):
                    if verb: print('\t\tGetting sens for dof, ', i)
                    new_dofs[i] += float(coeff * step_frac)
                    models[key] = modeli.update_dof(new_dofs)
                    inp_mat[idx: idx + shape, idx + i] =\
                        (new_dofs - modeli.get_dof())
                    resp_mat[:, idx + i] = -self.compare(
                        self(models),
                        initial_data)
                    new_dofs[i] -= float(coeff * step_frac)
                # end
                models[key] = modeli.update_dof(new_dofs)
            else:
                # If the simulation is not sensitive to the model, keep all
                # entries equal to zero
                pass
            # end
            idx += shape
        # end

        sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T
        return np.where(np.fabs(sens_matrix) > self.get_option('fd_tol'),
                        sens_matrix,
                        np.zeros(sens_matrix.shape))

    def _get_hessian(self, models, model_key, initial_data=None):
        r"""Gets the Hessian (matrix of second derivatives) of the simulated
        experiments to the EOS

        .. math::

            H(f_i) = \begin{smallmatrix}
            \frac{\partial^2 f_i}{\partial \mu_1^2} &
            \frac{\partial^2 f_i}{\partial \mu_1 \partial \mu_2}&
            \ldots & \frac{\partial^2 f_i}{\partial \mu_1 \partial \mu_n}\\

            \frac{\partial^2 f_i}{\partial \mu_2 \partial \mu_1}&
            \frac{\partial^2 f_i}{\partial \mu_2^2}&
            \ldots & \frac{\partial^2 f_i}{\partial \mu_2 \partial \mu_n}\\

            \frac{\partial^2 f_i}{\partial \mu_n \partial \mu_1}&
            \frac{\partial^2 f_i}{\partial \mu_n \partial \mu_2}&
            \ldots & \frac{\partial^2 f_i}{\partial \mu_n^2}

            \end{smallmatrix}

        .. math::

            H(f) = (H(f_1), H(f_2), \ldots , H(f_n))

        where

        .. math::

          f \in \mathcal{R}^m \\
          \mu \in \mathcal{R}^m
        """

        model_dict = copy.deepcopy(models)
        model = model_dict[model_key]

        if initial_data is None:
            initial_data = self(models)
        # end

        fd_step = 2E-2
        hessian = np.zeros((model.shape(),
                            self.shape(),
                            model.shape()))

        initial_sens = self.get_sens(model_dict, model_key)
        initial_dof = model.get_dof()
        for i, dof in enumerate(initial_dof):
            initial_dof[i] += fd_step * dof
            model_dict[model_key] = model.update_dof(
                initial_dof)
            step_sens = self.get_sens(model_dict, model_key)
            hessian[i] = (step_sens - initial_sens) / (fd_step * dof)
            # hessian[i, :, :] = np.linalg.lstsq(delta_mat,
            #                                    (step_sens - initial_sens).T
            #                                    )[0].T
            initial_dof[i] -= fd_step * dof
        # end

        return hessian
