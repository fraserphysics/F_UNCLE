#!/usr/bin/pyton
"""

pyExperiment

Abstract class for experiments, both physical and computational

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
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Struc import Struc
    from F_UNCLE.Utils.PhysicsModel import PhysicsModel
    from F_UNCLE.Utils.mpi_loop import pll_loop
else:
    from .Struc import Struc
    from .PhysicsModel import PhysicsModel
    from .mpi_loop import pll_loop
# end

# =========================
# Main Code
# =========================


class Experiment(Struc):
    """Abstract class for experiments

    A child of the Struc class. This abstract class contains methods common to
    all Experiment  objects. This class can be used to model two different cases

    **Definitions**

    Simulation
        Makes use of a single model or set of models internal to the object to
        simulate some physical process

    Experiment
        Can be of two types

        1. A "computational experiment" where a simulation is performed using a
           nominal *true* model
        2. A representation of a real experiment using tabulated values

    In order for an Experiment to work with the F_UNCLE framework, it must
    implement **all** the inherited methods from `Experiment`, regardless if
    it is a Simulation or Experiment

    **Attributes**

    None

    **Methods**
    """
    def __init__(self, req_models, name='Experiment', model_attribute=None,
                 *args, **kwargs):
        """Instantiates the object.

        Options can be set by passing them as keyword arguments

        """

        def_opts = {}

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

        self.model_attribute = model_attribute

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

    def __call__(self, models=None, **kwargs):
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
                 independent variable vector.
              2. (list): A list of other attributes of the simulation result.
                 The composition of this list is problem dependent
        """

        return self._on_call(*self.check_models(models))

    def _on_call(self, *models):
        """The overloaded method where the Experiment does its work
        """

        return NotImplemented

    def check_models(self, models):
        """Checks that the models passed to the Experiment are valid
        """

        # In F_UNCLE some experiments have hard coded models to represent real
        # experiments. This statement ignores the passed models and uses the
        # built in model. This behavior is not physically meaningful so a
        # warning is raised

        if self.model_attribute is not None:
            warnings.warn("{:} using the model attribute rather than a passed"
                          " model".format(self.get_warn(0)), UserWarning)
            if isinstance(self.model_attribute, PhysicsModel):
                return (self.model_attribute,)
            elif isinstance(self.model_attribute, (list, tuple)):
                return self.model_attribute

        # Checks that the dictionary models contains all needed models
        if models is None:
            raise TypeError("{:} if `model_attribute` is not set, must provide"
                            "a dict of models".format(self.get_inform(1)))
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

    def compare(self, indep, dep, model_data):
        """Compares a set of experimental data to the model

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
           indep(list): The list of independent variables for comparison
           dep(list): The list or array of dependent variables for comparison
           model_data(tuple): Complete output of a `__call__` to an `Experiment`
                              object  which `dep` is compared to at every point
                              in `indep`
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
           model(PhysicsModel): The model being analysed
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
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
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           experiment(Experiment): A valid Experiment object

        Return:
            (float): The log of the likelihood of the simulation
        """

        return NotImplemented

    def get_sens_mpi(self, models, model_key, initial_data=None, comm=None):
        """MPI evaluation of the model gradients

        Args:
            models(dict): The dictionary of models
            model_key(str): The key of the model for which the sensitivity is
                            desired
        Keyword Args:
            initial_data(np.ndarray): The response for the nominal model DOF, if
                it is `None`, it is calculated when this method is called
            comm(): A MPI communicator

        """
        models = copy.deepcopy(models)
        model = models[model_key]

        step_frac = 2E-2

        if initial_data is None:
            initial_data = self(models)
        # end

        resp_mat = np.zeros((initial_data[0].shape[0],
                             model.shape()))
        inp_mat = np.zeros((model.shape(),
                            model.shape()))
        new_dof_mat = []
        new_dofs = np.array(copy.deepcopy(model.get_dof()),
                            dtype=np.float64)
        # Build the input matrix and a lift of dof's to test
        for i, coeff in enumerate(model.get_dof()):
            new_dofs[i] += float(coeff * step_frac)
            inp_mat[:, i] = (new_dofs - model.get_dof())
            new_dof_mat.append(copy.deepcopy(new_dofs))
            new_dofs[i] -= float(coeff * step_frac)
        # end

        # The function to be evaluated in parallel
        def get_resp(new_dofi, exp, model_dct, mkey, init_dat):
            """Class method used in the parallel map function
            """
            model_dct[mkey] = model_dct[mkey].update_dof(new_dofi)                
            return -exp.compare(init_dat[0], init_dat[1][0],
                                exp(model_dct))

        pll_out = pll_loop(new_dof_mat, get_resp,
                           comm=comm,
                           exp=self,
                           model_dct=models,
                           mkey=model_key,
                           init_dat=initial_data)

        for key in pll_out:
            resp_mat[:,int(key)] = pll_out[key]
        #end
        
        sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T
        return np.where(np.fabs(sens_matrix) > 1E-21,
                        sens_matrix,
                        np.zeros(sens_matrix.shape))

    def get_sens_pll(self, models, model_key, initial_data=None):
        """Parallel evaluation of each model's sensitivities

        Args:
            models(dict): The dictionary of models
            model_key(str): The key of the model for which the sensitivity is
                            desired
        Keyword Args:
            initial_data(np.ndarray): The response for the nominal model DOF, if
                it is `None`, it is calculated when this method is called
        """

        import concurrent.futures

        models = copy.deepcopy(models)
        model = models[model_key]

        step_frac = 2E-2

        if initial_data is None:
            initial_data = self(models)
        # end

        resp_mat = np.zeros((initial_data[0].shape[0],
                             model.shape()))
        inp_mat = np.zeros((model.shape(),
                            model.shape()))
        new_dof_mat = []
        new_dofs = np.array(copy.deepcopy(model.get_dof()),
                            dtype=np.float64)

        for i, coeff in enumerate(model.get_dof()):
            new_dofs[i] += float(coeff * step_frac)
            inp_mat[:, i] = (new_dofs - model.get_dof())
            new_dof_mat.append(copy.deepcopy(new_dofs))
            new_dofs[i] -= float(coeff * step_frac)
        # end
            
        # with concurrent.futures.ThreadPoolExecutor(max_workers=50)\
        #      as executor:
        #     for i, resp in enumerate(executor.map(
        #             Experiment._get_resp,
        #             new_dof_mat,
        #             [copy.deepcopy(self) for i in range(len(new_dof_mat))],
        #             [copy.deepcopy(models) for i in range(len(new_dof_mat))],
        #             [copy.deepcopy(model_key) for i in range(len(new_dof_mat))],
        #             [copy.deepcopy(initial_data) for i in range(len(new_dof_mat))],
        #             )):
        #         resp_mat[:, i] = resp
        #     # end                
        # # end
        def _get_resp(new_dofi, exp, model_dct, mkey, init_dat):
            """Class method used in the parallel map function
            """
            model_dct[mkey] = model_dct[mkey].update_dof(new_dofi)   
            return -exp.compare(init_dat[0], init_dat[1][0],
                                exp(model_dct))
        # end
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = {}
            for i, coeff in enumerate(model.get_dof()):
                new_dofs[i] += float(coeff * step_frac)
                inp_mat[:, i] = (new_dofs - model.get_dof())
                futures[executor.submit(
                    _get_resp,
                    new_dofs,
                    copy.deepcopy(self),
                    copy.deepcopy(models),
                    copy.deepcopy(model_key),
                    copy.deepcopy(initial_data))] = i

                new_dofs[i] -= float(coeff * step_frac)
            # end

            for ftr in concurrent.futures.as_completed(futures):
                i = futures[ftr]

                try:
                    resp = ftr.result()
                except Exception as exc:
                    raise RuntimeError('{:s} sensitivities generated an'
                                       ' exception {:s}'.
                                       format(self.get_inform(1), exc))
                else:
                    resp_mat[:, i] = resp
                # end
            # end
        # end
 
        sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T
        return np.where(np.fabs(sens_matrix) > 1E-21,
                        sens_matrix,
                        np.zeros(sens_matrix.shape))

        
    def get_sens(self, models, model_key, initial_data=None):
        """Gets the sensitivity of the experiment response to the model DOF

        Args:
            models(dict): The dictionary of models
            model_key(str): The key of the model for which the sensitivity is
                            desired
        Keyword Args:
            initial_data(np.ndarray): The response for the nominal model DOF, if
                it is `None`, it is calculated when this method is called

        """

        models = copy.deepcopy(models)
        model = models[model_key]

        step_frac = 2E-2

        if initial_data is None:
            initial_data = self(models)
        # end

        resp_mat = np.zeros((initial_data[0].shape[0],
                             model.shape()))
        inp_mat = np.zeros((model.shape(),
                            model.shape()))
        new_dofs = np.array(copy.deepcopy(model.get_dof()),
                            dtype=np.float64)

        for i, coeff in enumerate(model.get_dof()):
            new_dofs[i] += float(coeff * step_frac)
            models[model_key] = model.update_dof(new_dofs)
            inp_mat[:, i] = (new_dofs - model.get_dof())
            resp_mat[:, i] = -self.compare(
                initial_data[0],
                initial_data[1][0],
                self(models))
            new_dofs[i] -= float(coeff * step_frac)
        # end
   
        sens_matrix = np.linalg.lstsq(inp_mat, resp_mat.T)[0].T
        return np.where(np.fabs(sens_matrix) > 1E-21,
                        sens_matrix,
                        np.zeros(sens_matrix.shape))

    def _get_hessian(self, models, model_key, initial_data=None):
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

    def get_fisher_matrix(self, models, use_hessian=False, exp=None,
                          sens_matrix=None, sigma=None):
        """Returns the fisher information matrix of the simulation

        Args:
            models(dict): Dictionary of models

        Keyword Args:
            use_hessian(bool): Flag to toggle wheather or not to use the hessian
            exp(Experiment): An experiment object, only used if use_hessian is
                             true
            sens_matrix(np.ndarray): the sensitivity matrix
                                     *Default None*

        Return:
            (np.ndarray): The fisher information matrix, a nxn matrix where
            `n` is the degrees of freedom of the model.
        """

        if sens_matrix is None:
            sens_matrix = self._get_sens(models)
        # end
        
        if sigma is None:
            sigma = inv(self.get_sigma(models))
        else:
            sigma = inv(sigma)
        # end
        
        if use_hessian:
            hessian = self._get_hessian(models, self.req_models.keys()[0])
            if exp is None:
                raise ValueError('{:} Must provide an experiment object when'
                                 'using the hessian'.
                                 format(self.get_inform(1)))
            else:
                model_data = self(models)
                exp_data = exp()
                epsilon = self.compare(exp_data[0], exp_data[1][0], model_data)
            # end
        # end

        if not use_hessian:
            return np.dot(sens_matrix.T, np.dot(sigma, sens_matrix))
        else:
            tmp = np.dot(epsilon.T,
                         np.sum([np.dot(sigma, hessian[i, :, :])
                                 for i in range(hessian.shape[0])]))
            return np.dot(sens_matrix.T, np.dot(sigma, sens_matrix)) + tmp


class GausianExperiment(Experiment):
    """An experiment class which can generate probability data assuming Gaussian
    errors.

    """

    def get_pq(self, models, opt_key, sim_data, experiment, sens_matrix,
               scale=False):
        """Generates the P and q matrix for the Bayesian analysis

        Args:
           models(dict): The dictionary of models
           opt_key(str): The key for the model being optimized
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           experiment(Experiment): A valid Experiment object
           sens_matrix(np.ndarray): The sensitivity matrix

        Keyword Arguments:
           scale(bool): Flag to use the model scaling

        Return:
            (tuple): Elements are:
                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """

        exp_data = experiment()
        epsilon = self.compare(exp_data[0], exp_data[1][0], sim_data)

        p_mat = np.dot(np.dot(sens_matrix.T,
                              inv(experiment.get_sigma(models))),
                       sens_matrix)
        q_mat = -np.dot(np.dot(epsilon,
                               inv(experiment.get_sigma(models))),
                        sens_matrix)

        if scale:
            prior_scale = models[opt_key].get_scaling()
            p_mat = np.dot(prior_scale, np.dot(p_mat, prior_scale))
            q_mat = np.dot(prior_scale, q_mat)
        # end

        return p_mat, q_mat

    def get_log_like(self, models, sim_data, experiment):
        """Gets the log likelihood of the current simulation

        Args:
           model(PhysicsModel): The model under investigation
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           experiment(Experiment): A valid Experiment object

        Return:
            (float): The log of the likelihood of the simulation
        """

        exp_data = experiment()
        epsilon = self.compare(exp_data[0], exp_data[1][0], sim_data)
        return -0.5 * np.dot(epsilon,
                             np.dot(inv(experiment.get_sigma(models)),
                                    epsilon))
