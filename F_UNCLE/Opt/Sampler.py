"""Tool to sample about some optimal set of model coefficients
"""
import copy
import numpy as np
import scipy.stats as spstat
from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline
from ..Utils.Struc import Struc
from .Bayesian import Bayesian

class Sampler(Bayesian):
    """Class to sample
    """

    def __init__(self, opt_models, simulations, fisher):
        """

        Args:
            opt_models(OrderedDict): Optimal set of models from a previous analysis
            simulations(OrderedDict): The set of simulations and experiments used
            fisher(dict): The fisher information
        """

        self.models = opt_models
        self.simulations = simulations
        self.fisher = fisher


    def __call__(self, simid, modkey):
        """Returns a set of samples from the posterior distribution for the
        model

        Args:


        Return:
            (tuple): Elements are:

                [0](list): List of numpy arrays for the model DOF
                [1](list): List of dependant variables for aligned results
        """

        all_models = copy.deepcopy(self.models)
        model = self.models[modkey]
        sim = self.simulations[simid]
        all_fisher = self.fisher[simid]

        idx = 0
        for key in all_models:
            shape = all_models[key].shape()
            if key == modkey:
                fisher = all_fisher[idx: idx + shape, idx: idx + shape]
            else:
                idx += shape
            # end
        # end

        pcnt_list = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        dof_list = np.empty(
            (len(pcnt_list),
             model.shape()
             )
        )

        data_list = np.empty(
            (len(pcnt_list),
             self.simulations[simid]['exp'].shape()
             )
        )

        mean = model.get_dof()
        G, h = model.get_constraints(scale=True)
        new_variance = model.get_sigma()#self.cramer_rao(model, fisher)

        new_variance = np.dot(new_variance, np.diag(model.get_t()[:-4]**3/2.56E9))
        eigs, vects = np.linalg.eigh(new_variance)

        assert eigs.shape[0] == model.shape(), 'Too few eigenvalues'
        assert np.all(eigs >= 0.0), 'Negative eigenvalues'
        assert np.all(np.isreal(eigs)), 'Imaginary eigenvalues'
        assert np.all(np.fabs(
            np.dot(vects.T, vects) - np.eye(model.shape())) < 1E-14),\
            'Eigenvectors are not orthogonal'


        # Calculate map from N(0,I) to N(0,E)
        a_mat = np.dot(np.linalg.inv(vects), np.dot(np.diag(np.sqrt(eigs)), vects))

        lim = np.dot(np.linalg.inv(np.dot(a_mat, G)),h)
        # import pdb
        # pdb.set_trace()
        count = 0
        x = np.zeros((model.shape(),) )
        # for i in range(model.shape()):
        #     v = spstat.norm.rvs(
        #         loc=None,
        #         scale=1,
        #         random_state=None)
        #     x[i] = v
        #     if np.all(v < lim):
        #         break
        #     else:
        #         print(count, len(np.where(v > lim)[0]))
        #         count += 1
        # for j, pcnt in enumerate(pcnt_list):
        #     feasible=False
        #     count=0
        #     print(pcnt)
        #     while not feasible and count < 1000:
        #         #x_prob = spstat.multivariate_normal.rvs(mean=mean, cov=new_variance)
        #         x_prob = self.sample_at_prob(pcnt, mean, new_variance)
        #         new_model = model.update_dof(x_prob)
        #         conval = np.dot((mean-x_prob), G)
        #         feasible = np.all(conval<=h)
        #         print("\t", count, feasible)
        #         count += 1
        #         feasible=True
        #     # end
        #     all_models.update({modkey:new_model})
        #     data =self.simulations[simid]['sim'](all_models)
        #     data_list[j, :] = self.simulations[simid]['exp'].align(data)[1][0]
        #     dof_list[j, :] = x_prob
        # # end

        return dof_list, data_list

    def spectral_decomp(self, simid, modkey, tol=1E-2):
        """
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
        all_models = copy.deepcopy(self.models)
        model = all_models[modkey]
        sim = self.simulations[simid]
        all_fisher = self.fisher[simid]

        idx = 0
        for key in all_models:
            shape = all_models[key].shape()
            if key == modkey:
                fisher = all_fisher[idx: idx + shape, idx: idx + shape]
                break
            else:
                idx += shape
            # end
        # end

        # Spectral decomposition of info matrix and sort by eigenvalues
        eig_vals, eig_vecs = np.linalg.eigh(fisher)

        eps = np.finfo(np.float64).eps
        assert np.all(eig_vals >= -eps *  eig_vals.max()), 'Negative eigenvalues'

        eig_vals = np.maximum(eig_vals, 0)        # info is positive definite

        vals = eig_vals[np.argsort(eig_vals)[::-1]]
        vects = eig_vecs[:,np.argsort(eig_vals)[::-1]]
        n_vals = max(len(np.where(vals > vals[0] * tol)[0]), 3)
        n_vecs = max(len(np.where(vals > vals[0] * tol)[0]), 1)

        # Find range of v that includes support of eigenfunctions
        # knots = self.models[self.opt_key].get_t()
        knots = model._get_knot_spacing()
        funcs = []

        # Create a matrix of the basis functions evaluated at each knot
        basis_mtx = np.zeros((model.shape(), model.shape()))
        basis_fn = model.get_basis()
        for i, b_fn in enumerate(basis_fn):
                basis_mtx[:, i] = b_fn(knots)
        # end

        print(eig_vals)
        print(vals[:n_vals])
        for j in range(n_vecs):
            # funcs.append(IUSpline(knots, np.dot(basis_mtx,
            #                                     vects[j])))
            funcs.append(model.set_c(vects[:,j]))
        # end

        return vals[:n_vals], vects[:, :n_vecs], funcs

    def pointwise_bounds(self, simid, modkey, sim_solve=True):
        """Creates pointwise bounds of the function

        Args:
            simid(str): The key for the simulation of interest
            modkey(str): The key for the model of interest

        Keyword Args:
           simsolve(bool): Flag to solve the simulations at each level

        Return:
            (tuple): Elements are

                (list): The `n` probability levels
                (np.ndarray): `nxm` array of the `m` DOF at each probability
                (np.ndarray): `nxk` array of the `k` aligned data at each
                              probability
        """

        all_models = copy.deepcopy(self.models)
        model = all_models[modkey]
       # model.set_option('spline_sigma', 0.25)
        # Create dictionary of fisher information for this model
        mod_fisher = {}
        for sim in self.fisher:
            idx = 0
            for mkey in all_models:
                shape = all_models[mkey].shape()
                if mkey == modkey:
                    mod_fisher[sim] = self.fisher[sim][idx: idx + shape,
                                                       idx: idx + shape]
                else:
                    idx += shape
                # end
            # end
        # end
        pcnt_list = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]

        dof_list = []

        dof_array = np.empty(
            (len(pcnt_list),
             model.shape()
             )
        )

        old_dof_array = np.empty(
            (len(pcnt_list),
             model.shape()
             )
        )

        mean = model.get_dof()
        old_variance = model.get_sigma()
        new_variance = self.cramer_rao(model, mod_fisher, simid)
        eigs, vects = np.linalg.eigh(new_variance)

        eps = np.finfo(np.float64).eps
        assert eigs.shape[0] == model.shape(), 'Too few eigenvalues'
        assert np.all(eigs >= -eps * eigs.max()), 'Negative eigenvalues'
        assert np.all(np.isreal(eigs)), 'Imaginary eigenvalues'
        vec_orth_err = np.fabs(np.dot(vects.T, vects) - np.eye(model.shape()))
        assert np.all( vec_orth_err <= 20.0 * eps * vects.max()),\
            'Eigenvectors are not orthogonal err {:f}'.format(vec_orth_err.max())

        for j, pcnt in enumerate(pcnt_list):
            scale = spstat.norm.ppf(pcnt, loc=0, scale=1)
            dof_list.append(mean + scale * np.sqrt(np.diag(new_variance)))
            old_dof_array[j, :] = mean + scale * np.sqrt(np.diag(old_variance))
        # end

        if sim_solve:
            sim = self.simulations[simid]
            data_array = np.empty(
                (len(pcnt_list),
                 self.simulations[simid]['exp'].shape()
                )
            )

            # Solve for each simulation using multi_solve (Allows run_job and mpi)
            data_list = self.simulations[simid]['sim'].multi_solve(
                all_models,
                [modkey, ],
                dof_list
            )

            for j, (data, dof) in enumerate(zip(data_list, dof_list)):
                data_array[j, :] = self.simulations[simid]['exp'].align(data)[1][0]
                dof_array[j, :] = dof
            # end
        else:
            for i, dof in enumerate(dof_list):
                dof_array[i, :] = dof
            # end
            data_array = None
        # end

        return pcnt_list, dof_array, data_array, old_dof_array

    def cramer_rao(self, model, fisher_in, simid):
        """Estimates the varaince of the model given the fisher info

        Uses the Cramer-Rao bound saying that the lower bound on the variance
        is given by the invrese of the fisher informaiton

        The fisher information matrix is inverted using a NAME procedure

        """

        fisher = np.zeros(2 * (model.shape(), ))
        if simid.strip().lower() == 'all':
            for key in fisher_in:
                if key.lower() == 'all':
                    continue
                fisher += fisher_in[key]
            # end
        elif simid in fisher_in:
            fisher += fisher_in[simid]
        else:
            raise KeyError("{:s} not in fisher".format(simid))
        # end

        fisher += np.linalg.inv(model.get_sigma())
        return np.linalg.inv(fisher)
        # Check that the fisher information matrix is positive semi-definate
        # and that the eigenvectors are orthogonal
        # eigval_1, eigvec_1 = np.linalg.eigh(fisher)

        # eps = np.finfo(np.float64).eps
        # assert eigval_1.shape[0] == model.shape(), 'Too few eigenvalues'
        # assert np.all(eigval_1 >= -eps *  eigval_1.max()), 'Negative eigenvalues'
        # vec_orth_err = np.fabs(np.dot(eigvec_1.T, eigvec_1) - np.eye(model.shape()))
        # assert np.all( vec_orth_err <= 20 * eps),\
        #     'Eigenvectors are not orthogonal err {:e}'.format(vec_orth_err.max())

        # # Invert the degenerate eigenvalue matrix
        # val_inv = np.empty(eigval_1.shape)
        # largest_eig = eigval_1.max()
        # for i in range(eigval_1.shape[0]):
        #     if eigval_1[i] / largest_eig  > 1E-12:
        #         val_inv[i] = 1/eigval_1[i]
        #     else:
        #         val_inv[i] = 0.0
        #         #eigvec_1[:,i] = 0.0
        #     # end
        # # end

        # # Create the new variance matrix using the Cramer Rao bound
        # new_var = np.dot(eigvec_1,
        #                  np.dot(np.diag(val_inv),
        #                         eigvec_1.T
        #                  ))

        # # import pdb
        # # pdb.set_trace()
        # new_var = np.diag(new_var)
        # new_var=np.diag(np.where(new_var / original_var > 20 * eps, new_var, original_var))

        # # for i in range(eigval_1.shape[0]):
        # #     if eigval_1[i] == 0.0:
        # #         new_var[i,:] = np.zeros(eigval_1.shape[0])
        # #         new_var[i,i] = np.sqrt(original_var[i])
        # #     # end
        # # # end
        # #return np.diag(original_var)
        # return new_var**2

    def sample_at_prob(self, prob, mean, var, rstate=None):
        """
        """
        shape = mean.shape[0]
        # Get a sample from a distribution N(0,I)
        scale = spstat.norm.ppf(prob, loc=0, scale=1)
        v = spstat.multivariate_normal.rvs(
            mean=None,
            cov=1,
            size=shape,
            random_state=rstate)

        v *= np.fabs(scale) / np.sqrt((v**2).sum())

        # Spectral decomposition of target dist covariance
        eigs, vects = np.linalg.eigh(var)

        assert eigs.shape[0] == shape, 'Too few eigenvalues'
        assert np.all(eigs >= 0.0), 'Negative eigenvalues'
        assert np.all(np.isreal(eigs)), 'Imaginary eigenvalues'
        assert np.all(np.fabs(
            np.dot(vects.T, vects) - np.eye(shape)) < 1E-14),\
            'Eigenvectors are not orthogonal'


        # Calculate map from N(0,I) to N(0,E)
        a_mat = np.dot(vects.T, np.dot(np.diag(np.sqrt(eigs)), vects))

        # Add the mean to get N(u,E)
        #return mean + np.copysign(np.dot(a_mat, v), scale)

        return mean + scale * np.diag(np.sqrt(var))
