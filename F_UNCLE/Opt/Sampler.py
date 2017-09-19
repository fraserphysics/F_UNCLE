"""Tool to sample about some optimal set of model coefficients
"""
import copy
import numpy as np
import scipy.stats as spstat

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

        lim = np.dot(np.linalg.inv(np.dot(a_mat, G)), h))
        import pdb
        pdb.set_trace()
        count = 0
        x = np.zeros((model.shape(),) )
        for i in range(model.shape())
            v = spstat.norm.rvs(
                loc=None,
                scale=1,
                random_state=None)
            x[i] = v
            if np.all(v < lim):
                break
            else:
                print(count, len(np.where(v > lim)[0]))
                count += 1
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

    def cramer_rao(self, model, fisher):
        """
        """
        original_var = np.diag(model.get_sigma())

        # Check that the fisher information matrix is positive semi-definate
        eigval_1, eigvec_1 = np.linalg.eigh(fisher)

        # assert eigval_1.shape[0] == model.shape(), 'Too few eigenvalues'
        # assert np.all(eigval_1 >= 0.0), 'Negative eigenvalues'
        # assert np.all(np.isreal(eigval_1)), 'Imaginary eigenvalues'

        # Invert the degenerate eigenvalue matrix
        for i in range(eigval_1.shape[0]):
            if eigval_1[i]/eigval_1.max() > 1E-6 and eigval_1[i] > 1E-21:
               eigval_1[i] = np.sqrt(eigval_1[i]**-1)
            else:
               eigval_1[i] = np.sqrt(original_var[i])
               # eigvec_1[:,i] = np.zeros(eigvec_1.shape[1])
            # end
        # end

        assert np.all(np.fabs(
            np.dot(eigvec_1.T, eigvec_1) - np.eye(model.shape()))<1E-12),\
            'Eigenvectors are not orthogonal'

        # Create the new variance matrix using the Cramer Rao bound
        new_var = np.dot(eigvec_1.T,
                         np.dot(np.diag(eigval_1),
                                eigvec_1
                         ))

        # for i in range(eigval_1.shape[0]):
        #     if eigval_1[i] == 0.0:
        #         new_var[i,:] = np.zeros(eigval_1.shape[0])
        #         new_var[i,i] = np.sqrt(original_var[i])
        return new_var**2

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
