"""A simplified strength model


 sigma = k epsilon + c epsilon'

"""

import copy


import numpy as np
from ..Utils.PhysicsModel import GaussianModel
from ..Utils.Struc import Struc


class SimpleStr(GaussianModel):
    """Simplified strength model
    """
    coeff = np.array([0.0, 0.0])  # The model coefficients
    prior = None

    def __init__(self, prior_coeff, name='Simple Strength Model',
                 *args, **kwargs):
        """
        """
        def_opts = {
            'sigma': [float, 0.005, 0, None, '',
                      "Normalized variance for each coefficeint"],
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        self.coeff = np.array(prior_coeff)

        Struc.__init__(self, name, def_opts=def_opts, *args, **kwargs)

        self.update_dof(prior_coeff)

        self.prior = copy.deepcopy(self)

    def plot(self, axes=None, labels=['Coeff'], linestyles=['ok'],
             *args, **kwargs):
        """
        """
        axes.plot(self.get_dof(), linestyles[0], label=labels[0])
    # end

    def get_sigma(self, *args, **kwargs):
        """
        """

        sigma = self.get_option('sigma')
        return np.diag((sigma * self.prior.get_dof())**2)

    def get_scaling(self):
        return np.diag(self.prior.get_dof())
#        return np.diag(np.ones((self.shape(),)))

    def get_constraints(self, scale=False):
        g_mat = np.zeros(2 * (self.shape(),))
        h_vec = np.ones((self.shape(),))
        return g_mat, h_vec

    def shape(self):
        """
        """
        return len(self.coeff)

    def update_dof(self, dof_in):
        """
        """

        if not len(dof_in) == 2:
            raise ValueError("{:} DOF must be a length two iterable"
                             .format(self.get_inform(1)))
        # end
        new_model = copy.deepcopy(self)
        new_model.coeff = np.array(dof_in)

        return new_model

    def __call__(self, epsilon, epsilon_dot, *args, **kwargs):
        """Calculates the stress corresonding to the strain

        .. math::

            \sigma = A \epsilon + B \dot{\epsilon}

        Args:
            epsilon(np.ndarray): nx1 array of true strain
            epsilon_dot(np.ndarray): nx1 array of true strain rate

        Return:
            (np.ndarray): nx1 array of stress

        """

        return self.coeff[0] * epsilon + self.coeff[1] * epsilon_dot

    def get_dof(self):
        """
        """
        return self.coeff

    def _on_str(self):
        """
        """

        dof = self.get_dof()
        out_str = "\n\n"
        out_str += "Degrees of Freedom\n"
        out_str += "==================\n\n"

        out_str += "Elastic Modulus          {:4.3e} GPa\n".format(dof[0] / 1E9)
        out_str += "Strain rate dependency   {:4.3e} GPa s-1\n\n"\
                   .format(dof[1] / 1E9)

        return out_str
