"""A model of a normally distributed random process of known variance
"""

import copy


import numpy as np
from ..Utils.PhysicsModel import GaussianModel
from ..Utils.Struc import Struc

class RandomProcess(GaussianModel):
    """A model of a normally distributed random process of  known variance
    """
    mean = None
    var = None
    prior = None
    
    def __init__(self, prior_mean, prior_var, name='Random Process',
                 *args, **kwargs):
        """
        """
        def_opts = {
            }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end
        
        Struc.__init__(self, name, def_opts = def_opts, *args, **kwargs)

        self.mean = prior_mean
        self.update_dof(prior_mean, prior_var)
        self.var = prior_var
        self.prior = copy.deepcopy(self)

    def plot(self, axes = None, labels=['Coeff'], linestyles=['ok'],
             *args, **kwargs):
        """
        """
        axes.plot(self.get_dof(), linestyles[0], label=labels[0])
    # end
    
    def get_sigma(self, *args, **kwargs):
        """
        """
        return np.diag(self.var**2)

    def get_scaling(self):
        return np.diag(self.prior.get_dof())

    def get_constraints(self, scale=False):
        g_mat = np.zeros(2 * (self.shape(),))
        h_vec = np.ones((self.shape(),))        
        return g_mat, h_vec
    
    def shape(self):
        """
        """
        return len(self.mean)

    def update_dof(self, dof_in, var_in=None):
        """
        """

        if isinstance(dof_in, (int, float)):
            raise TypeError("{:} Random process DOF must be iterable"
                            .format(self.get_inform(1)))
        elif len(dof_in) < 2:
            raise TypeError("{:} Random process DOF must have two or more"
                            " degrees of freedom"
                            .format(self.get_inform(1)))
        # end

        if not isinstance(dof_in, np.ndarray):
            dof_in = np.array(dof_in)

        if var_in is None:
            var_in = self.var
        elif not isinstance(var_in, np.ndarray):
            var_in = np.array(var_in)

        if not dof_in.shape[0] == self.mean.shape[0]:
            raise IndexError("{:} DOF must have the same dimensions"
                             " as the prior".format(self.get_inform(1)))
        
        if var_in is not None and not dof_in.shape[0] == var_in.shape[0]:
            raise IndexError("{:} Variance must have the same dimensions"
                             " as DOF".format(self.get_inform(1)))
        
        new_model = copy.deepcopy(self)
        new_model.mean = dof_in
        new_model.var = var_in
        return new_model
    
    def __call__(self, *args, **kwargs):
        """Returns the mean of the distribution
        
        Args:
            None

        Return:
           (float): The mean

        """

        return self.mean
    
    def get_dof(self):
        """
        """
        return self.mean

    def _on_str(self):
        """
        """

        dof = self.get_dof()
        out_str = "\n\n"
        out_str += "Random Process\n"
        out_str += "==============\n\n"

        for i in range(self.mean.shape[0]):
        out_str += "\tNumber   Mean       Variance\n"
        out_str += "\t------   ----       --------\n"                   
        out_str += "\t{:02d}   {:4.3e}    {:4.3e}\n"\
                   .format(i, self.mean[i], self.var[i])

        return out_str

    
