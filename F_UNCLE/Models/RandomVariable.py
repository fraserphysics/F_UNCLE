"""A model of a normally distributed random variable of known mean


 sigma = k epsilon + c epsilon'

"""

import copy


import numpy as np
from ..Utils.PhysicsModel import GaussianModel
from ..Utils.Struc import Struc

class RandomVariable(GaussianModel):
    """A model of a normally distributed random variable of  known variance
    """
    mean = np.array([0.0]) # The model coefficients
    prior = None
    
    def __init__(self, prior_mean, name='Random Variable', *args, **kwargs):
        """
        """
        def_opts = {
            'sigma': [float, 0.005 , 0, None, '',
                      "Normalized variance for each coefficeint"],
            }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        if isinstance(prior_mean, (int, float)):
            prior_mean = [prior_mean,]
        # end
        self.mean = np.array(prior_mean)

        Struc.__init__(self, name, def_opts = def_opts, *args, **kwargs)

        self.update_dof(prior_mean)
            
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
        sigma = self.get_option('sigma')
        return np.diag(sigma**2 * np.ones((self.shape(),)))

    def get_scaling(self):
        return np.diag(self.prior.get_dof())

    def get_constraints(self, scale=False):
        g_mat = np.zeros(2 * (self.shape(),))
        h_vec = np.ones((self.shape(),))        
        return g_mat, h_vec
    
    def shape(self):
        """
        """
        return 1

    def update_dof(self, dof_in):
        """
        """

        if isinstance(dof_in, (int, float)):
            dof_in = [dof_in,]
        # end
        
        if not len(dof_in) == 1:
            raise ValueError("{:} DOF must be length 1"
                             .format(self.get_inform(1)))
        # end
        new_model = copy.deepcopy(self)
        new_model.mean = np.array(dof_in)

        return new_model
    
    def __call__(self, *args, **kwargs):
        """Returns the mean of the distribution
        
        Args:
            None

        Return:
           (float): The mean

        """

        return float(self.mean)
    
    def get_dof(self):
        """
        """
        return np.array([self.mean])

    def _on_str(self):
        """
        """

        dof = self.get_dof()
        out_str = "\n\n"
        out_str += "Random Variable\n"
        out_str += "===============\n\n"

        out_str += "Mean         {:4.3e} GPa\n"\
                   .format(float(dof))
        out_str += "Prior Mean   {:4.3e} GPa s-1\n"\
                   .format(float(self.prior.get_dof()))
        out_str += "Variance     {:4.3e} GPa s-1\n\n"\
                   .format(self.get_option('sigma'))        

        return out_str

    
