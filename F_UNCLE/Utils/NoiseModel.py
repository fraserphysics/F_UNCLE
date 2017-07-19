"""A model for noise
"""


import numpy.random as nprand
import numpy as np
import scipy.stats as spstat

from .Struc import Struc

class NoiseModel(Struc):
    """Model for biased, correlated white noise
    """

    def __init__(self):

        # ampl 'The variance for the noise'
        # corr_length 'Noise correlation length'
        # bias 'Bias for the noise'
        
        Struc.__init__(self, name="Noise Model")
    
    def __call__(self, times, values, rstate=None):
        """Returns a random vector based 

        Args:
           shape(tuple): A tuple of the shape you want out
        
        Return:
           (np.ndarray): The random noise, has the specified shape       
        """

        # Creates a new random state if one was not passed
        if rstate is None:
            print('Generateing a new random state')
            rstate = nprand.RandomState(seed=None)
        # end
        
        shape = times.shape
        
        if len(shape)>1:
            raise ValueError('{:} can only generate 1d noise'
                             .format(self.get_inform(1)))
        # end
        ampl = 0.75 * values
        
        corr = np.diag(np.ones((shape[0],)))
        var = np.diag(ampl * np.ones((shape[0],)))
        sigma = np.dot(np.dot(np.sqrt(var), corr), np.sqrt(var))
        
        noise = spstat.multivariate_normal.rvs(mean=None,
                                               cov=sigma,
                                               size=1,
                                               random_state=rstate)

        bias = 0.0

        return noise + bias
