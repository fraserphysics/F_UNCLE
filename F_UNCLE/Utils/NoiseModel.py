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
    
    def __call__(self, times, values, lmb = 20, rstate=None):
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
        
        ampl = values
        
        corr = np.eye(shape[0])

        # for i in range(shape[0]):
        #     x_list = times[i] - times
        #     var = 1E-2
        #     x_list[i] = 0.0            
        #     corr[i, :] = np.exp(-0.5 * (x_list/var)**2)
        # # Test that the correlation matrix is positive definite
        # eig = np.linalg.eigvalsh(corr)
        # assert(eig.shape[0] == corr.shape[0])
        # assert(np.all(np.isreal(eig)))
        # assert(np.all(eig >= -1E-10))

        var = np.diag(ampl * np.ones((shape[0],)))
        sigma = np.dot(np.dot(np.sqrt(var), corr), np.sqrt(var))
        
        noise = spstat.multivariate_normal.rvs(mean=None,
                                               cov=sigma,
                                               size=1,
                                               random_state=rstate)

        assert(np.all(np.isfinite(noise)))
        bias = 0.0
        
        return noise + bias
