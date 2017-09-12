"""A model for noise
"""


import numpy.random as nprand
import numpy as np
import scipy.stats as spstat
import scipy.integrate as spint

from .Struc import Struc


class NoiseModel(Struc):
    """Model for biased, correlated white noise
    """

    def __init__(self):

        # ampl 'The variance for the noise'
        # corr_length 'Noise correlation length'
        # bias 'Bias for the noise'

        Struc.__init__(self, name="Noise Model")

    def __call__(self, times, values, white_frac = 0.1, cor_time=0.5E-6, rstate=None):
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

        if len(shape) > 1:
            raise ValueError('{:} can only generate 1d noise'
                             .format(self.get_inform(1)))
        # end

        ampl = (white_frac * values)**2

        corr = np.eye(shape[0])

        for i in range(shape[0]):
            x_list = (times[i] - times)
            corr[i, :] = np.exp(-0.5 * (x_list/cor_time)**2)
        # end

        # Test that the correlation matrix is positive definite
        eig, vect = np.linalg.eig(corr)
        eig = np.real(eig)
        vect = np.real(vect)
        
        for i in np.where(eig > 0)[0]:
            vect[:, i] = np.zeros(vect.shape[0])
        # end
        eig = np.where(eig > 0, eig, 0.0)

        assert eig.shape[0] == corr.shape[0], "Too few eigenvalues"
        assert np.all(np.isreal(eig)), "Imaginary eigenvalues"
        assert np.all(eig >= 0.0), "Negative eigenvalues {:}".format(eig)
        corr = np.dot(np.diag(eig * ampl), vect)

        # var = np.diag(ampl**(0.5))

        # sigma = np.dot(np.dot(var, np.eye(shape[0])), var)
        # sigma += corr
        
        import pdb
        pdb.set_trace()
        
        noise = spstat.multivariate_normal.rvs(mean=None,
                                               cov=corr,
                                               size=1,
                                               random_state=rstate)

        assert np.all(np.isfinite(noise))
        bias = 0.0#0.005E4
        return noise + bias
