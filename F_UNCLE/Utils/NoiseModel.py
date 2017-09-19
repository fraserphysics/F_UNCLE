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

    def __call__(self, times, values, white_frac=0.1, cor_time=0.1E-6,
                 rstate=None):
        """Returns a random vector based

        Args:
           shape(tuple): A tuple of the shape you want out

        Return:
           (np.ndarray): The random noise, has the specified shape
        """

        # Creates a new random state if one was not passed
        if rstate is None:
            print('Generating a new random state')
            rstate = nprand.RandomState(seed=None)
        # end

        times = np.array(times, dtype=np.float64)
        shape = times.shape

        if len(shape) > 1:
            raise ValueError('{:} can only generate 1d noise'
                             .format(self.get_inform(1)))
        # end

        var = np.eye(shape[0], dtype=np.float64)
        var *= (white_frac * values.mean())**2

        corr = np.eye(shape[0], dtype=np.float64)

        eps = np.finfo(np.float64).eps
        for i in range(shape[0]):
            exponent = 0.5 * ((times[i] - times) / cor_time)**2
            corr[i, :] = np.where(exponent> np.log(eps),
                                  np.exp(-exponent),
                                  0.0)

        # end


        # Test that the correlation matrix is positive definite
        eig, vect = np.linalg.eigh(corr)

        eig = np.where(eig > eig.max() * eps, eig, 0.0)
        assert eig.shape[0] == corr.shape[0], "Too few eigenvalues"
        assert np.all(np.isreal(eig)), "Imaginary eigenvalues"
        assert np.all(eig >= 0.0), "Negative eigenvalues {:}".format(eig)

        for i in range(vect.shape[1]):
            if eig[i] == 0:
                pass
                #vect[:,i] = 0.0
            # end
        # end

        corr = np.dot(vect.T, np.dot(np.diag(eig), vect))

        sigma = np.dot(np.sqrt(var).T, np.dot(corr, np.sqrt(var)))
        noise = spstat.multivariate_normal.rvs(mean=None,
                                               cov=sigma,
                                               size=1,
                                               random_state=rstate)
        assert np.all(np.isfinite(noise))
        bias = 0.0E4
        return noise + bias
