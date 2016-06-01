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

# =========================
# Python Standard Libraries
# =========================

import sys
import os
# import pdb
# import copy
import unittest

# =========================
# Python Packages
# =========================
# import numpy as np

# from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Utils.Struc import Struc


# =========================
# Main Code
# =========================

class Experiment(Struc):
    """Abstract class for experiments

    Attributes:
       dof(int): The degrees of freedom of the experiment

    """
    def __init__(self, name='Experiment', *args, **kwargs):

        def_opts = {}

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        #end



        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)
    # end

    def get_sigma(self, *args, **kwargs):
        """Gets the covariance matrix of the experiment
        """

        raise NotImplementedError('{} has not defined a covariance matrix'\
                                  .format(self.get_inform(1)))

    def shape(self, *args, **kwargs):
        """Gets the degrees of freedom of the experiment and retursn them
        """

        return 0

    def compare(self, indep, dep, model_data):
        """Compares a set of experimental data to the model

        Args:
           indep: The list or array of independent variables
           dep: The list or array of dependant variables
           model_data: The list of list of model outputs

        Retrurns
           (np.ndarray): The error between the dependant variables
                         and the model for each value of independent variable
        """

        pass

class TestExperiment(unittest.TestCase):
    """Test of the experiment class
    """

    def test_instantiation(self):
        """Tests that the cass can instantiate correctlt
        """
        exp = Experiment()

        self.assertIsInstance(exp, Experiment)

        print '\n'+str(exp)
    # end
# end

if __name__ == '__main__':
    unittest.main(verbosity=4)
