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

    A child of the Struc class. This abstract class contains methods common to
    all Experiment  objects. This class can be used to model two different cases

    Simulation
        Makes use of a single model or set of models internal to the object to
        simulate some physical process

    Experiment
        Can be of two types

        1. A "computational experiment" where a simulation is performed using a
           nominal *true* model
        2. A representation of a real experiment using tabulated values

    In order for an Experiment to work with the F_UNCLE framework, it must
    implement **all** the inherited methods from `Experiment`, regardless if
    it is a Simulation or Experiment

    **Attributes**

    None

    **Methods**
    """
    def __init__(self, name='Experiment', *args, **kwargs):
        """Instantiates the object.

        Options can be set by passing them as keyword arguments

        """

        def_opts = {}

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        #end
        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)
    # end

    def get_sigma(self, *args, **kwargs):
        """**ABSTRACT** Gets the co-variance matrix of the experiment

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (np.ndarray):
                A nxn array of the co-variance matrix of the simulation.
                Where n is the length of the independent variable vector, given
                by `shape()`
        """

        raise NotImplementedError('{} has not defined a co-variance matrix'\
                                  .format(self.get_inform(1)))

    def shape(self, *args, **kwargs):
        """**ABSTRACT** Gives the length of the independent variable vector

        Return:
            (int): The number of independent variables in the experiment
        """

        raise NotImplementedError('{} has no shape specified'\
                                  .format(self.get_inform(1)))


    def __call__(self, *args, **kwargs):
        """**ABSTRACT** Runs the simulation.

        The simulation should be structured so that the attributes and options
        of simulation should provide all the needed initial conditions are
        instantiated breadth.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
           (numpy.ndarray):
               The single vector of the simulation independent
               variable
           (list):
                A list of np.ndarray objects representing the various
                dependent variables for the problem. Element zero is the
                most important quantity. By default, comparisons to other
                data-sets are made to element zero. The length of each
                element of this list must be equal to the length of the
                independent variable vector.
           (list):
                A list of other attributes of the simulation result. The
                composition of this list is problem dependent
        """

        raise NotImplementedError('{} has no __call__ method instantiated'\
                                  .format(self.get_inform(1)))

    def compare(self, indep, dep, model_data):
        """Compares a set of experimental data to the model

        Args:
           indep(list): The list of independent variables for comparison
           dep(list): The list or array of dependent variables for comparison
           model_data(tuple): Complete output of a `__call__` to an `Experiment` object
                       which `dep` is compared to at every point in `indep`

        Returns
           (np.ndarray): The error between the dependent variables
               and the model for each value of independent variable
        """

        raise NotImplementedError('{} has not compare method instantiated'\
                          .format(self.get_inform(1)))


class TestExperiment(unittest.TestCase):
    """Test of the experiment class
    """

    def test_instantiation(self):
        """Tests that the class can instantiate correctly
        """
        exp = Experiment()

        self.assertIsInstance(exp, Experiment)

        print '\n'+str(exp)
    # end
# end

if __name__ == '__main__':
    unittest.main(verbosity=4)
