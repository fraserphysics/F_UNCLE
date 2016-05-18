#/usr/bin/pyton
"""

pyBayesian

An object to extract properties of the bayesian analysis of experiments

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (03-16-2016)


"""

# =========================
# Python Standard Libraries
# =========================

import sys
import os
# import pdb
import unittest

# =========================
# Python Packages
# =========================
import numpy as np

# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../'))
from FUNCLE.pyExperiment import Experiment
from FUNCLE.pyIsentrope import EOSBump, EOSModel, Isentrope
from FUNCLE.utils.pyStruc import Struc

# =========================
# Main Code
# =========================

class Bayesian(Struc):
    """A calss for performing bayesian inference on a model given data

    Attributes:
       sim_exp(Experiment): The simulated experimental data
       true_exp(Experiment): The true experimental data
       prior(Experiment): The prior for the physics model

    """

    def __init__(self, model, data, prior, name='Bayesian', *args, **kwargs):
        """

        Instantiates the Bayesian analysis

        Args:
           sim_exp(Experiment): The simulated experimental data
           true_exp(Experiment): The true experimental data
           prior(Struc): The prior for the physics model

        Keyword Args:
            name(str): Name for the analysis.('Bayesian')

        Return:
           None
        """

        Struc.__init__(self, name=name)

        if isinstance(model, Experiment):
            self.model = model
        else:
            raise TypeError("{:} the simulated experiment must be an Experiment type"\
                            .format(self.get_inform(2)))
        # end

        if isinstance(data, Experiment):
            self.data = data
        else:
            raise TypeError("{:} the true experiment must be an Experiment type"\
                            .format(self.get_inform(2)))
        # end

        if isinstance(prior, Struc):
            self.priora = prior
        else:
            raise TypeError("{:} the prior must be a Struc type"\
                            .format(self.get_inform(2)))
        # end

    #end

    def update(self, sim_exp=None, true_exp=None, prior=None):
        """Updates the properties of the bayesian analtsis

        Keyword Args:
           sim_exp(Experiment): The simulated experimental data (Default None)
           true_exp(Experiment): The true experimental data (Default None)
           prior(Experiment): The prior for the physics model (Default None)

        Return:
            None

        """

        if isinstance(sim_exp, Experiment):
            self.sim_exp = sim_exp
        elif sim_exp is None:
            pass
        else:
            raise TypeError("{:} the simulated experiment must be an Experiment type"\
                            .format(self.get_inform(2)))
        # end

        if isinstance(true_exp, Experiment):
            self.true_exp = true_exp
        elif true_exp is None:
            pass
        else:
            raise TypeError("{:} the true experiment must be an Experiment type"\
                            .format(self.get_inform(2)))
        # end


        if isinstance(prior, Struc):
            self.priora = prior
        elif prior is None:
            pass
        else:
            raise TypeError("{:} the prior must be a Struc type"\
                            .format(self.get_inform(2)))
        # end

    def get_exp_sens(self):
        """Gets the sensitivity of the experimental data to changes to the EOS
        """

        pass
    # end

    def get_sim_sens(self):
        """Gets the sensitivity of the simulated experiment to the EOS
        """
        pass
    # end

class TestBayesian(unittest.TestCase):
    """Test class for the bayesian object
    """
    def setUp(self):
        """Setup script for each test
        """

        self.exp = Experiment(name="Dummy experiment")
        self.mod = Experiment(name="Dummy model")
        self.prior = Experiment(name="Dummy prior")

    # end

    def test_instantiation(self):
        """Test that the object can instantiate correctly
        """

        bayes = Bayesian(self.exp, self.mod, self.prior)

        print bayes
        
        self.assertIsInstance(bayes, Bayesian)
    # end

    def test_bad_instantiaion(self):
        """Tets impropper instantiation raises the correct errors
        """
        
        pass
    # end

if __name__ == '__main__':

    unittest.main(verbosity=4)
