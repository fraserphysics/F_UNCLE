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
       simulations([Experiment]): The simulated experimental data
       experiments([Experiment]): The true experimental data
       prior(Isentrope): The prior for the physics model
       dimensions(tuple):
       sens_matrix(nump.ndarray): The (nxm) sensitivity matrix
                                  n - model degrees of freedom
                                  m - total experiment DOF
                                  [i,j] - sensitivity of model DOF i 
                                          to experiment DOF j

    """

    def __init__(self, simulations, experiments, prior, name='Bayesian', *args, **kwargs):
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


        
        def_opts = {
            'outer_atol' : [float, 1E-6, 0.0, 1.0, '-',
                            'Absolute tolerance on change in likelyhood for\
                            outer loop convergence'],
            'outer_rtol' : [float, 1E-6, 0.0, 1.0, '-',
                            'Relative tolerance on change in likelyhood for\
                            outer loop convergence'],
            ''
        }
        
        Struc.__init__(self, name=name, def_opt=def_opts)

        self.update(simulations=simulations,
                    experiments=experiments,
                    prior=prior) 

        self.sens_matrix = np.nan * np.ones(self.shape)        
    #end
    def _on_str(self):
        """Print method for bayesian model
        """
        
        out_str = ''
        out_str += 'Prior\n'
        out_str += '=====\n'
        out_str += str(self.prior)
        
        out_str += 'Experiments\n'
        out_str += '===========\n'
        for exp in self.experiments:
            out_str += str(exp)
        #end

        out_str += 'Simulations\n'
        out_str += '===========\n'
        for exp in self.simulations:
            out_str += str(exp)
        #end
        
    def update(self, simulations=None, experiments=None, prior=None):
        """Updates the properties of the bayesian analtsis

        Keyword Args:
           sim_exp(Experiment): The simulated experimental data (Default None)
           true_exp(Experiment): The true experimental data (Default None)
           prior(Experiment): The prior for the physics model (Default None)

        Return:
            None

        """
        if experiments is None:
            pass
        elif isinstance(experiments, Experiment):
            self.experiments = [experiments]
        elif isinstance(experiments, (list, tuple)):
            self.experiments = []
            for exp in experiments:
                if isinstance(exp, Experiment):
                    self.experiments.append(exp)
                else:
                    raise TypeError("{:} each experiment must be an\
                                    Experiment type".format(self.get_inform(2)))
                #end
            #end
        else:
            raise TypeError("{:} the simulated experiment must be an Experiment type"\
                            .format(self.get_inform(2)))
        # end
        if simulations is None:
            pass
        elif isinstance(simulations, Experiment):
            self.simulations = [simulations]
        elif isinstance(simulations, (list, tuple)):
            self.simulations = []
            for exp in simulations:
                if isinstance(exp, Experiment):
                    self.simulations.append(exp)
                else:
                    raise TypeError("{:} each simulation must be an\
                                    Experiment type".format(self.get_inform(2)))
                #end
            #end
        else:
            raise TypeError("{:} the true experiment must be an Experiment type"\
                            .format(self.get_inform(2)))
        # end

        if prior is None:
            pass
        elif isinstance(prior, Struc) and hasattr(prior, 'get_sigma'):
            self.prior = prior
        else:
            raise TypeError("{:} the prior must be a Struc type"\
                            .format(self.get_inform(2)))
        # end

        self.shape = self._get_dim()

    def _get_dim(self):
        """Gets the dimenstions of the problem

        Return:
           (tuple): The n x m dimensions of the problem
        """

        n = self.prior.get_option('spline_N')
        m = 0
        for exp in self.experiments:
            m += exp.exp_points
        #end
        
        return (n, m)
        

    def __call__(self):
        """Determines the best candidate EOS function for the models       
        
        Return:
           (Isentrope): The isentrope which gives best agreement over the space
           (list): The history of candiate prior DOF's
        """
        
        atol = 1
        reltol = 1
        maxiter = 1
        
        prior = self.prior        
        best_model = copy.deepcopy(prior)

        conv = False
        for i in xrange(maxiter):

            
            absdif, reldif = self.conv_check(new_c, best_model.get_c())

            if absdif < abstol and reldif < reltol:
                conv = True
                break
            #end            
        #end

        if conv:
            raise Exception("{}: Outer loop could not converge to the given\
                             tolerance in the maximum number of iterations"\
                            .format(self.get_inform(1)))
        #end

        

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
