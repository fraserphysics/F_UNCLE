#/usr/bin/pyton
"""

pyPhysicsModel

Abstract class for a physics model used in the Bayesian analysis

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
import unittest

# =========================
# Python Packages
# =========================

# =========================
# Custom Packages
# =========================

sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Utils.Struc import Struc

# =========================
# Main Code
# =========================


class PhysicsModel(Struc):
    """

    Abstract class for a pysics model
    
    Attributes:
       prior(PhysicsModel): the prior


    """

    def __init__(self, prior, name='Abstract Physics Model', *args, **kwargs):
        """

        Args:
           None

        Keyword Args
           prior(PhysicsModel): The prior for the pysics model. *Default = None*

        """

        Struc.__init__(self, name, *args, **kwargs)

        self.prior = None
        self.update_prior(prior)

        return
    # end

    def update_prior(self, prior):
        """

        Updates the prior for the pyhsics model

        Args:
           prior(PhysicsModel): The prior

        """

        if prior is None and self.prior is None:
            raise ValueError("{}: requires a prior".format(self.get_inform(1)))
        else:
            self._on_update_prior(prior)
        # end

        return

    # end

    def get_sigma(self, *args, **kwargs):
        """Gets the covariance matrix of the model
        """

        raise NotImplementedError('{} has not defined a covariance matrix'\
                                  .format(self.get_inform(1)))    

    def shape(self):
        """Gets the degrees of freedom of the experiment and retursn them
        
        Return:
           (tuple): Dimensions

        """
        return (0)

    def set_dof(self):
        """Gets the model degrees of freedom
        
        Args:
           None

        Return:
           (Iterable): The iterable defining the model degrees of freedom

        """

        raise NotImplementedError()
    
    def get_dof(self, c_in):
        """Sets the model degrees of freedom
        
        Args:
           (Iterable): The new values for *all* modek degrees of freedom
        
        Return:
           None
        """

        raise NotImplementedError()
    
    def _on_update_prior(self, prior):
        """Instance specific prior update
        """
        
        self.prior = prior
# end

class TestPhysModel(unittest.TestCase):

    def test_standard_instantiation(self):
        model = PhysicsModel(prior = 3.5)
    # end

    def test_update_prior(self):
        model = PhysicsModel(prior = 3.5)

        self.assertEqual(model.prior, 3.5)

        model.update_prior(2.5)

        self.assertEqual(model.prior, 2.5)
    # end

    def test_bad_update_prior(self):

        model = PhysicsModel(prior = 3.5)

        with self.assertRaises(TypeError):
            model.update_prior('two point five')
        # end
    # end
 # end

if __name__ == '__main__':

    unittest.main(verbosity = 4)
