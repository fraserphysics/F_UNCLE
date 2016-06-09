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

    A physics model is computer code that represents how some physical process
    responds to changes in the regime.
    
    DOF
        A physics model has degrees of freedom, dof, which represent how many 
        parameters the model has which can be adjusted to affect its response
    
    Prior
         A physics model has a prior, which represents the best esimate of the
         model's degrees of freedom. This prior is used by Bayesian methods
    
    ..note::
    
        **all** abstract methods must be overloaded for a physics model to 
        work in the `F_UNCLE` framework
    
    Attributes:
       prior(PhysicsModel): the prior

    """

    def __init__(self, prior, name='Abstract Physics Model', *args, **kwargs):
        """
        
        Args:
           prior: Can be either a PhysicsModel' object or a function or a vector
                  which defines the prior

        Keyword Args:
            name(str): A name for the model

        """

        Struc.__init__(self, name, *args, **kwargs)

        self.prior = None
        self.update_prior(prior)

        return
    # end

    def update_prior(self, prior):
        """Updates the prior for the pyhsics model

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

    def get_scale(self):
        """**ABSTRACT** Returns a matrix to scale the model degrees of fredom

        Scaling the model dofs may be necessary where there are large changes in
        the magnitude of the dofs
        
        Return:
            (np.ndarray): 
        """
        return
    def get_sigma(self, *args, **kwargs):
        """**Abstract** Gets the covariance matrix of the model
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """

        raise NotImplementedError('{} has not defined a covariance matrix'\
                                  .format(self.get_inform(1)))    

    def shape(self):
        """**ABSTRACT** Returns the degrees of freedom of the experiment
        
        Return:
           (tuple): Dimensions

        """

        raise NotImplementedError("{} does not have a shape"\
                                  .format(self.get_inform(1)))
        

    def set_dof(self):
        """**ABSTRACT** Gets the model degrees of freedom
        
        Args:
           None

        Return:
           (Iterable): The iterable defining the model degrees of freedom

        """

        raise NotImplementedError("{} does not set the model dof"\
                                  .format(self.get_inform(1)))
    
    def get_dof(self, dof_in):
        """**ABSTRACT** Sets the model's degrees of freedom
        
        Args:
           dof_in(Iterable): The new values for *all* model degrees of freedom
        
        Return:
           None
        """

        raise NotImplementedError("{} does not provide model dofs"\
                                  .format(self.get_inform(1)))

    
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
