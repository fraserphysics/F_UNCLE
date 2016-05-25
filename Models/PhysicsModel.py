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
    
    Abstract class for Batesian analysis
    
    """

    def __init__(self, name = 'Abstract Physics Model',  prior = None, *args, **kwargs):
        """
        
        **Arguments**
        
        None

        **Keyword Arguments**
        
        - prior -> PhysicsModel: The prior for the pysics model. *Default = None*
        
        """

        Struc.__init__(self, name, *args, **kwargs)
        
        self.prior = prior

        return
    # end

    def update_prior(self, prior):
        """
        
        Updates the prior for the pyhsics model

        **Arguments**
        
        - prior -> PhysicsModel: The prior
        
        """

        if hasattr(prior, '__call__'):
            self.on_update_prior(prior)
        elif isinstance(prior,type(self.prior)):
            self.on_update_prior(prior)
        elif self.prior == None:
            self.on_update_prior(prior)
        else:
            raise TypeError("{:} prior for update must be the same type as previous prior".format(self.getInform(2)))
        # end

        return

    # end

    def _on_update_prior(self, prior):
        """
        """
        self.prior = prior
# end 


if __name__ == '__main__':
    import unittest

    class test_object(unittest.TestCase):

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
        
    unittest.main(verbosity = 4)

    
    
