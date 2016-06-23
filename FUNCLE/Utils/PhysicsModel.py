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
import unittest
import sys
import os
# =========================
# Python Packages
# =========================

# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from FUNCLE.Utils.Struc import Struc

# =========================
# Main Code
# =========================

class PhysicsModel(Struc):
    """

    Abstract class for a physics model

    A physics model is computer code that represents how some physical process
    responds to changes in the regime.

    **Definitions**

    DOF
        A physics model has degrees of freedom, dof, which represent how many
        parameters the model has which can be adjusted to affect its response

    Prior
         A physics model has a prior, which represents the best estimate of the
         model's degrees of freedom. This prior is used by Bayesian methods

    .. note::

        **all** abstract methods must be overloaded for a physics model to
        work in the `F_UNCLE` framework

    **Attributes**

    Attributes:
       prior(PhysicsModel): the prior

    **Methods**
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
        """Updates the prior for the physics model

        Args:
           prior(PhysicsModel): The prior

        """

        if self.prior is None:
            pass
        elif not isinstance(prior, type(self.prior)):
            raise TypeError("{} New prior must be the same type as the old".\
                            format(self.get_inform(1)))
        #end
        self._on_update_prior(prior)
    def get_scaling(self):
        """Returns a matrix to scale the model degrees of freedom

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Scaling the model dofs may be necessary where there are large changes in
        the magnitude of the dofs

        Return:
            (np.ndarray):
                 a n x n matrix of scaling factors to make all dofs of the same
                 order of magnitude.
        """

        raise NotImplementedError('{} has not defined a co-variance matrix'\
                          .format(self.get_inform(1)))

    def get_sigma(self, *args, **kwargs):
        """Gets the co-variance matrix of the model

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (np.ndarray):
                A n x n diagonal matrix of the uncertainty in each dof.
                where n is the model degrees of freedom
        """

        raise NotImplementedError('{} has not defined a co-variance matrix'\
                                  .format(self.get_inform(1)))

    def shape(self):
        """Returns the shape of the model dof space

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Return:
           (tuple): Dimensions

        """

        raise NotImplementedError("{} does not have a shape"\
                                  .format(self.get_inform(1)))


    def set_dof(self, dof_in):
        """Sets the model degrees of freedom

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
           dof_in(Iterable): The new values for *all* model degrees of freedom

        Return:
            None

        """

        raise NotImplementedError("{} does not set the model dof"\
                                  .format(self.get_inform(1)))

    def get_dof(self):
        """Returns the model degrees of freedom

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
           None

        Return:
           (np.ndarray): The model degrees of freedom
        """

        raise NotImplementedError("{} does not provide model dofs"\
                                  .format(self.get_inform(1)))


    def _on_update_prior(self, prior):
        """Instance specific prior update

        Args:
            prior(PhysicsModel): The prior

        Return:
            None
        """

        self.prior = prior
# end

class TestPhysModel(unittest.TestCase):
    """Test of PhysicsModel object
    """
    def test_standard_instantiation(self):
        """Tests that teh model can be instantiated
        """
        model = PhysicsModel(prior=3.5)

        self.assertIsInstance(model, PhysicsModel)
    # end

    def test_update_prior(self):
        """Tests setting and updating the prior
        """
        model = PhysicsModel(prior=3.5)

        self.assertEqual(model.prior, 3.5)

        model.update_prior(2.5)

        self.assertEqual(model.prior, 2.5)
    # end

    def test_bad_update_prior(self):
        """Tests bad use of prior setting
        """
        model = PhysicsModel(prior=3.5)

        with self.assertRaises(TypeError):
            model.update_prior('two point five')
        # end
    # end
 # end

if __name__ == '__main__':

    unittest.main(verbosity=4)
