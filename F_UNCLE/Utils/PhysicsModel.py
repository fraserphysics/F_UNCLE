# /usr/bin/pyton
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# =========================
# Python Standard Libraries
# =========================
import sys
import os
import copy
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
# =========================
# Custom Packages
# =========================
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Struc import Struc
else:
    from .Struc import Struc
# end

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

    def __init__(self, prior, name=u'Abstract Physics Model', *args, **kwargs):
        """

        Args:
           prior: Can be either a PhysicsModel' object or a function or a vector
                  which defines the prior

        Keyword Args:
            name(str): A name for the model

        """

        Struc.__init__(self, name, *args, **kwargs)

        self.prior = prior

    # end

    def update_prior(self, prior):
        """Creates a new PhyscisModel with a new prior

        Args:
           prior(PhysicsModel): The prior

        Return:
           (PhysicsModel): A copy of `self` with the new prior

        """

        if self.prior is None:
            pass
        elif not isinstance(prior, type(self.prior)):
            raise TypeError("{} New prior must be the same type as the old".
                            format(self.get_inform(1)))
        # end
        return self._on_update_prior(prior)

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

        raise NotImplementedError('{} has not defined a co-variance matrix'
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

        raise NotImplementedError('{} has not defined a co-variance matrix'
                                  .format(self.get_inform(1)))

    def shape(self):
        """Returns the shape of the model dof space

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Return:
           (tuple): Dimensions

        """

        return NotImplemented

    def update_dof(self, dof_in):
        """Sets the model degrees of freedom

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
           dof_in(Iterable): The new values for *all* model degrees of freedom

        Return:
           (PhysicsModel):A copy of `self` with the new DOF values

        """

        return NotImplemented

    def _on_update_dof(self, model):
        """An extra method to perform special post-pocessing tasks when the DOF has
        been updated
        
        Args:
            model(PhysicsModel): The new physics model
        
        Return:
            (PhyscisModel): The post-processed model
        """

        return NotImplemented
        
    def get_dof(self):
        """Returns the model degrees of freedom

        .. note::

           Abstract Method: Must be overloaded to work with `F_UNCLE`

        Args:
           None

        Return:
           (np.ndarray): The model degrees of freedom
        """

        return NotImplemented

    def _on_update_prior(self, prior):
        """Instance specific prior update

        Args:
            prior(PhysicsModel): The prior

        Return:
            (PhysicsModel): A copy of `self` with the new prior
        """
        new_model = copy.deepcopy(self)
        new_model.prior = copy.deepcopy(prior)
        return new_model

    def get_log_like(self):
        """Returns the log likelyhood of the model
        """

        return NotImplemented

    def get_constraints(self):
        """Generates the constraints on the physics model
        """

        return NotImplemented

    def get_pq(self, scale=False):
        """Returns the p and q matricies for the model
        """

        return NotImplemented


class GaussianModel(PhysicsModel):
    """Generates model statistics assuming a gausian error
    """

    def get_log_like(self):
        """Returns the log liklyhood of the model, given the prior
        """
        return float(-0.5 * np.dot(np.dot(self.get_dof() - self.prior.get_dof(),
                                    inv(self.get_sigma())),
                             self.get_dof() - self.prior.get_dof()))


    def get_pq(self, scale=False):
        """Returns the P and q matrix components for the model

        Keyword Args:
            scale(bool): Flag to scale the model values
        """

        prior_scale = self.get_scaling()
        prior_var = inv(self.get_sigma())

        prior_delta = self.get_dof() - self.prior.get_dof()

        if scale:
            return np.dot(prior_scale, np.dot(prior_var, prior_scale)),\
                -np.dot(prior_scale, np.dot(prior_delta, prior_var))
        else:
            return prior_var, -np.dot(prior_delta, prior_var)
        # end



