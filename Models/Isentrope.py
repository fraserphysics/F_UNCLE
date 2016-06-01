#!/usr/bin/pyton
"""

pyIsentrope

Abstract class for an isentrope

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
import pdb
import copy
import unittest
import numpy.testing as npt

# =========================
# Python Packages
# =========================
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Utils.Struc import Struc
from F_UNCLE.Models.PhysicsModel import PhysicsModel
# =========================
# Main Code
# =========================


class Isentrope(PhysicsModel):
    """Abstract class for an isentrope
    """

    def __init__(self, name='Isentrope', *args, **kwargs):
        """

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            name(str): Name if the isentrope *Def 'Isentrope'*

        """

        def_opts = {
            'spline_N' : [int, 50, 7, None, '',
                          "Number of knots in the eos spline"],
            'spline_min': [float, 0.1, 0.0, None, 'cm**3/g',
                           "Minimum value of volume modeled by eos"],
            'spline_max': [float, 1.0, 0.0, None, 'cm**3/g',
                           "Maximum value of volume modeled by eos"],
            'spline_end' : [float, 4, 0, None, '',
                            "Number of zero nodes at end of spline"]
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        Struc.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)
        return
    # end

    def shape(self):
        """Overloaded class to get isentrope DOF's
        """

        return (self.get_option('spline_N'),1)
   
class Spline(IU_Spline, Struc):
    """Overloaded scipy spline to work with like_eos

    Child of the Scipy IU spline class which provides access to details on the
    knots

    """
       
    def get_t(self):
        """Gives the knot locations

        Return:
            (numpy.ndarray): knot locations

        """

        return self._eval_args[0]
    # end

    def get_c(self, spline_end = None):
        """Return the coefficients for the basis functions
        
        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline

        Return:
            (numpy.ndarray): basis function spline coefficients
        """
        if spline_end is None:
            spline_end = self.get_option('spline_end')
        elif not isinstance(spline_end, int):
            raise TypeError("Error in Spline: spline_end must be an integer")
        else:
            pass
        #end
        
        return self._eval_args[1][:-spline_end]

    def set_c(self, c_in, spline_end = None):
        """Updates the new spline with updated coefficients

        Return a new Spline_eos instance that is copy of self except
        that the coefficients for the basis functions are c.

        Args:
            c_in(numpy.ndarray): The new set of spline coefficeints

        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline
        
        Return
            None

        """
        
        if spline_end is None:
            spline_end = self.get_option('spline_end')
        elif not isinstance(spline_end, int):
            raise TypeError("Error in Spline: spline_end must be an integer")
        else:
            pass
        #end

        c_new = np.zeros(self._eval_args[1].shape)
        c_new[:-spline_end] = c_in
        self._eval_args =(self._eval_args[0],  c_new, self._eval_args[2])

        return None

    def get_basis(self, indep_vect, spline_end = None):
        """Returns the matrix of basis functions of the spline
        
        Args:
            indep_vect(np.ndarray): A vector of the independent variables over
                                    which the basis function should be
                                    calculated

        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline
        
        Return:
           (np.ndarray): The n x m matrix of basis functions where the n rows 
                         are the response over the independent variable vector
                         to a unit step in the m'th spline coefficient                
        """

        initial_c = self.get_c(spline_end=spline_end)
        tmp_c = np.zeros(initial_c.shape)
        basis = np.zeros((len(indep_vect), len(initial_c)))
        for j in range(len(tmp_c)):
            tmp_c[j] = 1.0
            self.set_c(tmp_c, spline_end=spline_end)
            basis[:, j] = self.__call__(indep_vect)
            tmp_c[j] = 0.0
        #end

        self.set_c(initial_c, spline_end = spline_end)
        
        return basis
        
class EOSBump(Isentrope):
    """Model of an ideal isentrope with gausian bumps

    """

    def __init__(self, name='Bump EOS', *args, **kwargs):
        """Instantiate the bump EOS

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            name(str): Name if the isentrope *Def 'Bump EOS'*

        """

        def_opts = {
            'const_C': [float, 2.56e9, 0.0, None, 'Pa',
                        "Constant p = C/v**3"],
            'bumps' : [list, [(0.35, 0.06, 0.2)], None, None, '',
                       "Gausian bumps to the eos"]
            }

        Isentrope.__init__(self, name, def_opts=def_opts, *args, **kwargs)

    def __call__(self, vs):
        """Solve the EOS

        Calculates the pressure for a given volume, replicates the EOS model
        but uses underlying equation rather than the spline

        Args:
            vs(float): Specific volume

        Return:
            pr(float): Pressure

        """
        C = self.get_option('const_C')
        bumps = self.get_option('bumps')
        pr = C/vs**3
        for v_0, w, s in bumps:
            z = (vs-v_0)/w
            pr += np.exp(-z*z/2)*s*C/(v_0**3)
        return pr
    # end

    def derivative(self, n=1):
        """Returns the nth order derrivative

        Keyword Args:
            n(int): The order of the derrivative. *Def 1*

        Retrun
            d1_fun(function): Function object yeilded first derrivative of
                pressure w.r.t volume
        """

        const_c = self.get_option('const_C')
        bumps = self.get_option('bumps')

        def d1_fun(v):
            """Derrivative function
            """
            rv = -3*const_c/v**4
            for v_0, w, s in bumps:
                z = (v-v_0)/w
                rv -= (z/w)*np.exp(-z*z/2)*s*const_c/(v_0**3)
            return rv
        # end

        if n == 1:
            return d1_fun
        else:
            raise IOError(
                '{:} bump EOS can only return fist derrivative function'.\
                format(self.get_inform(2)))
        # end




class EOSModel(Spline, Isentrope):
    """Spline based EOS model
    """

    def __init__(self, p_fun, name='Equation of State Spline', *args, **kwargs):
        """

        **Arguments**


        Args:
            p_fun(function): A function defining the initial EOS
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            name(str): Name of the isentrope *Def 'Equation of State Spline'*

        """

        def_opts = {
            'spline_sigma': [float, 5e-3, 0.0, None, '??',
                             "Multiplicative uncertainty of the prior (1/2%)"],
            'precondition' : [bool, False, None, None, '',
                              "Precondition flag"]
            }

        Isentrope.__init__(self, name, def_opts=def_opts, *args, **kwargs)

        # Update the prior of this PhysicsModel with the spline generated from
        # the nominal EOS given in __init__
        if not hasattr(p_fun, '__call__'):
            raise TypeError("{:} the initial eos estimate must be a function".\
                            format(self.get_inform(0)))
        #end
        
        v = np.logspace(np.log10(self.get_option('spline_min')),
                        np.log10(self.get_option('spline_max')),
                        self.get_option('spline_N')
                       )
        IU_Spline.__init__(self, v, p_fun(v))
        
        self.prior = copy.deepcopy(self)
        
    def get_sigma(self):
        """Returns the covariance matrix of the spline
        
        Return:
           (np.ndarray): Covariance matrix for the eos
                         shape is (nxn) where n is the dof
        """

        sigma = self.get_option('spline_sigma')

        return np.diag(sigma * self.get_c())
        
    def update_prior(self, prior, *args, **kwargs):
        """

        Updated the values and statistics of the prior

        ** Arguments **

        - prior -> function: A function which defines the prior EOS shape

        """
        
        if isinstance(prior, EOSModel):
            self.prior = prior
        else:
            raise TypeError("{:} Model required an EOSModel type prior"\
                           .format(self.get_inform(1)))
        #end

    def get_dof(self):
        """Returns the spline coefficients as the model degrees of fredom
        """
        return self.get_c()

    def set_dof(self, c_in):
        """Sets the spline coeffecients
        
        Args:
           c_in(Iterable): The knot positions of the spline
        """

        self.set_c(c_in)
        
class TestIsentrope(unittest.TestCase):
    """Test of the isentrope object
    """
    def test_standard_instantiation(self):
        """Test basic use of isentrope
        """
        model = Isentrope()

        self.assertEqual(model.get_option('spline_N'), 50)
        self.assertEqual(model.get_option('spline_min'), 0.1)
        self.assertEqual(model.get_option('spline_max'), 100)
        self.assertEqual(model.get_option('spline_end'), 4)
    # end

    def test_custom_instantiation(self):
        """Test instantiation with non default values
        """
        model = Isentrope(name="Francis", prior=3.5, spline_N=55,
                          spline_sigma=0.0)

        self.assertEqual(model.name, "Francis")
        self.assertEqual(model.get_option('spline_N'), 55)


# end

class test_eos_model(unittest.TestCase):
    """Test the spline EOS functions
    """
    def setUp(self):
        """Create test eos function
        """

        p_fun = lambda v: 2.56e9 / v**3

        self.p_fun = p_fun

    def test_standard_instantiation(self):
        """Test normal instantiation of the EOS
        """

        eos = EOSModel(self.p_fun)

        print eos
        
        self.assertEqual(eos.get_option('spline_sigma'), 5e-3)
        self.assertEqual(eos.get_option('precondition'), False)
    # end

    def test_bad_instantiation(self):
        """Test inproper instantiation
        """

        with self.assertRaises(TypeError):
            EOSModel(2.5)
        # end

    # end
    def test_custom_instantiation(self):
        """Tets instantiation with non default values
        """
        eos = EOSModel(self.p_fun, name="Ajax", spline_N=65,
                       spline_sigma=2.5e-3)

        self.assertEqual(eos.name, "Ajax")
        self.assertEqual(eos.get_option('spline_N'), 65)
        self.assertEqual(eos.get_option('spline_sigma'), 2.5e-3)

    def test_spline_get_t(self):
        """Test spline interaction method, get knots
        """

        # generate an eos spline
        eos = EOSModel(self.p_fun)

        # get t
        t_list = eos.get_t()

        # check it is a numpy array

        self.assertIsInstance(t_list, np.ndarray)


        # check it is long enough
        self.assertEqual(
            eos.get_option('spline_N')+eos.get_option('spline_end'),
            len(t_list)
        )

        # check is spans the v min and v max limits
        self.assertEqual(eos.get_option('spline_min'), t_list.min())
        self.assertEqual(eos.get_option('spline_max'), t_list.max())

    def test_spline_get_c(self):
        """Test spline interaction method, get coefficients
        """
        # generate an eos spline
        eos = EOSModel(self.p_fun)

        # get c
        c_list = eos.get_c()

        self.assertEqual(eos.get_option('spline_N'), len(c_list))

    def test_spline_new_c(self):
        """Test spline interaction method, set new coefficients
        """

        eos = EOSModel(self.p_fun)

        # get c
        c_list = eos.get_c()

        # add 0.1 to all c
        # get a new spline with updated c
        eos_update = eos.new_c(c_list + 0.1)

        # get its c
        c_update = eos_update.get_c()

        # check it matches
        npt.assert_array_equal(c_update, c_list + 0.1)

        # check the original eos is unchanged
        npt.assert_array_equal(c_list, eos.get_c())

    def test_eos_get_sigma(self):
        """ Tests that the covariance matrix is generated correctly       
        """

        eos = EOSModel(self.p_fun)

        sigma_eos = eos.get_sigma()

        n_spline = eos.get_option('spline_N')
        spline_var = eos.get_option('spline_sigma')

        self.assertEqual(sigma_eos.shape, (n_spline, n_spline))

        npt.assert_array_equal(np.diag(sigma_eos), eos.get_c() * spline_var)
        
class TestBumpEOS(unittest.TestCase):
    """Test of the bump EOS
    """
    def test_instantiation(self):
        """Tests that the object is properly instantiated
        """
        bump_eos = EOSBump()

        self.assertEqual(bump_eos.get_option('const_C'), 5e-3)
        self.assertEqual(bump_eos.get_option('bumps')[0][0], 0.35)
        self.assertEqual(bump_eos.get_option('bumps')[0][1], 0.06)
        self.assertEqual(bump_eos.get_option('bumps')[0][2], 0.2)

    def test_custom_instatntiation(self):
        """Test non default instantiation
        """
        bump_eos = EOSBump(const_C=6e-3,
                           bumps=[(0.25, 0.07, 0.3),
                                  (0.35, 0.08, 0.4),
                                 ])

        bump_eos(0.25)

        self.assertEqual(bump_eos.get_option('const_C'), 6e-3)
        self.assertEqual(bump_eos.get_option('bumps')[0][0], 0.25)
        self.assertEqual(bump_eos.get_option('bumps')[0][1], 0.07)
        self.assertEqual(bump_eos.get_option('bumps')[0][2], 0.3)
        self.assertEqual(bump_eos.get_option('bumps')[1][0], 0.35)
        self.assertEqual(bump_eos.get_option('bumps')[1][1], 0.08)
        self.assertEqual(bump_eos.get_option('bumps')[1][2], 0.4)

    def test_bad_instatntiation(self):
        """Test improper instantiation
        """
        pass

    def test_call(self):
        """Test that the bump EOS can be called
        """
        bump_eos = EOSBump()

        v = np.logspace(np.log10(001), np.log10(100), 50)

        p = bump_eos(v)

        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(50, len(p))

    def test_derivative(self):
        """Test the derrivative function
        """
        bump_eos = EOSBump()

        v = np.logspace(np.log10(001), np.log10(100), 50)

        p = bump_eos.derivative()(v)

        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(50, len(p))

    def test_bad_derivative(self):
        """Tests that derrivative errors are caught
        """
        bump_eos = EOSBump()

        v = np.logspace(np.log10(001), np.log10(100), 50)

        with self.assertRaises(IOError):
            p = bump_eos.derivative(n = 2)(v)
        # end

if __name__ == '__main__':
    unittest.main(verbosity = 4)