#!/usr/bin/pyton
"""

Isentrope

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
import copy
import unittest
import math
import sys
import os

# =========================
# Python Packages
# =========================
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================
sys.path.append(os.path.abspath('./../../'))
from F_UNCLE.Utils.PhysicsModel import PhysicsModel


# =========================
# Main Code
# =========================


class Isentrope(PhysicsModel):
    """Abstract class for an isentrope

    The equation of state for an isentropic expansion of high explosive is
    modeled by this class. The experiments for which this model is used
    occur at such short timescales that the process can be considered
    adiabatic

    **Units**

    Isentropes are assumed to be in CGS units

    **Diagram**

    .. figure:: /_static/eos.png

       The assumed shape of the equation of state isentrope

    **Options**

    +------------+-------+-----+-----+-----+---------+--------------------------+
    |    Name    |Type   |Def  |Min  |Max  |Units    |Description               |
    +============+=======+=====+=====+=====+=========+==========================+
    |`spline_N`  |(int)  |50   |7    |None |''       |Number of knots in the EOS|
    |            |       |     |     |     |         |spline                    |
    +------------+-------+-----+-----+-----+---------+--------------------------+
    |`spline_min`|(float)|0.1  |0.0  |None |'cm**3/g'|Minimum value of volume   |
    |            |       |     |     |     |         |modeled by EOS            |
    +------------+-------+-----+-----+-----+---------+--------------------------+
    |`spline_max`|(float)|1.0  |0.0  |None |'cm**3/g'|Maximum value of volume   |
    |            |       |     |     |     |         |modeled by EOS            |
    +------------+-------+-----+-----+-----+---------+--------------------------+
    |`spline_end`|(float)|4    |0    |None |''       |Number of zero nodes at   |
    |            |       |     |     |     |         |end of spline             |
    +------------+-------+-----+-----+-----+---------+--------------------------+

    """

    def __init__(self, name='Isentrope', *args, **kwargs):
        """

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            name(str): Name if the isentrope *Def 'Isentrope'*

        Return:
            None
        """

        def_opts = {
            'spline_N' : [int, 50, 7, None, '',
                          "Number of knots in the EOS spline"],
            'spline_min': [float, 0.1, 0.0, None, 'cm**3/g',
                           "Minimum value of volume modeled by EOS"],
            'spline_max': [float, 1.0, 0.0, None, 'cm**3/g',
                           "Maximum value of volume modeled by EOS"],
            'spline_end' : [float, 4, 0, None, '',
                            "Number of zero nodes at end of spline"]
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        PhysicsModel.__init__(self, None, name=name, def_opts=def_opts, *args, **kwargs)

    def shape(self):
        """Overloaded class to get isentrope DOF's

        Overloads :py:meth:`F_UNCLE.Utils.PhysModel.PhysModel.shape`

        Return:
            (tuple): (n,1) where n is the number of dofs
        """

        return (self.get_option('spline_N'), 1)

    def plot(self, axis=None, hardcopy=None, style = '-k', *args, **kwargs):
        """Plots the EOS

        Overloads the :py:meth:`F_UNCLE.Utils.Struc.Struc.plot` method to plot
        the eos over the range of volumes.

        Args:
            axis(plt.Axes): The axis on which to plot the figure, if None,
                creates a new figure object on which to plot.
            hard-copy(str): If a string, write the figure to the file specified
            style(str): A :py:meth:`plt.Axis.plot` format string for the eos
        Return:
            (plt.Figure): A reference to the figure containing the plot

        """

        if axis is None:
            fig = plt.figure()
            ax1 = fig.gca()
        elif isinstance(axis, plt.Axes):
            fig = None
            ax1 = axis
        else:
            raise TypeError("{} axis must be a matplotlib Axis object".\
                            format(self.get_inform(1)))
        #end

        # v_spec = np.logspace(np.log10(self.get_option('spline_min')),\
        #                 np.log10(self.get_option('spline_max')),\
        #                 50)
        v_spec = np.linspace(0.2,
                             0.6,
                             200)        
        ax1.plot(v_spec, self(v_spec), style, *args, **kwargs)
        ax1.set_xlabel(r'Specific volume / cm$^3$ g$^{-1}$')
        ax1.set_ylabel('Pressure / Pa')

        return fig


class Spline(IU_Spline):
    """Overloaded scipy spline to work as a PhysicsModel

    Child of the Scipy IU spline class which provides access to details to
    the knots which are treated as degrees of freedom

    """

    def get_t(self):
        """Gives the knot locations

        Return:
            (np.ndarray): knot locations

        """

        return self._eval_args[0]
    # end

    def get_c(self, spline_end=None):
        """Return the coefficients for the basis functions

        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline

        Return:
            (np.ndarray): basis function spline coefficients
        """
        if spline_end is None:
            spline_end = self.get_option('spline_end')
        elif not isinstance(spline_end, int):
            raise TypeError("Error in Spline: spline_end must be an integer")
        else:
            pass
        #end

        return copy.deepcopy(self._eval_args[1][:-spline_end])

    def set_c(self, c_in, spline_end=None):
        """Updates the new spline with updated coefficients

        Sets the spline coefficients of this instance to the given values

        Args:
            c_in(np.ndarray): The new set of spline coefficients

        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline

        Returns:
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
        self._eval_args = (self._eval_args[0], c_new, self._eval_args[2])

        return None

    def get_basis(self, indep_vect, spline_end=None):
        """Returns the matrix of basis functions of the spline

        Args:
            indep_vect(np.ndarray): A vector of the independent variables over
                which the basis function should be calculated

        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline

        Return:
            (np.ndarray):
                The n x m matrix of basis functions where the n rows
                are the response over the independent variable vector to a unit
                step in the m'th spline coefficient
        """

        initial_c = self.get_c(spline_end=spline_end)
        tmp_c = np.zeros(initial_c.shape)
        basis = np.zeros((len(initial_c), len(indep_vect)))
        for j in range(len(tmp_c)):
            tmp_c[j] = 1.0
            self.set_c(tmp_c, spline_end=spline_end)
            basis[j, :] = self.__call__(indep_vect)
            tmp_c[j] = 0.0
        #end

        self.set_c(initial_c, spline_end=spline_end)

        return basis

class EOSBump(Isentrope):
    """Model of an ideal isentrope with Gaussian bumps

    This is treated as the *true* EOS
    
    **Options**

    +---------+-------+----------------+-----+-----+-----+----------------------+
    |Name     |Type   |Def             |Min  |Max  |Units|Description           |
    +---------+-------+----------------+-----+-----+-----+----------------------+
    |`const_C`|(float)|2.56e9          |0.0  |None |'Pa' |'Constant p = C/v**3' |
    |         |       |                |     |     |     |                      |
    +---------+-------+----------------+-----+-----+-----+----------------------+
    |`bumps`  |(list) |[(0.4 0.1 0.25) |None |None |''   |'Gaussian bumps to the|
    |         |       |                |     |     |     |EOS'                  |
    |         |       |(0.5 0.1 -0.3)] |     |     |     |                      |
    +---------+-------+----------------+-----+-----+-----+----------------------+
    
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
            'bumps' : [list, [(0.4, 0.1, 0.25),
                              (0.5, 0.1, -0.3)], None, None, '',
                       "Gaussian bumps to the EOS"]
            }

        Isentrope.__init__(self, name, def_opts=def_opts, *args, **kwargs)

    def __call__(self, vol):
        """Solve the EOS

        Calculates the pressure for a given volume, is called the same way as
        the EOS model but uses underlying equation rather than the spline

        Args:
            vol(np.ndarray): Specific volume

        Return:
            pr(np.ndarray): Pressure

        """
        const_c = self.get_option('const_C')
        bumps = self.get_option('bumps')
        pressure = const_c/vol**(3.0)
        for v_0, width, scale in bumps:
            center = (vol-v_0)/width
            pressure += np.exp(-center*center/2)*scale*const_c/(v_0**3)
        return pressure
    # end

    def derivative(self, order=1):
        """Returns the nth order derivative

        Keyword Args:
            order(int): The order of the derivative. *Def 1*

        Return:
            d1_fun(function):
                Function object yielded first derivative of pressure w.r.t volume
        """

        const_c = self.get_option('const_C')
        bumps = self.get_option('bumps')

        def d1_fun(vol):
            """Derivative function
            """
            derr1 = -3*const_c/vol**4
            for v_0, width, scale in bumps:
                center = (vol-v_0)/width
                derr1 -= (center/width)*np.exp(-center*center/2)*scale*const_c/(v_0**3)
            return derr1
        # end

        if order == 1:
            return d1_fun
        else:
            raise IOError(
                '{:} bump EOS can only return fist derivative function'.\
                format(self.get_inform(2)))
        # end


class EOSModel(Spline, Isentrope):
    """Spline based EOS model

    Multiply inherited structure from both `Isentrope` and `Spline`

    **Options**

    +--------------+---------+------+-----+-----+-----+-------------------------+
    |Name          |Type     |Def   |Min  |Max  |Units|Description              |
    +==============+=========+======+=====+=====+=====+=========================+
    |`Spline_sigma`|float    |5e-3  |0.0  |None |'??' |'Multiplicative          |
    |              |         |      |     |     |     |uncertainty of the prior |
    |              |         |      |     |     |     |(1/2%)                   |
    +--------------+---------+------+-----+-----+-----+-------------------------+
    |`precondition`|bool     |False |None |None |''   |Precondition flag        |
    +--------------+---------+------+-----+-----+-----+-------------------------+

    """

    def __init__(self, p_fun, name='Equation of State Spline', *args, **kwargs):
        """Instantiates the object

        Args:
            p_fun(function): A function defining the prior EOS
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
            raise TypeError("{:} the initial EOS estimate must be a function".\
                            format(self.get_inform(0)))
        #end

        vol = np.logspace(np.log10(self.get_option('spline_min')),
                          np.log10(self.get_option('spline_max')),
                          self.get_option('spline_N'))
        Spline.__init__(self, vol, p_fun(vol))

        self.prior = copy.deepcopy(self)

    def get_scaling(self):
        """Returns a scaling matrix to make the dofs of the same scale

        The scaling matrix is a diagonal matrix with off diagonal terms zero
        the terms along the diagonal are the prior DOFs times the variance in
        the DOF values.

        Return:
            (np.ndarray): A nxn matrix where n is the number of model DOFs.

        """
        dev = self.prior.get_dof() * self.get_option('spline_sigma')
        return np.diag(dev)

    def _on_str(self):
        """Addeed information on the EOS
        """

        dof = self.get_dof()
        out_str="\n\n"
        out_str+="Degrees of Freedom\n"
        out_str+="==================\n\n"
        k = 8
        for i in xrange(int(math.ceil(len(dof)*1.0/k))):
            for j in xrange(min(8, len(dof)-k*i)):
                out_str += "{: 3.2e} ".format(dof[k*i+j])
            #end
            out_str += '\n'
        #end

        return out_str
            

    def get_sigma(self):
        """Returns the co-variance matrix of the spline

        Return:
            (np.ndarray):
                Co-variance matrix for the EOS shape is (nxn) where n is the dof
                of the model
        """

        sigma = self.get_option('spline_sigma')

        return np.diag((sigma * self.get_c())**2)

    def update_prior(self, prior, *args, **kwargs):
        """

        Updated the prior

        Args:
            prior(EOSModel): A function which defines the prior EOS shape
        """

        if prior is None:
            pass
        elif isinstance(prior, EOSModel):
            self.prior = prior
        else:
            raise TypeError("{:} Model required an EOSModel type prior"\
                           .format(self.get_inform(1)))
        #end

    def get_dof(self, *args, **kwargs):
        """Returns the spline coefficients as the model degrees of freedom

        Return:
            (np.ndarray): The degrees of freedom of the model
        """
        return self.get_c()

    def set_dof(self, c_in, *args, **kwargs):
        """Sets the spline coefficients

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
        self.assertEqual(model.get_option('spline_max'), 1.0)
        self.assertEqual(model.get_option('spline_end'), 4)
    # end

    def test_custom_instantiation(self):
        """Test instantiated with non default values
        """
        model = Isentrope(name="Francis",  spline_N=55,
                          spline_sigma=0.0)

        self.assertEqual(model.name, "Francis")
        self.assertEqual(model.get_option('spline_N'), 55)


# end

class TestEosModel(unittest.TestCase):
    """Test the spline EOS functions
    """
    def setUp(self):
        """Create test EOS function
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
        """Test improper instantiation
        """

        with self.assertRaises(TypeError):
            EOSModel(2.5)
        # end

    # end
    def test_custom_instantiation(self):
        """Tests instantiation with non default values
        """
        eos = EOSModel(self.p_fun, name="Ajax", spline_N=65,
                       spline_sigma=2.5e-3)

        self.assertEqual(eos.name, "Ajax")
        self.assertEqual(eos.get_option('spline_N'), 65)
        self.assertEqual(eos.get_option('spline_sigma'), 2.5e-3)

    def test_spline_get_t(self):
        """Test spline interaction method, get knots
        """

        # generate an EOS spline
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
        # generate an EOS spline
        eos = EOSModel(self.p_fun)

        # get c
        c_list = eos.get_c()

        self.assertEqual(eos.get_option('spline_N'), len(c_list))

    def test_spline_new_c(self):
        """Test spline interaction method, set new coefficients
        """

        eos = EOSModel(self.p_fun)

        # get c
        c_list = eos.get_dof()

        # add 0.1 to all c
        # get a new spline with updated c
        eos.set_dof(c_list + 0.1)

        # get its c
        c_update = eos.get_dof()

        # check it matches
        npt.assert_array_equal(c_update, c_list + 0.1)

    def test_eos_get_sigma(self):
        """ Tests that the co-variance matrix is generated correctly
        """

        eos = EOSModel(self.p_fun)

        sigma_eos = eos.get_sigma()

        n_spline = eos.get_option('spline_N')
        spline_var = eos.get_option('spline_sigma')

        self.assertEqual(sigma_eos.shape, (n_spline, n_spline))
        npt.assert_array_equal(np.diag(sigma_eos), (eos.get_c() * spline_var)**2)

class TestBumpEOS(unittest.TestCase):
    """Test of the bump EOS
    """
    def test_instantiation(self):
        """Tests that the object is properly instantiated
        """
        bump_eos = EOSBump()

        self.assertEqual(bump_eos.get_option('const_C'), 2.56e9)
        self.assertEqual(bump_eos.get_option('bumps')[0][0], 0.2)
        self.assertEqual(bump_eos.get_option('bumps')[0][1], 0.1)
        self.assertEqual(bump_eos.get_option('bumps')[0][2], 0.4)

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

        vol = np.logspace(np.log10(001), np.log10(100), 50)

        pressure = bump_eos(vol)

        self.assertIsInstance(pressure, np.ndarray)
        self.assertEqual(50, len(pressure))

    def test_derivative(self):
        """Test the derivative function
        """
        bump_eos = EOSBump()

        vol = np.logspace(np.log10(001), np.log10(100), 50)

        pressure = bump_eos.derivative()(vol)

        self.assertIsInstance(pressure, np.ndarray)
        self.assertEqual(50, len(pressure))

    def test_bad_derivative(self):
        """Tests that derivative errors are caught
        """
        bump_eos = EOSBump()

        vol = np.logspace(np.log10(001), np.log10(100), 50)

        with self.assertRaises(IOError):
            pressure = bump_eos.derivative(order=2)(vol)
        # end

if __name__ == '__main__':
    unittest.main(verbosity=4)
