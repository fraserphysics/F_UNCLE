#!/usr/bin/pyton
"""

Isentrope

Abstract class for an isentrope

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraser (AMF)

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
import copy
import math
import sys
import os

# =========================
# Python Packages
# =========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IU_Spline
from scipy.optimize import brentq
# For scipy.interpolate.InterpolatedUnivariateSpline. See:
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/fitpack2.py


# =========================
# Custom Packages
# =========================
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.PhysicsModel import GausianModel
else:
    from ..Utils.PhysicsModel import GausianModel


# =========================
# Main Code
# =========================

class Isentrope(GausianModel):
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

    def __init__(self, name=u'Isentrope', *args, **kwargs):
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
            'spline_N': [int, 50, 7, None, '',
                         "Number of knots in the EOS spline"],
            'spline_min': [float, 0.1, 0.0, None, 'cm**3/g',
                           "Minimum value of volume modeled by EOS"],
            'spline_max': [float, 1.0, 0.0, None, 'cm**3/g',
                           "Maximum value of volume modeled by EOS"],
            'spline_end': [float, 4, 0, None, '',
                           "Number of zero nodes at end of spline"],
            'basis': [str, 'volume', None, None, '',
                      "The basis of the isentrope, can be `volume` or"
                      " `density`"],
            'vcj_lower': [float, 1.0E5, 0.0, None, 'cm s-1',
                          "Lower bound on the CJ velocity used in"
                          "bracketing search"],
            'vcj_upper':[float, 10.0E5, 0.0, None, 'cm s-1',
                         "Upper bound on the CJ velocity used in"
                         " bracketing search"]
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        GausianModel.__init__(self, None, name=name, def_opts=def_opts,
                              *args, **kwargs)

    def shape(self):
        """Overloaded class to get isentrope DOF's

        Overloads :py:meth:`F_UNCLE.Utils.PhysModel.PhysModel.shape`

        Return:
            (tuple): (n,1) where n is the number of dofs
        """

        return self.get_option('spline_N')

    def _get_cj_point(self, vol_0, pres_0=0.0):
        """Find CJ conditions using two nested line searches.

        The CJ point is the location on the EOS where a Rayleigh line
        originating at the pre-detonation volume and pressure is tangent to
        the equation of state isentrope.

        This method uses two nested line searches implemented by the
        :py:meth:`scipy.optimize.brentq` algorithm to locate the velocity
        corresponding to this tangent Rayleigh line

        Args:
            vol_0(float): The specific volume of the equation of state before
                the shock arrives

        Keyword Args:
            pres_0(float): The pressure of the reactants

        Return:
            (tuple): Length 3 elements are:

                0.  (float): The detonation velocity
                1.  (float): The specific volume at the CJ point
                2.  (float): The pressure at the CJ point
                3.  (function): A function defining the Rayleigh line which
                    passes through the CJ point

        """
        eos = self
        # Search for det velocity between 1 and 10 km/sec
        d_min = eos.get_option('vcj_lower')  # cm s**-1
        d_max = eos.get_option('vcj_upper')  # cm s**-1
        if eos.get_option('basis').lower()[:3] == 'vol':
            v_min = eos.get_option('spline_min')
            v_max = eos.get_option('spline_max')
        elif eos.get_option('basis').lower()[:3] == 'den':
            v_min = eos.get_option('spline_max')**-1
            v_max = eos.get_option('spline_min')**-1
        else:
            raise IOError('Unknown Isentrope basis, must be `vol` or `den`'
                          .format(self.get_inform(1)))

        # R is Rayleigh line
        def rayl_line(vel, vol, eos, vol_0):
            return pres_0 + (vel**2) * (vol_0 - vol) / (vol_0**2)

        if eos.get_option('basis').lower()[:3] == 'vol':
            # F is self - R
            def rayl_err(vel, vol, eos, vol_0):
                return eos(vol) - rayl_line(vel, vol, eos, vol_0)

            # d_F is derivative of F wrt vol
            def derr_dvol(vol, vel, eos, vol_0):
                return eos.derivative(1)(vol) + (vel / vol_0)**2

        else:
            # Density based EOS
            def rayl_err(vel, vol, eos, vol_0):
                return eos(vol**-1) - rayl_line(vel, vol, eos, vol_0)

            # d_F is derivative of F wrt vol
            def derr_dvol(vol, vel, eos, vol_0):
                return -eos.derivative(1)(vol**-1) + (vol * vel / vol_0)**2
        # end
        # arg_min(vel, self) finds volume that minimizes self(v) - R(v)

        def arg_min(vel, eos, vol_0):
            return brentq(derr_dvol, v_min, v_max,
                          args=(vel, eos, vol_0))

        def error(vel, eos, vol_0):
            return rayl_err(vel, arg_min(vel, eos, vol_0),
                            eos, vol_0)

        try:
            vel_cj = brentq(error, d_min, d_max, args=(eos, vol_0))
            vol_cj = arg_min(vel_cj, eos, vol_0)

            if self.get_option('basis').lower()[:3] == 'vol':
                p_cj = eos(vol_cj)
            else:
                p_cj = eos(vol_cj**-1)
            #end

        except:
            import pdb
            pdb.set_trace()


        return vel_cj, vol_cj, float(p_cj), rayl_line

    def plot(self, axes=None, figure=None, linestyles=['-k'],
             labels=['Isentrope'], *args, **kwargs):
        """Plots the EOS

        Overloads the :py:meth:`F_UNCLE.Utils.Struc.Struc.plot` method to plot
        the eos over the range of volumes.

        Args:
            axes(plt.Axes): The axes on which to plot the figure, if None,
                creates a new figure object on which to plot.
            figure(plt.Figure): The figure on which to plot *ignored*
            linstyles(list): Strings for the linestyles
                0. '-k', The Isentrope
            labels(list): Strings for the plot labels
                0. 'Isentrope'

        Return:
            (plt.Figure): A reference to the figure containing the plot

        """

        if axes is None:
            fig = plt.figure()
            ax1 = fig.gca()
        elif isinstance(axes, plt.Axes):
            fig = None
            ax1 = axes
        else:
            raise TypeError("{} axis must be a matplotlib Axis object"
                            .format(self.get_inform(1)))
        # end

        # v_spec = np.logspace(np.log10(self.get_option('spline_min')),\
        #                 np.log10(self.get_option('spline_max')),\
        #                 50)
        v_spec = np.linspace(self.get_option('spline_min'),
                             self.get_option('spline_max'),
                             200)
        ax1.plot(v_spec, self(v_spec), linestyles[0], label=labels[0])
        ax1.set_xlabel(r'Specific volume / cm$^3$g^{-1}')
        ax1.set_ylabel(r'Pressure / Pa')
        # ax1.set_xlabel(r'Specific volume / $\si{\cubic\centi\meter\per\gram}$')
        # ax1.set_ylabel(r'Pressure / $\si{\pascall}$')

        return fig

    def get_constraints(self, scale=False):
        """Returns the G and h matricies corresponding to the model

        Args:
           model(GausianModel): The physics model subject to
                                physical constraints
        Return:
           ():
           ():

        Method

        Calculate constraint matrix G and vector h.  The
        constraint enforced by :py:class:`cvxopt.solvers.qp` is

        .. math::

           G*x \leq  h

        Equivalent to :math:`max(G*x - h) \leq 0`

        Since

        .. math::

           c_{f_{new}} = c_f+x,

        .. math::

           G(c_f+x) \leq 0

        is the same as

        .. math::

           G*x \leq -G*c_f,

        and :math:`h = -G*c_f`

        Here are the constraints for :math:`p(v)`:

        p'' positive for all v
        p' negative for v_max
        p positive for v_max

        For cubic splines between knots, f'' is constant and f' is
        affine.  Consequently, :math:`f''rho + 2f'` is affine between knots
        and it is sufficient to check eq:star at the knots.

        """

        c_model = copy.deepcopy(self)
        spline_end = c_model.get_option('spline_end')
        dim = c_model.shape()

        v_unique = c_model.get_t()[spline_end - 1:1 - spline_end]
        n_constraints = len(v_unique) + 2

        G_mat = np.zeros((n_constraints, dim))
        c_tmp = np.zeros(dim)
        c_init = c_model.get_dof()
        scaling = c_model.get_scaling()
        for i in range(dim):
            c_tmp[i] = 1.0
            mod_tmp = c_model.update_dof(c_tmp)
            G_mat[:-2, i] = -mod_tmp.derivative(2)(v_unique)
            G_mat[-2, i] = mod_tmp.derivative(1)(v_unique[-1])
            G_mat[-1, i] = -mod_tmp(v_unique[-1])
            c_tmp[i] = 0.0
        # end

        h_vec = -np.dot(G_mat, c_init)

        hscale = np.abs(h_vec)
        hscale = np.maximum(hscale, hscale.max() * 1e-15)
        # Scale to make |h[i]| = 1 for all i
        HI = np.diag(1.0 / hscale)
        h_vec = np.dot(HI, h_vec)
        G_mat = np.dot(HI, G_mat)

        if scale:
            G_mat = np.dot(G_mat, scaling)
        # end

        return G_mat, h_vec


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
        # end

        return copy.deepcopy(self._eval_args[1][:-spline_end])

    def set_c(self, c_in, spline_end=None):
        """Updates the new spline with updated coefficients

        Sets the spline coefficients of this instance to the given values

        Args:
            c_in(np.ndarray): The new set of spline coefficients

        Keyword Args:
           spline_end(int): The number of fixed nodes at the end of the spline

        Returns:
            (Spline): A copy of `self` with the new coefficients

        """

        if spline_end is None:
            spline_end = self.get_option('spline_end')
        elif not isinstance(spline_end, int):
            raise TypeError("Error in Spline: spline_end must be an integer")
        else:
            pass
        # end

        c_new = np.zeros(self._eval_args[1].shape)
        c_new[:-spline_end] = c_in
        new_spline = copy.deepcopy(self)
        new_spline._eval_args = (new_spline._eval_args[0],
                                 c_new,
                                 new_spline._eval_args[2])
        return new_spline

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
            tmp_spline = self.set_c(tmp_c, spline_end=spline_end)
            basis[j, :] = tmp_spline.__call__(indep_vect)
            tmp_c[j] = 0.0
        # end

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

    def __init__(self, name=u'Bump EOS', *args, **kwargs):
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
            'bumps': [list, [(0.4, 0.1, 0.25),
                             (0.5, 0.1, -0.3)], None, None, '',
                      "Gaussian bumps to the EOS"]
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

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
        pressure = const_c / vol**(3.0)
        for v_0, width, scale in bumps:
            center = (vol - v_0) / width
            pressure += np.exp(-center * center / 2) * scale * const_c /\
                (v_0**3)
        return pressure
    # end

    def derivative(self, order=1):
        """Returns the nth order derivative

        Keyword Args:
            order(int): The order of the derivative. *Def 1*

        Return:
            d1_fun(function):
                Function object yielded first derivative of pressure w.r.t
                volume
        """

        const_c = self.get_option('const_C')
        bumps = self.get_option('bumps')

        def d1_fun(vol):
            """Derivative function
            """
            derr1 = -3 * const_c / vol**4
            for v_0, width, scale in bumps:
                center = (vol - v_0) / width
                derr1 -= (center / width) * np.exp(-center * center / 2)\
                    * scale * const_c / (v_0**3)
            return derr1
        # end

        if order == 1:
            return d1_fun
        else:
            raise IOError(
                '{:} bump EOS can only return fist derivative function'
                .format(self.get_inform(2)))
        # end


class EOSModel(Spline, Isentrope):
    """Spline based EOS model

    Multiply inherited structure from both `Isentrope` and `Spline`

    **Options**

   +--------------+---------+------+-----+-----+-----+-------------------------+
   |Name          |Type     |Def   |Min  |Max  |Units|Description              |
   +==============+=========+======+=====+=====+=====+=========================+
   |`spline_sigma`|float    |5e-3  |0.0  |None |'??' |'Multiplicative          |
   |              |         |      |     |     |     |uncertainty of the prior |
   |              |         |      |     |     |     |(1/2%)                   |
   +--------------+---------+------+-----+-----+-----+-------------------------+
   |`precondition`|bool     |False |None |None |''   |Precondition flag        |
   +--------------+---------+------+-----+-----+-----+-------------------------+
    """

    def __init__(self, p_fun, name=u'Equation of State Spline',
                 *args, **kwargs):
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
            'precondition': [bool, False, None, None, '',
                             "Precondition flag"]
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        Isentrope.__init__(self, name, def_opts=def_opts, *args, **kwargs)

        # Update the prior of this GausianModel with the spline generated from
        # the nominal EOS given in __init__
        if not hasattr(p_fun, '__call__'):
            raise TypeError("{:} the initial EOS estimate must be a function"
                            .format(self.get_inform(0)))
        # end

        vol = np.logspace(np.log10(self.get_option('spline_min')),
                          np.log10(self.get_option('spline_max')),
                          self.get_option('spline_N'))

        # vol = np.linspace(self.get_option('spline_min'),
        #           self.get_option('spline_max'),
        #           self.get_option('spline_N'))

        Spline.__init__(self, vol, p_fun(vol))
        self.prior = copy.deepcopy(self)
        self = self._on_update_dof(self)


    def get_scaling(self):
        """Returns a scaling matrix to make the dofs of the same scale

        The scaling matrix is a diagonal matrix with off diagonal terms zero
        the terms along the diagonal are the prior DOFs times the variance in
        the DOF values.

        Return:
            (np.ndarray): A nxn matrix where n is the number of model DOFs.

        """
        dev = self.prior.get_dof()  # * self.get_option('spline_sigma')
        return np.diag(dev)

    def _on_str(self):
        """Addeed information on the EOS
        """

        dof = self.get_dof()
        out_str = "\n\n"
        out_str += "Degrees of Freedom\n"
        out_str += "==================\n\n"
        k = 8
        for i in range(int(math.ceil(len(dof) * 1.0 / k))):
            for j in range(min(8, len(dof) - k * i)):
                out_str += "{: 3.2e} ".format(dof[k * i + j])
            # end
            out_str += '\n'
        # end

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

    def get_dof(self, *args, **kwargs):
        """Returns the spline coefficients as the model degrees of freedom

        Return:
            (np.ndarray): The degrees of freedom of the model
        """
        return self.get_c()

    def update_dof(self, c_in, *args, **kwargs):
        """Sets the spline coefficients

        Args:
           c_in(Iterable): The knot positions of the spline

        Return:
           (EOSModel): A copy of `self` with the new dofs

        """

        return self._on_update_dof(self.set_c(c_in))

    def _on_update_dof(self, model):
        """An extra method to perform special post-pocessing tasks when the DOF
        has been updated

        Args:
            model(EOSModel): The new physics model

        Return:
            (EOSModel): The post-processed model
        """

        return model

    def plot_diff(self, isentropes,
                  axes=None,
                  figure=None,
                  linestyles=['-k', '--k', '-.k'],
                  labels=None):
        """Plots the difference between the current EOS and ither isentropes

        Plots the difference vs the prior

        Args:
            isentropes(list): A list of isentrope objects to compare
            axes(plt.Axes): The Axes on which to plot
            figure(plt.Figure): The figure object *Ignored*
            linestyles(list): A list of styles to plot
            labels(list): A list of labels for the isentropes

        Return:
            (plt.Axes)
        """

        if axes is None:
            fig = plt.figure()
            axes = fig.gca()
        elif isinstance(axes, plt.Axes):
            pass
        else:
            raise TypeError("{:} axis muse be a valid matplotlib Axes object"
                            .format(self.get_inform(1)))
        # end

        v_list = np.linspace(0.2,  # self.get_option('spline_min'),
                             0.6,  # self.get_option('spline_max'),
                             200)

        axes.plot(v_list, self(v_list) - self.prior(v_list), label="Model")

        if labels is None:
            labels = [None] * len(isentropes)
        elif not len(labels) == len(isentropes):
            raise IndexError("{:} must provide either no labels or one for each"
                             "isentrope".format(self.get_inform(1)))
        # end

        if not isinstance(isentropes, (list, tuple)):
            raise TypeError("{:} isentropes must be a list or tuple of"
                            "isentropes".format(self.get_inform(1)))
        # end
        for isen, lbl in zip(isentropes, labels):
            axes.plot(v_list, isen(v_list) - self.prior(v_list), label=lbl)
        # end

        axes.legend(loc='best')
        axes.set_xlabel(r'Specific Volume / ${cm}^3{g}^{-1}$')
        axes.set_ylabel(r'Pressure difference / Pa')
        return axes

    def plot_basis(self, axes=None, fig=None, labels=[], linstyles=[]):
        """Plots the basis function and their first and second derivatives

        Args:
            fig(plt.Figure): A valid figure object on which to plot
            axes(plt.Axes): A valid axes, *Ignored*
            labels(list): The labels, *Ignored*
            linestyles(list): The linestyles, *Ignored*
        Return:
            (plt.Figure): The figure

        """

        if fig is None:
            fig = plt.figure()
        else:
            fig = fig
        # end

        dof_init = copy.deepcopy(self.get_dof())

        basis = []
        dbasis = []
        ddbasis = []

        v_list = np.linspace(self.get_option('spline_min'),
                             self.get_option('spline_max'),
                             300)

        for i, coeff in enumerate(dof_init):
            new_dof = np.zeros(dof_init.shape[0])
            new_dof[i] = 1.0  # coeff
            tmp_spline = self.update_dof(new_dof)
            basis.append(tmp_spline(v_list))
            dbasis.append(tmp_spline.derivative(n=1)(v_list))
            ddbasis.append(tmp_spline.derivative(n=2)(v_list))
        # end

        basis = np.array(basis)
        dbasis = np.array(dbasis)
        ddbasis = np.array(ddbasis)

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        knots = tmp_spline.get_t()

        for i in range(basis.shape[0]):
            ax1.plot(v_list, basis[i, :])
            ax1.plot(knots, np.zeros(knots.shape), 'xk')
            ax2.plot(v_list, dbasis[i, :])
            ax2.plot(knots, np.zeros(knots.shape), 'xk')
            ax3.plot(v_list, ddbasis[i, :])
            ax3.plot(knots, np.zeros(knots.shape), 'xk')

        return fig
