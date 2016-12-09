"""

Stick: A simplified model of a rate stick

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (06-06-2016)

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
import unittest
import sys
import os

# =========================
# Python Packages
# =========================
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Custom Packages
# =========================

if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Utils.Experiment import GausianExperiment
    from F_UNCLE.Models.Isentrope import EOSBump, EOSModel, Isentrope
else:
    from ..Utils.Experiment import GausianExperiment
    from ..Models.Isentrope import EOSBump, EOSModel, Isentrope

# =========================
# Main Code
# =========================


class Stick(GausianExperiment):
    """A toy physics model representing a rate stick
    **TO DO**

    - Update the __call__ method to not use hardcoded sensor positions

    **Units**

    Units are based on CGS system

    **Diagram**

    .. figure:: /_static/stick.png

       The assumed geometry of the rate stick

    **Attributes**

    Attributes:
        eos(Isentrope): The products-of-detonation equation of state

    **Methods**
    """
    def __init__(self, name="Rate Stick Computational Experiment",
                 *args, **kwargs):
        """Instantiate the Experiment object

        Keyword Args:
            name(str): A name. (Default = "Rate Stick Computational Experiment")

        """

        # 'Name': [Type, Default, Min, Max, Units, Description]
        def_opts = {
            'sigma_t': [float, 1.0e-9, 0.0, None, 's',
                        'Variance attributed to t measurements'],
            'sigma_x': [float, 2e-3, 0.0, None, 'cm',
                        'Variance attributed to x positions'],
            'd_min': [float, 1.0e5, 0.0, None, 'cm sec**-1',
                      'Lower search range for detonation velocity'],
            'd_max': [float, 1.0e6, 0.0, None, 'cm sec**-1',
                      'Upper search range for detonation velocity'],
            'vol_0': [float, 1.835**-1, 0.0, None, 'cm**3 g**-1',
                      'The pre detonation specific volume of HE'],
            'x_min': [float, 0.0, 0.0, None, 'cm',
                      'The position of the first sensor on the rate stick'],
            'x_max': [float, 17.7, 0.0, None, 'cm',
                      'The position of the last sensor on the rate stick'],
            'n_x': [int, 7, 0, None, '',
                    'Number of sensor positions']
        }

        GausianExperiment.__init__(self, {'eos': Isentrope}, name=name,
                                   def_opts=def_opts, *args, **kwargs)

    def _on_check_models(self, models):
        """Checks that the model is valid

        Args:
            model(dict): A dictionary of models

        Return:
            (GunModel): A copy of self with the new eos model
        """
        return (models['eos'],)

    def _on_str(self, *args, **kwargs):
        """Print method of the gun model

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Return:
            (str): A string representing the object
        """

        out_str = ''

        return out_str

    def get_sigma(self, models):
        r"""Returns the variance matrix

        variance is

        .. math::

            \Sigma_i = \sigma_t^2 + \frac{\sigma_x^2}{V_{CJ}}

        **Where**
            - :math:`\sigma_t` is the error in time measurements
            - :math:`\sigma_x` is the error in sensor position
            - :math:`V_{CJ}` is the detonation velocity

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`

        """

        eos = self.check_models(models)[0]

        vol_0 = self.get_option('vol_0')

        vel_cj = eos._get_cj_point(vol_0)[0]

        var = np.ones(self.get_option('n_x'))
        var *= (self.get_option('sigma_t')**2 + (self.get_option('sigma_x') /
                                                 vel_cj)**2)
        return np.diag(var)

    def shape(self):
        """Returns the shape of the object

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.shape`

        """

        return self.get_option('n_x')

    def _on_call(self, eos):
        """Performs the rate stick experiment

        Args:
            eos(Isentrope): A valid EOS model

        Return:
            (tuple): Length 3. Elements are

                0. (np.ndarray): The independent variable, the `n` sensor
                   positions
                1. (tuple): The dependent variables, elements are:

                   0. (None)
                   1. (np.ndarray): The arrival `n` times at each sensor
                2. (tuple): The other solution data

                   0. the detonation velocity
                   1. the specific volume at the CJ point
                   2. the pressure at the CJ point
                   3. a Rayleigh line function, see below

        *Rayleigh Line Function*

            `p = ray(v, vel, vol0, eos)`

            Args:

                - v(np.ndarray): The specific volume
                - vel(float): Detonation velocity
                - vol_0(float): Specific volume ahead of the shock
                - eos(Isentrope): An equation of state model

            Return:

                - p(float): The pressure along the Rayleigh line at v

        """
        x_min = self.get_option('x_min')
        x_max = self.get_option('x_max')
        n_x = self.get_option('n_x')
        vol_0 = self.get_option('vol_0')

        # x_list = np.array(  # mm, page 8
        #    [25.9, 50.74, 75.98, 101.8, 125.91, 152.04, 177.61])/10
        x_list = np.linspace(x_min, x_max, n_x)

        cj_vel, cj_vol, cj_p, ray_fun = eos._get_cj_point(vol_0)

        t_list = x_list / cj_vel

        return x_list, [t_list], (cj_vel, cj_vol, cj_p, ray_fun)

    def compare(self, indep, dep, data):
        """Compares the model instance to other data

        The error is the difference in arrival times, dep less data.

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.compare`

        """

        det_vel = data[2][0]
        err = dep - indep / det_vel

        return np.where(np.fabs(err) > np.finfo(float).eps,
                        err,
                        np.zeros(err.shape))

    def plot(self, models, axes=None, fig=None, data=None, level=1,
             linestyles=['-k', ':k', 'ok', '+k'],
             labels=['Fit EOS', 'Rayleigh Line', r'($v_o$, $p_o$)',
                     'Inital point']):
        """Plots the EOS and Rayleigh line
        Plots the critical Rayleigh line corresponding to the detonation
        velocity tangent to the EOS.

        Args:
            models(dict): Dict of models

        Keyword Arguments:
            axes(plt.Axes): The Axes on which to plot
            fig(plt.Figure): The figure on which to plot *ignored*
            data(list): The output from a call to Stick
            level(int): Specified what to plot
                 1. Plots the EOS with the Raylight line intersecting the CJ
                    point
                 2. Plots the output from a simulation
            linestyles(list): Format strings for the trends, entries as follow
                 0. Stlye for the best fit EOS OR The data trend
                 1. Style for the Rayleigh line
                 2. Style for the CJ point
                 3. Style for the initial condiations
            labels(list): Strings for the legend
                 0. 'Fit EOS' (Change to Data for level 2 plot)
                 1. 'Rayleigh line'
                 2. 'v_o, p_o'
                 3. 'Initial point'
        see :py:meth:`F_UNCLE.Utils.Struc.Struc.plot`
        """

        eos = self.check_models(models)[0]

        v_min = eos.get_option('spline_min')
        v_max = eos.get_option('spline_max')
        v_0 = self.get_option('vol_0')

        if axes is None:
            fig = plt.figure()
            ax1 = fig.gca()
        else:
            fig = None
            ax1 = axes
        # end

        if level == 1:
            eos.plot(axes=ax1, linestlyes=[linestyles[0]], labels=[labels[0]])
            vel_cj, vol_cj, p_cj, rayl_line =\
                eos._get_cj_point(1.835**-1)

            # v_eos = np.logspace(np.log10(v_min), np.log10(v_max), 30)
            v_eos = np.linspace(v_min, v_max, 30)
            ax1.plot(v_eos, rayl_line(vel_cj, v_eos, v_0, 0.0), linestyles[1],
                     label="Rayl Line {:4.3f} km/s".format(vel_cj/1E5))
            ax1.plot(vol_cj, p_cj, linestyles[2], label=labels[2])
            ax1.plot(v_0, 0.0, linestyles[3], label=labels[3])
        elif level == 2:
            ax1.plot(data[0], 1E-3 * data[1][0], linestyles[0], label=labels[0])
            ax1.set_xlabel("Sensor position / cm")
            ax1.set_ylabel("Shock arrival time / ms")
        # end

        return fig
