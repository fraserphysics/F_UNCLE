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
from ..Utils.Simulation import Simulation
from ..Models.Isentrope import EOSBump, EOSModel, Isentrope, Spline
from ..Utils.Experiment import GaussianExperiment

# =========================
# Main Code
# =========================


class Stick(Simulation):
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
    def __init__(self, name=u"Rate Stick Computational Experiment",
                 *args, **kwargs):
        """Instantiate the Experiment object

        Keyword Args:
            name(str): A name. (Default = "Rate Stick Computational Experiment")

        """

        # 'Name': [Type, Default, Min, Max, Units, Description]
        def_opts = {
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

        Simulation.__init__(self, {'eos': Isentrope}, name=name,
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
        """Print method of the gun model.  Called by Struct.__str__

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

        return np.diag(np.ones(self.shape()))

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
                1. (list): The dependent variables, elements are:

                   0. (np.ndarray): The arrival `n` times at each sensor
                   1. (list): The lables
        
                2. (dict): The other solution data
                   - 'mean_fn'(Function): A function returning shock arrival 
                                          time as a function of position
                   - 'vel_CJ'(float): The detonation velocity
                   - 'vol_CJ'(float): The specific volume at the_CJ point
                   - 'pres_CJ'(float): The pressure at the_CJ point
                   - 'Rayl_fn'(Function): A Rayleigh line function, see below

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

        return x_list, [t_list, ['times']],\
            {'mean_fn': Spline(x_list, t_list),
             'vel_CJ': cj_vel,
             'vol_CJ': cj_vol,
             'pres_CJ': cj_p,
             'Rayl_fn': ray_fun}

    def compare(self, simdata1, simdata2):
        """Compares the model instance to other data

        The error is the difference in arrival times, dep less data.

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.compare`

        """


        err = simdata2[1][0] - simdata1[2]['mean_fn'](simdata2[0])

        return np.where(np.fabs(err) > np.finfo(float).eps,
                        err,
                        np.zeros(err.shape))

    def plot(self, models, axes=None, fig=None, data=None, level=1,
             linestyles=['-k', ':k', 'ok', '+k'],
             labels=['Fit EOS', 'Rayleigh Line', r'($v_o$, $p_o$)',
                     'Inital point'],
             vrange=None):
        """Plots the EOS and Rayleigh line

        Plots the critical Rayleigh line corresponding to the detonation
        velocity tangent to the EOS.

        Args:
            models(dict): Dict of models

        Keyword Arguments:
            axes(plt.Axes): The Axes on which to plot
            fig(plt.Figure): The figure on which to plot *ignored*
            data(list): The output from a call to Stick
            level(int): Specifies what to plot
		1. Plots the EOS with the Raylight line intersecting the CJ point
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
            vrange(tuple): Range of volumes to plot
        see :py:meth:`F_UNCLE.Utils.Struc.Struc.plot`
        """

        eos = self.check_models(models)[0]

        if vrange is not None:
            v_min = vrange[0]
            v_max = vrange[1]
        else:
            v_min = eos.get_option('spline_min')
            v_max = eos.get_option('spline_max')
        # end
        
        v_0 = self.get_option('vol_0')

        if axes is None:
            fig = plt.figure()
            ax1 = fig.gca()
        else:
            fig = None
            ax1 = axes
        # end

        if level == 1:
            eos.plot(axes=ax1,
                     linestlyes=[linestyles[0]],
                     labels=[labels[0]],
                     vrange=vrange)
            vel_cj, vol_cj, p_cj, rayl_line =\
                eos._get_cj_point(1.835**-1)

            # v_eos = np.logspace(np.log10(v_min), np.log10(v_max), 30)
            v_eos = np.linspace(v_min, v_max, 30)
            ax1.plot(v_eos, rayl_line(vel_cj, v_eos, v_0, 0.0), linestyles[1],
                     label="Rayl Line {:4.3f} km/s".format(vel_cj/1E5))
            ax1.plot(vol_cj, p_cj, linestyles[2], label=labels[2])
            ax1.plot(v_0, 0.0, linestyles[3], label=labels[3])
            if vrange is not None:
                ax1.set_xlim(*vrange)
            # end
            ax1.set_ylim(bottom=0.0, top = None)
        elif level == 2:
            ax1.plot(data[0], 1E6 * data[1][0], linestyles[0], label=labels[0])
            ax1.set_xlabel("Sensor position / cm")
            ax1.set_ylabel(r"Shock arrival time / $\mu$s")
        # end

        return fig

class StickExperiment(GaussianExperiment):
    """A class representing pseudo experimental data for a stick
    """

    def __init__(self, name="Stick pseudo experimental data", *args, **kwargs):
        """Instantiates the stick experiment
        """

        def_opts = {
            'sigma_t': [float, 1.0e-9, 0.0, None, 's',
                        'Variance attributed to t measurements'],
            'sigma_x': [float, 2e-3, 0.0, None, 'cm',
                        'Variance attributed to x positions']
        }

        GaussianExperiment.__init__(self, name=name, def_opts=def_opts, *args, **kwargs)
        
    def _get_data(self, model=None, *args, **kwargs):
        """Creates a simulated set of experimental data from a user provided
        model
        """

        sim = Stick(**kwargs)

        simdata = sim({'eos': model})
        self.detvel = simdata[2]['vel_CJ']
        return simdata[0], simdata[1][0],\
            np.zeros(simdata[0].shape)
    # end

    def get_splines(self,*args, **kwargs):
        return Spline(self.data[0], self.data[1]), None
    
    def get_sigma(self):
        """Returns the co-variance matrix

        see :py:meth:`F_UNCLE.Utils.Experiment.Experiment.get_sigma`
        """

        return np.diag(np.ones(self.shape())
                       * (self.get_option('sigma_t')**2
                          + (self.get_option('sigma_x') / self.detvel)**2)
        )
    
