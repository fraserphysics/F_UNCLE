# /usr/bin/pyton
"""

Experiment

Abstract class for experimental data

Authors
-------

- Stephen Andrews (SA)
- Andrew M. Fraiser (AMF)

Revisions
---------

0 -> Initial class creation (03-07-2017)

ToDo
----

None


"""

# =========================
# Python Standard Libraries
# =========================
import sys
import os
import copy
import pdb
# =========================
# Python Packages
# =========================
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline as IUSpline
from scipy.interpolate import UnivariateSpline as USpline
from scipy.interpolate import LSQUnivariateSpline as LSQSpline
from scipy import stats as stats

# =========================
# Custom Packages
# =========================
from .Struc import Struc


class Experiment(Struc):
    """Abstract class to represent true experimental data
    DataExperiments are immutable and load data on instantiation

    The following pre-processing activities occur on the data

    1. The data is loaded from file
    2. The data is checked for missing values and Nan's
    3. A smoothing spline is passed through the data along regularly
       spaced knots
    4. Other pre-processing activities are carried out by `_on_init`

    Attributes:

        data(list): A list of the data

            [0]. The independent variable, typically time
            [1]. The dependent variable
            [2]. The variance of the dependent variable

        mean_fn(scipy.interpolate.LSQSpline): A smoothing spline passing
            through the data

        var_fn(scipy.interpolate.IUSpline): An interpolated spline for the
            difference between the data and mean_fn

        trigger(None): An object to align experiments to the data.
                       Is None for basic case

    """
    data = None
    mean_fn = None
    var_fn = None


    def __init__(self, name="Data Experiment", *args, **kwargs):
        """Loads the data and performs pre-processing activities
        """

        # name: type, default, min, max, units, note
        def_opts = {
            'data_bounds': [tuple, (0, 1E21), None, None, '-',
                              'The bounds for the spline to represent the '
                              'data'],
            'n_knots': [int, 200, 5, None, '-',
                        'The number of knots to represent the data'],
            'n_smooth': [int, 0, 0, None, '-',
                        'The number of knots to smooth over'],            
            'exp_var': [float, 1E-2, 0.0, 1.0, '-',
                        'Percent variance in the data'],
        }

        if 'def_opts' in kwargs:
            def_opts.update(kwargs.pop('def_opts'))
        # end

        Struc.__init__(self, name, def_opts=def_opts,
                       *args, **kwargs)

        data = self._get_data(*args, **kwargs)
        self.data = self._check_finite(data)
        self.mean_fn, self.var_fn = self.get_splines(
            self.data[0],
            self.data[1],
            self.data[2]
        )

        # Performs class specific instantiation, i.e. making a trigger
        self._on_init(*args, **kwargs)

        # Obtains the time shift from t=0 for the curent experiment
        self.tau_exp = self.trigger(self.data[0], self.data[1])

        # Creates a window which masks the experimental data to
        # process only those within a certain time of the experimental
        # data trigger
        self.window = np.where((self.data[0] <=
                                self.tau_exp + self.get_option('data_bounds')[1])
                               &(self.data[0] >=
                                 self.tau_exp + self.get_option('data_bounds')[0]))
        
    def simple_trigger(self, x, y):
        """This is the most basic trigger object for data which does not need
        to be aling. Returns a shift if zero for all values of trial
        data
        """
        return x[0]

    def _on_init(self, *args, **kwargs):
        """Experiment specific instantiation tasks

        Creates the trigger object, In the base case it is a function which
        always returns zero
        """

        
        self.trigger = self.simple_trigger

        return

    def get_splines(self, x_data, y_data, var_data=0.0):
        """Returns a spline representing the mean and variance of the data

        Args:

            None

        Return:

            (tuple): Elements

                0. (scipy.interpolate.LSQSpline): A smoothing spline
                   representing the data
                1. (scipy.interpolate.IUSpline): An interpolated spline for the
                   difference between the data and mean_fn
        """

        # x_data = self.data[0]
        # y_data = self.data[1]
        # var_data = self.data[2]

        # bounds = self.get_option('bounds')
        bounds = (x_data[2], x_data[-3])
        knotlist = np.linspace(
            bounds[0],
            bounds[1],
            self.get_option('n_knots')
        )

        n_width = self.get_option('n_smooth')

        # If smoohting knots > 0 then apply gausian
        # smoothing. Otherwise smooth with LSQ splines
        if n_width > 0:
            width = knotlist[n_width]-knotlist[0]
            d = np.empty((len(knotlist), len(x_data)))
            for i in range(len(knotlist)):
                d[i,:] = x_data - knotlist[i]
            # end
            g = np.where(d * d / (2 * width * width) < 50.0,
                         -d * d / (2 * width * width),
                         -50.0) # Gaussian weight matrix
            g = np.exp(g)
           
            norm = g.sum(axis=1)             # Normalization vector

            mean_fn = IUSpline(
                knotlist,
                np.dot(g, y_data) / norm,
                ext=3
            )
            # mean_fn = LSQSpline(
            #     x_data,
            #     y_data,
            #     knotlist,
            #     ext=3
            # )

            error = (y_data - mean_fn(x_data))

            var_fn = IUSpline(
                knotlist,
                np.dot(g, error * error) / norm,
                ext=3
            )
            # end
        else:
            mean_fn = LSQSpline(
                x_data,
                y_data,
                knotlist,
                ext = 3
            )
            
            var_fn = LSQSpline(
                x_data,
                (y_data - mean_fn(x_data))**2,
                knotlist,
                ext = 3
            )
        # end
        
        return mean_fn, var_fn

    def __call__(self):
        """Returns the experimental data

        Args:
            None

        Returns:
            (tuple): Values are

                [0]. (np.ndarray): The independent data from the experiment
                [1]. (list): Dependent data

                    [0]. (np.ndarray): The raw data form the experiment
                    [1]. (np.ndarray): The mean_fn evaluated at the independent
                        data points
                    [2]. (np.ndarray): The raw variance from the experiment
                    [3]. (list): labels

                [2]. (dict): Auxiliary information

                    - `mean_fn` (function): The smoothed function representing
                        the dependent variable
                    - `var_fn` (function): The interpolated function
                        representing the difference between mean_fn and the
                        measured data
                    - `trigger` (Trigger): An object to align
                                           simulations to the data
        """
    # self.data[0],\
    #         [self.data[1], self.mean_fn(self.data[0]), self.data[2],

        return\
            self.data[0][self.window],\
            [self.data[1][self.window], self.mean_fn(self.data[0][self.window]), self.data[2],
             ['vel data', 'vel fn', 'variance']],\
            {'mean_fn': self.mean_fn, 'var_fn': self.var_fn,
             'trigger': self.trigger}

    def compare(self, sim_data, plot=False, fig=None):
        """Compares the results of a simulation to this these data

        Performs the following actions

        1. Calculates the time shift required to align the simulation data
           with the experiment
        2. Evaluates the simulation data at the experimental time-stamps
        3. Calculates the difference between the simulation and the experiment
           for every experimental time step. Experiment **less** simulation

        Args:

            sim_data(list): The output from a call to a F_UNCLE Experiment

        Keyword Args:

            plot(bool): Flag to plot the comparison
            fig(plt.Figure): If not none, the figure object on which to plot

        Return:

            (np.ndarray): The difference between sim_data and this experiment
                at each experimental time-stamp

        """
        tau = sim_data[2]['tau']
        epsilon = self.data[1][self.window]\
                  - sim_data[2]['mean_fn'](self.data[0][self.window] - tau)
        return epsilon

    def align(self, sim_data, plot=False, fig=None):
        """Updates the simulation data so it is aligned with the experiments

        Method
        ------

        1. Obtains the time-steps of the experiment
        2. Calculates the time shift to align the simulation with the experiment
        3. Evaluates the function representing the simulation output at the
           experimental time steps
        4. Updates the simulation `trigger` and `tau` values with the
           experimental Trigger object and the calculated `tau`
        5. Returns the simulation data aligned with the experiments

        Args:

            sim_data(list): A list returned by a F_UNCLE experiment

        Keyword Args:

            plot(bool): Flag to plot the comparison
            fig(plt.Figure): If not none, the figure object on which to plot

        Returns:

             (list): The updated simulation data

                 0. (np.ndarray): The experimental time steps
                 1. (list): The simulation dependent results

                     0. (np.ndarray): The simulation `mean_fn` shifted and
                        evaluated at the experimental times
                     1. The remaining elements are unchanged

                 2. (dict): The simulation summary data

                     - 'tau': Updated to show the time shift, in seconds,
                       required to bring the simulations and experiments
                       inline
                     - 't_0': The experimental time step when the experiment
                       starts
                     - 'trigger': The Trigger object used by the DataExperiment
        """

        tau_sim = self.trigger(sim_data[0], sim_data[1][0])
        #print('tau_sim_calc      us', tau_sim)
        # Trigger returns the time shift to align the simulations with
        # t=0 To align the simulations with the experiments, we need
        # the time shift which brings the experiments back to t=0. To
        # convert from experiment time to simulation time
        #t_sim = t_exp - tau_exp + tau_sim
        #t_sim = t_exp - (tau_exp - tau_sim)
        #t_sim = t_exp - tau
        tau = self.tau_exp - tau_sim 
        sim_data[2]['tau'] = tau        
        sim_data[2]['t_0'] = self.tau_exp
        sim_data[2]['trigger'] = copy.deepcopy(self.trigger)

        sim_data[1][0] = sim_data[2]['mean_fn'](
            self.data[0][self.window] - tau)

        return self.data[0][self.window], sim_data[1], sim_data[2]

    def shape(self):
        """Returns the number of independent data for the experiment

        Args:

            None

        Return:

           (int): The number of data
        """
        return self.data[0][self.window].shape[0]

    def get_sigma(self):
        """Returns the variance matrix.

        Variance is given as:

            self.data[1] * `exp_var`

        Args:

            tmp(dict): The physics models - not used

        Returns:

            (np.ndarray): nxn variance matrix where n is the number of data

        .. warning::

            This is a work in progress. The true variance should
            be calculated from the data.

        """
        # This test is needed so that the variance matrix is not
        # sigular for zero values of experimental data
        return np.diag(np.where(
            np.fabs(self.data[1][self.window]) > 1E-12,
            (self.data[1][self.window] * self.get_option('exp_var'))**2,
            1E-12))


    def _check_finite(self, data):
        """Removes Nan from data
        """
        data_out = []
        for trend in data:
            if trend is not None:
                data_out.append(
                    trend[np.where(np.isfinite(trend))[0]]
                )
            else:
                data_out.append(None)
            # end
        # end

        for i, trend1 in enumerate(data):
            for j, trend2 in enumerate(data):
                if trend1 is not None and trend2 is not None\
                   and not trend1.shape[0] == trend2.shape[0]:
                    raise ValueError(
                        "{:} Shape of data trends {:d} and {:d}"
                        " do not agree"
                        .format(self.get_inform(1), i, j)
                    )
                # end
            # end
        # end
        self.window = np.array([True] * data_out[0].shape[0])
        return data_out

    def _get_data(self, *args, **kwargs):
        """Abstract class for parsing datafiles or generating synthetic data

        Args:
            *args(list): Variable length list of arguments passed from
                         __init__
            *kwargs(list): Variable length list of keyword args passed from
                         __init__
        
        Return:
            (tuple): The data, elements are 
         
                0. (np.ndarray): The independent variable
                1. (np.ndarray): The **single** dependent variable
                2. (np.ndarray or None): The variance of the dependent
                   variable if availible 
        """
        
        raise NotImplementedError("Must be implemented in a child of the"
                                  "Experiment class")


    def get_pq(self, models, opt_key, sim_data, sens_matrix,
               scale=False):
        """Generates the P and q matrix for the Bayesian analysis

        Args:
           models(dict): The dictionary of models
           opt_key(str): The key for the model being optimized
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           sens_matrix(np.ndarray): The sensitivity matrix

        Keyword Arguments:
           scale(bool): Flag to use the model scaling

        Return:
            (tuple): Elements are:
                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """

        raise NotImplementedError('{:} cannot compute likelyhoods without a'
                                  ' statistical model for the data. Use a child'
                                  ' of Experiment which has implemented a model'
                                  .format(self.get_inform(1)))

    def get_log_like(self, sim_data):
        """Gets the log likelihood of the current simulation

        Args:
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           experiment(Experiment): A valid Experiment object

        Return:
            (float): The log of the likelihood of the simulation
        """

        raise NotImplementedError('{:} cannot compute likelyhoods without a'
                                  ' statistical model for the data. Use a child'
                                  ' of Experiment which has implemented a model'
                                  .format(self.get_inform(1)))
    
    def get_fisher_matrix(self, sens_matrix):
        """Returns the fisher information matrix of the simulation

        Args:
            sens_matrix(np.ndarray): the sensitivity matrix

        Keyword Args:
            use_hessian(bool): Flag to toggle wheather or not to use the hessian

        Return:
            (np.ndarray): The fisher information matrix, a nxn matrix where
            `n` is the degrees of freedom of the model.
        """

        sigma = inv(self.get_sigma())
        
        return np.dot(sens_matrix.T, np.dot(sigma, sens_matrix))
    
    def parse_datafile(self, folder, filename, root_dir, header_lines, delimiter,
                       indepcol, depcols, varcols):
        """Parser to read experimental datafiles

        Args:
            folder(str): The folder within fit_9501/Data where the experimental
                         data are located
            filename(str): The name of the datafile
            root_dir(str): The path of the file caling this method
            header_lines(int): The number of header lines to skip
            delimiter(str): The delimiter for the file
            indepcol(int): The index of the independent variable column
            depcols(list): The indicies of the dependant variables
            varcols(list): The indicies of the variance of the dep var

        """
        fname = os.path.abspath(os.path.normcase(os.path.join(
            root_dir,
            './../Data/',
            folder,
            filename)))

        with open(fname, 'rb') as fid:
            data = np.genfromtxt(fid,
                                 skip_header=header_lines,
                                 delimiter=delimiter,
                                 missing_values=np.nan)
        # end
        if varcols is not None:
            return data[:, indepcol],\
                data[:, depcols][:,0],\
                data[:, varcols]
        else:
            return data[:, indepcol],\
                data[:, depcols][:,0],\
                np.nan * np.ones(data.shape[0])
    
class GaussianExperiment(Experiment):
    """Class to represent experimental data which is modelled as being
    normally distributed
    """
    
    def get_pq(self, models, opt_key, sim_data, sens_matrix,
           scale=False):
        """Generates the P and q matrix for the Bayesian analysis

        Args:
           models(dict): The dictionary of models
           opt_key(str): The key for the model being optimized
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           sens_matrix(np.ndarray): The sensitivity matrix

        Keyword Arguments:
           scale(bool): Flag to use the model scaling

        Return:
            (tuple): Elements are:
                0. (np.ndarray): `P`, a nxn matrix where n is the model DOF
                1. (np.ndarray): `q`, a nx1 matrix where n is the model DOF

        """


        epsilon = self.compare(sim_data)

        p_mat = np.dot(np.dot(sens_matrix.T,
                              inv(self.get_sigma())),
                       sens_matrix)
        q_mat = -np.dot(np.dot(epsilon,
                               inv(self.get_sigma())),
                        sens_matrix)
        #pdb.set_trace()
        # q_mat = -np.dot(np.dot(epsilon.shape,
        #                        inv(self.get_sigma())),
        #                 sens_matrix)

        if scale:
            prior_scale = models[opt_key].get_scaling()
            p_mat = np.dot(prior_scale, np.dot(p_mat, prior_scale))
            q_mat = np.dot(prior_scale, q_mat)
        # end

        return p_mat, q_mat

    def get_log_like(self, sim_data):
        """Gets the log likelihood of the current simulation

        Args:
           sim_data(list): Lengh three list corresponding to the `__call__` from
                           a Experiment object
           experiment(Experiment): A valid Experiment object

        Return:
            (float): The log of the likelihood of the simulation
        """

        epsilon = self.compare(sim_data)
        return -0.5 * np.dot(epsilon,
                             np.dot(inv(self.get_sigma()),
                                    epsilon))
