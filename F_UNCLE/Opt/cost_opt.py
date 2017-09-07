"""

cost_opt.py

An overloaded version of Bayesian which optimizes the cost function directly

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
import matplotlib.pyplot as plt

try:
    import pyOpt
except:
    print('''You must import pyOpt to run CostOpt, but you can build the
documentations without it.''')

# =========================
# Custom Packages
# =========================

from ..Utils.Experiment import Experiment
from ..Utils.PhysicsModel import PhysicsModel
from ..Utils.Struc import Struc
from .Bayesian import Bayesian


class CostOpt(Bayesian):
    '''Code for experimenting with alternative optimization algorithms

    This class is experimental.  It Requires importing pyOpt.

    '''

    def __call__(self):
        """Overloaded call function to optimize cost directly
        """

        def objfun(x, param={}):
            """The objective function, calculates costs and constraints

            Args:
                x(list[50]): A list of model DOF,

            Keyword Args:
                param(dict): Dictonary of parameters passed to the function.
                    ['bayes']: The bayesian object

            Return:
                (float): f, The cost
                (list): g, list of constarints
                (bool): fail, failure flag
            """

            if len(x) > 50: pdb.set_trace()

            bayes = param['bayes']

            model_dict = bayes.models
            opt_key = bayes.opt_key
            model = model_dict[opt_key]

            x = np.dot(model.get_scaling(), x[:50])

            n_end = model.get_option('spline_end')
            n_dof = model.shape()

            # Update the model with the new data
            new_model = model.update_dof(x)
            model_dict[opt_key] = new_model

            new_bayes = bayes.update(models=model_dict)
            initial_data = new_bayes.get_data()

            log_like = 0
            log_like += new_bayes.model_log_like()
            log_like += new_bayes.sim_log_like(initial_data)

            n_dof = new_bayes.shape()[1]
            model_indep = new_model.get_t()

            g = np.zeros(n_dof + 2)
            g[:n_dof] = -model.derivative(n=2)(model_indep[:-n_end])
            g[n_dof] = model.derivative(n=1)(model_indep[n_end])
            g[n_dof + 1] = -model(model_indep[-n_end])

            return float(-log_like), g, False
        # end

        def gradfun(x, f, g, param={}, *args, **kwargs):
            """Function to calculate the gradients directly
            """

            bayes = param['bayes']
            model_dict = bayes.models
            opt_key = bayes.opt_key
            model = model_dict[opt_key]

            f1, g1, fail = objfun(x, param)

            step = 1E-6

            dg = []
            df = []

            old_dof = copy.deepcopy(x)
            for i in xrange(x.shape[0]):
                x[i] += old_dof[i] * step
                f2, g2, fail = objfun(x, param)
                df.append((f2 - f1) / (old_dof[i] * step))
                dg.append((g2 - g1) / (old_dof[i] * step))
                x[i] -= old_dof[i] * step

            return df, np.array(dg).T, False

        opt_prob = pyOpt.Optimization('Cost Optimization', objfun)

        opt_model = self.models[self.opt_key]
        ndof = opt_model.shape()

        scaled_dof = np.dot(np.linalg.inv(opt_model.get_scaling()),
                            opt_model.get_dof())

        opt_prob.addObj('cost')
        opt_prob.addVarGroup('dof', ndof, 'c',
                             lower=0.5,
                             upper=1.5,
                             value=scaled_dof)
        opt_prob.addConGroup('Convexity', ndof, 'i')
        opt_prob.addCon('Monotonicity', 'i')
        opt_prob.addCon('Positive', 'i')

        optimizer = pyOpt.SLSQP()
        optimizer.setOption('IPRINT', 0)
        optimizer.setOption('MAXIT', 100)

        # optimizer = pyOpt.PSQP()
        # optimizer.setOption('IPRINT', 2)
        # optimizer.setOption('XMAX', 1e16)

        # optimizer = pyOpt.ALPSO()
        # optimizer.setOption('xinit', 1)
        # optimizer.setOption('fileout', 2)
        # optimizer.setOption('stopCriteria', 0)
        # optimizer.setOption('Scaling', 0)

        # optimizer = pyOpt.CONMIN()
        # optimizer.setOption('IPRINT', 4)

        [fstr, xstr, inform] = optimizer(
            opt_prob,
            sens_type='fd',
            # sens_step=1.0E-4,
            param={'bayes': self})

        print(opt_prob.solution(0))
        model_dict = self.models
        opt_key = self.opt_key
        model = model_dict[opt_key]

        new_model = model.update_dof(np.dot(model.get_scaling(), xstr))
        model_dict[opt_key] = new_model

        new_bayes = self.update(models=model_dict)

        sens_matrix = new_bayes.get_senns()

        return new_bayes, (None, None), sens_matrix

if __name__ == '__main__':
    from F_UNCLE.Experiments.GunModel import Gun
    from F_UNCLE.Experiments.Stick import Stick
    from F_UNCLE.Experiments.Sphere import Sphere
    from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
    from F_UNCLE.Models.Ptw import Ptw

    # 1. Generate a functional form for the prior
    init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

    # 2. Create the model and *true* EOS
    eos_model = EOSModel(init_prior, Spline_sigma=0.05)
    eos_true = EOSBump()

    # 3. Create the objects to generate simulations and pseudo experimental data
    gun_experiment = Gun(model_attribute=eos_true, mass_he=1.0)
    gun_simulation = Gun(mass_he=1.0, sigma=1.0)

    stick_experiment = Stick(model_attribute=eos_true)
    stick_simulation = Stick(sigma_t=1E-9, sigma_x=2E-3)

    sphere_experiment = Sphere(model_attribute=(eos_true, Ptw()))
    sphere_simulation = Sphere()

    # 4. Create the analysis object
    analysis = CostOpt(
        simulations={'Gun': [gun_simulation, gun_experiment]},
        models={'eos': eos_model, 'strength': Ptw()},
        opt_key='eos',
        constrain=True,
        outer_reltol=1E-6,
        precondition=False,
        debug=False,
        verb=True)

    gun_prior_sim = gun_simulation(analysis.models)

    opt_model, hist, sens = analysis()

    g_time_s, (g_vel_s, g_pos_s), g_spline_s =\
        gun_simulation(opt_model.models)
    g_time_e, (g_vel_e, g_pos_e), g_spline_e =\
        gun_experiment()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    opt_model.models['eos'].plot(axes=ax1, linestyles=['-k'],
                                 labels=['Fit EOS'])
    eos_model.prior.plot(axes=ax1, linestyles=['--b'],
                         labels=['Prior EOS'])
    eos_true.plot(axes=ax1, linestyles=['-.g'], labels=['True EOS'])
    ax1.legend(loc='best')

    gun_simulation.plot(axes=ax2, linestyles=['-k', '-r'],
                        labels=['Fit EOS', 'Error'],
                        data=[(g_time_s, (g_vel_s, g_pos_s), g_spline_s),
                              (g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axes=ax2, linestyles=['-.g'], labels=['True EOS'],
                        data=[(g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axes=ax2, linestyles=['--b'], labels=['Prior EOS'],
                        data=[gun_prior_sim])
    ax2.legend(loc='upper left', framealpha=0.5)

    fig.tight_layout()
    fig.savefig('cost_opt_gun.pdf')

    fig = plt.figure()
    opt_model.models['eos'].plot_diff(
        axes=fig.gca(), isentropes=[eos_true], labels=['True'])
    fig.savefig('cost_opt_diff.pdf')
