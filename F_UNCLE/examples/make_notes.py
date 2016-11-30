"""Script for re-generating the notes figures

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# Standard python packages
import sys
import os
import pdb
import argparse

# External python packages
import numpy as np
import matplotlib.pyplot as plt

# F_UNLCE packages
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Experiments.GunModel import Gun
    from F_UNCLE.Experiments.Stick import Stick
    from F_UNCLE.Experiments.Sphere import Sphere
    from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
    from F_UNCLE.Opt.Bayesian import Bayesian
    from F_UNCLE.Models.Ptw import Ptw
else:
    from ..Experiments.GunModel import Gun
    from ..Experiments.Stick import Stick
    from ..Experiments.Sphere import Sphere
    from ..Models.Ptw import Ptw
    from ..Models.Isentrope import EOSModel, EOSBump
    from ..Opt.Bayesian import Bayesian
# end


parser = argparse.ArgumentParser(description='Generate plots for notes.tex')
parser.add_argument('--show', dest='show', action='store_true')
parser.add_argument('--fig_dir', type=str, dest='fig_dir', default='./',
                    help='Directory of figures')
for s, h in (
        ('eos_diff', 'Plot the difference between the prior and true eos'),
        ('eos', 'Plot prior and true eos'),
        ('eos_basis', 'Plots the eos basis functions'),
        ('info_gun',
         'Plot the eigenvalues and eigenfunctions for the gun experiment'),
        ('info_stick',
         'Plot the eigenvalues and eigenfunctions for the stick experiment'),
        ('info_sphere',
         'Plot the eigenvalues and eigenfunctions for the sphere experiment'),
        ('gun_sens', 'Plot the sensitivity of the gun results to the model'),
        ('stick_sens',
         'Plot the sensitivity of the stick results to the model'),
        ('sphere_sens',
         'Plot the sensitivity of the sphere results to the model'),
        ('conv', 'Plot the difference between the prior and true eos'),
        ('stick_results', 'Results from the stick experiment'),
        ('gun_results', 'Results from the gun experiment'),
        ('sphere_results', 'Results from the sphere experiment'),
        ('rayl_line', 'Rayleigh line results'),
            ):

    parser.add_argument('--' + s, dest=s, action='store_const', const=True,
                        default=False, help=h)

options = parser.parse_args()

if __name__ == '__main__':
    #################
    #    Get Data   #
    #################

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
    analysis = Bayesian(
        simulations={
            'Gun': [gun_simulation, gun_experiment],
            'Stick': [stick_simulation, stick_experiment],
            'Sphere': [sphere_simulation, sphere_experiment]},
        models={'eos': eos_model,
                'strength': Ptw()},
        opt_key='eos',
        constrain=True,
        outer_reltol=1E-6,
        precondition=True,
        debug=False,
        verb=True,
        sens_mode='pll',
        maxiter=6)

    # 5. Generate data from the simulations using the prior
    gun_prior_sim = gun_simulation(analysis.models)
    stick_prior_sim = stick_simulation(analysis.models)
    sphere_prior_sim = sphere_simulation(analysis.models)

    # 6. Run the analysis
    opt_model, history, sens_matrix = analysis()

    # 7. Update the simulations and get new data
    g_time_s, (g_vel_s, g_pos_s), g_spline_s =\
        opt_model.simulations['Gun']['sim'](opt_model.models)
    g_time_e, (g_vel_e, g_pos_e), g_spline_e =\
        opt_model.simulations['Gun']['exp']()

    s_pos_s, (s_time_s), s_data_s =\
        opt_model.simulations['Stick']['sim'](opt_model.models)
    s_pos_e, (s_time_e), s_data_e = opt_model.simulations['Stick']['exp']()

    sp_res_s = opt_model.simulations['Sphere']['sim'](opt_model.models)
    sp_res_e = opt_model.simulations['Sphere']['exp']()

    ####################
    # Generate Figures #
    ####################

    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 7
    rcParams['legend.handlelength'] = 3.0
    rcParams['backend'] = 'Agg'

    pagewidth = 360  # pts
    au_ratio = (np.sqrt(5) - 1.0) / 2.0
    figwidth = 1.0  # fraction of \pagewidth for figure
    figwidth *= pagewidth / 72.27
    figtype = '.pdf'
    out_dir = options.fig_dir
    square = (figwidth, figwidth)
    tall = (figwidth, 1.25 * figwidth)

    def eos_diff():
        ''' Compare EOS functions
        '''
        fig = plt.figure(figsize=square)
        opt_model.models['eos'].plot_diff(
            axes=fig.gca(), isentropes=[eos_true], labels=['True'])
        return fig

    def rayl_line():
        ''' Rayleigh line
        '''
        fig = plt.figure(figsize=square)
        ax = fig.gca()
        opt_model.simulations['Stick']['sim'].\
            plot(opt_model.models, axes=ax)

        eos_model.prior.plot(axes=ax, linestyles=['--b'], labels=['Prior EOS'])
        eos_true.plot(axes=ax, linestyles=['-.g'], labels=['True EOS'])
        ax.legend(loc='best')
        fig.tight_layout()
        return fig

    def eos():
        ''' Nominal and true EOS:
        '''
        fig = plt.figure(figsize=square)
        ax = fig.gca()
        eos_model.prior.plot(axes=ax, linestyles=['--b'], labels=['Prior EOS'])
        eos_true.plot(axes=ax, linestyles=['-.g'], labels=['True EOS'])
        ax.legend(loc='best')
        fig.tight_layout()
        return fig

    def info_gun():
        ''' Fisher information about the gun experiment
        '''
        fisher = opt_model.simulations['Gun']['sim'].\
            get_fisher_matrix(opt_model.models,
                              use_hessian=False,
                              exp=opt_model.simulations['Gun']['exp'],
                              sens_matrix=sens_matrix['Gun'])
        spec_data = opt_model.fisher_decomposition(fisher)

        fig = plt.figure(figsize=tall)
        fig = opt_model.plot_fisher_data(spec_data, fig=fig)
        fig.set_size_inches(tall)
        fig.tight_layout()
        return fig

    def info_stick():
        ''' Fisher information about the stick
        '''
        fisher = opt_model.simulations['Stick']['sim'].\
            get_fisher_matrix(opt_model.models,
                              sens_matrix=sens_matrix['Stick'])
        spec_data = opt_model.fisher_decomposition(fisher)

        fig = plt.figure(figsize=tall)
        fig = opt_model.plot_fisher_data(spec_data, fig=fig)
        fig.tight_layout()
        return fig

    def info_sphere():
        ''' Fisher information about the sphere
        '''
        fisher = opt_model.simulations['Sphere']['sim'].\
            get_fisher_matrix(opt_model.models,
                              sens_matrix=sens_matrix['Sphere'])

        spec_data = opt_model.fisher_decomposition(fisher)
        fig = plt.figure(figsize=tall)
        fig = opt_model.plot_fisher_data(spec_data, fig=fig)
        fig.tight_layout()
        return fig

    def stick_results():
        ''' Results of the optimized stick simulation
        '''
        fig = plt.figure(figsize=square)
        ax = fig.gca()

        stick_simulation.plot(opt_model.models,
                              axes=ax, linestyles=['-k'],
                              labels=['Fit EOS'], level=2,
                              data=(s_pos_s, (s_time_s), s_data_s))

        stick_simulation.plot(opt_model.models,
                              axes=ax, linestyles=['+g'],
                              labels=['True EOS'], level=2,
                              data=(s_pos_e, (s_time_e), s_data_e))

        stick_simulation.plot(opt_model.models,
                              axes=ax, linestyles=['--b'],
                              labels=['Prior EOS'], level=2,
                              data=stick_prior_sim)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig

    def gun_results():
        ''' Results of the optimized gun simulation
        '''
        fig = plt.figure(figsize=tall)
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
        return fig

    def sphere_results():
        ''' Results of the optimized Sphere simulation
        '''
        fig = plt.figure(figsize=tall)
        fig = sphere_simulation.plot(sp_res_s, fig=fig)
        fig.tight_layout()
        return fig

    def conv():
        '''Convergence history
        '''
        fig = plt.figure(figsize=square)
        opt_model.plot_convergence(history, axes=fig.gca())
        fig.tight_layout()
        return fig

    gun_sens = lambda: opt_model.plot_sens_matrix(
        simid='Gun',
        fig=plt.figure(figsize=square),
        sens_matrix=sens_matrix)             # Gun sensitivity plot

    stick_sens = lambda: opt_model.plot_sens_matrix(
        simid='Stick',
        fig=plt.figure(figsize=square),
        sens_matrix=sens_matrix)             # Stick sensitivity plot

    sphere_sens = lambda: opt_model.plot_sens_matrix(
        simid='Sphere',
        fig=plt.figure(figsize=square),
        sens_matrix=sens_matrix)             # Sphere sensitivity plot

    eos_basis = lambda: eos_model.plot_basis(
        fig=plt.figure(figsize=square))    # EOS basis functions

    L = locals()
    for name in '''eos_diff rayl_line eos info_gun info_stick info_sphere
    stick_results gun_results sphere_results conv gun_sens stick_sens
    sphere_sens eos_basis '''.split():
        if name in options:
            L[name]().savefig(out_dir + name + figtype, dpi=1000)
    if options.show:
        plt.show()
