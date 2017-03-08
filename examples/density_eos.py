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
import time
# External python packages
import numpy as np
import matplotlib.pyplot as plt

# F_UNLCE packages
sys.path.append(os.path.abspath('./../../fit_9501'))    
sys.path.append(os.path.abspath('./..'))
from F_UNCLE.Experiments.GunModel import Gun, GunExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Experiments.Sphere import Sphere
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from fit_9501.Models.equation_of_state import AugmentedIsentrope
from F_UNCLE.Opt.Bayesian import Bayesian
from F_UNCLE.Models.Ptw import Ptw

parser = argparse.ArgumentParser(description='Generate plots for notes.tex')
parser.add_argument('--show', dest='show', action='store_true')
parser.add_argument('--fig_dir', type=str, dest='fig_dir', default='./',
                    help='Directory of figures')
for s, h in (
        ('eos_diff', 'Plot the difference between the prior and true eos'),
        ('eos', 'Plot prior and true eos'),
        ('eos_basis', 'Plots the eos basis functions'),
        ('info_all',
         'Plot the eigenvalues and eigenfunctions for all experiments'),
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
    # Prior is the JWL isentrope. Values are from [TarverManganin2001]
    def jwl_isentrope(dens, rho_o=1.844, A=16.689, B=0.5969, C=0.018229,
                      R1=5.9, R2=2.1, omega=0.45):
    # def jwl_isentrope(dens, rho_o=1.891, A=7.783, B=0.07071, C=0.00643,
    #                   R1=4.20, R2=1.00, omega=0.30):
        """The JWL Isentrope.

        Eqiation come from the Ps eqiation along an isentrope in sec 8.3.1
        pp 8-21 of Ref [1]
        Args:
            dens(float or np.ndarray): The density of the gas g cm-3

        Keyword Args:
            rho_o(float): Reactants density from ref [1] in g cm-3
            A(float): PBX-9501 coefficeint from 
                      [TarverManganin2001] in Mbar
            B(float): PBX-9501 coefficeint from 
                      TarverManganin2001] in Mbar
            C(float): PBX-9501 coefficeint from 
                      [TarverManganin2001] in Mbar
            R1(float): PBX-9501 coefficeint from 
                       [TarverManganin2001], dimensionless
            R1(float): PBX-9501 coefficeint from 
                       [TarverManganin2001], dimensionless
            omega(float): PBX-9501 coefficeint from 
                          [TarverManganin2001], dimensionless

        Returns:
            (float or np.ndarray): The pressure at the given density in Pa

        References
        [1] "LLNL Explosives Handbook" LLNL-URCRL-52997Ve
        """
        v = rho_o / dens
        return 1e11 * (A * np.exp(-R1 * v)
                       + B * np.exp(-R2 * v)
                       + C * v**(-omega - 1))
    # end


    eos_model = AugmentedIsentrope(
#        jwl_isentrope,
        lambda r: 2.56E9 * r**3,
        spline_min=0.1,
        spline_max=4.0,
#        vcj_lower=5E5,
#        vcj_upper=11.0E5,
        cj_vol_range=(3.5**-1, 1.7**-1),
        rho_0=1.844,
        pres_0=101325,
        spline_sigma=0.05,
        basis='dens',
        spacing='lin'
    )

    # eos_model = AugmentedIsentrope(
    #     lambda r: 2.56E9 / r**3,
    #     spline_min=0.1,
    #     spline_max=1.0,
    #     vcj_lower=5E5,
    #     vcj_upper=11.0E5,
    #     cj_vol_range=(0.1, 1.0),
    #     rho_0=1.844,
    #     pres_0=101325,
    #     spline_sigma=0.005,
    #     basis='vol',
    #     spacing='lin'
    # )

    # 2. Create the model and *true* EOS
    eos_true = EOSBump()

    # 3. Create the objects to generate simulations and pseudo experimental data
    gun_simulation = Gun(mass_he=1.0, sigma=1.0)
    gun_experiment = GunExperiment(model=eos_true, mass_he=1.0)
    stick_simulation = Stick(model_attribute=eos_true)
    stick_experiment = StickExperiment(
        model=eos_true,
        sigma_t=1E-9,
        sigma_x=2E-3
    )

    # sphere_experiment = Sphere(model_attribute=(eos_true, Ptw()))
    # sphere_simulation = Sphere()

    # 4. Create the analysis object
    analysis = Bayesian(
        simulations={
            'Gun': [gun_simulation, gun_experiment],
            'Stick': [stick_simulation, stick_experiment],
            # 'Sphere': [sphere_simulation, sphere_experiment]
        },
        models={'eos': eos_model,
                'strength': Ptw()},
        opt_key='eos',
        constrain=True,
        outer_reltol=1E-6,
        precondition=True,
        debug=False,
        verb=True,
        sens_mode='ser',
        maxiter=6)

    # 5. Generate data from the simulations using the prior
    gun_prior_sim = gun_simulation(analysis.models)
    stick_prior_sim = stick_simulation(analysis.models)
    # sphere_prior_sim = sphere_simulation(analysis.models)

    # 6. Run the analysis
    to = time.time()
    opt_model, history, sens_matrix = analysis()
    print('time taken ', to - time.time() )

    # 7. Update the simulations and get new data
    g_time_s, (g_vel_s, g_pos_s, labels), g_spline_s =\
        opt_model.simulations['Gun']['sim'](opt_model.models)
    g_time_e, (g_vel_e, g_pos_e, tmp, labels), g_spline_e =\
        opt_model.simulations['Gun']['exp']()

    s_pos_s, (s_time_s, labels), s_data_s =\
        opt_model.simulations['Stick']['sim'](opt_model.models)
    s_pos_e, (s_time_e, tmp, tmp, lables), s_data_e = opt_model.simulations['Stick']['exp']()

    # sp_res_s = opt_model.simulations['Sphere']['sim'](opt_model.models)
    # sp_res_e = opt_model.simulations['Sphere']['exp']()

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
    vrange=(0.1,4.0)
    
    def eos_diff():
        ''' Compare EOS functions
        '''
        fig = plt.figure(figsize=square)
        # opt_model.models['eos'].plot_diff(
        #     axes=fig.gca(),
        #     isentropes=[eos_true],
        #     labels=['True'],
        #     vrange=vrange)
        return fig

    def rayl_line():
        ''' Rayleigh line
        '''
        fig = plt.figure(figsize=square)
        ax = fig.gca()
        opt_model.simulations['Stick']['sim'].\
            plot(opt_model.models,
                 axes=ax,
                 vrange=vrange)
        
        eos_model.prior.plot(axes=ax,
                             linestyles=['--b'],
                             labels=['Prior EOS'],
                             vrange=vrange)
        eos_true.plot(axes=ax,
                      linestyles=['-.g'],
                      labels=['True EOS'],
                      vrange=vrange)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig

    def eos():
        ''' Nominal and true EOS:
        '''
        fig = plt.figure(figsize=square)
        ax = fig.gca()
        # opt_model.models['eos'].prior.plot(axes=ax,
        #                      linestyles=['--b'],
        #                      labels=['Prior EOS'],
        #                      vrange=vrange)
        # opt_model.models['eos'].plot(axes=ax,
        #                linestyles=['--b'],
        #                labels=['Prior EOS'],
        #                vrange=vrange)
        rho_list = np.linspace(0.5, 3.5, 200)
        ax.plot(rho_list**-1, opt_model.models['eos'].prior(rho_list),
                label='prior')
        ax.plot(rho_list**-1, opt_model.models['eos'](rho_list),
                label='result')
        # ax.plot(rho_list**-1, eos_true(rho_list**-1),
        #         label='true')
        ax.set_xlabel("Specific volume / cm3 g-1")
        ax.set_ylabel("Pressure / Pa")
        # eos_true.plot(axes=ax,
        #               linestyles=['-.g'],
        #               labels=['True EOS'],
        #               vrange=vrange)
        ax.legend(loc='best')
        fig.tight_layout()
        return fig

    def info_all():
        fisher = opt_model.simulations['Gun']['sim'].\
            get_fisher_matrix(opt_model.models,
                              use_hessian=False,
                              exp=opt_model.simulations['Gun']['exp'],
                              sens_matrix=sens_matrix['Gun'])
        fisher += opt_model.simulations['Sphere']['sim'].\
            get_fisher_matrix(opt_model.models,
                              sens_matrix=sens_matrix['Sphere'])
        fisher += opt_model.simulations['Stick']['sim'].\
            get_fisher_matrix(opt_model.models,
                              sens_matrix=sens_matrix['Stick'])

        spec_data = opt_model.fisher_decomposition(fisher)

        fig = plt.figure(figsize=tall)
        fig = opt_model.plot_fisher_data(spec_data, fig=fig)
        fig.set_size_inches(tall)
        fig.tight_layout()
        return fig

    def info_gun():
        ''' Fisher information about the gun experiment
        '''
        fisher = opt_model.simulations['Gun']['exp'].\
            get_fisher_matrix(opt_model.models,
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
        fisher = opt_model.simulations['Stick']['exp'].\
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
        fisher = opt_model.simulations['Sphere']['exp'].\
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
                              data=(s_pos_s, (s_time_s,), s_data_s))

        stick_simulation.plot(opt_model.models,
                              axes=ax, linestyles=['+g'],
                              labels=['True EOS'], level=2,
                              data=(s_pos_e, (s_time_e,), s_data_e))

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

        opt_model.models['eos'].plot(axes=ax1,
                                     linestyles=['-k'],
                                     labels=['Fit EOS'],
                                     vrange=vrange)
        eos_model.prior.plot(axes=ax1,
                             linestyles=['--b'],
                             labels=['Prior EOS'],
                             vrange=vrange)
        # eos_true.plot(axes=ax1,
        #               linestyles=['-.g'],
        #               labels=['True EOS'],
        #               vrange=vrange)
        
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
    # for name in '''eos_diff rayl_line eos info_all info_gun info_stick info_sphere stick_results gun_results sphere_results conv gun_sens stick_sens sphere_sens eos_basis '''.split():
    for name in """gun_results gun_sens eos_diff info_gun stick_results stick_sens info_stick eos""".split():
    # for name in '''eos_diff rayl_line eos info_stick conv stick_sens
    # stick_results eos_basis '''.split():    
        # if name in options:     
        L[name]().savefig(out_dir + name + figtype, dpi=1000)
    if options.show:
        plt.show()
