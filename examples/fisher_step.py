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
sys.path.append(os.path.abspath('./../'))
from F_UNCLE.Experiments.GunModel import Gun
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Opt.cost_opt import CostOpt
from F_UNCLE.Opt.Bayesian import Bayesian
from F_UNCLE.Models.Ptw import Ptw


parser = argparse.ArgumentParser(description='Generate plots for notes.tex')
parser.add_argument('--show', dest='show', action='store_true')
parser.add_argument('--fig_dir', type=str, dest='fig_dir', default='./',
                    help='Directory of figures')

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

    # 4. Create the analysis object
    analysis = Bayesian(
        simulations={
            'Gun': [gun_simulation, gun_experiment]},
        models={'eos': eos_model},
        opt_key='eos',
        constrain=True,
        outer_reltol=1E-6,
        precondition=True,
        debug=False,
        verb=True,
        maxiter=10)

    # 5. Generate data from the simulations using the prior
    gun_prior_sim = gun_simulation(analysis.models)

    # 6. Run the analysis
    opt_model, history, sens_matrix = analysis()

    # 7. Update the simulations and get new data
    g_time_s, (g_vel_s, g_pos_s), g_spline_s =\
        opt_model.simulations['Gun']['sim'](opt_model.models)
    g_time_e, (g_vel_e, g_pos_e), g_spline_e =\
        opt_model.simulations['Gun']['exp']()

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
    out_dir = './'
    square = (figwidth, figwidth)
    tall = (figwidth, 1.25 * figwidth)

    '''Gets the fisher information matrix
    '''

    fisher = opt_model.simulations['Gun']['sim'].\
        get_fisher_matrix(opt_model.models,
                          sens_matrix=sens_matrix['Gun'])
    spec_data = opt_model.fisher_decomposition(fisher)

    fig = plt.figure(figsize=tall)
    fig = opt_model.plot_fisher_data(spec_data, fig=fig)
    fig.set_size_inches(tall)
    fig.tight_layout()
    fig.savefig('fisher_info_step.png')

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
    fig.savefig('gun_init.png')

    model_dict = opt_model.models
    model = model_dict['eos']

    z_list = [0.5, 0.75, 0.95]
    styles = ['--k', '-.k', ':k']
    fig = plt.figure(figsize=square)
    ax1 = fig.gca()
    for z_sigma, style in zip(z_list, styles):
        model_dict['eos'] = model.update_dof(model.get_dof() *
                                             (1 + z_sigma * spec_data[1][0]))
        data_plus = opt_model.simulations['Gun']['sim'](model_dict)

        model_dict['eos'] = model.update_dof(model.get_dof() *
                                             (1 - z_sigma * spec_data[1][0]))
        data_minus = opt_model.simulations['Gun']['sim'](model_dict)

        gun_simulation.plot(axes=ax1, linestyles=[style],
                            data=[data_plus])
        gun_simulation.plot(axes=ax1, linestyles=[style],
                            data=[data_minus])
        pdb.set_trace()


    gun_simulation.plot(axes=ax1, linestyles=['-k', '-r'],
                        labels=['nominal', 'Error'],
                        data=[(g_time_s, (g_vel_s, g_pos_s), g_spline_s)])
    fig.savefig('gun_steps2.png')
