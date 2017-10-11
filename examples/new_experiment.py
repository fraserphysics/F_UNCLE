"""F_UNCLE Optimization of the new experiments
which more closely model true HE tests
"""

## Standard python packages
import sys
import os
import argparse
import pickle
import time
import copy
from collections import OrderedDict
import pdb
## External python packages
import numpy as np
import numpy.linalg as nplin

import scipy.stats as spstat

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rcParams
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 7
rcParams['legend.handlelength'] = 3.0

figdir = './new_fig'
pagewidth = 390 # pt
au_ratio = (np.sqrt(5) - 1.0) / 2.0
figwidth = 1.0  # fraction of \pagewidth for figure
figwidth *= pagewidth / 72.27
figtype = '.pdf'
fig_square = (figwidth, figwidth)
fig_golden = (0.75 * figwidth, 0.75 * au_ratio * figwidth)
fig_half = (0.5 * figwidth, 0.5 * figwidth)
fig_tall = (figwidth, 1.25 * figwidth)
fig_vtall = (2.0 * figwidth, 1.0 *  figwidth)

## F_UNLCE packages
from F_UNCLE.Experiments.Sandwich import ToySandwich, ToySandwichExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Experiments.Cylinder import ToyCylinder, ToyCylinderExperiment
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Models.SimpleStr import SimpleStr
from F_UNCLE.Opt.Bayesian import Bayesian
from F_UNCLE.Opt.Sampler import Sampler

## Make the models
str_model = SimpleStr([101E9, 100.0], sigma=0.05) # Youngs modulis for Cu from Hibbeler
str_true  = SimpleStr([101E9, 100.0]) # Youngs modulis for Cu from Hibbeler

eos_model = EOSModel(
    lambda v: 2.56E9/v**3, # famma=3 gas for HE
    spline_sigma = 0.25,
    spline_min = 5.5**-1,
    spline_max = 2.0,
    spline_N = 20,
    spacing='log'
)
eos_true = EOSBump()

#isen.write_tex('isentrope.tex')


## Make the simulations and experiments

rstate = np.random.RandomState(seed=1122548) # Random state with fixed seed

sandwich_simulation = ToySandwich()
sandwich_experiment = ToySandwichExperiment(model=eos_true,
                                            exp_corr = 0.1,
                                            rstate=rstate )

stick_simulation = Stick(model_attribute=eos_true)
stick_experiment = StickExperiment(model=eos_true,
                                   sigma_t=1E-10,
                                   sigma_x=2E-4)

cylinder_simulation = ToyCylinder()
cylinder_experiment = ToyCylinderExperiment(
    models={'eos': eos_true, 'strength': str_true},
    rstate=rstate
    )

simulations = OrderedDict()
#simulations['Sand'] =  [sandwich_simulation, sandwich_experiment]
#simulations['Stick'] = [stick_simulation, stick_experiment]
simulations['Cyl'] = [cylinder_simulation, cylinder_experiment]

models = OrderedDict()
models['eos'] = eos_model
models['strength'] = str_model

analysis = Bayesian(
    simulations=simulations,
    models=models,
    opt_keys=['eos'],
    constrain=True,
    outer_reltol=1E-6,
    precondition=True,
    debug=False,
    verb=True,
    sens_mode='ser',
    maxiter=1
)

to = time.time()
opt_model, history, sens_matrix, fisher_matrix  = analysis()
print('time taken ', to - time.time() )


samp = Sampler(opt_model.models, opt_model.simulations, fisher_matrix)



for simid in ['Cyl']:# ['Sand', 'Stick', 'Cyl']:
    new_var = samp.cramer_rao(opt_model.models['eos'], fisher_matrix[simid])
    pcnt_list, dof_list, data_list = samp.pointwise_bounds(simid, 'eos')
    eigval, eigvec, eigfun = samp.spectral_decomp(simid, 'eos')

    models= opt_model.models

    knots = np.linspace(
        models['eos'].get_option('spline_min'),
        models['eos'].get_option('spline_max'),
        100)

    true_knots = models['eos']._get_knot_spacing()

    data = opt_model.simulations[simid]['exp']()

    fig00 = plt.figure(figsize = fig_half)
    fig01 = plt.figure(figsize = fig_half)
    fig02 = plt.figure(figsize = fig_half)
    fig03 = plt.figure(figsize = fig_half)
    fig04 = plt.figure(figsize = fig_half)
    fig05 = plt.figure(figsize = fig_half)

    ax00 = fig00.gca()
    ax01 = fig01.gca()
    ax02 = fig02.gca()
    ax03 = fig03.gca()
    ax04 = fig04.gca()


    # models['eos'] = models['eos'].update_dof(dof_in)
    ax00.semilogy(true_knots, models['eos'].get_dof(), '-', label='Value')
    ax00.semilogy(true_knots, np.sqrt(np.diag(models['eos'].get_sigma())),
                  ls = ':',
                  label='Initial variance')
    ax00.semilogy(true_knots, np.sqrt(np.diag(new_var)),
                  '-',
                  label='Final variance')

    styles = ['r:', 'r--', 'r-.', 'r-', 'k-', 'g-', 'g-.', 'g--', 'g:']
    for j in range(dof_list.shape[0]):
        model = models['eos'].update_dof(dof_list[j])
        ax01.plot(1E6 * data[0], 1E-4 * data_list[j, :], styles[j],
                  label='Pcnt = {:f}'.format(pcnt_list[j]))

        ax02.semilogy(knots, 1E0 * model(knots), styles[j],
                      label='{:3.2f}'.format(pcnt_list[j]))
    # end

    ax03.plot(eigval, 'sk')

    for k, func in enumerate(eigfun):
        ax04.plot(knots, func(knots), label='{:02d}'.format(k))
    # end

    ax03.set_xlabel('Eigenvalue rank / -')
    ax03.set_ylabel(r'Eigenvalue nuber / Pa$^2$')

    ax04.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
    ax04.set_ylabel(r'Eigenfunction / Pa')
    ax04.legend(loc='best', title='Eig. No.')


    ax01.set_xlabel(r'Time / $\mu$s')
    ax01.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')

    ax02.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
    ax02.set_ylabel(r'Pressure / Pa')
    ax02.legend(loc='upper right', title='Percentile')

    ax00.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
    ax00.set_ylabel(r'Pressure / Pa')
    ax00.legend(loc='upper right')


    fig00.tight_layout()
    fig01.tight_layout()
    fig02.tight_layout()
    fig03.tight_layout()
    fig04.tight_layout()


    fig00.savefig("{:}/{:}_fisher_modvar{:}".format(figdir, simid, figtype))
    fig01.savefig("{:}/{:}_fisher_sim_pointwise{:}".format(figdir, simid, figtype))
    fig02.savefig("{:}/{:}_fisher_mod_pointwise{:}".format(figdir, simid, figtype))
    fig03.savefig("{:}/{:}_fisher_eigval{:}".format(figdir, simid, figtype))
    fig04.savefig("{:}/{:}_fisher_eigfun{:}".format(figdir, simid, figtype))

    plt.close(fig00)
    plt.close(fig01)
    plt.close(fig02)
    plt.close(fig03)
    plt.close(fig04)
# end



## Plot the EOS
fig = plt.figure(figsize=fig_golden)
fig = eos_model.prior.plot(figure=fig, labels=['Prior'], linestyles = ['-k'])
fig = opt_model.models['eos'].plot(figure=fig, labels = ['Postrior'], linestyles = ['--r'])
fig = eos_true.plot(figure=fig, labels = ['True'], linestyles = ['-.g'])
ax1 = fig.gca()
ax1.set_xlim((0.1, 0.75))
ax1.set_ylim((0.0, 0.5E12))
ax1.legend(loc='best')
fig.tight_layout()
fig.savefig('{:}/eos_comparisson{:}'.format(figdir, figtype))
plt.close(fig)

fig = plt.figure(figsize=fig_golden)
vol_list = np.linspace(
            models['eos'].get_option('spline_min'),
            models['eos'].get_option('spline_max'),
            100)

ax01 = fig.gca()
ax01.plot(vol_list,
          (opt_model.models['eos'](vol_list) - eos_true(vol_list))/1.0,#eos_true(vol_list),
          label="Optimal Model")
ax01.plot(vol_list,
          (opt_model.models['eos'].prior(vol_list) - eos_true(vol_list))/1.0,#eos_true(vol_list),
          label="Prior")
ax01.set_xlabel(r'Specific Volume / cm$^{3}$g$^{-1}$')
ax01.set_ylabel(r'Difference to True Model / Pa')
ax01.legend(loc='lower right')
fig.tight_layout()
fig.savefig('{:}/eos_delta{:}'.format(figdir, figtype))
plt.close(fig)

## Get the data for the sandwich
sand_prior = sandwich_simulation({'eos': eos_model.prior})
sand_post = sandwich_simulation(opt_model.models)
sand_exp = sandwich_experiment()
fig = plt.figure(figsize=fig_golden)
fig = sandwich_simulation.plot(fig=fig, data=[sand_prior, sand_post, sand_exp],
                         labels=['Prior', 'Posterior', 'Experiment'],
                         linestyles=['-k', '--r', '-.g'])
fig.tight_layout()
fig.savefig('{:}/sand_results{:}'.format(figdir, figtype))
plt.close(fig)
## Get the data for the cylinder
cyl_prior = cylinder_simulation({'eos': eos_model.prior, 'strength': str_true})
cyl_post = cylinder_simulation(opt_model.models)
cyl_exp = cylinder_experiment()
fig = plt.figure(figsize=fig_golden)
fig = cylinder_simulation.plot(fig=fig, data=[cyl_prior, cyl_post, cyl_exp],
                         labels=['Prior', 'Posterior', 'Experiment'],
                         linestyles=['-k', '--r', '-.g'])
fig.tight_layout()
fig.savefig('{:}/cyl_results{:}'.format(figdir, figtype))


# # ## Plot the sensitivity of the cylinder to isen
# # fig = EOSModel.plot_sens_matrix(sens_matrix, 'Cyl', opt_model.models, 'eos')
# # fig.savefig('cyl_eos_sens.pdf')

# # fig = SimpleStr.plot_sens_matrix(sens_matrix, 'Cyl', opt_model.models, 'strength')
# # fig.savefig('cyl_str_sens.pdf')

# # ## Plot the sensitivity of the sandwich to isen
# # fig = EOSModel.plot_sens_matrix(sens_matrix, 'Sand', opt_model.models, 'eos')
# # fig.savefig('sand_eos_sens.pdf')

# # fig = SimpleStr.plot_sens_matrix(sens_matrix, 'Sand', opt_model.models, 'strength')
# # fig.savefig('sand_str_sens.pdf')


# ## Get the fisher information
# def make_fisher_image(data):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     ax1.semilogy(data[0], 'sk')
#     for i in range(data[2].shape[0]):
#         ax2.plot(data[3], data[2][i])
#     # end
#     return fig

# data = Bayesian.fisher_decomposition(fisher_matrix, 'Cyl', opt_model.models, 'eos')
# fig = make_fisher_image(data)
# fig.savefig('cyl_eos_info.pdf')


# data = Bayesian.fisher_decomposition(fisher_matrix, 'Sand', opt_model.models, 'eos')
# fig = make_fisher_image(data)
# fig.savefig('sand_eos_info.pdf')
