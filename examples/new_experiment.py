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
    spline_max = 2.0
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
    maxiter=6
)

to = time.time()
opt_model, history, sens_matrix, fisher_matrix  = analysis()
print('time taken ', to - time.time() )


samp = Sampler(opt_model.models, opt_model.simulations, fisher_matrix)



for simid in ['Cyl']:# ['Sand', 'Stick', 'Cyl']:
    dof_list, data_list = samp(simid, 'eos')
    models= opt_model.models
#     # Create the fisher information matrix
#     filter_fisher = fisher_matrix[simid]
#     sigma_aug = filter_fisher

#     original_var = np.sqrt(np.diag(opt_model.models['eos'].get_sigma()))

#     # Check that the fisher information matrix is positive semi-definate
#     eigval_1, eigvec_1 = nplin.eigh(sigma_aug)
#     print(eigval_1)
#     assert(np.all(eigval_1 >= -1E21))
#     assert(np.all(np.isreal(eigval_1)))

#     eigs, vects, funcs, vol = Bayesian.fisher_decomposition(
#         {simid:filter_fisher},
#         simid,
#         opt_model.models,
#         'eos'
#     )

#     # Invert the degenerate eigenvalue matrix
#     for i in range(eigval_1.shape[0]):
#         if eigval_1[i]/eigval_1.max() > 1E-6 and eigval_1[i] > 1E-21:
#            eigval_1[i] = np.sqrt(eigval_1[i]**-1)
#         else:
#            eigval_1[i] = original_var[i,i]
#         # end
#     # end
#     print(eigval_1)

#     # Create the new variance matrix using the Cramer Rao bound
#     new_var = np.dot(eigvec_1,
#                      np.dot(np.diag(eigval_1),
#                             eigvec_1.T
#                      ))

#     # Use the prior variance when there is not information
#     #new_var = np.diag(np.where(np.diag(new_var) == 0.0, original_var, np.diag(new_var)))
#     eigval_2, eigvec_2 = nplin.eigh(new_var)

#     pcnt_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
#     dof_in = opt_model.models['eos'].get_dof()
#     dof_list = np.empty((9, opt_model.models['eos'].shape()))
#     sand_data_list = np.empty((9, opt_model.simulations[simid]['exp'].shape()))
#     sand_data_list_init = np.empty((9, opt_model.simulations[simid]['exp'].shape()))
#     models = copy.deepcopy(opt_model.models)

    knots = np.linspace(
        models['eos'].get_option('spline_min'),
        models['eos'].get_option('spline_max'),
        100)

#     # Evaluate the simulations at each credible level
#     for j, pcnt in enumerate(pcnt_list):
#         # Solve for the augmented variance
#         feasible = False
#         count = 0
#         while not feasible and count < 1:
#             scale = spstat.norm.ppf(pcnt, loc=0, scale=1)
#             #print(scale)
#             dof_list[j,:] = scale * np.diag(new_var) + dof_in
#             #dof_list[j, :] = np.array(
#             #    [spstat.norm.ppf(pcnt, loc=dof_in[i], scale=new_var[i, i])
#             #    for i in range(dof_in.shape[0])])

#             models['eos'] = models['eos'].update_dof(dof_list[j, :])
#             G, h = models['eos'].get_constraints(scale=False)
#             conval = np.dot((dof_list[j, :] - dof_in), G)
#             err = np.where(conval <= h, 0.0, conval - h )
#             feasible = np.all(err == 0)
#             if feasible and count>0:
#                 print(models['eos'].derivative(2)(knots))
#                 assert(np.all(models['eos'].derivative(2)(knots) > 0))
#                 #print(np.dot(err, err))
#             # if count%10 == 0 and not feasible:
#             #     print('no feasible point found in {:03d} draws'.format(count))
#             # if feasible:
#             #     print('feasible point found on draw {:03d}'.format(count))
#             count += 1
#             #feasible = True
#             #break
#         # end
#         data =opt_model.simulations[simid]['sim'](models)
#         sand_data_list[j, :] = opt_model.simulations[simid]['exp'].align(data)[1][0]

#         # Use the prior variance
#         tmp_dof = np.array(
#             [spstat.norm.ppf(pcnt, loc=dof_in[i], scale=original_var[i])
#              for i in range(dof_in.shape[0])])
#         models['eos'] = models['eos'].update_dof(tmp_dof)
#         data =opt_model.simulations[simid]['sim'](models)
#         sand_data_list_init[j, :] = opt_model.simulations[simid]['exp'].align(data)[1][0]
#     # end

    fig00 = plt.figure(figsize = fig_half)
    fig01 = plt.figure(figsize = fig_half)
    fig001 = plt.figure(figsize = fig_half)
    fig002 = plt.figure(figsize = fig_half)
    fig02 = plt.figure(figsize = fig_golden)
    fig03 = plt.figure()
    ax00 = fig00.gca()
    ax01 = fig01.gca()
    ax001 = fig001.gca()
    ax002 = fig002.gca()
    ax02 = fig02.gca()
    ax03 = fig03.gca()

    # true_knots = models['eos'].get_t()[:-4]
    # models['eos'] = models['eos'].update_dof(dof_in)
    # ax02.semilogy(true_knots, models['eos'].get_dof(), label='Value')
    # ax02.semilogy(true_knots, np.sqrt(np.diag(models['eos'].get_sigma())),
    #               label='Initial variance')
    # ax02.semilogy(true_knots, np.diag(new_var),
    #               label='Final variance')

    data = opt_model.simulations[simid]['exp']()
    styles = ['r:', 'r--', 'r-.', 'r-', 'k-', 'g-', 'g-.', 'g--', 'g:']
    for j, dof in enumerate(dof_list):
        print(dof_list[j])
        model = models['eos'].update_dof(dof_list[j])
        ax00.plot(1E6 * data[0], 1E-4 * data_list[j, :], styles[j],
                  label='Pcnt = {:f}'.format(.0))

    #     models['eos'] = models['eos'].update_dof(dof_list[j, :])
        ax01.semilogy(knots, 1E0 * model(knots), styles[j],
                  label='{:3.2f}'.format(0.0))

    #     ax03.plot(1E6 * data[0], 1E-4 * sand_data_list_init[j, :], styles[j],
    #               label='{:3.2f}'.format(pcnt))

    # # end

    # ax001.semilogy(eigs/1E9, 'sk')
    # for i in range(funcs.shape[0]):
    #     ax002.plot(vol, funcs[i],
    #              label="{:d}".format(i))
    # end
    # ax2.axvline(opt_model.models['eos'].state_cj.dens)
    # ax2.axvline(opt_model.models['eos'].get_option('rho_0'))
    ax001.xaxis.set_major_locator(MultipleLocator(1))
    ax001.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax001.set_xlabel('Eigenvalue rank')
    ax001.set_ylabel('Eigenvalue')
    ax002.set_xlabel(r'Density / g cm$^{-3}$')
    ax002.set_ylabel(r'Presure / Pa')
    #ax002.legend(loc='upper right', title='Eigenvalue\nrank')
    #ax002.get_legend().get_title().set_fontsize(str(rcParams['legend.fontsize']))
    ax00.set_xlabel(r'Time / $\mu$s')
    ax00.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')

    ax01.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
    ax01.set_ylabel(r'Pressure / Pa')
    ax01.legend(loc='upper right', title='Percentile')

    ax02.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
    ax02.set_ylabel(r'Pressure / Pa')
    ax02.legend(loc='upper right')

    ax03.set_xlabel(r'Time / $\mu$s')
    ax03.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')

    fig00.tight_layout()
    fig01.tight_layout()
    fig02.tight_layout()
    fig001.tight_layout()
    fig002.tight_layout()

    fig001.savefig("{:}/{:}_fisher_eval{:}".format(figdir, simid, figtype))
    fig002.savefig("{:}/{:}_fisher_efun{:}".format(figdir, simid, figtype))
    fig01.savefig("{:}/{:}_fisher_modvar{:}".format(figdir, simid, figtype))
    fig00.savefig("{:}/{:}_fisher_simvar{:}".format(figdir,simid, figtype))

    fig02.savefig("{:}/{:}_fisher_var{:}".format(figdir, simid, figtype))
    plt.close(fig00)
    plt.close(fig01)
    plt.close(fig001)
    plt.close(fig002)
    plt.close(fig002)
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
