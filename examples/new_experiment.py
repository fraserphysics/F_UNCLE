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
## External python packages
import numpy as np
import numpy.linalg as nplin

import scipy.stats as spstat

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rcParams
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['legend.handlelength'] = 3.0

## F_UNLCE packages
from F_UNCLE.Experiments.Sandwich import ToySandwich, ToySandwichExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Experiments.Cylinder import ToyCylinder, ToyCylinderExperiment
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Models.SimpleStr import SimpleStr
from F_UNCLE.Opt.Bayesian import Bayesian


## Make the models
str_model = SimpleStr([101E9, 100.0], sigma=0.05) # Youngs modulis for Cu from Hibbeler
str_true  = SimpleStr([101E9, 100.0]) # Youngs modulis for Cu from Hibbeler

eos_model = EOSModel(
    lambda v: 2.56E9/v**3, # famma=3 gas for HE
    spline_sigma = 0.25,
    spline_max = 2.0
)
eos_true = EOSBump()


## Make the simulations and experiments

rstate = np.random.RandomState(seed=1122548) # Random state with fixed seed

sandwich_simulation = ToySandwich()
sandwich_experiment = ToySandwichExperiment(model=eos_true, rstate=rstate )

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
simulations['Sand'] =  [sandwich_simulation, sandwich_experiment]
simulations['Stick'] = [stick_simulation, stick_experiment]
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
    debug=True,
    verb=True,
    sens_mode='ser',
    maxiter=6
)

## Run the analysis
solve = True

if solve:
    to = time.time()
    opt_model, history, sens_matrix, fisher_matrix  = analysis()
    print('time taken ', to - time.time() )
    print(opt_model.models['eos'])
    print(opt_model.models['strength'])
    # with open('opt_model.pkl', 'wb') as fid:
    #     pickle.dump(opt_model, fid)

    # with open('history.pkl', 'wb') as fid:
    #     pickle.dump(history, fid)

    # with open('sens_matrix.pkl', 'wb') as fid:
    #     pickle.dump(sens_matrix, fid)
else:
    with open('opt_model.pkl', 'rb') as fid:
        opt_model = pickle.load(fid)

    with open('history.pkl', 'rb') as fid:
        history = pickle.load(fid)

    with open('sens_matrix.pkl', 'rb') as fid:
        sens_matrix = pickle.load(fid)
# end

simid = 'Cyl'
filter_fisher = np.where(fisher_matrix[simid]<1E21, fisher_matrix[simid], 0.0)
original_var = np.sqrt(np.diag(opt_model.models['eos'].get_sigma()))


# sigma_aug = filter_fisher + nplin.inv(opt_model.models['eos'].get_sigma())
# eigval_1, eigvec_1 = nplin.eig(sigma_aug)
# assert(np.all(eigval_1 > 0.0))
# assert(np.all(np.isreal(eigval_1)))

# for i in range(eigval_1.shape[0]):
#     if eigval_1[i] > 1E-21:
#        eigval_1[i] = eigval_1[i]**-1
#     else:
#        eigval_1[i] = 0
#     # end
# # end
# new_var = np.sqrt(np.dot(eigvec_1,
#                  np.dot(np.diag(eigval_1),
#                         nplin.inv(eigvec_1)
#                  )
# ))


new_var = np.diag(np.sqrt(np.diag(
    nplin.inv(filter_fisher + nplin.inv(opt_model.models['eos'].get_sigma())))))

dof_in = opt_model.models['eos'].get_dof()
dof_list = np.empty((9, opt_model.models['eos'].shape()))
sand_data_list = np.empty((9, opt_model.simulations[simid]['exp'].shape()))
sand_data_list_init = np.empty((9, opt_model.simulations[simid]['exp'].shape()))
models = copy.deepcopy(opt_model.models)
knots = np.linspace(
    models['eos'].get_option('spline_min'),
    models['eos'].get_option('spline_max'),
    100)
fig0 = plt.figure()
ax00 = fig0.add_subplot(222)
ax01 = fig0.add_subplot(223)
ax02 = fig0.add_subplot(224)
ax03 = fig0.add_subplot(221)

true_knots = models['eos'].get_t()[:-4]
ax02.semilogy(true_knots, models['eos'].get_dof(), label='Value')
ax02.semilogy(true_knots, np.sqrt(np.diag(models['eos'].get_sigma())),
              label='Initial variance')
ax02.semilogy(true_knots, np.diag(new_var),
              label='Final variance')

styles = ['r:', 'r--', 'r-.', 'r-', 'k-', 'g-', 'g-.', 'g--', 'g:']
for j, pcnt in enumerate([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]):
    # Solve for the augmented variance
    dof_list[j,:] = np.array(
        [spstat.norm.ppf(pcnt, loc=dof_in[i], scale=new_var[i, i])
         for i in range(dof_in.shape[0])])
    
    models['eos'] = models['eos'].update_dof(dof_list[j, :])
    data =opt_model.simulations[simid]['sim'](models)
    sand_data_list[j, :] = opt_model.simulations[simid]['exp'].align(data)[1][0]
    ax00.plot(1E6 * data[0], 1E-4 * sand_data_list[j, :], styles[j],
              label='Pcnt = {:f}'.format(pcnt))
    
    ax01.semilogy(knots, 1E0 * models['eos'](knots), styles[j],
              label='{:3.2f}'.format(pcnt))    

    # Use the prior variance
    dof_list[j,:] = np.array(
        [spstat.norm.ppf(pcnt, loc=dof_in[i], scale=original_var[i])
         for i in range(dof_in.shape[0])])
    
    models['eos'] = models['eos'].update_dof(dof_list[j, :])
    data =opt_model.simulations[simid]['sim'](models)
    sand_data_list_init[j, :] = opt_model.simulations[simid]['exp'].align(data)[1][0]
    ax03.plot(1E6 * data[0], 1E-4 * sand_data_list_init[j, :], styles[j],
              label='{:3.2f}'.format(pcnt))
    
# end


ax00.set_xlabel(r'Time / $\mu$s')
ax00.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')
ax00.set_title('With fisher informaiton')

ax01.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
ax01.set_ylabel(r'Pressure / Pa')
ax01.legend(loc='upper right', title='Percentile')
ax01.set_title('Equation of State')

ax02.set_xlabel(r'Specific Volume / cm$^{3}$ g$^{-1}$')
ax02.set_ylabel(r'Pressure / Pa')
ax02.legend(loc='upper right')
ax02.set_title('Variances')

ax03.set_xlabel(r'Time / $\mu$s')
ax03.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')
ax03.set_title('Prior variance')


fig0.tight_layout()
fig0.savefig('percentiles.pdf')

    

# for key in fisher_data:
#     fisher = fisher_data[key]
#     eigs, vects, funcs, vol = Bayesian.fisher_decomposition(
#         fisher, opt_model.models['eos']
#     )

#     fig = plt.figure()
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212)

#     ax1.semilogy(eigs/1E9, 'sk')

#     for i in range(funcs.shape[0]):
#         ax2.plot(vol, funcs[i],
#                  label="{:d}".format(i))
#     # end
#     # ax2.axvline(opt_model.models['eos'].state_cj.dens)
#     # ax2.axvline(opt_model.models['eos'].get_option('rho_0'))
#     ax1.xaxis.set_major_locator(MultipleLocator(1))
#     ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

#     ax1.set_xlabel('Eigenvalue rank')
#     ax1.set_ylabel('Eigenvalue')
#     ax2.set_xlabel(r'Density / g cm$^{-3}$')
#     ax2.set_ylabel(r'Presure / Pa')        
#     fig.tight_layout()
#     fig.savefig("{:}_fisher.eps".format(key))
#     plt.close(fig)
# # end   


## Plot the EOS
fig = eos_model.prior.plot(labels=['Prior'], linestyles = ['-k'])
fig = opt_model.models['eos'].plot(figure=fig, labels = ['Postrior'], linestyles = ['--r'])
fig = eos_true.plot(figure=fig, labels = ['True'], linestyles = ['-.g'])
ax1 = fig.gca()
ax1.set_xlim((0.1, 0.75))
ax1.set_ylim((0.0, 0.5E12))
ax1.legend(loc='best')
fig.savefig('eos_comparisson.pdf')


## Get the data for the sandwich
sand_prior = sandwich_simulation({'eos': eos_model.prior})
sand_post = sandwich_simulation(opt_model.models)
sand_exp = sandwich_experiment()
fig = sandwich_simulation.plot(data=[sand_prior, sand_post, sand_exp],
                         labels=['Prior', 'Posterior', 'Experiment'],
                         linestyles=['-k', '--r', '-.g'])

fig.savefig('sand_results.pdf')

## Get the data for the cylinder
cyl_prior = cylinder_simulation({'eos': eos_model.prior, 'strength': str_true})
cyl_post = cylinder_simulation(opt_model.models)
cyl_exp = cylinder_experiment()

fig = cylinder_simulation.plot(data=[cyl_prior, cyl_post, cyl_exp],
                         labels=['Prior', 'Posterior', 'Experiment'],
                         linestyles=['-k', '--r', '-.g'])

fig.savefig('cyl_results.pdf')


# ## Plot the sensitivity of the cylinder to isen
# fig = EOSModel.plot_sens_matrix(sens_matrix, 'Cyl', opt_model.models, 'eos')
# fig.savefig('cyl_eos_sens.pdf')

# fig = SimpleStr.plot_sens_matrix(sens_matrix, 'Cyl', opt_model.models, 'strength')
# fig.savefig('cyl_str_sens.pdf')

# ## Plot the sensitivity of the sandwich to isen
# fig = EOSModel.plot_sens_matrix(sens_matrix, 'Sand', opt_model.models, 'eos')
# fig.savefig('sand_eos_sens.pdf')

# fig = SimpleStr.plot_sens_matrix(sens_matrix, 'Sand', opt_model.models, 'strength')
# fig.savefig('sand_str_sens.pdf')


## Get the fisher information
def make_fisher_image(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.semilogy(data[0], 'sk')
    for i in range(data[2].shape[0]):
        ax2.plot(data[3], data[2][i])
    # end
    return fig

data = Bayesian.fisher_decomposition(fisher_matrix, 'Cyl', opt_model.models, 'eos')
fig = make_fisher_image(data)
fig.savefig('cyl_eos_info.pdf')


data = Bayesian.fisher_decomposition(fisher_matrix, 'Sand', opt_model.models, 'eos')
fig = make_fisher_image(data)
fig.savefig('sand_eos_info.pdf')
