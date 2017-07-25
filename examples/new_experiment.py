"""F_UNCLE Optimization of the new experiments
which more closely model true HE tests
"""

## Standard python packages
import sys
import os
import argparse
import pickle
import time

from collections import OrderedDict
## External python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
## F_UNLCE packages
from F_UNCLE.Experiments.Sandwich import ToySandwich, ToySandwichExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Experiments.Cylinder import ToyCylinder, ToyCylinderExperiment
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Models.SimpleStr import SimpleStr
from F_UNCLE.Opt.Bayesian import Bayesian


## Make the models
str_model = SimpleStr([101E9, 100.0]) # Youngs modulis for Cu from Hibbeler
str_true  = SimpleStr([101E9, 100.0]) # Youngs modulis for Cu from Hibbeler

eos_model = EOSModel(
    lambda v: 2.56E9/v**3, # famma=3 gas for HE
    spline_max = 2.0
)
eos_true = EOSBump()


## Make the simulations and experiments

rstate = np.random.RandomState(seed=1122548) # Random state with fixed seed

sandwich_simulation = ToySandwich()
sandwich_experiment = ToySandwichExperiment(model=eos_true, rstate=rstate )

stick_simulation = Stick(model_attribute=eos_true)
stick_experiment = StickExperiment(model=eos_true,
                                   sigma_t=1E-9,
                                   sigma_x=2E-3)

cylinder_simulation = ToyCylinder()
cylinder_experiment = ToyCylinderExperiment(
    models={'eos': eos_true, 'strength': str_true},
    rstate=rstate
    )


simulations = OrderedDict()
simulations['Sand'] =  [sandwich_simulation, sandwich_experiment]
#simulations['Stick'] = [stick_simulation, stick_experiment]
simulations['Cyl'] = [cylinder_simulation, cylinder_experiment]

models = OrderedDict()
models['eos'] = eos_model
models['strength'] = str_model

analysis = Bayesian(
    simulations=simulations,
    models=models,
    opt_keys=['eos'],#, 'strength'],
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
    opt_model, history, sens_matrix = analysis()
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

fisher_data = {}
# Get fisher info for everyone
fisher_all = np.empty((analysis.shape()[1], analysis.shape()[1]))
for key in simulations:
    fisher = simulations[key]['exp'].get_fisher_matrix(sens_matrix[key])
    fisher_all += fisher
    fisher_data[key] = fisher 
# end

fisher_data['All'] = fisher_all
with open('fisher_matrix.pkl', 'wb') as fid:
    pickle.dump(fisher_data, fid)
# end

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
