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

from scipy.interpolate import LSQUnivariateSpline as LSQSpline
import scipy.fftpack as spfft

## F_UNLCE packages
from F_UNCLE.Experiments.Sandwich import ToySandwich, ToySandwichExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Experiments.Cylinder import ToyCylinder, ToyCylinderExperiment
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Models.SimpleStr import SimpleStr
from F_UNCLE.Opt.Bayesian import Bayesian
from F_UNCLE.Utils.NoiseModel import NoiseModel

## Make the models
str_model = SimpleStr([101E9, 100.0], sigma=0.05) # Youngs modulis for Cu from Hibbeler
str_true  = SimpleStr([101E9, 100.0]) # Youngs modulis for Cu from Hibbeler

eos_model = EOSModel(
    lambda v: 2.56E9/v**3, # famma=3 gas for HE
    spline_sigma = 0.05,
    spline_max = 2.0
)
eos_true = EOSBump()


## Make the simulations and experiments

rstate = np.random.RandomState()#seed=11223344) # Random state with fixed seed

sandwich_simulation = ToySandwich()
sandwich_experiment = ToySandwichExperiment(model=eos_true,
                                            rstate=rstate,
                                            exp_var=0.10,
                                            exp_corr=0.00)

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
models['eos'] = eos_true
models['strength'] = str_model

exp_data = simulations['Sand'][1]()
initial_data = simulations['Sand'][0](models)
aligned_data = simulations['Sand'][1].align(initial_data)


fig = plt.figure()

ax0 = fig.add_subplot(121)
ax01 = fig.add_subplot(122)
ax0.plot(1E6 * exp_data[0], 1E-4 * exp_data[1][0], label='exp')

var_list = np.linspace(0.05, 0.15, 10)
like_list = []
for var in var_list:
    simulations['Sand'][1].set_option('exp_corr', var)
    like = simulations['Sand'][1].get_log_like(aligned_data)
    like_list.append(like)
    #print(var, like)
# end

ax01.plot(var_list, like_list)
ax0.plot(1E6 * aligned_data[0], 1E-4 * aligned_data[1][0], label='sim')

ax0.set_xlabel(r'Time / $\mu$s')
ax0.set_ylabel(r'Error / cm $\mu$s$^{-1}$')
ax0.legend(loc='best')

ax01.set_xlabel(r'Correlation distance / $\mu$s')
ax01.set_ylabel(r'Log Likelihood / -')
fig.tight_layout()
fig.savefig('corrtest.pdf')

dx = exp_data[0].max() - exp_data[0].min()
knots = np.linspace(
    exp_data[0].min() + 0.01 * dx,
    exp_data[0].max() - 0.01 * dx,
    8
)

smooth =  LSQSpline(
    x=exp_data[0],
    y=exp_data[1][0],
    t=knots,
    w=None,
    k=3,
    ext=3,
    check_finite=True
)

#err = exp_data[1][0] - smooth(exp_data[0])
nmod = NoiseModel()
err = nmod(exp_data[0], exp_data[1][0], rstate=rstate)

N = exp_data[0].shape[0]
T = (exp_data[0].max() - exp_data[0].min()) / N

psd = spfft.fft(err)

psd_int = np.array([np.abs(psd[:i]).sum() for i in range(N//2)])
ppsd = spfft.fft(psd[:N//2]**2)[1:N//4]
fig2 = plt.figure()

ax20 = fig2.add_subplot(322)
ax21 = fig2.add_subplot(323)
ax22 = fig2.add_subplot(324)
ax23 = fig2.add_subplot(321)
ax24 = fig2.add_subplot(325)

ax23.plot(1E6 * exp_data[0],
          1E-4 * exp_data[1][0],
          label='exp')
ax23.plot(1E6 * exp_data[0],
          1E-4 * smooth(exp_data[0]),
          label='smoothed')

ax20.plot(1E6 * exp_data[0], 1E-4 * err)

ax23.set_xlabel(r'Time / $\mu$s')
ax23.set_ylabel(r'Velocity / cm $\mu$s$^{-1}$')
ax20.set_xlabel(r'Time / $\mu$s')
ax20.set_ylabel(r'Velocity Error / cm $\mu$s$^{-1}$')
ax23.legend(loc='best')

xf = np.linspace(0, 1E-6 * 1/(2 * T), N//2)
ax21.plot(xf, psd_int)
ax22.plot(np.abs(ppsd))
ax24.plot(xf, np.abs(psd[:N//2]))
#ax24.plot(spfft.ifft(psd**2))

ax24.set_ylabel("FFT Output")
ax24.set_xlabel("Frequency / MHz")
ax21.set_ylabel("FFT Integral")
ax21.set_xlabel("Frequency / MHz")
ax22.set_ylabel("FFT of FFT")
fig2.tight_layout()
fig2.savefig('fft.pdf')

analysis = Bayesian(
    simulations=simulations,
    models=models,
    opt_keys=['eos', 'strength'],
    constrain=True,
    outer_reltol=1E-6,
    precondition=True,
    debug=True,
    verb=True,
    sens_mode='ser',
    maxiter=6
)

## Run the analysis
# solve = False

# if solve:
#     to = time.time()
#     opt_model, history, sens_matrix, fisher_matrix  = analysis()
#     print('time taken ', to - time.time() )
#     print(opt_model.models['eos'])
#     print(opt_model.models['strength'])
#     # with open('opt_model.pkl', 'wb') as fid:
#     #     pickle.dump(opt_model, fid)

#     # with open('history.pkl', 'wb') as fid:
#     #     pickle.dump(history, fid)

#     # with open('sens_matrix.pkl', 'wb') as fid:
#     #     pickle.dump(sens_matrix, fid)
# else:
#     with open('opt_model.pkl', 'rb') as fid:
#         opt_model = pickle.load(fid)

#     with open('history.pkl', 'rb') as fid:
#         history = pickle.load(fid)

#     with open('sens_matrix.pkl', 'rb') as fid:
#         sens_matrix = pickle.load(fid)
# end
