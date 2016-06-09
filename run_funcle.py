import sys
import os
import copy
import pdb

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath('./../'))

from F_UNCLE.Experiments.GunModel import Gun
from F_UNCLE.Experiments.Stick import Stick
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Opt.Bayesian import Bayesian

# Initial estimate of prior functional form
init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

# Create the model and *true* EOS
eos_model = EOSModel(init_prior)
eos_true = EOSBump()

# Create the objects to generate simulations and pseudo experimental data
gun_experiment = Gun(eos_true)
gun_simulation = Gun(eos_model)
gun_prior_sim =gun_simulation()

stick_experiment = Stick(eos_true)
stick_simulation = Stick(eos_model)

stick_prior_sim = stick_simulation()
# Create the analysis object
analysis = Bayesian(simulations = [(gun_simulation, gun_experiment)],
#                                   (stick_simulation, stick_experiment)],
                    model = eos_model, constrain = True, precondition = False)

# Run the analysis
best_eos, history = analysis()


plt.figure()
ax1 = plt.gca()
ax1.plot(xrange(len(history)), history)
ax1.set_title('Log likelyhood convergence')
ax1.set_xlabel('Iteration number')
ax1.set_ylabel('A posteriori log likelyhood')

plt.savefig('log_likelyhood.pdf')

v_min = eos_model.get_option('spline_min')
v_max = eos_model.get_option('spline_max')

v = np.logspace(np.log10(v_min), np.log10(v_max), 50)
plt.figure()
ax1 = plt.gca()
ax1.plot(v, best_eos(v)-best_eos.prior(v), '--r', label = 'Best EOS model')
# ax1.plot(v, best_eos.prior(v), label = 'Prior')
ax1.plot(v, eos_true(v) - best_eos.prior(v), '-.k', label ='True EOS')
ax1.set_title('The Equation of state models')
ax1.set_xlabel('Specific volume')
ax1.set_ylabel('Pressure')
ax1.legend(loc = 'best')

plt.savefig('eos_models.pdf')

gun_simulation.update(model = best_eos)
time_s, (vel_s, pos_s), spline_s = gun_simulation()
time_e, (vel_e, pos_e), spline_e = gun_experiment()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
l1 = ax1.plot(gun_prior_sim[0], gun_prior_sim[1][0], '-b', label = 'Prior EOS')
l2 = ax1.plot(time_s, vel_s, '--r', label = 'Best EOS')
l3 = ax1.plot(time_e, vel_e, '-.k', label = 'True EOS')
l4 = ax2.plot(time_e, vel_e - spline_s(time_e), '-g', label = 'Final error')
l5 = ax2.plot(time_e, vel_e - gun_prior_sim[2](time_e), '-.g', label = 'Prior error')         

ax1.legend([l1[0],l2[0],l3[0],l4[0],l5[0]],
           ['Prior EOS',
            'Best EOS',
            'True EOS',
            'Final Error',
            'Prior Error'], loc = 'best')

ax1.set_title('Velocity histories')
ax1.set_xlabel('Time')
ax1.set_ylabel('Velocity')
ax2.set_ylabel('Error')

plt.savefig('vel_hist.pdf')

# stick_simulation.update(model = best_eos)

# pos_s, (time_s), data_s = stick_simulation()
# pos_e, (time_e), data_e = stick_experiment()


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(pos_s, time_s, '-k', label = 'Best EOS')
# ax1.plot(pos_e, time_e, 'ok', label = 'Experiment')
# ax1.plot(stick_prior_sim[0], stick_prior_sim[1][0], '-b', label = 'Prior')
# ax1.legend(loc = 'best')
plt.show()
