"""Script for re-generating the figures used in scipy2016

This script generates the figures used in ref 1

**Dependencies**

- python 2.7
- numpy
- matplotlib
- scipy
- cvxopt

**Useage**

from the command line::

   $ python scipy2016.py

**References**

1. Fraser, A. M. and Andrews, S. A. 2016 "Functional Uncertanty Constrained by
   Law and Experiment." Proceedings of the 15th Python in Science Conference
   LA-UR-23717


Figures
-------

.. figure:: /_static/scipy2016_figure1.png

   Figure 1

.. figure:: /_static/scipy2016_figure2.png

   Figure 2

.. figure:: /_static/scipy2016_figure3.png

   Figure 3

.. figure:: /_static/scipy2016_figure4.png

   Figure 4

.. figure:: /_static/scipy2016_figure5.png

   Figure 5

.. figure:: /_static/scipy2016_figure6.png

   Figure 6


"""

# Standard python packages

import sys
import os
import pdb

# External python packages

import numpy as np
import matplotlib.pyplot as plt

# F_UNLCE packages
sys.path.append(os.path.abspath('./../'))
from F_UNCLE.Experiments.GunModel import Gun
from F_UNCLE.Experiments.Stick import Stick
from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
from F_UNCLE.Opt.Bayesian import Bayesian

if __name__ == '__main__':
    #################
    #    Get Data   #
    #################

    # 1. Generate a functional form for the prior
    init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

    # 2. Create the model and *true* EOS
    eos_model = EOSModel(init_prior)
    #eos_true = EOSBump(bumps=[], const_C=2.4e9)
    eos_true = EOSBump()
    
    # 3. Create the objects to generate simulations and pseudo experimental data
    gun_experiment = Gun(eos_true, mass_he=1.0)
    gun_simulation = Gun(eos_model, mass_he=1.0)

    stick_experiment = Stick(eos_true)
    stick_simulation = Stick(eos_model)

    # 4. Generage data from the simulations using the prior
    gun_prior_sim = gun_simulation ()
    stick_prior_sim = stick_simulation()

    # 5. Create the analysis object
    analysis = Bayesian(simulations=[(gun_simulation, gun_experiment),
                                     (stick_simulation, stick_experiment)],
                        model=eos_model,
                        prior_weight=1.0,
                        constrain=True,
                        precondition=True,
                        maxiter=10)

    # 6. Run the analysis
    best_eos, history = analysis()


    # 7. Update the simulations and get new data
    gun_simulation.update(model = best_eos)
    g_time_s, (g_vel_s, g_pos_s), g_spline_s = gun_simulation()
    g_time_e, (g_vel_e, g_pos_e), g_spline_e = gun_experiment()

    stick_simulation.update(model = best_eos)
    s_pos_s, (s_time_s), s_data_s = stick_simulation()
    s_pos_e, (s_time_e), s_data_e = stick_experiment()


    ####################
    # Generate Figures #
    ####################

    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 7
    rcParams['legend.handlelength'] = 3.0

    pagewidth = 253 # pts
    au_ratio = (np.sqrt(5) - 1.0) / 2.0
    figwidth = 1.0 # fraction of \pagewidth for figure
    figwidth *= pagewidth/72.27
    figtype = '.png'

    square = (figwidth, figwidth)
    tall = (figwidth, 1.25*figwidth)

    # Figure 1

    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    stick_simulation.plot(axis=f1ax1,
                          eos_style='-k',
                          ray_style=':k',
                          cj_style='ok')
    eos_model.prior.plot(axis=f1ax1, style='--k')
    eos_true.plot(axis=f1ax1, style='-.k')
    f1ax1.legend(['Fit EOS',
                  'Rayleigh line',
                  'CJ point',
                  'Prior EOS',
                  'True EOS'],
                 loc='best')

    fig1.tight_layout()
    fig1.savefig('scipy2016_figure1'+figtype, dpi = 1000)

    # Figure 5

    fisher = analysis.get_fisher_matrix(simid=0, sens_calc=True)
    spec_data = analysis.fisher_decomposition(fisher)

    fig5 = analysis.plot_fisher_data(spec_data)
    fig5.set_size_inches(tall)
    fig5.tight_layout()
    fig5.savefig('scipy2016_figure5'+figtype, dpi = 1000)
    
    # Figure 2

    fisher = analysis.get_fisher_matrix(simid=1, sens_calc=True)
    spec_data = analysis.fisher_decomposition(fisher)

    fig2 = analysis.plot_fisher_data(spec_data)
    fig2.set_size_inches(tall)
    fig2.savefig('scipy2016_figure2'+figtype, dpi = 1000)

    # Figure 3

    fig3 = plt.figure(figsize=square)
    f3ax1 = fig3.gca()

    stick_simulation.plot(axis=f3ax1, data_style='-k', level=2,
                          data=(s_pos_s, (s_time_s), s_data_s))


    stick_simulation.plot(axis=f3ax1, data_style='+k', level=2,
                          data=(s_pos_e, (s_time_e), s_data_e))


    stick_simulation.plot(axis=f3ax1, data_style='--k', level=2,
                          data=stick_prior_sim)

    f3ax1.legend(['Fit EOS', 'True EOS', 'Prior EOS'], loc='best')

    fig3.savefig('scipy2016_figure3'+figtype, dpi = 1000)

    # Figure 4

    fig4 = plt.figure(figsize = tall)
    f4ax1 = fig4.add_subplot(211)
    f4ax2 = fig4.add_subplot(212)

    best_eos.plot(axis=f4ax1, style='-k')
    eos_model.prior.plot(axis=f4ax1, style='--k')
    eos_true.plot(axis=f4ax1, style='-.k')
    f4ax1.legend(['Fit EOS',
                  'Prior EOS',
                  'True EOS'], loc='best')


    gun_simulation.plot(axis=f4ax2, style='-k' , err_style='-r',
                        data=[(g_time_s, (g_vel_s, g_pos_s), g_spline_s),
                                (g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axis=f4ax2, style='-.k' ,
                        data=[(g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axis=f4ax2, style='--k' ,
                        data=[gun_prior_sim])
    fig4.tight_layout()
    fig4.savefig('scipy2016_figure4'+figtype, dpi = 1000)


    # Figure 6

    fig6 = plt.figure(figsize=square)
    f6a1 = fig6.gca()
    analysis.plot_convergence(history, axis=f6a1)
    fig6.tight_layout()
    fig6.savefig('scipy2016_figure6'+figtype, dpi = 1000)

#    plt.show()
