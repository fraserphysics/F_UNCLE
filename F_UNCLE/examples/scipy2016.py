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

.. figure:: /_static/scipy2016_figure1eos.png

   Figure 0

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
import time

# External python packages
import numpy as np
import matplotlib.pyplot as plt

# F_UNLCE packages
if __name__ == '__main__':
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Experiments.GunModel import Gun
    from F_UNCLE.Experiments.Stick import Stick
    from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
    from F_UNCLE.Opt.Bayesian import Bayesian
else:
    from ..Experiments.GunModel import Gun
    from ..Experiments.Stick import Stick
    from ..Models.Isentrope import EOSModel, EOSBump
    from ..Opt.Bayesian import Bayesian
#end

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

    # 4. Create the analysis object
    analysis = Bayesian(
        simulations={
            'Gun': [gun_simulation, gun_experiment],
            'Stick': [stick_simulation, stick_experiment],
        },
        models={'eos': eos_model},
        opt_key='eos',
        constrain=True,
        outer_reltol=1E-6,
        precondition=True,
        debug=False,
        verb=True,
        sens_mode='ser',
        maxiter=6)

    # 5. Generage data from the simulations using the prior
    gun_prior_sim = gun_simulation(analysis.models)
    stick_prior_sim = stick_simulation(analysis.models)

    
    # 6. Run the analysis
    to = time.time()
    opt_model, history, sens_matrix = analysis()
    print('time taken ', to - time.time() )


    # 7. Update the simulations and get new data
    g_time_s, (g_vel_s, g_pos_s), g_spline_s =\
        opt_model.simulations['Gun']['sim'](opt_model.models)
    g_time_e, (g_vel_e, g_pos_e), g_spline_e =\
        opt_model.simulations['Gun']['exp']()

    s_pos_s, (s_time_s), s_data_s =\
        opt_model.simulations['Stick']['sim'](opt_model.models)
    s_pos_e, (s_time_e), s_data_e = opt_model.simulations['Stick']['exp']()


    ####################
    # Generate Figures #
    ####################

    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 7
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['legend.handlelength'] = 3.0

    pagewidth = 253 # pts
    au_ratio = (np.sqrt(5) - 1.0) / 2.0
    figwidth = 1.0 # fraction of \pagewidth for figure
    figwidth *= pagewidth/72.27
    figtype = '.pdf'
    out_dir = os.path.join('.', '..', '..',
                           'reports', 'poster', 'figures')+os.sep
    #out_dir = '/Users/saandrews/Documents/projects/tmp/scipy_proceedings/papers/andrew_fraser/'
    square = (figwidth, figwidth)
    tall = (figwidth, 1.25*figwidth)

    # Figure 1
    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    opt_model.simulations['Stick']['sim'].\
        plot(opt_model.models, axes=f1ax1)

    eos_model.prior.plot(axes=f1ax1, linestyles=['--b'], labels=['Prior EOS'])
    eos_true.plot(axes=f1ax1, linestyles=['-.g'], labels=['True EOS'])
    f1ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(out_dir+'scipy2016_figure1'+figtype, dpi=1000)

    # Figure 1
    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    eos_model.prior.plot(axes=f1ax1, linestyles=['--b'], labels=['Prior EOS'])
    eos_true.plot(axes=f1ax1, linestyles=['-.g'], labels=['True EOS'])
    f1ax1.legend(loc='best')
    fig1.tight_layout()
    fig1.savefig(out_dir+'scipy2016_figure1eos'+figtype, dpi=1000)

    # Figure 5
    fisher = opt_model.simulations['Gun']['sim'].\
        get_fisher_matrix(opt_model.models,
                          use_hessian=False,
                          exp=opt_model.simulations['Gun']['exp'],
                          sens_matrix=sens_matrix['Gun'])

    spec_data = opt_model.fisher_decomposition(fisher)

    fig5 = plt.figure(figsize=tall)
    fig5 = opt_model.plot_fisher_data(spec_data, fig=fig5)
    fig5.set_size_inches(tall)
    fig5.tight_layout()
    fig5.savefig(out_dir+'scipy2016_figure5'+figtype, dpi=1000)

    # Figure 2

    fisher = opt_model.simulations['Stick']['sim'].\
        get_fisher_matrix(opt_model.models,
                          sens_matrix=sens_matrix['Stick'])
    spec_data = opt_model.fisher_decomposition(fisher)

    fig2 = plt.figure(figsize=tall)
    fig2 = opt_model.plot_fisher_data(spec_data, fig=fig2)
    fig2.tight_layout()

    fig2.axes[1].axvline(s_data_s[1])
    fig2.axes[1].annotate(r'$v_{CJ}$',
                          xy=(s_data_s[1],0),
                          xytext=(30,30),
                          xycoords='data',
                          textcoords='offset points',
                          arrowprops=dict(facecolor='black',
                                          arrowstyle='->'))
    fig2.set_size_inches(tall)
    fig2.tight_layout()
    fig2.savefig(out_dir+'scipy2016_figure2'+figtype, dpi=1000)

    # Figure 3

    fig3 = plt.figure(figsize=square)
    f3ax1 = fig3.gca()

    stick_simulation.plot(opt_model.models,
                          axes=f3ax1, linestyles=['-k'],
                          labels=['Fit EOS'], level=2,
                          data=(s_pos_s, (s_time_s), s_data_s))

    stick_simulation.plot(opt_model.models,
                          axes=f3ax1, linestyles=['+g'],
                          labels=['True EOS'], level=2,
                          data=(s_pos_e, (s_time_e), s_data_e))

    stick_simulation.plot(opt_model.models,
                          axes=f3ax1, linestyles=['--b'],
                          labels=['Prior EOS'], level=2,
                          data=stick_prior_sim)
    f3ax1.legend(loc='best')
    fig3.tight_layout()
    fig3.savefig(out_dir+'scipy2016_figure3'+figtype, dpi=1000)

    # Figure 4

    fig4 = plt.figure(figsize=tall)
    f4ax1 = fig4.add_subplot(211)
    f4ax2 = fig4.add_subplot(212)

    opt_model.models['eos'].plot(axes=f4ax1, linestyles=['-k'],
                                 labels=['Fit EOS'])
    eos_model.prior.plot(axes=f4ax1, linestyles=['--b'],
                         labels=['Prior EOS'])
    eos_true.plot(axes=f4ax1, linestyles=['-.g'], labels=['True EOS'])
    f4ax1.legend(loc='best')

    gun_simulation.plot(axes=f4ax2, linestyles=['-k', '-r'],
                        labels=['Fit EOS', 'Error'],
                        data=[(g_time_s, (g_vel_s, g_pos_s), g_spline_s),
                              (g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axes=f4ax2, linestyles=['-.g'], labels=['True EOS'],
                        data=[(g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axes=f4ax2, linestyles=['--b'], labels=['Prior EOS'],
                        data=[gun_prior_sim])
    f4ax2.legend(loc='upper left', framealpha=0.5)

    fig4.tight_layout()

    fig4.tight_layout()
    fig4.savefig(out_dir+'scipy2016_figure4'+figtype, dpi=1000)

    # Figure 6

    fig6 = plt.figure(figsize=square)
    f6a1 = fig6.gca()
    opt_model.plot_convergence(history, axes=f6a1)
    fig6.tight_layout()
    fig6.savefig(out_dir+'scipy2016_figure6'+figtype, dpi=1000)

#    plt.show()
