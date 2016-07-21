"""Script for re-generating the notes figures

"""

# Standard python packages
import sys
import os
import pdb

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
    gun_experiment = Gun(eos_true, mass_he=1.0)
    gun_simulation = Gun(eos_model, mass_he=1.0, sigma=1.0)

    stick_experiment = Stick(eos_true)
    stick_simulation = Stick(eos_model, sigma_t=1E-9, sigma_x=2E-3)


    # 4. Generage data from the simulations using the prior
    gun_prior_sim = gun_simulation()
    stick_prior_sim = stick_simulation()

    # 5. Create the analysis object
    analysis = Bayesian(simulations=[(gun_simulation, gun_experiment),
                                     (stick_simulation, stick_experiment)],
                        model=eos_model,
                        constrain=True,
                        outer_reltol=1E-6,
                        precondition=True,
                        debug=False,
                        maxiter=10)

    # 6. Run the analysis
    best_eos, history = analysis()


    # 7. Update the simulations and get new data
    gun_simulation.update(model=best_eos)
    g_time_s, (g_vel_s, g_pos_s), g_spline_s = gun_simulation()
    g_time_e, (g_vel_e, g_pos_e), g_spline_e = gun_experiment()

    stick_simulation.update(model=best_eos)
    s_pos_s, (s_time_s), s_data_s = stick_simulation()
    s_pos_e, (s_time_e), s_data_e = stick_experiment()


    ####################
    # Generate Figures #
    ####################

    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 7
    rcParams['legend.handlelength'] = 3.0

    pagewidth = 360 # pts
    au_ratio = (np.sqrt(5) - 1.0) / 2.0
    figwidth = 1.0 # fraction of \pagewidth for figure
    figwidth *= pagewidth/72.27
    figtype = '.pdf'
    out_dir = os.path.join('.', '..', '..',
                           'reports', 'notes', 'figures')+os.sep

    square = (figwidth, figwidth)
    tall = (figwidth, 1.25*figwidth)

    # Rayleigh line
    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    stick_simulation.plot(axis=f1ax1,
                          eos_style='-k',
                          ray_style=':k',
                          cj_style='xk')
    eos_model.prior.plot(axis=f1ax1, style='--b')
    eos_true.plot(axis=f1ax1, style='-.g')
    f1ax1.legend(['Fit EOS',
                  'Rayleigh line',
                  'CJ point',
                  r'($v_o$, $p_o$)',
                  'Prior EOS',
                  'True EOS'],
                 loc='best')

    fig1.tight_layout()
    fig1.savefig(out_dir+'rayl_line'+figtype, dpi=1000)

    # Nominal and true EOS
    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    eos_model.prior.plot(axis=f1ax1, style='--b')
    eos_true.plot(axis=f1ax1, style='-.g')
    f1ax1.legend(['Prior EOS',
                  'True EOS'])
    fig1.tight_layout()
    fig1.savefig(out_dir+'nom_true_eos'+figtype, dpi=1000)
    
    # Fisher information about the gun experiment
    fisher = analysis.get_fisher_matrix(simid=0, sens_calc=True)
    spec_data = analysis.fisher_decomposition(fisher)

    fig5 = analysis.plot_fisher_data(spec_data)
    fig5.set_size_inches(tall)
    fig5.tight_layout()
    fig5.savefig(out_dir+'info_gun'+figtype, dpi=1000)

    # Fisher information about the stick
    fisher = analysis.get_fisher_matrix(simid=1, sens_calc=True)
    spec_data = analysis.fisher_decomposition(fisher)

    fig2 = analysis.plot_fisher_data(spec_data)
    fig2.set_size_inches(tall)
    fig2.tight_layout()
    fig2.savefig(out_dir+'info_stick'+figtype, dpi=1000)

    # Results of the optimized stick simulation

    fig3 = plt.figure(figsize=square)
    f3ax1 = fig3.gca()

    stick_simulation.plot(axis=f3ax1, data_style='-k', level=2,
                          data=(s_pos_s, (s_time_s), s_data_s))


    stick_simulation.plot(axis=f3ax1, data_style='+g', level=2,
                          data=(s_pos_e, (s_time_e), s_data_e))

    stick_simulation.plot(axis=f3ax1, data_style='--b', level=2,
                          data=stick_prior_sim)

    f3ax1.legend(['Fit EOS', 'True EOS', 'Prior EOS'], loc='best')
    fig3.tight_layout()
    fig3.savefig(out_dir+'stick_results'+figtype, dpi=1000)

    # Results of the optimized gun simulation

    fig4 = plt.figure(figsize=tall)
    f4ax1 = fig4.add_subplot(211)
    f4ax2 = fig4.add_subplot(212)

    best_eos.plot(axis=f4ax1, style='-k')
    eos_model.prior.plot(axis=f4ax1, style='--b')
    eos_true.plot(axis=f4ax1, style='-.g')
    f4ax1.legend(['Fit EOS',
                  'Prior EOS',
                  'True EOS'], loc='best')

    gun_simulation.plot(axis=f4ax2, style='-k', err_style='-r',
                        data=[(g_time_s, (g_vel_s, g_pos_s), g_spline_s),
                              (g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axis=f4ax2, style='-.g',
                        data=[(g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    gun_simulation.plot(axis=f4ax2, style='--b',
                        data=[gun_prior_sim])
    f4ax2.plot([None],[None], '-r')
    f4ax2.legend(['Fit EOS',
                  'Prior EOS',
                  'True EOS',
                  'Error'], loc='upper left', framealpha = 0.5)

    fig4.tight_layout()
    fig4.savefig(out_dir+'gun_results'+figtype, dpi=1000)

    # Convergence history
    fig6 = plt.figure(figsize=square)
    f6a1 = fig6.gca()
    analysis.plot_convergence(history, axis=f6a1)
    fig6.tight_layout()
    fig6.savefig(out_dir+'conv'+figtype, dpi=1000)

    # Gun sensitivity plot
    fig7 = plt.figure(figsize=square)
    analysis.plot_sens_matrix(fig = fig7, simid = 0)
    fig7.savefig(out_dir+'gun_sens'+figtype, dpi=1000)

    # Stick sensitivity plot
    fig8 = plt.figure(figsize=square)
    analysis.plot_sens_matrix(fig = fig8, simid = 1)
    fig8.savefig(out_dir+'stick_sens'+figtype, dpi=1000)    


    # EOS basis functions
    fig9 = plt.figure(figsize=square)
    fig9 = eos_model.plot_basis(fig = fig9)
    fig9.savefig(out_dir+'eos_basis'+figtype, dpi=1000)    
    
    # Knots 

#    plt.show()
