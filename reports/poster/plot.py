"""plot.py makes plots for notes.pdf.

"""
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
import sys
import os
import matplotlib as mpl
import numpy as np
import time

pagewidth = 500 # pts
au_ratio = (np.sqrt(5) - 1.0) / 2.0
figwidth = 1.0 # fraction of \pagewidth for figure
figwidth *= pagewidth/72.27
figtype = '.pdf'
square = (figwidth, figwidth)
tall = (figwidth, 1.25*figwidth)

def basis(plt, sim):
    sys.path.append(os.path.abspath('./../../'))
    from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
    fig = plt.figure()
    EOSModel(lambda x: 1/x**3, spline_N=10).plot_basis(fig=fig)
    return fig
plot_dict['basis'] = basis

def stick_xt(plt, sim):
    fig3 = plt.figure(figsize=square)
    f3ax1 = fig3.gca()

    sim.stick_simulation.plot(sim.opt_model.models,
                          axes=f3ax1, linestyles=['-k'],
                          labels=['Fit EOS'], level=2,
                          data=(sim.s_pos_s, (sim.s_time_s), sim.s_data_s))

    sim.stick_simulation.plot(sim.opt_model.models,
                          axes=f3ax1, linestyles=['+g'],
                          labels=['True EOS'], level=2,
                          data=(sim.s_pos_e, (sim.s_time_e), sim.s_data_e))

    sim.stick_simulation.plot(sim.opt_model.models,
                          axes=f3ax1, linestyles=['--b'],
                          labels=['Prior EOS'], level=2,
                          data=sim.stick_prior_sim)
    f3ax1.legend(loc='best')
    fig3.tight_layout()
    return fig3

plot_dict['stick_xt'] = stick_xt

def stick_fisher(plt, sim):
    fisher = sim.opt_model.simulations['Stick']['sim'].\
        get_fisher_matrix(sim.opt_model.models,
                          sens_matrix=sim.sens_matrix['Stick'])
    spec_data = sim.opt_model.fisher_decomposition(fisher)

    fig2 = plt.figure(figsize=tall)
    fig2 = sim.opt_model.plot_fisher_data(spec_data, fig=fig2)
    fig2.tight_layout()

    fig2.axes[1].axvline(sim.s_data_s[1])
    fig2.axes[1].annotate(r'$v_{CJ}$',
                          xy=(sim.s_data_s[1],0),
                          xytext=(30,30),
                          xycoords='data',
                          textcoords='offset points',
                          arrowprops=dict(facecolor='black',
                                          arrowstyle='->'))
    fig2.set_size_inches(tall)
    fig2.tight_layout()
    return fig2
plot_dict['stick_fisher'] = stick_fisher

def stick_CJ(plt, sim):
    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    vrange=(.2, .5)
    sim.opt_model.simulations['Stick']['sim'].\
        plot(sim.opt_model.models,
             axes=f1ax1,
             vrange=vrange)

    sim.eos_model.prior.plot(
        axes=f1ax1,
        linestyles=['--b'],
        labels=['Prior EOS'],
        vrange=vrange
    )
    sim.eos_true.plot(
        axes=f1ax1,
        linestyles=['-.g'],
        labels=['True EOS'],
        vrange=vrange

    )
    f1ax1.legend(loc='best')
    fig1.tight_layout()
    return fig1
plot_dict['stick_CJ'] = stick_CJ

def eos_nom_true(plt, sim):
    fig1 = plt.figure(figsize=square)
    f1ax1 = fig1.gca()
    vrange=(.2, .5)        
    sim.eos_model.prior.plot(
        axes=f1ax1,
        linestyles=['--b'],
        labels=['Prior EOS'],
        vrange=vrange
    )
    sim.eos_true.plot(
        axes=f1ax1,
        linestyles=['-.g'],
        labels=['True EOS'],
        vrange=vrange
    )
    f1ax1.legend(loc='best')
    fig1.tight_layout()
    return fig1
plot_dict['eos_nom_true'] = eos_nom_true

def gun_tv(plt, sim):
    fig4 = plt.figure(figsize=tall)
    f4ax1 = fig4.add_subplot(211)
    f4ax2 = fig4.add_subplot(212)
    vrange=(.2, .5)
    sim.opt_model.models['eos'].plot(axes=f4ax1,
                                     linestyles=['-k'],
                                     labels=['Fit EOS'],
                                     vrange=vrange)
    sim.eos_model.prior.plot(axes=f4ax1,
                             linestyles=['--b'],
                             labels=['Prior EOS'],
                             vrange=vrange)
    sim.eos_true.plot(axes=f4ax1,
                      linestyles=['-.g'],
                      labels=['True EOS'],
                      vrange=vrange)
    f4ax1.legend(loc='best')

    sim.gun_simulation.plot(axes=f4ax2, linestyles=['-k', '-r'],
                        labels=['Fit EOS', 'Error'],
                        data=[(sim.g_time_s,
                               (sim.g_vel_s, sim.g_pos_s),
                               sim.g_spline_s),
                              (sim.g_time_e,
                               (sim.g_vel_e, sim.g_pos_e),
                               sim.g_spline_e)])

    sim.gun_simulation.plot(axes=f4ax2, linestyles=['-.g'], labels=['True EOS'],
                        data=[(sim.g_time_e,
                               (sim.g_vel_e, sim.g_pos_e),
                               sim.g_spline_e)])

    sim.gun_simulation.plot(axes=f4ax2, linestyles=['--b'], labels=['Prior EOS'],
                        data=[sim.gun_prior_sim])
    f4ax2.legend(loc='upper left', framealpha=0.5)
    fig4.tight_layout()
    return fig4
plot_dict['gun_tv'] = gun_tv

def gun_fisher(plt, sim):
    fisher = sim.opt_model.simulations['Gun']['sim'].\
        get_fisher_matrix(sim.opt_model.models,
                          use_hessian=False,
                          exp=sim.opt_model.simulations['Gun']['exp'],
                          sens_matrix=sim.sens_matrix['Gun'])

    spec_data = sim.opt_model.fisher_decomposition(fisher)

    fig5 = plt.figure(figsize=tall)
    fig5 = sim.opt_model.plot_fisher_data(spec_data, fig=fig5)
    fig5.set_size_inches(tall)
    fig5.tight_layout()
    return fig5
plot_dict['gun_fisher'] = gun_fisher

class Sim():
    def __init__(self):
        '''Suggested improvements:

        self to have attributes: stick, gun, eos_model, eos_true and
        eos_best.  Then self.stick.update(eos_true) returns new stick
        instance with specified eos.  The call also leaves self.stick
        unmodified.

        Take plotting functions out of objects created here and have
        methods of those objects return data for plotting.  If those
        plotting functions are useful for debugging, leave them in, but
        don't use them here.
        '''
        import sys, os
        sys.path.append(os.path.abspath('./../../'))
        from F_UNCLE.Experiments.GunModel import Gun
        from F_UNCLE.Experiments.Stick import Stick
        from F_UNCLE.Models.Isentrope import EOSModel, EOSBump
        from F_UNCLE.Opt.Bayesian import Bayesian

        # 1. Generate a functional form for the prior
        init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

        # 2. Create the model and *true* EOS
        self.eos_model = EOSModel(init_prior, Spline_sigma=0.05)
        self.eos_true = EOSBump()

        # 3. Create the objects to generate simulations and pseudo
        #    experimental data
        self.gun_experiment = Gun(model_attribute=self.eos_true, mass_he=1.0)
        self.gun_simulation = Gun(mass_he=1.0, sigma=1.0)

        self.stick_experiment = Stick(model_attribute=self.eos_true)
        self.stick_simulation = Stick(sigma_t=1E-9, sigma_x=2E-3)

        # 4. Create the analysis object
        analysis = Bayesian(
            simulations={
                'Gun': [self.gun_simulation, self.gun_experiment],
                'Stick': [self.stick_simulation, self.stick_experiment],
            },
            models={'eos': self.eos_model},
            opt_key='eos',
            constrain=True,
            outer_reltol=1E-6,
            precondition=True,
            debug=False,
            verb=True,
            sens_mode='ser',
            maxiter=6)

        # 5. Generage data from the simulations using the prior
        self.gun_prior_sim = self.gun_simulation(analysis.models)
        self.stick_prior_sim = self.stick_simulation(analysis.models)


        # 6. Run the analysis
        to = time.time()
        self.opt_model, history, self.sens_matrix = analysis()
        print('time taken ', to - time.time() )


        # 7. Update the simulations and get new data
        self.g_time_s, (self.g_vel_s, self.g_pos_s), self.g_spline_s =\
            self.opt_model.simulations['Gun']['sim'](self.opt_model.models)
        self.g_time_e, (self.g_vel_e, self.g_pos_e), self.g_spline_e =\
            self.opt_model.simulations['Gun']['exp']()

        self.s_pos_s, (self.s_time_s), self.s_data_s =\
            self.opt_model.simulations['Stick']['sim'](self.opt_model.models)
        self.s_pos_e, (self.s_time_e), self.s_data_e =\
            self.opt_model.simulations['Stick']['exp']()

        
def main(argv=None):
    import argparse
    import os
    import os.path

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    help_dict = {
        'basis':'Ten basis functions',
        'eos_nom_true':'Nominal and true EOS',
        'stick_CJ':'CJ construction with fit, prior and true EOSs',
        'stick_fisher':'Spectral analysis of Fisher information',
        'stick_xt':'t vs x plot',
        'gun_tv':'v vs t plot',
        'gun_fisher':'Spectral analysis of Fisher information'}
    parser = argparse.ArgumentParser(description='Make plots for poster.pdf')
    parser.add_argument('--show', action='store_true')
    # Plot requests
    for key,value in help_dict.items():
        parser.add_argument('--{0}'.format(key), type=str, help=value)
    args = parser.parse_args(argv)
    
    params = {'axes.labelsize': 18,     # Plotting parameters for latex
              'font.size': 15,
              'legend.fontsize': 15,
              'text.usetex': True,
              'font.family':'serif',
              'font.serif':'Computer Modern Roman',
              'xtick.labelsize': 15,
              'ytick.labelsize': 15}
    mpl.rcParams.update(params)
    if not args.show:
        mpl.use('PDF')  # Enables running without DISPLAY enviroment variable
    import matplotlib.pyplot as plt  # must be after mpl.use

    sim = Sim()
    # Make requested plots
    do_show = args.show
    for key in args.__dict__:
        if key not in plot_dict:
            continue
        if args.__dict__[key] == None:
            continue
        print('work on %s'%(key,))
        fig = plot_dict[key](plt, sim) # This calls a plotting function
        dir_name = getattr(args, key)
        if args.show or dir_name == 'show':
            do_show = True
        else:
            fig.savefig(os.path.join(dir_name,key)+'.pdf', format='pdf')
    if do_show:
        plt.show()
    return 0

if __name__ == "__main__":
    if sys.argv[1] == 'test':
        sys.exit(0)
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
