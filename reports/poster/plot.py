"""plot.py makes plots for notes.pdf.

"""
plot_dict = {} # Keys are those keys in args.__dict__ that ask for
               # plots.  Values are the functions that make those
               # plots.
import sys
import matplotlib as mpl
import numpy as np

def stick_xt(plt, sim):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    s_pos_s, (s_time_s), s_data_s = sim.stick_simulation()
    s_pos_e, (s_time_e), s_data_e = sim.stick_experiment()
    sim.stick_simulation.plot(axis=ax, data_style='-k', level=2,
                          data=(s_pos_s, (s_time_s), s_data_s))


    sim.stick_simulation.plot(axis=ax, data_style='+g', level=2,
                          data=(s_pos_e, (s_time_e), s_data_e))

    sim.stick_simulation.update(sim.eos_model)
    sim.stick_simulation.plot(axis=ax, data_style='--b', level=2,
                          data=sim.stick_simulation())
    sim.stick_simulation.update(sim.best_eos)

    ax.legend(['Fit EOS', 'True EOS', 'Prior EOS'], loc='best')
    return fig
plot_dict['stick_xt'] = stick_xt

def stick_fisher(plt, sim):
    fisher = sim.analysis.get_fisher_matrix(simid=1, sens_calc=True)
    spec_data = sim.analysis.fisher_decomposition(fisher)
    s_pos_s, (s_time_s), s_data_s = sim.stick_simulation()

    fig = sim.analysis.plot_fisher_data(spec_data)
    fig.axes[1].axvline(s_data_s[1])
    fig.axes[1].annotate(r'$v_{CJ}$',
                          xy=(s_data_s[1], 0),
                          xytext=(30,30),
                          xycoords='data',
                          textcoords='offset points',
                          arrowprops=dict(facecolor='black',
                                          arrowstyle='->'))
    return fig
plot_dict['stick_fisher'] = stick_fisher

def stick_CJ(plt, sim):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sim.stick_simulation.plot(axis=ax,
                          eos_style='-k',
                          ray_style=':k',
                          cj_style='xk')
    sim.eos_model.prior.plot(axis=ax, style='--b')
    sim.eos_true.plot(axis=ax, style='-.g')
    ax.legend(['Fit EOS',
                  'Rayleigh line',
                  'CJ point',
                  r'($v_o$, $p_o$)',
                  'Prior EOS',
                  'True EOS'],
                 loc='best')
    return fig
plot_dict['stick_CJ'] = stick_CJ

def eos_nom_true(plt, sim):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sim.eos_model.prior.plot(axis=ax, style='--b')
    sim.eos_true.plot(axis=ax, style='-.g')
    ax.legend(['Prior EOS',
                  'True EOS'])
    fig.tight_layout()
    return fig
plot_dict['eos_nom_true'] = eos_nom_true

def gun_tv(plt, sim):
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    sim.best_eos.plot(axis=ax, style='-k')
    sim.eos_model.prior.plot(axis=ax, style='--b')
    sim.eos_true.plot(axis=ax, style='-.g')
    ax.legend(['Fit EOS',
                  'Prior EOS',
                  'True EOS'], loc='best')

    ax = fig.add_subplot(2,1,2)
    g_time_s, (g_vel_s, g_pos_s), g_spline_s = sim.gun_simulation()
    g_time_e, (g_vel_e, g_pos_e), g_spline_e = sim.gun_experiment()
    sim.gun_simulation.plot(axis=ax, style='-k', err_style='-r',
                        data=[(g_time_s, (g_vel_s, g_pos_s), g_spline_s),
                              (g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    sim.gun_simulation.plot(axis=ax, style='-.g',
                        data=[(g_time_e, (g_vel_e, g_pos_e), g_spline_e)])

    sim.gun_simulation.update(model=sim.eos_model)
    sim.gun_simulation.plot(axis=ax, style='--b', data=[sim.gun_simulation()])
    sim.gun_simulation.update(model=sim.best_eos)
    ax.plot([None],[None], '-r')
    ax.legend(['Fit EOS',
                  'Prior EOS',
                  'True EOS',
                  'Error'], loc='upper left', framealpha = 0.5)

    return fig
plot_dict['gun_tv'] = gun_tv

def gun_fisher(plt, sim):
    fisher = sim.analysis.get_fisher_matrix(simid=0, sens_calc=True)
    spec_data = sim.analysis.fisher_decomposition(fisher)

    fig = sim.analysis.plot_fisher_data(spec_data)
    fig.set_size_inches((6,7.5))
    return fig
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
        import Models.Isentrope
        import Experiments.GunModel
        import Experiments.Stick
        import Opt.Bayesian
        # 1. Generate a functional form for the prior
        init_prior = np.vectorize(lambda v: 2.56e9 / v**3)

        # 2. Create the model and *true* EOS
        eos_model = Models.Isentrope.EOSModel(init_prior, Spline_sigma=0.05)
        eos_true = Models.Isentrope.EOSBump()
        self.eos_model = eos_model
        self.eos_true = eos_true

        # 3. Create the objects to generate simulations and pseudo experimental data
        gun_experiment = Experiments.GunModel.Gun(eos_true, mass_he=1.0)
        gun_simulation = Experiments.GunModel.Gun(eos_model, mass_he=1.0, sigma=1.0)
        self.gun_simulation = gun_simulation
        self.gun_experiment = gun_experiment

        stick_experiment = Experiments.Stick.Stick(eos_true)
        stick_simulation = Experiments.Stick.Stick(
            eos_model,
            sigma_t=1E-9,
            sigma_x=2E-3)
        self.stick_simulation = stick_simulation
        self.stick_experiment = stick_experiment



        # 4. Generate data from the simulations using the prior
        gun_prior_sim = gun_simulation()
        stick_prior_sim = stick_simulation()
        self.gun_prior_sim = gun_prior_sim
        self.stick_prior_sim = stick_prior_sim

        # 5. Create the analysis object
        analysis = Opt.Bayesian.Bayesian(
            simulations=[
                (gun_simulation, gun_experiment),
                (stick_simulation, stick_experiment)],
            model=eos_model,
            constrain=True,
            outer_reltol=1E-6,
            precondition=True,
            debug=False,
            maxiter=10)

        # 6. Run the analysis
        best_eos, history = analysis()
        self.analysis = analysis
        self.best_eos = best_eos

def main(argv=None):
    import argparse
    import os
    import os.path

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    help_dict = {
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
