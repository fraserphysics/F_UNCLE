"""Script for re-generating the notes figures

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# Standard python packages
import sys
import os
import time
# External python packages
import numpy as np
import matplotlib.pyplot as plt

# F_UNLCE packages
sys.path.append(os.path.abspath('./..'))
from F_UNCLE.Experiments.GunModel import Gun, GunExperiment
from F_UNCLE.Experiments.Stick import Stick, StickExperiment
from F_UNCLE.Models.Isentrope import DensityEOS, EOSBump, EOSModel
from F_UNCLE.Opt.Bayesian import Bayesian

sys.path.append(os.path.abspath('./../../fit_9501'))
from fit_9501.Models.equation_of_state import AugmentedIsentrope

if __name__ == '__main__':
    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 7
    rcParams['legend.handlelength'] = 3.0
    rcParams['backend'] = 'Agg'

    pagewidth = 360  # pts
    au_ratio = (np.sqrt(5) - 1.0) / 2.0
    figwidth = 1.0  # fraction of \pagewidth for figure
    figwidth *= pagewidth / 72.27
    figtype = '.pdf'
    out_dir = './'
    square = (figwidth, figwidth)
    tall = (figwidth, 1.25 * figwidth)
    vrange=(0.1,4.0)
    

    #################
    #    Get Data   #
    #################
    vol_min = 0.1
    vol_max = 1.0
    
    density_linear = DensityEOS(
        lambda r: 2.56E9 * r**3,
        name='density_linear',
        spline_min=vol_max**-1,
        spline_max=vol_min**-1,
        spline_sigma=0.05,
        basis='dens',
        spacing='lin'
    )

    augmented_lin = AugmentedIsentrope(
        lambda r: 2.56E9 * r**3,
        name='augmented_linear',
        spline_min=vol_max**-1,
        spline_max=vol_min**-1,
        spline_sigma=0.05,
        basis='dens',
        spacing='lin'
    )
    
    density_log = DensityEOS(
        lambda r: 2.56E9 * r**3,
        name='density_log',        
        spline_min=vol_max**-1,
        spline_max=vol_min**-1,
        spline_sigma=0.05,
        basis='dens',
        spacing='log'
    )
    
    volume_linear = EOSModel(
        lambda r: 2.56E9 * r**-3,
        name='volume_linear',
        spline_min=vol_min,
        spline_max=vol_max,
        rho_0=1.844,
        pres_0=101325,
        spline_sigma=0.05,
        basis='vol',
        spacing='lin'
    )

    volume_log = EOSModel(
        lambda r: 2.56E9 * r**-3,
        name='volume_log',
        spline_min=vol_min,
        spline_max=vol_max,
        rho_0=1.844,
        pres_0=101325,
        spline_sigma=0.05,
        basis='vol',
        spacing='log'
    )
    
    eos_true = EOSBump()

    # 3. Create the objects to generate simulations and pseudo experimental data
    gun_simulation = Gun(mass_he=1.0, sigma=1.0)
    gun_experiment = GunExperiment(model=eos_true, mass_he=1.0)
    stick_simulation = Stick()
    stick_experiment = StickExperiment(
        model=eos_true,
        sigma_t=1E-9,
        sigma_x=2E-3
    )
    
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.set_xlabel('Specific Volume')
    ax2.set_ylabel('Pressure')
    volume_list = np.linspace(vol_min, vol_max, 200)
    ax2.plot(volume_list, 2.56E9 * volume_list**-3, '-', label='Prior')
    
    for eos_model in [density_linear, density_log, volume_log, volume_linear, augmented_lin]:
        print('Working on ' + eos_model.name)
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
            verb=False,
            sens_mode='ser',
            maxiter=6)

        to = time.time()
        opt_model, history, sens_matrix = analysis()
        print('time taken ', to - time.time() )

        if eos_model.get_option('basis') == 'dens':
            ax2.plot(volume_list, opt_model.models['eos'](volume_list**-1),
                     '-' ,label=eos_model.name)
        else:
            ax2.plot(volume_list, opt_model.models['eos'](volume_list),
                     '-' ,label=eos_model.name)            
        # end
        
        for key in ['Gun', 'Stick']:
            fisher = opt_model.simulations['Gun']['exp'].\
                get_fisher_matrix(opt_model.models,
                                  sens_matrix=sens_matrix['Gun'])
            spec_data = opt_model.fisher_decomposition(fisher)

            fig = plt.figure(figsize=tall)
            fig = opt_model.plot_fisher_data(spec_data, fig=fig)
            fig.set_size_inches(tall)
            fig.tight_layout()
            fig.savefig('fisher_' + key + '_' + eos_model.name + '.pdf')
        # end
    # end
    ax2.set_xlim((0.1,0.9))
    ax2.set_ylim((0,6E11))    
    ax2.legend(loc='best')
    fig2.savefig('EOSComparisson.pdf')

