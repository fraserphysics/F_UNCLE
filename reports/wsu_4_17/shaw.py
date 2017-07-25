"""shaw.py:
Copied from ~/projects/eos with minor edits
Plots of isentrope from Hixson et al. J Appl. Phys. 88, 6287 (2000)
and simple power law
"""

import sys
import matplotlib as mpl
import numpy as np

# Parameters from Hixson et al.
c = np.array(( # eqn 4.9 pg 6292
    2.4287,    # c_0
    0.0378,    # c_1
    0.1005,    # c_2
    -0.7166,   # c_3
    0.4537,    # c_4
    0.8062,    # c_5
    -0.8473    # c_6
    ))
rho_CJ = 2.4403 # gm/cm^3After eqn 4.9 pg 6292
P_CJ = 35.3     # GPa after eqn 4.9 pg 6292
gamma_0 = 3     # Top of second column page 6290

def f_x(x_): # eqn 4.9 pg 6292
    rv = np.empty(x_.shape)
    for i in range(len(x_)):
        x = x_[i]
        X = np.array([
            1.0,
            x,
            x*x,
            x**3,
            x**4,
            x**5,
            x**6])*1e9
        rv[i] = np.dot(X,c)
    return rv

def P(v # Specific volume, ie volume per gram
      ):
    rho = 1/v
    return rho**gamma_0 * f_x(rho - rho_CJ)

def f_nom(v):
    C = 2.56e10
    return C/v**3

size1 = 24
size2 = 28
params = {'axes.labelsize': size2,     # Plotting parameters for latex
          'text.fontsize': size1,
          'legend.fontsize': size1,
          'text.usetex': True,
          'font.family':'serif',
          'font.serif':'Computer Modern Roman',
          'xtick.labelsize': size1,
          'ytick.labelsize': size1}
mpl.rcParams.update(params)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) not in (0, 2):
        raise RuntimeError,'argv=%s'%(argv,)
    if len(argv) == 0:
        DEBUG = True
    else:
        DEBUG = False
    if DEBUG:
        mpl.rcParams['text.usetex'] = False
    else:
        mpl.use('PDF')
    import matplotlib.pyplot as plt  # must be after mpl.use

    def plot_func_v(v, func):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(v,func(v),'b')
        return fig

    figs = []
    
    v = np.arange(.31, .52, .01)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    nom = P(v)/1e10
    ax.plot(v,nom,'r')
    ax.plot(v,nom*1.05,'b')
    ax.plot(v,nom*0.95,'b')
    ticks = np.array([.3, .4, .5])
    ax.set_xticks(ticks)
    ax.set_xticklabels(['%3.1f'%t for t in ticks])
    yticks = np.array([2, 4, 6, 8])
    ax.set_yticks(yticks)
    #ax.set_xticklabels(['%3.1f'%t for t in ticks])
    ax.set_xlim(.3, 0.53)
    ax.set_xlabel(r'$v/({\rm{cm}}^3 \! /{\rm{gm}}$)')
    ax.set_ylabel(r'$p/(10^{10}$Pa)')
    fig.subplots_adjust(left=0.15, bottom=0.15) # Make more space for label
    if not DEBUG:
        figs.append((argv[0],fig))
    
    v = np.arange(.4, 4.0, .01)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    nom = f_nom(v)
    x = np.log(v)
    ticks = np.array([.4, 1, 2, 4])
    ax.semilogy(x,nom,'r')
    ax.semilogy(x,nom*1.25,'b')
    ax.semilogy(x,nom*0.8,'b')
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(['%3.1f'%t for t in ticks])
    ax.set_xlabel(r'$v$ (log scale)')
    ax.set_ylabel(r'$p$ (log scale)')
    fig.subplots_adjust(left=0.2, bottom=0.15) # Make more space for label
    if not DEBUG:
        figs.append((argv[1],fig))

    if DEBUG:
        plt.show()
    else:
        for name, fig in figs:
            fig.savefig(name, format='pdf')
        pass
if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# eval: (python-mode)
# End:
