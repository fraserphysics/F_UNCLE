import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

import numpy as np
def plot_sens_matrix(sens_matrix, exp,  model=None, axes=None, fig=None,
                     labels=[], linestyles=[]):
    """Prints the sensitivity matrix

    Args:
        model(PhysicsModel): The model the sensitivity is in respect to
        exp(Experiment): The experiment the matrix is compared to
        sens_matrix(np.ndarray): The sensitivity matrix
    
    Keyword Args:
        axes(plt.Axes): The axes object *Ignored*
        fig(plt.Figure): A valid matplotlib figure on which to plot.
                         If `None`, creates a new figure
        sens_matrix(dict): A dict of the total sensitivity
        labels(list): Strings for labels *Ignored*
        linestyles(list): Strings for linestyles *Ignored*

    Return:
        (plt.Figure): The figure
    """
    if fig is None:
        fig = plt.figure()
    else:
        fig = fig
    # end

    gs = gridspec.GridSpec(3, 4,
                           width_ratios=[6, 1, 6, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[4])
    ax4 = fig.add_subplot(gs[6])
    ax5 = fig.add_subplot(gs[8])

    if model is not None:
        knot_post = model.get_t()
    else:
        knot_post = np.arange(sens_matrix.shape[1])
    # end

    resp_val = exp[0]

    style = ['-r', '-g', '-b', ':r', ':g', ':b',
             '--r', '--g', '--b', '--k']
    for i in range(10):
        ax1.plot(sens_matrix[:, i],
                 style[i], label="{:4.3f}".format(knot_post[i]))
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax1.get_legend().set_title("knots",
    #   prop = {'size':rcParams['legend.fontsize']})
    for i in range(10, 20):
        ax2.plot(sens_matrix[:, i],
                 style[i - 10], label="{:4.3f}".format(knot_post[i]))
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax2.get_legend().set_title("knots",
    #   prop = {'size':rcParams['legend.fontsize']})
    for i in range(20, 30):
        ax3.plot(sens_matrix[:, i],
                 style[i - 20], label="{:4.3f}".format(knot_post[i]))
    ax3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax3.get_legend().set_title("knots",
    #   prop = {'size':rcParams['legend.fontsize']})

    for i in range(30, 40):
        ax4.plot(sens_matrix[:, i],
                 style[i - 30], label="{:4.3f}".format(knot_post[i]))
    ax4.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax4.get_legend().set_title("knots",
    #   prop = {'size':rcParams['legend.fontsize']})

    for i in range(40, 47):
        ax5.plot(sens_matrix[:, i],
                 style[i - 40], label="{:4.3f}".format(knot_post[i]))
    ax5.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax5.get_legend().set_title("knots",
    #   prop = {'size':rcParams['legend.fontsize']})

    ax1.set_ylabel('Sensitivity')
    ax3.set_ylabel('Sensitivity')
    ax5.set_ylabel('Sensitivity')
    ax5.set_xlabel('Model resp. indep. var.')
    ax4.set_xlabel('Model resp. indep. var.')

    # xlocator = (max(resp_val) - min(resp_val)) / 4
    # ax1.xaxis.set_major_locator(MultipleLocator(xlocator))
    # ax2.xaxis.set_major_locator(MultipleLocator(xlocator))
    # ax3.xaxis.set_major_locator(MultipleLocator(xlocator))
    # ax4.xaxis.set_major_locator(MultipleLocator(xlocator))
    # ax5.xaxis.set_major_locator(MultipleLocator(xlocator))

    fig.tight_layout()
    plt.show()
    return fig
