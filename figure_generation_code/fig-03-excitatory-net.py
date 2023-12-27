import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf


def plot_single_excitatory_boundary(ax):
    """
    Plots panel (a), the input-output voltage relationship for a single excitatory neuron.
    :param ax: Figure axis.
    :return:
    """

    xlims = [-1, 1]
    ylims = [0, 2.5]
    E = 2
    F = 3.5
    T = 1.5
    D = 1

    # boundary
    x = np.linspace(xlims[0], xlims[1], 21)
    y = scnf.single_nrn_boundary(x, E, F, T)

    ax.plot(x, y, linewidth=1, c=pu.excitatory_red)
    ax.fill_between(x, y, 2.5 * np.ones_like(y), color=pu.excitatory_red, alpha=0.1)
    ax.quiver([-0.1], [0.95], [0.005 * F], [0.005 * E], color=pu.excitatory_red,
              width=0.01, headlength=5, headwidth=5, scale=0.05, scale_units='y')
    ax.text(0.2, 1.0, r'$\begin{pmatrix} F_1 \\ E_1 \end{pmatrix}$', color=pu.excitatory_red, fontsize=pu.fs1)

    # leak quivers
    xlim = np.arange(-0.85, 0.95, 0.2)
    ylim = np.arange(0.1, 2.4, 0.37)
    X, Y = np.meshgrid(xlim, ylim)
    u = 0
    v = -Y-0.2
    v[F * X + E * Y - T > -0.4] = np.nan
    ax.quiver(X, Y, u, v, color='gray', alpha=0.5, width=0.005, headlength=5, headwidth=5)

    # decoder
    ax.quiver([-0.4], [1.45], [0], [0.025 * D], color=pu.excitatory_red, width=0.01, headlength=5, headwidth=5,
              scale=0.05, scale_units='y')
    ax.text(-0.3, 1.65, r'$\begin{drcases} & \\ & \end{drcases}D_1$', color=pu.excitatory_red, fontsize=pu.fs1)

    # sub/supra text
    ax.text(-0.1, 2.25, r'\textbf{suprathreshold}', color=pu.excitatory_red, fontsize=pu.fs1)
    ax.text(-0.9, 0.075, r'\textbf{subthreshold}', fontsize=pu.fs1)

    # formatting
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['0', '1', '2'], fontsize=pu.fs1)
    ax.set_xlabel('Input signal, $x$', fontsize=pu.fs2)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2, labelpad=7)
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine(ax=ax)


def plot_excitatory_boundary(ax1=None, ax2=None):
    """
    Plots panels (b) and (c), the input-output boundaries for a network of 10 excitatory neurons.
    :param ax1: Figure axis for the version with individual neuron thresholds
    :param ax2: Figure axis for the version with a single boundary, plus decoder arrows
    :return:
    """

    # boundary parameters
    a = -1.0
    b = 0
    c = 1.5

    # get neuron parameters
    N = 10
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, [-0.9, 0.9])
    D *= 0.25  # set decoder to be larger so that it is easier to see

    # plotting boundaries
    xlim = np.arange(-0.875, 0.85, 0.195)
    ylim = np.arange(0.3, 2, 0.3)
    X, Y = np.meshgrid(xlim, ylim)
    u = 0
    v = -Y

    x_range = [-1, 1]
    x_ideal = np.linspace(x_range[0], x_range[1], 101)
    y_ideal = scnf.y_quad_fcn(a, b, c)(x_ideal)
    if ax1 is not None:
        ax1.plot(x_ideal, y_ideal, c='gray', alpha=0.5, linestyle=':', linewidth=2)

    cmap = pu.excitatory_cmap(np.linspace(0, 0.9, N))
    yb = np.zeros((N, 51))
    x = np.linspace(x_range[0], x_range[1], 51)
    for i in range(N):
        y = scnf.single_nrn_boundary(x, E[i], F[i], T[i])
        yb[i, :] = y
        ax1.plot(x, y, linewidth=1, c=cmap[i], alpha=0.5)
        v[F[i] * X + E[i] * Y - T[i] > -0.15] = np.nan
        ax2.quiver(xvals[i], yvals[i],
                   np.array([0]), np.array([0.05 * D[i]]), color=pu.excitatory_red,
                   width=0.0075, headlength=5, headwidth=5, scale=0.05, scale_units='y')

    ax2.plot(x, np.min(yb, 0), c=pu.excitatory_red, linewidth=1)
    ax2.fill_between(x, np.min(yb, 0), 15 * np.ones_like(x), color=pu.excitatory_red, alpha=0.1)
    ax2.quiver(X, Y, u, v, color='gray', alpha=0.5,
               width=0.005, headlength=5, headwidth=5)
    ax1.text(-0.85, 0.5, '$y=-f_{cvx}(x)$', fontsize=pu.fs1)
    ax2.text(-0.95, 1.825, r'\textbf{suprathreshold}', color=pu.excitatory_red, fontsize=pu.fs1)
    ax2.text(0.1, 0.075, r'\textbf{subthreshold}', fontsize=pu.fs1)

    # formatting
    ytcks = [0, 1, 2]
    for ax in [ax1, ax2]:
        ax.set_xlim(x_range)
        ax.set_ylim([0, 2])
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
        ax.set_yticks(ytcks)
        ax.set_yticklabels([str(s) for s in ytcks], fontsize=pu.fs1)
        ax.set_xlabel('Input signal, $x$', fontsize=pu.fs2)
        ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        sns.despine(ax=ax)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2, labelpad=7)


def main(save_pdf=False, show_plot=True):

    f = plt.figure(figsize=(4.5, 1.55), dpi=150)
    gs = f.add_gridspec(1, 3)
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[0, 2])

    # call plotting functions
    plot_single_excitatory_boundary(ax1)
    plot_excitatory_boundary(ax1=ax2, ax2=ax3)
    f.tight_layout()

    # add panel labels
    ax1.text(-0.35, 1.075, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.35, 1.075, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.35, 1.075, r'\textbf{c}', transform=ax3.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_03_excitation.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
