import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plot_utils as pu
import net_sim_code as scnf
from scipy import ndimage

# parameters for the simulations
N = 50  # number of neurons
leak_s = 200.  # synaptic timescale  # pw = 5
D_scale = 0.1  # scaling of decoder


def plot_noisy_inhib_pop_dynamics(ax1=None, ax2=None, ax3=None, seed=None):
    """
    Plots for panels (a,b,c): noisy inhibitory population with slow, finite synapses
    :param ax1: Figure axis for the latent variable readout (trial 1)
    :param ax2: Figure axis for the spike raster (trial 1)
    :param ax3: Figure axis for two neurons' voltages (trial 1)
    :param seed: Random seed.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5
    x_lims = [-1.0, 1.0]  # [-0.9, 0.9]

    # get neuron parameters
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, x_lims)

    # decoders: make larger & variable for illustration purposes
    D *= -1.

    # deterministic simulation
    mu = 0.01
    sigma_v = 0.02
    D = D_scale * D[None, :]
    E = E[:, None]
    F = F[:, None]
    dt = 1e-4
    Tend = 0.3
    times = np.arange(0, Tend, dt)
    nT = len(times)
    init_pd = int(0.05 / dt)
    xrange = [-1, 1]
    x = np.concatenate((xrange[0] * np.ones((int((1. / 6) * nT),)),
                        np.linspace(xrange[0], xrange[1], int((5. / 6) * nT))))[None, :]

    s, V, g = scnf.run_exp_spiking_net(x, D, E, F, T, mu=mu, dt=dt, sigma_v=sigma_v, leak_s=leak_s)
    r = scnf.exp_filter(g, times, incl_dt=True)

    s = s[:, init_pd:]
    r = r[:, init_pd:]
    V = V[:, init_pd:]
    x = x[:, init_pd:]
    times = times[:-init_pd]
    y = D @ r
    spk_times = scnf.get_spike_times(s, dt)

    # simulation plotting
    _ = ax1.plot(times, x[0, :], '-', c='black', alpha=0.25, linewidth=1)
    _ = ax1.plot(times, scnf.y_quad_fcn(a, b, c)(x[0, :]), '--', c='black', alpha=0.75,
                 linewidth=1, label=r"$y$ bndry.")
    _ = ax1.plot(times, y[0, :], c='black', linewidth=0.5, alpha=0.5, label=r"$y$ sim.")
    # ax1.legend(fontsize=pu.fs1, ncols=2, frameon=False, handlelength=1.5, columnspacing=0.75,
    #            loc="lower left", bbox_to_anchor=(0.15, -0.1))
    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    for i in range(N):
        ax2.plot(spk_times[i], i * np.ones_like(spk_times[i]), '.', color=cmap[i], markersize=2)

    # plot voltages
    for i in [20, 35]:
        ax2.plot([0, 0.25], [i, i], alpha=0.25, linewidth=1, color=cmap[i])
        V[i, s[i, :].astype(bool)] = 2.
        ax3.plot(times, V[i, :], color=cmap[i], linewidth=0.75)  # ,alpha=0.5)
    ax3.axhline(y=0, c='gray', alpha=0.5, linewidth=0.5)
    ax3.axhline(y=1, c='black', alpha=0.5, linewidth=0.5, linestyle='--')

    # formatting
    for ax in [ax1, ax2, ax3]:
        if ax is not None:
            ax.set_xlim([0.0, 0.25])
            ax.set_xticks(np.linspace(0.0, 0.25, 6))
            ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,0.25,6)],fontsize=pfcn.fs1)
            ax.tick_params(axis='both', width=0.5, length=3, pad=1)
            sns.despine(ax=ax)
    ytcks1 = [-2, -1, 0, 1]
    ytcks2 = [0, 25, 50]
    ax1.set_yticks(ytcks1)
    ax2.set_yticks(ytcks2)
    ax1.set_yticklabels([str(s) for s in ytcks1], fontsize=pu.fs1)
    ax2.set_yticklabels([str(s) for s in ytcks2], fontsize=pu.fs1)
    ax1.set_ylim([-2, 1])
    ax2.set_ylim([-1, N])
    ax1.set_ylabel('Latent var.', fontsize=pu.fs2)
    ax2.set_ylabel('Neuron ID', fontsize=pu.fs2)
    ax3.set_ylabel('Voltages', fontsize=pu.fs2, labelpad=4.25)
    ax3.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, 0.25, 6)], fontsize=pu.fs1)
    ax3.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    ax1.text(0.2, 0.8, 'input, $x$', fontsize=pu.fs1, color='gray', alpha=0.5, rotation=11)
    ax1.text(0.17, -0.35, '$y$, boundary', fontsize=pu.fs1, color='black', alpha=0.75)
    ax1.text(0.047, -1.75, '$y$, simulation', fontsize=pu.fs1, color='black', alpha=0.5)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.7],
                facecolor=[0, 0, 0, 0.7])
    ax1.annotate("", xy=(0.215, -0.9), xytext=(0.225, -0.5), arrowprops=prop, color='black', alpha=0.75)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.4],
                facecolor=[0, 0, 0, 0.4])
    ax1.annotate("", xy=(0.031, -1.3), xytext=(0.041, -1.6), arrowprops=prop, color='black', alpha=0.5)

    ax2.text(0.02, 22, 'neuron 1', fontsize=pu.fs1, color=cmap[20])
    ax2.text(0.09, 37, 'neuron 2', fontsize=pu.fs1, color=cmap[35])
    ax3.text(0.01, 0.2, 'neuron 1', fontsize=pu.fs1, color=cmap[20])
    ax3.text(0.1, 0.3, 'neuron 2', fontsize=pu.fs1, color=cmap[35])

    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['0', '1'], fontsize=pu.fs1)
    ax3.set_ylim([0, 1.75])


def plot_rate_inhib_pop_dynamics(ax1=None, ax2=None):
    """
    Plots for panels (d,e): (sigmoidal) rate version of the inhibitory population
    :param ax1: Figure axis for the latent variable readout (trial 1)
    :param ax2: Figure axis for the rates over time.
    :param seed: Random seed.
    :return:
    """

    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5
    x_lims = [-1.0, 1.0]

    # get neuron parameters
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, x_lims)

    # decoders: make larger & variable for illustration purposes
    D *= -5.

    # deterministic simulation
    D = D_scale * D[None, :]
    E = E[:, None]
    F = F[:, None]
    dt = 1e-4
    Tend = 1.05
    init_pd = int(0.05 / dt)
    times = np.arange(0, Tend, dt)
    nT = len(times)
    xrange = [-1, 1]
    x = np.concatenate((xrange[0] * np.ones((int((1. / 20) * nT),)),
                        np.linspace(xrange[0], xrange[1], int((19. / 20) * nT))))[None, :]

    V, r = scnf.run_rate_net(x, D, E, F, T, dt=dt, beta=50)
    y = D @ r
    r = r[:, init_pd:]
    y = y[:, init_pd:]
    x = x[:, init_pd:]
    times = times[:-init_pd]

    # simulation plotting
    if ax1 is not None:
        _ = ax1.plot(times, x[0, :], '-', c='black', alpha=0.25, linewidth=1)
        _ = ax1.plot(times, scnf.y_quad_fcn(a, b, c)(x[0, :]), '--', c='black', alpha=0.75,
                     linewidth=1, label=r"$\bar{y}$ bndry.")
        _ = ax1.plot(times, y[0, :], c='black', linewidth=0.5, alpha=0.5, label=r"$\bar{y}$ sim.")
        # ax1.legend(fontsize=pu.fs1, ncols=2, frameon=False, handlelength=1.5, columnspacing=0.75,
        #            loc="lower left", bbox_to_anchor=(0.15, -0.1))

    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    avals = 0.25 * np.ones((N,))
    avals[20] = 1.0
    avals[35] = 1.0
    for i in range(N):
        _ = ax2.plot(times, r[i, :], linewidth=0.5, alpha=avals[i], color=cmap[i])

    # formatting
    for ax in [ax1, ax2]:
        ax.set_xlim([0.0, 1.0])
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,0.25,6)],fontsize=pfcn.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        sns.despine(ax=ax)

    if ax1 is not None:
        ytcks1 = [-2, -1, 0, 1]
        ax1.set_yticks(ytcks1)
        ax1.set_yticklabels([str(s) for s in ytcks1], fontsize=pu.fs1)
        ax1.set_ylim([-2, 1])
        ax1.set_ylabel('Latent var.', fontsize=pu.fs2)
        # ax1.text(0.05, -0.45, '$x$', fontsize=pu.fs1, color='gray')
        # ax1.text(0.75, -1.5, '$y$', fontsize=pu.fs1, color='black')
        ax1.text(3.9*0.2, 0.8, 'input, $x$', fontsize=pu.fs1, color='gray', alpha=0.5, rotation=11)
        ax1.text(4*0.17, -0.35, r'$\bar{y}$, boundary', fontsize=pu.fs1, color='black', alpha=0.75)
        ax1.text(3.6*0.047, -1.75, r'$\bar{y}$, simulation', fontsize=pu.fs1, color='black', alpha=0.5)
        prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.7],
                    facecolor=[0, 0, 0, 0.7])
        ax1.annotate("", xy=(4*0.215, -0.9), xytext=(4*0.225, -0.5), arrowprops=prop, color='black', alpha=0.75)
        prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.4],
                    facecolor=[0, 0, 0, 0.4])
        ax1.annotate("", xy=(3.6*0.031, -1.3), xytext=(3.6*0.041, -1.6), arrowprops=prop, color='black', alpha=0.5)

    if ax2 is not None:
        ax2.set_ylabel(r'Rates, $r_F$', fontsize=pu.fs2, labelpad=4.25)
        ax2.set_yticks([0, 0.25, 0.5])
        ax2.set_yticklabels(['0', '0.25', '0.5'], fontsize=pu.fs1)
        ax2.set_ylim([0, 0.5])


def plot_trialavg_spikes(ax, ntrials=100):
    """
    Plot for panel (f): comparison of rates and trial-averaged spiking
    :param ax: Figure axis.
    :param ntrials: Number of spiking trials with which to compute the average.
    :return:
    """

    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5
    x_lims = [-1.0, 1.0]  # [-0.9, 0.9]

    # get neuron parameters
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, x_lims)

    # decoders: make larger & variable for illustration purposes
    D *= -1.

    # deterministic simulation
    mu = 0.01
    sigma_v = 0.02
    D = D_scale * D[None, :]
    E = E[:, None]
    F = F[:, None]
    dt = 1e-4
    Tend = 0.3
    times = np.arange(0, Tend, dt)
    nT = len(times)
    xrange = [-1, 1]
    x = np.concatenate((xrange[0] * np.ones((int((1. / 6) * nT),)),
                        np.linspace(xrange[0], xrange[1], int((5. / 6) * nT))))[None, :]

    s_tot = np.zeros((N, nT))
    for i in range(ntrials):
        s, V, g = scnf.run_exp_spiking_net(x, D, E, F, T, mu=mu, dt=dt, sigma_v=sigma_v, leak_s=leak_s)
        s_tot += s

    init_pd = int(0.05 / dt)
    s_tot = s_tot[:, init_pd:]
    times = times[:-init_pd]

    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    avals = 0.25 * np.ones((N,))
    avals[20] = 1.0
    avals[35] = 1.0
    for i in range(N):
        _ = ax.plot(times, ndimage.gaussian_filter1d(s_tot[i, :], 20, axis=0) / 5, linewidth=0.5, alpha=avals[i],
                    color=cmap[i])

    ax.set_xlim([0.0, 0.25])
    ax.set_xticks(np.linspace(0.0, 0.25, 6))
    ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,0.25,6)],fontsize=pfcn.fs1)
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine(ax=ax)

    ytcks2 = [0.0, 0.25, 0.5]
    ax.set_yticks(ytcks2)
    ax.set_yticklabels([str(s) for s in ytcks2], fontsize=pu.fs1)
    ax.set_ylim([0, 0.55])
    ax.set_ylabel('Avg. spikes', fontsize=pu.fs2)
    ax.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, 0.25, 6)], fontsize=pu.fs1)
    ax.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)


def main(save_pdf=False, show_plot=True):

    f = plt.figure(figsize=(4.5, 2.25), dpi=150)
    gs = gridspec.GridSpec(3, 8)
    ax1 = f.add_subplot(gs[0, 1:4])
    ax2 = f.add_subplot(gs[0, 4:7])
    ax3 = f.add_subplot(gs[1, 1:4])
    ax4 = f.add_subplot(gs[1, 4:7])
    ax5 = f.add_subplot(gs[2, 1:4])
    ax6 = f.add_subplot(gs[2, 4:7])

    plot_noisy_inhib_pop_dynamics(ax1=ax1, ax2=ax3, ax3=ax5, seed=61)  # seed=59)
    plot_rate_inhib_pop_dynamics(ax1=ax2, ax2=ax4)
    plot_trialavg_spikes(ax=ax6)

    f.tight_layout()

    ax1.text(-0.3, 1.1, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax3.text(-0.3, 1.1, r'\textbf{b}', transform=ax3.transAxes, **pu.panel_prms)
    ax5.text(-0.3, 1.1, r'\textbf{c}', transform=ax5.transAxes, **pu.panel_prms)
    ax2.text(-0.3, 1.1, r'\textbf{d}', transform=ax2.transAxes, **pu.panel_prms)
    ax4.text(-0.3, 1.1, r'\textbf{e}', transform=ax4.transAxes, **pu.panel_prms)
    ax6.text(-0.3, 1.1, r'\textbf{f}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_10_slow_inhib_net.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
