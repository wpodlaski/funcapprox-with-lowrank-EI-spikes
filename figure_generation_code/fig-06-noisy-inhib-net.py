import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plot_utils as pu
import net_sim_code as scnf


def plot_noisy_inhib_pop_boundary(ax1=None, ax2=None, seed=None):
    """
    Plots for panels (a) and (b) -- inhibitory boundary with random jitter applied to each neuron.
    Note that this is an illustration --- no direct relationship to the simulations in the rest of the figure.
    :param ax1: Figure axis for noise realization 1
    :param ax2: Figure axis for noise realization 2
    :param seed: Random seed
    :return:
    """

    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5
    N = 10
    x_lims = [-1.0, 1.0]

    # get neuron parameters
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, x_lims)

    x_range = [-1, 1]
    x_ideal = np.linspace(x_range[0], x_range[1], 101)
    y_ideal = scnf.y_quad_fcn(a, b, c)(x_ideal)
    ax1.plot(x_ideal, y_ideal, c='gray', alpha=0.5, linestyle=':', linewidth=1.5)
    ax2.plot(x_ideal, y_ideal, c='gray', alpha=0.5, linestyle=':', linewidth=1.5)

    if seed is not None:
        np.random.seed(seed)
    noise1 = 0.2 * np.random.normal(size=(N,))
    noise2 = 0.2 * np.random.normal(size=(N,))
    yb1 = np.zeros((N, 51))
    yb2 = np.zeros((N, 51))
    x = np.linspace(x_range[0], x_range[1], 51)
    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    for i in range(N):
        y = scnf.single_nrn_boundary(x, E[i], F[i], T[i])
        ax1.plot(x, y + noise1[i], linewidth=0.75, c=cmap[i], alpha=0.25)
        ax2.plot(x, y + noise2[i], linewidth=0.75, c=cmap[i], alpha=0.25)
        yb1[i, :] = y + noise1[i]
        yb2[i, :] = y + noise2[i]
    yb1max = np.min(yb1, 0)
    yb2max = np.min(yb2, 0)
    for i in range(N):
        ytmp = yb1[i, :].copy()
        ytmp[ytmp > yb1max] = np.nan
        ax1.plot(x, ytmp, linewidth=1.25, c=cmap[i], alpha=1.)
        ytmp = yb2[i, :].copy()
        ytmp[ytmp > yb2max] = np.nan
        ax2.plot(x, ytmp, linewidth=1.25, c=cmap[i], alpha=1.)

    # formatting
    ytcks = [-2, -1, 0]
    for ax in [ax1, ax2]:
        ax.set_xlim(x_range)
        ax.set_ylim([-2, 0])
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([])  # ,fontsize=pfcn.fs1)
        ax.set_yticks(ytcks)
        ax.set_yticklabels([str(s) for s in ytcks], fontsize=pu.fs1)
        ax.set_ylabel('Latent var., $y$', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.axhline(y=0, linewidth=1., color='black')
        sns.despine(ax=ax, bottom=True)
    ax2.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    ax2.set_xlabel('Input, $x$', fontsize=pu.fs2)


def plot_noisy_inhib_pop_dynamics(ax1=None, ax2=None, ax3=None, ax4=None, seed1=None, seed2=None):
    """
    Plots for panels (c), (d), (e) and (f): noisy inhibitory population, run for two trials
    :param ax1: Figure axis for the latent variable readout (trial 1)
    :param ax2: Figure axis for the spike raster (trial 1)
    :param ax3: Figure axis for two neurons' voltages (trial 1)
    :param ax4: Figure axis for two neurons' voltages (trial 2)
    :param seed1: Random seed for trial 1
    :param seed2: Random seed for trial 2
    :return:
    """

    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5
    N = 50
    x_lims = [-1.0, 1.0]  # [-0.9, 0.9]

    # get neuron parameters
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, x_lims)

    # decoders: make larger & variable for illustration purposes
    D *= -1.

    # noisy simulation
    mu = 0.02
    sigma_v = 0.02
    D_scale = 0.1  # 2.#0.25
    D = D_scale * D[None, :]
    E = E[:, None]
    F = F[:, None]
    dt = 1e-4
    Tend = 0.3
    times = np.arange(0, Tend, dt)
    init_pd = int(0.05 / dt)
    nT = len(times)
    xrange = [-1, 1]
    x = np.concatenate((xrange[0] * np.ones((int((1. / 6) * nT),)),
                        np.linspace(xrange[0], xrange[1], int((5. / 6) * nT))))[None, :]

    # trial 1
    (s, V, _) = scnf.run_spiking_net(x, D, E, F, T, mu=mu, dt=dt, sigma_v=sigma_v, seed=seed1)
    # trial 2
    (s2, V2, _) = scnf.run_spiking_net(x, D, E, F, T, mu=mu, dt=dt, sigma_v=sigma_v, seed=seed2)

    # get rates, readouts, etc
    r = scnf.exp_filter(s, times)
    s = s[:, init_pd:]
    r = r[:, init_pd:]
    V = V[:, init_pd:]
    s2 = s2[:, init_pd:]
    V2 = V2[:, init_pd:]
    x = x[:, init_pd:]
    times = times[:-init_pd]
    y = D @ r
    spk_times = scnf.get_spike_times(s, dt=dt)

    # simulation plotting
    _ = ax1.plot(times, x[0, :], '-', c='black', alpha=0.25, linewidth=1)
    _ = ax1.plot(times, scnf.y_quad_fcn(a, b, c)(x[0, :]), '--', c='black', alpha=0.75,
                 linewidth=1, label=r"$y$ bndry.")
    _ = ax1.plot(times, y[0, :], c='black', linewidth=0.5, alpha=0.5, label=r"$y$ sim.")
    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    for i in range(N):
        ax2.plot(spk_times[i], i * np.ones_like(spk_times[i]), '.', color=cmap[i], markersize=2)

    # plot voltages
    for i in [20, 35]:
        V[i, s[i, :].astype(bool)] = 2.
        ax3.plot(times, V[i, :], color=cmap[i], linewidth=0.5)  # ,alpha=0.5)
    for i in [20, 35]:
        V2[i, s2[i, :].astype(bool)] = 2.
        ax4.plot(times, V2[i, :], color=cmap[i], linewidth=0.5)  # ,alpha=0.5)
        ax2.plot([0, 0.25], [i, i], alpha=0.25, linewidth=1, color=cmap[i])
    for ax in [ax3, ax4]:
        ax.axhline(y=0, c='gray', alpha=0.5, linewidth=0.5)
        ax.axhline(y=1, c='black', alpha=0.5, linewidth=0.5, linestyle='--')
    ax2.text(0.02, 22, 'neuron 1', fontsize=pu.fs1, color=cmap[20])
    ax2.text(0.09, 37, 'neuron 2', fontsize=pu.fs1, color=cmap[35])
    ax3.text(0.01, 0.2, 'neuron 1', fontsize=pu.fs1, color=cmap[20])
    ax3.text(0.1, 0.3, 'neuron 2', fontsize=pu.fs1, color=cmap[35])

    # formatting
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([0.0, 0.25])
        ax.set_xticks(np.linspace(0.0, 0.25, 6))
        ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,0.25,6)],fontsize=pfcn.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        sns.despine(ax=ax)
    ax2.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, 0.25, 6)], fontsize=pu.fs1)
    ax4.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, 0.25, 6)], fontsize=pu.fs1)
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
    ax2.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    ax3.set_ylabel('Voltages\n (trial 1)', fontsize=pu.fs2, labelpad=4.25)
    ax4.set_ylabel('Voltages\n (trial 2)', fontsize=pu.fs2, labelpad=4.25)
    ax4.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    ax1.text(0.2, 0.8, 'input, $x$', fontsize=pu.fs1, color='gray', alpha=0.5, rotation=11)
    ax1.text(0.17, -0.35, '$y$, boundary', fontsize=pu.fs1, color='black', alpha=0.75)
    ax1.text(0.047, -1.75, '$y$, simulation', fontsize=pu.fs1, color='black', alpha=0.5)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.7],
                facecolor=[0, 0, 0, 0.7])
    ax1.annotate("", xy=(0.215, -0.9), xytext=(0.225, -0.5), arrowprops=prop, color='black', alpha=0.75)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.4],
                facecolor=[0, 0, 0, 0.4])
    ax1.annotate("", xy=(0.031, -1.3), xytext=(0.041, -1.6), arrowprops=prop, color='black', alpha=0.5)

    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['0', '1'], fontsize=pu.fs1)
    ax3.set_ylim([0, 1.75])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['0', '1'], fontsize=pu.fs1)
    ax4.set_ylim([0, 1.75])


def main(save_pdf=False, show_plot=True):

    f = plt.figure(figsize=(4.5, 1.75), dpi=150)
    gs = gridspec.GridSpec(2, 5)
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[1, 0])
    ax3 = f.add_subplot(gs[0, 1:3])
    ax4 = f.add_subplot(gs[1, 1:3])
    ax5 = f.add_subplot(gs[0, 3:])
    ax6 = f.add_subplot(gs[1, 3:])

    plot_noisy_inhib_pop_boundary(ax1=ax1, ax2=ax2, seed=32)
    plot_noisy_inhib_pop_dynamics(ax1=ax3, ax2=ax4, ax3=ax5, ax4=ax6, seed1=45, seed2=55)

    # adjust spacing and subplots
    f.tight_layout()
    f.subplots_adjust(wspace=1.25, hspace=0.5)

    for ax in [ax1, ax2]:
        box = ax.get_position()
        box.x1 = box.x1 + 0.02
        ax.set_position(box)
    for ax in [ax3, ax4]:
        box = ax.get_position()
        box.x0 = box.x0 + 0.005
        box.x1 = box.x1 + 0.005
        ax.set_position(box)

    # panel labels
    ax1.text(-0.65, 1.15, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax3.text(-0.25, 1.15, r'\textbf{b}', transform=ax3.transAxes, **pu.panel_prms)
    ax4.text(-0.25, 1.15, r'\textbf{c}', transform=ax4.transAxes, **pu.panel_prms)
    ax5.text(-0.3, 1.15, r'\textbf{d}', transform=ax5.transAxes, **pu.panel_prms)
    ax6.text(-0.3, 1.15, r'\textbf{e}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_06_noisy_inhib_net.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
