import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf


def plot_noisy_inhib_pop_boundary(ax1=None):
    """
    Plots for panel (a) -- inhibitory boundary, with one neuron highlighted.
    :param ax1: Figure axis.
    :return:
    """

    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5
    N = 20
    x_lims = [-1.0, 1.0]  # [-0.9, 0.9]

    # get neuron parameters
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, x_lims)

    x_range = [-1, 1]
    x_ideal = np.linspace(x_range[0], x_range[1], 101)
    y_ideal = scnf.y_quad_fcn(a, b, c)(x_ideal)
    ax1.plot(x_ideal, y_ideal, c='gray', alpha=0.5, linestyle=':', linewidth=1.5)

    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    yb = np.zeros((N, 21))
    x = np.linspace(x_range[0], x_range[1], 21)
    for i in range(N):
        y = scnf.single_nrn_boundary(x, E[i], F[i], T[i])
        yb[i, :] = y
        if i == 5:
            ax1.plot(x, y, linewidth=1.5, c=cmap[i], alpha=1.0)
        else:
            ax1.plot(x, y, linewidth=0.5, c=cmap[i], alpha=0.25)
    ax1.fill_between(x, np.zeros_like(x), np.min(yb, 0), color=pu.inhibitory_blue, alpha=0.1)
    ax1.axvline(x=-0.5, c='gray', alpha=0.5, linewidth=0.5)

    # formatting
    ytcks = [-2, -1, 0]
    for ax in [ax1]:
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
    ax1.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    ax1.set_xlabel('Input, $x$', fontsize=pu.fs2)


def plot_noisy_inhib_pop_dynamics(ax1=None, ax2=None, ax3=None, seed1=None, recurrence=True):
    """
    Plots for panels (c), (d), (e) and (f): noisy inhibitory population, run for two trials
    :param ax1: Figure axis for the latent variable readout (trial 1)
    :param ax2: Figure axis for the spike raster (trial 1)
    :param ax3: Figure axis for two neurons' voltages (trial 1)
    :param ax4: Figure axis for two neurons' voltages (trial 2)
    :param seed1: Random seed for trial 1
    :param seed2: Random seed for trial 2
    :param recurrence: TBD.
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
    # D += np.linspace(-0.75,0.25,N)

    # noisy simulation
    mu = 0.0
    D_scale = 0.25  # 2.#0.25
    D = D_scale * D[None, :]
    E = E[:, None]
    F = F[:, None]
    if not recurrence:
        pw = 1
    else:
        pw = None
    #     T = 0.35 * np.ones_like(T)
    dt = 1e-5
    Tend = 0.075
    times = np.arange(0, Tend, dt)
    init_pd = int(0.025 / dt)
    nT = len(times)
    xrange = [-1, 1]
    # x = np.concatenate((xrange[0] * np.ones((int((1. / 6) * nT),)),
    #                     np.linspace(xrange[0], xrange[1], int((5. / 6) * nT))))[None, :]
    x = -0.5 * np.ones((nT,))[None, :]
    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))

    nrn_nr = 12
    n_trials = 20
    sigma_v_vals = np.array([0, 0.025])
    for i, (sigma_v, ax) in enumerate(zip(sigma_v_vals, [ax1, ax2])):
        for nt in range(n_trials):
            if sigma_v > 0:
                mu = 0.025
            else:
                mu = 0.0
            (s, V, _) = scnf.run_spiking_net(x, D, E, F, T, mu=mu, dt=dt, sigma_v=sigma_v, seed=seed1,
                                             recurrence=recurrence, pw=pw)
            spk_times = scnf.get_spike_times(s, dt=dt)
            ax.plot(spk_times[nrn_nr], (nt + 1) * np.ones_like(spk_times[nrn_nr]), '.',
                    color=cmap[nrn_nr], markersize=2)

    # formatting
    ytcks = [1, int(n_trials/2), n_trials]
    for ax in [ax1, ax2]:
        ax.set_xlim([0.025, 0.075])
        ax.set_xticks(np.linspace(0.025, 0.075, 3))
        ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,0.25,6)],fontsize=pfcn.fs1)
        ax.set_yticks(ytcks)
        ax.set_yticklabels([str(s) for s in ytcks], fontsize=pu.fs1)
        ax.set_ylim([0, n_trials+2])
        ax.set_ylabel('Trial Nr.', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        sns.despine(ax=ax)
    ax2.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, 0.05, 3)], fontsize=pu.fs1)
    ax2.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)

    # now do a longer one
    n_trials = 1000
    sigma_v_vals = np.linspace(0, 0.2, 11)   # np.array([0, 0.0125, 0.025, 0.05, 0.1, 0.25])  # , 0.6, 0.7, 0.8, 0.9, 1.0])
    spike_counts = np.zeros((len(sigma_v_vals), n_trials))  # , N))
    for i, sigma_v in enumerate(sigma_v_vals):
        print(f'{i + 1} of {len(sigma_v_vals)}')
        for nt in range(n_trials):
            (s, V, _) = scnf.run_spiking_net(x, D, E, F, T, mu=mu, dt=dt, sigma_v=sigma_v, seed=seed1,
                                             recurrence=recurrence)
            # spike_counts[i, nt, :] = np.sum(s, 1)
            spike_counts[i, nt] = np.sum(s[nrn_nr, :])

    return sigma_v_vals, spike_counts


def main(save_pdf=False, show_plot=True):

    # set up figure and subplots
    f = plt.figure(figsize=(4.5, 2.5), dpi=150)
    gs = f.add_gridspec(6, 9)
    ax1 = f.add_subplot(gs[:3, :3])
    ax2 = f.add_subplot(gs[3:, :3])
    ax3 = f.add_subplot(gs[:3, 3:6])
    ax4 = f.add_subplot(gs[3:, 3:6])
    ax6 = f.add_subplot(gs[:3, 6:])
    ax7 = f.add_subplot(gs[3:, 6:])

    plot_noisy_inhib_pop_boundary(ax1)
    sigma_v, spike_counts = plot_noisy_inhib_pop_dynamics(ax1=ax3, ax2=ax4, seed1=None)
    _, spike_counts2 = plot_noisy_inhib_pop_dynamics(ax1=ax6, ax2=ax7, seed1=None, recurrence=False)
    ax3.text(0.99, 0.95, r'$\sigma_V$$=$$0$', transform=ax3.transAxes, fontsize=pu.fs1, color='black',
             horizontalalignment='right')
    ax4.text(0.99, 0.95, r'$\sigma_V$$=$$0.01$', transform=ax4.transAxes, fontsize=pu.fs1, color='black',
             horizontalalignment='right')
    ax3.set_title('Recurrent inhibition', fontsize=pu.fs2)
    ax6.set_title('Feedforward', fontsize=pu.fs2)
    for ax in [ax6, ax7]:
        ax.set_ylabel('')

    ax2.plot(sigma_v, np.std(spike_counts, 1) / np.mean(spike_counts, 1), color='black', linewidth=1)
    ax2.plot(sigma_v, np.std(spike_counts2, 1) / np.mean(spike_counts2, 1), color='gray', linewidth=1)
    ax2.text(0.75, 1.0, 'default', transform=ax2.transAxes, fontsize=pu.fs1, color='black',
             horizontalalignment='right')
    ax2.text(0.75, 0.2, 'feedforward', transform=ax2.transAxes, fontsize=pu.fs1, color='gray',
             horizontalalignment='right')

    ax2.set_xlim([0.0, np.max(sigma_v)])
    ax2.set_xticks([0, np.max(sigma_v)/2, np.max(sigma_v)])
    ax2.set_xticklabels(['0.00', '%.2f' % (np.max(sigma_v)/2), '%.2f' % np.max(sigma_v)], fontsize=pu.fs1)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax2.set_ylim([0, 1.35])
    ax2.set_ylabel('Spike count CV', fontsize=pu.fs2)
    ax2.set_xlabel(r'Voltage noise stdev., $\sigma_V$', fontsize=pu.fs2)
    ax2.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine(ax=ax2)
    f.subplots_adjust(wspace=4, hspace=1.5)

    # add panel labels
    ax1.text(-0.4, 1.1, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.4, 1.075, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.3, 1.15, r'\textbf{c}', transform=ax3.transAxes, **pu.panel_prms)
    ax6.text(-0.3, 1.15, r'\textbf{d}', transform=ax6.transAxes, **pu.panel_prms)

    box = ax1.get_position()
    box.x1 = box.x1 - 0.02
    box.y0 = box.y0 + 0.02
    ax1.set_position(box)
    box = ax2.get_position()
    box.x1 = box.x1 - 0.04
    box.y1 = box.y1 - 0.04
    ax2.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_s2_noise_irregularity_relationship.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
