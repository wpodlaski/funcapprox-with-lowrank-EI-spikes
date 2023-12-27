import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plot_utils as pu
import net_sim_code as scnf


cmap = plt.get_cmap('gist_earth')
cmap = cmap(np.linspace(0, 0.9, 10))
cmap = [cmap[2], cmap[5]]
clrs = ['darkseagreen', pu.inhibitory_blue]


def plot_noiseless_two_neurons(ax1, ax2, ax3):
    """
    Plots for panels (a,b,c) -- The two-neuron network with no noise
    :param ax1: Figure axis for (r_1,r_2) space illustrating threshold boundaries.
    :param ax2: Figure axis for (r_1,r_2) space illustrating dynamics, and readout and orthogonal spaces.
    :param ax3: Figure axis for latent and orthogonal readouts over time.
    :return:
    """

    N = 2
    mu = 0.
    sigma_v = 0.0
    D_scale = 0.5
    D = D_scale * np.array([1., 1.])[None, :]
    Dorth = D_scale * np.array([-1., 1.])[None, :]
    E = np.array([1., 1.])[:, None]
    F = np.array([1., 1.])[:, None]
    T = np.array([0.25, 1.25])
    dt = 1e-4
    Tend = 0.2
    times = np.arange(0, Tend, dt)
    nT = len(times)
    init_pd = int(0.05 / dt)
    x0 = 4.
    x = x0 * np.ones((1, nT))
    r0 = np.array([6, 4])

    s, V, _ = scnf.run_spiking_net(x, -D, E, F, T, mu=mu, sigma_v=sigma_v, r0=r0)
    r = scnf.exp_filter(s, times, r0=r0)

    # first plot the initial period
    ax1.plot(r[0, :110], r[1, :110], c='black', linewidth=0.5)

    # next cut out initial period and plot the steady state
    r = r[:, init_pd:]
    times = times[:-init_pd]
    y = D @ r
    y_orth = Dorth @ r

    offsets = [-0, 0]
    arrows = [[1, 0], [0, 1]]
    idxs = [50, 60]
    r0 = np.linspace(-4, 10, 141)
    for i in range(N):
        r1 = (-D[0, 0] * r0 - (T[i] - F[i] * x0) / E[i]) / D[0, 1] + offsets[i]
        ax1.plot(r0, r1, c=cmap[i], linewidth=1.0)
        ax1.fill_between(r0, -np.ones_like(r1), r1, color=cmap[i], alpha=0.1)
        ax1.quiver(r0[idxs[i]], r1[idxs[i]],
                   np.array([0.075 * arrows[i][0]]), np.array([0.075 * arrows[i][1]]), color=cmap[i],
                   width=0.015, headlength=5, headwidth=5, scale=0.05, scale_units='y')

    ax2.plot(r[0, :], r[1, :] + 0.025 * np.random.normal(size=(r.shape[1],)), c='black', linewidth=0.5)

    ax3.plot(times, y[0, :], c=clrs[1], linewidth=0.5)
    ax3.plot(times, y_orth[0, :], c='silver', linewidth=0.5)
    ax3.legend(('latent readout', 'orthogonal readout'), ncol=1, fontsize=pu.fs1, frameon=False)

    ax1.text(4.5, 6.5, r'\textbf{sub-}''\n 'r'\textbf{threshold}', color='black', fontweight='bold',
             fontsize=pu.fs1)
    ax1.text(0.25, 0.25, r'\textbf{supra-}''\n 'r'\textbf{threshold}', color=cmap[0], fontweight='bold',
             fontsize=pu.fs1)

    ax2.text(1., 6.0, 'orthogonal\n subspace', color='silver', fontsize=pu.fs1)
    ax2.text(5.0, 3, 'latent\n subspace', color=clrs[1], fontsize=pu.fs1)
    ax2.annotate('fixed pt.', xy=(7.8, 0.05), xytext=(5.45, 1.25), xycoords='data',
                 arrowprops=dict(facecolor='black', lw=0.5, arrowstyle='-|>'),
                 bbox=dict(pad=0.4, facecolor="none", edgecolor="none"),
                 fontsize=pu.fs1)


def plot_noisy_two_neurons(ax1, ax2, ax3, seed=None):
    """
    Plots for panels (d,e,f) -- The two-neuron network WITH NOISE.
    :param ax1: Figure axis for (r_1,r_2) space illustrating noisy threshold boundaries.
    :param ax2: Figure axis for (r_1,r_2) space illustrating dynamics and readout and orthogonal spaces.
    :param ax3: Figure axis for latent and orthogonal readouts over time.
    :param seed: Random seed for the noise.
    :return:
    """

    N = 2
    mu = 0.05
    sigma_v = 0.2
    D_scale = 0.5
    D = D_scale * np.array([1., 1.])[None, :]
    Dorth = D_scale * np.array([-1., 1.])[None, :]
    E = np.array([1., 1.])[:, None]
    F = np.array([1., 1.])[:, None]
    T = np.array([0.5, 1.0])
    dt = 1e-4
    Tend = 0.2
    times = np.arange(0, Tend, dt)
    nT = len(times)
    init_pd = int(0.05 / dt)
    x0 = 4.
    x = x0 * np.ones((1, nT))

    offsets = [-0, 0]
    r0 = np.linspace(0, 10, 21)
    for i in range(N):
        r1 = (-D[0, 0] * r0 - (T[i] - F[i] * x0) / E[i]) / D[0, 1] + offsets[i]
        for j in range(10):
            ax1.plot(r0, r1 + 0.5 * np.random.normal(), c=cmap[i], linewidth=0.5, alpha=0.5)
        ax1.plot(r0, r1, c=cmap[i], linewidth=1.0)

    r0 = np.array([5, 5])
    T = np.array([0.7, 0.8])

    s, V, _ = scnf.run_spiking_net(x, -D, E, F, T, mu=mu, sigma_v=sigma_v, r0=r0, seed=seed)
    r = scnf.exp_filter(s, times, r0=r0)

    # first plot the initial period
    ax1.plot(r[0, :100], r[1, :100], c='black', linewidth=0.5)

    # next cut out initial period and plot the steady state
    r = r[:, init_pd:]
    times = times[:-init_pd]
    y = D @ r
    y_orth = Dorth @ r
    ax2.plot(r[0, :], r[1, :], c='black', linewidth=0.5)
    ax3.plot(times, y[0, :], c=clrs[1], linewidth=0.5)
    ax3.plot(times, y_orth[0, :], c='silver', linewidth=0.5)


def main(save_pdf=False, show_plot=True):

    f = plt.figure(figsize=(4.5, 2.25), dpi=150)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.5, 1])

    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[0, 2])
    ax4 = f.add_subplot(gs[0, 3])
    ax5 = f.add_subplot(gs[1, :2])
    ax6 = f.add_subplot(gs[1, 2:])

    plot_noiseless_two_neurons(ax1, ax2, ax5)
    plot_noisy_two_neurons(ax3, ax4, ax6, seed=13)

    for ax in [ax2, ax4]:
        ax.plot([0, 6.5], [6.5, 0], color='silver', linewidth=1.5, linestyle=':')
        ax.arrow(3, 3, 2, 2, color=clrs[1], linewidth=1, head_length=0.5, head_width=0.5, zorder=1)
        ax.arrow(3, 3, -2, -2, color=clrs[1], linewidth=1, head_length=0.5, head_width=0.5, zorder=1)

    # FORMATTING
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([0, 4, 8])
        ax.set_xticklabels(['0', '4', '8'], fontsize=pu.fs1)
        ax.set_yticks([0, 4, 8])
        ax.set_yticklabels(['0', '4', '8'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Rate, $r_1$', fontsize=pu.fs2)
        ax.set_ylabel(r'Rate, $r_2$', fontsize=pu.fs2)
        ax.set_xlim([-0.5, 8])
        ax.set_ylim([-0.5, 8])
        ax.axhline(y=0, c='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, c='gray', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('', fontsize=pu.fs2)
    ax4.set_ylabel('', fontsize=pu.fs2)

    for ax in [ax5, ax6]:
        ax.set_xticks([0, 0.075, 0.15])
        ax.set_xticklabels(['0', '7.5', '15'], fontsize=pu.fs1)
        ax.set_xlim([0, 0.15])
        ax.set_yticks([-4, 0, 4])
        ax.set_yticklabels(['-4', '0', '4'], fontsize=pu.fs1)
        ax.set_ylim([-5, 5])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
        ax.set_ylabel(r'Readouts', fontsize=pu.fs2)

    f.subplots_adjust(wspace=-1.25, hspace=-1)
    sns.despine()
    f.tight_layout()

    for ax in [ax1, ax3]:
        box = ax.get_position()
        box.x1 = box.x1 + 0.02
        ax.set_position(box)

    for ax in [ax2, ax4]:
        box = ax.get_position()
        box.x0 = box.x0 - 0.02
        ax.set_position(box)

    for ax in [ax5, ax6]:
        box = ax.get_position()
        box.y0 = box.y0 - 0.02
        ax.set_position(box)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax1.text(-0.375, 1.1, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.2, 1.1, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.375, 1.1, r'\textbf{d}', transform=ax3.transAxes, **pu.panel_prms)
    ax4.text(-0.2, 1.1, r'\textbf{e}', transform=ax4.transAxes, **pu.panel_prms)
    ax5.text(-0.165, 1.1, r'\textbf{c}', transform=ax5.transAxes, **pu.panel_prms)
    ax6.text(-0.165, 1.1, r'\textbf{f}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_07_two_neuron_nullspace_noise.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
