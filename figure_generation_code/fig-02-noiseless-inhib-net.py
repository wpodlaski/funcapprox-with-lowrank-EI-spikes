import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf


def plot_single_inhibitory_boundary(ax):
    """
    Plots panel (a), the input-output voltage relationship for a single inhibitory neuron.
    :param ax: Figure axis.
    :return:
    """

    # set parameters
    xlims = [-1, 1]
    ylims = [-2, 0]
    E = 1.75
    F = 2.5
    T = -1  # neuron is spontaneously active (equivalent to including a background current)
    D = -1

    # boundary
    x = np.linspace(xlims[0], xlims[1], 21)
    y = scnf.single_nrn_boundary(x, E, F, T)

    ax.plot(x, y, linewidth=1, c=pu.inhibitory_blue)
    ax.plot(x, y - 0.45, linewidth=1, c=pu.inhibitory_blue, linestyle=':', alpha=0.5)
    ax.fill_between(x, np.zeros_like(x), y, color=pu.inhibitory_blue, alpha=0.1)
    ax.quiver([0.4], [(T - F * 0.4) / E], [0.005 * F], [0.005 * E], color=pu.inhibitory_blue,
              width=0.01, headlength=5, headwidth=5, scale=0.05, scale_units='y')
    ax.text(0.42, -0.75, r'$\begin{pmatrix} F_1 \\ E_1 \end{pmatrix}$', color=pu.inhibitory_blue, fontsize=pu.fs1)

    # leak quivers
    xlim = np.arange(-0.92, 0.95, 0.2)
    ylim = np.arange(-1.75, -0., 0.33)
    X, Y = np.meshgrid(xlim, ylim)
    u = 0
    v = -Y
    v[F * X + E * Y - T > -0.5] = np.nan
    ax.quiver(X, Y, u, v, color='gray', alpha=0.5, width=0.005, headlength=5, headwidth=5)

    # decoder
    xloc = 0.056  # x location of decoder arrow
    ax.quiver([xloc], [scnf.single_nrn_boundary(xloc, E, F, T)], [0], [0.02167 * D], color=pu.inhibitory_blue,
              width=0.01, headlength=5, headwidth=5, scale=0.05, scale_units='y')
    ax.text(-0.37, -0.92, r'$D_1\begin{cases} & \\ & \end{cases}$', color=pu.inhibitory_blue, fontsize=pu.fs1)

    # sub/supra text
    ax.text(-0.9, -1.92, r'\textbf{subthreshold}', fontweight='bold', fontsize=pu.fs1)
    ax.text(-0.1, -0.2, r'\textbf{suprathreshold}', color=pu.inhibitory_blue, fontweight='bold', fontsize=pu.fs1)

    # formatting
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    ax.set_yticks([-2, -1, 0])
    ax.set_yticklabels(['-2', '-1', '0'], fontsize=pu.fs1)
    ax.set_xlabel('Input signal, $x$', fontsize=pu.fs2)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2, labelpad=7)
    ax.axhline(y=0, linewidth=1.0, color='black')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)

    # move x-axis to the top
    ax.axhline(y=0, linewidth=0.5, color='black')
    sns.despine(ax=ax, bottom=True)


def plot_inhibitory_boundary(ax1=None, ax2=None):
    """
    Plots panels (b) and (c), the input-output boundaries for a network of 10 inhibitory neurons.
    :param ax1: Figure axis for the version with individual neuron thresholds
    :param ax2: Figure axis for the version with a single boundary, plus decoder arrows
    :return:
    """

    # boundary parameters: y = a*x^2 + b*x + c
    a = -1.0
    b = 0
    c = -0.5

    # get neuron parameters
    N = 10
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, [-0.9, 0.9])

    # decoders: make larger & variable for illustration purposes
    D *= -0.25
    D += np.linspace(-0.15, 0.05, N)

    # plotting boundaries
    xlim = np.arange(-1, 1, 0.2)  # np.arange(-0.92, 0.95, 0.2)
    ylim = np.arange(-1.75, -0., 0.33)  # np.arange(-2.75, -0.1, 0.3)
    X, Y = np.meshgrid(xlim, ylim)
    u = 0
    v = -Y

    x_range = [-1, 1]
    x_ideal = np.linspace(x_range[0], x_range[1], 101)
    y_ideal = scnf.y_quad_fcn(a, b, c)(x_ideal)
    if ax1 is not None:
        ax1.plot(x_ideal, y_ideal, c='gray', alpha=0.5, linestyle=':', linewidth=2)

    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    yb = np.zeros((N, 21))
    x = np.linspace(x_range[0], x_range[1], 21)
    for i in range(N):
        y = scnf.single_nrn_boundary(x, E[i], F[i], T[i])
        yb[i, :] = y
        ax1.plot(x, y, linewidth=1, c=cmap[i], alpha=0.5)
        v[F[i] * X + E[i] * Y - T[i] > -0.2] = np.nan
        ax2.quiver(xvals[i], yvals[i],
                   np.array([0]), np.array([0.05 * D[i]]), color=pu.inhibitory_blue,
                   width=0.01, headlength=5, headwidth=5, scale=0.05, scale_units='y')
    ax2.fill_between(x, np.min(yb + np.tile(D, (21, 1)).T, 0), np.min(yb, 0), color='gray', alpha=0.1)
    ax2.plot(x, np.min(yb, 0), c=pu.inhibitory_blue, linewidth=1)
    ax2.fill_between(x, np.zeros_like(x), np.min(yb, 0), color=pu.inhibitory_blue, alpha=0.1)
    ax2.quiver(X, Y, u, v, color='gray', alpha=0.5, width=0.005, headlength=5, headwidth=5)

    ax1.text(-0.3, -1.5, '$y=-f_{cvx}(x)$', fontsize=pu.fs1)
    ax2.text(-0.9, -1.95, r'\textbf{subthreshold}', fontsize=pu.fs1)
    ax2.text(-0.06, -0.2, r'\textbf{suprathreshold}', fontsize=pu.fs1,
             color=pu.inhibitory_blue)

    # formatting
    for ax in [ax1, ax2]:
        ax.set_xlim(x_range)
        ax.set_ylim([-2, 0])
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
        ax.set_yticks([-2, -1, 0])
        ax.set_yticklabels(['-2', '-1', '0'], fontsize=pu.fs1)
        ax.set_xlabel('Input signal, $x$', fontsize=pu.fs2)
        ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.axhline(y=0, linewidth=1.0, color='black')
        sns.despine(ax=ax, bottom=True)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2, labelpad=7)


def plot_inhibitory_dynamics(ax1=None, ax2=None):
    """
    Plots the two suplots of panel (d), the dynamics of the inhibitory network for x from -1 to 1.
    :param ax1: Figure panel for the latent variable output
    :param ax2: Figure panel for the spike raster
    :return:
    """
    # boundary parameters
    a = -1.0
    b = 0
    c = -0.5

    N = 10
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, [-0.9, 0.9])

    # decoders: make larger & variable for illustration purposes
    D *= -0.25
    D += np.linspace(-0.15, 0.05, N)

    # deterministic simulation
    mu = 0.
    sigma_v = 0.
    D_scale = 2.
    D = D_scale * D[None, :]
    E = E[:, None]
    F = F[:, None]
    dt = 1e-4
    leak = 100.
    Tend = 0.3
    init_pd = int(0.05 / dt)
    times = np.arange(0, Tend, dt)
    nT = len(times)
    xrange = [-1, 1]
    x = np.concatenate((xrange[0] * np.ones((int((1. / 6) * nT),)),
                        np.linspace(xrange[0], xrange[1], int((5. / 6) * nT))))[None, :]
    s, _, _ = scnf.run_spiking_net(x, D, E, F, T, mu=mu, dt=dt, leak=leak, sigma_v=sigma_v)
    r = scnf.exp_filter(s, times)
    s = s[:, init_pd:]
    r = r[:, init_pd:]
    x = x[:, init_pd:]
    times = times[:-init_pd]
    y = D @ r
    spk_times = scnf.get_spike_times(s, dt=dt)

    # simulation plotting
    _ = ax1.plot(times, x[0, :], '-', c='gray', alpha=0.5, linewidth=1)
    _ = ax1.plot(times, scnf.y_quad_fcn(a, b, c)(x[0, :]), '--', c='black', alpha=0.75,
                 linewidth=1, label=r"$y$ bndry.")
    _ = ax1.plot(times, y[0, :], c='black', linewidth=0.5, alpha=0.5, label=r"$y$ sim.")
    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    for i in range(N):
        ax2.plot(spk_times[i], i * np.ones_like(spk_times[i]), '.', color=cmap[i], markersize=2)

    # formatting
    for ax in [ax1, ax2]:
        ax.set_xlim([0.0, 0.25])
        ax.set_xticks(np.linspace(0.0, 0.25, 6))
        ax.set_xticklabels([])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        sns.despine(ax=ax)
    ax2.set_xticklabels(['%.0f' % (leak * x) for x in np.linspace(0.0, 0.25, 6)], fontsize=pu.fs1)
    ytcks1 = [-2, -1, 0, 1]
    ytcks2 = [0, 5, 10]
    ax1.set_yticks(ytcks1)
    ax2.set_yticks(ytcks2)
    ax1.set_yticklabels([str(s) for s in ytcks1], fontsize=pu.fs1)
    ax2.set_yticklabels([str(s) for s in ytcks2], fontsize=pu.fs1)
    ax1.set_ylim([-2, 1])
    ax2.set_ylim([-1, N])
    ax1.set_ylabel(r'Latent var., $y$', fontsize=pu.fs2)
    ax2.set_ylabel('Neuron ID', fontsize=pu.fs2)
    ax2.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    ax1.text(0.2, 0.8, 'input, $x$', fontsize=pu.fs1, color='gray', alpha=0.5, rotation=10)
    ax1.text(0.19, -0.35, '$y$, boundary', fontsize=pu.fs1, color='black', alpha=0.75)
    ax1.text(0.1, -1.75, '$y$, simulation', fontsize=pu.fs1, color='black', alpha=0.5)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.7],
                facecolor=[0, 0, 0, 0.7])
    ax1.annotate("", xy=(0.215, -0.9), xytext=(0.225, -0.5), arrowprops=prop, color='black', alpha=0.75)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.2", shrinkA=0, shrinkB=0, edgecolor=[0, 0, 0, 0.4],
                facecolor=[0, 0, 0, 0.4])
    ax1.annotate("", xy=(0.087, -1.3), xytext=(0.097, -1.6), arrowprops=prop, color='black', alpha=0.5)


def plot_3d_inhibitory_boundary(ax):
    """
    Plots panel (e), the 3D plot for an inhibitory network that receives a 2D input
    :param ax: Figure axis for the plot, must be a 3d axis.
    :return:
    """
    # boundary parameters
    a1 = -0.5
    b1 = 0.5
    a2 = -0.5
    b2 = -0.5
    a3 = -0.1
    c = -5
    N_3d = 36
    x1_range = [-4.5, 4.5]
    x2_range = [-4.5, 4.5]
    (D, E, F1, F2, T, _, _, _) = scnf.fcn2d_to_nrns(scnf.y_3d_quad_fcn(a1, a2, a3, b1, b2, c),
                                                    scnf.ydx1_quad_fcn(a1, a3, b1),
                                                    scnf.ydx2_quad_fcn(a2, a3, b2),
                                                    N_3d, x1_range, x2_range, sigma=0.5, switch_order=True)

    X1, X2 = np.meshgrid(np.linspace(-5, 5, 501), np.linspace(-5, 5, 501))
    Z = []
    for i in range(N_3d):
        Z.append((F1[i] * X1 + F2[i] * X2 - T[i]) / E[i])
    Z_stack = np.stack(Z)
    Z_min = np.min(Z_stack, 0)
    inh_2d_cmap_list = pu.get_2d_inh_cmap(N_3d)
    for i in range(N_3d):
        Z_tmp = Z_stack[i, :, :]
        Z_tmp[Z_tmp > Z_min] = np.nan
        _ = ax.plot_surface(X1, X2, Z_tmp, cmap=inh_2d_cmap_list[i],
                            linewidth=0, antialiased=True, alpha=0.5, vmin=0, vmax=30)

    # formatting
    pu.set_3d_plot_specs(ax)
    ax.set_xlabel('Input 1, $x_1$', fontsize=pu.fs2, labelpad=-14)
    ax.set_ylabel('Input 2, $x_2$', fontsize=pu.fs2, labelpad=-14)
    ax.set_zlabel('Latent variable, $y$', fontsize=pu.fs2, rotation=90, labelpad=-12)
    _ = ax.set_xticks([-5, 0, 5])
    _ = ax.set_yticks([-5, 0, 5])
    _ = ax.set_zticks([-40, -20, 0])
    _ = ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    _ = ax.set_yticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    _ = ax.set_zticklabels(['-4', '-2', '0'], fontsize=pu.fs1)
    ax.tick_params(axis='x', which='major', pad=-6, width=0.5, length=3)
    ax.tick_params(axis='y', which='major', pad=-6, width=0.5, length=3)
    ax.tick_params(axis='z', which='major', pad=-5, width=0.5, length=3)
    ax.set_box_aspect((np.ptp(X1), np.ptp(X2), 0.3 * np.ptp(Z_min[~np.isnan(Z_min)])), zoom=1.42)
    ax.view_init(elev=25, azim=305)
    ax.text(-5, -5, -33, r'$y=-f_{cvx}(\mathbf{x})$', fontsize=pu.fs1)


def main(save_pdf=False, show_plot=True):

    # set up figure and subplots
    f = plt.figure(figsize=(4.5, 3), dpi=150)
    gs = f.add_gridspec(4, 6, height_ratios=[1, 1, 1.5, 1.5])  # 0.025
    ax1 = f.add_subplot(gs[0:2, 0:2])
    ax2 = f.add_subplot(gs[0:2, 2:4])
    ax3 = f.add_subplot(gs[0:2, 4:6])
    ax4 = f.add_subplot(gs[2, 0:3])
    ax5 = f.add_subplot(gs[3, 0:3])
    ax6 = f.add_subplot(gs[2:4, 3:6], projection='3d')

    # call plotting functions
    plot_single_inhibitory_boundary(ax1)
    plot_inhibitory_boundary(ax1=ax2, ax2=ax3)
    plot_inhibitory_dynamics(ax1=ax4, ax2=ax5)
    f.tight_layout()  # tightening layout before 3d plot works better
    plot_3d_inhibitory_boundary(ax6)

    # add panel labels
    ax1.text(-0.35, 1.075, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.35, 1.075, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.35, 1.075, r'\textbf{c}', transform=ax3.transAxes, **pu.panel_prms)
    ax4.text(-0.20, 1.175, r'\textbf{d}', transform=ax4.transAxes, **pu.panel_prms)
    ax6.text(-9, -9, 4, r'\textbf{e}', **pu.panel_prms)

    # manual adjustment of subplots to look nicer
    f.subplots_adjust(wspace=2.0, hspace=1)

    box = ax4.get_position()
    box.y0 = box.y0 - 0.06
    box.y1 = box.y1 - 0.06
    ax4.set_position(box)

    box = ax5.get_position()
    box.y0 = box.y0 + 0.0
    box.y1 = box.y1 + 0.0
    ax5.set_position(box)

    box = ax6.get_position()
    box.x0 = box.x0 + 0.02
    box.x1 = box.x1 + 0.02
    box.y0 = box.y0 + 0.02
    box.y1 = box.y1 + 0.02
    ax6.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_02_noiseless_inhibition.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
