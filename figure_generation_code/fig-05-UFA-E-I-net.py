import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.interpolate import RegularGridInterpolator
import plot_utils as pu
import net_sim_code as scnf


def plot_3d_EI_intuition_plot(ax0):
    """
    Plots panel (a), the 3d plot illustrating the E/I boundary intersection as a function of the input x.
    Note that this is an illustration and is not simulated.
    :param ax0: Figure axis.
    :return:
    """

    # define parameters
    a_y_I = 0.1
    b_y_I = 3.
    d_I = 1.
    a_y_E = 0.09
    b_y_E = 20.
    d_E = 25.

    def yI(yE, x_):
        return a_y_I * (yE - b_y_I) ** 2 + 0.32 + d_I + 0. * x_

    def yI_yEinv(yE, x_):
        return -np.sqrt((yE - d_E + 1.6 ** 2) / (-a_y_E)) + b_y_E + 0. * x_

    x_vals = np.linspace(0, 10, 201)[::-1]
    yE_vals = np.linspace(0, 20, 201)
    YE, X = np.meshgrid(yE_vals, x_vals)
    Z_I = yI(YE, X)
    Z_I[Z_I > 30] = np.nan
    Z_E = yI_yEinv(YE, X)
    Z_E += 0.5 * X
    Z_I -= 0.2 * X

    # plot the lines
    xs = [0, 5, 10]
    alphas = [1.0, 0.6, 0.3]
    for i in range(3):
        x = xs[i]
        alph = alphas[i]
        idx = np.where(X[:, 0] == x)[0][0]
        ax0.plot(YE[idx, :], X[idx, :], Z_I[idx, :], c=pu.inhibitory_blue, alpha=alph, linewidth=1.)
        ax0.plot(YE[idx, :], X[idx, :], Z_E[idx, :], c=pu.excitatory_red, alpha=alph, linewidth=1.)

    Z_E2 = Z_E.copy()
    Z_I2 = Z_I.copy()
    Z_E2[Z_E2 < Z_I2] = np.nan
    Z_I2[Z_I2 < Z_E2] = np.nan

    _ = ax0.plot_surface(YE, X, Z_I2, cmap=pu.blue_cmap,
                         linewidth=0, antialiased=True, alpha=0.25, vmin=0, vmax=30)

    _ = ax0.plot_surface(YE, X, Z_E2, cmap=pu.red_cmap,
                         linewidth=0, antialiased=True, alpha=0.25, vmin=0, vmax=30)

    ax0.set_box_aspect((np.ptp(YE), 1.5 * np.ptp(X), 0.75 * np.ptp(Z_I[~np.isnan(Z_I)])), zoom=1.31)
    ax0.view_init(elev=25, azim=240)

    ylabs = [r'Input, $x$', r'Input ($x$)']
    xlabs = ['Excit. activity, ' + r'$y^E$', r'Inhibitory\n latent ($y_I$)']
    zlabs = ['Inhib. activity, ' + r'$|y^I|$', r'Excit. latent ($y_E$)']
    ax0.axes.set_xlim3d(left=-2, right=20)
    ax0.axes.set_ylim3d(bottom=-1, top=11)
    ax0.axes.set_zlim3d(bottom=-5, top=31)
    _ = ax0.set_yticks([0, 5, 10])
    _ = ax0.set_xticks([0, 10, 20])
    _ = ax0.set_zticks([0, 10, 20, 30])
    _ = ax0.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)
    _ = ax0.set_xticklabels(['0', '10', '20'], fontsize=pu.fs1)
    _ = ax0.set_zticklabels(['0', '10', '20', '30'], fontsize=pu.fs1)
    ax0.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax0.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax0.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax0.zaxis.set_rotate_label(False)

    ax0.set_xlabel(xlabs[0], fontsize=pu.fs2, labelpad=-14)
    ax0.set_ylabel(ylabs[0], fontsize=pu.fs2, labelpad=-14)
    ax0.set_zlabel(zlabs[0], fontsize=pu.fs2, rotation=90, labelpad=-10)
    ax0.tick_params(axis='x', which='major', pad=-6)
    ax0.tick_params(axis='y', which='major', pad=-6)
    ax0.tick_params(axis='z', which='major', pad=-4)


def plot_UFA_intro(f, ax1, ax2, ax3, ax4):
    """
    Plots panel (b), which illustrates function approximation (UFA) through the difference of convex functions.
    :param f: Figure handle.
    :param ax1: Axis for the function itself.
    :param ax2: Axis for the excitatory boundary as a function of x
    :param ax3: Axis for the inhibitory boundary as a function of x
    :param ax4: Axis for the intersection of the two boundaries (latent outputs).
    :return:
    """

    dx = 0.05
    xmax = 10
    xvals = np.linspace(0, xmax, int(xmax / dx) + 1)

    # define parameters
    g_E_x = 0.4
    g_E_y = 0.3
    g_I_x = 0.4
    g_I_y = 0.9
    b_E_x = 0.25
    b_I_x = 0.1

    # define functions
    C_E_x = np.zeros_like(xvals)
    C_I_x = np.zeros_like(xvals)
    C_E_x += (0.8 / dx) * scnf.delta_kernel(2, xvals)
    C_I_x += (1.6 / dx) * scnf.delta_kernel(4, xvals)
    C_E_x += (1.6 / dx) * scnf.delta_kernel(6, xvals)
    C_I_x += (1.6 / dx) * scnf.delta_kernel(8, xvals)
    G_E_x = np.cumsum(dx * C_E_x) + g_E_x
    G_I_x = np.cumsum(dx * C_I_x) + g_I_x
    B_E_x = np.cumsum(dx * G_E_x) + b_E_x
    B_I_x = np.cumsum(dx * G_I_x) + b_I_x

    Einds = [0, 20 * 2, 20 * 6, 20 * 10]
    Iinds = [0, 20 * 4, 20 * 8, 20 * 10]
    avals = [0.35, 0.7, 1.0]

    ax1.plot(xvals, B_E_x - B_I_x, c='black', linewidth=1)
    for i in range(3):
        ax2.plot(xvals[Einds[i]:Einds[i + 1]], B_E_x[Einds[i]:Einds[i + 1]], c=pu.excitatory_red, linewidth=1,
                 alpha=avals[i])
        ax3.plot(xvals[Iinds[i]:Iinds[i + 1]], B_I_x[Iinds[i]:Iinds[i + 1]], c=pu.inhibitory_blue, linewidth=1,
                 alpha=avals[i])
    ax4.plot(xvals, B_E_x - B_I_x, c=pu.excitatory_red, linewidth=1)
    twin = ax4.twinx()
    twin.plot(xvals, -1 * (B_I_x - (g_I_y / g_E_y) * B_E_x), c=pu.inhibitory_blue, linewidth=1)

    # dotted lines between the plots
    con = patches.ConnectionPatch(xyA=(xvals[Einds[1]], 0.0),
                                  xyB=(xvals[Einds[1]], 3),
                                  coordsA="data", coordsB="data",
                                  axesA=ax1, axesB=ax2, color=pu.excitatory_red,
                                  linestyle='--', alpha=0.5, linewidth=0.5)
    f.add_artist(con)
    con2 = patches.ConnectionPatch(xyA=(xvals[Einds[2]], 0.0),
                                   xyB=(xvals[Einds[2]], 7.),
                                   coordsA="data", coordsB="data",
                                   axesA=ax1, axesB=ax2, color=pu.excitatory_red,
                                   linestyle='--', alpha=0.5, linewidth=0.5)
    f.add_artist(con2)
    con3 = patches.ConnectionPatch(xyA=(xvals[Iinds[1]], 1.35),
                                   xyB=(xvals[Iinds[1]], 3),
                                   coordsA="data", coordsB="data",
                                   axesA=ax1, axesB=ax3, color=pu.inhibitory_blue,
                                   linestyle='--', alpha=0.5, linewidth=0.5)
    f.add_artist(con3)
    con4 = patches.ConnectionPatch(xyA=(xvals[Iinds[2]], 1.35),
                                   xyB=(xvals[Iinds[2]], 12),
                                   coordsA="data", coordsB="data",
                                   axesA=ax1, axesB=ax3, color=pu.inhibitory_blue,
                                   linestyle='--', alpha=0.5, linewidth=0.5)
    f.add_artist(con4)

    for ax in [ax1, ax2, ax3, ax4, twin]:
        ax.set_xticks([0, 5, 10])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax4.set_xticklabels(['0', '5', '10'], fontsize=pu.fs1)

    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels([])
    ax2.set_yticks([0, 5, 10, 15])
    ax2.set_yticklabels([])
    ax3.set_yticks([0, 5, 10, 15])
    ax3.set_yticklabels([])
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels([])
    twin.set_yticks([0, 10, 20, 30])
    twin.set_yticklabels([])

    ax1.set_ylabel('Desired\n output', fontsize=pu.fs2)
    ax2.set_ylabel(r'$q(x)$', fontsize=pu.fs1)
    ax3.set_ylabel(r'$p(x)$', fontsize=pu.fs1)
    ax4.set_ylabel(r'$q$$-$$p$', fontsize=pu.fs1)
    twin.set_ylabel(r'$p$$-$$aq$', fontsize=pu.fs1, labelpad=1)
    ax4.set_xlabel(r'Input, $x$', fontsize=pu.fs2, labelpad=2.5)

    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    sns.despine(ax=ax4, left=False, right=False)
    sns.despine(ax=twin, left=False, right=False)


def plot_UFA_dynamics(ax1, ax2, ax3, ax4, prms):
    """
    Plots panels (c,d), the function approximation (UFA) example, in 3d and plots over time
    :param ax1: Figure axis for the 3d plot of the boundaries in (x, yE, yI) space
    :param ax2: Figure axis. Input, x.
    :param ax3: Figure axis. Spike rasters.
    :param ax4: Figure axis. Latent readouts, yE, yI.
    :param prms: Network parameters for the boundaries and simulation.
    :return:
    """

    dx = 0.05
    dyE = 0.05
    xmax = 10
    yEmax = 12
    xvals = np.linspace(0, xmax, int(xmax / dx) + 1)
    yEvals = np.linspace(0, yEmax, int(yEmax / dx) + 1)

    # define parameters
    g_E_x = 0.4
    g_E_y = 0.3
    g_I_x = 0.4
    g_I_y = 0.9

    b_E_x = 0.25
    b_E_y = 0.25
    b_I_x = 0.1
    b_I_y = -1.3

    # define functions
    C_E_x = np.zeros_like(xvals)
    C_E_y = np.zeros_like(yEvals)
    C_I_x = np.zeros_like(xvals)
    C_I_y = np.zeros_like(yEvals)

    # modifications to curvature
    C_E_x += (0.8 / dx) * scnf.delta_kernel(2, xvals)
    C_I_x += (1.6 / dx) * scnf.delta_kernel(4, xvals)
    C_E_x += (1.6 / dx) * scnf.delta_kernel(6, xvals)
    C_I_x += (1.6 / dx) * scnf.delta_kernel(8, xvals)

    G_E_x = np.cumsum(dx * C_E_x) + g_E_x
    G_E_y = np.cumsum(dyE * C_E_y) + g_E_y
    G_I_x = np.cumsum(dx * C_I_x) + g_I_x
    G_I_y = np.cumsum(dx * C_I_y) + g_I_y

    B_E_x = np.cumsum(dx * G_E_x) + b_E_x
    B_E_y = np.cumsum(dyE * G_E_y) + b_E_y
    B_I_x = np.cumsum(dx * G_I_x) + b_I_x
    B_I_y = np.cumsum(dx * G_I_y) + b_I_y

    x_mesh, yE_mesh = np.meshgrid(xvals, yEvals)
    B_E_x_mesh, B_E_y_mesh = np.meshgrid(B_E_x, B_E_y)
    B_I_x_mesh, B_I_y_mesh = np.meshgrid(B_I_x, B_I_y)

    B_E = B_E_x_mesh + B_E_y_mesh
    B_I = B_I_x_mesh + B_I_y_mesh

    # CODE FROM ABOVE
    X = x_mesh.copy()
    YE = yE_mesh.copy()
    YI_E = B_E.copy()
    YI_I = B_I.copy()

    ########################################################
    # get single neuron parameters and run #################
    ########################################################
    dt = 5e-5
    # leak = 100.
    Tend = 0.25
    times = np.arange(0, Tend, dt)
    nT = len(times)
    x = np.linspace(-2.5, 10, nT)[None, :]
    init_time = 0.05
    init_pd = int(init_time/dt)

    NE = 3
    NI = 3

    f_E = RegularGridInterpolator((xvals, yEvals), B_E.T)
    f_I = RegularGridInterpolator((xvals, yEvals), B_I.T)

    G_E_x_mesh, G_E_y_mesh = np.meshgrid(G_E_x, G_E_y.T)
    G_I_x_mesh, G_I_y_mesh = np.meshgrid(G_I_x, G_I_y.T)
    dfyE_E = RegularGridInterpolator((xvals, yEvals), G_E_y_mesh.T)
    dfx_E = RegularGridInterpolator((xvals, yEvals), G_E_x_mesh.T)
    dfyE_I = RegularGridInterpolator((xvals, yEvals), G_I_y_mesh.T)
    dfx_I = RegularGridInterpolator((xvals, yEvals), G_I_x_mesh.T)

    # define functions:
    yE_range = [5]
    x_range = [0, 10]
    (E_EE, E_EI, F_E, T_E, D_E) = scnf.new_interp_fcn_to_nrns(f_E, dfyE_E, dfx_E, yE_range, x_range,
                                                              N=NE, redundancy=1, D_scale=1.0)

    yE_range = [5]
    x_range = [0, 10]
    (E_IE, E_II, F_I, T_I, D_I) = scnf.new_interp_fcn_to_nrns(f_I, dfyE_I, dfx_I, yE_range, x_range,
                                                              N=NI, redundancy=1, D_scale=1.0)

    # fix decoders
    D_E = prms['D_E']
    D_I = prms['D_I']

    # No single-neuron cost or noise for this figure
    mu_E = 0.0
    sigma_vE = 0.0
    mu_I = 0.0
    sigma_vI = 0.0

    (s_E, s_I, V_E, V_I, g_E, g_I) = scnf.run_EI_spiking_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I,
                                                             T_I,
                                                             dt=dt, mu_E=mu_E, mu_I=mu_I,
                                                             sigma_vE=sigma_vE, sigma_vI=sigma_vI,
                                                             pw_E=prms['pw_E'], pw_I=prms['pw_I'],
                                                             tref_E=prms['tref_E'], tref_I=prms['tref_I'],
                                                             )

    # get readouts and spike times
    r_E = scnf.exp_filter(s_E, times)
    r_I = scnf.exp_filter(s_I, times)
    y_E = D_E @ r_E
    y_I = D_I @ r_I

    # cut out initial pd
    Tend = Tend - init_time
    times = times[:-init_pd]
    x = x[:, init_pd:]
    s_E = s_E[:, init_pd:]
    s_I = s_I[:, init_pd:]
    y_E = y_E[:, init_pd:]
    y_I = y_I[:, init_pd:]

    spk_times_E = scnf.get_spike_times(s_E, dt=dt)
    spk_times_I = scnf.get_spike_times(s_I, dt=dt)

    # latent dynamics over time
    x_trace = np.linspace(0, 10, len(times))
    ids = np.argmin(np.abs(YI_E - YI_I), 0)
    yE_soln_yE = YE[ids, np.arange(YE.shape[1])]
    yI_soln_yI = YI_I[ids, np.arange(YI_I.shape[1])]

    twin = ax4.twinx()
    ax2.plot(times, x_trace, '-', c='black', alpha=0.5, linewidth=0.5)
    mod_times = np.linspace(0, Tend, yE_soln_yE.shape[0])
    ax4.plot(mod_times, yE_soln_yE, '--', c=pu.excitatory_red, label=r'$y^E$ boundary')
    twin.plot(mod_times, yI_soln_yI, '--', c=pu.inhibitory_blue, label=r'$|y^I|$ boundary')
    _ = ax4.plot(times, y_E[0, :], c=pu.excitatory_red, alpha=0.5, linewidth=0.75, label=r'$y^E$ sim.')
    _ = twin.plot(times, y_I[0, :], c=pu.inhibitory_blue, alpha=0.5, linewidth=0.75, label=r'$|y^I|$ sim.')
    _ = ax4.legend(fontsize=pu.fs1, ncols=1, frameon=False,
                   handlelength=2, columnspacing=1, labelspacing=0.15, loc="lower left", bbox_to_anchor=(0.41, -0.09))
    _ = twin.legend(fontsize=pu.fs1, ncols=1, frameon=False,
                    handlelength=2, columnspacing=1, labelspacing=0.15, loc="lower left", bbox_to_anchor=(0.075, 0.6))
    ax1.plot(y_E[0, :], x[0, :], y_I[0, :], c='black', alpha=0.75, linewidth=0.75)

    # spike raster
    for i in range(NE):
        ax3.plot(spk_times_E[i], i * np.ones_like(spk_times_E[i]), '.', color=pu.excitatory_red, markersize=1)
    for i in range(NI):
        ax3.plot(spk_times_I[i], NE + i * np.ones_like(spk_times_I[i]), '.', color=pu.inhibitory_blue,
                 markersize=1)

    for ax in [ax2, ax3, ax4]:
        ax.set_xlim([0.0, Tend])
        ax.set_xticks(np.linspace(0.0, Tend, 6))
        ax.set_xticklabels([])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax2.set_yticks([0, 10])
    ax2.set_yticklabels(['0', '10'], fontsize=pu.fs1)
    ax3.set_yticks([0, NE, NE + NI])
    ax3.set_yticklabels(['0', str(NE), str(NE + NI)], fontsize=pu.fs1)
    ax4.set_yticks([0, 5, 10])
    ax4.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)
    twin.set_yticks([0, 10, 20])
    twin.set_yticklabels(['0', '10', '20'], fontsize=pu.fs1)
    twin.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax4.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, Tend, 6)], fontsize=pu.fs1)
    ax4.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)

    ax2.set_ylabel(r'Input, $x$', fontsize=pu.fs2)
    ax4.set_ylabel(r'Excit. activity, $y^E$', fontsize=pu.fs2)
    twin.set_ylabel(r'Inhib. activity, $|y^I|$', fontsize=pu.fs2)
    ax3.set_ylabel('Neuron\n ID', fontsize=pu.fs2)
    ax4.set_ylim([0, 10])
    twin.set_ylim([0, 20])

    # 3d plot (ax1)
    YI_E[YI_E < YI_I] = np.nan
    YI_I[YI_I < YI_E] = np.nan

    YI_E_A = YI_E.copy()
    YI_E_B = YI_E.copy()
    YI_E_C = YI_E.copy()
    YI_I_A = YI_I.copy()
    YI_I_B = YI_I.copy()
    YI_I_C = YI_I.copy()

    YI_E_A[:, xvals > 2] = np.nan
    YI_E_B[:, xvals < 2] = np.nan
    YI_E_B[:, xvals > 6] = np.nan
    YI_E_C[:, xvals < 6] = np.nan
    YI_I_A[:, xvals > 4] = np.nan
    YI_I_B[:, xvals < 4] = np.nan
    YI_I_B[:, xvals > 8] = np.nan
    YI_I_C[:, xvals < 8] = np.nan

    alphas = [0.25, 0.5, 0.75]
    YI_Is = [YI_I_A, YI_I_B, YI_I_C]
    YI_Es = [YI_E_A, YI_E_B, YI_E_C]

    for i in range(3):
        _ = ax1.plot_surface(YE.T, X.T, YI_Is[i].T, cmap=pu.blue_cmap,
                             linewidth=0, antialiased=True, alpha=alphas[i], vmin=0, vmax=20)
        _ = ax1.plot_surface(YE.T, X.T, YI_Es[i].T, cmap=pu.red_cmap,
                             linewidth=0, antialiased=True, alpha=alphas[i], vmin=0, vmax=20)
    ax1.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.zaxis.set_rotate_label(False)
    ax1.set_box_aspect((10, 10, 8.2), zoom=1.15)
    ax1.view_init(elev=20, azim=240)

    ax1.axes.set_xlim3d(left=-1, right=11)
    ax1.axes.set_ylim3d(bottom=-1, top=13)
    ax1.axes.set_zlim3d(bottom=0, top=24)

    ax1.set_ylabel(r'Input, $x$', fontsize=pu.fs2, labelpad=-12)
    ax1.set_xlabel(r'Excit. activity, $y^E$', fontsize=pu.fs2, labelpad=-12)  # ,linespacing=1.2)
    ax1.set_zlabel(r'Inhib. activity, $|y^I|$', fontsize=pu.fs2, rotation=90, labelpad=-10)
    _ = ax1.set_xticks([0, 5, 10])
    _ = ax1.set_yticks([0, 5, 10])
    _ = ax1.set_zticks([0, 10, 20])
    _ = ax1.set_xticklabels(['0', '5', '10'], fontsize=pu.fs1)
    _ = ax1.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)
    _ = ax1.set_zticklabels(['0', '10', '20'], fontsize=pu.fs1)

    ax1.tick_params(axis='x', which='major', pad=-6)  # , length=50, width=2)
    ax1.tick_params(axis='y', which='major', pad=-6)
    ax1.tick_params(axis='z', which='major', pad=-4)
    ax1.tick_params(axis='both', width=0.5, length=3)  # ,pad=1)

    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    sns.despine(ax=ax4, left=False, right=False)
    sns.despine(ax=twin, left=False, right=False)

    return spk_times_E, spk_times_I


def main(save_pdf=False, show_plot=True):
    f = plt.figure(figsize=(4.1*0.75, 4.5*0.75), dpi=150)
    gs = f.add_gridspec(5, 2, width_ratios=[0.75, 1], height_ratios=[2.5, 0.5, 0.5, 0.5, 0.5])

    ax1a = f.add_subplot(gs[0, 0], projection='3d')
    ax2a = f.add_subplot(gs[1, 0])
    ax3a = f.add_subplot(gs[2, 0])
    ax4a = f.add_subplot(gs[3, 0])
    ax5a = f.add_subplot(gs[4, 0])

    f.tight_layout()

    plot_3d_EI_intuition_plot(ax1a)
    plot_UFA_intro(f, ax2a, ax3a, ax4a, ax5a)

    ax1b = f.add_subplot(gs[0, 1], projection='3d')
    ax2b = f.add_subplot(gs[1, 1])
    ax3b = f.add_subplot(gs[2, 1])
    ax4b = f.add_subplot(gs[3:, 1])

    prms = {'D_E': 0.75 * np.array([[1.05, 1.075, 1.025]]),
            'D_I': 0.75 * np.array([[0.9, 1.5, 2.5]]),
            'pw_E': None,
            'pw_I': None,
            'tref_E': 0.,
            'tref_I': 0.}
    plot_UFA_dynamics(ax1b, ax2b, ax3b, ax4b, prms)

    # adjust subplots and manually position some
    f.subplots_adjust(wspace=0.5, hspace=0.25)

    down_shift = 0.03

    box = ax2a.get_position()
    box.y0 = box.y0 - 0.02 - down_shift
    box.y1 = box.y1 - 0.02 - down_shift
    ax2a.set_position(box)

    box = ax3a.get_position()
    box.y0 = box.y0 - 0.015 - down_shift
    box.y1 = box.y1 - 0.015 - down_shift
    ax3a.set_position(box)

    box = ax4a.get_position()
    box.y0 = box.y0 - 0.005 - down_shift
    box.y1 = box.y1 - 0.005 - down_shift
    ax4a.set_position(box)

    box = ax5a.get_position()
    box.y0 = box.y0 - 0.005 - down_shift
    box.y1 = box.y1 - 0.005 - down_shift
    ax5a.set_position(box)

    box = ax2b.get_position()
    box.y0 = box.y0 - 0.02 - down_shift
    box.y1 = box.y1 - 0.02 - down_shift
    ax2b.set_position(box)

    box = ax3b.get_position()
    box.y0 = box.y0 - 0.015 - down_shift
    box.y1 = box.y1 - 0.015 - down_shift
    ax3b.set_position(box)

    box = ax4b.get_position()
    box.y0 = box.y0 - 0.0 - down_shift
    box.y1 = box.y1 - 0.01 - down_shift
    ax4b.set_position(box)

    # panel labels
    ax1a.text2D(-0.35, 1.1, r'\textbf{a}', transform=ax1a.transAxes, **pu.panel_prms)
    ax1b.text2D(-0.35, 0.95, r'\textbf{c}', transform=ax1b.transAxes, **pu.panel_prms)
    ax2a.text(-0.35, 1.2, r'\textbf{b}', transform=ax2a.transAxes, **pu.panel_prms)
    ax2b.text(-0.35, 1.16, r'\textbf{d}', transform=ax2b.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_05_UFA_EI_net.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
