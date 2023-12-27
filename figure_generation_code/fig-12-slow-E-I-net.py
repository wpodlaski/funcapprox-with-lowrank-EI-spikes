import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf
from scipy.interpolate import RegularGridInterpolator


def plot_UFA_example(ax1, ax2, ax3, prms, legend=False):
    """
    Plots panels (a) and (c), the function approximation (UFA) example, with slow synapses.
    :param ax1: Figure axis for the 3d plot of the boundaries in (x, yE, yI) space
    :param ax2: Figure axis. Spike rasters.
    :param ax3: Figure axis. Input, x, and latent readouts, yE, yI.
    :param prms: Network parameters for the boundaries and simulation.
    :param legend: If True, puts a legend in the bottom plot.
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

    ######################################################################
    # get single neuron parameters and run #################
    ######################################################################
    dt = 1e-4
    Tend = 0.25
    times = np.arange(0, Tend, dt)
    nT = len(times)
    x = np.linspace(-2.5, 10, nT)[None, :]
    init_time = 0.05
    init_pd = int(init_time / dt)

    NE = 3
    NI = 3

    # get numerical approximations to functions
    f_E = RegularGridInterpolator((xvals, yEvals), B_E.T)
    f_I = RegularGridInterpolator((xvals, yEvals), B_I.T)

    G_E_x_mesh, G_E_y_mesh = np.meshgrid(G_E_x, G_E_y)
    G_I_x_mesh, G_I_y_mesh = np.meshgrid(G_I_x, G_I_y)
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

    mu_E = prms['mu_E']
    sigma_vE = 0.01
    mu_I = prms['mu_I']
    sigma_vI = 0.01

    s_E, s_I, V_E, V_I, g_E, g_I = scnf.run_EI_spiking_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I, T_I,
                                                           dt=dt, mu_E=mu_E, mu_I=mu_I,
                                                           sigma_vE=sigma_vE, sigma_vI=sigma_vI,
                                                           pw_E=prms['pw_E'], pw_I=prms['pw_I'],
                                                           tref_E=prms['tref_E'], tref_I=prms['tref_I'])

    # get readouts and spike times
    r_E = scnf.exp_filter(g_E, times)
    r_I = scnf.exp_filter(g_I, times)
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

    twin = ax3.twinx()
    ax3.plot(times, x_trace, '-', c='black', alpha=0.25, linewidth=0.75)
    mod_times = np.linspace(0, Tend, yE_soln_yE.shape[0])
    ax3.plot(mod_times, yE_soln_yE, '--', c=pu.excitatory_red, label=r'$y^E$ boundary')  # ,alpha=0.5)
    twin.plot(mod_times, yI_soln_yI, '--', c=pu.inhibitory_blue, label=r'$|y^I|$ boundary')  # ,alpha=0.5)
    _ = ax3.plot(times, y_E[0, :], c=pu.excitatory_red, alpha=0.5, linewidth=0.75, label=r'$y^E$ sim.')
    _ = twin.plot(times, y_I[0, :], c=pu.inhibitory_blue, alpha=0.5, linewidth=0.75, label=r'$|y^I|$ sim.')
    if legend:
        _ = ax3.legend(fontsize=pu.fs1, ncols=1, frameon=False,
                       handlelength=2, columnspacing=1, labelspacing=0.15, loc="lower left",
                       bbox_to_anchor=(0.41, -0.09))
        _ = twin.legend(fontsize=pu.fs1, ncols=1, frameon=False,
                        handlelength=2, columnspacing=1, labelspacing=0.15, loc="lower left",
                        bbox_to_anchor=(0.075, 0.6))

    # spike raster
    for i in range(NE):
        ax2.plot(spk_times_E[i], i * np.ones_like(spk_times_E[i]), '.', color=pu.excitatory_red, markersize=1)
    for i in range(NI):
        ax2.plot(spk_times_I[i], NE + i * np.ones_like(spk_times_I[i]), '.', color=pu.inhibitory_blue,
                 markersize=1)  # cmap[i])

    for ax in [ax2, ax3]:
        ax.set_xlim([0.0, Tend])
        ax.set_xticks(np.linspace(0.0, Tend, 6))
        ax.set_xticklabels([])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax2.set_yticks([0, NE, NE + NI])
    ax2.set_yticklabels(['0', str(NE), str(NE + NI)], fontsize=pu.fs1)
    ax3.set_yticks([0, 5, 10])
    ax3.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)
    twin.set_yticks([0, 10, 20])
    twin.set_yticklabels(['0', '10', '20'], fontsize=pu.fs1)
    twin.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax3.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, Tend, 6)], fontsize=pu.fs1)
    ax3.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)

    ax3.set_ylabel(r'$x$', fontsize=pu.fs2)
    ax3.set_ylabel(r'$y^E$', fontsize=pu.fs2)
    twin.set_ylabel(r'$|y^I|$', fontsize=pu.fs2)
    ax2.set_ylabel('Neuron ID', fontsize=pu.fs2)
    ax3.set_ylim([0, 10])
    twin.set_ylim([0, 20])

    # 3d plot
    ax1.set_box_aspect((10, 10, 8.2), zoom=1.2)

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

    ax1.plot(y_E[0, :], x[0, :], y_I[0, :], c='black', alpha=0.75, linewidth=0.75)

    pu.set_3d_plot_specs(ax1, transparent_panes=False, juggle=False)
    ax1.axes.set_xlim3d(left=-1, right=11)
    ax1.axes.set_ylim3d(bottom=-1, top=13)
    ax1.axes.set_zlim3d(bottom=0, top=24)
    ax1.set_ylabel(r'$x$', fontsize=pu.fs2)
    ax1.set_xlabel(r'$y^E$', fontsize=pu.fs2)  # ,linespacing=1.2)
    ax1.xaxis.labelpad = -14
    ax1.yaxis.labelpad = -14
    ax1.zaxis.labelpad = -12
    ax1.set_zlabel(r'$|y^I|$', fontsize=pu.fs2, rotation=90)
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

    ax1.view_init(elev=20, azim=240)

    sns.despine(ax=ax2)
    sns.despine(ax=ax3, left=False, right=False)
    sns.despine(ax=twin, left=False, right=False)


def plot_UFA_rate_example(ax1, ax2, ax3, prms):
    """
    Plots panel (b), rate version of the function approximation (UFA) example.
    :param ax1: Figure axis for the 3d plot of the boundaries in (x, yE, yI) space
    :param ax2: Figure axis. Rates over time.
    :param ax3: Figure axis. Input, x, and latent readouts, yE, yI.
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

    ######################################################################
    # get single neuron parameters and run #################
    ######################################################################
    dt = 1e-4
    Tend = 0.2
    times = np.arange(0, Tend, dt)
    nT = len(times)
    x = np.linspace(0, 10, nT)[None, :]

    NE = 3
    NI = 3

    # get numerical approximations to functions
    f_E = RegularGridInterpolator((xvals, yEvals), B_E.T)
    f_I = RegularGridInterpolator((xvals, yEvals), B_I.T)

    G_E_x_mesh, G_E_y_mesh = np.meshgrid(G_E_x, G_E_y)
    G_I_x_mesh, G_I_y_mesh = np.meshgrid(G_I_x, G_I_y)

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

    r_E0 = np.array([0.35, 0.01, 0.01])
    r_I0 = np.array([0.25, 0.01, 0.01])

    r_E, r_I, V_E, V_I = scnf.run_EI_rate_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I, T_I,
                                              dt=dt, beta=prms['beta'], r_E0=r_E0, r_I0=r_I0)
    y_E = D_E @ r_E
    y_I = D_I @ r_I

    # latent dynamics over time
    x_trace = np.linspace(0, 10, len(times))
    ids = np.argmin(np.abs(YI_E - YI_I), 0)
    yE_soln_yE = YE[ids, np.arange(YE.shape[1])]
    yI_soln_yI = YI_I[ids, np.arange(YI_I.shape[1])]

    twin = ax3.twinx()
    ax3.plot(times, x_trace, '-', c='black', alpha=0.25, linewidth=0.75)
    mod_times = np.linspace(0, Tend, yE_soln_yE.shape[0])
    ax3.plot(mod_times, yE_soln_yE, '--', c=pu.excitatory_red)  # ,alpha=0.5)
    twin.plot(mod_times, yI_soln_yI, '--', c=pu.inhibitory_blue)  # ,alpha=0.5)
    _ = ax3.plot(times, y_E[0, :], c=pu.excitatory_red, alpha=0.5, linewidth=0.75)
    _ = twin.plot(times, y_I[0, :], c=pu.inhibitory_blue, alpha=0.5, linewidth=0.75)

    # spike raster
    alphas = [0.25, 0.5, 0.75]
    for i in range(NE):
        ax2.plot(times, r_E[i, :], color=pu.excitatory_red, alpha=alphas[i], linewidth=1)
    for i in range(NI):
        ax2.plot(times, r_I[i, :], color=pu.inhibitory_blue, alpha=alphas[i], linewidth=1)

    for ax in [ax2, ax3]:
        ax.set_xlim([0.0, Tend])
        ax.set_xticks(np.linspace(0.0, Tend, 6))
        ax.set_xticklabels([])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(['0', '0.5', '1.0'], fontsize=pu.fs1)
    ax3.set_yticks([0, 5, 10])
    ax3.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)
    twin.set_yticks([0, 10, 20])
    twin.set_yticklabels(['0', '10', '20'], fontsize=pu.fs1)
    twin.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax3.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, Tend, 6)], fontsize=pu.fs1)
    ax3.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)

    ax3.set_ylabel(r'$x$', fontsize=pu.fs2)
    ax3.set_ylabel(r'$y^E$', fontsize=pu.fs2)
    twin.set_ylabel(r'$|y^I|$', fontsize=pu.fs2)
    ax2.set_ylabel('Rate, $r_F$', fontsize=pu.fs2)
    ax3.set_ylim([0, 10])
    twin.set_ylim([0, 20])

    # 3d plot
    ax1.set_box_aspect((10, 10, 8.2), zoom=1.2)

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

    ax1.plot(y_E[0, :], x[0, :], y_I[0, :], c='black', alpha=0.75, linewidth=0.75)

    pu.set_3d_plot_specs(ax1, transparent_panes=False, juggle=False)

    ax1.axes.set_xlim3d(left=-1, right=11)
    ax1.axes.set_ylim3d(bottom=-1, top=13)
    ax1.axes.set_zlim3d(bottom=0, top=24)

    ax1.set_ylabel(r'$x$', fontsize=pu.fs2)
    ax1.set_xlabel(r'$y^E$', fontsize=pu.fs2)  # ,linespacing=1.2)
    ax1.xaxis.labelpad = -14
    ax1.yaxis.labelpad = -14
    ax1.zaxis.labelpad = -12
    ax1.set_zlabel(r'$|y^I|$', fontsize=pu.fs2, rotation=90)
    _ = ax1.set_xticks([0, 5, 10])
    _ = ax1.set_yticks([0, 5, 10])
    _ = ax1.set_zticks([0, 10, 20])
    _ = ax1.set_xticklabels(['0', '5', '10'], fontsize=pu.fs1)
    _ = ax1.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)
    _ = ax1.set_zticklabels(['0', '10', '20'], fontsize=pu.fs1)

    ax1.tick_params(axis='x', which='major', pad=-6, width=0.5, length=3)  # , length=50, width=2)
    ax1.tick_params(axis='y', which='major', pad=-6, width=0.5, length=3)
    ax1.tick_params(axis='z', which='major', pad=-4, width=0.5, length=3)

    ax1.view_init(elev=20, azim=240)

    sns.despine(ax=ax2)
    sns.despine(ax=ax3, left=False, right=False)
    sns.despine(ax=twin, left=False, right=False)

    return r_E, r_I, V_E, V_I


def main(save_pdf=False, show_plot=True):
    f = plt.figure(figsize=(4.5, 2.5), dpi=150)
    gs = f.add_gridspec(3, 3, height_ratios=[2, 0.5, 1])

    ax1a = f.add_subplot(gs[0, 0], projection='3d')
    ax2a = f.add_subplot(gs[1, 0])
    ax3a = f.add_subplot(gs[2, 0])

    ax1b = f.add_subplot(gs[0, 1], projection='3d')
    ax2b = f.add_subplot(gs[1, 1])
    ax3b = f.add_subplot(gs[2, 1])

    ax1c = f.add_subplot(gs[0, 2], projection='3d')
    ax2c = f.add_subplot(gs[1, 2])
    ax3c = f.add_subplot(gs[2, 2])

    f.tight_layout()

    prms = {'D_E': 0.9 * np.array([[0.9, 1.6, 1.85]]),  # 1.2 for all
            'D_I': 0.9 * np.array([[1., 1.5, 2.5]]),
            'pw_E': 2.5,  # 2.5,
            'pw_I': 1.0,  # 1.0,
            'mu_E': 0.25,
            'mu_I': 0.05,
            'tref_E': 0.0,  # 1.5,
            'tref_I': 0.0}  # 0.25}
    plot_UFA_example(ax1a, ax2a, ax3a, prms, legend=False)
    ax3a.text(0.135, 7.75, 'input, $x$', fontsize=pu.fs1, color='gray', alpha=0.5, rotation=29)

    prms = {'D_E': 15 * np.array([[0.5, 1.2, 1.8]]),
            'D_I': 15 * np.array([[0.5, 1.5, 3.5]]),
            'beta': 10.}
    plot_UFA_rate_example(ax1b, ax2b, ax3b, prms)

    prms = {'D_E': 1.25 * np.array([[1., 1.2, 1.2]]),
            'D_I': np.array([[1., 1.5, 2.5]]),
            'pw_E': 2.5,
            'pw_I': 2.5,
            'mu_E': 0.25,
            'mu_I': 0.05,
            'tref_E': 0.0,  # 1.,
            'tref_I': 0.0}  # 0.5}
    plot_UFA_example(ax1c, ax2c, ax3c, prms)

    f.subplots_adjust(wspace=0.9, hspace=0.3)

    # add panel labels
    ax1a.text2D(-0.35, 1.05, r'\textbf{a}', transform=ax1a.transAxes, **pu.panel_prms)
    ax1b.text2D(-0.35, 1.05, r'\textbf{b}', transform=ax1b.transAxes, **pu.panel_prms)
    ax1c.text2D(-0.35, 1.05, r'\textbf{c}', transform=ax1c.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_12_slow_E_I_UFA.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
