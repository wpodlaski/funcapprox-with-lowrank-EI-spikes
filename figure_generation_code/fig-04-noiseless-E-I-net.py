import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import plot_utils as pu
import net_sim_code as scnf

# boundary parameters for panels a,b,c
a_y_I = -0.5
b_y_I = 0.6
a_x_I = -0.75
b_x_I = 0.
d_I = -0.2
a_y_E = 0.45
b_y_E = 4.0
a_x_E = 0.5
b_x_E = 0.
d_E = 4.0


# inhibitory boundary function: yI = f_I(x, yE)
def yI(yE_, x_, a_y_I_=a_y_I, b_y_I_=b_y_I, a_x_I_=a_x_I, b_x_I_=b_x_I, d_I_=d_I):
    return a_y_I_ * (yE_ - b_y_I_) ** 2 + a_x_I_ * (x_ - b_x_I_) ** 2 + d_I_


# excitatory boundary function: yE = f_E(x, yI)
def yE(yI_, x_, a_y_E_=a_y_E, b_y_E_=b_y_E, a_x_E_=a_x_E, b_x_E_=b_x_E, d_E_=d_E):
    return -a_y_E_ * (yI_ - b_y_E_) ** 2 - a_x_E_ * (x_ - b_x_E_) ** 2 + d_E_


# inverse excitatory boundary function: yI = f_E^{-1}(x, yE)
def yI_yEinv(yE_, x_, a_y_E_=a_y_E, b_y_E_=b_y_E, a_x_E_=a_x_E, b_x_E_=b_x_E, d_E_=d_E):
    return -np.sqrt((yE_ - d_E_ + a_x_E_ * (x_ - b_x_E_) ** 2) / (-a_y_E_)) + b_y_E_


# inverse excitatory boundary function: yI = f_E^{-1}(x, yE)
def yI_yEinv2(yE_, x_, a_y_E_=a_y_E, b_y_E_=b_y_E, a_x_E_=a_x_E, b_x_E_=b_x_E, d_E_=d_E):
    return -1 * yI_yEinv(yE_, x_, a_y_E_=a_y_E_, b_y_E_=b_y_E_, a_x_E_=a_x_E_, b_x_E_=b_x_E_, d_E_=d_E_)


# inverse inhibitory boundary function: yE = f_I^{-1}(x, yI)
def yE_yIinv(yI_, x_, a_y_I_=a_y_I, b_y_I_=b_y_I, a_x_I_=a_x_I, b_x_I_=b_x_I, d_I_=d_I):
    return np.sqrt((yI_ - np.abs(d_I_) - np.abs(a_x_I_) * (x_ - b_x_I_) ** 2) / np.abs(a_y_I_)) + b_y_I_


def plot_3d_inhibitory_boundary(ax):
    """
    Plots panel (a), the 3d inhibitory boundary yI = f_I(x, yE).
    :param ax: Figure axis.
    :return:
    """

    # plot curved 2d surface in the 3d plot
    x_vals = np.linspace(-2, 2, 201)
    yE_vals = np.linspace(-0.6, 3.4, 201)
    X, YE = np.meshgrid(x_vals, yE_vals)
    Z_I = yI(YE, X)
    Z_I[Z_I < -8.0] = np.nan
    ax.set_box_aspect((20, 25, 25))
    _ = ax.plot_surface(X, YE, Z_I, cmap=pu.blue_cmap, linewidth=0, antialiased=True, alpha=0.5, vmin=-6.0, vmax=0)

    # plot 1d slice for fixed x with decoder arrows
    x = 0.8
    idx = np.where(np.abs(X[0, :]-x) < 1e-6)[0][0]
    ax.plot(X[:, idx], YE[:, idx], Z_I[:, idx], c=pu.inhibitory_blue, alpha=1., linewidth=1.)
    arr_ids = [10, 60, 110, 160]
    for i in range(len(arr_ids)):
        arrw = pu.Arrow3D([X[arr_ids[i], idx], X[arr_ids[i], idx]], [YE[arr_ids[i], idx], YE[arr_ids[i], idx]],
                          [Z_I[arr_ids[i], idx] + 0.2, Z_I[arr_ids[i], idx] - 1.4], lw=1, linewidth=0.1,
                          arrowstyle="-|>,head_length=1.5,head_width=0.75", color=pu.inhibitory_blue)
        ax.add_artist(arrw)

    # formatting
    pu.set_3d_plot_specs(ax)
    ax.axes.set_xlim3d(left=-11./5, right=11./5)
    ax.axes.set_ylim3d(bottom=-5./5, top=16./5)
    ax.axes.set_zlim3d(bottom=-33./5, top=0)
    _ = ax.set_xticks([-10/5, 0, 10/5])
    _ = ax.set_yticks([0, 10/5])
    _ = ax.set_zticks([-30/5, -20/5, -10/5, 0])
    _ = ax.set_xticklabels(['-2', '0', '2'], fontsize=pu.fs1)
    _ = ax.set_yticklabels(['0', '2'], fontsize=pu.fs1)
    _ = ax.set_zticklabels(['-6', '-4', '-2', '0'], fontsize=pu.fs1)
    ax.set_xlabel(r'Input, $x$', fontsize=pu.fs2, labelpad=-12)
    ax.set_ylabel('Excitatory\n latent var., ' + r'$y^E$', fontsize=pu.fs2, labelpad=-10)  # ,linespacing=1.2)
    ax.set_zlabel('Inhibitory\n latent var., ' + r'$y^I$', fontsize=pu.fs2, rotation=90, labelpad=-10)
    ax.tick_params(axis='x', which='major', pad=-6, width=0.5, length=3)
    ax.tick_params(axis='y', which='major', pad=-6, width=0.5, length=3)
    ax.tick_params(axis='z', which='major', pad=-4, width=0.5, length=3)
    ax.view_init(elev=10, azim=320)


def plot_3d_excitatory_boundary(ax):
    """
    Plots panel (b), the 3d excitatory boundary yE = f_E(x, yI).
    :param ax: Figure axis.
    :return:
    """

    # plot curved 2d surface in the 3d plot
    x_vals = np.linspace(-2, 2, 101)
    yI_vals = np.linspace(1, 6, 101)[::-1]
    yI_vals2 = np.linspace(-6, -1, 101)
    X, YI = np.meshgrid(x_vals, yI_vals)
    X2, YI2 = np.meshgrid(x_vals, yI_vals2)
    Z_E = yE(YI, X)
    ax.set_box_aspect((20, 25, 25))
    _ = ax.plot_surface(X, YI2, Z_E, cmap=pu.red_cmap,
                        linewidth=0, antialiased=True, alpha=0.5, vmin=0, vmax=6.0)

    # plot 1d slice for fixed x with decoder arrows
    x = 0.8
    idx2 = np.where(np.abs(X[0, :] - x) < 1e-6)[0][0]
    ax.plot(X[:, idx2], YI2[:, idx2], Z_E[:, idx2], c=pu.excitatory_red, alpha=1., linewidth=1.)
    arr_ids = [10, 30, 50, 70]
    for i in range(len(arr_ids)):
        arrw = pu.Arrow3D([X[arr_ids[i], idx2], X[arr_ids[i], idx2]], [YI2[arr_ids[i], idx2], YI2[arr_ids[i], idx2]],
                          [Z_E[arr_ids[i], idx2] - 0.2, Z_E[arr_ids[i], idx2] + 1.4], lw=1, linewidth=0.1,
                          arrowstyle="-|>,head_length=1.5,head_width=0.75", color=pu.excitatory_red)
        ax.add_artist(arrw)

    # formatting
    pu.set_3d_plot_specs(ax)
    ax.axes.set_xlim3d(left=-11./5, right=11./5)
    ax.axes.set_ylim3d(bottom=-30./5, top=2./5)
    ax.axes.set_zlim3d(bottom=-9./5, top=22./5)
    _ = ax.set_xticks([-2, 0, 2])
    _ = ax.set_yticks([-4, 0])
    _ = ax.set_zticks([0, 2, 4])
    _ = ax.set_xticklabels(['-2', '0', '2'], fontsize=pu.fs1)
    _ = ax.set_yticklabels(['-4', '0'], fontsize=pu.fs1)
    _ = ax.set_zticklabels(['0', '2', '4'], fontsize=pu.fs1)
    ax.set_xlabel(r'Input, $x$', fontsize=pu.fs2, labelpad=-12)
    ax.set_ylabel('Inhibitory\n latent var., ' + r'$y^I$', fontsize=pu.fs2, labelpad=-10)
    ax.set_zlabel('Excitatory\n latent var., ' + r'$y^E$', fontsize=pu.fs2, rotation=90, labelpad=-10)
    ax.tick_params(axis='x', which='major', pad=-6, width=0.5, length=3)
    ax.tick_params(axis='y', which='major', pad=-6, width=0.5, length=3)
    ax.tick_params(axis='z', which='major', pad=-4, width=0.5, length=3)
    ax.view_init(elev=10, azim=320)


def plot_2d_EI_slice_boundary(ax):
    """
    Plots panel (c), the E and I boundaries for fixed x in (yE, yI) space.
    :param ax: Figure axis.
    :return:
    """

    x = 0.8
    yE_1 = np.linspace(-1., 0.6, 51)
    yE_2 = np.linspace(0.6, 4., 201)
    yI_1 = np.linspace(0.8, 1.15, 51)
    yI_2 = np.linspace(1.15, 4., 131)
    yI_3 = np.linspace(4., 5.4, 101)

    # the numerical crossing point
    yI_soln = 2.34
    yI_soln2 = 4.72

    #
    yI_fb1 = np.linspace(1, yI_soln, 101)
    yI_fb2 = np.linspace(yI_soln, yI_soln2, 101)
    yE_fb5 = np.linspace(0, 2.34, 101)
    yE_fb6 = np.linspace(2.34, 4.6, 101)

    # fill in the different areas of the plot
    ax.fill_between(yE(yI_fb1, x), np.abs(yI(yE(yI_fb1, x), x)), yI_fb1, color=pu.excitatory_red,
                    alpha=0.25, edgecolor='white')
    ax.fill_between(yE(yI_fb2, x), np.abs(yI(yE(yI_fb2, x), x)), yI_fb2, color=pu.inhibitory_blue,
                    alpha=0.25, edgecolor='white')
    ax.fill_between(np.concatenate((yE_fb5, yE_fb6)),
                    np.concatenate((yI_yEinv(yE_fb5, x), np.abs(yI(yE_fb6, x)))),
                    7.0 * np.ones_like(np.concatenate((yE_fb5, yE_fb6))),
                    color='gray', alpha=0.1, edgecolor='white')

    # delineate the suprathreshold area to fill in purple
    x0 = np.linspace(0, 5.0, 101)
    y0 = np.zeros_like(x0)
    y1 = np.linspace(0, 7.0, 101)
    x1 = 5.0 * np.ones_like(y1)
    y3 = np.linspace(2.34, 7.0, 101)
    x3 = np.maximum(yE(np.linspace(2.34, 7.0, 101), x), yE_yIinv(np.linspace(2.34, 7.0, 101), x))
    y3 = y3[::-1]
    x3 = x3[::-1]
    x2 = np.linspace(x3[0], 5.0, 21)
    y2 = 7.0 * np.ones_like(x2)
    x2 = x2[::-1]
    y2 = y2[::-1]
    x4 = np.linspace(0, 5.0, 101)
    y4 = np.minimum(2.356 * np.ones((101,)), np.abs(yI(np.linspace(0, 5.0, 101), x)))
    y4 = y4[x4 <= 2.4]
    x4 = x4[x4 <= 2.4]
    x4 = x4[::-1]
    y4 = y4[::-1]
    y5 = np.linspace(y4[-1], 0, 11)
    x5 = np.zeros_like(y5)
    xs = np.concatenate((x0, x1, x2, x3, x4, x5))
    ys = np.concatenate((y0, y1, y2, y3, y4, y5))
    ax.fill_between(xs, ys, color='rebeccapurple', alpha=0.25, edgecolor='w')

    # plot the boundaries themselves
    ax.plot(yE_2, np.abs(yI(yE_2, x)), '-', c=pu.inhibitory_blue, linewidth=1)
    ax.plot(yE(yI_2, x), yI_2, '-', c=pu.excitatory_red, linewidth=1)
    ax.plot(yE_1, np.abs(yI(yE_1, x)), ':', c=pu.inhibitory_blue)
    ax.plot(yE(yI_1, x), yI_1, ':', c=pu.excitatory_red)
    ax.plot(yE(yI_3, x), yI_3, ':', c=pu.excitatory_red)

    # arrows along the two slices of the boundaries
    exc_arrows = [1.2, 2.0, 3.4]
    for j in range(len(exc_arrows)):
        ax.arrow(yE(exc_arrows[j], x), exc_arrows[j], 0.4, 0, color=pu.excitatory_red,
                 width=0.02, head_length=0.1, head_width=0.2, linewidth=0.5, zorder=10)
    inh_arrows = [1.4, 2.8, 3.6]
    for j in range(len(inh_arrows)):
        ax.arrow(inh_arrows[j], np.abs(yI(inh_arrows[j], x)), 0, 0.5, color=pu.inhibitory_blue,
                 width=0.02, head_length=0.15, head_width=0.15, linewidth=0.5, zorder=10)

    # formatting
    ax.set_ylabel(r'Inhibitory activity, $|y^I|$', fontsize=pu.fs2)
    ax.set_xlabel(r'Excitatory activity, $y^E$', fontsize=pu.fs2)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(['0', '2', '4'], fontsize=pu.fs1)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(['0', '2', '4', '6'], fontsize=pu.fs1)
    ax.set_xlim([0, 4.4])
    ax.set_ylim([0, 6.4])
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax.text(0.6, 1.8, r'\textbf{E}', fontsize=pu.fs2, color=pu.excitatory_red)
    ax.text(3.0, 4.2, r'\textbf{I}', fontsize=pu.fs2, color=pu.inhibitory_blue)
    ax.text(0.2, 5.8, r'\textbf{subthreshold}', fontweight='bold', fontsize=pu.fs1)
    ax.text(2.0, 0.2, r'\textbf{suprathreshold}', color='rebeccapurple', fontweight='bold', fontsize=pu.fs1)


def plot_EI_net_results(axs, prms):
    """
    Plots the E and I boundaries and simulation first for panels d,e and then for panel f.
    :param axs: Figure axes.
    :param prms: Network boundary parameters.
    :return:
    """

    # network parameters
    NE = prms['NE']  # 5
    NI = prms['NI']  # 5

    def Efunc(x__):
        return prms['aE'] * scnf.rect(x__) * scnf.rect(x__) + prms['bE'] * scnf.rect(x__) + prms['cE']

    def Edfunc(x__):
        return 2 * prms['aE'] * scnf.rect(x__) + prms['bE']

    def Ifunc(x__):
        return prms['aI'] * scnf.rect(x__) * scnf.rect(x__) + prms['bI'] * scnf.rect(x__) + prms['cI']

    def Idfunc(x__):
        return 2 * prms['aI'] * scnf.rect(x__) + prms['bI']

    # E and I parameters
    (D_E, E_EI, E_EE, FT_E, xvals, yvals) = scnf.old_fcn_to_nrns(Efunc, Edfunc, NE, prms['x_range'])
    (D_I, E_II, E_IE, FT_I, xvals, yvals) = scnf.old_fcn_to_nrns(Ifunc, Idfunc, NI, prms['x_range'])
    D_E *= prms['D_E']
    D_I *= prms['D_I']
    T_E = np.ones((NE,))
    T_I = np.ones((NI,))

    if prms['x'] == 0:
        F_E = np.zeros((NE,))
        F_I = np.zeros((NI,))
        T_E = FT_E
        T_I = FT_I
    else:
        F_E = (T_E - FT_E) / prms['x']
        F_I = (T_I - FT_I) / prms['x']

    # fix size issues
    D_E = D_E[None, :]
    E_EI = E_EI[:, None]
    E_EE = E_EE[:, None]
    F_E = F_E[:, None]
    D_I = D_I[None, :]
    E_II = E_II[:, None]
    E_IE = E_IE[:, None]
    F_I = F_I[:, None]

    # additional parameters
    sigma_vE = prms.get('sigma_vE', 0.)
    sigma_vI = prms.get('sigma_vI', 0.)
    mu_E = prms.get('mu_E', 0.)
    mu_I = prms.get('mu_I', 0.)

    # input & simulation parameters
    dt = 5e-5
    Tend = 0.06
    times = np.arange(0, Tend, dt)
    nT = len(times)
    x = prms['x'] * np.ones((nT,))[None, :]
    x[0, :100] = 0

    # inhibitory perturbation
    I_E = np.zeros((NE, nT))
    I_I = np.zeros((NI, nT))

    # run simulation
    (s_E, s_I, V_E, V_I, g_E, g_I) = scnf.run_EI_spiking_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I, T_I,
                                                             sigma_vE=sigma_vE, sigma_vI=sigma_vI,
                                                             mu_E=mu_E, mu_I=mu_I, dt=dt, I_I=I_I, I_E=I_E)

    # get readouts and spike times
    r_E = scnf.exp_filter(s_E, times)
    r_I = scnf.exp_filter(s_I, times)
    y_E = D_E @ r_E
    y_I = D_I @ r_I
    spk_times_E = scnf.get_spike_times(s_E, dt=dt)
    spk_times_I = scnf.get_spike_times(s_I, dt=dt)

    # calculate fixed point from equations
    (yE_sol1, yE_sol2) = scnf.solve_quad(prms['aI'] - prms['aE'], prms['bI'] - prms['bE'], prms['cI'] - prms['cE'])
    fixpt_yE = yE_sol1
    fixpt_yI = Efunc(np.array([fixpt_yE]))[0]

    if axs[0] is not None:
        xvals = np.linspace(prms['xlim'][0], prms['xlim'][1], 101)
        yEvals = Efunc(xvals)
        yIvals = Ifunc(xvals)
        axs[0].plot(xvals, yEvals, c=pu.excitatory_red, linewidth=1, alpha=0.75)
        axs[0].plot(xvals, yIvals, c=pu.inhibitory_blue, linewidth=1, alpha=0.75)

        for i in range(NE):
            (x_, y_) = scnf.old_boundary(prms['x_range'], E_EI[i, 0], E_EE[i, 0], T_E[i])
            inds = np.where(y_ >= 0)[0]
            axs[0].plot(x_[inds], y_[inds], linewidth=1, c=pu.excitatory_red, alpha=0.25)
        for i in range(NI):
            (x_, y_) = scnf.old_boundary(prms['x_range'], E_II[i, 0], E_IE[i, 0], T_I[i])
            inds = np.where(y_ >= 0)[0]
            axs[0].plot(x_[inds], y_[inds], linewidth=1, c=pu.inhibitory_blue, alpha=0.25)

        # plot dynamics
        axs[0].plot(y_E[0, 1000:1500], y_I[0, 1000:1500], c='black', alpha=1, linewidth=1, label='spiking sim.')
        axs[0].plot([fixpt_yE], [fixpt_yI], '.', c='black', markersize=3, alpha=1, label='boundary\n intersection')
        if prms['boundary_labels']:
            axs[0].legend(fontsize=pu.fs1, ncols=1, frameon=False, handlelength=1.5, columnspacing=0.75,
                          loc="lower left", bbox_to_anchor=(0.35, -0.05))

        # plot three triangles over the dynamics indicating direction
        yE_min = np.min(y_E[0, 1000:1500])
        yE_max = np.max(y_E[0, 1000:1500])
        yE_mid = 0.5 * yE_min + 0.5 * yE_max
        yI_min = np.min(y_I[0, 1000:1500])
        yI_mid = 0.5 * yI_min + 0.5 * np.max(y_I[0, 1000:1500])
        style = "Simple, tail_width=0.5, head_width=3, head_length=2.5"
        kw = dict(arrowstyle=style, color="black", alpha=1, linewidth=0.1, clip_on=True)
        a2 = patches.FancyArrowPatch((yE_mid, yI_min), (yE_mid + 0.25, yI_min), **kw, zorder=10)  # horizontal
        axs[0].add_patch(a2)
        a1 = patches.FancyArrowPatch((yE_max, yI_mid - 0.25), (yE_max, yI_mid + 0.25), **kw, zorder=10)  # vertical
        axs[0].add_patch(a1)
        a3 = patches.FancyArrowPatch((yE_mid + 0.0625, yI_mid + 0.125), (yE_mid - 0.0625, yI_mid - 0.125),
                                     **kw, zorder=10)  # diagonal
        axs[0].add_patch(a3)

        # add general trajectory arrows
        if prms['trajectory_arrows']:
            kw = dict(alpha=0.25, linewidth=0.5, head_width=0.15, head_length=0.15, color='black')
            axs[0].arrow(1.6, 3.75, -0.1, -0.1 * 3 / 2, **kw)
            axs[0].arrow(0.35, 1.25, 0.1, -0.1 * 3 / 2, **kw)
            axs[0].arrow(2.25, 2, 0.1, 0.1 * 3 / 2, **kw)
            axs[0].arrow(3.5, 4.6, -0.1, 0.1 * 3 / 2, **kw)
        if prms['boundary_labels']:
            axs[0].text(0.5, 2.0, r'\textbf{E}', fontsize=pu.fs2, color=pu.excitatory_red)
            axs[0].text(3.5, 5.5, r'\textbf{I}', fontsize=pu.fs2, color=pu.inhibitory_blue)

        axs[0].set_xlabel(r'Excitatory activity, $y^E$', fontsize=pu.fs2)
        axs[0].set_ylabel(r'Inhibitory activity, $|y^I|$', fontsize=pu.fs2)
        axs[0].set_xticks(prms['xticks'])
        axs[0].set_yticks(prms['yticks'])
        axs[0].set_xticklabels([str(s) for s in prms['xticks']], fontsize=pu.fs1)
        axs[0].set_yticklabels([str(s) for s in prms['yticks']], fontsize=pu.fs1)

        axs[0].tick_params(axis='both', width=0.5, length=3, pad=1)
        axs[0].set_xlim(prms['xlim'])
        axs[0].set_ylim(prms['ylim'])

    # latent dynamics over time
    if axs[1] is not None:
        _ = axs[1].plot(times, -y_I[0, :], c=pu.inhibitory_blue, linewidth=0.5, alpha=0.5, label='sim.')
        fixpt_I_dyn = np.zeros_like(x)
        fixpt_I_dyn[0, x[0, :] == 0] = 0
        fixpt_I_dyn[0, x[0, :] > 0] = -fixpt_yI
        axs[1].plot(times, fixpt_I_dyn[0, :], linestyle='--', c=pu.inhibitory_blue, alpha=1.0, linewidth=1,
                    label='boundary')
        axs[1].legend(fontsize=pu.fs1, ncols=2, frameon=False, handlelength=1.5, columnspacing=0.75,
                      loc="lower left", bbox_to_anchor=(0.15, 0.3))
        for i in range(NI):
            axs[1].plot(spk_times_I[i], 0.75 * np.ones_like(spk_times_I[i]), '.', color=pu.inhibitory_blue,
                        markersize=1)  # cmap[i])
        axs[1].set_ylabel(r'$y^I$', fontsize=pu.fs2)
        axs[1].set_yticks([-4, -2, 0])
        axs[1].set_xticks([0, Tend / 2, Tend])
        axs[1].set_yticklabels(['-4', '-2', '0'], fontsize=pu.fs1)
        axs[1].set_xticklabels([])
        axs[1].tick_params(axis='both', width=0.5, length=3, pad=1)
        axs[1].set_xlim([0, Tend])
    if axs[2] is not None:
        _ = axs[2].plot(times, y_E[0, :], c=pu.excitatory_red, linewidth=0.5, alpha=0.5, label='sim.')
        fixpt_E_dyn = np.zeros_like(x)
        fixpt_E_dyn[0, x[0, :] == 0] = 0
        fixpt_E_dyn[0, x[0, :] > 0] = fixpt_yE
        axs[2].plot(times, fixpt_E_dyn[0, :], linestyle='--', c=pu.excitatory_red, alpha=1.0, linewidth=1,
                    label='boundary')
        axs[2].legend(fontsize=pu.fs1, ncols=2, frameon=False, handlelength=1.5, columnspacing=0.75,
                      loc="lower left", bbox_to_anchor=(0.15, 0.))
        for i in range(NE):
            axs[2].plot(spk_times_E[i], 2.8 * np.ones_like(spk_times_E[i]), '.', color=pu.excitatory_red,
                        markersize=1)
        axs[2].set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
        axs[2].set_ylabel(r'$y^E$', fontsize=pu.fs2)
        axs[2].set_yticks([0, 1, 2])
        axs[2].set_xticks([0., Tend / 2, Tend])
        axs[2].set_yticklabels(['0', '1', '2'], fontsize=pu.fs1)
        axs[2].set_xticklabels(['%.0f' % (100. * x) for x in [0, Tend / 2, Tend]], fontsize=pu.fs1)
        axs[2].tick_params(axis='both', width=0.5, length=3, pad=1)
        axs[2].set_xlim([0, Tend])

    # choose one neuron and plot synaptic input and voltage
    n1 = 0

    if axs[3] is not None:
        # synaptic input
        Vpos = F_E[n1, :] @ x + E_EE[n1, :] @ y_E
        Vneg = -E_EI[n1, :] @ y_I
        Vsum = Vpos + Vneg
        axs[3].plot(times, Vpos, color=pu.excitatory_red, linewidth=0.75)
        axs[3].plot(times, Vneg, color=pu.inhibitory_blue, linewidth=0.75)
        axs[3].plot(times, Vsum, color='gray', linewidth=0.75)

    if axs[4] is not None:
        # voltage
        V1 = V_E[n1, :]
        V1[s_E[n1, :].astype(bool)] = 3.
        axs[4].plot(times, V_E[n1, :], color=pu.excitatory_red, linewidth=0.5)
        axs[4].axhline(y=1, linestyle='--', c='black', alpha=0.5, linewidth=1)
        axs[4].axhline(y=0, linestyle='-', c='gray', alpha=0.5, linewidth=1)


def draw_I_diagram(ax):
    """
    Draws the diagram at the top of panel (a)
    :param ax: Figure axis.
    :return:
    """

    style = "Simple, tail_width=0.5, head_width=2.5, head_length=2.5"
    kw = dict(arrowstyle=style, color="k", linewidth=0.1, clip_on=False)
    ax.set_axis_off()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # INHIB ONLY & EXC ONLY
    centX = [0.33, 0.27]
    centY = [-0.5, -0.5]
    circ_clrs = [pu.inhibitory_blue, pu.excitatory_red]
    x2_strs = [r'$y^E$', r'$y^I$']
    out_strs = [r'$y^I$', r'$y^E$']
    in_space = [0, 0.02]
    i = 0
    c = patches.Ellipse((centX[i], centY[i]), width=0.1, height=1., angle=0, color=circ_clrs[i], alpha=0.5,
                        clip_on=False)
    ax.add_patch(c)
    arr = patches.FancyArrowPatch((centX[i] - 0.2, centY[i] + 0.45), (centX[i] - 0.06, centY[i] + 0.2), **kw)
    ax.add_patch(arr)
    arr = patches.FancyArrowPatch((centX[i] - 0.2, centY[i] - 0.5), (centX[i] - 0.06, centY[i] - 0.2), **kw)
    ax.add_patch(arr)
    arr = patches.FancyArrowPatch((centX[i] + 0.06, centY[i]), (centX[i] + 0.2, centY[i]), **kw)
    ax.add_patch(arr)
    pu.drawCirc(ax, 0.08, 0.8, centX[i], centY[i] + 0.95, 0, 275, 195, 195, color_='black', linewidth_=0.5,
                headwidth_=0.2, headlength_=0.42, xscale=0.1)
    ax.text(centX[i] - 0.26, centY[i] + 0.45, r'$x$', color='black', fontsize=pu.fs1)
    ax.text(centX[i] - 0.31 + in_space[i], centY[i] - 0.5, x2_strs[i], color='black', fontsize=pu.fs1)
    ax.text(centX[i] + 0.19, centY[i], out_strs[i], color='black', fontsize=pu.fs1)
    ax.text(centX[i] + 0.05, centY[i] + 1.45, out_strs[i], color='black', fontsize=pu.fs1)


def draw_E_diagram(ax):
    """
    Draws the diagram at the top of panel (b)
    :param ax: Figure axis.
    :return:
    """

    style = "Simple, tail_width=0.5, head_width=2.5, head_length=2.5"
    kw = dict(arrowstyle=style, color="k", linewidth=0.1, clip_on=False)
    ax.set_axis_off()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # INHIB ONLY & EXC ONLY
    centX = [0.33, 0.27]
    centY = [-0.5, -0.5]
    circ_clrs = [pu.inhibitory_blue, pu.excitatory_red]
    x2_strs = [r'$y^E$', r'$y^I$']
    out_strs = [r'$y^I$', r'$y^E$']
    in_space = [0, 0.02]
    i = 1
    c = patches.Ellipse((centX[i], centY[i]), width=0.1, height=1., angle=0, color=circ_clrs[i], alpha=0.5,
                        clip_on=False)
    ax.add_patch(c)
    arr = patches.FancyArrowPatch((centX[i] - 0.2, centY[i] + 0.45), (centX[i] - 0.06, centY[i] + 0.2), **kw)
    ax.add_patch(arr)
    arr = patches.FancyArrowPatch((centX[i] - 0.2, centY[i] - 0.5), (centX[i] - 0.06, centY[i] - 0.2), **kw)
    ax.add_patch(arr)
    arr = patches.FancyArrowPatch((centX[i] + 0.06, centY[i]), (centX[i] + 0.2, centY[i]), **kw)
    ax.add_patch(arr)
    pu.drawCirc(ax, 0.08, 0.8, centX[i], centY[i] + 0.95, 0, 275, 195, 195, color_='black', linewidth_=0.5,
                headwidth_=0.2, headlength_=0.42, xscale=0.1)
    ax.text(centX[i] - 0.26, centY[i] + 0.45, r'$x$', color='black', fontsize=pu.fs1)
    ax.text(centX[i] - 0.31 + in_space[i], centY[i] - 0.5, x2_strs[i], color='black', fontsize=pu.fs1)
    ax.text(centX[i] + 0.19, centY[i], out_strs[i], color='black', fontsize=pu.fs1)
    ax.text(centX[i] + 0.05, centY[i] + 1.45, out_strs[i], color='black', fontsize=pu.fs1)


def draw_EI_diagram(ax):
    """
    Draws the diagram at the top of panel (c)
    :param ax: Figure axis.
    :return:
    """

    style = "Simple, tail_width=0.5, head_width=2.5, head_length=2.5"
    kw = dict(arrowstyle=style, color="k", linewidth=0.1, clip_on=False)
    ax.set_axis_off()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # EI TOGETHER
    centX = 0.32
    centY = -0.8
    c1 = patches.Ellipse((centX, centY), width=0.1, height=1., angle=0, color=pu.excitatory_red, alpha=0.5,
                         clip_on=False)
    c2 = patches.Ellipse((centX + 0.28, centY), width=0.1, height=1., angle=0, color=pu.inhibitory_blue, alpha=0.5,
                         clip_on=False)
    ax.add_patch(c1)
    ax.add_patch(c2)
    a1 = patches.FancyArrowPatch((centX + 0.055, centY + 0.05), (centX + 0.23, centY + 0.05),
                                 connectionstyle="arc3,rad=-0.5", **kw)
    a2 = patches.FancyArrowPatch((centX + 0.23, centY - 0.05), (centX + 0.055, centY - 0.05),
                                 connectionstyle="arc3,rad=-0.5", **kw)
    ax.add_patch(a1)
    ax.add_patch(a2)
    pu.drawCirc(ax, 0.08, 0.8, centX - 0.09, centY, 0, 80, 270, 280, color_='black', linewidth_=0.5, headwidth_=0.2,
                headlength_=0.42, xscale=0.1)
    pu.drawCirc(ax, 0.08, 0.8, centX + 0.38, centY, 0, 260, 100, 120, color_='black', linewidth_=0.5, headwidth_=0.2,
                headlength_=0.42, xscale=0.1)
    a3 = patches.FancyArrowPatch((centX, centY + 0.45), (centX, centY + 1.75), connectionstyle="arc3,rad=0", **kw)
    a4 = patches.FancyArrowPatch((centX + 0.28, centY + 0.45), (centX + 0.28, centY + 1.75),
                                 connectionstyle="arc3,rad=0", **kw)
    ax.add_patch(a3)
    ax.add_patch(a4)
    ax.text(centX - 0.04, centY + 1.75, r'$y^E$', color='black', fontsize=pu.fs1)
    ax.text(centX + 0.24, centY + 1.75, r'$y^I$', color='black', fontsize=pu.fs1)


def main(save_pdf=False, show_plot=True):
    f = plt.figure(figsize=(4.95, 4.85), dpi=150)
    gs = f.add_gridspec(14, 14)

    # rest of the panels
    ax1 = f.add_subplot(gs[1:6, :5], projection='3d')
    ax2 = f.add_subplot(gs[1:6, 4:9], projection='3d')
    ax3 = f.add_subplot(gs[1:5, 10:])
    ax4 = f.add_subplot(gs[6:10, :4])
    ax5a = f.add_subplot(gs[6:8, 5:9])
    ax5b = f.add_subplot(gs[8:10, 5:9])
    ax6 = f.add_subplot(gs[6:10, 10:])

    # top three panels, for network schematic diagrams
    ax0_1 = f.add_subplot(gs[0, :4])
    ax0_2 = f.add_subplot(gs[0, 5:9])
    ax0_3 = f.add_subplot(gs[0, 10:])
    axs0 = [ax0_1, ax0_2, ax0_3]

    # EI boundary in panel c
    plot_2d_EI_slice_boundary(ax3)

    # Homogeneous (two-neuron) EI boundary and results in panels d,e
    prms = {'NE': 1, 'NI': 1, 'D_E': 0.62, 'D_I': 1.0,
            'aE': 0., 'bE': 1., 'cE': 1., 'aI': 0., 'bI': 1.5, 'cI': 0.,
            'x_range': [2, 2], 'x': 1, 'xlim': [0, 4], 'ylim': [0, 6],
            'xticks': [0, 2, 4], 'yticks': [0, 2, 4, 6],
            'trajectory_arrows': True, 'boundary_labels': True}
    plot_EI_net_results([ax4, ax5a, ax5b, None, None], prms)

    # Heterogeneous EI boundary in panel f
    prms = {'NE': 10, 'NI': 10, 'D_E': 2.45/2, 'D_I': 3.6/2,
            'aE': 2 * 0.075, 'bE': 0., 'cE': 5./2, 'aI': 2 * 0.12, 'bI': 0., 'cI': 3.5/2,
            'x_range': [0, 5], 'x': 0, 'xlim': [0, 5], 'ylim': [0, 7.5],
            'xticks': [0, 2, 4], 'yticks': [0, 2, 4, 6],
            'trajectory_arrows': False, 'boundary_labels': False}
    plot_EI_net_results([ax6, None, None, None, None], prms)

    # I and E 3d boundaries in panels a,b (do this last, it works better)
    plot_3d_inhibitory_boundary(ax1)
    plot_3d_excitatory_boundary(ax2)

    # draw the three network diagrams at the top of panels a,b,c
    draw_I_diagram(axs0[0])
    draw_E_diagram(axs0[1])
    draw_EI_diagram(axs0[2])

    # subplot formatting
    f.subplots_adjust(wspace=2, hspace=2)
    sns.despine()

    box = ax1.get_position()
    box.x0 = box.x0 - 0.03
    box.x1 = box.x1 - 0.05
    box.y0 = box.y0 + 0.04
    box.y1 = box.y1 + 0.04
    ax1.set_position(box)

    box = ax2.get_position()
    box.x0 = box.x0 + 0.02
    box.x1 = box.x1 + 0.02
    box.y0 = box.y0 + 0.04
    box.y1 = box.y1 + 0.04
    ax2.set_position(box)

    # panel labels
    axs0[0].text(-0.4, 1.2, r'\textbf{a}', transform=axs0[0].transAxes, **pu.panel_prms)
    axs0[1].text(-0.4, 1.2, r'\textbf{b}', transform=axs0[1].transAxes, **pu.panel_prms)
    axs0[2].text(-0.4, 1.2, r'\textbf{c}', transform=axs0[2].transAxes, **pu.panel_prms)
    ax4.text(-0.4, 1.075, r'\textbf{d}', transform=ax4.transAxes, **pu.panel_prms)
    ax5a.text(-0.4, 1.19, r'\textbf{e}', transform=ax5a.transAxes, **pu.panel_prms)
    ax6.text(-0.4, 1.075, r'\textbf{f}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_04_noiseless_EI_net.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
