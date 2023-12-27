import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plot_utils as pu
import net_sim_code as scnf


def plot_ideal_boundary(ax):
    """
    Plot for panel (a) -- The ideal, infinitely-steep inhibitory boundary.
    :param ax: Figure axis.
    :return:
    """
    ax.plot([-2, -5], [2, 5], c=pu.inhibitory_blue, linewidth=1)
    ax.plot([-2, -2], [-10, 2], c=pu.inhibitory_blue, linewidth=0.75, linestyle=':')
    ax.plot([0, -2], [-10, -10], c=pu.inhibitory_blue, linewidth=1)
    ax.axhline(y=0, linewidth=0.5, c='gray', alpha=0.5)

    ax.set_xlim([-4, 0])
    ax.set_ylim([-11, 8])
    ax.set_yticks([-10, 0])
    ax.set_yticklabels([r'-$\infty$', '0'], fontsize=pu.fs1)
    ax.set_ylabel(r'$dy/dt$', fontsize=pu.fs2)
    ax.set_xticks(np.arange(-4, 1))
    ax.set_xticklabels(['-4', '-3', '-2', '-1', '0'], fontsize=pu.fs1)
    ax.set_xlabel(r'$y$', fontsize=pu.fs2)
    ax.text(-2.4, -5.5, r'$y$$=$$y_0$', fontsize=pu.fs2, rotation='vertical')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    p1 = patches.Rectangle((-2, -11), 2, 19, facecolor=pu.inhibitory_blue, edgecolor='white', alpha=0.15)
    ax.add_patch(p1)
    p2 = patches.Rectangle((-4.5, -8), 6, 1, facecolor='white', edgecolor='white', alpha=1.0, clip_on=False, zorder=15)
    ax.add_patch(p2)
    ax.plot([-4.25, -3.75], [-8.5, -7.5], linewidth=0.5, color='black', zorder=15, clip_on=False)
    ax.plot([-4.25, -3.75], [-7.5, -6.5], linewidth=0.5, color='black', zorder=15, clip_on=False)


def simulate_y_exp_alpha_dynamics(tsteps=600, dt=1e-4, leak=100., leak_s=500., D=-4.):
    """
    Simulate the dynamics of the latent variable, y, and synaptic variable alpha, for exponentially-decaying synapses
    with inverse time constant leak_s.
    NOTE: THIS IS NOT USED IN THE FIGURE.
    :param tsteps: Number of simulated time steps.
    :param dt: Time step.
    :param leak: Inverse membrane time constant, 1/s.
    :param leak_s: Inverse synaptic time constant, 1/s.
    :param D: Decoder magnitude.
    :return:
    """

    y = np.zeros((tsteps,))
    alpha = np.zeros((tsteps,))
    y[0] = -4.
    for t in range(1, tsteps):
        y[t] = y[t - 1] - dt * leak * y[t - 1] + dt * D * alpha[t - 1]
        if leak_s == 0.:
            alpha[t] = 0.
        else:
            alpha[t] = alpha[t - 1] - dt * leak_s * alpha[t - 1]
        if y[t] >= -2.:
            if leak_s == 0.:
                alpha[t] = 1. / dt
            else:
                alpha[t] = leak_s
    return y, dt * alpha


def simulate_y_pulse_alpha_dynamics(tsteps=2500, dt=1e-4, leak=100., leak_s=500., D=-4.):
    """
    Simulate the dynamics of the latent variable, y, and synaptic variable alpha, for square pulse synapses.
    :param tsteps: Number of simulated time steps.
    :param dt: Time step.
    :param leak: Inverse membrane time constant, 1/s.
    :param leak_s: Inverse synaptic time constant, 1/s.
    :param D: Decoder magnitude.
    :return:
    """
    pw = int(1. / (dt * leak_s))
    y = np.zeros((tsteps,))
    alpha = np.zeros((tsteps,))
    y[0] = -2.
    for t in range(1, tsteps):
        y[t] = y[t - 1] - dt * leak * y[t - 1] + dt * D * alpha[t - 1]
        if leak_s == 0.:
            alpha[t] = 0.
        if y[t] >= -2.:
            if leak_s == 0.:
                alpha[t] = 1. / dt
            else:
                alpha[t:(t + pw)] = 1. / pw / dt
        if np.abs(alpha[t-1]-alpha[t]) > 1e-3:
            y[t] = y[t - 1]
    return y, alpha


def plot_y_exp_pulse_alpha_comparison(ax):
    """
    Plot for panel (b) -- The latent variable-synaptic variable dynamics.
    :param ax: Figure axis.
    :return:
    """

    p1 = patches.Rectangle((-2, 0), 2, 1500, facecolor='gray', edgecolor='white', alpha=0.15)
    ax.add_patch(p1)
    ax.axhline(y=0, linewidth=0.5, c='gray', alpha=0.5)

    D = -1
    tsteps = 600
    y, alpha = simulate_y_pulse_alpha_dynamics(tsteps=tsteps, D=D, leak_s=1000., dt=1e-5)
    ax.plot(y, alpha, c='black', linewidth=0.75, alpha=1, linestyle='-')
    ax.plot(y[1:2], alpha[1:2], '^', c='black', alpha=1, markersize=2)
    ax.plot(y[99:100], alpha[99:100], '<', c='black', alpha=1, markersize=2)
    ax.plot(y[101:102], alpha[101:102], 'v', c='black', alpha=1, markersize=2)
    ax.plot(y[202:400:95], alpha[202:400:95], '>', c='black', alpha=1, markersize=2)

    y, alpha = simulate_y_pulse_alpha_dynamics(tsteps=tsteps, D=D, leak_s=500., dt=1e-5)
    ax.plot(y, alpha, c='gray', linewidth=0.75, alpha=1, linestyle='-')
    ax.plot(y[1:2], alpha[1:2], '^', c='gray', alpha=1, markersize=2)
    ax.plot(y[97:200:95], alpha[97:200:95], '<', c='gray', alpha=1, markersize=2)
    ax.plot(y[201:202], alpha[201:202], 'v', c='gray', alpha=1, markersize=2)
    ax.plot(y[302:400:95], alpha[302:400:95], '>', c='gray', alpha=1, markersize=2)

    y, alpha = simulate_y_pulse_alpha_dynamics(tsteps=tsteps, D=D, leak_s=250., dt=1e-5)
    ax.plot(y, alpha, c='silver', linewidth=0.75, linestyle='-')
    ax.plot(y[1:2], alpha[1:2], '^', c='silver', alpha=1, markersize=2)
    ax.plot(y[97:400:95], alpha[97:400:95], '<', c='silver', alpha=1, markersize=2)
    ax.plot(y[401:402], alpha[401:402], 'v', c='silver', alpha=1, markersize=2)

    ax.set_xlim([-3, -1.5])
    ax.set_ylim([0, 1300])
    ax.set_yticks([0, 500, 1000])
    ax.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)  # '0', '5', '10'], fontsize=pu.fs1)
    ax.set_ylabel(r'$\alpha_k$', fontsize=pu.fs2)

    ax.set_xticks(np.arange(-3, -1.4, 0.5))
    ax.set_xticklabels(['-3', '-2.5', '-2', '-1.5'], fontsize=pu.fs1)
    ax.set_xlabel(r'$y$', fontsize=pu.fs2)  # , loc='left')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    ax.set_ylabel(r'$s_k$', fontsize=pu.fs2)
    ax.text(-2.45, 325, r'$\tau_s$$=$$\tau / 2$', fontsize=pu.fs1, rotation=0, color='silver')
    ax.text(-2.7, 600, r'$\tau_s$$=$$\tau / 5$', fontsize=pu.fs1, rotation=0, color='gray')
    ax.text(-2.95, 1100, r'$\tau_s$$=$$\tau / 10$', fontsize=pu.fs1, rotation=0, color='black')
    ax.set_xlim([-3, -1.75])


# def plot_y_alpha_boundary(ax, leak_s=200.):
#     """
#     Simulates the exponentially-decaying
#     NOTE: this is not used in the figure.
#     :param ax: Figure axis.
#     :param leak_s: Inverse synaptic time constant.
#     :return:
#     """
#
#     D = -6.
#     y, alpha = simulate_y_exp_alpha_dynamics(D=D, leak_s=leak_s)
#     ax.plot(y, D * alpha, c='black', linewidth=0.75)
#     ax.axhline(y=0, linewidth=0.5, c='gray', alpha=0.5)
#
#     ax.set_xlim([-4, 0])
#     ax.set_yticks(np.arange(-6, 5, 3))
#     ax.set_yticklabels(['-6', '-3', '0', '3'], fontsize=pu.fs1)
#     ax.set_ylabel(r'$D\alpha$', fontsize=pu.fs2)
#     ax.set_xticks(np.arange(-4, 1))
#     ax.set_xticklabels(['-4', '-3', '-2', '-1', '0'], fontsize=pu.fs1)
#     ax.set_xlabel(r'$y$', fontsize=pu.fs2)  # , loc='left')
#     ax.tick_params(axis='both', width=0.5, length=3, pad=1)
#     sns.despine()
#
#     p1 = patches.Rectangle((-2, -11), 2, 19, facecolor='gray', edgecolor='white', alpha=0.15)
#     ax.add_patch(p1)
#
#     style = "Simple, tail_width=0.5, head_width=3.5, head_length=3"
#     kw = dict(arrowstyle=style, color="black", alpha=1, linewidth=0.5, clip_on=False)
#     a = patches.FancyArrowPatch((-3, 0), (-2.75, 0), **kw, zorder=10)  # horizontal
#     ax.add_patch(a)
#     a2 = patches.FancyArrowPatch((-2, -3), (-2, -3.25), **kw, zorder=10)  # horizontal
#     ax.add_patch(a2)
#     a3 = patches.FancyArrowPatch((-2.68, -4.1), (-2.82, -3.5), **kw, zorder=10)  # horizontal
#     ax.add_patch(a3)


# def plot_time_dynamics(ax1, ax2, ax3):
#     # plot a trajectory
#     tsteps = 600
#     dt = 1e-4
#     Tmax = dt*tsteps
#     leak = 100.
#     D = -1.
#     y = np.zeros((tsteps,))
#     y[0] = -4.
#     for t in range(1, tsteps):
#         y[t] = y[t - 1] - dt * leak * y[t - 1]
#         if y[t] >= -2.:
#             y[t] += D
#     ax1.plot(np.linspace(0, Tmax, tsteps), y, color='black', linewidth=0.75)
#
#     y, alpha = simulate_y_exp_alpha_dynamics(D=-6.)
#     ax2.plot(np.linspace(0, Tmax, tsteps), alpha, color='black', linewidth=0.75)
#     ax2.plot(np.linspace(0, Tmax, tsteps), 0.5 * np.ones_like(alpha), color='gray', alpha=0.5, linewidth=0.5)
#     ax3.plot(np.linspace(0, Tmax, tsteps), y, color='black', linewidth=0.75)
#     ax3.plot(np.linspace(0, Tmax, tsteps), -2.4 * np.ones_like(y), alpha=0.5, color='gray', linewidth=0.5)
#
#     yticks = [[-4, -2], [0, 1], [-4, -2]]
#     xticks = [0, 0.02, 0.04, 0.06]
#     ylabs = [r'$y$', r'$\alpha$', r'$y$']
#     for i, ax in enumerate([ax1, ax2, ax3]):
#         ax.tick_params(axis='both', width=0.5, length=3, pad=1)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels([])
#         ax.set_yticks(yticks[i])
#         ax.set_yticklabels([str(x) for x in yticks[i]], fontsize=pu.fs1)
#         ax.set_ylabel(ylabs[i], fontsize=pu.fs2)
#     ax3.set_xticklabels([str(int(100.*x)) for x in xticks], fontsize=pu.fs1)
#     ax3.set_xlabel(r'time ($\tau$)', fontsize=pu.fs2)
#     ax1.set_ylim([-4.5, -1.75])
#     ax2.set_ylim([0, 1.5])
#     ax3.set_ylim([-4.5, -1.75])


def plot_alpha_function(ax):
    """
    Plot for panel (c), the finite, heaviside-function approximation to the synaptic dynamics.
    :param ax: Figure axis.
    :return:
    """

    fp = -2
    ax.axhline(y=0, linewidth=0.5, c='gray', alpha=0.5)
    y = np.linspace(-5, 0, 101)
    ax.plot(y[y < fp], np.zeros_like(y[y < fp]), c='black', linewidth=1)
    ax.plot(y[y > fp], 1000 * np.ones_like(y[y > fp]), c='black', linewidth=1)
    ax.plot(y[y > fp], 500 * np.ones_like(y[y > fp]), c='gray', linewidth=1)
    ax.plot(y[y > fp], 250 * np.ones_like(y[y > fp]), c='silver', linewidth=1)
    ax.plot([fp, fp], [0, 1000], c='black', linewidth=0.75, linestyle=':')
    ax.plot([fp, fp], [0, 500], c='gray', linewidth=0.75, linestyle=':')
    ax.plot([fp, fp], [0, 250], c='silver', linewidth=0.75, linestyle=':')

    ax.set_xlim([-4, 0])
    ax.set_ylim([0, 1300])
    ax.set_yticks([0, 500, 1000])
    ax.set_yticklabels(['0', '5', '10'], fontsize=pu.fs1)  # '0', '5', '10'], fontsize=pu.fs1)
    ax.set_ylabel(r'$\bar{s}_k(\bar{y})$', fontsize=pu.fs2)
    ax.set_xticks(np.arange(-4, 1))
    ax.set_xticklabels(['-4', '-3', '-2', '-1', '0'], fontsize=pu.fs1)
    ax.set_xlabel(r'$\bar{y}$', fontsize=pu.fs2)  # , loc='left')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()

    ax.text(-2.1, 225, r'$\tau_s$$=$$\tau / 2$', fontsize=pu.fs1, rotation=0, color='silver',
            horizontalalignment='right')
    ax.text(-2.1, 475, r'$\tau_s$$=$$\tau / 5$', fontsize=pu.fs1, rotation=0, color='gray',
            horizontalalignment='right')
    ax.text(-2.1, 975, r'$\tau_s$$=$$\tau / 10$', fontsize=pu.fs1, rotation=0, color='black',
            horizontalalignment='right')

    p1 = patches.Rectangle((fp, -1), 3, 13, facecolor='gray', edgecolor='white', alpha=0.15)
    ax.add_patch(p1)


def plot_one_step_boundary(ax, seed=None):
    """
    Plot for panel (d), the finite boundary in latent space based on the heaviside synaptic function.
    :param ax: Figure axis.
    :param seed: Random seed.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    fp = -2

    y = np.linspace(-5, 0, 501)
    dy = np.zeros_like(y)
    dy[y > fp] = -8
    dy2 = np.zeros_like(y)
    dy2[y > fp] = -4

    ax.plot(y, -y, c=pu.inhibitory_blue, linewidth=1, linestyle='--', alpha=0.25)
    ax.plot(y[y < fp], dy[y < fp], c=pu.inhibitory_blue, linewidth=1, linestyle='--', alpha=0.25)
    ax.plot(y[y > fp], dy[y > fp], c=pu.inhibitory_blue, linewidth=1, linestyle='--', alpha=0.25)
    ax.plot(y[y < fp], dy[y < fp] - y[y < fp], c=pu.inhibitory_blue, linewidth=1)
    ax.plot(y[y > fp], dy[y > fp] - y[y > fp], c=pu.inhibitory_blue, linewidth=1)
    ax.plot([fp, fp], [2, -6], color=pu.inhibitory_blue, linestyle=':', linewidth=0.75)
    ax.plot([fp, fp], [0, -8], color=pu.inhibitory_blue, linestyle=':', linewidth=0.75, alpha=0.5)
    ax.axhline(y=0, linewidth=0.5, c='gray', alpha=0.5)

    p1 = patches.Rectangle((fp, -21), 3, 29, facecolor=pu.inhibitory_blue, edgecolor='white', alpha=0.15)
    ax.add_patch(p1)

    ax.set_xlim([-4, 0])
    ax.set_ylim([-11, 7])
    ax.set_yticks([0])  # np.arange(-10, 8, 5))
    ax.set_yticklabels(['0'], fontsize=pu.fs1)  # '-10', '-5', '0', '5'], fontsize=pu.fs1)
    ax.set_ylabel(r'$d\bar{y}/dt$', fontsize=pu.fs2)
    ax.set_xticks(np.arange(-4, 1))
    ax.set_xticklabels(['-4', '-3', '-2', '-1', '0'], fontsize=pu.fs1)
    ax.set_xlabel(r'$\bar{y}$', fontsize=pu.fs2)  # , loc='left')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    ax.text(-1.35, 2.1, r'leak, $-\bar{y}$', fontsize=pu.fs1, color=pu.inhibitory_blue, alpha=0.5)
    ax.text(-3.4, -5.25, r'$\frac{D_k}{\tau_s}$', fontsize=pu.fs3, color=pu.inhibitory_blue, alpha=0.75)
    ax.text(-2.6, -4.8, r'$\begin{cases} & \\ & \\ & \\ & \end{cases}$', fontsize=pu.fs1-0.45,
            color=pu.inhibitory_blue, alpha=0.75)


def plot_trialavg_boundary(ax):
    """
    Plot for panel (e), the smoothed, sigmoidal boundary.
    :param ax: Figure axis.
    :return:
    """

    # (1) first do a real simulation and estimate dy/dt as a function of y
    ntrials = 151
    ntrials2 = 10
    yvals = []
    dyvals = []
    y0vals = np.linspace(-4, 1, ntrials)
    for n in range(ntrials):

        for n2 in range(ntrials2):
            Tend = 0.001
            dt = 1e-4
            leak = 100.
            time = np.arange(0, Tend, dt)
            tsteps = len(time)
            y0 = y0vals[n]  # -4 * np.random.rand(1)
            D = -2 * np.ones((1, 1))
            F = np.zeros((1, 1))
            E = np.ones((1, 1))
            x = np.zeros((1, tsteps))
            T = -2 * np.ones((1,)) + 0.4 * np.random.normal(size=(1,))
            (s, V, g) = scnf.run_spiking_net(x, D, E, F, T, dt=1e-4, leak=leak, mu=0., current_inj=None, sigma_v=0.,
                                             pw=2, tref=100, y0=np.array([y0])[:, None])
            r0 = np.array([y0 / D[0, 0]])
            r = scnf.exp_filter(g, time, r0=r0)
            y = D @ r
            dy = np.diff(y) / dt / leak
            yvals.append(y[:, :-1])
            dyvals.append(dy)

    yvals = np.concatenate(yvals).flatten()
    dyvals = np.concatenate(dyvals).flatten()
    idxs = np.argsort(yvals)
    yvals = yvals[idxs]
    dyvals = dyvals[idxs]

    y_avg = np.linspace(-4, 1, 51)
    dy = y_avg[1] - y_avg[0]
    dydt_avg = np.zeros_like(y_avg)
    for i in range(len(y_avg)):
        ymin = y_avg[i]-dy
        ymax = y_avg[i]+dy
        idxs = np.logical_and(yvals >= ymin, yvals < ymax)
        dydt_avg[i] = np.mean(dyvals[idxs])

    ax.plot(y_avg, dydt_avg, c='gray', label='sim.')

    # (2) now just make a sigmoid by averaging over heavisides
    y = np.linspace(-5, 0, 101)
    dy = np.zeros_like(y)
    dy1 = np.zeros_like(y)
    dy2 = np.zeros_like(y)

    ntrials = 200
    N = 8

    for n in range(ntrials):

        dy -= y
        dy1 -= y

        locs = -2.25 + 0.5 * np.random.normal(size=(N,))  # 2*np.ones((N,))
        for i in range(N):
            step = -1.1 * np.ones_like(y)
            step[np.where(y < locs[i])] = 0.
            dy += step
            dy2 += step
    ax.plot(y, dy / ntrials, c=pu.inhibitory_blue, linewidth=1, label='ideal')
    ax.legend(fontsize=pu.fs1, ncols=1, frameon=False, handlelength=1.5, columnspacing=0.75,
              loc="upper right", bbox_to_anchor=(1.05, 0.55))
    ax.plot(y, dy1 / ntrials, c=pu.inhibitory_blue, alpha=0.25, linewidth=1, linestyle='--')
    ax.plot(y, dy2 / ntrials, c=pu.inhibitory_blue, alpha=0.25, linewidth=1, linestyle='--')
    ax.axhline(y=0, linewidth=0.5, c='gray', alpha=0.5)

    ax.set_xlim([-4, 0])
    ax.set_ylim([-11, 7])
    ax.set_yticks([0])
    ax.set_yticklabels(['0'], fontsize=pu.fs1)
    ax.set_ylabel(r'$d\bar{y}/dt$', fontsize=pu.fs2)
    ax.set_xticks(np.arange(-4, 1))
    ax.set_xticklabels(['-4', '-3', '-2', '-1', '0'], fontsize=pu.fs1)
    ax.set_xlabel(r'$\bar{y}$', fontsize=pu.fs2)
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    ax.text(-1.35, 2.1, r'leak, $-\bar{y}$', fontsize=pu.fs1, color=pu.inhibitory_blue, alpha=0.5)
    ax.text(-3.1, -8, r'$\sigma_\beta(\bar{y})$', fontsize=pu.fs2, color=pu.inhibitory_blue, alpha=0.75)


def sigmoid(x, beta=5):
    return 1./(1. + np.exp(-beta*x))


def plot_boundary_contours(ax):
    """
    Plot for panel (f), the softer inhibitory boundary as in Fig 2b,c.
    :param ax: Figure axis.
    :return:
    """

    a = -1.0
    b = 0
    c = -0.5

    y = np.linspace(-2, 0, 201)
    x = np.linspace(-1, 1, 201)
    Y, X = np.meshgrid(y, x)
    C = -sigmoid(Y - (a * X**2 + b * X + c), beta=5) - 0.1 * Y
    m = ax.contourf(X, Y, C, cmap='Blues_r')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-2, 0])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-2, -1, 0])
    ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
    ax.set_yticklabels(['-2', '-1', '0'], fontsize=pu.fs1)
    ax.set_ylabel(r'$\bar{y}$', fontsize=pu.fs2)  # ,loc='top')
    ax.set_xlabel(r'$x$', fontsize=pu.fs2)  # ,loc='right')

    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax.axhline(y=0, linewidth=1.0, color='black')
    sns.despine(ax=ax, bottom=True)

    axins = inset_axes(ax, width="5%", height="80%", loc='right',  # borderpad=-0.75,
                       bbox_to_anchor=(.41, -0.1, .75, 1.1),
                       bbox_transform=ax.transAxes)
    cb = plt.colorbar(m, ax=ax, cax=axins, orientation="vertical")
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0])
    cb.set_ticklabels(['0'])
    ax.text(1.05, -0.05, r'$\frac{d\bar{y}}{dt}$', fontsize=pu.fs2)
    return axins


def main(save_pdf=False, show_plot=True):
    f = plt.figure(figsize=(4.5, 2.75), dpi=150)
    gs = f.add_gridspec(2, 3)
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[0, 2])
    ax4 = f.add_subplot(gs[1, 0])
    ax5 = f.add_subplot(gs[1, 1])
    ax6 = f.add_subplot(gs[1, 2])

    plot_ideal_boundary(ax1)
    plot_y_exp_pulse_alpha_comparison(ax2)
    plot_alpha_function(ax3)
    plot_one_step_boundary(ax4)
    plot_trialavg_boundary(ax5)
    axins = plot_boundary_contours(ax6)

    f.tight_layout()
    f.subplots_adjust(wspace=0.5)

    box = axins.get_position()
    box.x0 = box.x0 + 0.1
    box.x1 = box.x1 + 0.1
    axins.set_position(box)

    ax1.text(-0.35, 1.05, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.35, 1.05, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.35, 1.05, r'\textbf{c}', transform=ax3.transAxes, **pu.panel_prms)
    ax4.text(-0.35, 1.05, r'\textbf{d}', transform=ax4.transAxes, **pu.panel_prms)
    ax5.text(-0.35, 1.05, r'\textbf{e}', transform=ax5.transAxes, **pu.panel_prms)
    ax6.text(-0.35, 1.05, r'\textbf{f}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_09_soft_inhib_boundary.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
