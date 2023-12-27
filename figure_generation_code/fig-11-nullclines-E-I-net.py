import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plot_utils as pu


def sigmoid(x, beta=5):
    return 1./(1. + np.exp(-beta*x))


def inv_sigmoid(y, beta=5):
    return (1./beta)*np.log(y/(1.-y))


def plot_inh_trialavg_boundary(ax):
    """
    Plots panel (a), the smoothed inhibitory boundary as d|yI|/dt vs yI
    :param ax: Figure axis.
    :return:
    """
    y = np.linspace(0, 5, 101)
    dy = np.zeros_like(y)
    dy1 = np.zeros_like(y)
    dy2 = np.zeros_like(y)
    dy0 = np.zeros_like(y)

    ntrials = 200
    N = 8

    for n in range(ntrials):

        dy -= y
        dy1 -= y
        dy0 -= y

        locs = 2 + 0.5 * np.random.normal(size=(N,))  # 2*np.ones((N,))
        for i in range(N):
            step = 1 * np.ones_like(y)
            step[np.where(y > locs[i])] = 0.
            dy += step
            dy2 += step
    dy0[np.where(y < 2)] = 1e6

    ax.plot(y, dy / ntrials, c=pu.inhibitory_blue, linewidth=1)
    ax.plot(y, dy1 / ntrials, c=pu.inhibitory_blue, alpha=0.5, linewidth=1, linestyle='--')
    ax.plot(y, dy2 / ntrials, c=pu.inhibitory_blue, alpha=0.5, linewidth=1, linestyle='--')

    ax.set_xlim([0, 5])
    ax.set_ylim([-5, 2 + N])
    ax.set_yticks([0])  # np.arange(-4, N + 2, 2))
    ax.set_yticklabels(['0'], fontsize=pu.fs1)
    ax.set_ylabel(r'$d|\bar{y}^I|/dt$', fontsize=pu.fs2)
    ax.set_xticks([0])  # np.arange(0, 5))
    ax.set_xticklabels(['0'], fontsize=pu.fs1)
    ax.set_xlabel(r'$|\bar{y}^I|$', fontsize=pu.fs2)  # , loc='right')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    ax.axhline(y=0, c='gray', alpha=0.5, linewidth=0.75)
    ax.text(0.5, -3.4, r'leak, $-\bar{y}^I$', fontsize=pu.fs1, color=pu.inhibitory_blue, alpha=0.5)
    ax.text(2.5, 2, r'$\sigma_\beta$', fontsize=pu.fs1, color=pu.inhibitory_blue, alpha=0.5)


def plot_I_contours(ax):
    """
    Plots panel (c), the contour plot of dyI/dt in (yE, |yI|) space.
    :param ax: Figure axis.
    :return:
    """

    NI = 50
    tau = 1
    DI = -0.25  # -0.15

    x = 2.
    FI = 1.
    TI = 2.5
    E_IE = 2.
    E_II = 1.

    ye = np.linspace(-1, 5, 201)
    yi = np.linspace(-1, 10, 201)
    YE, YI = np.meshgrid(ye, yi)
    # C = sigmoid(FI * x + E_IE * YE - TI - E_II * YI, beta=3) - 0.04 * YI
    C = sigmoid(FI * x + E_IE * YE - TI - E_II * YI, beta=3) + tau * YI / (NI * DI)
    m = ax.contourf(YE, YI, C, cmap='Blues')

    yI = np.linspace(-10, 5, 201)
    # yE = (1. / E_IE) * (inv_sigmoid(- 0.04 * yI, beta=3) - FI * x + TI - E_II * yI)
    yE = (1. / E_IE) * (inv_sigmoid(tau * yI / (NI * DI), beta=3) - FI * x + TI - E_II * yI)
    ax.plot(yE, -yI, color='black', alpha=1, linewidth=1)
    print(np.nanmin(yE), np.nanmax(yE), np.nanmin(yI), np.nanmax(yI))

    ax.set_xticks([0])  # , 1.5, 3])
    ax.set_yticks([0])  # , 3, 6])
    ax.set_xticklabels(['0'], fontsize=pu.fs1)
    ax.set_yticklabels(['0'], fontsize=pu.fs1)
    ax.set_ylabel(r'$|\bar{y}^I|$', fontsize=pu.fs2)  # ,loc='top')
    ax.set_xlabel(r'$\bar{y}^E$', fontsize=pu.fs2)  # ,loc='right')

    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    cb = plt.colorbar(m, ax=ax)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0])
    cb.set_ticklabels(['0'])
    ax.text(3.3, 7.25, r'$d|\bar{y}^I|/dt$', fontsize=pu.fs1)

    ax.set_xlim([0, 3.75])
    ax.set_ylim([0, 7])


def plot_exc_trialavg_boundary(ax):
    """
    Plots panel (b), the smoothed excitatory boundary as dyE/dt vs yE
    :param ax: Figure axis.
    :return:
    """
    y = np.linspace(0, 5, 101)
    dy = np.zeros_like(y)
    dy1 = np.zeros_like(y)
    dy2 = np.zeros_like(y)
    dy0 = np.zeros_like(y)

    ntrials = 200
    N = 8

    for n in range(ntrials):

        dy -= y
        dy1 -= y
        dy0 -= y

        locs = 3 + 0.5 * np.random.normal(size=(N,))  # 2*np.ones((N,))
        for i in range(N):
            step = 1 * np.ones_like(y)
            step[np.where(y < locs[i])] = 0.
            dy += step
            dy2 += step
    dy0[np.where(y > 3)] = 1e6

    ax.plot(y, dy / ntrials, c=pu.excitatory_red, linewidth=1)
    ax.plot(y, dy1 / ntrials, c=pu.excitatory_red, alpha=0.5, linewidth=1, linestyle='--')
    ax.plot(y, dy2 / ntrials, c=pu.excitatory_red, alpha=0.5, linewidth=1, linestyle='--')

    ax.set_xlim([0, 5])
    ax.set_ylim([-5, 2 + N])
    ax.set_yticks([0])  # np.arange(-4, N + 2, 2))
    ax.set_yticklabels([0], fontsize=pu.fs1)
    ax.set_ylabel(r'$d\bar{y}^E/dt$', fontsize=pu.fs2)
    ax.set_xticks([0])  # np.arange(0, 5))
    ax.set_xticklabels([0], fontsize=pu.fs1)
    ax.set_xlabel(r'$\bar{y}^E$', fontsize=pu.fs2)  # , loc='right')
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    ax.axhline(y=0, c='gray', alpha=0.5, linewidth=0.75)
    ax.text(1.1, -4.2, r'leak, $-\bar{y}^E$', fontsize=pu.fs1, color=pu.excitatory_red, alpha=0.5)
    ax.text(2.5, 5, r'$\sigma_\beta$', fontsize=pu.fs1, color=pu.excitatory_red, alpha=0.5)


def plot_E_contours(ax):
    """
    Plots panel (d), the contour plot of dyE/dt in (yE, |yI|) space.
    :param ax: Figure axis.
    :return:
    """

    NE = 50
    tau = 1
    DE = 0.125  # 0.075

    x = 2.
    FE = 1.
    TE = 1.
    E_EE = 1.25
    E_EI = 1.

    # yE3 = np.linspace(-1,5,201)
    # ax.plot(yE3,(1./E_II)*(FI*x+E_IE*yE3-TI),c=pfcn.blue_color,linestyle='-',alpha=1.)

    ye = np.linspace(-1, 5, 201)
    yi = np.linspace(-1, 10, 201)
    YE, YI = np.meshgrid(ye, yi)
    # C = sigmoid(FE * x + E_EE * YE - TE - E_EI * YI, beta=3) - 0.04 * YE
    C = sigmoid(FE * x + E_EE * YE - TE - E_EI * YI, beta=3) - tau * YE / (NE * DE)
    m = ax.contourf(YE, YI, C, cmap='Reds')

    yE2 = np.linspace(-5, 5, 201)
    # yI2 = (1. / E_EI) * (inv_sigmoid(0.04 * yE2, beta=3) - FE * x + TE - E_EE * yE2)
    yI2 = (1. / E_EI) * (inv_sigmoid(tau * yE2 / (NE * DE), beta=3) - FE * x + TE - E_EE * yE2)

    ax.plot(yE2, -1 * yI2, color='black', alpha=1, linewidth=1)

    ax.set_xticks([0])  # , 1.5, 3])
    ax.set_yticks([0])  # , 3, 6])
    ax.set_xticklabels([0], fontsize=pu.fs1)
    ax.set_yticklabels([0], fontsize=pu.fs1)
    ax.set_ylabel(r'$|\bar{y}^I|$', fontsize=pu.fs2)  # ,loc='top')
    ax.set_xlabel(r'$\bar{y}^E$', fontsize=pu.fs2)  # ,loc='right')

    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    cb = plt.colorbar(m, ax=ax)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0])
    cb.set_ticklabels(['0'])
    ax.text(3.3, 7.25, r'$d\bar{y}^E/dt$', fontsize=pu.fs1)

    ax.set_xlim([0, 3.75])
    ax.set_ylim([0, 7])


def plot_EI_linear_rate_nullclines(ax):
    """
    Plots panel (e), the E-I nullclines for linear (homogeneous) boundaries.
    :param ax: Figure axis.
    :return:
    """

    NE = 50
    NI = 50
    tau = 1
    DI = -0.25  # -0.15
    DE = 0.125  # 0.075

    x = 2.
    FI = 1.
    FE = 1.
    TI = 2.5
    TE = 1.
    E_IE = 2.
    E_II = 1.
    E_EE = 1.25
    E_EI = 1.

    betas = [10, 3, 1]
    alphas = [1.0, 0.3, 0.1]
    for i in range(len(betas)):
        yI = np.linspace(-10, 5, 201)
        yE = (1. / E_IE) * (inv_sigmoid(tau * yI / (NI * DI), beta=betas[i]) - FI * x + TI - E_II * yI)

        ax.plot(yE, -1 * yI, color=pu.inhibitory_blue, alpha=alphas[i], linewidth=1)

    for i in range(len(betas)):
        yE2 = np.linspace(-5, 5, 201)
        yI2 = (1. / E_EI) * (inv_sigmoid(tau * yE2 / (NE * DE), beta=betas[i]) - FE * x + TE - E_EE * yE2)

        ax.plot(yE2, -1 * yI2, color=pu.excitatory_red, alpha=alphas[i], linewidth=1)

    ax.legend(('', '', '', r'$\beta=10$', r'$\beta=3$', r'$\beta=1$'), loc='upper left',
              ncol=2, fontsize=pu.fs1, frameon=False, handlelength=1, columnspacing=0.1,
              bbox_to_anchor=(0.05, 1.0))
    ax.text(0.5, 6.75, r'$I$', fontsize=pu.fs1)
    ax.text(1.0, 6.75, r'$E$', fontsize=pu.fs1)

    yE3 = np.linspace(-1, 5, 201)
    ax.plot(yE3, (1. / E_EI) * (FE * x + E_EE * yE3 - TE), c=pu.excitatory_red, linestyle=':', alpha=0.5)
    ax.plot(yE3, (1. / E_II) * (FI * x + E_IE * yE3 - TI), c=pu.inhibitory_blue, linestyle=':', alpha=0.5)

    ax.set_xticks([0])  # , 1.5, 3])
    ax.set_yticks([0])  # , 3, 6])
    ax.set_xticklabels([0], fontsize=pu.fs1)
    ax.set_yticklabels([0], fontsize=pu.fs1)
    ax.set_ylabel(r'$|\bar{y}^I|$', fontsize=pu.fs2)  # ,loc='top')
    ax.set_xlabel(r'$\bar{y}^E$', fontsize=pu.fs2)  # ,loc='right')

    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()

    ax.set_xlim([0, 3.75])
    ax.set_ylim([0, 7])


def plot_EI_quadratic_rate_nullclines(ax):
    """
    Plots panel (f), the E-I nullclines for nonlinear (heterogeneous) boundaries.
    :param ax: Figure axis.
    :return:
    """
    NE = 50
    NI = 50
    tau = 1
    DI = -0.15
    DE = 0.125

    x = 2.
    FI = 1.
    FE = 1.
    TI = 2.5
    TE = 1.
    bI = FI * x - TI + 2
    bE = FE * x - TE + 2

    aE = 0.1
    aI = 0.2

    betas = [10, 3, 1]
    alphas = [1.0, 0.3, 0.1]
    for i in range(len(betas)):
        # I
        yI = np.linspace(-10, 10, 201)
        yE = np.sqrt((1. / aI) * (inv_sigmoid(tau * yI / (NI * DI), beta=betas[i]) - yI - bI))

        # E
        yE2 = np.linspace(-10, 10, 201)
        yI2 = (aE * yE2 ** 2 - inv_sigmoid(tau * yE2 / (NE * DE), beta=betas[i]) + bE)

        ax.plot(yE, -1 * yI, color=pu.inhibitory_blue, alpha=alphas[i], linewidth=1)
        ax.plot(yE2, yI2, color=pu.excitatory_red, alpha=alphas[i], linewidth=1)

    yE3 = np.linspace(-1, 7, 201)
    ax.plot(yE3, aE * yE3 ** 2 + bE, c=pu.excitatory_red, linestyle=':', alpha=0.5)
    ax.plot(yE3, aI * yE3 ** 2 + bI, c=pu.inhibitory_blue, linestyle=':', alpha=0.5)

    ax.set_xlim([0, 6.3])
    ax.set_ylim([0, 8])
    ax.set_yticks([0])  # 1, 4, 7])
    ax.set_xticks([0])  # , 3, 6])
    ax.set_xticklabels([0], fontsize=pu.fs1)
    ax.set_yticklabels([0], fontsize=pu.fs1)
    ax.set_ylabel(r'$|\bar{y}^I|$', fontsize=pu.fs2)  # ,loc='top')
    ax.set_xlabel(r'$\bar{y}^E$', fontsize=pu.fs2)  # ,loc='right')

    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()


def main(save_pdf=False, show_plot=True):
    f = plt.figure(figsize=(4.5, 3.1), dpi=150)

    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.25, 1])
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[0, 2])
    ax4 = f.add_subplot(gs[1, 0])
    ax5 = f.add_subplot(gs[1, 1])
    ax6 = f.add_subplot(gs[1, 2])

    plot_inh_trialavg_boundary(ax1)
    plot_I_contours(ax2)
    plot_exc_trialavg_boundary(ax4)
    plot_E_contours(ax5)

    plot_EI_linear_rate_nullclines(ax3)
    plot_EI_quadratic_rate_nullclines(ax6)
    f.tight_layout()

    ax1.text(-0.2, 1.075, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.2, 1.075, r'\textbf{c}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.2, 1.075, r'\textbf{e}', transform=ax3.transAxes, **pu.panel_prms)
    ax4.text(-0.2, 1.075, r'\textbf{b}', transform=ax4.transAxes, **pu.panel_prms)
    ax5.text(-0.2, 1.075, r'\textbf{d}', transform=ax5.transAxes, **pu.panel_prms)
    ax6.text(-0.2, 1.075, r'\textbf{f}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_11_E_I_nullclines.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
