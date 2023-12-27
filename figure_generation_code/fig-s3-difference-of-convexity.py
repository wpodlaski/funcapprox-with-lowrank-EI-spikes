
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf
from scipy.stats import norm,gamma
from scipy.interpolate import interp2d


# generate data from a Gaussian process (for variable single-neuron targets)
def cov(params, kernel, x, x_star):
    if len(x) == len(x_star):  # if the data is square...
        cov_m = np.zeros((len(x), len(x_star)))
        for i in range(len(x)):
            for j in range(i + 1, len(x_star)):  # only need to compute upper triangular
                cov_m[i, j] = kernel(params, x[i], x_star[j])
        cov_m += cov_m.T  # append the transpose to gain square matrix
        for i in range(len(x)):
            for j in range(len(x)):  # make sure diagonal is only added once
                cov_m[i, j] = kernel(params, x[i], x_star[j])
        return cov_m
    else:  # otherwise just produce full rectangular matrix
        cov_m = np.zeros((len(x), len(x_star)))
        for i in range(len(x)):
            for j in range(len(x_star)):
                cov_m[i, j] = kernel(params, x[i], x_star[j])
    return cov_m


def squared_exp(params, x1, x2):
    # Defining a covariance function
    l = params[0]  # length scale
    sig = params[1]  # variance
    sig_n = params[2]  # noise parameter
    return (sig ** 2 * np.exp(-((x1 - x2) ** 2 / 2 * l ** 2))) + sig_n


def diff_x(x):
    return (x + 0.5 * (x[1] - x[0]))[:-1]


def find_numerical_roots(x, a):
    a0 = a[0]
    x0 = x[0]
    x_roots = []
    idx_roots = []
    for i in range(1, a.shape[0]):
        a1 = a[i]
        x1 = x[i]
        if a0 * a1 < 0:
            x_roots.append(0.5 * (x0 + x1))
            idx_roots.append(i)
        a0 = a1
        x0 = x1
    return x_roots, idx_roots


def get_piecewise_fcn(x, f, x_roots):
    # build a piecewise linear version of the function
    y_pw = np.zeros_like(x)
    y_roots = []
    xnew_roots = []

    # first data point
    idx0 = 0
    x0 = x[idx0]
    y0 = f[idx0]
    xnew_roots.append(x0)
    y_roots.append(y0)

    for i in range(1, len(x_roots)):

        x1 = x_roots[i]
        idx1 = np.argmin(np.abs(x - x1))  # find the closes value of x to x1
        y1 = f[idx1]

        print(i, idx0, idx1)

        if np.abs(y1 - y0) > 0.25 or i == len(x_roots) - 1:
            y_roots.append(y1)
            xnew_roots.append(x1)
            y_pw[idx0:(idx1 + 1)] = np.linspace(y0, y1, idx1 + 1 - idx0)
            idx0 = idx1
            y0 = y1

    return xnew_roots, y_roots, y_pw


def plot_function_non_monotonic(axs):

    np.random.seed(44)
    y = np.linspace(0, 20, 100)
    l = 0.3
    sig = 0.5
    sig_n = 1
    params = [l, sig, sig_n]
    mu_y = [-0. for i in range(len(y))]
    cov_y = cov(params, squared_exp, y, y)
    sample = np.random.multivariate_normal(mu_y, cov_y)

    xlims = [0, 10]
    x = np.linspace(xlims[0], xlims[1] - xlims[1] / 100., sample.shape[0])
    d_x = x[1] - x[0]
    dx = diff_x(x)
    ddx = diff_x(dx)
    dddx = diff_x(ddx)

    data = sample
    data -= np.min(data)
    data /= np.max(data)
    data = 1 + 4 * data

    B = data
    G = (1. / d_x) * np.diff(B)
    C = (1. / d_x) * np.diff(G)
    dC = (1. / d_x) * np.diff(C)

    # convex decompositions
    lamb = -1 * min(0, np.min(C))
    B_cvx1 = B + lamb * 0.5 * (x - 5) ** 2
    B_cvx2 = lamb * 0.5 * (x - 5) ** 2
    lamb2 = lamb
    B_cvx3 = B + lamb2 * 0.5 * x ** 2
    B_cvx4 = lamb2 * 0.5 * x ** 2

    G_cvx1 = (1. / d_x) * np.diff(B_cvx1)
    C_cvx1 = (1. / d_x) * np.diff(G_cvx1)
    dC_cvx1 = (1. / d_x) * np.diff(C_cvx1)
    G_cvx2 = (1. / d_x) * np.diff(B_cvx2)
    C_cvx2 = (1. / d_x) * np.diff(G_cvx2)
    dC_cvx2 = (1. / d_x) * np.diff(C_cvx2)
    G_cvx3 = (1. / d_x) * np.diff(B_cvx3)
    C_cvx3 = (1. / d_x) * np.diff(G_cvx3)
    dC_cvx3 = (1. / d_x) * np.diff(C_cvx3)
    G_cvx4 = (1. / d_x) * np.diff(B_cvx4)
    C_cvx4 = (1. / d_x) * np.diff(G_cvx4)
    dC_cvx4 = (1. / d_x) * np.diff(C_cvx4)

    x_roots, idx_roots = find_numerical_roots(dx, G)  # find_numerical_roots(dddx, dC)
    x_roots = np.array([xlims[0]] + x_roots + [xlims[1]])

    x_roots, B_roots, B_pw = get_piecewise_fcn(x, B, x_roots)

    G_pw = (1. / d_x) * np.diff(B_pw)
    C_pw = (1. / d_x) * np.diff(G_pw)
    dC_pw = (1. / d_x) * np.diff(C_pw)

    d_x = 0.1  # x[1] - x[0]

    C_E_x = np.zeros_like(C_pw)
    C_I_x = np.zeros_like(C_pw)

    g_E_x = 0.25
    g_I_x = g_E_x - G_pw[0]
    b_E_x = B_pw[0]
    b_I_x = 0.

    C_E_x[C_pw > 1e-4] = C_pw[C_pw > 1e-4]
    C_I_x[C_pw < -1e-4] = -C_pw[C_pw < -1e-4]

    G_E_x = np.cumsum(d_x * C_E_x) + g_E_x
    G_I_x = np.cumsum(d_x * C_I_x) + g_I_x

    B_E_x = np.cumsum(d_x * G_E_x) + b_E_x
    B_I_x = np.cumsum(d_x * G_I_x) + b_I_x

    # plot function itself
    axs[0, 0].plot(x, B, linewidth=1, color='black')
    # axs[0, 1].plot(x, B, linewidth=1, color='black')
    axs[1, 0].plot(x, B, c='gray', alpha=0.5, linewidth=0.75)
    # axs[0, 3].plot(x, B, c='gray', alpha=0.5, linewidth=0.75)
    axs[1, 0].plot(x, B_pw, linewidth=1, color='black')
    # axs[0, 3].plot(x, B_pw, linewidth=1, color='black')

    # plot decomposition into convex functions
    axs[0, 1].plot(x, B_cvx1, linewidth=1, color=pu.excitatory_red)
    axs[0, 1].plot(x, B_cvx2, linewidth=1, color=pu.inhibitory_blue)
    # axs[1, 1].plot(x, B_cvx3, linewidth=1, color=pu.excitatory_red)
    # axs[1, 1].plot(x, B_cvx4, linewidth=1, color=pu.inhibitory_blue)

    axs[1, 1].plot(x[1:-1], B_E_x, color=pu.excitatory_red, linewidth=1)
    axs[1, 1].plot(x[1:-1], B_I_x, color=pu.inhibitory_blue, linewidth=1)

    axs[1, 2].plot(x[:-1] + d_x, G_pw, linewidth=0.75, color='black', linestyle=':')
    axs[1, 2].plot(x[1:-1] + d_x, G_E_x, color=pu.excitatory_red, linewidth=0.75, linestyle=':')
    axs[1, 2].plot(x[1:-1] + d_x, G_I_x, color=pu.inhibitory_blue, linewidth=0.75, linestyle=':')

    # remove a point from the gradient
    g_prev = G_pw[0]
    for i, g in enumerate(G_pw):
        if np.abs(g - g_prev) > 0.1:
            G_pw[i] = np.nan
        g_prev = g
    g_prev = G_E_x[0]
    for i, g in enumerate(G_E_x):
        if np.abs(g - g_prev) > 0.1:
            G_E_x[i] = np.nan
        g_prev = g
    g_prev = G_I_x[0]
    for i, g in enumerate(G_I_x):
        if np.abs(g - g_prev) > 0.1:
            G_I_x[i] = np.nan
        g_prev = g

    # plot gradients
    axs[0, 2].plot(x[:-1]+d_x, G, linewidth=1, color='black')
    # axs[2, 1].plot(x[:-1]+d_x, G, linewidth=1, color='black')
    axs[1, 2].plot(x[:-1]+d_x, G, c='gray', alpha=0.5, linewidth=0.75)
    # axs[2, 3].plot(x[:-1]+d_x, G, c='gray', alpha=0.5, linewidth=0.75)
    axs[1, 2].plot(x[:-1]+d_x, G_pw, linewidth=1, color='black')
    # axs[2, 3].plot(x[:-1]+d_x, G_pw, linewidth=1, color='black')

    axs[0, 2].plot(x[:-1] + d_x, G_cvx1, linewidth=1, color=pu.excitatory_red)
    axs[0, 2].plot(x[:-1] + d_x, G_cvx2, linewidth=1, color=pu.inhibitory_blue)
    # axs[2, 1].plot(x[:-1] + d_x, G_cvx3, linewidth=1, color=pu.excitatory_red)
    # axs[2, 1].plot(x[:-1] + d_x, G_cvx4, linewidth=1, color=pu.inhibitory_blue)

    axs[1, 2].plot(x[1:-1] + d_x, G_E_x, color=pu.excitatory_red, linewidth=1)
    axs[1, 2].plot(x[1:-1] + d_x, G_I_x, color=pu.inhibitory_blue, linewidth=1)

    # plot curvatures
    axs[0, 3].plot(x[1:-1], C, linewidth=1, color='black')
    # axs[3, 1].plot(x[1:-1], C, linewidth=1, color='black')
    axs[1, 3].plot(x[1:-1], C, c='gray', alpha=0.5, linewidth=0.75)
    # axs[3, 3].plot(x[1:-1], C, c='gray', alpha=0.5, linewidth=0.75)
    # axs[1, 3].plot(x[1:-1], C_pw, linewidth=1, color='black')
    # axs[3, 3].plot(x[1:-1], C_pw, linewidth=1, color='black')

    axs[0, 3].plot(x[1:-1], C_cvx1, linewidth=1, color=pu.excitatory_red)
    axs[0, 3].plot(x[1:-1], C_cvx2, linewidth=1, color=pu.inhibitory_blue)
    # axs[3, 1].plot(x[1:-1], C_cvx3, linewidth=1, color=pu.excitatory_red)
    # axs[3, 1].plot(x[1:-1], C_cvx4, linewidth=1, color=pu.inhibitory_blue)

    # axs[1, 3].plot(x[1:-1], C_E_x, color=pu.excitatory_red, linestyle=':', linewidth=1)
    # axs[1, 3].plot(x[1:-1], C_I_x, color=pu.inhibitory_blue, linestyle=':', linewidth=1)
    x_p = 0
    axs[1, 3].plot([0, 10], [0, 0], color='black', linewidth=1.5)
    axs[1, 3].plot([0, 10], [0, 0], color=pu.excitatory_red, linewidth=1.0)
    axs[1, 3].plot([0, 10], [0, 0], color=pu.inhibitory_blue, linewidth=0.5)
    up_down = [1, -1, 1]
    for i, x_r in enumerate(x_roots[1:-1]):
        axs[1, 3].arrow(x_r, 0, 0, up_down[i]*2.5, head_width=0.5, head_length=0.2, width=0.05)
        if i == 1:
            axs[1, 3].arrow(x_r, 0, 0, 2, head_width=0.5, head_length=0.2, width=0.01,
                            color=pu.inhibitory_blue)
        else:
            axs[1, 3].arrow(x_r, 0, 0, 2, head_width=0.5, head_length=0.2, width=0.01,
                            color=pu.excitatory_red)


    # ax4a.plot(dddx, dC)
    # ax4b.plot(dddx, dC_pw)
    # ax4a.plot(x_roots, np.zeros_like(x_roots), '.')
    # ax1a.plot(x_roots,B_roots,'-',linewidth=0.5)
    # ax1.plot(x,B_pw,'.-')

    # for ax in [ax3a, ax4a, ax3b, ax4b]:
    #    ax.axhline(y=0, c='gray', alpha=0.5)

    fmt = mpl.ticker.StrMethodFormatter("{x: .0f}")
    for i in range(2):
        for j in range(4):
            ax = axs[i, j]
            ax.tick_params(axis='both', width=0.5, length=3, pad=1, labelsize=pu.fs1)
            sns.despine(ax=ax)
            axs[i, j].xaxis.set_major_formatter(fmt)
            axs[i, j].yaxis.set_major_formatter(fmt)
            # ax.set_xticks([])
            # ax.set_yticks([])
            axs[i, j].set_xticks([0, 5, 10])
            if i == 0:
                axs[i, j].set_xticklabels([])
            else:
                axs[i, j].set_xticklabels(['0', '5', '10'], fontsize=pu.fs1)
                axs[i, j].set_xlabel(r'$x$', fontsize=pu.fs2)
            if j > 1:
                ax.axhline(y=0, c='gray', linewidth=0.5, alpha=0.5)

    # axs[0, 0].set_ylabel(r'$f(x)$', fontsize=pu.fs2)
    # axs[1, 0].set_ylabel(r'$f_{PL}(x)$', fontsize=pu.fs2)
    # axs[0, 1].set_ylabel(r'$g(x), h(x)$', fontsize=pu.fs2)
    # axs[1, 1].set_ylabel(r'$g_{PL}(x), h_{PL}(x)$', fontsize=pu.fs2)
    # axs[0, 1].set_ylabel(r'$\partial f(x), \partial g(x), \partial h(x)$', fontsize=pu.fs2)
    # axs[1, 1].set_ylabel(r'$\partial f_{PL}(x), \partial g_{PL}(x), \partial h_{PL}(x)$', fontsize=pu.fs2)
    # axs[0, 2].set_ylabel(r'$\partial^2 f(x), \partial^2 g(x), \partial^2 h(x)$', fontsize=pu.fs2)
    # axs[1, 2].set_ylabel(r'$\partial^2 f_{PL}(x), \partial^2 g_{PL}(x), \partial^2 h_{PL}(x)$', fontsize=pu.fs2)
    axs[0, 0].text(0.2, 0.6, r'$f(x)$', fontsize=pu.fs1, transform=axs[0, 0].transAxes)
    axs[1, 0].text(0.1, 0.7, r'$f_{PL}(x)$', fontsize=pu.fs1, transform=axs[1, 0].transAxes)
    axs[0, 1].text(0.55, 0.5, r'$q(x)$', fontsize=pu.fs1, transform=axs[0, 1].transAxes, color=pu.excitatory_red)
    axs[0, 1].text(0.775, 0.2, r'$p(x)$', fontsize=pu.fs1, transform=axs[0, 1].transAxes, color=pu.inhibitory_blue)
    axs[1, 1].text(0.25, 0.6, r'$q_{PL}(x)$', fontsize=pu.fs1, transform=axs[1, 1].transAxes, color=pu.excitatory_red)
    axs[1, 1].text(0.7, 0.3, r'$p_{PL}(x)$', fontsize=pu.fs1, transform=axs[1, 1].transAxes, color=pu.inhibitory_blue)

    axs[0, 2].text(0.1, 0.6, r"$f'(x)$", fontsize=pu.fs1, transform=axs[0, 2].transAxes)
    axs[1, 2].text(0.15, 0.1, r"$f_{PL}'(x)$", fontsize=pu.fs1, transform=axs[1, 2].transAxes)
    axs[0, 2].text(0.2, 0.1, r"$q'(x)$", fontsize=pu.fs1, transform=axs[0, 2].transAxes, color=pu.excitatory_red)
    axs[0, 2].text(0.4, 0.3, r"$p'(x)$", fontsize=pu.fs1, transform=axs[0, 2].transAxes, color=pu.inhibitory_blue)
    axs[1, 2].text(0.05, 0.7, r"$q_{PL}'(x)$", fontsize=pu.fs1, transform=axs[1, 2].transAxes, color=pu.excitatory_red)
    axs[1, 2].text(0.35, 0.9, r"$p_{PL}'(x)$", fontsize=pu.fs1, transform=axs[1, 2].transAxes, color=pu.inhibitory_blue)

    axs[0, 3].text(0.15, 0.2, r"$f''(x)$", fontsize=pu.fs1, transform=axs[0, 3].transAxes)
    axs[1, 3].text(0.15, 0.2, r"$f_{PL}''(x)$", fontsize=pu.fs1, transform=axs[1, 3].transAxes)
    axs[0, 3].text(0.4, 0.85, r"$q''(x)$", fontsize=pu.fs1, transform=axs[0, 3].transAxes, color=pu.excitatory_red)
    axs[0, 3].text(0.1, 0.55, r"$p''(x)$", fontsize=pu.fs1, transform=axs[0, 3].transAxes, color=pu.inhibitory_blue)
    axs[1, 3].text(0.75, 0.35, r"$q_{PL}''(x)$", fontsize=pu.fs1, transform=axs[1, 3].transAxes, color=pu.excitatory_red)
    axs[1, 3].text(0.35, 0.95, r"$p_{PL}''(x)$", fontsize=pu.fs1, transform=axs[1, 3].transAxes, color=pu.inhibitory_blue)


def main(save_pdf=False, show_plot=True):

    # set up figure and subplots
    f, axs = plt.subplots(nrows=2, ncols=4, figsize=(4.5, 2.25), dpi=150)

    plot_function_non_monotonic(axs)

    # call plotting functions
    # f.subplots_adjust(wspace=4, hspace=1.5)
    f.tight_layout()  # tightening layout before 3d plot works better

    # add panel labels
    axs[0, 0].text(-0.3, 1.05, r'\textbf{a}', transform=axs[0, 0].transAxes, **pu.panel_prms)
    axs[0, 1].text(-0.3, 1.05, r'\textbf{b}', transform=axs[0, 1].transAxes, **pu.panel_prms)
    axs[0, 2].text(-0.3, 1.05, r'\textbf{c}', transform=axs[0, 2].transAxes, **pu.panel_prms)
    axs[0, 3].text(-0.3, 1.05, r'\textbf{d}', transform=axs[0, 3].transAxes, **pu.panel_prms)
    axs[1, 0].text(-0.3, 1.05, r'\textbf{e}', transform=axs[1, 0].transAxes, **pu.panel_prms)
    axs[1, 1].text(-0.3, 1.05, r'\textbf{f}', transform=axs[1, 1].transAxes, **pu.panel_prms)
    axs[1, 2].text(-0.3, 1.05, r'\textbf{g}', transform=axs[1, 2].transAxes, **pu.panel_prms)
    axs[1, 3].text(-0.3, 1.05, r'\textbf{h}', transform=axs[1, 3].transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_s3_difference_of_convexity.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
