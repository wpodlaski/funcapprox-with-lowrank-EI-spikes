import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf


def plot_random_inhibitory_boundary(ax1=None, ax2=None):
    """
    Plots panels (a) and (b), the input-output boundaries for a network of 10 inhibitory neurons with
    randomly-distributed parameters.
    :param ax1: Figure axis for the version with random E, F, T.
    :param ax2: Figure axis for the version with random E and F, with T tuned to the boundary.
    :return:
    """
    # boundary parameters: y = a*x^2 + b*x + c
    a = -1.0
    b = 0
    c = -0.5

    # get neuron parameters
    N = 10
    (D, E, F, T, xvals, yvals) = scnf.fcn_to_nrns(scnf.y_quad_fcn(a, b, c), scnf.ydx_quad_fcn(a, b), N, [-0.9, 0.9])
    print(np.min(E), np.max(E), np.mean(E), np.std(E))
    print(np.min(F), np.max(F), np.mean(F), np.std(F))
    print(np.min(T), np.max(T), np.mean(T), np.std(T))

    np.random.seed(19)  # 46
    E = 1 + 0.5 * np.random.normal(size=(N,))
    F = 1 * np.random.normal(size=(N,))
    T = 0.25 + 0.25 * np.random.normal(size=(N,))

    print(np.min(E), np.max(E), np.mean(E), np.std(E))
    print(np.min(F), np.max(F), np.mean(F), np.std(F))
    print(np.min(T), np.max(T), np.mean(T), np.std(T))

    x_range = [-1, 1]
    x_ideal = np.linspace(x_range[0], x_range[1], 101)
    y_ideal = scnf.y_quad_fcn(a, b, c)(x_ideal)
    ax2.plot(x_ideal, y_ideal, c='gray', alpha=0.5, linestyle=':', linewidth=2)

    # optimize thresholds
    T2 = np.zeros_like(T)
    for i in range(N):
        slope = -F[i] / E[i]
        xval = (slope - b) / 2 / a
        yval = a * xval ** 2 + b * xval + c
        T2[i] = F[i] * xval + E[i] * yval

    # plotting boundaries
    xlim = np.arange(-1, 1, 0.2)  # np.arange(-0.92, 0.95, 0.2)
    ylim = np.arange(-1.75, 0., 0.33)  # np.arange(-2.75, -0.1, 0.3)
    X, Y = np.meshgrid(xlim, ylim)
    u = 0
    v = -Y

    cmap = pu.inhibitory_cmap(np.linspace(0, 0.9, N))
    yb = np.zeros((N, 21))
    x = np.linspace(x_range[0], x_range[1], 21)

    # first one
    for i in range(N):
        y = scnf.single_nrn_boundary(x, E[i], F[i], T[i])
        yb[i, :] = y
        ax1.plot(x, y, linewidth=1, c=cmap[i], alpha=0.5)
        v[F[i] * X + E[i] * Y - T[i] > -0.2] = np.nan
    yb = np.minimum(np.min(yb, 0), np.zeros_like(x))
    ax1.plot(x, yb, c=pu.inhibitory_blue, linewidth=1)
    ax1.fill_between(x, np.ones_like(x), yb,
                     color=pu.inhibitory_blue, alpha=0.1)
    ax1.quiver(X, Y, u, v, color='gray', alpha=0.5, width=0.005, headlength=5, headwidth=5)

    # second one
    v = -Y
    yb = np.zeros((N, 21))
    for i in range(N):
        y = scnf.single_nrn_boundary(x, E[i], F[i], T2[i])
        yb[i, :] = y
        ax2.plot(x, y, linewidth=1, c=cmap[i], alpha=0.5)
        v[F[i] * X + E[i] * Y - T2[i] > -0.2] = np.nan
    yb = np.minimum(np.min(yb, 0), np.zeros_like(x))
    ax2.plot(x, yb, c=pu.inhibitory_blue, linewidth=1)
    ax2.fill_between(x, np.ones_like(x), yb, color=pu.inhibitory_blue, alpha=0.1)
    ax2.quiver(X, Y, u, v, color='gray', alpha=0.5, width=0.005, headlength=5, headwidth=5)

    # formatting
    for ax in [ax1, ax2]:
        ax.text(-0.06, -1.95, r'\textbf{subthreshold}', fontsize=pu.fs1)
        ax.text(-0.9, 0.3, r'\textbf{suprathreshold}', fontsize=pu.fs1,
                 color=pu.inhibitory_blue)
        ax.set_xlim(x_range)
        ax.set_ylim([-2, 0.5])
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=pu.fs1)
        ax.set_yticks([-2, -1, 0])
        ax.set_yticklabels(['-2', '-1', '0'], fontsize=pu.fs1)
        ax.set_xlabel('Input signal, $x$', fontsize=pu.fs2)
        ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.axhline(y=0.5, linewidth=1.0, color='black')
        ax.axhline(y=0, linewidth=0.5, color='gray', alpha=0.5)
        sns.despine(ax=ax, bottom=True)
    ax.set_ylabel('Latent variable, $y$', fontsize=pu.fs2, labelpad=7)


def main(save_pdf=False, show_plot=True):

    # set up figure and subplots
    f = plt.figure(figsize=(4.5, 1.65), dpi=150)
    gs = f.add_gridspec(1, 6)  # 0.025
    ax1 = f.add_subplot(gs[0, 1:3])
    ax2 = f.add_subplot(gs[0, 3:5])

    # call plotting functions
    plot_random_inhibitory_boundary(ax1=ax1, ax2=ax2)
    f.tight_layout()  # tightening layout before 3d plot works better

    # add panel labels
    ax1.text(-0.35, 1.075, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.35, 1.075, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_s1_random_boundary.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
