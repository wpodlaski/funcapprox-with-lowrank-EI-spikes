import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plot_utils as pu
from scipy.ndimage import gaussian_filter1d


def smooth_lif_io_curve(input_current, weight, bias, tau=0.02, tau_ref=0.005, threshold=20.0):
    """
    Generate the f-I curve of a LIF neuron, Gaussian smoothed (just because it looks nicer)
    :param input_current: input current values
    :param weight: input weight
    :param bias: input bias
    :param tau: membrane time constant (in seconds)
    :param tau_ref: refractory period (in seconds)
    :param threshold: voltage threshold
    :return: output firing rate, f (Hz)
    """
    tmp = 1 - threshold / (weight * input_current + bias)
    tmp[tmp <= 0] = 0.
    out = 1.0 / (tau_ref - tau * np.log(tmp))
    out[out <= 0.] = 0.
    out[np.isnan(out)] = 0.
    return gaussian_filter1d(out, 6)


def plot_fi_curves(ax1, ax2):
    """
    Plots panels (a) and (d), the LIF frequency-input (f-I) curves.
    :param ax1: Figure axis of the "rate code" panel
    :param ax2: Figure axis of the "spike code" panel
    :return:
    """
    a = 15.
    b = -10.
    input_current = np.linspace(0, 5, 101)
    frequency = smooth_lif_io_curve(input_current, a, b)

    titles = ['Rate code', '(Balanced) Spike code']
    for i, ax in enumerate([ax1, ax2]):
        ax.plot(input_current, frequency, 'k', linewidth=1)
        ax.axhline(y=0, c='gray', alpha=0.5, linewidth=0.5)
        # formatting
        ax.set_ylim([-20, 90])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Time-averaged\n output (Hz)', fontsize=pu.fs2)
        ax.set_xlabel('Time-averaged input', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_title(titles[i], fontsize=pu.fs3)

    p1 = Ellipse((2.5, 0), 5, 30, color='mediumseagreen', fill=None, linewidth=2)
    ax1.add_patch(p1)
    p2 = Ellipse((1.35, 0), 2., 30, color='mediumseagreen', fill=None, linewidth=2)
    ax2.add_patch(p2)
    ax1.text(0.25, 40.0, 'rate-based\n operating\n regime', fontsize=6, color='mediumseagreen')
    ax2.text(0.25, 40.0, 'balanced\n operating\n regime', fontsize=6, color='mediumseagreen')
    prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.3",
                shrinkA=0, shrinkB=0, color='mediumseagreen')
    ax1.annotate("", xy=(1.25, 20), xytext=(1., 32), arrowprops=prop)
    ax2.annotate("", xy=(1.05, 20), xytext=(1., 32), arrowprops=prop)


def simulate_lif_neuron(threshold=1., tau=1, dt=0.1, time=20., seed=213):
    """
    Simulate a single leaky integrate-and-fire neuron receiving white noise input.
    :param threshold: spiking threshold
    :param tau: membrane time constant
    :param dt: time step
    :param time: total simulation time
    :param seed: random seed for white noise
    :return: LIF voltage time series
    """
    np.random.seed(seed)
    time_steps = int(time / dt)
    x = 0.5 + 1.5 * np.random.normal(size=(time_steps,))
    V = np.zeros((time_steps,))
    V[0] = 0.7
    for t in range(1, time_steps):

        V[t] = V[t - 1] + (dt / tau) * (-V[t - 1] + x[t])

        if V[t] > threshold:
            V[t - 1] = 3.
            V[t] = 0.

    return V


def plot_spiking_inset(ax):
    """
    Plot a noisy LIF voltage to illustrate the "balanced" regime (inset for panel (d))
    :param ax: Figure axis where the inset will be located
    :return:
    """
    axin = inset_axes(ax, width="50%", height="50%", loc='lower right',
                      bbox_to_anchor=(0.2, -0.05, 1, 1), bbox_transform=ax.transAxes)

    V = simulate_lif_neuron()
    axin.plot(V, linewidth=0.5, color='black')
    p = FancyBboxPatch((3.4, -17.), 2.45, 60, boxstyle='round,pad=0.2', color='mediumseagreen', fill=None,
                       linewidth=1, linestyle=':', clip_on=False)
    ax.add_patch(p)

    axin.set_xticks([])
    axin.set_yticks([])
    sns.despine(ax=axin, left=True, bottom=True)


def plot_spike_responses(ax1, ax2):
    """
    Plots panels (b) and (e), three trials of spike trains, hand-designed.
    :param ax1: Figure axis for the "rate code" spike trains.
    :param ax2: Figure axis for the "balanced spike code" spike trains
    :return:
    """
    s1 = 0.01 * np.array([23, 33, 43, 53, 63, 73, 83, 93])
    s1b = 0.01 * (4 + np.array([23, 33, 43, 53, 63, 73, 83, 93]))
    s1c = 0.01 * (2 + np.array([23, 33, 43, 53, 63, 73, 83, 93]))

    s2 = 0.01 * np.array([23, 35, 43, 48, 61, 75, 96])
    s2b = 0.01 * np.array([25, 43, 66, 77, 80, 91])
    s2c = 0.01 * np.array([27, 35, 43, 47, 66, 69, 83, 96])

    ax1.eventplot(s1, color='black', lineoffsets=0.4, linelengths=0.3, linewidths=1.5)
    ax2.eventplot(s2, color='black', lineoffsets=0.4, linelengths=0.3, linewidths=1.5)
    ax1.eventplot(s1b, color='black', lineoffsets=0.85, linelengths=0.3, linewidths=1.5)
    ax2.eventplot(s2b, color='black', lineoffsets=0.85, linelengths=0.3, linewidths=1.5)
    ax1.eventplot(s1c, color='black', lineoffsets=1.3, linelengths=0.3, linewidths=1.5)
    ax2.eventplot(s2c, color='black', lineoffsets=1.3, linelengths=0.3, linewidths=1.5)
    for ax in [ax1, ax2]:
        p = Rectangle((0.21, 1.55), 1.5, 0.55, color='gray', alpha=0.25, linewidth=0)
        ax.add_patch(p)
        ax.set_ylim([0, 2])
        ax.set_xlim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Single-\n trial I/Os', fontsize=pu.fs2)
        ax.set_xlabel('Time', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.text(0.4, 1.6, 'stimulus on', color='black', fontsize=pu.fs2, alpha=0.5)


def plot_network_computation(ax1, ax2):
    """
    Plots panels (c) and (f), the "network computation" part.
    :param ax1: Figure axis for the "rate code" network computation.
    :param ax2: Figure axis for the "balanced spike code" network computation.
    :return:
    """
    input_current = np.linspace(0, 5, 101)
    input_weights = [10., -10., 10., -10.]
    input_biases = [0., 50., 15., 65.]
    frequencies = []
    for weight, bias in zip(input_weights, input_biases):
        frequencies.append(smooth_lif_io_curve(input_current, weight, bias))
    output_weights = np.array([2.5, -2.5, -2.5, 2.5])
    output = output_weights @ np.array(frequencies)

    clrs = ['blue', 'green', 'purple', 'orange']
    for i in range(4):
        ax1.plot(input_current, np.sign(output_weights[i]) * frequencies[i], clrs[i],
                 linewidth=1, alpha=0.25)
    ax1.plot(input_current, output, 'black', linewidth=1)
    ax1.axhline(y=0, c='gray', alpha=0.5, linewidth=0.5)

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Network\n output', fontsize=pu.fs2)
        ax.set_xlabel('Input', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax1.text(3.75, -62, 'n3', color='purple', alpha=0.5, fontsize=pu.fs1, fontweight='bold')
    ax1.text(0.8, 47, 'n4', color='orange', alpha=0.5, fontsize=pu.fs1, weight='bold')
    ax1.text(0.8, -75, 'n1', color='green', alpha=0.5, fontsize=pu.fs1, fontweight='bold')
    ax1.text(3.75, 63, 'n2', color='blue', alpha=0.5, fontsize=pu.fs1, weight='bold')
    ax2.text(0.44, 0.35, '?', fontsize=18, weight='bold')


def main(save_pdf=False, show_plot=True):

    # figure setup
    f = plt.figure(figsize=(4.5, 2.75), dpi=150)
    gs = f.add_gridspec(3, 4, height_ratios=[1, 0.5, 0.75], width_ratios=[1, 2, 2, 1])
    ax1 = f.add_subplot(gs[0, 1])
    ax2 = f.add_subplot(gs[0, 2])
    ax3 = f.add_subplot(gs[1, 1])
    ax4 = f.add_subplot(gs[1, 2])
    ax5 = f.add_subplot(gs[2, 1])
    ax6 = f.add_subplot(gs[2, 2])

    # call plotting functions
    plot_fi_curves(ax1, ax2)  # PANELS a,d
    plot_spike_responses(ax3, ax4)  # PANELS b,e
    plot_network_computation(ax5, ax6)  # PANELS c,f

    sns.despine()
    f.tight_layout()

    plot_spiking_inset(ax2)  # inset in panel d

    # panel labels
    ax1.text(-0.25, 1.15, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax3.text(-0.25, 1.1, r'\textbf{b}', transform=ax3.transAxes, **pu.panel_prms)
    ax5.text(-0.25, 1.1, r'\textbf{c}', transform=ax5.transAxes, **pu.panel_prms)
    ax2.text(-0.25, 1.15, r'\textbf{d}', transform=ax2.transAxes, **pu.panel_prms)
    ax4.text(-0.25, 1.1, r'\textbf{e}', transform=ax4.transAxes, **pu.panel_prms)
    ax6.text(-0.25, 1.1, r'\textbf{f}', transform=ax6.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_01_operating_regimes.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
