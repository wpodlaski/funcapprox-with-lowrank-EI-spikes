import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import plot_utils as pu
import net_sim_code as scnf


def plot_noisy_EI_dynamics(ax1=None, ax2=None, ax3=None, ax4=None, ax5=None, ax6=None, seed=None):
    """
    Plots for panels (a,b,c,d) -- The noisy, homogeneous E-I network
    :param ax1: Figure axis for the E-I boundaries before the inhibitory perturbation.
    :param ax2: Figure axis for the E-I boundaries after the inhibitory perturbation.
    :param ax3: Figure axis for the latent readouts over time.
    :param ax4: Figure axis for the spike raster over time.
    :param ax5: Figure axis for the synaptic inputs over time.
    :param ax6: Figure axis for the voltages over time.
    :param seed: Random seed for the noise.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    # network parameters
    NE = 51
    NI = 50
    N2 = NE + NI
    JE = 1
    JI = 1
    Jx = 1

    D_E = 0.075 * np.ones((JE, NE))
    E_EE = 1.25 * np.ones((NE, JE))
    E_EI = np.ones((NE, JI))
    F_E = np.ones((NE, Jx))
    T_E = np.zeros((NE,))

    D_I = 0.15 * np.ones((JI, NI))
    E_II = np.ones((NI, JI))
    E_IE = 2. * np.ones((NI, JE))
    F_I = np.ones((NI, Jx))
    T_I = 1.5 * np.ones((NI,))

    # simulation parameters
    dt = 5e-5
    Tend2 = 0.25
    times2 = np.arange(0, Tend2, dt)
    nT2 = len(times2)
    x2 = 1. * np.ones((nT2,))[None, :]
    x2[0, :int(0.005/dt)] = 0.
    I_E = np.zeros_like(x2)
    I_I = np.zeros_like(x2)
    I_I[0, int(0.125/dt):] = 1.

    mu_E = 0.5
    sigma_vE = 0.15
    mu_I = 0.5
    sigma_vI = 0.15

    # change the boundary of a single excitatory neuron
    E_EE[50:, :] = 1.75
    T_E[50:] = 1.5

    s_E, s_I, V_E, V_I, _, _ = scnf.run_EI_spiking_net(x2, D_E, E_EE, E_EI, F_E, T_E,
                                                       D_I, E_IE, E_II, F_I, T_I, dt=dt,
                                                       mu_E=mu_E, mu_I=mu_I, I_E=I_E, I_I=I_I,
                                                       sigma_vE=sigma_vE, sigma_vI=sigma_vI)
    r_E = scnf.exp_filter(s_E, times2)
    r_I = scnf.exp_filter(s_I, times2)
    y_E = D_E @ r_E
    y_I = D_I @ r_I
    spk_times_E = scnf.get_spike_times(s_E, dt=dt)
    spk_times_I = scnf.get_spike_times(s_I, dt=dt)

    # calculate the inputs
    Vpos_E = E_EE @ D_E @ r_E + F_E @ x2
    Vneg_E = -E_EI @ D_I @ r_I
    Vsum_E = Vpos_E + Vneg_E

    # PLOTTING
    yE = np.linspace(0, 4, 21)
    alphas = np.ones((NE,))
    alphas[50:] = 0.5
    for i in [0, 50]:
        yI = (F_E[i, 0] * x2[0, int(0.025 / dt)] + E_EE[i, 0] * yE - T_E[i]) / E_EI[i, 0]
        ax1.plot(yE, yI, color=pu.excitatory_red, alpha=alphas[i], linewidth=0.75)
        ax2.plot(yE, yI, color=pu.excitatory_red, alpha=alphas[i], linewidth=0.75)
    i = 0
    yI = (F_I[i, 0] * x2[0, int(0.025 / dt)] + E_IE[i, 0] * yE - T_I[i]) / E_II[i, 0]
    ax1.plot(yE, yI, color=pu.inhibitory_blue, linewidth=0.75)
    ax2.plot(yE, yI, color=pu.inhibitory_blue, linewidth=0.5, linestyle='--')
    yI = (F_I[i, 0] * x2[0, int(0.025 / dt)] + E_IE[i, 0] * yE - T_I[i] + 1.0) / E_II[i, 0]
    ax2.plot(yE, yI, color=pu.inhibitory_blue, linewidth=0.75)

    ax1.plot(y_E[0, int(0.025 / dt):int(0.05 / dt)], y_I[0, int(0.025 / dt):int(0.05 / dt)], c='black',
             alpha=0.75, linewidth=1)
    ax2.plot(y_E[0, int(0.15 / dt):int(0.175 / dt)], y_I[0, int(0.15 / dt):int(0.175 / dt)], c='black',
             alpha=0.75, linewidth=1)

    ax1.text(2.7, 6, r'\textbf{I}', c=pu.inhibitory_blue, fontsize=pu.fs1)
    ax1.text(3.6, 4.1, r'\textbf{E}', c=pu.excitatory_red, fontsize=pu.fs1)
    ax1.text(1.7, 1.0, r'\textbf{E}$_2$', c=pu.excitatory_red, fontsize=pu.fs1, alpha=0.5)

    ax1.text(0.2, 4, 'spiking\n sim.', c='black', fontsize=pu.fs1)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.15",
                shrinkA=0, shrinkB=0, color='black')
    ax1.annotate("", xy=(1.7, 3.5), xytext=(1.5, 4.4), arrowprops=prop)
    ax2.text(0.3, 5, 'stim. inh.\n pop', c=pu.inhibitory_blue, fontsize=pu.fs1)
    prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.15",
                shrinkA=0, shrinkB=0, color=pu.inhibitory_blue)
    ax2.annotate("", xy=(1.9, 4.6), xytext=(1.55, 5.5), arrowprops=prop)

    # define the expected traces
    y_E_trace = 0.71 * np.ones_like(times2)
    y_E_trace[:int(0.125 / dt)] = 1.9
    y_I_trace = 2.1 * np.ones_like(times2)
    y_I_trace[:int(0.125 / dt)] = 3.35

    # latent var readouts
    sim_E, = ax3.plot(times2, y_E[0, :], c=pu.excitatory_red, linewidth=0.75, alpha=0.5, label=r"$y^E$ sim.")
    sim_I, = ax3.plot(times2, y_I[0, :], c=pu.inhibitory_blue, linewidth=0.75, alpha=0.5, label=r"$|y^I|$ sim.")
    bndry_E, = ax3.plot(times2, y_E_trace, '--', c=pu.excitatory_red, alpha=1.0, linewidth=1., label=r"$y^E$ boundary")
    bndry_I, = ax3.plot(times2, y_I_trace, '--', c=pu.inhibitory_blue, alpha=1.0, linewidth=1.,
                        label=r"$|y^I|$ boundary")
    l1 = ax3.legend([sim_E, bndry_E], [r"$y^E$ sim.", r"$y^E$ boundary"], fontsize=pu.fs1, ncols=1, frameon=False,
                    handlelength=2, columnspacing=1, labelspacing=0.15, loc="lower left", bbox_to_anchor=(0.04, -0.11))
    _ = ax3.legend([sim_I, bndry_I], [r"$|y^I|$ sim.", r"$|y^I|$ boundary"], fontsize=pu.fs1, ncols=1, frameon=False,
                   handlelength=2, columnspacing=1, labelspacing=0.15, loc="lower left", bbox_to_anchor=(0.55, 0.5))
    ax3.add_artist(l1)

    # spike raster
    for i in range(NE):
        ax4.plot(spk_times_E[i], i * np.ones_like(spk_times_E[i]), '.', color=pu.excitatory_red, markersize=1)
    for i in range(NI):
        ax4.plot(spk_times_I[i], NE + i * np.ones_like(spk_times_I[i]), '.', color=pu.inhibitory_blue,
                 markersize=1)  # cmap[i])

    # voltage (of a single neuron)
    n1 = np.random.choice(np.arange(50))
    n2 = 50  # + np.random.choice(np.arange(50))
    V1 = V_E[n1, :]
    V1[s_E[n1, :].astype(bool)] = 2.
    V2 = V_E[n2, :]
    V2[s_E[n2, :].astype(bool)] = 2.
    ax5.plot(times2, Vpos_E[n1, :], color=pu.excitatory_red, linewidth=0.75)
    ax5.plot(times2, Vneg_E[n1, :], color=pu.inhibitory_blue, linewidth=0.75)
    ax5.plot(times2, Vsum_E[n1, :], color='gray', linewidth=0.75)
    ax6.plot(times2, V_E[n1, :], color=pu.excitatory_red, linewidth=0.5)
    ax6.plot(times2, V_E[n2, :], color=pu.excitatory_red, linewidth=0.5, alpha=0.5)
    ax6.plot(times2, np.ones_like(times2), '--', c='black', linewidth=0.5)

    ax6.text(0.15, 0.175, r'\textbf{E}', c=pu.excitatory_red, fontsize=pu.fs1)
    ax6.text(0.1, -0.15, r'\textbf{E}$_2$', c=pu.excitatory_red, fontsize=pu.fs1, alpha=0.5)

    # formatting
    for ax in [ax1, ax2]:
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 7])
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([0, 3, 6])
        ax.set_xticklabels([0, 2, 4], fontsize=pu.fs1)
        ax.set_yticklabels([0, 3, 6], fontsize=pu.fs1)
        ax.set_ylabel(r'Inhib. act. $|y^I|$', fontsize=pu.fs2)
    ax1.set_xticklabels([])
    ax2.set_xlabel(r'Excit. act. $y^E$', fontsize=pu.fs2)

    for ax in [ax3, ax4, ax5, ax6]:
        ax.set_xlim([0.02, Tend2])
        ax.set_xticks(np.linspace(0.02, Tend2 + 0.02, 6)[:-1])
        ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,Tend,6)])
    ax4.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, Tend2, 6)][:-1], fontsize=pu.fs1)
    ax6.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, Tend2, 6)][:-1], fontsize=pu.fs1)
    ax3.set_yticks([0, 2, 4])
    ax3.set_yticklabels(['0', '2', '4'], fontsize=pu.fs1)
    ax4.set_yticks([0, 50, 100, 150])
    ax4.set_yticklabels(['0', '50', '100', '150'], fontsize=pu.fs1)
    ax4.set_ylim([-0.5, N2])
    ax5.set_yticks([-5, 0, 5])
    ax5.set_yticklabels(['-5', '0', '5'], fontsize=pu.fs1)
    ax5.set_ylim([-5, 5])
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['0', '1'], fontsize=pu.fs1)
    ax6.set_ylim([-0.35, 1.75])
    ax3.set_ylabel('Latent vars.', fontsize=pu.fs2)  # , labelpad=7.25)
    ax4.set_ylabel('Neuron ID', fontsize=pu.fs2)  # , labelpad=1)
    ax5.set_ylabel('Syn. inputs', fontsize=pu.fs2)  # , labelpad=1)
    ax6.set_ylabel('Voltage', fontsize=pu.fs2)  # , labelpad=7.25)
    ax4.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    ax6.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()


def plot_mistuned_cross_intuition(ax):
    """
    Plots for panel (e), illustrating the readouts and orthogonal spaces and effects of mistuning in (r_1,r_2) space.
    :param ax: Figure axis for the (r_1,r_2) space plot.
    :return:
    """

    ax.plot([0, 4], [3, 0.6], color=pu.excitatory_red, alpha=1, linestyle=':')
    ax.plot([0.2, 3], [5-0.2*5/3, 0], color=pu.excitatory_red, alpha=0.5, linestyle=':')
    arrow_kws = dict(color=pu.excitatory_red, linewidth=1, head_length=0.35, head_width=0.35, zorder=1)
    ax.arrow(2, 2, 0.25 * 3, 0.25 * 5, **arrow_kws)
    ax.arrow(2, 2, 0.25 * 5, 0.25 * 3, **arrow_kws, alpha=0.5)
    ax.arrow(2, 2, -0.25 * 3, -0.25 * 5, **arrow_kws)
    ax.arrow(2, 2, -0.25 * 5, -0.25 * 3, **arrow_kws, alpha=0.5)

    # load spiking data from previous figure to use as illustration (not a real sim.)
    r = np.loadtxt('fig-08-stuff/noisy_rates.txt')
    r[0, :] *= 5. / 6
    r[1, :] *= 3. / 6
    r[0, r[0, :] > 4] = np.nan
    ax.plot(r[0, :], r[1, :]-0.5, c='black', alpha=0.25, linewidth=0.5)

    ax.set_ylim([0, 6])
    ax.set_xlim([0, 6])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    ax.text(1.5, 3.5, r'$EE$', fontsize=pu.fs1, color=pu.excitatory_red)
    ax.text(3.7, 2.5, r'$IE$', fontsize=pu.fs1, color=pu.excitatory_red, alpha=0.5)
    ax.text(2.75, 4, 'latent\n subspaces', fontsize=pu.fs1, color=pu.excitatory_red)
    ax.text(2.75, -1, 'latent\n nullspaces', fontsize=pu.fs1, color='black', alpha=0.5, clip_on=False)

    ax.arrow(-0.75, -0.75, 1, 0, color='black', linewidth=0.75, head_length=0.2, head_width=0.25,
             zorder=1, clip_on=False)
    ax.arrow(-0.75, -0.75, 0, 1, color='black', linewidth=0.75, head_length=0.2, head_width=0.25,
             zorder=1, clip_on=False)
    ax.text(-0.5, -1.3, r'$r_1$', fontsize=pu.fs2, color='black', clip_on=False)
    ax.text(-1.5, -0.4, r'$r_2$', fontsize=pu.fs2, color='black', rotation=90, clip_on=False)


def get_running_average(x, y, n_per_bin=10, n_jump=5, median=False):
    """
    Given pairs of values in (x,y) this generates a running average of the data by simple binning, either by computing
    the mean (plus standard deviation) or median (plus quartiles).
    :param x: x-axis values for the data (not assumed to be in order).
    :param y: y-axis values for the data.
    :param n_per_bin: number of data points used in each bin.
    :param n_jump: controls the amount of overlap between subsequent bins by how many points are jumped from one
        bin to the next.
    :param median: If True, then the median is calculated instead of the mean for each bin.
    :return: tuple of four arrays, (x_avg, y_avg, y_stdev_pos, y_stdev_neg), with mean (or median) values for x and y,
        plus the standard deviation (or quartiles) for each bin in the positive and negative directions.
    """

    # first sort data according to x
    inds = np.argsort(x)
    x_ = x[inds]
    y_ = y[inds]

    # compute running averages by binning the data
    x_avg = []
    y_avg = []
    y_stdev_pos = []
    y_stdev_neg = []
    i = 0
    done = False
    while not done:
        if median:
            x_avg.append(np.mean(x_[i:i + n_per_bin]))
            y_avg.append(np.median(y_[i:i + n_per_bin]))
            y_stdev_pos.append(np.quantile(y_[i:i + n_per_bin], 0.75))
            y_stdev_neg.append(np.quantile(y_[i:i + n_per_bin], 0.25))
        else:
            x_avg.append(np.mean(x_[i:i+n_per_bin]))
            y_avg.append(np.mean(y_[i:i+n_per_bin]))
            y_stdev_pos.append(np.std(y_[i:i+n_per_bin]))
            y_stdev_neg.append(np.std(y_[i:i+n_per_bin]))
        i += n_jump
        if (i+n_per_bin) > x_.shape[0]:
            done = True

    return np.array(x_avg), np.array(y_avg), np.array(y_stdev_pos), np.array(y_stdev_neg)


def plot_mistuned_cross_prmsweep(ax, use_saved_data=True):
    """
    Plots for panel (f) -- The mean squared error (MSE) as a function of decoder mistuning.
    :param ax: Figure axis for the E-I boundaries before the inhibitory perturbation.
    :param use_saved_data: If True, then this function will load pre-simulated data. If False, the simulation will run,
        but this may take a while.
    :return:
    """

    if use_saved_data:

        with open('fig-08-stuff/mistuned_cross_connection_data.pkl', 'rb') as file:
            mistuned_data = pickle.load(file)

        D_E_corr = mistuned_data['D_E_corr']
        D_I_corr = mistuned_data['D_I_corr']
        yE_err = mistuned_data['yE_err']
        yI_err = mistuned_data['yI_err']

    else:

        NE = 50
        NI = 50
        JE = 1
        JI = 1
        Jx = 1

        dt = 5e-5
        Tend2 = 0.12
        times2 = np.arange(0, Tend2, dt)
        nT2 = len(times2)
        x2 = 1. * np.ones((nT2,))[None, :]
        x2[0, :int(0.005 / dt)] = 0.

        mu_E = 0.5
        sigma_vE = 0.15
        mu_I = 0.5
        sigma_vI = 0.15

        E_EE = 1.25 * np.ones((NE, JE))
        E_EI = np.ones((NE, JI))
        F_E = np.ones((NE, Jx))
        T_E = np.zeros((NE,))

        E_II = np.ones((NI, JI))
        E_IE = 2. * np.ones((NI, JE))
        F_I = np.ones((NI, Jx))
        T_I = 1.5 * np.ones((NI,))

        sigma_vals = np.linspace(0, 0.5, 11)
        ntrials = 50

        D_E_corr = np.zeros((len(sigma_vals), ntrials))
        D_I_corr = np.zeros((len(sigma_vals), ntrials))
        yE_err = np.zeros((len(sigma_vals), ntrials))
        yI_err = np.zeros((len(sigma_vals), ntrials))

        for i in range(len(sigma_vals)):
            for j in range(ntrials):

                print(i, j)

                # define decoders
                sigma_DE = sigma_vals[i]
                sigma_DI = sigma_vals[i]
                D_E_E = 0.075 * (np.ones((JE, NE)) + sigma_DE * np.random.normal(size=(JE, NE)))
                D_E_I = 0.075 * (np.ones((JE, NE)) + sigma_DE * np.random.normal(size=(JE, NE)))
                D_I_I = 0.15 * (np.ones((JI, NI)) + sigma_DI * np.random.normal(size=(JI, NI)))
                D_I_E = 0.15 * (np.ones((JI, NI)) + sigma_DI * np.random.normal(size=(JI, NI)))
                D_E_E[D_E_E < 0.] = 0.
                D_E_I[D_E_I < 0.] = 0.
                D_I_I[D_I_I < 0.] = 0.
                D_I_E[D_I_E < 0.] = 0.

                s_E, s_I, V_E, V_I, _, _ = scnf.run_EI_spiking_net(x2, D_E_E, E_EE, E_EI, F_E, T_E,
                                                                   D_I_I, E_IE, E_II, F_I, T_I, dt=dt,
                                                                   mu_E=mu_E, mu_I=mu_I,
                                                                   sigma_vE=sigma_vE, sigma_vI=sigma_vI,
                                                                   D_EI=D_I_E, D_IE=D_E_I)
                r_E = scnf.exp_filter(s_E, times2)
                r_I = scnf.exp_filter(s_I, times2)
                y_E = D_E_E @ r_E
                y_I = D_I_I @ r_I

                D_E_corr[i, j] = 180./np.pi * np.arccos((np.dot(D_E_E[0, :], D_E_I[0, :])
                                                         / np.linalg.norm(D_E_E[0, :])
                                                         / np.linalg.norm(D_E_I[0, :])))
                D_I_corr[i, j] = 180./np.pi * np.arccos((np.dot(D_I_I[0, :], D_I_E[0, :])
                                                         / np.linalg.norm(D_I_I[0, :])
                                                         / np.linalg.norm(D_I_E[0, :])))
                yE_err[i, j] = np.mean((y_E[0, int(0.075/dt):] - 1.9) ** 2)
                yI_err[i, j] = np.mean((y_I[0, int(0.075/dt):] - 3.35) ** 2)

        # save data
        mistuned_data = {'NE': NE, 'NI': NI, 'JE': JE, 'JI': JI, 'Jx': Jx, 'dt': dt, 'Tend2': Tend2,
                         'sigma_vals': sigma_vals, 'ntrials': ntrials,
                         'D_E_corr': D_E_corr, 'D_I_corr': D_I_corr,
                         'yE_err': yE_err, 'yI_err': yI_err}
        with open('fig-08-stuff/mistuned_cross_connection_data.pkl', 'wb') as file:
            pickle.dump(mistuned_data, file)

    avg_corr = 0.5 * D_E_corr.flatten() + 0.5 * D_I_corr.flatten()
    avg_err = 0.5 * yE_err.flatten() + 0.5 * yI_err.flatten()
    runavg_corr, runavg_err, runavg_std, runavg_std2 = get_running_average(avg_corr, avg_err,
                                                                           n_per_bin=50, n_jump=25, median=True)
    ax.plot(runavg_corr, runavg_err, color='black', linewidth=1, markersize=4)
    ax.fill_between(runavg_corr, runavg_std2, runavg_std, color='black', alpha=0.25)

    ax.set_xticks([0, 15, 30])
    ax.set_xticklabels(['0', '15', '30'], fontsize=pu.fs1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=pu.fs1)
    ax.set_ylabel('Avg. latent MSE', fontsize=pu.fs2)
    ax.set_xlabel(r'Avg. decoder angle ($^\circ$)', fontsize=pu.fs2)
    ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()


def plot_mistuned_cross_example(ax1, ax2, seed=None):
    """
    Plot for panel (g) -- Example simulation of the E-I network with mistuned decoders.
    :param ax1: Figure axis for the latent readouts over time.
    :param ax2: Figure axis for the spike raster over time.
    :param seed: Random seed for the noise.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    # network parameters
    NE = 50
    NI = 50
    N2 = NE + NI
    JE = 1
    JI = 1
    Jx = 1

    E_EE = 1.25 * np.ones((NE, JE))
    E_EI = np.ones((NE, JI))
    F_E = np.ones((NE, Jx))
    T_E = np.zeros((NE,))

    E_II = np.ones((NI, JI))
    E_IE = 2. * np.ones((NI, JE))
    F_I = np.ones((NI, Jx))
    T_I = 1.5 * np.ones((NI,))

    # define decoders
    sigma_DE = 0.25
    sigma_DI = 0.25
    D_E_E = 0.075 * (np.ones((JE, NE)) + sigma_DE * np.random.normal(size=(JE, NE)))
    D_E_I = 0.075 * (np.ones((JE, NE)) + sigma_DE * np.random.normal(size=(JE, NE)))
    D_I_I = 0.15 * (np.ones((JI, NI)) + sigma_DI * np.random.normal(size=(JI, NI)))
    D_I_E = 0.15 * (np.ones((JI, NI)) + sigma_DI * np.random.normal(size=(JI, NI)))
    D_E_E[D_E_E < 0.] = 0.
    D_E_I[D_E_I < 0.] = 0.
    D_I_I[D_I_I < 0.] = 0.
    D_I_E[D_I_E < 0.] = 0.

    # simulation parameters
    dt = 5e-5
    Tend2 = 0.12
    times2 = np.arange(0, Tend2, dt)
    nT2 = len(times2)
    x2 = 1. * np.ones((nT2,))[None, :]
    x2[0, :int(0.005 / dt)] = 0.
    I_E = np.zeros_like(x2)
    I_I = np.zeros_like(x2)

    mu_E = 0.5
    sigma_vE = 0.15
    mu_I = 0.5
    sigma_vI = 0.15

    s_E, s_I, V_E, V_I, _, _ = scnf.run_EI_spiking_net(x2, D_E_E, E_EE, E_EI, F_E, T_E,
                                                       D_I_I, E_IE, E_II, F_I, T_I, dt=dt,
                                                       mu_E=mu_E, mu_I=mu_I, I_E=I_E, I_I=I_I,
                                                       sigma_vE=sigma_vE, sigma_vI=sigma_vI,
                                                       D_EI=D_I_E, D_IE=D_E_I)
    r_E = scnf.exp_filter(s_E, times2)
    r_I = scnf.exp_filter(s_I, times2)
    y_E = D_E_E @ r_E
    y_I = D_I_I @ r_I
    spk_times_E = scnf.get_spike_times(s_E, dt=dt)
    spk_times_I = scnf.get_spike_times(s_I, dt=dt)

    D_E_corr = 180./np.pi * np.arccos(np.dot(D_E_E[0, :], D_E_I[0, :])
                                      / np.linalg.norm(D_E_E[0, :]) / np.linalg.norm(D_E_I[0, :]))
    D_I_corr = 180./np.pi * np.arccos(np.dot(D_I_I[0, :], D_I_E[0, :])
                                      / np.linalg.norm(D_I_I[0, :]) / np.linalg.norm(D_I_E[0, :]))
    yE_err = np.mean((y_E[0, int(0.075 / dt):] - 1.9) ** 2)
    yI_err = np.mean((y_I[0, int(0.075 / dt):] - 3.35) ** 2)

    # define the expected traces
    y_E_trace = 1.9 * np.ones_like(times2)
    y_I_trace = 3.35 * np.ones_like(times2)

    # latent var readouts
    _ = ax1.plot(times2, y_E[0, :], c=pu.excitatory_red, linewidth=0.75, alpha=0.5)
    _ = ax1.plot(times2, y_I[0, :], c=pu.inhibitory_blue, linewidth=0.75, alpha=0.5)
    ax1.plot(times2, y_E_trace, '--', c=pu.excitatory_red, alpha=1.0, linewidth=1.)
    ax1.plot(times2, y_I_trace, '--', c=pu.inhibitory_blue, alpha=1.0, linewidth=1.)

    # spike raster
    for i in range(NE):
        ax2.plot(spk_times_E[i], i * np.ones_like(spk_times_E[i]), '.', color=pu.excitatory_red, markersize=1)
    for i in range(NI):
        ax2.plot(spk_times_I[i], NE + i * np.ones_like(spk_times_I[i]), '.', color=pu.inhibitory_blue,
                 markersize=1)  # cmap[i])

    for ax in [ax1, ax2]:  # i in range(4):
        ax.set_xlim([0.02, Tend2])
        ax.set_xticks(np.linspace(0.02, Tend2 + 0.02, 6)[:-1])
        ax.set_xticklabels([])  # '%.2f'%x for x in np.linspace(0.0,Tend,6)])
    ax2.set_xticklabels(['%.0f' % (100. * x) for x in np.linspace(0.0, Tend2, 6)][:-1], fontsize=pu.fs1)
    ax1.set_yticks([0, 2, 4, 6])
    ax1.set_yticklabels(['0', '2', '4', '6'], fontsize=pu.fs1)
    ax2.set_yticks([0, 50, 100, 150])
    ax2.set_yticklabels(['0', '50', '100', '150'], fontsize=pu.fs1)
    ax2.set_ylim([-0.5, N2])
    ax1.set_ylabel('Latent vars.', fontsize=pu.fs2)  # , labelpad=7.25)
    ax2.set_ylabel('Neuron ID', fontsize=pu.fs2)  # , labelpad=1)
    ax2.set_xlabel(r'Time ($\tau$)', fontsize=pu.fs2)
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    sns.despine()
    return 0.5 * D_E_corr + 0.5 * D_I_corr, 0.5 * yE_err + 0.5 * yI_err


def main(save_pdf=False, show_plot=True):
    f = plt.figure(figsize=(4.95, 3), dpi=100)
    gs = f.add_gridspec(4, 5, height_ratios=[1, 1, 0.6, 0.6])

    # top three panels, for network schematic diagrams
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[1, 0])
    ax3 = f.add_subplot(gs[0, 1:3])
    ax4 = f.add_subplot(gs[1, 1:3])
    ax5 = f.add_subplot(gs[0, 3:])
    ax6 = f.add_subplot(gs[1, 3:])
    ax7 = f.add_subplot(gs[2:, 0])
    ax8 = f.add_subplot(gs[2:, 1:3])
    ax9 = f.add_subplot(gs[2, 3:])
    ax10 = f.add_subplot(gs[3, 3:])

    plot_noisy_EI_dynamics(ax1=ax1, ax2=ax2, ax3=ax3, ax4=ax4, ax5=ax5, ax6=ax6, seed=66)
    plot_mistuned_cross_intuition(ax7)
    plot_mistuned_cross_prmsweep(ax8, use_saved_data=True)
    corr, err = plot_mistuned_cross_example(ax9, ax10, seed=12)  # 12 is okay
    ax8.plot([corr], [1.0], '*', c='black', markersize=4)

    f.tight_layout()

    for ax in [ax1, ax3, ax5]:
        box = ax.get_position()
        box.y0 = box.y0 - 0.04
        box.y1 = box.y1 + 0.0
        ax.set_position(box)
    for ax in [ax2, ax4, ax6]:
        box = ax.get_position()
        box.y0 = box.y0 + 0.01
        box.y1 = box.y1 + 0.05
        ax.set_position(box)
    for ax in [ax3, ax4, ax5, ax6, ax9, ax10]:
        box = ax.get_position()
        box.x0 = box.x0 - 0.02
        box.x1 = box.x1 + 0.0
        ax.set_position(box)
    ax = ax7
    box = ax.get_position()
    box.x0 = box.x0 + 0.0
    box.x1 = box.x1 + 0.08
    ax.set_position(box)
    ax = ax8
    box = ax.get_position()
    box.x0 = box.x0 + 0.04
    box.x1 = box.x1 - 0.03
    ax.set_position(box)
    ax = ax9
    box = ax.get_position()
    box.y0 = box.y0 - 0.05
    box.y1 = box.y1 + 0.0
    ax.set_position(box)
    ax = ax10
    box = ax.get_position()
    box.y0 = box.y0 + 0.0
    box.y1 = box.y1 + 0.05
    ax.set_position(box)

    ax1.text(-0.75, 1.1, r'\textbf{a}', transform=ax1.transAxes, **pu.panel_prms)
    ax2.text(-0.75, 1.1, r'\textbf{b}', transform=ax2.transAxes, **pu.panel_prms)
    ax3.text(-0.2, 1.1, r'\textbf{c}', transform=ax3.transAxes, **pu.panel_prms)
    ax5.text(-0.2, 1.1, r'\textbf{d}', transform=ax5.transAxes, **pu.panel_prms)
    ax7.text(-0.35, 1.05, r'\textbf{e}', transform=ax7.transAxes, **pu.panel_prms)
    ax8.text(-0.3, 1.05, r'\textbf{f}', transform=ax8.transAxes, **pu.panel_prms)
    ax9.text(-0.3, 1.1, r'\textbf{g}', transform=ax9.transAxes, **pu.panel_prms)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/fig_08_noisy_E_I_net.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
