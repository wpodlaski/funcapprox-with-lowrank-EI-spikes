import numpy as np


def run_spiking_net(x, D, E, F, T, dt=1e-4, leak=100., mu=0.0, current_inj=None, sigma_v=0.0, pw=None, tref=0.,
                    y0=None, r0=None, V0=None, seed=None, recurrence=True):
    """
    Description here.
    :param x:
    :param D:
    :param E:
    :param F:
    :param T:
    :param dt:
    :param leak:
    :param mu:
    :param current_inj:
    :param sigma_v:
    :param pw:
    :param tref:
    :param y0:
    :param r0:
    :param V0:
    :param seed:
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    N = D.shape[1]  # number of neurons
    num_bins = x.shape[1]  # number of time bins

    # set recurrent weights
    # separate self connections, which are always fast
    W = E @ D
    W -= np.diag(np.diag(W))
    W_self = np.diag(np.diag(E @ D)) - mu * np.eye(N)

    if not recurrence:
        W = np.zeros_like(W)

    # set all thresholds to 1 and reintroduce bias input currents
    T0 = np.ones((N,))
    b = T0 - T

    # absolute refractory periods
    ref = np.ones((N,))

    # synaptic dynamics
    pwsteps = None
    if pw is not None:
        pwsteps = int(pw / (dt * 1000))
    trsteps = int(tref / (dt * 1000))

    # initialize main vectors
    s = np.zeros((N, num_bins))  # spikes
    V = np.zeros((N, num_bins))  # voltages
    g = np.zeros((N, num_bins))  # synaptic inputs
    if current_inj is None:  # external inputs (perturbations)
        current_inj = np.zeros_like(V)

    if y0 is not None:
        V[:, 0] = F @ x[:, 0] + E @ y0 + 1 - T
    elif r0 is not None:
        V[:, 0] = F @ x[:, 0] + (W + W_self) @ r0 - (T - 1)
    elif V0 is not None:
        V[:, 0] = V0

    # run the Euler method to solve the differential equations
    for t in range(1, num_bins):

        # calculate feedforward inputs, bias currents and external inputs
        c = (x[:, t] - x[:, t - 1]) / (dt * leak) + x[:, t - 1]
        ff_inputs = F @ c + b + current_inj[:, t - 1]

        # calculate recurrent inputs
        rec_inputs = (W @ g[:, t - 1] + W_self @ s[:, t - 1]) / (dt * leak)

        # generate noise inputs
        noise_E = np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)

        # update membrane potentials
        V[:, t] = V[:, t - 1] + dt * leak * (-V[:, t - 1] + ff_inputs + rec_inputs) + noise_E

        # Check for spiking
        if pw is None:  # one spike per timestep rule

            spiking_index = np.argmax(V[:, t] - T0)
            if V[spiking_index, t] >= T0[spiking_index]:
                g[spiking_index, t] = 1
                s[spiking_index, t] = 1
                V[spiking_index, t] = T0[spiking_index]

        else:  # slower synapses; multiple spikes per timestep

            spiking_indices = np.arange(N)[V[:, t] >= T0]
            if spiking_indices.size > 0:
                spiking_indices = spiking_indices[ref[spiking_indices] <= 0]
                g[spiking_indices, t:t + pwsteps] += 1. / pwsteps
                s[spiking_indices, t] = 1
                ref[spiking_indices] = trsteps

        ref -= 1

    return s, V, g


def run_exp_spiking_net(x, D, E, F, T, dt=1e-4, leak=100., mu=0.0, current_inj=None, sigma_v=0.0, leak_s=None, tref=0.,
                        y0=None, r0=None, V0=None, seed=None):
    """
    Description here.
    :param x:
    :param D:
    :param E:
    :param F:
    :param T:
    :param dt:
    :param leak:
    :param mu:
    :param current_inj:
    :param sigma_v:
    :param leak_s:
    :param tref:
    :param y0:
    :param r0:
    :param V0:
    :param seed:
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    N = D.shape[1]  # number of neurons
    num_bins = x.shape[1]  # number of time bins

    # set recurrent weights
    # separate self connections, which are always fast
    W = E @ D
    W -= np.diag(np.diag(W))
    W_self = np.diag(np.diag(E @ D)) - mu * np.eye(N)

    # set all thresholds to 1 and reintroduce bias input currents
    T0 = np.ones((N,))
    b = T0 - T

    # absolute refractory periods
    ref = np.ones((N,))
    trsteps = int(tref / (dt * 1000))

    # initialize main vectors
    s = np.zeros((N, num_bins))  # spikes
    V = np.zeros((N, num_bins))  # voltages
    g = np.zeros((N, num_bins))  # synaptic inputs
    if current_inj is None:  # external inputs (perturbations)
        current_inj = np.zeros_like(V)

    if y0 is not None:
        V[:, 0] = np.dot(-E, y0) + (1 - T)
    elif r0 is not None:
        V[:, 0] = F @ x[:, 0] + (W + W_self) @ r0 - (T - 1)
    elif V0 is not None:
        V[:, 0] = V0

    # run the Euler method to solve the differential equations
    for t in range(1, num_bins):

        # calculate feedforward inputs, bias currents and external inputs
        c = (x[:, t] - x[:, t - 1]) / (dt * leak) + x[:, t - 1]
        ff_inputs = F @ c + b + current_inj[:, t - 1]

        # calculate recurrent inputs
        rec_inputs = (1. / leak) * (W @ g[:, t - 1] + (1. / dt) * W_self @ s[:, t - 1])

        # generate noise inputs
        noise_E = np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)

        # update membrane potentials
        V[:, t] = V[:, t - 1] + dt * leak * (-V[:, t - 1] + ff_inputs + rec_inputs) + noise_E

        # Check for spiking
        if leak_s is None:  # one spike per timestep rule

            spiking_index = np.argmax(V[:, t] - T0)
            if V[spiking_index, t] >= T0[spiking_index]:
                g[spiking_index, t] = 1. / dt
                s[spiking_index, t] = 1

        else:  # slower synapses; multiple spikes per timestep

            # dynamics on g
            g[:, t] = g[:, t - 1] - dt * leak_s * g[:, t - 1]

            spiking_indices = np.arange(N)[V[:, t] >= T0]
            if spiking_indices.size > 0:
                spiking_indices = spiking_indices[ref[spiking_indices] <= 0]
                g[spiking_indices, t] += leak_s
                s[spiking_indices, t] = 1
                ref[spiking_indices] = trsteps

        ref -= 1

    return s, V, g


def run_EI_spiking_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I, T_I,
                       dt=1e-4, leak_E=100., leak_I=100., mu_E=0.0, mu_I=0.0,
                       I_E=None, I_I=None, sigma_vE=0.0, sigma_vI=0.0,
                       pw_E=None, pw_I=None, tref_E=0., tref_I=0.,
                       D_EI=None, D_IE=None, y0=None, r0=None, V0=None, seed=None):
    """
    Function to simulate a EI spiking network.
    :param x:
    :param D_E:
    :param E_EE:
    :param E_EI:
    :param F_E:
    :param T_E:
    :param D_I:
    :param E_IE:
    :param E_II:
    :param F_I:
    :param T_I:
    :param dt:
    :param leak_E:
    :param leak_I:
    :param mu_E:
    :param mu_I:
    :param I_E:
    :param I_I:
    :param sigma_vE:
    :param sigma_vI:
    :param pw_E:
    :param pw_I:
    :param tref_E:
    :param tref_I:
    :param D_EI:
    :param D_IE:
    :param seed:
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    NE = D_E.shape[1]  # number of E neurons
    NI = D_I.shape[1]  # number of I neurons
    num_bins = x.shape[1]  # number of time bins

    # mistuned cross-connections
    if D_EI is None:
        D_EI = D_I
    if D_IE is None:
        D_IE = D_E

    # set recurrent weights
    W_EE = E_EE @ D_E  # - mu_E*np.eye(NE)
    W_EI = -E_EI @ D_EI
    W_IE = E_IE @ D_IE
    W_II = -E_II @ D_I  # - mu_I*np.eye(NI)

    # separate self resets, which are always fast
    # W_EE -= np.diag(np.diag(W_EE))
    # W_EE_self = np.diag(np.diag(E_EE @ D_E)) - mu_E * np.eye(NE)
    W_EE_self = -mu_E * np.eye(NE)  # only make the self-reset part fast
    W_II -= np.diag(np.diag(W_II))
    W_II_self = np.diag(np.diag(-E_II @ D_I)) - mu_I * np.eye(NI)

    # set all thresholds to 1 and reintroduce bias input currents
    T_E0 = np.ones((NE,))
    T_I0 = np.ones((NI,))
    b_E = T_E0 - T_E
    b_I = T_I0 - T_I

    # absolute refractory periods
    ref_E = np.ones((NE,))
    ref_I = np.ones((NI,))

    # synaptic dynamics
    pwsteps_E = None
    pwsteps_I = None
    if pw_E is not None:
        pwsteps_E = int(pw_E / (dt * 1000))
    if pw_I is not None:
        pwsteps_I = int(pw_I / (dt * 1000))
    trsteps_E = int(tref_E / (dt * 1000))
    trsteps_I = int(tref_I / (dt * 1000))

    # initialize main vectors
    s_E = np.zeros((NE, num_bins))  # spikes
    V_E = np.zeros((NE, num_bins))  # voltages
    g_E = np.zeros((NE, num_bins))  # synaptic inputs
    s_I = np.zeros((NI, num_bins))  # spikes
    V_I = np.zeros((NI, num_bins))  # voltages
    g_I = np.zeros((NI, num_bins))  # synaptic inputs
    if I_E is None:  # external inputs (perturbations)
        I_E = np.zeros_like(V_E)
    if I_I is None:
        I_I = np.zeros_like(V_I)

    if y0 is not None:
        V_E[:, 0] = F_E @ x[:, 0] + E_EE @ y0['yE'] + E_EI @ y0['yI'] - (T_E - 1)
        V_I[:, 0] = F_I @ x[:, 0] + E_IE @ y0['yE'] + E_II @ y0['yI'] - (T_I - 1)
    elif r0 is not None:
        V_E[:, 0] = F_E @ x[:, 0] + E_EE @ D_E @ r0['rE'] + E_EI @ D_I @ r0['rI'] - (T_E - 1)
        V_I[:, 0] = F_I @ x[:, 0] + E_IE @ D_E @ r0['rE'] + E_II @ D_I @ r0['rI'] - (T_I - 1)
    elif V0 is not None:
        V_E[:, 0] = V0['VE']
        V_I[:, 0] = V0['VI']

    # run the Euler method to solve the differential equations
    for t in range(1, num_bins):

        # calculate feedforward inputs, bias currents and external inputs
        c_E = (x[:, t] - x[:, t - 1]) / (dt * leak_E) + x[:, t - 1]
        ff_inputs_E = F_E @ c_E + b_E + I_E[:, t - 1]
        c_I = (x[:, t] - x[:, t - 1]) / (dt * leak_I) + x[:, t - 1]
        ff_inputs_I = F_I @ c_I + b_I + I_I[:, t - 1]

        # calculate recurrent inputs
        rec_inputs_E = (W_EE @ g_E[:, t - 1] + W_EI @ g_I[:, t - 1] + W_EE_self @ s_E[:, t - 1]) / (dt * leak_E)
        rec_inputs_I = (W_IE @ g_E[:, t - 1] + W_II @ g_I[:, t - 1] + + W_II_self @ s_I[:, t - 1]) / (dt * leak_I)

        # generate noise inputs
        noise_E = np.sqrt(2 * dt * leak_E) * sigma_vE * np.random.randn(NE)
        noise_I = np.sqrt(2 * dt * leak_I) * sigma_vI * np.random.randn(NI)

        # update membrane potentials
        V_E[:, t] = V_E[:, t - 1] + dt * leak_E * (-V_E[:, t - 1] + ff_inputs_E + rec_inputs_E) + noise_E
        V_I[:, t] = V_I[:, t - 1] + dt * leak_I * (-V_I[:, t - 1] + ff_inputs_I + rec_inputs_I) + noise_I

        # Check for spiking
        if pw_E is None and pw_I is None:  # one spike per timestep rule; inhibition always wins over excitation

            spiking_index = np.argmax(V_I[:, t] - T_I0)
            if V_I[spiking_index, t] >= T_I0[spiking_index]:
                g_I[spiking_index, t] = 1
                s_I[spiking_index, t] = 1
            else:
                spiking_index = np.argmax(V_E[:, t] - T_E0)
                if V_E[spiking_index, t] >= T_E0[spiking_index]:
                    g_E[spiking_index, t] = 1
                    s_E[spiking_index, t] = 1

        else:  # slower synapses; multiple spikes per timestep, both I and E together

            # INHIB
            spiking_indices = np.arange(NI)[V_I[:, t] >= T_I0]
            if spiking_indices.size > 0:
                spiking_indices = spiking_indices[ref_I[spiking_indices] <= 0]
                g_I[spiking_indices, t:t + pwsteps_I] += 1. / pwsteps_I
                s_I[spiking_indices, t] = 1
                ref_I[spiking_indices] = trsteps_I

            # EXCIT
            spiking_indices = np.arange(NE)[V_E[:, t] >= T_E0]
            if spiking_indices.size > 0:
                spiking_indices = spiking_indices[ref_E[spiking_indices] <= 0]
                g_E[spiking_indices, t:t + pwsteps_E] += 1. / pwsteps_E
                s_E[spiking_indices, t] = 1
                ref_E[spiking_indices] = trsteps_E

        ref_I -= 1
        ref_E -= 1

    return s_E, s_I, V_E, V_I, g_E, g_I


def run_exp_EI_spiking_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I, T_I,
                           dt=1e-4, leak_E=100., leak_I=100., mu_E=0.0, mu_I=0.0,
                           I_E=None, I_I=None, sigma_vE=0.0, sigma_vI=0.0,
                           leak_s_E=None, leak_s_I=None, tref_E=0., tref_I=0.,
                           D_EI=None, D_IE=None, seed=None):
    """
    Function to simulate a EI spiking network.
    :param x:
    :param D_E:
    :param E_EE:
    :param E_EI:
    :param F_E:
    :param T_E:
    :param D_I:
    :param E_IE:
    :param E_II:
    :param F_I:
    :param T_I:
    :param dt:
    :param leak_E:
    :param leak_I:
    :param mu_E:
    :param mu_I:
    :param I_E:
    :param I_I:
    :param sigma_vE:
    :param sigma_vI:
    :param leak_s_E:
    :param leak_s_I:
    :param tref_E:
    :param tref_I:
    :param D_EI:
    :param D_IE:
    :param seed:
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    NE = D_E.shape[1]  # number of E neurons
    NI = D_I.shape[1]  # number of I neurons
    num_bins = x.shape[1]  # number of time bins

    # mistuned cross-connections
    if D_EI is None:
        D_EI = D_I
    if D_IE is None:
        D_IE = D_E

    # set recurrent weights
    W_EE = E_EE @ D_E  # - mu_E*np.eye(NE)
    W_EI = -E_EI @ D_EI
    W_IE = E_IE @ D_IE
    W_II = -E_II @ D_I  # - mu_I*np.eye(NI)

    # separate self connections, which are always fast
    W_EE -= np.diag(np.diag(W_EE))
    W_EE_self = np.diag(np.diag(E_EE @ D_E)) - mu_E * np.eye(NE)
    W_II -= np.diag(np.diag(W_II))
    W_II_self = np.diag(np.diag(-E_II @ D_I)) - mu_I * np.eye(NI)

    # set all thresholds to 1 and reintroduce bias input currents
    T_E0 = np.ones((NE,))
    T_I0 = np.ones((NI,))
    b_E = T_E0 - T_E
    b_I = T_I0 - T_I

    # absolute refractory periods
    ref_E = np.ones((NE,))
    ref_I = np.ones((NI,))
    trsteps_E = int(tref_E / (dt * 1000))
    trsteps_I = int(tref_I / (dt * 1000))

    # initialize main vectors
    s_E = np.zeros((NE, num_bins))  # spikes
    V_E = np.zeros((NE, num_bins))  # voltages
    g_E = np.zeros((NE, num_bins))  # synaptic inputs
    s_I = np.zeros((NI, num_bins))  # spikes
    V_I = np.zeros((NI, num_bins))  # voltages
    g_I = np.zeros((NI, num_bins))  # synaptic inputs
    if I_E is None:  # external inputs (perturbations)
        I_E = np.zeros_like(V_E)
    if I_I is None:
        I_I = np.zeros_like(V_I)

    # run the Euler method to solve the differential equations
    for t in range(1, num_bins):

        # calculate feedforward inputs, bias currents and external inputs
        c_E = (x[:, t] - x[:, t - 1]) / (dt * leak_E) + x[:, t - 1]
        ff_inputs_E = F_E @ c_E + b_E + I_E[:, t - 1]
        c_I = (x[:, t] - x[:, t - 1]) / (dt * leak_I) + x[:, t - 1]
        ff_inputs_I = F_I @ c_I + b_I + I_I[:, t - 1]

        # calculate recurrent inputs
        # rec_inputs_E = (W_EE @ g_E[:, t - 1] + W_EI @ g_I[:, t - 1] + W_EE_self @ s_E[:, t - 1]) / (dt * leak_E)
        # rec_inputs_I = (W_IE @ g_E[:, t - 1] + W_II @ g_I[:, t - 1] + + W_II_self @ s_I[:, t - 1]) / (dt * leak_I)
        rec_inputs_E = (1. / leak_E) * (W_EE @ g_E[:, t - 1] + W_EI @ g_I[:, t - 1]
                                        + (1. / dt) * W_EE_self @ s_E[:, t - 1])
        rec_inputs_I = (1. / leak_I) * (W_IE @ g_E[:, t - 1] + W_II @ g_I[:, t - 1]
                                        + (1. / dt) * W_II_self @ s_I[:, t - 1])

        # generate noise inputs
        noise_E = np.sqrt(2 * dt * leak_E) * sigma_vE * np.random.randn(NE)
        noise_I = np.sqrt(2 * dt * leak_I) * sigma_vI * np.random.randn(NI)

        # update membrane potentials
        V_E[:, t] = V_E[:, t - 1] + dt * leak_E * (-V_E[:, t - 1] + ff_inputs_E + rec_inputs_E) + noise_E
        V_I[:, t] = V_I[:, t - 1] + dt * leak_I * (-V_I[:, t - 1] + ff_inputs_I + rec_inputs_I) + noise_I

        # Check for spiking
        if leak_s_E is None and leak_s_I is None:  # one spike per timestep rule; inhibition always wins over excitation

            spiking_index = np.argmax(V_I[:, t] - T_I0)
            if V_I[spiking_index, t] >= T_I0[spiking_index]:
                g_I[spiking_index, t] = 1. / dt
                s_I[spiking_index, t] = 1
            else:
                spiking_index = np.argmax(V_E[:, t] - T_E0)
                if V_E[spiking_index, t] >= T_E0[spiking_index]:
                    g_E[spiking_index, t] = 1. / dt
                    s_E[spiking_index, t] = 1

        else:  # slower synapses; multiple spikes per timestep, both I and E together

            # dynamics on g
            g_I[:, t] = g_I[:, t - 1] - dt * leak_s_I * g_I[:, t - 1]
            g_E[:, t] = g_E[:, t - 1] - dt * leak_s_E * g_E[:, t - 1]

            # INHIB
            spiking_indices = np.arange(NI)[V_I[:, t] >= T_I0]
            if spiking_indices.size > 0:
                spiking_indices = spiking_indices[ref_I[spiking_indices] <= 0]
                g_I[spiking_indices, t] += leak_s_I
                s_I[spiking_indices, t] = 1
                ref_I[spiking_indices] = trsteps_I

            # EXCIT
            spiking_indices = np.arange(NE)[V_E[:, t] >= T_E0]
            if spiking_indices.size > 0:
                spiking_indices = spiking_indices[ref_E[spiking_indices] <= 0]
                g_E[spiking_indices, t] += leak_s_E
                s_E[spiking_indices, t] = 1
                ref_E[spiking_indices] = trsteps_E

        ref_I -= 1
        ref_E -= 1

    return s_E, s_I, V_E, V_I, g_E, g_I


def run_rate_net(x, D, E, F, T, dt=1e-4, leak=100., V0=None, r0=None, beta=1.):
    """

    :param x:
    :param D:
    :param E:
    :param F:
    :param T:
    :param dt:
    :param leak:
    :param V0:
    :param r0:
    :param beta:
    :return:
    """

    # initialize main arrays
    N = D.shape[1]  # number of neurons
    num_bins = x.shape[1]  # number of time bins
    V = np.zeros((N, num_bins))  # filtered spikes through leak
    r = np.zeros((N, num_bins))  # filtered spikes through leak
    W = E @ D

    if r0 is not None:
        r[:, 0] = r0
        V[:, 0] = F @ x[:, 0] + W @ r0 - (T - 1)
    if V0 is not None:
        V[:, 0] = V0

    # run the Euler method to solve the differential equations
    for t in range(1, num_bins):
        V[:, t] = V[:, t - 1] + dt * leak * (-V[:, t - 1] + F @ x[:, t - 1] + W @ r[:, t - 1])
        r[:, t] = 1. / (1. + np.exp(-beta * (V[:, t] - T)))

    return V, r


def run_EI_rate_net(x, D_E, E_EE, E_EI, F_E, T_E, D_I, E_IE, E_II, F_I, T_I,
                    dt=1e-4, leak_E=100., leak_I=100., beta=1., r_E0=None, r_I0=None):
    NE = D_E.shape[1]  # number of E neurons
    NI = D_I.shape[1]  # number of I neurons
    num_bins = x.shape[1]  # number of time bins

    # set recurrent weights
    W_EE = E_EE @ D_E  # - mu_E*np.eye(NE)
    W_EI = -E_EI @ D_I
    W_IE = E_IE @ D_E
    W_II = -E_II @ D_I  # - mu_I*np.eye(NI)

    # initialize main arrays
    V_E = np.zeros((NE, num_bins))  # filtered spikes through leak
    r_E = np.zeros((NE, num_bins))  # filtered spikes through leak
    V_I = np.zeros((NI, num_bins))  # filtered spikes through leak
    r_I = np.zeros((NI, num_bins))  # filtered spikes through leak

    if r_E0 is not None:
        r_E[:, 0] = r_E0
    if r_I0 is not None:
        r_I[:, 0] = r_I0

    V_E[:, 0] = (1. / beta) * np.log(
        r_E[:, 0] / (1 - r_E[:, 0])) + T_E  # F_E @ x[:, 0] + W_EE @ r_E[:, 0] + W_EI @ r_I[:, 0]
    V_I[:, 0] = (1. / beta) * np.log(
        r_I[:, 0] / (1 - r_I[:, 0])) + T_I  # F_I @ x[:, 0] + W_IE @ r_E[:, 0] + W_II @ r_I[:, 0]

    # run the Euler method to solve the differential equations
    for t in range(1, num_bins):
        V_E[:, t] = V_E[:, t - 1] + dt * leak_E * (
                    -V_E[:, t - 1] + F_E @ x[:, t - 1] + W_EE @ r_E[:, t - 1] + W_EI @ r_I[:, t - 1])
        V_I[:, t] = V_I[:, t - 1] + dt * leak_I * (
                    -V_I[:, t - 1] + F_I @ x[:, t - 1] + W_IE @ r_E[:, t - 1] + W_II @ r_I[:, t - 1])
        r_E[:, t] = 1. / (1. + np.exp(-beta * (V_E[:, t] - T_E)))
        r_I[:, t] = 1. / (1. + np.exp(-beta * (V_I[:, t] - T_I)))

    return r_E, r_I, V_E, V_I
