import numpy as np
from scipy.stats import norm


def rect(x):
    x[x < 0] = 0
    return x


def solve_quad(a,b,c):
    if a == 0:
        return -c / b, None
    else:
        det = b ** 2 - 4 * a * c
        return (-b + np.sqrt(det)) / (2 * a), (-b - np.sqrt(det)) / (2 * a)


def single_nrn_boundary(x, E, F, T):
    if E == 0:
        raise Exception('Degenerate case: boundary must be calculated manually.')
    return (1. / E) * (T - F * x)


def old_boundary(x_lims, E, F, T):
    x = np.linspace(x_lims[0], x_lims[-1], 501)
    y = (1. / E) * (F * x - T)
    return x, y


# functions devoted to quadratic boundaries:
# the actual boundary function is E(x,y) = y - ax^2 - bx - c = 0
# but solving for y, we get: y = ax^2 - bx - c
def y_quad_fcn(a, b, c):
    return lambda x: a * x ** 2 + b * x + c


def ydx_quad_fcn(a, b):
    return lambda x: -2 * a * x - b


def y_3d_quad_fcn(a1, a2, a3, b1, b2, c):
    return lambda x1, x2: a1 * x1 ** 2 + a2 * x2 ** 2 + a3 * x1 * x2 + b1 * x1 + b2 * x2 + c


def ydx1_quad_fcn(a1, a3, b1):
    return lambda x1, x2: 2 * a1 * x1 + a3 * x2 + b1


def ydx2_quad_fcn(a2, a3, b2):
    return lambda x1, x2: 2 * a2 * x2 + a3 * x1 + b2


def exp_filter(s, times, leak=100., r0=None, dt=1e-4, incl_dt=False):
    if incl_dt:
        s *= dt
    r = np.zeros_like(s)
    if r0 is not None:
        r = r0[:, None] * np.tile(np.exp(-leak*times), (s.shape[0], 1))
    for n in range(s.shape[0]):
        r[n, :] += np.convolve(s[n, :], np.exp(-leak*times), mode='full')[:s.shape[1]]

    return r


def get_spike_times(s,dt=1e-4):
    spk_times = []
    for n in range(s.shape[0]):
        tmp = dt*np.where(s[n,:]==1)[0]
        spk_times.append(list(tmp))
    return spk_times


def get_ISIs(spk_times):
    isis = []
    for n in range(len(spk_times)):
        isis += list(np.diff(spk_times[n]))
    return np.array(isis)


def fcn_to_nrns(func, dfunc, N, x_range):
    xvals = np.linspace(x_range[0], x_range[1], N)
    # xvals = distribute_points_on_1d_curve(dfunc,xlims=x_range,npoints=N)
    yvals = func(xvals)
    E = np.ones_like(xvals)
    F = dfunc(xvals)
    T = F * xvals + E * yvals
    D = np.ones_like(xvals)
    return D, E, F, T, xvals, yvals


def old_fcn_to_nrns(func,dfunc,N,x_range):
    xvals = np.linspace(x_range[0], x_range[1], N)
    yvals = func(xvals)
    y_range = [min(0, np.min(yvals)), np.max(yvals)]
    dvals = dfunc(xvals)
    ivals = yvals - dvals*xvals
    F = dvals
    E = np.ones_like(xvals)
    D = np.ones_like(xvals)
    T = -ivals
    return D, E, F, T, xvals, yvals


# equally distribute along x-axes
def fcn2d_to_nrns(func, dx1func, dx2func, N, x1_range, x2_range, sigma=0., switch_order=False):
    m = int(np.floor(np.sqrt(N)))
    if switch_order:
        X2, X1 = np.meshgrid(np.linspace(x2_range[0],x2_range[1],m),
                             np.linspace(x1_range[0],x1_range[1],m))
    else:
        X1, X2 = np.meshgrid(np.linspace(x1_range[0],x1_range[1],m),
                             np.linspace(x2_range[0],x2_range[1],m))
    X1 = X1.flatten() + sigma*np.random.normal(size=(N,))
    X2 = X2.flatten() + sigma*np.random.normal(size=(N,))
    Y = func(X1,X2)
    E = np.ones_like(Y)
    F1 = dx1func(X1,X2)
    F2 = dx2func(X1,X2)
    T = F1*X1 + F2*X2 - Y
    D = np.ones_like(Y)
    return D, E, F1, F2, T, X1, X2, Y


def delta_kernel(val,xvals):
    tmp = np.zeros_like(xvals)
    idx = np.argmin(np.abs(xvals-val))
    tmp[idx] = 1.
    return tmp


def gauss_kernel(val, xvals, sigma=0):
    if sigma == 0:
        return delta_kernel(val, xvals)
    else:
        tmp = norm.pdf(xvals, loc=val, scale=sigma)
        tmp /= np.sum(tmp)
        return tmp


def interp_fcn_to_nrns(yI_func, dyE_func, dx_func, yE_range, x_range, N=50, redundancy=1, D_scale=0.1):
    # determine the x,yE values for each neuron
    if redundancy > 1:
        M = int(np.floor(1. * N / redundancy))
    else:
        M = N
    if (len(yE_range) == 1) and (len(x_range) == 1):
        x_vals = x_range[0] * np.ones((N,))
        yE_vals = yE_range[0] * np.ones((N,))
    elif (len(yE_range) == 1) and (len(x_range) == 2):
        x_vals = np.linspace(x_range[0], x_range[1], M)
        x_vals = np.repeat(x_vals, redundancy)
        yE_vals = yE_range[0] * np.ones_like(x_vals)
    elif (len(yE_range) == 2) and (len(x_range) == 1):
        yE_vals = np.linspace(yE_range[0], yE_range[1], M)
        yE_vals = np.repeat(yE_vals, redundancy)
        x_vals = x_range[0] * np.ones_like(yE_vals)
    elif (len(yE_range) == 2) and (len(x_range) == 2):
        m = int(np.floor(np.sqrt(M)))
        x_vals = np.linspace(x_range[0], x_range[1], m)
        yE_vals = np.linspace(yE_range[0], yE_range[1], m)
        x_vals, yE_vals = np.meshgrid(x_vals, yE_vals)
        x_vals = np.repeat(x_vals.flatten(), redundancy)
        yE_vals = np.repeat(yE_vals.flatten(), redundancy)
    else:
        raise Exception('Error: yE_range and/or x_range inputs not understood...')

    # set neuronal parameters
    yI_vals = np.zeros_like(x_vals)
    E_E = np.zeros_like(x_vals)
    F = np.zeros_like(x_vals)
    for i in range(len(x_vals)):
        yI_vals[i] = yI_func(x_vals[i], yE_vals[i])
        E_E[i] = dyE_func(x_vals[i], yE_vals[i])
        F[i] = dx_func(x_vals[i], yE_vals[i])
    E_I = np.ones_like(x_vals)
    T = E_E * yE_vals - E_I * yI_vals + F * x_vals
    D = D_scale * np.ones_like(x_vals)

    # deal with a different number of neurons
    # for now just repeat
    if E_E.shape[0] < N:
        extra = N - E_E.shape[0]
        E_I = np.append(E_I, np.array(extra * [E_I[-1]]))
        E_E = np.append(E_E, np.array(extra * [E_E[-1]]))
        F = np.append(F, np.array(extra * [F[-1]]))
        T = np.append(T, np.array(extra * [T[-1]]))
        D = np.append(D, np.array(extra * [D[-1]]))

    # deal with putting in the correct format
    E_E = E_E[:, None]
    E_I = E_I[:, None]
    F = F[:, None]
    D = D[None, :]

    return E_E, E_I, F, T, D


def new_interp_fcn_to_nrns(yI_func, dyE_func, dx_func, yE_range, x_range, N=50, redundancy=1, D_scale=0.1):
    # determine the x,yE values for each neuron
    if redundancy > 1:
        M = int(np.floor(1. * N / redundancy))
    else:
        M = N
    if (len(yE_range) == 1) and (len(x_range) == 1):
        x_vals = x_range[0] * np.ones((N,))
        yE_vals = yE_range[0] * np.ones((N,))
    elif (len(yE_range) == 1) and (len(x_range) == 2):
        x_vals = np.linspace(x_range[0], x_range[1], M)
        x_vals = np.repeat(x_vals, redundancy)
        yE_vals = yE_range[0] * np.ones_like(x_vals)
    elif (len(yE_range) == 2) and (len(x_range) == 1):
        yE_vals = np.linspace(yE_range[0], yE_range[1], M)
        yE_vals = np.repeat(yE_vals, redundancy)
        x_vals = x_range[0] * np.ones_like(yE_vals)
    elif (len(yE_range) == 2) and (len(x_range) == 2):
        m = int(np.floor(np.sqrt(M)))
        x_vals = np.linspace(x_range[0], x_range[1], m)
        yE_vals = np.linspace(yE_range[0], yE_range[1], m)
        x_vals, yE_vals = np.meshgrid(x_vals, yE_vals)
        x_vals = np.repeat(x_vals.flatten(), redundancy)
        yE_vals = np.repeat(yE_vals.flatten(), redundancy)
    else:
        raise Exception('Error: yE_range and/or x_range inputs not understood...')

    # set neuronal parameters
    yI_vals = np.zeros_like(x_vals)
    E_E = np.zeros_like(x_vals)
    F = np.zeros_like(x_vals)
    for i in range(len(x_vals)):
        yI_vals[i] = yI_func(np.array([x_vals[i], yE_vals[i]]))
        E_E[i] = dyE_func(np.array([x_vals[i], yE_vals[i]]))
        F[i] = dx_func(np.array([x_vals[i], yE_vals[i]]))
    E_I = np.ones_like(x_vals)
    T = E_E * yE_vals - E_I * yI_vals + F * x_vals
    D = D_scale * np.ones_like(x_vals)

    # deal with a different number of neurons
    # for now just repeat
    if E_E.shape[0] < N:
        extra = N - E_E.shape[0]
        E_I = np.append(E_I, np.array(extra * [E_I[-1]]))
        E_E = np.append(E_E, np.array(extra * [E_E[-1]]))
        F = np.append(F, np.array(extra * [F[-1]]))
        T = np.append(T, np.array(extra * [T[-1]]))
        D = np.append(D, np.array(extra * [D[-1]]))

    # deal with putting in the correct format
    E_E = E_E[:, None]
    E_I = E_I[:, None]
    F = F[:, None]
    D = D[None, :]

    return E_E, E_I, F, T, D
