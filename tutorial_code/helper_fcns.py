import numpy as np

# generate data from a Gaussian process
def cov(params,kernel,x,x_star):
    if len(x) == len(x_star):                # if the data is square...
        cov_m = np.zeros((len(x),len(x_star)))
        for i in range(len(x)):
            for j in range(i+1,len(x_star)): # only need to compute upper triangular
                cov_m[i,j] = kernel(params,x[i],x_star[j])
        cov_m += cov_m.T                     # append the transpose to gain square matrix
        for i in range(len(x)):
            for j in range(len(x)):          # make sure diagonal is only added once 
                cov_m[i,j] = kernel(params,x[i],x_star[j])
        return cov_m  
    else:                                    # otherwise just produce full rectangular matrix 
        cov_m = np.zeros((len(x),len(x_star)))
        for i in range(len(x)):
            for j in range(len(x_star)):
                cov_m[i,j] = kernel(params,x[i],x_star[j])
    return cov_m 

def squared_exp(params,x1,x2):
    # Defining a covariance function 
    l = params[0]     # length scale
    sig = params[1]   # variance
    sig_n = params[2] # noise parameter 
    return (sig**2 * np.exp(-((x1-x2)**2/2*l**2)))+sig_n

def generate_random_smooth_function(l=1, sig=0.5, sig_n=1, xlims=[0, 10], seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    # generate random Gaussian function
    y_range = np.linspace(0,20,100)
    params = [l,sig,sig_n]
    mu_y = [0. for i in range(len(y_range))]
    cov_y = cov(params,squared_exp,y_range,y_range)
    y = np.random.multivariate_normal(mu_y,cov_y)
    
    # shift and normalize so that it works better
    # (generally, for best performance the function should be non-negative and values not too large, e.g., between 1 and 5 here)
    y -= np.min(y)
    y /= np.max(y)
    y = 1 + 4*y
    
    # assign this function to x values (I am arbitrarily choosing x between 0 and 10 by default
    x = np.linspace(xlims[0],xlims[1],y.shape[0]+1)[1:]
    
    return x, y


def diff_x(x):
    return (x + 0.5*(x[1]-x[0]))[:-1]

def find_numerical_roots(x,a):
    
    a0 = a[0]
    x0 = x[0]
    x_roots = []
    idx_roots = []
    for i in range(1,a.shape[0]):
        a1 = a[i]
        x1 = x[i]
        if a0*a1<0:
            x_roots.append(0.5*(x0+x1))
            idx_roots.append(i)
        a0 = a1
        x0 = x1
    return (x_roots,idx_roots)

def get_piecewise_fcn(x,f,x_roots):
    
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
    
    for i in range(1,len(x_roots)):
        
        x1 = x_roots[i]
        idx1 = np.argmin(np.abs(x-x1)) # find the closes value of x to x1
        y1 = f[idx1]
        
        if np.abs(y1-y0)>0.75 or i==len(x_roots)-1:
            y_roots.append(y1)
            xnew_roots.append(x1)
            y_pw[idx0:(idx1+1)] = np.linspace(y0,y1,idx1+1-idx0)
            idx0 = idx1
            y0 = y1

    return xnew_roots,y_roots,y_pw