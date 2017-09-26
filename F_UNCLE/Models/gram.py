'''Functions that implement the prior for the spline coefficients of
an isentrope for F_UNCLE

'''
import numpy as np

def k_g(
        tau,   # log of ratio of volumes
        alpha, # Characteristic length of correlation
        beta   # log of fractional uncertainty
):
    '''Kernel function in log-log coordinates is beta times a normalized
    Gaussian with width alpha.

    '''
    return beta/(alpha*np.sqrt(2*np.pi)) * np.exp(-tau**2/(2*alpha**2))

def mu_v(
        v,        # Volume (scalar float or np array)
        c=1.0,      # Constant
        gamma=3   # Exponent of density for pressure
):
    ''' Nominal pressure function in p,v coordinates
    '''
    return c/np.power(v,gamma)

def test_mu_v():
    '''Demonstrates that mu_v works on vectors
    '''
    v = np.linspace(1.0,10.0,20)
    p = mu_v(v)
    return(0)

def k_v(
        v,          # First specific volume
        v_,         # Second specific volume
        alpha=0.05,  # Characteristic length of correlation
        beta=0.05   # log of fractional uncertainty
):
    ''' Kernel function in p,v coordinates
    '''
    kg_tau = k_g(np.log(v/v_), alpha, beta)
    return mu_v(v)*mu_v(v_)*kg_tau

def inner_k(
        f,           # First function
        g,           # Second function
        k,           # Kernel
        v_min=1.0,   # Lower bound for integration
        v_max=10.0   # Upper bound for integration
):
    '''Calculate and return the inner product:

    <f|k|g> = \int_v_min^v_max dx \int_v_min^v_max dy f(x)k(x,y)g(y)
    '''
    import scipy.integrate
    def first(f,k,v):
        '''  \int_v_min^v_max f(x)k(x,v) dx
        '''
        return scipy.integrate.quad(lambda x: f(x)*k(x,v), v_min, v_max)[0]
    return scipy.integrate.quad(lambda y: first(f,k,y) * g(y), v_min, v_max)[0]

def gram(
        funcs,        # List of basis functions
        k,            # kernel
        v_min=1.0,
        v_max=10.0,
):
    '''Calculate and return the Gram matrix of the functions with the
    inner product:

    <f|k|g> = \int_v_min^v_max dx \int_v_min^v_max dy f(x)k(x,y)g(y)
    '''
    n = len(funcs)
    rv = np.empty((n,n))
    for i,f in enumerate(funcs):
        for j,g in enumerate(funcs):
            if j < i:  # Exploit symmetry because inner_k is slow
                rv[i,j] = rv[j,i]
            else:
                print(i, j)
                rv[i,j] = inner_k(f,g,k)
            # end
    return rv

#---------------
# Local Variables:
# eval: (python-mode)
# End:
