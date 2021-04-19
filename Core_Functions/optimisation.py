from .gmm_base import GMM_Transport, Wasserstein_Gradients, Gaussian_Mixture
import numpy as np
from scipy.optimize import minimize

# def Line_Search(f,g,reg,num_iter=10):
#     f_k = f
#     g_k = g
#     GW2 = np.zeros(num_iter+1)

#     for k in range(1,num_iter+1):
#         GW2[k-1], P, alpha = GMM_Transport(f_k,g_k,reg)
#         dGW_dp, dGW_dmu, dGW_dsigma = Wasserstein_Gradients(f_k,g_k,P,alpha)

#         #now want to do gradient descent for a variety of step sizes
#         GW_a = np.zeros(20)
#         i = 0

#         for a in np.linspace(0,0.5,20):
#             adGW_dp = a * dGW_dp
#             adGW_dmu = a * dGW_dmu
#             adGW_dsigma = a * dGW_dsigma

#             f_k_a = Gaussian_Mixture(f.n,f.d)

#             f_k_a.assign_w(f_k.w - adGW_dp)
#             f_k_a.assign_m(f_k.m - adGW_dmu)
#             f_k_a.assign_cov(f_k.cov - adGW_dsigma)
            
#             # TODO need to do a check that the covariance matrix is valid - positive semidefinite?
#             # TODO if so, need to make GW_a[i] NaN and use nanargmin to get good step size 

#             GW_a[i] = GMM_Transport(f_k_a,g,reg)[0]

#             i += 1

#         step = np.argmin(GW_a)

        
# * To use the in-built optimisation, want to have source parameters in vector

def Flattener(f):
    """
    Converts the Gaussian_Mixture to a 1D array of its component parameters.
    Inputs:
        f: mixture model to be flattened (Gaussian_Mixture) 
    Outputs:
        f_flat: array of mixture model parameters (array)
        d: dimension of mixture model (int)
    """
    d = f.d
    w = f.w.flatten()
    mean = f.m.flatten()
    cov = f.cov.flatten()
    f_flat = np.concatenate((w,mean,cov))
    return f_flat, d

def Unflattener(f_flat,d):
    """
    Inverts Flattener by converting the 1D model parameter array into a Gaussian_Mixture.
    Inputs:
        f_flat: flattened model parameters (array)
        d: dimension of mixture model (int)
    Outputs:
        f: unflattened mixture model (Gaussian_Mixture)
    """
    n = np.size(f_flat) // (1 + d + d**2)

    f = Gaussian_Mixture(n,d)
    w = f_flat[:n]
    mean = f_flat[n:n+n*d].reshape([n,d])
    cov = f_flat[n+n*d:].reshape([n,d,d])

    f.assign_w(w)
    f.assign_m(mean)
    f.assign_cov(cov)

    return f

def GW_Flat(f_flat,g,reg,d):
    """
    Determines the Gaussian-Wasserstein distance between a flattened source distribution and target Gaussian_Mixture
    Inputs:
        f_flat: source model parameters as flattened array (array)
        g: target mixture (Gaussian_Mixture)
        reg: entropic regularisation parameter (float)
        d: dimension of the mixtures (int)
    Returns:
        GW: Gaussian-Wasserstein distance approximation under entropic regularisation (float)
    """
    f = Unflattener(f_flat,d)
    # * need to then use the normal calculation that it is a Gaussian_Mixture again
    GW = GMM_Transport(f,g,reg)[0]
    return GW

def dGW_Flat(f_flat,g,reg,d):
    """
    Determines the gradient vector of the entropically smoothed Gaussian-Wasserstein distance.
    Inputs:
        f_flat: source model parameters as flattened array (array)
        g: target mixture (Gaussian_Mixture)
        reg: entropic regularisation parameter (float)
        d: dimension of the mixtures (int)
    Returns:
        dGW_flat: derivative of Gaussian-Wasserstein with respect to each source model parameter in same order as f_flat (array)
    """
    f = Unflattener(f_flat,d)
    GW, P, alpha = GMM_Transport(f,g,reg)
    dGW_dp, dGW_dmu, dGW_dsigma = Wasserstein_Gradients(f,g,P,alpha)
    dGW_flat = np.concatenate((dGW_dp.flatten(),dGW_dmu.flatten(),dGW_dsigma.flatten()))
    return dGW_flat

def GMM_Inversion(f,g,reg):
    f_flat, d = Flattener(f)
    min_flat = minimize(GW_Flat,f_flat,args=(g,reg,d),method='BFGS',jac=dGW_Flat,options={'disp': True})
    #f_min = Unflattener(min_flat,d)
    return min_flat