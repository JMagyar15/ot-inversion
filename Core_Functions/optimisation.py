from .gmm_base import GMM_Transport, Wasserstein_Gradients, Gaussian_Mixture
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


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
    # ! this doesn't work anymore - output of Wasserstein_Gradients is now a Gaussian_Mixture
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


def Project_Simplex(p):
    """
    Projects an arbitrary weighting vector onto the unit simplex. 
    Inputs:
        p: weighting vector to be projected onto simplex (array)
    Outputs:
        x: the vector p projected onto the unit simplex (array)
    """
    n = len(p)

    p_sort = np.sort(p)

    i = n - 1

    while i > 0:
        t_i = (np.sum(p_sort[i:]) - 1) / (n - i)

        if t_i > p_sort[i-1]:
            t_hat = t_i
            x = p - t_hat
            x[x < 0] = 0
            return x
        else:
            i -= 1
    t_hat = (np.sum(p) - 1) / n
    x = p - t_hat
    x[x < 0] = 0
    return x

def Project_SemiPosDef(cov):
    """
    Projects the updated symmetric covariance matrix candidates onto the cone of semi positive definite matrices.
    Inputs:
        cov: candidate covariance matrices (n x d x d array)
    Outputs:
        proj_cov: the matrices in cov projected onto cone of semi positive definite matrices (n x d x d array)
    """
    n = np.shape(cov)[0]
    proj_cov = np.zeros(np.shape(cov))
    for i in range(n):
        lamb, U = eigh(cov[i,:,:])
        lamb[lamb < 0] = 0
        proj_cov[i,:,:] = np.real(U @ np.diag(lamb) @ U.transpose())
    return proj_cov

def Projection(f):
    """
    Projects candidate mixture parameters onto the space of valid mixture models.
    Inputs:
        f: GMM with candidate parameters (Gaussian_Mixture)
    Outputs:
        f_proj: the candidate mixture with projected parameters (Gaussian_Mixture)
    """
    f_proj = Gaussian_Mixture(f.n,f.d)
    f_proj.assign_w(Project_Simplex(f.w))
    f_proj.assign_m(f.m)
    f_proj.assign_cov(Project_SemiPosDef(f.cov))
    return f_proj


def Search_Point(f_old,descent,alpha):
    """
    For a given descent direction and step size, updates the GMM via a projection onto the set of valid GMMs.
    Inputs:
        f_old: GMM at the previous iteration (Gaussian_Mixture)
        descent: search direction (Gaussian_Mixture)
        alpha: step size (float)
    Outputs:
        f_new_p: search point projected onto valid set (Gaussian_Mixture)
    """
    f_new = Gaussian_Mixture(f_old.n,f_old.d)
    f_new.assign_w(f_old.w - alpha * descent.w)
    f_new.assign_m(f_old.m - alpha * descent.m)
    f_new.assign_cov(f_old.cov - alpha * descent.cov)

    f_new_p = Projection(f_new)
    return f_new_p

def Linear_Steps(f_k, grad_f, g, reg, a_max = 1, num_steps=20, line = False):
    steps = np.linspace(0,a_max,num_steps)
    GW = np.zeros(num_steps)

    i = 0
    for a in steps:
        f_a = Search_Point(f_k, grad_f, a)
        GW[i] = GMM_Transport(f_a, g, reg)[0]
        i += 1

    a_k = steps[np.nanargmin(GW)]
    if line == False:
        return a_k
    else:
        return a_k, GW


def Backtracking():
    return

def Steepest_Descent(f,g,reg,num_iter=20):
    
    GW_arr = np.zeros(num_iter)

    for k in range(num_iter):
        #do the OT calculations
        GW_arr[k], P, alpha = GMM_Transport(f,g,0.01)
        grad = Wasserstein_Gradients(f,g,P,alpha)

        a_k = Linear_Steps(f, grad, g, reg)
        f = Search_Point(f,grad,a_k)
    
    return f, GW_arr


# def Gradient_Descent():
#     """
#     Computes the search direction via steepest descent. 
#     """
#     # ! Just needs to compute steepest descent direction - the rest is done in GMM_Optimise
#     return 

# def BFGS():
#     # ! Again, just needs to get the search direction, the rest will be done later
#     return

# def Line_Search():
#     # ! takes a search direction and performs a line search, returns minimum on this path
#     return

# def GMM_Optimise(f0,g,reg,method='Gradient_Descent',num_iter=20):
#     # ! just loop through the above functions in this master function
    
#     for k in range(num_iter):
    
#         if method == 'Gradient_Descent':
        
#         if method == 'BFGS':
#             # TODO add some initialisation of the Hessian - some multiple of identity?
#     return