import numpy as np
import matplotlib.pyplot as plt
from Core_Functions import gmm_base as gb
from Core_Functions import gmm_plot as gp
from Core_Functions import optimisation as op
from scipy.linalg import eigh
from scipy.stats import multivariate_normal

def Update_H(H,s,y):
    n = np.size(y)
    I = np.eye(n)
    A = I - (np.outer(s,y) / np.inner(y,s))
    B = (np.outer(s,s)/ np.inner(y,s))
    new_H = (A @ H @ A) + B
    return new_H


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

    f = gb.Gaussian_Mixture(n,d)
    w = f_flat[:n]
    mean = f_flat[n:n+n*d].reshape([n,d])
    cov = f_flat[n+n*d:].reshape([n,d,d])

    f.assign_w(w)
    f.assign_m(mean)
    f.assign_cov(cov)

    return f

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
            #print('Projection applied!')
            return x + 1e-5
        else:
            i -= 1
    t_hat = (np.sum(p) - 1) / n
    x = p - t_hat
    x[x < 0] = 0
    #print('Projection applied!')
    return x + 1e-5

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
    f_proj = gb.Gaussian_Mixture(f.n,f.d)
    f_proj.assign_w(Project_Simplex(f.w))
    f_proj.assign_m(f.m)
    f_proj.assign_cov(Project_SemiPosDef(f.cov))
    return f_proj

#now we can implement a linear search method

def Linear_Steps(x, p, g, reg, a_max = 1, num_steps=20):
    steps = np.linspace(0,a_max,num_steps)
    GW = np.zeros(num_steps)

    i = 0
    for a in steps:
        x_a = x + a * p #get the parameters for this step size
        f_a = Unflattener(x_a,g.d) #turn them into a Gaussian_Mixture for computations
        f_a_p = Projection(f_a)
        #print(x_a)
        GW[i] = gb.GMM_Transport(f_a_p, g, reg)[0] #compute GW
        i += 1

    a_k = steps[np.nanargmin(GW)] #pick the step size that minimised GW
    return a_k, GW, steps






source = gb.Gaussian_Mixture(2,1)
target = gb.Gaussian_Mixture(2,1)

sw = np.array([0.5,0.5])
sm = np.zeros([2,1])
sc = np.zeros([2,1,1])

tw = np.array([0.7,0.3])
tm = np.zeros([2,1])
tc = np.zeros([2,1,1])

#assign some example values
sm[0,0] = -1
sm[1,0] = -4
sc[0,0,0] = 2
sc[1,0,0] = 5

tm[0,0] = 1
tm[1,0] = 5
tc[0,0,0] = 1.5
tc[1,0,0] = 3.2


source.assign_w(sw)
source.assign_m(sm)
source.assign_cov(sc)

target.assign_w(tw)
target.assign_m(tm)
target.assign_cov(tc)


x, d = Flattener(source)
n = np.size(x)
H = np.eye(n)
H[0,0] = 0
H[1,1] = 0


num_iter = 20
GW = np.zeros(num_iter)
fig, ax = plt.subplots(nrows=num_iter,ncols = 2,figsize=[10,30])

GW0, P, a = gb.GMM_Transport(source,target,0.01)
grad_mix = gb.Wasserstein_Gradients(source,target,P,a)
dGW, d = Flattener(grad_mix)

for k in range(num_iter):
    
    p = -(H @ dGW) #compute the search direction
    #p[0:target.n] *= 0.01 #damp the effect of the weighting gradients - they do too much!!!
    
    #now work out the step size with this direction
    alpha_k, GW_alpha, steps = Linear_Steps(x,p,target,0.01)
    ax[k,1].plot(steps,GW_alpha,'-k',linewidth=3)

    x_a = x + alpha_k * p #get the parameters for this step size
    f_a = Unflattener(x_a,target.d) #turn them into a Gaussian_Mixture for computations
    source = Projection(f_a)
    
    x_new, d = Flattener(source)
    
    #now want to update H
    GW[k], P, a = gb.GMM_Transport(source,target,0.01)
    grad_mix = gb.Wasserstein_Gradients(source,target,P,a)
    dGW_new, d = Flattener(grad_mix)
    y = dGW_new - dGW
    s = x_new - x
    H = Update_H(H,s,y)
    dGW = dGW_new
    x = x_new
    
    
    #---------------------------------------------------------------------------
    #do some plotting stuff
    #---------------------------------------------------------------------------
    domain = np.linspace(-10,10,100)
    f_d = np.zeros(100)
    g_d = np.zeros(100)

    for i in range(source.n):
        f_d += source.w[i] * multivariate_normal.pdf(domain, mean = source.m[i,:], cov = source.cov[i,:,:])

    for j in range(target.n):
        g_d += target.w[j] * multivariate_normal.pdf(domain, mean = target.m[j,:], cov = target.cov[j,:,:])


    ax[k,0].plot(domain,f_d,'-k',linewidth=3,label='Source')
    ax[k,0].plot(domain,g_d,'--k',linewidth=3,label='Target')