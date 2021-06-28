import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import multinomial
from scipy.linalg import fractional_matrix_power as fmp
from .gmm_base import Gaussian_Mixture
import ot

class Model_Points:
    """
    Defines a collection of points in the model space.
    """
    def __init__(self,x):
        self.all = x
        num_x, d = np.shape(x)
        self.num = num_x
        self.d = d
        
        
    def add(self,new_x):
        """
        Adds a collection of points to an existing Model_Points object.
        Inputs:
            new_x: the points that must be added to the collection (num x d)
        """
        # TODO add a check here that the new points have the same dimension as the old set
        add_x = np.concatenate((self.all,new_x),axis=0)
        self.all = add_x
        
        
    def pick(self,i):
        """
        Selects and returns the model point of a given index.
        Inputs:
            i: the index at which to pick the point
        Outputs:
            x_i: the point in the model space with index i
        """
        x_i = self.all[i,:]
        return x_i

def Evaluate_Mixture(f,m_points):
    """
    f: the mixture to be evaluated (Gaussian_Mixture)
    m_points: the points in the model space to evaluate the mixture (Model_Points)
    """
    pdf = np.zeros(m_points.num)
    for j in range(f.n):
        pdf += f.w[j] * multivariate_normal.pdf(m_points.all,mean=f.m[j,:],cov=f.cov[j,:,:])
    return pdf

def Sample_Mixture(f,num):
    """
    Inputs:
        f: the mixture to be sampled
        num: the number of points to sample from the mixture
    Outputs:
        m_points: the model points that have been selected (Model_Points)
    """
    #so we need a multinomial distribution for component selection
    m_all = []
    q_pick = multinomial(num,f.w).rvs(1)[0]
    for j in range(f.n):
        fj = multivariate_normal(mean=f.m[j,:],cov=f.cov[j,:,:])
        m_j = fj.rvs(q_pick[j]).reshape([q_pick[j],f.d])
        m_all.append(m_j)
    m_points = Model_Points(np.concatenate(m_all))
    return m_points

def Wasserstein(g,N,posterior=Evaluate_Posterior,reg=0.01):
    """
    Computes the discrete Wasserstein distance between the current mixture estimate and the posterior distribution 
    at a random set of points distributed according to the current estimate. Importance sampling is used to
    distribute these according to the posterior. Note that this will not match the real Wasserstein for a modest
    number of points when the estimate is well separated from the true posterior but will instead consider the tail
    segment.
    Inputs:
        g: the approximation of the posterior (Gaussian_Mixture)
        N: the number of points to evaluate the posterior at (int)
        posterior: a function that takes a model point and returns the unnormalised posteior likelihood (function)
        reg: the entropic regularisation parameter (float)
    Returns:
        W2: 2-Wasserstein distance (Sinkhorn approximation) for discrete transport between sampled points in mixture
        and posterior (float)
        alpha: left scaling vector from Sinkhorn iterations (N array)
        beta: right scaling vector from Sinkhorn iterations (N array)
    """
    #we want to randomly generate some points from the estimating mixture
    m_points = Sample_Mixture(g,N)

    mix_eval = Evaluate_Mixture(g,m_points)
    post_eval = posterior(m_points)

    #do the importance sampling
    p = mix_eval / mix_eval
    q = post_eval / mix_eval

    #now normalise these discrete densities
    p /= np.sum(p)
    q /= np.sum(q)

    print('Filling the cost matrix...')
    C = ot.dist(m_points.all,m_points.all)
    print('Cost matrix done, computing Wasserstein...')
    P, log = ot.bregman.sinkhorn_epsilon_scaling(p,q,C,reg,log=True)
    print('Wasserstein done, moving on to next task...')
    alpha = log['alpha']
    beta = log['beta']

    W2 = np.inner(p,alpha) + np.inner(q,beta)
    return W2, alpha, beta


def Gaussian_Derivs(f,m_points):
    """
    Computes the derivatives of the likelihood of a GMM at the points in m_points with respect to
    each parameter of the mixture
    Inputs:
        f: Gaussian mixture under consideration (Gaussian_Mixture)
        m_points: the points at which the derivatives should be computed (Model_Points)
    Outputs:
        dlnf_dw: derivative of the log-mixture with respect to the weights
        dlnf_dm: derivative of the log-mixture with respect to the mean vectors
        dlnf_dcov: derivative of the log-mixture with respect to the
        covariance matrices
    """
    #! the required derivatives can be found in the matrix cookbook
    n = f.n
    d = f.d
    N = m_points.num
    Q = np.zeros([n,N])
    dlnf_dw = np.zeros([n,N])
    dlnf_dm = np.zeros([n,d,N])
    dlnf_dcov = np.zeros([n,d,d,N])

    for j in range(n):
        Q[j,:] = f.w[j] * multivariate_normal.pdf(m_points.all,mean=f.m[j,:],cov=f.cov[j,:,:])
    
    sQ = np.sum(Q)

    for j in range(n):
        dlnf_dw[j,:] = (1 / f.w[j]) * (Q[j,:] / sQ)
        for k in range(d):
            dlnf_dm[j,:] = 
            for l in range(d):
                dlnf_dcov[j,k,l,:] = 

    return dlnf_dw, dlnf_dm, dlnf_dcov