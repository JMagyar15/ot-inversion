import numpy as np
import ot
from scipy.linalg import fractional_matrix_power as fmp

class Gaussian_Mixture:
    """
    Creates a new Gaussian Mixture Model. 
    """
    def __init__(self,n,d):
        """
        Creates a new Guassian_Mixture object with n components in d dimensions.
        """
        self.n = n
        self.d = d
        
    
    def assign_w(self,w_arr):
        """
        Assigns the weights to the corresponding Gaussian components.
        Inputs:
            w_arr: vector of weights (1 x n numpy array)
        """
        if np.size(w_arr) != (self.n):
            raise Exception('number of weights and components are not equal')
        else:
            self.w = w_arr / (np.sum(w_arr))
    
    def assign_m(self,m_arr):
        """
        Assigns a mean vector to each of the Gaussian components.
        Inputs:
            m_arr: array of mean vectors. Rows correspond to Gaussian components,
            columns to each coordinate (n x d numpy array)
        """
        #check that rows = n, cols = d
        if np.shape(m_arr) != (self.n,self.d):
            raise Exception('mean vectors are of wrong dimension or number is not equal to number of components')
        else:
            self.m = m_arr

    def assign_cov(self,cov_arr):
        """
        Assigns the covariance matrices to each of the Gaussian components. 
        Inputs:
            cov_arr: array containing concatenated covariance matrices (n*d x d numpy array)
        """
        # TODO some sort of check that the matrix is positive definite
        if np.shape(cov_arr) != (self.n,self.d,self.d):
            raise Exception('covariance matrices are of wrong dimension or number is not equal to number of components')
        else:
            self.cov = cov_arr


class Analytical_Gradients:
    def __init__(self,f,g):
        if f.d != g.d:
            raise Exception('source and target distributions are of different dimension')
        
        self.mean = np.zeros([f.n,g.n,f.d])
        self.cov = np.zeros([f.n,g.n,f.d,f.d])
        for i in range(f.n):
            for j in range(g.n):
                self.mean[i,j,:] = 2 * (f.m[i,:] - g.m[j,:])

                # * split this into lots of parts to make sure it is right!
                A1 = fmp(f.cov[i,:,:],-0.5)
                A2 = fmp(f.cov[i,:,:],0.5)
                B = g.cov[j,:,:]
                C = fmp(A2 @ B @ A2,0.5)
                D = A1 @ C @ A1
                self.cov[i,j,:,:] = np.eye(f.d) - D

    def dmu(self,i,j):
        """
        Derivative of the 2-Wasserstein distance between f_i and g_j with respect to each component of the f_i mean vector.
        Inputs:
            i: index of the source component (int)
            j: index of the target component (int)
        """
        return self.mean[i,j,:]

    def dsigma(self,i,j):
        """Derivative of the 2-Wasserstein distance between f_i and g_j with respect to each component of the f_i covariance matrix.
        Inputs:
            i: index of the source component (int)
            j: index of the target component (int)
        """
        return self.cov[i,j,:,:]

def Analytical_Wasserstein(f,g):
    """
    Computes the analytical (exact) solution for the 2-Wasserstein distance between Gaussian distributions of any dimension. Note that while the inputs are Gaussian Mixture Models,
    these MUST have one component for the result to be correct.
    Inputs:
        f: source distribution (Gaussian_Mixture)
        g: target distribution (Gaussian_Mixture)
    Outputs:
        W2: exact 2-Wasserstein distance between f and g (float)
    """
    #add error here if both f and g are not 1 component Gaussian_Mixture

    mean_dist = np.linalg.norm(f.m[0,:] - g.m[0,:])**2
    A1 = f.cov[0,:,:]
    B = g.cov[0,:,:]
    A2 = fmp(A1,0.5)
    C = fmp(A2 @ B @ A2, 0.5)
    bures = np.trace(A1 - B - 2 * C)
    W2 = mean_dist + bures

    return W2


def Wasserstein_Cost(f,g):
    """
    Generates a cost matrix for Mixture-Wasserstein computations. Each row corresponds to an unweighted source component while the columns are the unweighted target components.
    Inputs:
        f: source Gaussian mixture (Gaussian_Mixture)
        g: target Gaussian mixture (Gaussian_Mixture)
    Outputs:
        W: transport cost between all component combinations for GMM transport (n x m array)
    """
    W = np.zeros([f.n,g.n])

    for i in range(f.n):
        f_i = Gaussian_Mixture(1,f.d)
        f_i.assign_m(np.reshape(f.m[i,:],[1,f.d]))
        f_i.assign_cov(np.reshape(f.cov[i,:,:],[1,f.d,f.d]))
        for j in range(g.n):
            g_j = Gaussian_Mixture(1,g.d)
            g_j.assign_m(np.reshape(g.m[j,:],[1,g.d]))
            g_j.assign_cov(np.reshape(g.cov[j,:,:],[1,f.d,f.d]))
            
            W[i,j] = Analytical_Wasserstein(f_i,g_j) 

    return W


def GMM_Transport(f,g,reg):
    """
    Using log-domain Sinkhorn iterations to solve the entropically regularised Gaussian mixture model OT problem.
    Inputs:
        f: source mixture (Gaussian_Mixture)
        g: target mixture (Gaussian_Mixture)
        reg: entropic regularisation parameter (float)
    Outputs:
        GW: 2-Wasserstein distance on space of GMMs (float)
        P: transport plan between weighting vectors of f and g (n x m array)
        alpha: log domain scaling vector rescaled by regularisation parameter (1 x n array)   
    """
    W = Wasserstein_Cost(f,g)
    # now do log domain OT to get plan and the dictionary
    P, log = ot.bregman.sinkhorn_stabilized(f.w,g.w,W,reg,log=True)
    alpha = reg * log['logu']
    beta = reg * log['logv']
    GW = np.sum(np.dot(f.w,alpha)) + np.sum(np.dot(g.w,beta))
    return GW, P, alpha


def Wasserstein_Gradients(f,g,P,alpha):
    """
    Determines the derivatives of the Gauss-Wasserstein distance with respect to each of the source mixture model parameters. 
    Inputs:
        f: source mixture (Gaussian_Mixture)
        g: target mixture (Gaussian_Mixture)
        P: optimal transport plan between weighting vectors (n x m array)
        alpha: scaling vector from GMM_Transport (1 x n array)
    Outputs:
        dGW_dp: derivative of GW with respect to each component weight (1 x n array)
        dGW_dmu: derivative of GW with respect to each mean vector (n x d array)
        dGW_dsigma: derivative of GW with respect to each covariance matrix (n x d x d array)
    """
    # w.r.t. weights is simple: the weights are the data so normal entropic regularisation result from dual problem.
    dGW_dp = alpha - np.mean(alpha)

    #now for the mean and covariance.
    dW = Analytical_Gradients(f,g)
    dGW_dmu = np.zeros([f.n,g.n,f.d])
    dGW_dsigma = np.zeros([f.n,g.n,f.d,f.d])
    for i in range(f.n):
        for j in range(g.n):
            dGW_dmu[i,j,:] = P[i,j] * dW.dmu(i,j)
            dGW_dsigma[i,j,:,:] = P[i,j] * dW.dsigma(i,j)

    #add up over each source component (chain rule)
    dGW_dmu = np.sum(dGW_dmu,axis=1)
    dGW_dsigma = np.sum(dGW_dsigma,axis=1)

    return dGW_dp, dGW_dmu, dGW_dsigma




    



