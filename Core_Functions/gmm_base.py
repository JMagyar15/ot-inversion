import numpy as np
import Core_Functions.sinkhorn as sh

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
        if np.size(w_arr) != self.n:
            raise Exception('number of weights and components are not equal')
        else:
            self.w = w_arr
    
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
                self.cov[i,j,:,:] = (np.eye(f.d) - f.cov[i,:,:]**(-1/2) @ (f.cov[i,:,:]**(1/2) 
                @ g.cov[j,:,:] @ f.cov[i,:,:]**(1/2))**(1/2) @ f.cov[i,:,:]**(-1/2))

    def dmu(self,i,j):
        """
        Derivative of the 2-Wasserstein distance between f_i and g_j with respect to each component of the f_i mean vector.
        Inputs:
            i: index of the source component
            j: index of the target component
        """
        return self.mean[i,j,:]

    def dsigma(self,i,j):
        """Derivative of the 2-Wasserstein distance between f_i and g_j with respect to each component of the f_i covariance matrix.
        Inputs:
            i: index of the source component
            j: index of the target component
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
    bures = np.trace(f.cov[0,:,:] + g.cov[0,:,:] - 2 * (f.cov[0,:,:]**(1/2) @ g.cov[0,:,:] @ f.cov[0,:,:]**(1/2))**(1/2))
    W2 = mean_dist + bures

    return W2


def Wasserstein_Cost(f,g):
    """
    Generates a cost matrix for Mixture-Wasserstein computations. Each row corresponds to an unweighted source component while the columns are the unweighted target components.
    Inputs:
        f: source Gaussian mixture (Gaussian_Mixture)
        g: target Gaussian mixture (Gaussian_Mixture)
    Outputs:
        W: transport cost between all component combinations for GMM transport (array)
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


def GMM_Transport(f,g,reg,num_iter=100,plan=False):
    # ! Dont like this much - change when doin log domain, just get out Wasserstein, gradient, and plan for derivs in one go
    # ! Should be able to do log domain stuff using POT and returining log to get the scaling vectors from dictionary
    # ? potential outputs are therefore GW2, dGW_dp, P --> dGW_dmu/dSigma later
    W = Wasserstein_Cost(f,g)
    Sink_Div = sh.Sinkhorn_Divergence(f.w,g.w,W,reg,num_iter=num_iter,plan=plan)
    return Sink_Div




    



