import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import fractional_matrix_power as fmp

class Affine_Maps:
    """
    Defines collection of affine maps. 
    """
    def __init__(self,A,b):
        """
        Sets the affine maps according to collection of matrices and vectors where any given map
        is defined by T_i(x) = A_i * x + b_i.
        Inputs:
            A: set of matrices that define each of the n affine maps (n x d x d array)
            b: set of vectors that define each of the n affine maps (n x d array) 
        """
        self.A = A
        self.b = b
        
        #also give it the number of maps and dimension as attributes
        n, d = np.shape(b)
        self.n = n
        self.d = d

    def apply(self,i,x):
        """
        Applies the i-th affine map to the point x.
        Inputs:
            i: the index of the affine map to be used (int from 0 to n-1)
            x: the point to which the map should be applied (1 x d array)
        """
        #applies the ith map to the point x
        y = self.A[i,:,:] @ x + self.b[i,:]
        return y

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

    def map(self,Affine,i):
        """
        Maps all the points in the collection under a given affine transformation.
        Inputs:
            Affine: a collection of affine transformations (Affine_Maps)
            i: the index of the affine map to be applied
        Returns:
            Mapped: the points in self under the i-th affine map (Model_Points)
        """
        map_x = np.zeros([self.num,self.d])
        for k in range(self.num):
            map_x[k,:] = Affine.apply(i,self.pick(k))
        Mapped = Model_Points(map_x)
        return Mapped


def Alpha_Maps(f):
    """
    Computes the optimal transport maps from each component of f to the intermediate Gaussian 
    distribution with mean zero vector and identity covariance matrix.
    Inputs:
        f: the GMM that is being mapped from (generally the prior) (Gaussian_Mixture)
    Outputs:
        Alpha: the collection of n affine maps that optimally map the corresponding component (Affine_Maps)
    """
    #so given a mixture, find each of the n alpha maps
    n = f.n
    d = f.d
    
    A = np.zeros([n,d,d])
    b = np.zeros([n,d])
    
    for i in range(n):
        A[i,:,:] = fmp(f.cov[i,:,:],-1/2)
        b[i,:] = - A[i,:,:] @ f.m[i,:]
        
    Alpha = Affine_Maps(A,b)
    
    return Alpha

def Inv_Alpha_Maps(f):
    """
    Computes the optimal transport maps from the intermediate distribution (Gaussian with zero mean 
    and identity covariance) to the GMM defined by f.
    Inputs:
        f: the mixture that is being mapped to from the intermediate (Gaussian_Mixture)
    Outputs:
        Inv_Alpha: the n affine maps that take the intermediate Gaussian to each of the mixture
        components of f (Affine_Maps)
    """
    n = f.n
    d = f.d
    
    A_inv = np.zeros([n,d,d])
    b_inv = np.zeros([n,d])
    
    for i in range(n):
        A_inv[i,:,:] = fmp(f.cov[i,:,:],1/2)
        b_inv[i,:] = f.m[i,:]
    
    Inv_Alpha = Affine_Maps(A_inv,b_inv)
    
    return Inv_Alpha


def Partitioner(q,Inv_Beta,Posterior,x):
    # TODO Make the posterior and x a single object that is used for distribution evaluations
    # TODO and has pairs of points and evaluations at those points, can add or remove points
    # TODO with methods and map them according to affines
    """
    Returns the current estimate for each component based on the current beta affine maps.
    Inputs:
        q: estimated weighting vector for posterior components (array)
        Inv_Beta: estimated inverse beta maps (Affine_Maps)
        Posterior:
        x: points at which the posterior has been evaluated at (Model_Points)
    Outputs:
        Q: the partitioning functions that, when multiplying the mixture, gives the i-th
        component (array)
        part: the posterior evaluations separated into its estimated components
    """
    
    m = Inv_Beta.n #get the number of maps being used 
    Q = np.zeros([m,x.num]) #initialise the partition functions
    
    for j in range(m):
        y = x.map(Inv_Beta,j)
        Q[j,:] = q[j] * multivariate_normal.pdf(y.all,mean=np.zeros(x.d),cov=np.eye(x.d)) * np.linalg.det(Inv_Beta.A[j,:,:])
    
    mix = np.sum(Q,axis=0)
    part = np.zeros([m,x.num])

    for j in range(m):
        Q[j,:] /= mix
        part[j,:] = Posterior * Q[j,:]
    
    return Q, part


def Scaling_Factor(part,m_points,InvBM):
    """
    
    Inputs:
        part: each of the partitioned posterior evaluations (m x num points array)
        m_points: the model points at which the posterior has been evaluated (Model_Points)
        InvBM: the current estimate of the inverse beta maps (Affine_Maps)
    Returns:
        S: the scaling factor at each point for each inverse mapping (m x num points array)
        R: the residual of the scaling factor at each point from the mean (m x num points array)
    """
    #firstly get some of the key dimensions and shapes out of the inputs
    d = m_points.d
    m, num_x = np.shape(part)

    #want to initialise the scaling factors and their error
    S = np.zeros([m,num_x])
    R = np.zeros([m,num_x])

    #generate the reference distribution we are trying to backmap to
    ref = multivariate_normal(mean = np.zeros(d),cov = np.eye(d))

    #now loop through each of the partitions
    for j in range(m):
        #work out the backmapping locations
        backmap = m_points.map(InvBM,j)
        #now work out the scaling ratio with the reference normal for each along with the residual
        S[j,:] = part[j,:] / ref.pdf(backmap.all)
        R[j,:] = S[j,:] - np.mean(S[j,:])
    
    return S, R


def Gradients(S,InvBM,m_points):
    """
    Finds the derivative of the scaling ratio for each point with respect to the elements of the inverse beta
    transport map.
    Inputs:
        evid:
        InvBM: current estimate of inverse beta maps (Affine_Maps)
        m_points: the model coordinates for which the scaling ratio has been evaluated (Model_Points)
    Outputs:
        dR_dA: derivative of residual scaling factor at point with respect to affine matrix
        dR_db: derivative of residual scaling factor at point with respect to affine vector
    """
    #as opposed to the notebook, want to find the gradient for all maps as once
    #also make use of the inverse map for simpler gradient expressions
    #should be able to make this general for any dimension

    #some key values for the problem from the inputs
    m = InvBM.n
    d = InvBM.d

    #firstly, we will initialise the gradient arrays

    dS_dA = np.zeros([m,m_points.num,d,d])
    dS_db = np.zeros([m,m_points.num,d])

    mean_dS_dA = np.zeros([m,d,d])
    mean_dS_db = np.zeros([m,d])

    dR_dA = np.zeros([m,m_points.num,d,d])
    dR_db = np.zeros([m,m_points.num,d])


    #now we want to work with each component separately
    for j in range(m):
        #each component has a different backmapping
        backmap = m_points.map(InvBM,j)
        #now that we have all the backmapped points, want to see how these change with the map parameters
        for k in range(m_points.num):
            dS_dA[j,k,:,:] = S[j,k] * np.outer(backmap.pick(k),backmap.pick(k))
            dS_db[j,k,:] = S[j,k] * backmap.pick(k)
    
    #we also want the mean derivative
        mean_dS_dA[j,:,:] = np.sum(dS_dA[j,:,:],axis=0) / m_points.num
        mean_dS_db[j,:] = np.sum(dS_db[j,:],axis=0) / m_points.num

    #we now need to substract these from each of the original derivatives
    for j in range(m):
        for k in range(m_points.num):
            dR_dA[j,k,:,:] = dS_dA[j,k,:,:] - mean_dS_dA[j,:,:]
            dR_db[j,k,:] = dS_db[j,k,:] - mean_dS_db[j,:]
    
    return dR_dA, dR_db

def Map_Inversion(Maps):
    """
    Swaps between the forward and backward maps. Built on the assumption that the input maps are invertible.
    ! Note that this does not yet take advantage of finding the Cholesky decomposition of the affine matrix - should be able to use
    ! back or forward substitution to get the inverse more easily with the decomposition in hand
    """
    A = Maps.A
    b = Maps.b
    
    #get the number of maps and their dimension so we can start inverting them
    n, d = np.shape(b)
    
    A_Inv = np.zeros([n,d,d])
    b_Inv = np.zeros([n,d])
    
    for i in range(n):
        A_Inv[i,:,:] = np.linalg.inv(A[i,:,:])
        b_Inv[i,:] = - A_Inv[i,:,:] @ b[i,:]
        
    #now we can store these in the new set of inverted maps
    Inv_Maps = Affine_Maps(A_Inv,b_Inv)
    return Inv_Maps

def Least_Squares(dR_dA,dR_db,R):
    """
    Solves for the inverse mapping by minimising the L2 norm between the scaling ratio vector and its mean. This means it is attempting to make the
    scaling ratio constant across the domain.
    Inputs: 
        dR_dA:
        dR_db:
        R:
    Outputs:
        Delta:
    """
    m, num, d = np.shape(dR_db)
    Delta = np.zeros([m,d**2 + d])
    #now the rows of the matrix are the points while the columns are the mapping parameters

    #we will need to do this separately for each component
    for j in range(m):
        #lets initialise the matrix of gradients
        M = np.zeros([num,d**2 + d])
        #now we need to fill it so that rows are each data point and columns are each parameter
        for var in range(d**2):
            M[:,var] = dR_dA[j,:,var//d,var%d]
        for var in range(d):
            M[:,d**2 + var] = dR_db[j,:,var]

        Delta[j,:] = np.linalg.inv(M.T @ M) @ M.T @ R
    return Delta

def Weight_Gradients(q,InvBM,Posterior,m_points):
    """
    Takes the current inverse beta maps and posterior evaluations and determines the derivative of the evidence (as a function of the model)
    with respect to the posterior component weights.
    Inputs:
        q: current estimate of the posterior component weighting vector
        InvBM: current estimate of inverse beta maps (Affine_Maps)
        Posterior: posterior evaluations
        m_points: points in the model space at which the posterior was evaluated at (Model_Points)
    Outputs:
        ds_dq: derivative of the evidence at each point with respect to the components of the weighting vector q
    """

    m = InvBM.n
    d = InvBM.d

    #lets create our reference difstribution
    ref = multivariate_normal(mean=np.zeros(d),cov=np.eye(d))

    #now we will need some empty arrays to fill
    h = np.zeros([m,m_points.num])
    mix = np.zeros([m_points.num])
    scale = np.zeros([m,m_points.num])
    dQ_dq = np.zeros([m,m,m_points.num])
    ds_dq = np.zeros([m,m_points.num])
    dr_dq = np.zeros([m,m_points.num])

    #we will need to loop over each component
    for j in range(m):
        #compute the j backmapping and the reference
        back_j = m_points.map(InvBM,j)
        h[j,:] = ref.pdf(back_j.all) * np.linalg.det(InvBM.A[j,:,:])

        scale[j,:] = Posterior / (ref.pdf(back_j.all) * np.linalg.det(InvBM.A[j,:,:]))

        mix += q[j] * h[j,:]

    for j in range(m):
        for k in range(m):
            dQ_dq[j,k,:] =  - q[j] * ((h[j,:] * h[k,:]) / mix **2)
        dQ_dq[j,j,:] += (h[j,:] / mix)
        
    # now we need to combine this result with the scaling terms and add together to get the 
    # final derivatives to work with.

    for k in range(m):
        for j in range(m):
            ds_dq[k,:] += scale[j,:] * dQ_dq[j,k,:]
        
        dr_dq[k,:] = ds_dq[k,:] - np.mean(ds_dq[k,:])

    return dr_dq

def Refit_Mixture(dr_dq):
    """
    Adjusts the weights of the posterior mixture according to the new beta maps.
    """
    return