import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import fractional_matrix_power as fmp

class Affine_Maps:
    """
    Defines collection of affine maps. 
    """
    def __init__(self,L,b):
        """
        Sets the affine maps according to collection of matrices and vectors where any given map
        is defined by T_i(x) = A_i * x + b_i where A_i = L_i @ L_i^T
        Inputs:
            L: the set of lower triangular matrices that define each of the n affine maps (n x d x d array)
            b: set of vectors that define each of the n affine maps (n x d array) 
        """

        self.L = L
        self.b = b
        
        #also give it the number of maps and dimension as attributes
        n, d = np.shape(b)
        self.n = n
        self.d = d

    
        self.A = np.zeros([n,d,d])
        for j in range(n):
            self.A[j,:,:] = L[j,:,:] @ L[j,:,:].T

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
    
    def update(self,Delta):
        m = self.n
        d = self.d
        newL = np.zeros([m,d,d])
        newA = np.zeros([m,d,d])
        newb = np.zeros([m,d])

        Delta = Delta.reshape([m,d**2+d])

        for j in range(m):
            newL[j,:,:] = self.L[j,:,:] + Delta[j,:d**2].reshape([d,d])
            newb[j,:] = self.b[j,:] + Delta[j,d**2:].reshape([d])
        updated = Affine_Maps(newL,newb)
        return updated


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


def Partitioner(q,InvV,Posterior,m_points):
    """
    Returns the current estimate for each component based on the current beta affine maps.
    Inputs:
        q: estimated weighting vector for posterior components (array)
        InvV: estimated inverse V maps (Affine_Maps)
        Posterior: evaluations of the posterior weighted by the evidence (array)
        m_points: points at which the posterior has been evaluated at (Model_Points)
    Outputs:
        Q: the partitioning functions that, when multiplying the mixture, gives the i-th
        component (array)
        P: the posterior evaluations partitioned into its estimated components
        g_est: the current mixture approximation of the posterior distribution
    """
    
    m = InvV.n #get the number of maps being used 
    Q = np.zeros([m,m_points.num]) #initialise the partition functions
    
    for j in range(m):
        #backmap the points from the posterior to the intermediate
        backmap = m_points.map(InvV,j)
        #determine the current mixture using a change of variables
        det = InvV.L[j,:,:].diagonal().prod()**2
        Q[j,:] = q[j] * multivariate_normal.pdf(backmap.all,mean=np.zeros(m_points.d),cov=np.eye(m_points.d)) * det
    
    #now we have the total mixture
    g_est = np.sum(Q,axis=0)
    P = np.zeros([m,m_points.num])

    for j in range(m):
        #the partitioner can be found from these
        Q[j,:] /= g_est
        #apply the partitioner to the posterior evaluations to get the partitioned components
        P[j,:] = Posterior * Q[j,:]
    
    return Q, P, g_est

def Log_Scaling(P,q,InvV,m_points):
    """
    Determines the log ratio between the backmapped posterior evaluations and reference distribution for each backmapped point.
    Inputs:
        P: the partitioned posterior evaluations according to the current map estimates (m x N array)
        q: the current estimate of posterior component weights (m array)
        InvV: the current estimate of the backtracking maps (Affine_Maps)
        m_points: the points in the model space at which the posterior has been evaluated (Model_Points)
    Outputs:
        s: the log scaling ratio for each model point under each map (m x N array)
        r: the residual for each log ratio from the mean log ratio (m x N array)
    """
    m = InvV.n
    d = InvV.d
    ref = multivariate_normal(mean=np.zeros(d),cov=np.eye(d))
    s = np.zeros([m,m_points.num])
    r = np.zeros([m,m_points.num])

    for j in range(m):
        backtrack = m_points.map(InvV,j)
        det = InvV.L[j,:,:].diagonal().prod() ** 2
        s[j,:] = np.log(P[j,:]) - np.log(q[j]) - np.log(ref.pdf(backtrack.all)) - np.log(det)
        r[j,:] = s[j,:] - np.mean(s[j,:])
    return s, r

def Map_Gradients(InvV,m_points):
    #can firstly grab some of the array dimensions
    m = InvV.n
    d = InvV.d
    N = m_points.num

    #now use these to initialise the arrays
    ds_dL = np.zeros([m,d,d,N])
    ds_db = np.zeros([m,d,N])
    dM_dL = np.zeros([d,d,d,d])
    dy_dL = np.zeros([m,d,d,d])

    #we will also want the reference normal distribution
    ref = multivariate_normal(mean=np.zeros(d),cov=np.eye(d))

    for j in range(m):
        backtrack = m_points.map(InvV,j)
        for k in range(d):
            for l in range(d):
                for i in range(N):
                    ds_db[j,k,i] = backtrack.pick(i)[k]
                    for row in range(d):
                        for col in range(d):
                            dy_dL[j,l,k,l] += m_points.pick(i)[k] * dM_dL[j,row,col,k,l]
                    ds_dL[j,k,l,i] = np.inner(backtrack.pick(i),dy_dL[j,:,k,l])
    return ds_dL, ds_db

def Cholesky_Derivs(InvV,m_points):
    m = InvV.n
    d = InvV.d
    N = m_points.num

    dM_dL = np.zeros([m,d,d,d,d])

    for j in range(m):
        for k in range(d):
            for l in range(d):
                #take the l-th column of L and put it in row k
                dM_dL[j,k,l,k,:] += InvV.L[j,:,l]
                #take the k-th row of L and put it in row l
                dM_dL[j,k,l,l,:] += InvV.L[j,k,:] 
    return dM_dL

def Weight_Gradients():
    return