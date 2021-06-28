import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import multinomial
from scipy.linalg import fractional_matrix_power as fmp
from .gmm_base import Gaussian_Mixture


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
        
        #take the inputs and assign them to the mapping object
        self.L = L
        self.b = b
        
        #also give it the number of maps and dimension as attributes
        n, d = np.shape(b)
        self.n = n
        self.d = d

        
        #it is also of use to have the affine matrix so assign that now
        self.A = np.zeros([n,d,d])
        for j in range(n):
            #use the cholesky definition of the matrix L
            self.A[j,:,:] = L[j,:,:] @ L[j,:,:].T


    def apply(self,i,x):
        """
        Applies the i-th affine map to the point x.
        Inputs:
            i: the index of the affine map to be used (int from 0 to n-1)
            x: the point to which the map should be applied (d array)
        Outputs:
            y: the point x mapped under the i-th map (d array)
        """
        #applies the ith map to the point x
        y = self.A[i,:,:] @ x + self.b[i,:]
        return y
    
    
    def update(self,Delta):
        """
        Update the set of maps using the output of the optimisation algorithm
        Inputs:
            Delta: the proposed map updates from the optimisation (array)
        Outputs:
            updates: the set of maps updated according to Delta (Affine_Maps)
        """
        #get the required array dimensions from the inputs
        m = self.n
        d = self.d
        
        #initialise some arrays for the updates
        newL = np.zeros([m,d,d])
        newb = np.zeros([m,d])
        
        #want to reshape the update vector so that components are separated
        Delta = Delta.reshape([m,d**2+d])
        
        #loop through each of the maps and update separately
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


    def logGauss(self):
        """
        Determines the log-likelihood of the stardard normal at each of the points in self.
        Outputs:
            log: the log-likelihood of the stardard multivariate normal at the corresponding model point
        """
        #firstly initialise an array to store the values
        log = np.zeros([self.num])
        
        #now want to loop through each of the points in the collections
        for i in range(self.num):
            #get the point as an array
            point = self.pick(i)
            #key characteristic of standard normal: can treat as product of independent 1D normals
            log[i] = self.d - np.log(np.sqrt(2 * np.pi)) - 0.5 * np.sum(point**2)
        return log



def MapstoRef(f):
    """
    Computes the optimal transport maps from each component of f to the intermediate Gaussian 
    distribution with mean zero vector and identity covariance matrix.
    Inputs:
        f: the GMM that is being mapped from (generally the prior) (Gaussian_Mixture)
    Outputs:
        ToRef: the collection of n affine maps that optimally map the corresponding component (Affine_Maps)
    """
    #so given a mixture, find each of the n alpha maps
    n = f.n
    d = f.d
    
    A = np.zeros([n,d,d])
    L = np.zeros([n,d,d])
    b = np.zeros([n,d])
    
    for i in range(n):
        A[i,:,:] = fmp(f.cov[i,:,:],-1/2)
        L[i,:,:] = np.linalg.cholesky(A[i,:,:])
        b[i,:] = - A[i,:,:] @ f.m[i,:]
        
    ToRef = Affine_Maps(L,b)
    
    return ToRef

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

def InvMaptoMixture(InvV,q):
    m = InvV.n
    d = InvV.d
    
    A = np.zeros([m,d,d])
    b = np.zeros([m,d])
    
    fm = np.zeros([m,d])
    fcov = np.zeros([m,d,d])
    
    for j in range(m):
        A[j,:,:] = np.linalg.inv(InvV.A[j,:,:])
        b[j,:] = - A[j,:,:] @ InvV.b[j,:]
        fm[j,:] = b[j,:]
        fcov[j,:,:] = A[j,:,:] @ A[j,:,:].T
    
    f = Gaussian_Mixture(q,fm,fcov)
    
    return f

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

    for j in range(m):
        #the partitioner can be found from these
        Q[j,:] /= g_est
        #apply the partitioner to the posterior evaluations to get the partitioned components
    
    return Q

def Log_Scaling(Posterior,q,InvV,m_points):
    """
    Determines the log ratio between the backmapped posterior evaluations and reference distribution for each backmapped point.
    Inputs:
        Posterior: the posterior evaluations at each of the N points (N array)
        q: the current estimate of posterior component weights (m array)
        InvV: the current estimate of the backtracking maps (Affine_Maps)
        m_points: the points in the model space at which the posterior has been evaluated (Model_Points)
    Outputs:
        s: the log scaling ratio for each model point (N array)
        r: the residual for each log ratio from the mean log ratio (N array)
    """

    m = InvV.n #get the number of maps being used 
    d = InvV.d

    g_est = np.zeros(m_points.num)
    
    for j in range(m):
        #backmap the points from the posterior to the intermediate
        backmap = m_points.map(InvV,j)
        #determine the current mixture using a change of variables
        det = InvV.L[j,:,:].diagonal().prod()**2
        g_est += q[j] * multivariate_normal.pdf(backmap.all,mean=np.zeros(d),cov=np.eye(d)) * det
    
    #now we have the total mixture
    s = np.log(Posterior) - np.log(g_est)

    
    r = s - np.mean(s)
    misfit = 0.5 * np.linalg.norm(r)
    return s, r, misfit

def Cholesky_Derivs(InvV,m_points):
    """
    Computes the derivatives of the affine matrix for each map with respect to the elements of its Cholesky decomposition.
    Inputs:
        InvV: The set of backtracking affine maps (Affine_Maps)
        m_points: Collection of points at which posterior has been evaluated (Model_Points)
    Returns:
        dM_dL: derivative of each component of each affine matrix with respect to the components of its Cholesky decomposition
    """
    #get some of the dimensions from the inputs
    m = InvV.n
    d = InvV.d

    #initialise the required array
    dM_dL = np.zeros([m,d,d,d,d])
    
    #firstly loop through each map/component
    for j in range(m):
        #now want to loop over each of the Cholesky components
        for k in range(d):
            for l in range(d):
                #take the l-th column of L and put it in row k
                dM_dL[j,k,l,k,:] += InvV.L[j,:,l]
                #take the k-th row of L and put it in row l
                dM_dL[j,k,l,l,:] += InvV.L[j,k,:] 
    return dM_dL

def Map_Gradients(post_eval,q,InvV,m_points):
    """
    Determines the derivatives of the residual log ratio with respect to the parameters of each map.
    Inputs:
        post_eval: value of posterior * evidence at the chosen model points (N array)
        q: current estimate of posterior mixture weights (m array)
        InvV: current estimate of the inverse posterior maps (Affine_Maps)
        m_points: the points at which the posterior has been evaluated (m_points)
    """
    m = InvV.n
    N = m_points.num
    d = InvV.d
    
    ds_dq = np.zeros([m,N])
    dr_dq = np.zeros([m,N])
    
    ds_db = np.zeros([m,d,N])
    dr_db = np.zeros([m,d,N])
    
    ds_dL = np.zeros([m,d,d,N])
    dr_dL = np.zeros([m,d,d,N])
    
    dB_dL = np.zeros([m,d,d,d,N])
    dM_dL = Cholesky_Derivs(InvV,m_points)
    Q = Partitioner(q, InvV, post_eval, m_points)
    
    for j in range(m):
        backtrack = m_points.map(InvV,j)
        ds_dq[j,:] = - Q[j,:] / q[j]
        dr_dq[j,:] = ds_dq[j,:] - np.mean(ds_dq[j,:])
        
        for k in range(d):
            ds_db[j,k,:] = Q[j,:] * backtrack.all[:,k].T
            dr_db[j,k,:] = ds_db[j,k,:] - np.mean(ds_db[j,k,:])
            
            for l in range(d):
                for i in range(N):
                    for row in range(d):
                        for col in range(d):
                            dB_dL[j,row,k,l,i] += m_points.pick(i)[col] * dM_dL[j,row,col,k,l]
                    ds_dL[j,k,l,i] = Q[j,i] * np.inner(backtrack.pick(i),dB_dL[j,:,k,l,i])
                if k == l:
                    ds_dL[j,k,l,:] += (2/InvV.L[j,k,l])
                    
                dr_dL[j,k,l,:] = ds_dL[j,k,l,:] - np.mean(ds_dL[j,k,l,:])
                
    return dr_dq, dr_db, dr_dL



# posterior = multivariate_normal(mean=2,cov=1)
# ref = multivariate_normal(mean=0,cov=1)

# plot_dom = np.linspace(-5,5,100)

# fig, ax = plt.subplots()
# ax.plot(plot_dom,posterior.pdf(plot_dom),'-k',label='Posterior',linewidth=3)
# ax.plot(plot_dom,ref.pdf(plot_dom),'--k',label='Intermediate',linewidth=3)
# ax.legend(fontsize=12)

# #lets check that the point-wise derivatives are corrent first
# #we will need a map that we know is slightly incorrect
# L = np.array([1.4]).reshape([1,1,1])
# b = np.array([-2.7]).reshape([1,1])
# q = np.array([1])
# m_points = Model_Points(np.linspace(-5,5,20).reshape([20,1]))
# post_eval = posterior.pdf(m_points.all)

# InvV = Affine_Maps(L,b)
# Q = Partitioner(q,InvV,post_eval,m_points)
# s, r = Log_Scaling(post_eval,q,InvV,m_points)

# fig, ax = plt.subplots()
# ax.plot(m_points.all,s.T)
# dr_dq, dr_db,dr_dL = Map_Gradients(post_eval,q,InvV,m_points)


# # #now try the finite differences
# Lpos = np.array([1.45]).reshape([1,1,1])
# Lneg = np.array([1.35]).reshape([1,1,1])

# bpos = np.array([-2.65]).reshape([1,1])
# bneg = np.array([-2.75]).reshape([1,1])

# L_InvVpos = Affine_Maps(Lpos,b)
# L_InvVneg = Affine_Maps(Lneg,b)

# b_InvVpos = Affine_Maps(L,bpos)
# b_InvVneg = Affine_Maps(L,bneg)



# s_Lpos, r_Lpos = Log_Scaling(post_eval,q,L_InvVpos,m_points)
# s_Lneg, r_Lneg = Log_Scaling(post_eval,q,L_InvVneg,m_points)


# s_bpos, r_bpos = Log_Scaling(post_eval,q,b_InvVpos,m_points)
# s_bneg, r_bneg = Log_Scaling(post_eval,q,b_InvVneg,m_points)

# fd_ds_db = (r_bpos - r_bneg) / 0.1
# fd_dr_dL = (r_Lpos - r_Lneg) / 0.1

# # fd_ds_db = (b_spos - b_sneg) / 0.1
# # fd_dr_db = (b_rpos - b_rneg) / 0.1

# fig, ax = plt.subplots(ncols=2,figsize=[12,4])

# ax[0].plot(m_points.all,dr_db[0,0,:],'-k',linewidth=3,label='Analytical')
# ax[0].plot(m_points.all,fd_ds_db,'--r',linewidth=2,label='Numerical')
# ax[1].plot(m_points.all,dr_dL[0,0,0,:],'-k',linewidth=3,label='Analytical')
# ax[1].plot(m_points.all,fd_dr_dL,'--r',linewidth=2,label='Numerical')
# ax[0].set_title(r'$\partial r/\partial b$',fontsize=16)
# ax[1].set_title(r'$\partial r/\partial L$',fontsize=16)
# ax[0].legend(fontsize=12)
# ax[1].legend(fontsize=12)

# ax[0,2].plot(m_points.all,ds_db[0,0,:],'-k',linewidth=3,label='Analytical')
# ax[1,2].plot(m_points.all,dr_db[0,0,:],'-k',linewidth=3,label='Analytical')
# ax[0,2].plot(m_points.all,fd_ds_db[0,:],'--r',linewidth=2,label='Numerical')
# ax[1,2].plot(m_points.all,fd_dr_db[0,:],'--r',linewidth=2,label='Numerical')
# ax[0,2].set_title(r'$\partial s/\partial b$',fontsize=14)
# ax[1,2].set_title(r'$\partial r/\partial b$',fontsize=14)
# ax[0,2].legend(fontsize=12)
# ax[1,2].legend(fontsize=12)

# plt.subplots_adjust(hspace=0.3)


# L = np.array([1.4]).reshape([1,1,1])
# b = np.array([-2.7]).reshape([1,1])
# q = np.array([1])
# m_points = bm.Model_Points(np.linspace(-5,5,20).reshape([20,1]))
# post_eval = posterior.pdf(m_points.all)

# InvV = bm.Affine_Maps(L,b)

# num_iter = 5

# fig, ax = plt.subplots(nrows=num_iter,ncols=2,figsize=[12,10])


# for k in range(num_iter):
#     s, r = bm.Log_Scaling(post_eval,q,InvV,m_points)
#     ds_dL,ds_db = bm.Map_Gradients(InvV,m_points)
#     dr_dL,dr_db = bm.Centre_Gradients(ds_dL,ds_db)
#     all_grad = np.concatenate([dr_dL[0,0,:,:],dr_db[0,:,:]],axis=0)
#     M = all_grad.T
    
#     update = -1 * np.linalg.solve(M.T @ M,M.T @ r[0,:])
    
#     InvV = InvV.update(update)
    
#     new_mean = -InvV.b[0,0] / (InvV.L[0,0,0] **2)
#     new_var = 1 / (InvV.L[0,0,0]**2)
    
#     post_est = multivariate_normal(mean=new_mean,cov=new_var)
    
#     evid = round(np.exp(np.mean(s)),3)
#     misfit = round(np.linalg.norm(r))
    
#     string = 'evidence = ' + str(evid) + ' \n misfit = ' + str(misfit)
    
#     ax[k,0].plot(m_points.all,s[0,:],'-k',linewidth=2)
#     ax[k,1].plot(plot_dom,posterior.pdf(plot_dom),'--',color='grey',linewidth=2)
#     ax[k,1].plot(plot_dom,post_est.pdf(plot_dom),'-k',linewidth=2)
#     ax[k,1].text(1.1,0.4,string,fontsize=14,transform=ax[k,1].transAxes)
    
# ax[0,0].set_title('Log Ratio',fontsize=16)
# ax[0,1].set_title('Current Posterior Estimate',fontsize=16)