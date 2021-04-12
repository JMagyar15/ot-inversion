import numpy as np

def Sinkhorn(p,q,C,reg,num_iter = 100):
    """
    Determines the optimal transport plan between histograms under entropic regularisation using Sinkhorn's algorithm.
    
    Inputs:
        p: source histogram (array)
        q: target histogram (array)
        W: cost matrix for transport between corresponding points in source and target (array)
        gamma: entropic regularisation parameter (float)
        num_iter: number of Sinkhorn interations (int)
    Outputs:
        P: optimal transport plan under entropic regularisation (array)
        u_k: left optimal scaling vector after last interation (array)
        v_k: right optimal scaling vector after last interation (array)
    """
    # TODO do this in log domain for stability
    # TODO add a check in here for whether the histrograms sum to 1 (or close to 1)

    K = np.exp(-C/reg)
    
    k=0 
    v_k = np.ones(np.size(q))
    
    while k <= num_iter:
        u_k = p / (K @ v_k)
        v_k = q / (np.transpose(K) @ u_k)
        
        k += 1
        
    diag_u = np.diag(u_k)
    diag_v = np.diag(v_k)
    
    P = diag_u @ K @ diag_v
    
    return P, u_k, v_k

def Sinkhorn_Divergence(f,g,C,reg,num_iter,plan=False):
    P, u, v = Sinkhorn(f,g,C,reg,num_iter=num_iter)
    a = reg * np.log(u)
    b = reg * np.log(v)

    Sink_Div = np.sum(a * u) + np.sum(b * v)
    if plan == False:
        return Sink_Div
    else:
        return Sink_Div, P
