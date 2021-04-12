from Core_Functions import gmm_base as gm
import numpy as np
import ot

#firstly, let's consider a 1D example
f = gm.Gaussian_Mixture(3,1)
g = gm.Gaussian_Mixture(4,1)

f_m = np.zeros([3,1])
f_m[:,0] = [1,3,2]
f_c = np.zeros([3,1,1])
f_c[:,0,0] = [0.01,0.04,0.3]

g_m = np.zeros([4,1])
g_m[:,0] = [-1,0,7,3]
g_c = np.zeros([4,1,1])
g_c[:,0,0] = [0.1,0.01,0.25,0.04]

f.assign_w(np.array([0.3,0.2,0.5]))
f.assign_m(f_m)
f.assign_cov(f_c)

g.assign_w(np.array([0.1,0.3,0.4,0.2]))
g.assign_m(g_m)
g.assign_cov(g_c)

#print(gm.GMM_Transport(f, g, 0.01,plan=True))
#a = gm.Analytical_Gradients(f,g)
#print(a.dsigma(1,1))

W = gm.Wasserstein_Cost(f, g)
#print(W)
P, log = ot.bregman.sinkhorn_stabilized(f.w, g.w, W, 0.001,log=True)
print(log['logu'],log['logv'])