from Core_Functions.gmm_base import Wasserstein_Cost, Gaussian_Mixture, Analytical_Gradients, GMM_Transport, Wasserstein_Gradients
from Core_Functions.gmm_plot import Source_Target_1D as plot1D
from Core_Functions.optimisation import GMM_Inversion, Flattener, Unflattener
import numpy as np
import ot

#firstly, let's consider a 1D example
f = Gaussian_Mixture(3,1)
g = Gaussian_Mixture(3,1)

f_m = np.zeros([3,1])
f_m[:,0] = [1,3,2]
f_c = np.zeros([3,1,1])
f_c[:,0,0] = [0.01,0.04,0.3]

g_m = np.zeros([3,1])
g_m[:,0] = [-1,7,3]
g_c = np.zeros([3,1,1])
g_c[:,0,0] = [0.1,0.01,0.04]

f.assign_w(np.array([0.3,0.2,0.5]))
f.assign_m(f_m)
f.assign_cov(f_c)

g.assign_w(np.array([0.3,0.4,0.3]))
g.assign_m(g_m)
g.assign_cov(g_c)


W = Wasserstein_Cost(f, g)
P, log = ot.bregman.sinkhorn_stabilized(f.w, g.w, W, 0.001,log=True)
#print(log['logu'],log['logv'])

GW, P, alpha = GMM_Transport(f,g,0.01)
print(Wasserstein_Gradients(f,g,P,alpha))

#fig, ax = plot1D(f,g,[-3,4],res=300)

#print(GMM_Inversion(f,g,0.01))
