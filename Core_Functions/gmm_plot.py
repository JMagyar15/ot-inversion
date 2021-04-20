import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def Source_Target_1D(f,g,x_ran,res=100):
    x = np.linspace(x_ran[0],x_ran[1],res)
    f_d = np.zeros(res)
    g_d = np.zeros(res)

    for i in range(f.n):
        f_d += f.w[i] * multivariate_normal.pdf(x, mean = f.m[i,:], cov = f.cov[i,:,:])

    for j in range(g.n):
        g_d += g.w[j] * multivariate_normal.pdf(x, mean = g.m[j,:], cov = g.cov[j,:,:])

    fig, ax = plt.subplots(figsize=[8,6])

    ax.plot(x,f_d,'-k',linewidth=3,label='Source')
    ax.plot(x,g_d,'--k',linewidth=3,label='Target')

    ax.tick_params(labelsize=12)

    plt.show()
    return fig, ax

def Source_Target_2D(f,g,x_ran,y_ran,res=100):
    x = np.linspace(x_ran[0],x_ran[1],res)
    y = np.linspace(y_ran[0],y_ran[1],res)

    X, Y = np.meshgrid(x,y)
    
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    f_d = np.zeros([res,res])
    g_d = np.zeros([res,res])

    for i in range(f.n):
        f_d += f.w[i] * multivariate_normal.pdf(pos, mean = f.m[i,:], cov = f.cov[i,:,:])

    for j in range(g.n):
        g_d += g.w[j] * multivariate_normal.pdf(pos, mean = g.m[j,:], cov = g.cov[j,:,:])

    fig, ax = plt.subplots(ncols = 2, figsize=[12,6])
    ax[0].contourf(X,Y,f_d,cmap='binary')
    ax[1].contourf(X,Y,g_d,cmap='binary')

    ax[0].set_title('Source Distribution',fontsize=14)
    ax[1].set_title('Target Distribution',fontsize=14)

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)

    plt.show()
    return fig, ax