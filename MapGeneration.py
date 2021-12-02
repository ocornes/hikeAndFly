#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:23:01 2021

@author: olivier
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def cone(pos, glide, x_0, y_0):
    """Return a cone surface for the glide."""

    return -(1/glide)*(pos[:, :, 0]**2 + pos[:, :, 1]**2)**0.5 + 0.6

# Our 2-dimensional distribution will be over variables X and Y
N = 40
X = np.linspace(-5, 5, N) # Assuming a 10 km box
Y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu_0 = np.array([-4., -4.])
Sigma_0 = np.array([[ 5. , -0.5], [-0.5,  5.]])

mu_1 = np.array([1., 4.])
Sigma_1 = np.array([[ 4. , -1], [-1,  3.]])

mu_2 = np.array([3., -5.])
Sigma_2 = np.array([[ 4 , -0.8], [-0.8,  2]])

mu_3 = np.array([-4., 6.])
Sigma_3 = np.array([[ 4 , -0.1], [-1,  2]])

mu_4 = np.array([0., 0.])
Sigma_4 = np.array([[ 1 , -0.1], [-1,  1]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu_0, Sigma_0)
Z = Z + 0.8*multivariate_gaussian(pos, mu_1, Sigma_1)
Z = Z + multivariate_gaussian(pos, mu_2, Sigma_2)
Z = Z + 0.5*multivariate_gaussian(pos, mu_3, Sigma_3)
Z = 26*(Z + 0.1*multivariate_gaussian(pos, mu_4, Sigma_4)) # adding a 26 factor to have "mountains" that are 1.6 km high around the site

glide = 10
x_0 = 0
y_0 = 0 
Cone = cone(pos, glide, x_0, y_0)


# plot using subplots
fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1 = plt.axes(projection='3d')

Z1 = Cone-Z
Z1[Z1<0]= 0

ax1.plot_surface(X, Y, Z1, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

#ax1.plot_surface(X, Y, Cone, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                cmap=cm.inferno)
#ax1.view_init(55,-70)
#ax1.set_xticks([])
#ax1.set_yticks([])
#ax1.set_zticks([])
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')

plt.rcParams["figure.figsize"]=10,10
plt.show()

topo = {'X':X, 'Y':Y, 'Z':Z }

filename = 'dogs'
outfile = open(filename,'wb')
pickle.dump(topo,outfile)
outfile.close()


'''
#ax2 = fig.add_subplot(1,1,1,projection='3d')
ax2 = plt.axes(projection='3d')
ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
ax2.view_init(90, 270)

#ax2.grid(False)
#ax2.set_xticks([])
#ax2.set_yticks([])
#ax2.set_zticks([])
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')

plt.show()'''