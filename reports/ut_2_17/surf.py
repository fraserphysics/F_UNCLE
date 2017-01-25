'''surf.py for plotting EOS as surface

Derived from
http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#surface-plots

Use ideal gas law PV = nRT

From wikipedia:

n is the # of moles
R is the gas constant (8.314 J*K-1*mol-1)
'''

def ideal_P(V,T):
    R = 8.314 # The gas constant
    n = 0.1   # Number of moles
    return (n*R*T)/V

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

V = np.linspace(5, 10, 20)
T = np.linspace(200, 300, 20)
V, T = np.meshgrid(V, T)
P = ideal_P(V,T)
print('P.shape={0}'.format(P.shape))

# Plot the surface.
surf = ax.plot_surface(V,T,P, linewidth=.15, antialiased=False) #cmap=cm.jet, )

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(4))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.xaxis.set_major_locator(LinearLocator(6))
ax.yaxis.set_major_locator(LinearLocator(5))
ax.set_xlabel(r'$V$')
ax.set_ylabel(r'$T$')
ax.set_zlabel(r'$P$')

import sys
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1])
else:
    plt.show()
