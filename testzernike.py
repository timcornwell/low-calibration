import sys

import pylab
pylab.rcParams['figure.figsize'] = (6.0, 6.0)
#pylab.rcParams['image.cmap'] = 'rainbow'

import matplotlib.pyplot as plt
import numpy
from zernikecache import *
import subprocess
print("Working from Git repository %s" % 
      subprocess.check_output(["git",  "describe", "--long", "--all"]))

plt.clf()
r=numpy.arange(0.9, 1.0, 0.0001)
for j in range(1,10000,1000):
    m, n = noll_to_nm(j+1)
    z=zernike(m, n, rho=r, phi=numpy.pi/4.0)
    plt.plot(r, z, '-')
plt.show()

print ("Factorial %s" % str(fac.cache_info()))

