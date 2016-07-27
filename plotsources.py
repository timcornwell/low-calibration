import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np

from sources import *
from ionosphere import *


freqs=np.array([5e7, 1.0e8])
colors=['red', 'green']
bandwidth=1e6

plt.clf()

fluxes=np.power(10, np.arange(-4.0, 1.0, 0.1))
omega = numpy.power(3.0e8 / (2 * 35.0 * 50e6), 2)
numbers=numpy.zeros([len(fluxes)])
for k in range(len(fluxes)):
    numbers[k] = len(sources().randomsources(fluxes[k], freq=50e6, FOV=omega))
plt.loglog(fluxes, numbers, label='%dMHz' % 50, color='red')

omega = numpy.power(3.0e8 / (2 * 35.0 * 1e8), 2)
for k in range(len(fluxes)):
    numbers[k] = len(sources().randomsources(fluxes[k], freq=1e8, FOV=omega))
plt.loglog(fluxes, numbers, label='%dMHz' % (100), color='green')

plt.axes().set_ylim([1.0, 1e5])
plt.axvline(sources().tnoise(5e7, 10.0, bandwidth=bandwidth)*numpy.sqrt(512)*10.0, color='red', ls='--',
            label='50MHZ PPC')
plt.axvline(sources().tnoise(1e8, 10.0, bandwidth=bandwidth)*numpy.sqrt(512)*10.0, color='green', ls='--',
            label='100MHZ PPC')
plt.axvline(sources().tnoise(5e7, 10.0, bandwidth=bandwidth)*10.0, color='r', ls=':', label='50MHZ DFV')
plt.axvline(sources().tnoise(1e8, 10.0, bandwidth=bandwidth)*10.0, color='green', ls=':',
            label='100MHZ DFV')
plt.ylabel('Numbers of sources per station beam')
plt.xlabel('Flux (Jy)')
plt.title('Numbers of sources per station beam')
plt.legend(loc='lower left')
plt.savefig('numbers.pdf')
