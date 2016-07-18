import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np

from sources import *
from ionosphere import *


freqs=np.array([5e7, 1.0e8])
colors=['red', 'green']
bandwidth=1e6

plt.clf()
baselines=np.power(10.0, np.arange(-2.0, 3.0, 0.1))
conf=sources().confusion(freq=5e7, B=baselines)
plt.loglog(baselines, conf, color='red')
conf=sources().confusion(freq=1e8, B=baselines)
plt.loglog(baselines, conf, color='green')
plt.xlabel('Baseline length (km)')
plt.ylabel('Noise (Jy)')
plt.title('Noise (solid: confusion, dotted: thermal) 1MHz bandwidth')
plt.axvline(0.035, color='purple', ls='--')
plt.text(0.035, 1.5e-9, 'Station diameter')
plt.axvline(80.0, color='brown', ls='--')
plt.text(80.0, 1.5e-9, 'Array diameter')
plt.axhline(sources().tnoise(5e7, bandwidth=bandwidth), color='red', ls=':', label='50MHZ source')
plt.axhline(sources().tnoise(1e8, bandwidth=bandwidth), color='green', ls=':', label='100MHZ source')

plt.axhline(sources().tnoise(5e7, 10.0, bandwidth=bandwidth), color='red', ls='--', label='50MHZ cal')
plt.axhline(sources().tnoise(1e8, 10.0, bandwidth=bandwidth), color='green', ls='--', label='100MHZ cal')
plt.legend(loc='best')
plt.savefig('confusion.pdf')

plt.clf()
fluxes=np.power(10, np.arange(-6.0, 1.0, 0.1))

integratedflux = sources().integratedflux(fluxes, freq=50e6)
omega = numpy.power(3.0e8 / (2 * 35.0 * 50e6), 2)
plt.loglog(fluxes, omega * integratedflux, color='red')
integratedflux = sources().integratedflux(fluxes, freq=100e6)
omega = numpy.power(3.0e8 / (2 * 35.0 * 100e6), 2)
plt.loglog(fluxes, omega * integratedflux, color='green')

integratedflux=sources().integratedflux(fluxes, freq=50e6)
omega=numpy.power(3.0e8/(2*35.0*50e6), 2)
plt.loglog(fluxes, omega*integratedflux, color='red')

integratedflux=sources().integratedflux(fluxes, freq=1e8)
omega=numpy.power(3.0e8/(2*35.0*1e8), 2)
plt.loglog(fluxes, omega*integratedflux, color='green')

plt.xlabel('Flux (Jy)')
# plt.axes().set_xlim([1e-6, 1e2])
# plt.axes().set_ylim([1, 1e3])
plt.axvline(sources().tnoise(5e7, 10.0)*numpy.sqrt(1024.0), color='red', ls='--', label='50MHZ cal')
plt.axvline(sources().tnoise(1e8, 10.0)*numpy.sqrt(1024.0), color='green', ls='--', label='50MHZ cal')
plt.axvline(sources().tnoise(5e7), color='red', ls=':', label='50MHZ source')
plt.axvline(sources().tnoise(1e8), color='green', ls=':', label='100MHZ source')

plt.ylabel(r'Integrated Flux per station beam')
plt.legend(loc='lower left')
plt.title('Integrated flux per station beam')
plt.savefig('integratedflux.pdf')

plt.clf()
omega = numpy.power(3.0e8 / (2 * 35.0 * 50e6), 2)
numbers = omega * sources().numbers(fluxes, freq=50e6)
plt.loglog(fluxes, numbers, label='%dMHz' % 50, color='red')

omega = numpy.power(3.0e8 / (2 * 35.0 * 1e8), 2)
numbers = omega * sources().numbers(fluxes, freq=1e8)
plt.loglog(fluxes, numbers, label='%dMHz' % (100), color='green')

plt.axes().set_ylim([1.0, 1e6])
plt.axvline(sources().tnoise(5e7, 10.0)*numpy.sqrt(1024.0), color='red', ls='--', label='50MHZ cal')
plt.axvline(sources().tnoise(1e8, 10.0)*numpy.sqrt(1024.0), color='green', ls='--', label='100MHZ cal')
plt.axvline(sources().tnoise(5e7), color='red', ls=':', label='50MHZ source')
plt.axvline(sources().tnoise(1e8), color='green', ls=':', label='100MHZ source')
plt.ylabel('Numbers of sources per station beam')
plt.xlabel('Flux (Jy)')
plt.title('Numbers of sources per station beam')
plt.legend(loc='lower left')
plt.savefig('numbers.pdf')

plt.clf()
omega = numpy.power(3.0e8 / (2 * 35.0 * 50e6), 2)
numbers = omega * sources().numbers(fluxes, freq=50e6)
integratedflux = omega * sources().integratedflux(fluxes, freq=50e7)
plt.loglog(numbers, integratedflux, label='%dMHz' % 50)

omega = numpy.power(3.0e8 / (2 * 35.0 * 1e8), 2)
numbers = omega * sources().numbers(fluxes, freq=1e8)
plt.loglog(fluxes, numbers, label='%dMHz' % (100))

plt.axhline(sources().tnoise(5e7, 10.0)*numpy.sqrt(1024.0), color='red', ls='--', label='50MHZ cal')
plt.axhline(sources().tnoise(1e8, 10.0)*numpy.sqrt(1024.0), color='green', ls='--', label='100MHZ cal')
# plt.axes().set_xlim([1.0, 1e8])
plt.axes().set_ylim([0.1, 1e6])
plt.xlabel('Number of sources per station beam')
plt.ylabel('Flux per station beam')
plt.legend(loc='lower left')
plt.title('Flux vs Numbers of sources per station beam')
plt.savefig('numbersintegrated.pdf')