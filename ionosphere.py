from teliono import *
import numpy

import matplotlib.pyplot as plt
		
hiono=300.0
waves=[6.0, 3.0, 2.0]
base=numpy.arange(0.01, 100.0, 0.01)
ti=TelIono()

plt.title(r'Ionosphere (r:50MHz, g:100MHz, b:150MHz)')
plt.xlabel('Distance (km)')
plt.ylabel('Phase')
plt.text(1.05*numpy.sqrt(hiono*6.0/1000.0), 1.1e-3, 'Fresnel zone')
plt.axes().set_xlim([1e-2, 100.0])
plt.axes().set_ylim([1e-3, 1e2])
ind=0
color=['r', 'g', 'b']
for wave in waves:
	phase=ti.ionosphere(base)*wave
	plt.loglog(base, phase, color=color[ind])
	plt.axvline(numpy.sqrt(hiono*wave/1000.0), ls='--', color=color[ind])
	ind=ind+1

plt.axvline(0.035, color='purple', ls='--')
plt.text(0.035, 1.1e-3, 'Station beam')
	
plt.savefig('ionosphere.pdf')

plt.clf()
base=numpy.arange(0.01, 100.0, 0.01)
veliono=500.0/3600.0
timeiono=base/veliono
plt.loglog(base, timeiono, color='black', label='Ionosphere')
plt.title(r'Ionospheric motion')
plt.xlabel(r'Distance (km)')
plt.ylabel('Transit time (s)')

plt.axvline(0.035, color='red', ls='--', label='Station')
plt.axvline(1, color='green', ls='--', label='Core')
plt.axvline(80, color='blue', ls='--', label='Array')
plt.axvline(0.3, color='purple', ls='--', label='Fresnel')
plt.legend(loc="upper left")	
plt.savefig('timescales.pdf')
