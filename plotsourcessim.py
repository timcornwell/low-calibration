import numpy as numpy
import matplotlib.pyplot as plt

from sources import *
from ionosphere import *


bandwidth=1e6

plt.clf()

for realisation in range(10):
    omega = numpy.power(3.0 / (2 * 35.0), 2)
    S=sources().randomsources(smin=0.020, freq=1e8, FOV=omega)
    print("FOV = %f, number of sources = %d, max = %.3f" %(omega, len(S), numpy.max(S)))
    plt.semilogy(range(len(S)), S, '.', label='%dMHz' % 50, color='red')
plt.ylabel('Flux')
plt.savefig('sourcessim.pdf')
