import numpy
import math
import random
import numpy as np

from scipy import interpolate
from scipy.constants import k

class sources:
#  From Condon et al 2012
    def confusion(self, freq=1.0e8, B=35.0):
        theta=180.0*3600.0*3.0e8/(freq*B*1000.0*np.pi)
        return 1.2e-6 * np.power(freq / 3.02e9, -0.7) * np.power(theta/8.0, 10.0/3.0)

#  Integral source counts N(>S) from Condon et al 2012
    def numbersold(self, s=1.0, freq=1e8):
        return numpy.power(freq/1.4e9, 0.7)*9000.0*numpy.power(s, -1.7)

# Randomly chosen strength.         
    def randomsources(self, smin=1.0, nsources=1):
        S=numpy.ones(nsources)
        for i in range(nsources):
            S[i]=smin*math.pow(random.uniform(0,1),-1.0/0.7)
        return S
                
# Integrated source counts from Bregman (2012) 140 MHz values scaled to actual frequency
    def numbers(self, flux, freq=1e8):
        fluxes= [2e-5,     6e-4,   2e-3,   2e-2,   1e-1,   3e-1, 1.7,  20.0]
        numbers=[4.12e7, 4.45e5, 1.88e5, 2.98e4, 5.92e3, 1.36e3, 81.0, 0.75]
        s = interpolate.InterpolatedUnivariateSpline(numpy.log10(numpy.array(fluxes)),
                                                     numpy.log10(numpy.array(numbers)))
        return (10**s(numpy.log10(flux)))*numpy.power(freq/1.4e9, -0.7)


# Spot values for A over T from BDv2
    def tnoise(self, freq=1e8, time=1000.0*3600.0, bandwidth=1e5):
        # These are the single pol values
        s = interpolate.InterpolatedUnivariateSpline([50,   110,   160, 220, 280, 340],
                                                     [72.0, 380.0, 535, 530, 500, 453])
        SEFD = 1e26 * k / s(freq/1e6)
        RTB = numpy.sqrt(bandwidth*time)
        return SEFD / (2.0 * RTB)
