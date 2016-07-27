import numpy
import math
import random
import numpy as np

from scipy import interpolate
from scipy.constants import k

class sources:
# The number counts are a CDF when normalised correctly.
    def randomsources(self, smin, freq, FOV):
        nsources=int(FOV*self.numbers(smin, freq))
        cdf = lambda x: (FOV*self.numbers(x, freq)/float(nsources))
        S=numpy.ones(nsources)
        for nsource in range(nsources):
            p=random.uniform(0.0, 1.0)
            for expflux in numpy.arange(numpy.log10(smin), 1.0, 0.01):
                flux=10**expflux
                if (p > cdf(flux)):
                    S[nsource] = flux
                    break
        return sorted(S)
            
                
# Integrated source counts from Bregman (2012) 140 MHz values scaled to actual frequency
    def numbers(self, flux, freq=1e8):
        fluxes= [2e-5,     6e-4,   2e-3,   2e-2,   1e-1,   3e-1, 1.7,  20.0]
        numbers=[4.12e7, 4.45e5, 1.88e5, 2.98e4, 5.92e3, 1.36e3, 81.0, 0.75]
        s = interpolate.InterpolatedUnivariateSpline(numpy.log10(numpy.array(fluxes)),
                                                     numpy.log10(numpy.array(numbers)))
        return (10**s(numpy.log10(flux)))*numpy.power(freq/1.4e8, -0.8)

# Spot values for A over T from BDv2
    def tnoise(self, freq=1e8, time=1000.0*3600.0, bandwidth=1e5):
        # These are the single pol values
        s = interpolate.InterpolatedUnivariateSpline([50,   110,   160, 220, 280, 340],
                                                     [72.0, 380.0, 535, 530, 500, 453])
        SEFD = 1e26 * k / s(freq/1e6)
        RTB = numpy.sqrt(bandwidth*time)
        return SEFD / (2.0 * RTB)
