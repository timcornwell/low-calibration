import numpy
import math
import random
import matplotlib.pyplot as plt
import numpy as np


class sources:
#  From Condon et al 2012
    def confusion(self, freq=1.0e8, B=35.0):
        theta=180.0*3600.0*3.0e8/(freq*B*1000.0*np.pi)
        return 1.2e-6 * np.power(freq / 3.02e9, -0.7) * np.power(theta/8.0, 10.0/3.0)

#  Integral source counts N(>S) from Condon et al 2012
    def numbers(self, s=1.0, freq=1e8):
        return numpy.power(freq/1.4e9, 0.7)*9000.0*numpy.power(s, -1.7)

# Randomly chosen strength.         
    def randomsources(self, smin=1.0, nsources=1):
        S=numpy.ones(nsources)
        for i in range(nsources):
            S[i]=smin*math.pow(random.uniform(0,1),-1.0/0.7)
        return S
                
# Integrate S.dNdS over S
    def integratedflux(self, s=1.0, freq=1e8, smax=10000.0):
        return (1.7/0.7)*numpy.power(freq/1.4e9, 0.7)*9000.0*(numpy.power(s, -0.7)-numpy.power(smax, -0.7))
        
# Spot values from BDv1
    def noise(self):
        return {'50':25.1e-6, '110':3.1e-6, '160':3.4e-6, '220':3.4e-6} 

#  Simpler version
    def tnoise(self, freq=1e8, time=10000.0*3600.0, bandwidth=1e5):
        scale=numpy.sqrt(10000.0*3600.0/time)*numpy.sqrt(1e5/bandwidth)
        if freq<7.5e7:
            return  scale*25.1e-6
        else:
            return  scale*3.1e-6
