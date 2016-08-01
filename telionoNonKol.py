from functools import lru_cache

import numpy
from scipy.special import gamma as gammabackend

from zernikecache import noll_to_nm


@lru_cache(maxsize=None)
def gamma(n):
    return gammabackend(n)


class TelIono:
    def ionosphere(self, baseline, s0=7.0, wavelength=3.0, beta=1.8):
        """
        Mevius model
        """
        return numpy.power(baseline / (s0 * (3.0 / wavelength)), +beta / 2.0) / 1.5
    
    def varphase(self, J=1, r0=22.0, B=51.4, wavelength=3.0, beta=3.8):
        """
        Revised Noll formula for variance in phase after fitting J Zernikes
        """
        n, m = noll_to_nm(J)
        return (numpy.sin(numpy.pi * ((beta - 2) / 2.0)) * ((n + 1) / numpy.pi) *
                (gamma((2 * n + 2 - beta) / 2) * gamma((beta + 4) / 2) * gamma(beta / 2) /
                 gamma((2 * n + 4 + beta) / 2)) * numpy.power(B / (r0 * (3.0 / wavelength)), beta - 2))
    
    def dr(self, J=1, r0=22.0, B=55.0, tsky=10000. * 3600.0, tiono=10.0, wavelength=3.0, beta=3.8):
        """ Convert J etc to dynamic range
        TODO: Fix for new model
        """
        return numpy.sqrt(tsky / tiono) / (numpy.sqrt(2.0 * self.varphase(J, r0, B, wavelength, beta)))
    
    def tsky(self, J, DR=1e5, r0=22.0, B=51.4, tiono=10.0, wavelength=3.0, beta=3.8):
        """ Time on sky to obtain given dynamic range
        TODO: Fix for new model
        """
        return tiono * DR ** 2 * self.varphase(J, r0, B, wavelength, beta)
