import numpy


class TelIono:
    def ionosphere(self, baseline=51.4, s0=7.0, wavelength=3.0):
        """ Mevius model
        """
        return numpy.power(baseline / (s0 * (3.0 / wavelength)), +5 / 6) / 1.5
    
    def varphase(self, J=1, r0=22.0, B=51.4, wavelength=3.0):
        """ Noll formula for variance in phase after fitting J Zernikes
        """
        return 0.2994 * numpy.power(J, -numpy.sqrt(3) / 2.0) * numpy.power(B / (r0 * (3.0 / wavelength)), +5 / 6)
    
    def dr(self, J=1, r0=22.0, B=55.0, tsky=10000. * 3600.0, tiono=10.0, wavelength=3.0):
        """ Convert J etc to dynamic range
        """
        return numpy.sqrt(tsky / tiono) / (numpy.sqrt(2.0 * self.varphase(J, r0, B, wavelength)))
    
    def tsky(self, J, DR=1e5, r0=22.0, B=51.4, tiono=10.0, wavelength=3.0):
        """ Time on sky to obtain given dynamic range
        """
        return tiono * (DR / 1.304) ** 2 * numpy.power(J, -numpy.sqrt(3) / 2.0) * numpy.power(
            B / (r0 * (3.0 / wavelength)), +5 / 6)
