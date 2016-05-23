import numpy

class TelIono:
    def varphase(self, J=1, r0=14.0, B=80.0):
        return 0.2994 * numpy.power(J, -math.sqrt(3)/2.0) * numpy.power(B/r0, 5.0/3.0)
    def dr(self, J=1, r0=14.0, B=80.0, tobs=10000.*3600.0, tiono=10.0):
        return 1.854 * numpy.power(J, math.sqrt(3)/4.0) * numpy.power(B/r0, -5.0/6.0) * numpy.sqrt(tobs/tiono)
    def ionosphere(self, baseline):
        return numpy.power(baseline/14.0,+1.8/2.0)/1.5
    def tobs(self, J, DR=1e5, r0=14.0, B=80.0, tobs=10000.*3600.0, tiono=10.0):
        return tiono * (DR/3.397) * (DR/3.397) * numpy.power(J, -numpy.sqrt(3)/2.0) * numpy.power(B/r0, 5.0/6.0)
