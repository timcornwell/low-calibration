import multiprocessing
import sys
from collections import defaultdict

import numpy

from sources import sources


# Define our own logger to copy output to a file
class Logger(object):
    def __init__(self, logfile):
        print("Copying output to %s" % (logfile))
        self.terminal = sys.stdout
        self.log = open(logfile, "a")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def getinputs():
    d = defaultdict(list)
    for k, v in ((k.lstrip('-'), v) for k, v in (a.split('=') for a in sys.argv[1:])):
        d[k] = v
    
    import random
    random.seed(d.get("seed", 781490893))
    
    name = d.get("name", "test-low-arraycalibration")
    sys.stdout = Logger("%s.log" % name)
    print("Copying output to %s" % "%s.log" % name)
    
    nproc = int(d.get("nproc", multiprocessing.cpu_count()))
    print("Using %d processes in inner loop" % (nproc))
    nstations = 512
    nnoll = int(d.get("Noll", 1000))
    
    wavelength = float(d.get("wavelength", 3.0))
    stationdiameter = float(d.get('stationdiameter', 35.0))
    rcore = float(d.get('rcore', 0.5))
    rmin = float(d.get('rmin', 0.0))
    hiono = float(d.get('hiono', 300.0))
    
    HWZ = hiono * wavelength / stationdiameter  # Half width to Zero
    FOV = (0.5 * wavelength / stationdiameter) ** 2
    rhalo = HWZ
    rmax = HWZ
    print("FOV (HWZ) at ionosphere=%.1f km" % HWZ)
    
    # Frequency
    freq = 3.0e8 / wavelength
    print("Observing frequency = %.2f MHz" % (freq / 1e6))
    bandwidth = float(d.get("bandwidth", 1e5))
    print("Observing bandwidth = %.2f MHz" % (bandwidth / 1e6))
    
    tiono = float(d.get("hiono", 10.0 * numpy.power(wavelength / 3.0, -5.0 / 6.0)))
    
    print("Ionospheric coherence time = %.1f (s)" % (tiono))
    
    configs = ['LOWBD2', 'LOWBD2-CORE', 'LOWBD2-RASTERHALO']
    configs = [d.get('configs', 'LOWBD2')]
    print("Processing %s" % str(configs))
    
    ntrials = int(d.get('ntrials', 1))
    
    doplot = d.get('doplot', True)
    doFresnel = d.get('doFresnel', True)
    return name, nstations, nnoll, wavelength, stationdiameter, rcore, rmin, hiono, HWZ, FOV, rhalo, rmax, freq, \
           bandwidth, tiono, configs, ntrials, doplot, doFresnel, nproc


def calculatenoise(tiono, freq, bandwidth, nstations, stationdiameter):
    # Calculate weight of solution for each pierce point
    imgnoise = sources().tnoise(time=tiono, freq=freq, bandwidth=bandwidth) * (512 / float(nstations)) * (35.0 /
                                                                                                          stationdiameter) ** 2
    visnoise = sources().tnoise(time=tiono, freq=freq, bandwidth=bandwidth) * float(512) * (35.0 / stationdiameter) ** 2
    weight = 1.0 / visnoise ** 2
    gainnoise = visnoise / numpy.sqrt(512)
    return imgnoise, visnoise, weight, gainnoise
