# coding: utf-8

# ### Perform Zernike polynominal analysis of SKA1-LOW and LOFAR calibration

import subprocess

import pylab

from telback import definetel, plot, printstats, runtrials
from telinputs import *
from telutil import *
from zernikecache import fac, pre_fac, inv_sum_pre_fac

plt.switch_backend('pdf')
pylab.rcParams['figure.figsize'] = (8.0, 8.0)

print("Working from Git repository %s" % subprocess.check_output(["git", "describe", "--long", "--all"]))

#######################################################################################################
## Start of real processing

# Get the inputs from the arguments of the script invocation
name, nstations, nnoll, wavelength, stationdiameter, rcore, rmin, hiono, HWZ, FOV, rhalo, rmax, freq, \
bandwidth, tiono, configs, ntrials, doplot, doFresnel, nproc = getinputs()

tel, mst = definetel(configs)


stats = {}
for config in tel:
    stats[config] = {}
    imgnoise, visnoise, weight, gainnoise = calculatenoisearray(tiono, freq, bandwidth, tel[config].nstations,
                                                                     stationdiameter,
                                                                     FOV)
    tel[config].stations['weight']=weight * numpy.ones(tel[config].nstations)
    print("Processing %s" % (config))
    stats[config] = runtrials(nnoll, ntrials, tel[config], wavelength, stationdiameter, HWZ, bandwidth = bandwidth,
                              nproc=nproc, smin=10.0*imgnoise)

printstats(stats, mst)

if doplot:
    plot(name, stats, nsources, nnoll)

#######################################################################################################
# We are done: see cache statistics for zernike shortcuts
print("fac %s" % (str(fac.cache_info())))
print("pre_fac %s" % (str(pre_fac.cache_info())))
print("inv_sum_pre_fac %s" % (str(inv_sum_pre_fac.cache_info())))
