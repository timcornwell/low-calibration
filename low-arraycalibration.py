# coding: utf-8

# ### Perform Zernike polynominal analysis of SKA1-LOW and LOFAR calibration

import multiprocessing
import subprocess

import pylab

from telback import definetel, plot, printstats, runtrials
from telinputs import *
from telutil import *
from zernikecache import fac, pre_fac, inv_sum_pre_fac

plt.switch_backend('pdf')
pylab.rcParams['figure.figsize'] = (8.0, 8.0)

print("Working from Git repository %s" % subprocess.check_output(["git", "describe", "--long", "--all"]))

nproc = 2 * multiprocessing.cpu_count()
print("Using %d processes in inner loop" % (nproc))
#######################################################################################################
## Start of real processing

# Get the inputs from the arguments of the script invocation
name, nstations, nnoll, wavelength, stationdiameter, rcore, rmin, hiono, HWZ, FOV, rhalo, rmax, freq, \
bandwidth, tiono, configs, ntrials, doplot, doFresnel = getinputs()

imgnoise, visnoise, nsources, weight = calculatenoise(tiono, freq, bandwidth, nstations, stationdiameter, FOV)

tel, mst = definetel(configs, weight)

stats = {}
for config in tel:
    stats[config] = {}
    for nsource in [nsources]:
        print("Processing %s %d sources" % (config, nsource))
        stats[config][nsource] = runtrials(nnoll, ntrials, nsource, tel[config], wavelength,
                                           stationdiameter, HWZ, bandwidth = bandwidth, nproc=nproc)

printstats(stats, mst)

if doplot:
    plot(name, stats, nsources, nnoll)

#######################################################################################################
# We are done: see cache statistics for zernike shortcuts
print("fac %s" % (str(fac.cache_info())))
print("pre_fac %s" % (str(pre_fac.cache_info())))
print("inv_sum_pre_fac %s" % (str(inv_sum_pre_fac.cache_info())))
