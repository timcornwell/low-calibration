# coding: utf-8

# These are helper functions for running telutil

import matplotlib.pyplot as plt
import numpy

from telutil import TelArray, TelPiercings, TelArrayPiercing, TelSources
from teliono import TelIono



#####
# ### Define all configs and plot

def runtrials(nnoll, ntrials, config, wavelength, stationdiameter, HWZ, smin, tsky=10000 * 3600.0, weight=1.0,
           snrthreshold=5.0, bandwidth=1e5, nproc=4, doFresnel=True):


    tp = TelArrayPiercing()

    Save=0.0
    for trial in range(ntrials):
        ts = TelSources()
        FOV=(wavelength/(2.0*stationdiameter))**2
        print("Minimum flux in model %.3f Jy" % (smin))
        ts.construct(smin=smin, radius=wavelength/stationdiameter, FOV=FOV)
        S=ts.sources['flux']
        print("FOV = %f, number of sources = %d, max = %.3f" % (FOV, len(S), numpy.max(S)))
        print("Total flux in model = %.3f" % (numpy.sum(ts.sources['flux'])))
        s = numpy.sqrt(tp.assess(ts, config, nnoll=nnoll, rmax=HWZ, wavelength=3.0, weight=weight,
                                 hiono=300, limit=0.95, doplot=(trial == 0), doFresnel=doFresnel,
                                 nproc=nproc))
        print("Trial %d, max(s)=%.1f" % (trial, numpy.max(s)))

        if trial == 0:
            Save = s / float(ntrials)
        else:
            Save = Save + s / float(ntrials)
    J = len(Save[Save > snrthreshold])
    if J < 1:
        J = 1
    DR = TelIono().dr(J, tsky=tsky, B=2 * HWZ)
    itsky = TelIono().tsky(J, DR=1e5, B=2 * HWZ)
    stdphase = numpy.sqrt(TelIono().varphase(J, B=2 * HWZ))

    return {'nsources': len(ts.sources['flux']), 'dr': DR, 's': Save, 'J': J, 'tsky': itsky, 'stdphase': stdphase,
            'bandwidth': (bandwidth / 1e6)}


def printstats(stats, mst):
    print("Config, Bandwidth, Nsources, peak, J, stdphase, DR, sky, MST")

    for config in stats.keys():
        print("%s, %.1f, %d, %.1f, %d, %.2f, %.2f, %.2f, %.1f" % (config,
                                                                  stats[config]['bandwidth'],
                                                                  stats[config]['nsources'],
                                                                  stats[config]['s'][0],
                                                                  stats[config]['J'],
                                                                  stats[config]['stdphase'],
                                                                  10.0 * numpy.log10(stats[config]['dr']),
                                                                  stats[config]['tsky'] / (
                                                                      24.0 * 3600.0 * 365.0),
                                                                  mst[config]))


def definetel(configs, weight=1.0, doplot=False):

    tel = {}
    mst = {}
    for config in configs:
    # Define all config variants
        tel[config] = TelArray()
        if config == 'LOWBD2':
            tel[config].readKML(kmlfile='V4Drsst512red_2.kml', weight=weight)
        elif config == 'LOWBD2-CORE':
            tel[config].readCSV('LOWBD2-CORE', csvfile='LOWBD2-CORE.csv', recenter=True, weight=weight)
        elif config == 'LOWBD2-HALO':
            tel[config].readCSV('LOWBD2-HALO', csvfile='LOWBD2-HALO.csv', recenter=True, weight=weight)
        elif config == 'LOWBD2-RASTERHALO':
            tel['LOWBD2-CORE'] = TelArray()
            tel['LOWBD2-CORE'].readCSV('LOWBD2-CORE', csvfile='LOWBD2-CORE.csv', recenter=True)
            tel['RASTERHALO'] = TelArray()
            tel['RASTERHALO'].rasterBoolardy(nhalo=346, name='LOWRASTERHALO', nstations=346, scale=1.05, rhalo=40.0,
                                             rcore=0.0, weight=weight)
            tel['LOWBD2-RASTERHALO'] = TelArray().add(tel['RASTERHALO'], tel['LOWBD2-CORE'], name=config)
        elif config == 'LOWBD2-RASTERHALO25KM':
            tel['RASTERHALO25KM'] = TelArray()
            tel['RASTERHALO25KM'].rasterBoolardy(nhalo=346, name='LOWRASTERHALO25KM', nstations=346, scale=40.0 / 25.7,
                                                 rhalo=25.7, rcore=0.0, weight=weight)
            tel['LOWBD2-CORE'] = TelArray()
            tel['LOWBD2-CORE'].readCSV('LOWBD2-CORE', csvfile='LOWBD2-CORE.csv', recenter=True)
            tel[config] = TelArray().add(tel['RASTERHALO25KM'], tel['LOWBD2-CORE'], name=config)
        elif config == 'LOFAR':
            tel[config].readLOFAR(weight=weight)
    # Plot and save the figure

        if doplot:
            tel[config].plot()
            plt.show()
            plt.savefig('%s.pdf' % config)
    #
        mst[config] = tel[config].mst(doplot=doplot)

    newtel = {}
    for config in configs:
        newtel[config] = tel[config]
        
    return (newtel, mst)

def plot(name, stats, nnoll):
    plt.clf()
    ymax = 0.1
    Jmax = 0

    for config in stats.keys():
        y = stats[config]['s']
        if numpy.max(y) > ymax:
            ymax = numpy.max(y)
        x = numpy.arange(nnoll)
        J = x[y < 5.0]
        if len(J):
            J=J[0]
        if J > Jmax:
            Jmax = J
        plt.semilogy(x, y, label=config)
    plt.axes().axhline(5.0, color='grey', linestyle='dotted', label='5 sigma cutoff')
    plt.axes().axvline(Jmax, color='grey', linestyle='dotted', label='J=%d' % (int(round(J))))
    plt.xlim([1, nnoll])
    plt.ylim([1, ymax])
    plt.xlabel('Singular value index')
    plt.ylabel('Singular value')
    plt.title(name)

    plt.legend(loc="upper right")
    plt.show()

    plt.savefig('%s.pdf' % name)

    plt.clf()

