# coding: utf-8

# These are helper functions for running telutil

import matplotlib.pyplot as plt
import numpy

from telutil import TelArray, TelPiercings, TelArrayPiercing, TelSources
from teliono import TelIono



#####
# ### Define all configs and plot

def runtrials(nnoll, ntrials, nsources, config, wavelength, stationdiameter, HWZ, tsky=10000 * 3600.0, weight=1.0,
           snrthreshold=5.0, bandwidth=1e5, nproc=4, doFresnel=True):


    tp = TelArrayPiercing()

    Save=0.0
    for trial in range(ntrials):
        ts = TelSources()
        ts.construct(nsources=nsources, radius=wavelength / stationdiameter)
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

    return {'dr': DR, 's': Save, 'J': J, 'tsky': itsky, 'stdphase': stdphase, 'bandwidth': (bandwidth / 1e6)}


def printstats(stats, mst):
    print("Config, Bandwidth, Nsources, peak, J, stdphase, DR, sky, MST")

    for config in stats.keys():
        for nsources in stats[config].keys():
            print("%s, %.1f, %d, %.1f, %d, %.2f, %.1f, %.2f, %.1f" % (config,
                                                                      stats[config][nsources]['bandwidth'],
                                                                      nsources,
                                                                      stats[config][nsources]['s'][0],
                                                                      stats[config][nsources]['J'],
                                                                      stats[config][nsources]['stdphase'],
                                                                      10.0 * numpy.log10(stats[config][nsources]['dr']),
                                                                      stats[config][nsources]['tsky'] / (
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
            tel['RASTERHALO'] = TelArray()
            tel['RASTERHALO'].rasterBoolardy(nhalo=346, name='LOWRASTERHALO', nstations=346, scale=1.05, rhalo=40.0,
                                             rcore=0.0, weight=weight)
            tel[config] = TelArray().add(tel['RASTERHALO'], tel['LOWBD2-CORE'], name=config)
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

    return (tel, mst)

def plot(name, stats, nsources, nnoll):
    plt.clf()
    ymax = 0.1
    Jmax = 0

    for nsource in [nsources]:
        for config in stats.keys():
            y = stats[config][nsource]['s']
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
    plt.title('Singular value spectra: %d sources, max Noll %d' % (nsources, nnoll))

    plt.legend(loc="upper right")
    plt.show()

    plt.savefig('%s.pdf' % name)

    plt.clf()

