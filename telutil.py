import csv
import math
import random

import numpy
import scipy.spatial.distance as sd
from scipy import linalg
from scipy.misc import imread

from mst import *
from sources import sources
from zernikecache import *


# random.seed(781490893)

class TelUtils:
    def uniformcircle(self, n, rhalo=1.0):
        x = numpy.zeros(n)
        y = numpy.zeros(n)
        for i in range(n):
            phi = 2 * numpy.pi * random.random()
            r = rhalo * numpy.sqrt(random.random())
            x[i] = r * numpy.cos(phi)
            y[i] = r * numpy.sin(phi)
        return x, y


class TelMask:
    def _init_(self):
        self.name = ''

    def readMask(self, maskfile='Mask_BoolardyStation.png'):
        self.mask = imread('Mask_BoolardyStation.png')
        self.center = {}
        self.center['x'] = self.mask.shape[0] / 2
        self.center['y'] = self.mask.shape[1] / 2
        self.scale = {}
        self.scale['x'] = 2 * 80.0 / self.mask.shape[0]
        self.scale['y'] = 2 * 80.0 / self.mask.shape[1]

    def masked(self, x, y):
        mx = +int(-y / self.scale['x'] + self.center['x'])
        my = +int(+x / self.scale['y'] + self.center['y'])
        if mx < 0 or mx > self.mask.shape[0] - 1 or my < 0 or my > self.mask.shape[1] - 1:
            return False
        if self.mask[mx, my, 0] == 255:
            return True
        else:
            return False

    def readKML(self, name='BoolardyStation', kmlfile="BoolardyStation2(approx).kml"):
        long0 = 116.779167
        lat0 = -26.789267
        Re = 6371.0
        nsegments = 55
        self.segments = {}
        self.segments['x1'] = numpy.zeros(nsegments)
        self.segments['y1'] = numpy.zeros(nsegments)
        self.segments['x2'] = numpy.zeros(nsegments)
        self.segments['y2'] = numpy.zeros(nsegments)
        self.name = name
        f = open(kmlfile)
        segment = 0
        nextline = False
        for line in f:
            line = line.lstrip()
            if nextline:
                part = line.split(' ')[0].split(',')
                x = float(part[0])
                y = float(part[1])
                self.segments['x1'][segment] = (x - long0) * Re * numpy.pi / (
                    180.0 * numpy.cos(numpy.pi * lat0 / 180.0))
                self.segments['y1'][segment] = (y - lat0) * Re * numpy.pi / (180.0 * numpy.cos(numpy.pi * lat0 / 180.0))
                part = line.split(' ')[1].split(',')
                x = float(part[0])
                y = float(part[1])
                self.segments['x2'][segment] = (x - long0) * Re * numpy.pi / (
                    180.0 * numpy.cos(numpy.pi * lat0 / 180.0))
                self.segments['y2'][segment] = (y - lat0) * Re * numpy.pi / (180.0 * numpy.cos(numpy.pi * lat0 / 180.0))
                nextline = False
                segment = segment + 1
            if line.find('</coordinates>') > -1:
                nextline = False
            elif line.find('<coordinates>') > -1:
                nextline = True

    def plot(self, rmax=20):
        plt.clf()
        plt.fill(self.segments['x1'], self.segments['y1'], fill=False, color='blue')
        plt.axes().set_xlim([-80.0, 80.0])
        plt.axes().set_ylim([-80.0, 80.0])
        plt.axes().set_aspect('equal')
        plt.savefig('Mask_%s_frame.png' % self.name)
        plt.fill(self.segments['x1'], self.segments['y1'], fill=True, color='blue')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().set_frame_on(False)
        plt.show()
        plt.savefig('Mask_%s.png' % self.name)
        plt.show()


class TelArray:
    def _init_(self):
        self.name = ''
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
    
    def plot(self, rmax=40.0, plotfile=''):
        plt.clf()
        plt.title('Antenna locations %s' % self.name)
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.plot(self.stations['x'], self.stations['y'], '.')
        plt.axes().set_aspect('equal')
        circ = plt.Circle((0, 0), radius=rmax, color='g', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circ)
        maxaxis = 1.1 * max(numpy.max(abs(self.stations['x'])), numpy.max(abs(self.stations['y'])))
        plt.axes().set_xlim([-maxaxis, maxaxis])
        plt.axes().set_ylim([-maxaxis, maxaxis])
        mask = TelMask()
        mask.readKML()
        plt.fill(mask.segments['x1'], mask.segments['y1'], fill=False)
        if plotfile == '':
            plotfile = '%s_Array.pdf' % self.name
        plt.show()
        plt.savefig(plotfile)
        print("%s has %d" % (self.name, len(self.stations['x'])))
    
    def add(self, a1, a2, name):
        a = TelArray()
        a.mask = a1.mask
        a.diameter = a1.diameter
        a.nstations = a1.nstations + a2.nstations
        a.stations = {}
        a.stations['x'] = numpy.zeros(a.nstations)
        a.stations['y'] = numpy.zeros(a.nstations)
        a.stations['weight'] = numpy.zeros(a.nstations)
        for key in ['x', 'y', 'weight']:
            a.stations[key][:a1.nstations] = a1.stations[key]
            a.stations[key][a1.nstations:] = a2.stations[key]
        a.name = name
        return a

    def random(self, name='Stations', rhalo=40, rcore=1.0, nstations=512, nhalo=45, nantennas=256, fobs=1e8,
               diameter=35.0, weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.rhalo = rhalo
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        self.center['x'] = 0.0
        self.center['y'] = 0.0
        ncore = self.nstations - nhalo
        self.fobs = fobs
        self.diameter = diameter
        self.stations = {}
        self.stations['x'], self.stations['y'] = TelUtils().uniformcircle(self.nstations, self.rhalo)
        self.stations['x'][:ncore], self.stations['y'][:ncore] = TelUtils().uniformcircle(ncore, self.rcore)
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def coreBoolardy(self, name='CoreBoolardy', rcore=2.5, nstations=512, nantennas=256, fobs=1e8, diameter=35.0,
                     scale=1.0, weight=1.0):
        self.weightpergain = 100.0  # Only use 10 sigma points
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        self.center['x'] = 0.0
        self.center['y'] = 0.0
        self.ncore = nstations
        ncore = self.ncore
        self.fobs = fobs
        self.diameter = diameter
        self.stations = {}
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        self.stations['x'], self.stations['y'] = TelUtils().uniformcircle(self.ncore, self.rcore)
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def rasterBoolardy(self, name='RasterBoolardy', rhalo=40.0, rcore=0.0, nstations=512, nhalo=45, nantennas=256,
                       fobs=1e8, diameter=35.0, scale=1.0, weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.rhalo = rhalo
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        self.center['x'] = 0.0
        self.center['y'] = 0.0
        self.fobs = fobs
        self.diameter = diameter
        self.stations = {}
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        if rcore > 0.0:
            self.ncore = self.nstations - nhalo
            self.stations['x'][:self.ncore], self.stations['y'][:self.ncore] = TelUtils().uniformcircle(self.ncore,
                                                                                                        self.rcore)
        else:
            self.nstations = nhalo
            self.ncore = 0
        ncore = self.ncore
        rmax2 = rhalo * rhalo
        halo = ncore
        # 0.62 is the empirical overlap of Boolardy and a 40km radius circle
        iscale = scale * rhalo * math.sqrt(0.6 * math.pi / (nhalo - 1))
        for x in np.arange(-100 * iscale, +100 * iscale, iscale):
            for y in np.arange(-100 * iscale, +100 * iscale, iscale):
                r = x * x + y * y
                if ((halo < nhalo + ncore) and (r < rmax2) and (not self.mask.masked(x, y))):
                    self.stations['x'][halo] = x
                    self.stations['y'][halo] = y
                    halo = halo + 1
        self.nstations = halo
        self.stations['x'] = self.stations['x'][:self.nstations]
        self.stations['y'] = self.stations['y'][:self.nstations]
        self.stations['weight'] = weight * numpy.ones(self.nstations)
        self.shakehalo(iscale / 2.0)

    def rasterSKA2(self, name='RasterSKA2', rhalo=180.0, rcore=2.5, nstations=155, nhalo=45, nantennas=256, fobs=1e8,
                   diameter=180.0, scale=1.0, weight=1.0):
        self.name = name
        self.rhalo = rhalo
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        if rcore > 0.0:
            self.ncore = self.nstations - nhalo
        else:
            self.nstations = nhalo
            self.ncore = 0
        ncore = self.ncore
        self.stations['x'][:ncore], self.stations['y'][:ncore] = TelUtils().uniformcircle(ncore, self.rcore)
        self.fobs = fobs
        self.diameter = diameter
        self.stations = {}
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        self.stations['x'][:self.ncore], self.stations['y'][:self.ncore] = TelUtils().uniformcircle(self.ncore,
                                                                                                    self.rcore)
        rmax2 = rhalo * rhalo
        halo = self.ncore
        iscale = scale * rhalo * math.sqrt(math.pi / (nhalo - 1))
        for x in np.arange(-50 * iscale, +50 * iscale, iscale):
            for y in np.arange(-50 * iscale, +50 * iscale, iscale):
                r = x * x + y * y
                if (halo < nhalo) and (r < rmax2):
                    self.stations['x'][halo + ncore] = x
                    self.stations['y'][halo + ncore] = y
                    halo = halo + 1
        self.nstations = nhalo
        self.stations['x'] = self.stations['x'][:self.nstations]
        self.stations['y'] = self.stations['y'][:self.nstations]
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def randomBoolardy(self, name='RandomBoolardy', rhalo=40.0, rcore=1.0, nstations=512, nhalo=45, nantennas=256,
                       fobs=1e8, diameter=35.0, weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.rhalo = rhalo
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        self.center['x'] = 0.0
        self.center['y'] = 0.0
        ncore = self.nstations - nhalo
        self.fobs = fobs
        self.diameter = diameter
        self.stations = {}
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        self.stations['x'][:ncore], self.stations['y'][:ncore] = TelUtils().uniformcircle(ncore, self.rcore)
        inhalo = ncore
        self.stations['x'][inhalo] = 0.0
        self.stations['y'][inhalo] = 0.0
        while inhalo < nstations:
            x, y = TelUtils().uniformcircle(1, self.rhalo)
            if not self.mask.masked(x, y):
                r = numpy.sqrt(x * x + y * y)
                if r < rhalo:
                    self.stations['x'][inhalo] = x
                    self.stations['y'][inhalo] = y
                    inhalo = inhalo + 1
        self.nstations = inhalo
        self.stations['x'] = self.stations['x'][:self.nstations]
        self.stations['y'] = self.stations['y'][:self.nstations]
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def circles(self, name='Stations', rhalo=40, rcore=1.0, nstations=512, nhalo=44, fobs=1e8, diameter=35.0,
                weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.rhalo = rhalo
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        self.center['x'] = 0.0
        self.center['y'] = 0.0
        ncore = self.nstations - nhalo
        self.fobs = fobs
        self.diameter = diameter
        if nhalo == 60:
            self.nrings = 4
            self.r = [0.0, rhalo / 3.0, 2 * rhalo / 3.0, rhalo]
            self.nonring = [1, 9, 21, 29]
        elif nhalo == 46:
            self.nrings = 4
            self.r = [0.0, rhalo / 3.0, 2 * rhalo / 3.0, rhalo]
            self.nonring = [1, 9, 13, 23]
        elif nhalo == 30:
            self.nrings = 4
            self.r = [0.0, rhalo / 3.0, 2 * rhalo / 3.0, rhalo]
            self.nonring = [1, 5, 9, 15]
        elif nhalo == 12:
            self.nrings = 3
            self.r = [0.0, rhalo / 2.0, rhalo]
            self.nonring = [1, 5, 6]
        else:
            nhalo = 185
            self.nrings = 7
            self.r = [0.0, rhalo / 6.0, 2 * rhalo / 6.0, 3.0 * rhalo / 6.0, 4.0 * rhalo / 6.0, 5.0 * rhalo / 6.0, rhalo]
            self.nonring = [1, 9, 19, 27, 35, 43, 51]
        self.nstations = nhalo
        self.stations = {}
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        #       self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        station = 0
        for ring in range(self.nrings):
            dphi = 2 * numpy.pi / self.nonring[ring]
            phi = 0.0
            for spoke in range(self.nonring[ring]):
                self.stations['x'][station] = self.r[ring] * numpy.cos(phi)
                self.stations['y'][station] = self.r[ring] * numpy.sin(phi)
                phi = phi + dphi
                station = station + 1
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def circles(self, name='Stations', rhalo=40, rcore=1.0, nstations=512, nhalo=44, fobs=1e8, diameter=35.0,
                weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.rhalo = rhalo
        self.rcore = rcore
        self.nstations = nstations
        self.center = {}
        self.center['x'] = 0.0
        self.center['y'] = 0.0
        ncore = self.nstations - nhalo
        self.fobs = fobs
        self.diameter = diameter
        if nhalo == 60:
            self.nrings = 4
            self.r = [0.0, rhalo / 3.0, 2 * rhalo / 3.0, rhalo]
            self.nonring = [1, 9, 21, 29]
        elif nhalo == 46:
            self.nrings = 4
            self.r = [0.0, rhalo / 3.0, 2 * rhalo / 3.0, rhalo]
            self.nonring = [1, 9, 13, 23]
        elif nhalo == 30:
            self.nrings = 4
            self.r = [0.0, rhalo / 3.0, 2 * rhalo / 3.0, rhalo]
            self.nonring = [1, 5, 9, 15]
        elif nhalo == 12:
            self.nrings = 3
            self.r = [0.0, rhalo / 2.0, rhalo]
            self.nonring = [1, 5, 6]
        else:
            nhalo = 185
            self.nrings = 7
            self.r = [0.0, rhalo / 6.0, 2 * rhalo / 6.0, 3.0 * rhalo / 6.0, 4.0 * rhalo / 6.0, 5.0 * rhalo / 6.0, rhalo]
            self.nonring = [1, 9, 19, 27, 35, 43, 51]
        self.nstations = nhalo
        self.stations = {}
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        #       self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        station = 0
        for ring in range(self.nrings):
            dphi = 2 * numpy.pi / self.nonring[ring]
            phi = 0.0
            for spoke in range(self.nonring[ring]):
                self.stations['x'][station] = self.r[ring] * numpy.cos(phi)
                self.stations['y'][station] = self.r[ring] * numpy.sin(phi)
                phi = phi + dphi
                station = station + 1
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def shakehalo(self, rshake=5.0, one=False):
        newstations = {}
        newstations['x'] = self.stations['x'].copy()
        newstations['y'] = self.stations['y'].copy()
        newstations['x'][0] = 0.0
        newstations['y'][0] = 0.0
        newstations['weight'] = self.stations['weight'].copy()
        if one:
            stations = [int(random.uniform(1.0, self.nstations))]
        else:
            stations = range(1, self.nstations)
        for station in stations:
            cr = numpy.sqrt(self.stations['x'][station] * self.stations['x'][station] + self.stations['y'][station] *
                            self.stations['y'][station])
            if cr > self.rcore:
                phi = 2.0 * numpy.pi * random.random()
                r = rshake * numpy.sqrt(random.random())
                x = newstations['x'][station] + r * numpy.cos(phi)
                y = newstations['y'][station] + r * numpy.sin(phi)
                rdr = numpy.sqrt(x * x + y * y)
                if rdr < self.rhalo:
                    if not self.mask.masked(x, y):
                        newstations['x'][station] = x
                        newstations['y'][station] = y
        self.stations = newstations

    def addcore(self, weight=1.0):
        numpy.append(self.stations['x'], [0.0])
        numpy.append(self.stations['y'], [0.0])
        numpy.append(self.stations['weight'], [0.0])
        print("Extended array to %d stations including the single core" % self.nstations)

    def readCSV(self, name='LOWBD', rcore=0.0, csvfile='SKA-low_config_baseline_design_arm_stations_2013apr30.csv',
                rhalo=40, recenter=False, weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.nstations = 1
        self.stations = {}
        self.rhalo = rhalo
        self.fobs = 1e8
        self.diameter = 35.0
        meanx = 0
        meany = 0
        with open(csvfile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                meanx = meanx + float(row[0])
                meany = meany + float(row[1])
                self.nstations = self.nstations + 1
        meanx = meanx / self.nstations
        meany = meany / self.nstations
        if not recenter:
            meanx = 0.0
            meany = 0.0
        f.close()
        self.nstations = 0
        scale = 0.001
        with open(csvfile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x = scale * (float(row[0]) - meanx)
                y = scale * (float(row[1]) - meany)
                r = numpy.sqrt(x * x + y * y)
                if r > rcore:
                    self.nstations = self.nstations + 1
        
        f.close()
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = weight * numpy.ones(self.nstations)
        station = 0
        with open(csvfile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x = scale * (float(row[0]) - meanx)
                y = scale * (float(row[1]) - meany)
                r = numpy.sqrt(x * x + y * y)
                if r > rcore:
                    self.stations['x'][station] = x
                    self.stations['y'][station] = y
                    station = station + 1
        self.nstations = station
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def readLOWBD(self, name='LOWBD', rcore=0.0, csvfile='SKA-low_config_baseline_design_arm_stations_2013apr30.csv',
                  weight=1.0):
        return self.readCSV(name, rcore, csvfile, weight=weight)

    def saveCSV(self, filename='LOWBD.csv'):
        with open(filename, 'wb') as fp:
            rowwriter = csv.writer(fp)
            for station in range(self.nstations):
                rowwriter.writerow([1000.0 * self.stations['x'][station], 1000.0 * self.stations['y'][station]])

    def writeWGS84(self, filename='LOWBD.csv'):
        long0 = 116.779167
        lat0 = -26.789267
        height0 = 300.0
        Re = 6371.0
        with open(filename, 'wb') as fp:
            rowwriter = csv.writer(fp)
            rowwriter.writerow(['name', 'longitude', 'latitude', 'height'])
            for station in range(self.nstations):
                name = 'Low%d' % station
                long = long0 + 180.0 * (self.stations['x'][station]) * numpy.cos(numpy.pi * lat0 / 180.0) / (
                    Re * numpy.pi)
                lat = lat0 + 180.0 * (self.stations['y'][station]) * numpy.cos(numpy.pi * lat0 / 180.0) / (
                    Re * numpy.pi)
                rowwriter.writerow([name, long, lat, height0])

    def readLOWL1(self, name='LOWL1', rcore=0.0, csvfile='L1_configuration.csv', weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name = name
        self.nstations = 0
        self.stations = {}
        self.rhalo = 80
        self.fobs = 1e8
        self.diameter = 35.0
        meanx = 0
        meany = 0
        with open(csvfile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                meanx = meanx + float(row[1])
                meany = meany + float(row[0])
                self.nstations = self.nstations + 1
        meanx = meanx / self.nstations
        meany = meany / self.nstations
        f.close()
        self.nstations = 0
        scale = 6.371e6 * numpy.pi / 180000.0
        with open(csvfile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x = scale * (float(row[1]) - meanx) * numpy.cos(meanx * numpy.pi / 180.0)
                y = scale * (float(row[0]) - meany)
                r = numpy.sqrt(x * x + y * y)
                if r > rcore:
                    self.nstations = self.nstations + 1
        
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = weight * numpy.ones(self.nstations)
        station = 0
        scale = 6.371e6 * numpy.pi / 180000.0
        with open(csvfile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x = scale * (float(row[1]) - meanx) * numpy.cos(meanx * numpy.pi / 180.0)
                y = scale * (float(row[0]) - meany)
                r = numpy.sqrt(x * x + y * y)
                if r > rcore:
                    self.stations['x'][station] = x
                    self.stations['y'][station] = y
                    self.stations['weight'] = weight * numpy.ones(self.nstations)
                    station = station + 1

    def readLOFAR(self, name='LOFAR', stationtype='S', band='HBA', lfdef='LOFAR.csv', lat=52.7, weight=1):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        cs = numpy.cos(numpy.pi * lat / 180.0)
        sn = numpy.sin(numpy.pi * lat / 180.0)
        nantennas = 256
        self.name = name
        self.nstations = 0
        self.stations = {}
        self.rhalo = 80
        self.fobs = 1e8
        self.diameter = 35.0
        meanx = 0
        meany = 0
        meanz = 0
        with open(lfdef, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                type = row[0]
                if (type.find(stationtype) > -1) and (type.find(band) > -1):
                    meanx = meanx + float(row[2])
                    meany = meany + float(row[1])
                    meanz = meanz + float(row[3])
                    self.nstations = self.nstations + 1
        meanx = meanx / self.nstations
        meany = meany / self.nstations
        meanz = meanz / self.nstations
        f.close()
        station = 0
        self.stations['x'] = numpy.zeros(self.nstations)
        self.stations['y'] = numpy.zeros(self.nstations)
        self.stations['weight'] = numpy.zeros(self.nstations)
        with open(lfdef, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                type = row[0]
                if (type.find(stationtype) > -1) and (type.find(band) > -1):
                    x = (float(row[2]) - meanx) / 1000.0
                    y = (float(row[1]) - meany) / 1000.0
                    z = (float(row[3]) - meanz) / 1000.0
                    self.stations['x'][station] = x
                    self.stations['y'][station] = -cs * y + sn * z
                    station = station + 1
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def readKML(self, name='LOWBD2', kmlfile="V4Drsst512red_2", diameter=35.0, weight=1.0):
        self.mask = TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        long0 = 116.779167
        lat0 = -26.789267
        Re = 6371.0
        self.stations = {}
        self.stations['x'] = numpy.zeros(512)
        self.stations['y'] = numpy.zeros(512)
        self.stations['weight'] = weight * numpy.ones(512)
        self.nstations = 512
        self.name = name
        self.diameter = diameter
        f = open(kmlfile)
        self.nstations = 512
        station = 0
        for line in f:
            line = line.lstrip()
            if line.find("name") > 0:
                if line.find("Station") > 0:
                    station = int(line.split('Station')[1].split('<')[0])
                if line.find("Antenna") > 0:
                    station = int(line.split('Antenna')[1].split('<')[0])
            if line.find("coordinates") > 0:
                x = float(line.split('>')[1].split('<')[0].split(',')[0])
                y = float(line.split('>')[1].split('<')[0].split(',')[1])
                self.stations['x'][station] = (x - long0) * Re * numpy.pi / (180.0 * numpy.cos(numpy.pi * lat0 / 180.0))
                self.stations['y'][station] = (y - lat0) * Re * numpy.pi / (180.0 * numpy.cos(numpy.pi * lat0 / 180.0))
        self.stations['weight'] = weight * numpy.ones(self.nstations)

    def writeKML(self, kmlfile="LOW_CIRCLES.kml"):

        long0 = 116.779167
        lat0 = -26.789267
        height0 = 300.0
        Re = 6371.0
        s = ['<?xml version="1.0" encoding="UTF-8"?>', \
             '<kml xmlns="http://www.opengis.net/kml/2.2">', \
             '<Document>', \
             '<Style id="whitecirc">', \
             '<IconStyle>', \
             '<Icon>', \
             '<href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>', \
             '</Icon>', \
             '</IconStyle>', \
             '</Style>', \
             '<!--name></name-->']
        l = ['<Placemark>', \
             '<styleUrl>#whitecirc</styleUrl>', \
             '<name>S%d</name>', \
             '<Point>', \
             '<coordinates>%f, %f</coordinates>', \
             '</Point>', \
             '</Placemark>']
        e = ['</Document>', '</kml>']
        f = open(kmlfile, 'w')
        for ss in s:
            f.write(ss)
        for station in range(self.nstations):
            long = long0 + 180.0 * (self.stations['x'][station]) * numpy.cos(numpy.pi * lat0 / 180.0) / (Re * numpy.pi)
            lat = lat0 + 180.0 * (self.stations['y'][station]) * numpy.cos(numpy.pi * lat0 / 180.0) / (Re * numpy.pi)
            f.write(l[0])
            f.write(l[1])
            f.write(l[2] % station)
            f.write(l[3])
            f.write(l[4] % (long, lat))
            f.write(l[5])
            f.write(l[6])
        f.write(e[0])
        f.write(e[1])

    def excessDistance(self, MST=False):
        if MST:
            return 2.0 * self.distance() - self.mst(False) / float(self.nstations)
        else:
            return self.distance()
            # We maximise the minimum distance between stations

    def distance(self):
        P = numpy.zeros([self.nstations, 2])
        P[..., 0] = self.stations['x']
        P[..., 1] = self.stations['y']
        distancemat = sd.pdist(P)
        distance = numpy.min(distancemat)
        return distance

    # Evaluate the minimum spanning tree
    def mst(self, doplot=True, plotfile=''):
        P = numpy.zeros([self.nstations, 2])
        P[..., 0] = self.stations['x']
        P[..., 1] = self.stations['y']
        X = sd.squareform(sd.pdist(P))
        edge_list = minimum_spanning_tree(X)
        if doplot:
            plt.clf()
            plt.scatter(P[:, 0], P[:, 1])
            dist = 0
            for edge in edge_list:
                i, j = edge
                plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
                dist = dist + numpy.sqrt(
                    (P[i, 0] - P[j, 0]) * (P[i, 0] - P[j, 0]) + (P[i, 1] - P[j, 1]) * (P[i, 1] - P[j, 1]))
            plt.title('%s, MST=%.1f km' % (self.name, dist))
            plt.xlabel('X (km)')
            plt.ylabel('y (km)')
            plt.axes().set_aspect('equal')
            maxaxis = numpy.max(abs(P))
            plt.axes().set_xlim([-maxaxis, maxaxis])
            plt.axes().set_ylim([-maxaxis, maxaxis])
            mask = TelMask()
            mask.readKML()
            plt.fill(mask.segments['x1'], mask.segments['y1'], fill=False)
            if plotfile == '':
                plotfile = '%s_MST.pdf' % self.name
            plt.show()
            plt.savefig(plotfile)
            return dist
        else:
            dist = 0
            for edge in edge_list:
                i, j = edge
                dist = dist + numpy.sqrt(
                    (P[i, 0] - P[j, 0]) * (P[i, 0] - P[j, 0]) + (P[i, 1] - P[j, 1]) * (P[i, 1] - P[j, 1]))
            return dist


class TelUV:
    def _init_(self):
        self.name = ''
        self.uv = False
    
    def construct(self, t):
        self.name = t.name
        self.nbaselines = t.nstations * t.nstations
        self.uv = {}
        self.uv['x'] = numpy.zeros(self.nbaselines)
        self.uv['y'] = numpy.zeros(self.nbaselines)
        for station in range(t.nstations):
            self.uv['x'][station * t.nstations:(station + 1) * t.nstations] = t.stations['x'] - t.stations['x'][station]
            self.uv['y'][station * t.nstations:(station + 1) * t.nstations] = t.stations['y'] - t.stations['y'][station]

    def plot(self, plotfile=''):
        self.plotter = True
        plt.clf()
        plt.title('UV Sampling %s' % self.name)
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.plot(self.uv['x'], self.uv['y'], '.')
        plt.axes().set_aspect('equal')
        if plotfile == '':
            plotfile = 'UVcoverage_%s.pdf' % self.name
        plt.show()
        plt.savefig(plotfile)

    def assess(self):
        return 1.0


#
# Sources on the celestial sphere.
#
class TelSources:
    def _init_(self):
        self.name = 'Sources'
        self.nsources = 100
    
    def construct(self, name='Sources', nsources=100, radius=1, smin=0.66545, limit=0.95):
        """ Construct nsources sources above the flux limit
        """
        self.name = name
        # Make more sources than we need
        self.sources = {}
        self.sources['x'], self.sources['y'] = TelUtils().uniformcircle(100 * nsources, radius)
        x = self.sources['x']
        y = self.sources['y']
        r = numpy.sqrt(x * x + y * y)
        pb = numpy.exp(numpy.log(0.01) * (r / radius) ** 2)  # Model out to 1% of PB
        pb[r > 0.95 * radius] = 0.0
        f = pb * sources().randomsources(smin=smin, nsources=100 * nsources)
        # Avoid fields with bright sources
        x = x[f < 10.0 * smin]
        y = y[f < 10.0 * smin]
        f = f[f < 10.0 * smin]
        x = x[f > smin]
        y = y[f > smin]
        f = f[f > smin]
        self.sources['x'] = x[:nsources]
        self.sources['y'] = y[:nsources]
        self.sources['flux'] = f[:nsources]
        print("Source fluxes = %s" % sorted(self.sources['flux']))
        self.nsources = nsources
        self.radius = radius

    def plot(self):
        self.plotter = True
    
    def assess(self):
        return 1.0


#
# Piercings through the ionosphere
#
class TelPiercings:
    def _init_(self):
        self.name = 'Piercings'
        self.npiercings = 0
        self.hiono = 400
    
    def plot(self, rmax=70):
        plt.clf()
        plt.title(self.name)
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        x = self.piercings['x'][self.piercings['flux'] > 0.0]
        y = self.piercings['y'][self.piercings['flux'] > 0.0]
        r2 = x * x + y * y
        plt.plot(x, y, ',')
        plt.axes().set_aspect('equal')
        circ = plt.Circle((0, 0), radius=rmax, color='g', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circ)
        plt.show()
        plt.savefig('%s.pdf' % self.name)

        # weight in the inverse variance of the visibility noise in 10s

    def construct(self, sources, array, weight, hiono=300):
        self.hiono = hiono
        self.npiercings = sources.nsources * array.nstations
        self.name = '%s_sources%d_PC' % (array.name, sources.nsources)
        self.piercings = {}
        nstations = array.nstations
        self.piercings['x'] = numpy.zeros(self.npiercings)
        self.piercings['y'] = numpy.zeros(self.npiercings)
        self.piercings['weight'] = numpy.zeros(self.npiercings)
        self.piercings['flux'] = numpy.zeros(self.npiercings)
        for source in range(sources.nsources):
            self.piercings['x'][source * nstations:(source + 1) * nstations] = self.hiono * sources.sources['x'][
                source] + array.stations['x']
            self.piercings['y'][source * nstations:(source + 1) * nstations] = self.hiono * sources.sources['y'][
                source] + array.stations['y']
            self.piercings['flux'][source * nstations:(source + 1) * nstations] = sources.sources['flux'][source]
            self.piercings['weight'][source * nstations:(source + 1) * nstations] = \
                (sources.sources['flux'][source]) ** 2 * array.stations['weight']

    def assess(self, nnoll=20, rmax=40.0, doplot=True, limit=0.95):
        x = self.piercings['x']
        y = self.piercings['y']
        r = numpy.sqrt(x * x + y * y)
        weight = numpy.sqrt(self.piercings['weight'])[r < limit * rmax]
        x = x[r < limit * rmax]
        y = y[r < limit * rmax]
        r = numpy.sqrt(x * x + y * y)
        phi = numpy.arctan2(y, x)
        self.npiercings = len(r)
        A = numpy.zeros([len(r), nnoll], dtype='float')
        for noll in range(nnoll):
            A[:, noll] = weight * zernikel(noll, r / rmax, phi)

        print("Maximum in A matrix = %.1f" % numpy.max(A))
        Covar_A = numpy.zeros([nnoll, nnoll])
        print("Shape of A = %s" % str(A.shape))
        Covar_A = numpy.dot(A.T, A)
        print("Shape of A^T A = %s" % str(Covar_A.shape))
        # The singular value analysis is relatively cheap
        U, s, Vh = linalg.svd(Covar_A)
        
        if doplot:
            plt.clf()
            plt.title('%s A' % (self.name))
            plt.xlabel('Noll number')
            plt.ylabel('Piercing number')
            plt.imshow(numpy.sqrt(abs(A)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.show()
            plt.savefig('%s_A.pdf' % (self.name))
            plt.clf()
            plt.title('%s Covar A' % (self.name))
            plt.xlabel('Noll number')
            plt.ylabel('Noll number')
            plt.imshow(numpy.sqrt(abs(Covar_A)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.show()
            plt.savefig('%s_Covar_A.pdf' % (self.name))
            plt.clf()
            plt.title('%s U' % (self.name))
            plt.ylabel('Noll number')
            plt.xlabel('Noll number')
            plt.imshow(numpy.sqrt(abs(U)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.show()
            plt.savefig('%s_U.pdf' % (self.name))
        return np.array(s)


class TelArrayPiercing:
    def _init_(self):
        self.name = 'ArrayPiercings'
        self.npiercings = 0
        self.hiono = 300

    def assess(self, sources, array, rmax=40.0, nnoll=100, wavelength=3.0, hiono=300, weight=1.0, limit=0.9,
               rmin=0.3, doplot=True, doFresnel=True):
        nstations = array.nstations
        nconstraints = 2 * nstations * nstations
        print("Will generate %d constraint equations" % (nconstraints))
        self.name = '%s_sources%d_PC' % (array.name, sources.nsources)
        if doFresnel:
            print("Using Fraunhoffer and Fresnel terms")
        else:
            print("Fraunhoffer term only")

        x = array.stations['x']
        y = array.stations['y']

        P = numpy.zeros([len(x), len(x)])
        for station in range(len(x)):
            P[:, station] = numpy.sqrt((x - x[station]) ** 2 + (y - y[station]) ** 2)

        flux = sources.sources['flux']
        l = sources.sources['x']
        m = sources.sources['y']

        # Calculate phase from source to station: this loop is inexpensive
        phasor = numpy.ones([sources.nsources, nstations], dtype='complex')
        for source in range(sources.nsources):
            aphase = + 2.0 * 1000.0 * numpy.pi * (x * l[source] + y * m[source]) / wavelength
            if doFresnel:
                fx = hiono * l[source] + x
                fy = hiono * m[source] + y
                fphase = + 1000.0 * numpy.pi * (fx ** 2 + fy ** 2) / (wavelength * hiono)
                phasor[source, :] = numpy.sqrt(flux[source]) * numpy.exp(+1j * (aphase + fphase))
            else:
                phasor[source, :] = numpy.sqrt(flux[source]) * numpy.exp(+1j * aphase)
        print('Finished calculating phasors')

        # Calculate the summed visibility: this is very time consuming since it uses the
        # phase from each source to each station, multiplied by the appropriate Zernicke.
        # We do this by station and accumulate the covariance matrix so as not to form
        # a huge design matrix all at once.
        for station in range(nstations):
            print('Calculating station %d to all other stations' % (station))
            A = numpy.zeros([nnoll, 2, nstations])
            for noll in range(nnoll):
                for source in range(sources.nsources):
                    dist=numpy.sqrt((x-x[station])**2+(y-y[station])**2)
                    fx = hiono * l[source] + x
                    fy = hiono * m[source] + y
                    r = numpy.sqrt(fx ** 2 + fy ** 2)
                    phi = numpy.arctan2(fy, fx)
                    # This is a time sink unless factorial is cached
                    z = zernikel(noll, r / rmax, phi)
                    z[r > rmax] = 0.0
                    # All the previous operations are per station. In the code below we
                    # make the transition to baselines
                    vphasor = weight * phasor[source, :] * z * numpy.conj(phasor[source, station])
                    vphasor[dist < rmin] = 0.0
                    # In this approach, we only get access to the summed effects of all sources
                    # as seen through the screen and correlated with another station.
                    A[noll, 0, :] += numpy.real(vphasor)
                    A[noll, 1, :] += numpy.imag(vphasor)
            # Reshape so that the dot product can work. This sums over the last chunk of
            # visibilities.
            A = numpy.reshape(A, [nnoll, 2 * nstations])
            if station == 0:
                Covar_A = numpy.dot(A, A.T)
            else:
                Covar_A += numpy.dot(A, A.T)

        print("Shape of A A^T = %s" % str(Covar_A.shape))
        # The singular value analysis is relatively cheap
        print("Performing SVD of A A^T")
#        U, s, Vh = linalg.svd(Covar_A)
        s, U = linalg.eig(Covar_A)
        s = numpy.real(s)
        s[s<0.0]=0.0
        U = numpy.real(U)

        if doplot:
            plt.clf()
            plt.title('%s Covar A' % (self.name))
            plt.xlabel('Noll number')
            plt.ylabel('Noll number')
            plt.imshow(numpy.sqrt(abs(Covar_A)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.show()
            plt.savefig('%s_Covar_A.pdf' % (self.name))
            plt.clf()
            plt.title('%s U' % (self.name))
            plt.ylabel('Noll number')
            plt.xlabel('Noll number')
            plt.imshow(numpy.sqrt(abs(U)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.show()
            plt.savefig('%s_U.pdf' % (self.name))

        return np.array(s)
