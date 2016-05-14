import numpy
import math
import scipy
import matplotlib
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML
import matplotlib.pylab as pylab
import csv
import zernike
import numpy as np
import scipy.spatial.distance as sd 
from scipy import linalg
from mst import * 

#random.seed(781490893)

class TelIono:
    def dr(self, J=1, r0=14.0, B=80.0, tobs=10000.*3600.0, tiono=10.0):
        return 3.397 * numpy.power(J, math.sqrt(3)/4.0) * numpy.power(B/r0, -5.0/6.0) * numpy.sqrt(tobs/tiono)
    def ionosphere(self, baseline):
        return numpy.power(baseline/14.0,+1.8/2.0)/1.5
    def tobs(self, J, DR=1e5, r0=14.0, B=80.0, tobs=10000.*3600.0, tiono=10.0):
        return tiono * (DR/3.397) * (DR/3.397) * numpy.power(J, -math.sqrt(3)/2.0) * numpy.power(B/r0, 5.0/6.0)

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
        
# Spot values from L1
    def noise(self):
        return {'50':25.1e-6, '110':3.1e-6, '160':3.4e-6, '220':3.4e-6} 

#  Simpler version
    def tnoise(self, freq=1e8, time=10000.0*3600.0):
        scale=numpy.sqrt(10000.0*3600.0/time)
        if freq<7.5e7:
            return  scale*25.1e-6
        else:
            return  scale*3.1e-6

class TelUtils:

    def uniformcircle(self, n, rhalo=1.0):
        x=numpy.zeros(n)
        y=numpy.zeros(n)
        for i in range(n):
            phi=2*numpy.pi*random.random()
            r=rhalo*numpy.sqrt(random.random())
            x[i]=r*numpy.cos(phi)
            y[i]=r*numpy.sin(phi)
        return x, y
        
class TelMask:
    def _init_(self):
        self.name=''
        
    def readMask(self, maskfile='Mask_BoolardyStation.png'):
        self.mask = scipy.misc.imread('Mask_BoolardyStation.png')
        self.center={}
        self.center['x']=self.mask.shape[0]/2
        self.center['y']=self.mask.shape[1]/2
        self.scale={}
        self.scale['x']=2*80.0/self.mask.shape[0]
        self.scale['y']=2*80.0/self.mask.shape[1]
        
    def masked(self, x, y):
        mx=+int(-y/self.scale['x']+self.center['x'])
        my=+int(+x/self.scale['y']+self.center['y'])
        if mx < 0 or mx > self.mask.shape[0]-1 or my < 0 or my > self.mask.shape[1]-1:
            return False
        if  self.mask[mx,my,0] == 255:
            return True
        else:
            return False
        

    def readKML(self, name='BoolardyStation', kmlfile="BoolardyStation2(approx).kml"):
        self.weightpergain=100.0 # Only use 10 sigma points    
        long0=116.779167
        lat0=-26.789267
        Re=6371.0
        nsegments=55
        self.segments={}
        self.segments['x1']=numpy.zeros(nsegments)
        self.segments['y1']=numpy.zeros(nsegments)
        self.segments['x2']=numpy.zeros(nsegments)
        self.segments['y2']=numpy.zeros(nsegments)
        self.name=name
        f=open(kmlfile)
        segment=0
        nextline=False
        for line in f:
            line=line.lstrip()
            if nextline:
                part=line.split(' ')[0].split(',')
                x=float(part[0])
                y=float(part[1])
                self.segments['x1'][segment]=(x-long0)*Re*numpy.pi/(180.0*numpy.cos(numpy.pi*lat0/180.0))
                self.segments['y1'][segment]=(y-lat0)*Re*numpy.pi/(180.0*numpy.cos(numpy.pi*lat0/180.0))
                part=line.split(' ')[1].split(',')
                x=float(part[0])
                y=float(part[1])
                self.segments['x2'][segment]=(x-long0)*Re*numpy.pi/(180.0*numpy.cos(numpy.pi*lat0/180.0))
                self.segments['y2'][segment]=(y-lat0)*Re*numpy.pi/(180.0*numpy.cos(numpy.pi*lat0/180.0))
                nextline=False
                segment=segment+1
            if line.find('</coordinates>') > -1:
                nextline=False
            elif line.find('<coordinates>') > -1:
                nextline=True
                
    def plot(self, rmax=20):
        plt.clf()
        plt.fill(self.segments['x1'], self.segments['y1'],fill=False,color='blue')
        plt.axes().set_xlim([-80.0,80.0])
        plt.axes().set_ylim([-80.0,80.0])
        plt.axes().set_aspect('equal')
        plt.savefig('Mask_%s_frame.png' % self.name)
        plt.fill(self.segments['x1'], self.segments['y1'],fill=True,color='blue')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().set_frame_on(False)
        plt.show()
        plt.savefig('Mask_%s.png' % self.name)
        plt.show()

class TelArray:

    def _init_(self):
        self.name=''
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
    
    def plot(self, rmax=40.0, plotfile=''):
        plt.clf()
        plt.title('Antenna locations %s' % self.name)
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.plot(self.stations['x'], self.stations['y'], '.')
        plt.axes().set_aspect('equal')
        circ=plt.Circle((0,0), radius=rmax, color='g', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circ)
        maxaxis=1.1*max(numpy.max(abs(self.stations['x'])), numpy.max(abs(self.stations['y'])))
        plt.axes().set_xlim([-maxaxis,maxaxis])
        plt.axes().set_ylim([-maxaxis,maxaxis])
        mask=TelMask()
        mask.readKML()
        plt.fill(mask.segments['x1'], mask.segments['y1'], fill=False)
        if plotfile=='':
            plotfile='%s_Array.pdf' % self.name
        plt.show()
        plt.savefig(plotfile)
    
    def add(self, a1, a2, name):
        a=TelArray()
        a.mask=a1.mask
        a.diameter=a1.diameter
        a.nstations=a1.nstations+a2.nstations
        a.stations={}
        print a1.nstations, a2.nstations, a.nstations
        a.stations['x']=numpy.zeros(a.nstations)
        a.stations['y']=numpy.zeros(a.nstations)
        a.stations['weight']=numpy.zeros(a.nstations)
        for key in ['x', 'y', 'weight']:
            a.stations[key][:a1.nstations]=a1.stations[key]
            a.stations[key][a1.nstations:]=a2.stations[key]
        a.name=name
        return a
        
    def random(self, name='Stations', rhalo=40, rcore=1.0, nstations=512, nhalo=45, nantennas=256, fobs=1e8, diameter=35.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.rhalo=rhalo
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        self.center['x']=0.0
        self.center['y']=0.0
        ncore=self.nstations-nhalo
        self.fobs=fobs
        self.diameter=diameter
        self.stations={}
        self.stations['x'], self.stations['y']=TelUtils().uniformcircle(self.nstations, self.rhalo)
        self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        self.stations['weight']=numpy.ones(self.nstations)

    def coreBoolardy(self, name='CoreBoolardy', rcore=2.5, nstations=512, nantennas=256, fobs=1e8, diameter=35.0, scale=1.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        self.center['x']=0.0
        self.center['y']=0.0
        self.ncore=nstations
        ncore=self.ncore
        self.fobs=fobs
        self.diameter=diameter
        self.stations={}
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
        self.stations['x'], self.stations['y']=TelUtils().uniformcircle(self.ncore, self.rcore)
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)

    def rasterBoolardy(self, name='RasterBoolardy', rhalo=40.0, rcore=0.0, nstations=512, nhalo=45, nantennas=256, fobs=1e8, diameter=35.0, scale=1.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.rhalo=rhalo
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        self.center['x']=0.0
        self.center['y']=0.0
        self.fobs=fobs
        self.diameter=diameter
        self.stations={}
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
        if rcore>0.0:
            self.ncore=self.nstations-nhalo
            self.stations['x'][:self.ncore], self.stations['y'][:self.ncore]=TelUtils().uniformcircle(self.ncore, self.rcore)
        else:
            self.nstations=nhalo
            self.ncore=0
        ncore=self.ncore
        rmax2=rhalo*rhalo
        halo=ncore
        # 0.62 is the empirical overlap of Boolardy and a 40km radius circle
        iscale=scale*rhalo*math.sqrt(0.6*math.pi/(nhalo-1))
        for x in np.arange(-100*iscale,+100*iscale,iscale):
            for y in np.arange(-100*iscale,+100*iscale,iscale):
                r=x*x+y*y
                if ((halo < nhalo+ncore) and (r < rmax2) and (not self.mask.masked(x,y))):
                    self.stations['x'][halo]=x
                    self.stations['y'][halo]=y
                    halo=halo+1
        self.nstations=halo
        self.stations['x']=self.stations['x'][:self.nstations]
        self.stations['y']=self.stations['y'][:self.nstations]
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)
        self.shakehalo(iscale/4.0)

    def rasterSKA2(self, name='RasterSKA2', rhalo=180.0, rcore=2.5, nstations=155, nhalo=45, nantennas=256, fobs=1e8, diameter=180.0, scale=1.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.name=name
        self.rhalo=rhalo
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        if rcore>0.0:
            self.ncore=self.nstations-nhalo
        else:
            self.nstations=nhalo
            self.ncore=0
        ncore=self.ncore
        self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        self.fobs=fobs
        self.diameter=diameter
        self.stations={}
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
        self.stations['x'][:self.ncore], self.stations['y'][:self.ncore]=TelUtils().uniformcircle(self.ncore, self.rcore)
        rmax2=rhalo*rhalo
        halo=self.ncore
        iscale=scale*rhalo*math.sqrt(math.pi/(nhalo-1))
        for x in np.arange(-50*iscale,+50*iscale,iscale):
            for y in np.arange(-50*iscale,+50*iscale,iscale):
                r=x*x+y*y
                if (halo < nhalo) and (r < rmax2):
                    self.stations['x'][halo+ncore]=x
                    self.stations['y'][halo+ncore]=y
                    halo=halo+1
        self.nstations=nhalo
        self.stations['x']=self.stations['x'][:self.nstations]
        self.stations['y']=self.stations['y'][:self.nstations]
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)

    def randomBoolardy(self, name='RandomBoolardy', rhalo=40.0, rcore=1.0, nstations=512, nhalo=45, nantennas=256, fobs=1e8, diameter=35.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.rhalo=rhalo
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        self.center['x']=0.0
        self.center['y']=0.0
        ncore=self.nstations-nhalo
        self.fobs=fobs
        self.diameter=diameter
        self.stations={}
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
        self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        inhalo=ncore
        self.stations['x'][inhalo]=0.0
        self.stations['y'][inhalo]=0.0
        while inhalo < nstations:
            x, y=TelUtils().uniformcircle(1, self.rhalo)
            if not self.mask.masked(x,y):
                r=numpy.sqrt(x*x+y*y)
                if r<rhalo:
                    self.stations['x'][inhalo]=x
                    self.stations['y'][inhalo]=y
                    inhalo=inhalo+1
        self.nstations=inhalo
        self.stations['x']=self.stations['x'][:self.nstations]
        self.stations['y']=self.stations['y'][:self.nstations]
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)
       
    def circles(self, name='Stations', rhalo=40, rcore=1.0, nstations=512, nhalo=44, fobs=1e8, diameter=35.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.rhalo=rhalo
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        self.center['x']=0.0
        self.center['y']=0.0
        ncore=self.nstations-nhalo
        self.fobs=fobs
        self.diameter=diameter
        if nhalo==60:
            self.nrings=4
            self.r=[0.0, rhalo/3.0, 2*rhalo/3.0, rhalo]
            self.nonring=[1, 9, 21, 29]
        elif nhalo==46:
            self.nrings=4
            self.r=[0.0, rhalo/3.0, 2*rhalo/3.0, rhalo]
            self.nonring=[1, 9, 13, 23]
        elif nhalo==30:
            self.nrings=4
            self.r=[0.0, rhalo/3.0, 2*rhalo/3.0, rhalo]
            self.nonring=[1, 5, 9, 15]
        elif nhalo==12:
            self.nrings=3
            self.r=[0.0, rhalo/2.0, rhalo]
            self.nonring=[1, 5, 6]
        else:
            nhalo=185
            self.nrings=7
            self.r=[0.0, rhalo/6.0, 2*rhalo/6.0, 3.0*rhalo/6.0, 4.0*rhalo/6.0, 5.0*rhalo/6.0, rhalo]
            self.nonring=[1, 9, 19, 27, 35, 43, 51]
        self.nstations=nhalo
        self.stations={}
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
#       self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        station=0
        for ring in range(self.nrings):
            dphi=2*numpy.pi/self.nonring[ring]
            phi=0.0
            for spoke in range(self.nonring[ring]):
                self.stations['x'][station]=self.r[ring]*numpy.cos(phi)
                self.stations['y'][station]=self.r[ring]*numpy.sin(phi)
                phi=phi+dphi
                station=station+1
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)


    def circles(self, name='Stations', rhalo=40, rcore=1.0, nstations=512, nhalo=44, fobs=1e8, diameter=35.0):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.rhalo=rhalo
        self.rcore=rcore
        self.nstations=nstations
        self.center={}
        self.center['x']=0.0
        self.center['y']=0.0
        ncore=self.nstations-nhalo
        self.fobs=fobs
        self.diameter=diameter
        if nhalo==60:
            self.nrings=4
            self.r=[0.0, rhalo/3.0, 2*rhalo/3.0, rhalo]
            self.nonring=[1, 9, 21, 29]
        elif nhalo==46:
            self.nrings=4
            self.r=[0.0, rhalo/3.0, 2*rhalo/3.0, rhalo]
            self.nonring=[1, 9, 13, 23]
        elif nhalo==30:
            self.nrings=4
            self.r=[0.0, rhalo/3.0, 2*rhalo/3.0, rhalo]
            self.nonring=[1, 5, 9, 15]
        elif nhalo==12:
            self.nrings=3
            self.r=[0.0, rhalo/2.0, rhalo]
            self.nonring=[1, 5, 6]
        else:
            nhalo=185
            self.nrings=7
            self.r=[0.0, rhalo/6.0, 2*rhalo/6.0, 3.0*rhalo/6.0, 4.0*rhalo/6.0, 5.0*rhalo/6.0, rhalo]
            self.nonring=[1, 9, 19, 27, 35, 43, 51]
        self.nstations=nhalo
        self.stations={}
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
#       self.stations['x'][:ncore], self.stations['y'][:ncore]=TelUtils().uniformcircle(ncore, self.rcore)
        station=0
        for ring in range(self.nrings):
            dphi=2*numpy.pi/self.nonring[ring]
            phi=0.0
            for spoke in range(self.nonring[ring]):
                self.stations['x'][station]=self.r[ring]*numpy.cos(phi)
                self.stations['y'][station]=self.r[ring]*numpy.sin(phi)
                phi=phi+dphi
                station=station+1
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)

    def shakehalo(self, rshake=5.0, one=False):
        self.weightpergain=100.0 # Only use 10 sigma points
        newstations={}
        newstations['x']=self.stations['x'].copy()
        newstations['y']=self.stations['y'].copy()
        newstations['x'][0]=0.0
        newstations['y'][0]=0.0
        newstations['weight']=self.stations['weight'].copy()
        if one:
            stations=[int(random.uniform(1.0, self.nstations))]
        else:
            stations=range(1,self.nstations)
        for station in stations:
            cr=numpy.sqrt(self.stations['x'][station]*self.stations['x'][station]+self.stations['y'][station]*self.stations['y'][station])
            if cr>self.rcore:
                phi=2.0*numpy.pi*random.random()
                r=rshake*numpy.sqrt(random.random())
                x=newstations['x'][station]+r*numpy.cos(phi)
                y=newstations['y'][station]+r*numpy.sin(phi)
                rdr=numpy.sqrt(x*x+y*y)
                if rdr<self.rhalo:
                    if not self.mask.masked(x,y):
                        newstations['x'][station]=x
                        newstations['y'][station]=y
        self.stations=newstations

    def readCSV(self, name='LOWBD', rcore=0.0, l1def='SKA-low_config_baseline_design_arm_stations_2013apr30.csv', rhalo=40, recenter=False):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.nstations=1
        self.stations={}
        self.rhalo=rhalo
        self.fobs=1e8
        self.diameter=35.0
        meanx=0
        meany=0
        with open(l1def, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                meanx=meanx+float(row[0])
                meany=meany+float(row[1])
                self.nstations=self.nstations+1
        meanx=meanx/self.nstations
        meany=meany/self.nstations
        if not recenter:
            meanx=0.0
            meany=0.0
        f.close()
        self.nstations=0
        scale=0.001
        with open(l1def, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x=scale*(float(row[0])-meanx)
                y=scale*(float(row[1])-meany)
                r=numpy.sqrt(x*x+y*y)
                if r>rcore:
                    self.nstations=self.nstations+1
        
        f.close()
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)
        print "Number of stations = ", self.nstations
        station=0
        with open(l1def, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x=scale*(float(row[0])-meanx)
                y=scale*(float(row[1])-meany)
                r=numpy.sqrt(x*x+y*y)
                if r>rcore:
                    self.stations['x'][station]=x
                    self.stations['y'][station]=y
                    station=station+1
        self.nstations=station
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)

    def readLOWBD(self, name='LOWBD', rcore=0.0, l1def='SKA-low_config_baseline_design_arm_stations_2013apr30.csv'):
        return self.readCSV(name, rcore, l1def)

    def saveCSV(self, filename='LOWBD.csv'):
        with open(filename, 'wb') as fp:
            rowwriter = csv.writer(fp)
            for station in range(self.nstations):
                rowwriter.writerow([1000.0*self.stations['x'][station],1000.0*self.stations['y'][station]])
                
    def writeWGS84(self, filename='LOWBD.csv'):
        long0=116.779167
        lat0=-26.789267
        height0=300.0
        Re=6371.0
        with open(filename, 'wb') as fp:
            rowwriter = csv.writer(fp)
            rowwriter.writerow(['name','longitude','latitude','height'])
            for station in range(self.nstations):
                name='Low%d'%station
                long= long0+180.0*(self.stations['x'][station])*numpy.cos(numpy.pi*lat0/180.0)/(Re*numpy.pi)
                lat = lat0 +180.0*(self.stations['y'][station])*numpy.cos(numpy.pi*lat0/180.0)/(Re*numpy.pi)
                rowwriter.writerow([name,long,lat,height0])
                
    def readLOWL1(self, name='LOWL1', rcore=0.0, l1def='L1_configuration.csv'):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        self.name=name
        self.nstations=0
        self.stations={}
        self.rhalo=80
        self.fobs=1e8
        self.diameter=35.0
        meanx=0
        meany=0
        with open(l1def, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                meanx=meanx+float(row[1])
                meany=meany+float(row[0])
                self.nstations=self.nstations+1
        meanx=meanx/self.nstations
        meany=meany/self.nstations
        f.close()
        self.nstations=0
        scale=6.371e6*numpy.pi/180000.0
        with open(l1def, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x=scale*(float(row[1])-meanx)*numpy.cos(meanx*numpy.pi/180.0)
                y=scale*(float(row[0])-meany)
                r=numpy.sqrt(x*x+y*y)
                if r>rcore:
                    self.nstations=self.nstations+1
        
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)
        station=0
        scale=6.371e6*numpy.pi/180000.0
        with open(l1def, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                x=scale*(float(row[1])-meanx)*numpy.cos(meanx*numpy.pi/180.0)
                y=scale*(float(row[0])-meany)
                r=numpy.sqrt(x*x+y*y)
                if r>rcore:
                    self.stations['x'][station]=x
                    self.stations['y'][station]=y
                    self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)
                    station=station+1
        
    def readLOFAR(self, name='LOFAR', stationtype='S', band='HBA', lfdef='LOFAR.csv', lat=52.7):
        self.weightpergain=100.0 # Only use 10 sigma points
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        cs=numpy.cos(numpy.pi*lat/180.0)
        sn=numpy.sin(numpy.pi*lat/180.0)
        nantennas=256
        self.name=name
        self.nstations=0
        self.stations={}
        self.rhalo=80
        self.fobs=1e8
        self.diameter=35.0
        meanx=0
        meany=0
        meanz=0
        with open(lfdef, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                type=row[0]
                if (type.find(stationtype) > -1) and (type.find(band) > -1):
                    meanx=meanx+float(row[2])
                    meany=meany+float(row[1])
                    meanz=meanz+float(row[3])
                    self.nstations=self.nstations+1
        meanx=meanx/self.nstations
        meany=meany/self.nstations
        meanz=meanz/self.nstations
        f.close()
        station=0
        self.stations['x']=numpy.zeros(self.nstations)
        self.stations['y']=numpy.zeros(self.nstations)
        self.stations['weight']=numpy.zeros(self.nstations)
        with open(lfdef, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                type=row[0]
                if (type.find(stationtype) > -1) and (type.find(band) > -1):
                    x=(float(row[2])-meanx)/1000.0
                    y=(float(row[1])-meany)/1000.0
                    z=(float(row[3])-meanz)/1000.0
                    self.stations['x'][station]=x
                    self.stations['y'][station]=-cs*y+sn*z
                    station=station+1
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)

    def readKML(self, name='LOWBD2', kmlfile="V4Drsst512red_2", diameter=35.0):
        self.weightpergain=100.0 # Only use 10 sigma points    
        self.mask=TelMask()
        self.mask.readMask(maskfile='Mask_BoolardyStation.png')
        long0=116.779167
        lat0=-26.789267
        Re=6371.0
        self.stations={}
        self.stations['x']=numpy.zeros(512)
        self.stations['y']=numpy.zeros(512)
        self.stations['weight']=self.weightpergain*numpy.ones(512)
        self.nstations=512
        self.name=name
        self.diameter=diameter
        f=open(kmlfile)
        self.nstations=512
        station=0
        for line in f:
            line=line.lstrip()
            if line.find("name")>0:
                if line.find("Station")>0:
                    station=int(line.split('Station')[1].split('<')[0])
                if line.find("Antenna")>0:
                    station=int(line.split('Antenna')[1].split('<')[0])
            if line.find("coordinates")>0:
                x= float(line.split('>')[1].split('<')[0].split(',')[0])
                y= float(line.split('>')[1].split('<')[0].split(',')[1])
                self.stations['x'][station]=(x-long0)*Re*numpy.pi/(180.0*numpy.cos(numpy.pi*lat0/180.0))
                self.stations['y'][station]=(y-lat0)* Re*numpy.pi/(180.0*numpy.cos(numpy.pi*lat0/180.0))
        self.stations['weight']=self.weightpergain*numpy.ones(self.nstations)
                
    def writeKML(self, kmlfile="LOW_CIRCLES.kml"):
    
        long0=116.779167
        lat0=-26.789267
        height0=300.0
        Re=6371.0
        s=['<?xml version="1.0" encoding="UTF-8"?>', \
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
        l=['<Placemark>', \
            '<styleUrl>#whitecirc</styleUrl>', \
            '<name>S%d</name>', \
            '<Point>', \
            '<coordinates>%f, %f</coordinates>', \
            '</Point>', \
            '</Placemark>']
        e=['</Document>', '</kml>']
        f=open(kmlfile, 'w')
        for ss in s:
            f.write(ss)
        for station in range(self.nstations):
            long= long0+180.0*(self.stations['x'][station])*numpy.cos(numpy.pi*lat0/180.0)/(Re*numpy.pi)
            lat = lat0 +180.0*(self.stations['y'][station])*numpy.cos(numpy.pi*lat0/180.0)/(Re*numpy.pi)
            f.write( l[0])
            f.write( l[1])
            f.write( l[2] % station)
            f.write( l[3])
            f.write( l[4] % (long, lat))
            f.write( l[5])
            f.write( l[6])
        f.write( e[0])
        f.write( e[1])
        
    def excessDistance(self, MST=False):
        if MST:
            return 2.0*self.distance()-self.mst(False)/float(self.nstations)
        else:
            return self.distance()
# We maximise the minimum distance between stations 
    def distance(self):
        P=numpy.zeros([self.nstations,2])
        P[...,0]=self.stations['x']
        P[...,1]=self.stations['y']
        distancemat=sd.pdist(P)
        distance=numpy.min(distancemat)
        return distance
# Evaluate the minimum spanning tree    
    def mst(self, doplot=True, plotfile=''):
        P=numpy.zeros([self.nstations,2])
        P[...,0]=self.stations['x']
        P[...,1]=self.stations['y']
        X=sd.squareform(sd.pdist(P))
        edge_list = minimum_spanning_tree(X)
        if doplot:
            plt.clf()
            plt.scatter(P[:, 0], P[:, 1]) 
            dist=0    
            for edge in edge_list:
                i, j = edge
                plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
                dist=dist+numpy.sqrt((P[i,0]-P[j,0])*(P[i,0]-P[j,0])+(P[i,1]-P[j,1])*(P[i,1]-P[j,1]))
            plt.title('%s, MST=%.1f km' % (self.name, dist))
            plt.xlabel('X (km)')
            plt.ylabel('y (km)')
            plt.axes().set_aspect('equal')
            maxaxis=numpy.max(abs(P))
            plt.axes().set_xlim([-maxaxis,maxaxis])
            plt.axes().set_ylim([-maxaxis,maxaxis])
            mask=TelMask()
            mask.readKML()
            plt.fill(mask.segments['x1'], mask.segments['y1'], fill=False)
            if plotfile== '':
                plotfile='%s_MST.pdf' %self.name
            plt.show()
            plt.savefig(plotfile)
            return dist
        else:
            dist=0    
            for edge in edge_list:
                i, j = edge
                dist=dist+numpy.sqrt((P[i,0]-P[j,0])*(P[i,0]-P[j,0])+(P[i,1]-P[j,1])*(P[i,1]-P[j,1]))
            return dist

class TelUV:
    def _init_(self):
        self.name=''
        self.uv=False
    
    def construct(self, t):
        self.name=t.name
        self.nbaselines=t.nstations*t.nstations
        self.uv={}
        self.uv['x']=numpy.zeros(self.nbaselines)
        self.uv['y']=numpy.zeros(self.nbaselines)
        for station in range(t.nstations):
            self.uv['x'][station*t.nstations:(station+1)*t.nstations]=t.stations['x']-t.stations['x'][station]
            self.uv['y'][station*t.nstations:(station+1)*t.nstations]=t.stations['y']-t.stations['y'][station]
        
    
    def plot(self, plotfile=''):
        self.plotter=True
        plt.clf()
        plt.title('UV Sampling %s' % self.name)
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.plot(self.uv['x'], self.uv['y'], '.')
        plt.axes().set_aspect('equal')
        if plotfile == '':
            plotfile='UVcoverage_%s.pdf' % self.name
        plt.show()
        plt.savefig(plotfile)
        
    def assess(self):
        return 1.0
    
#
# Sources on the celestial sphere.
#
class TelSources:
    def _init_(self):
        self.name='Sources'
        self.nsources=100
    
    def construct(self, name='Sources', nsources=100, radius=1):
        self.name=name
        self.sources={}
        self.sources['x'], self.sources['y']=TelUtils().uniformcircle(nsources, radius)

        self.sources['x']=self.sources['x']-numpy.sum(self.sources['x'])/float(nsources)
        self.sources['y']=self.sources['y']-numpy.sum(self.sources['y'])/float(nsources)
        self.sources['flux']=sources().randomsources(smin=1.0, nsources=nsources) # We normalise so that the minimum source is 1 and the amplitude obeys logN/logS
        self.nsources=nsources
        self.radius=radius

    def plot(self):
        self.plotter=True
    
    def assess(self):
        return 1.0
#
# Piercings through the ionosphere
#
class TelPiercings:
    def _init_(self):
        self.name='Piercings'
        self.npiercings=0
        self.hiono=400
    
    def plot(self, rmax=70, rcore=70):
        plt.clf()
        plt.title(self.name)
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        r2=self.piercings['x']*self.piercings['x']+self.piercings['y']*self.piercings['y']
        npierce=len(r2>(rmax*rmax))
        P=numpy.zeros([npierce,2])
        P[...,0]=self.piercings['x']-numpy.average(self.piercings['x'])
        P[...,1]=self.piercings['y']-numpy.average(self.piercings['y'])
        distancemat=numpy.sort(sd.pdist(P))
        iondist=numpy.median(distancemat[:npierce])
        plt.plot(self.piercings['x']-numpy.average(self.piercings['x']), self.piercings['y']-numpy.average(self.piercings['y']), '.')
        plt.axes().set_aspect('equal')
        circ=plt.Circle((0,0), radius=rmax, color='g', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circ)
        circcore=plt.Circle((0,0), radius=rcore, color='r', fill=False)
        fig = plt.gcf()
        fig.gca().add_artist(circcore)
        maxaxis=max(numpy.max(abs(self.piercings['x'])), numpy.max(abs(self.piercings['y'])))
        plt.axes().set_xlim([-maxaxis,maxaxis])
        plt.axes().set_ylim([-maxaxis,maxaxis])
        plt.show()
        plt.savefig('%s.pdf' % self.name)   
        
    def construct(self, sources, array, rmin=1, hiono=300, rmax=200.0):
        self.hiono=hiono
        self.npiercings=sources.nsources*array.nstations
        self.name='%s_PC' % (array.name)
        self.piercings={}
        nstations=array.nstations
        self.piercings['x']=numpy.zeros(self.npiercings)
        self.piercings['y']=numpy.zeros(self.npiercings)
        self.piercings['weight']=numpy.zeros(self.npiercings)
        self.piercings['flux']=numpy.zeros(self.npiercings)
        for source in range(sources.nsources):
            self.piercings['x'][source*nstations:(source+1)*nstations]=self.hiono*sources.sources['x'][source]+array.stations['x']
            self.piercings['y'][source*nstations:(source+1)*nstations]=self.hiono*sources.sources['y'][source]+array.stations['y']
            r2=(self.piercings['x'][source*nstations:(source+1)*nstations])**2 + (self.piercings['y'][source*nstations:(source+1)*nstations])**2
            self.piercings['flux'][source*nstations:(source+1)*nstations]=sources.sources['flux'][source]*sources.sources['flux'][source]
            self.piercings['weight'][source*nstations:(source+1)*nstations]=array.stations['weight']
            self.piercings['weight'][source*nstations:(source+1)*nstations][r2>=rmax**2]=0.0

    def assess(self, nnoll=20, rmax=40.0, doplot=True):
        weight=self.piercings['weight']*self.piercings['flux']
        x=self.piercings['x']-numpy.average(self.piercings['x'])
        y=self.piercings['y']-numpy.average(self.piercings['y'])
        A=numpy.zeros([self.npiercings, nnoll])
        r=numpy.sqrt(x*x+y*y)
        pb2=numpy.exp(-numpy.log10(0.01)*(r/rmax)**2) # Model out to 10% of PB
        pb2[r>rmax]=0.0
        phi=numpy.arctan2(y,x)
        for noll in range(nnoll):
            A[:,noll]=zernike.zernikel(noll,r/rmax,phi)
        Covar_A=numpy.zeros([nnoll, nnoll])
        for nnol1 in range(nnoll):
            for nnol2 in range(nnoll):
                Covar_A[nnol1,nnol2]=numpy.sum(A[...,nnol1]*pb2*weight[...]*A[...,nnol2])/float(nnoll)
        print "Condition number = %f" % numpy.linalg.cond(Covar_A)
        U,s,Vh = linalg.svd(Covar_A)
        
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


        
        
    