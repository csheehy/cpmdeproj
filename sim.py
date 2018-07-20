import numpy as np
import healpy as hp
import beam
from astropy.io import fits
from functools import partial
from scipy.interpolate import interp1d
from matplotlib.pyplot import *
from copy import deepcopy as dc
ion()

def cosd(x):
    return np.cos(x*np.pi/180)
def sind(x):
    return np.sin(x*np.pi/180)
def tand(x):
    return np.tan(x*np.pi/180)

class sim(object):

    def __init__(self, Ba, Bb, Nside='TEB_lcdm_N256.fits', r=0, theta=0, dk=0.0):
        """Simulate given a beam object"""
        self.Ba = Ba
        self.Bb = Bb
        self.Nside = Nside
        self.r = r
        self.theta = theta
        self.dk = dk

        self.genmap()
        self.gettraj()

    def runsim(self, inclpol=True, Ttemptype='noiseless', QUtemptype='noiseless',
               cpmalpha=1e7, dk=0.0, sigmadet=0):
        """addtempnoise = False (none), 'planck' or 'spt'"""

        self.inclpol = inclpol
        self.Ttemptype = Ttemptype
        self.QUtemptype = QUtemptype
        self.cpmalpha = cpmalpha
        self.sigmadet = sigmadet

        self.gentempmap()

        self.gensig()
        self.addmapnoise()
        #self.filtermap()
        self.prepcpm()
        #self.filterTtemplate()
        self.cpm()

        return

    def hipassmap(self, hmap, lmin=30):
        """Hi pass signal and template maps"""

        npix = hmap.shape[-1]
        Nside = hp.npix2nside(npix)
        alm = list(hp.map2alm(hmap))
        l = np.arange(0, 3*self.Nside)
        fl = np.ones_like(l)
        fl[l<lmin] = 0
        alm = [hp.almxfl(val, fl) for val in alm]
        hmap_out = hp.alm2map(alm, Nside)
        return np.array(hmap_out)


    def genmap(self, Nside=None):
        """Generate a healpix CMB map with TT, EE, and BB"""
        
        if Nside is not None:
            self.Nside = Nside   
            
        if type(self.Nside) is str:
            self.hmap = np.array(hp.read_map(self.Nside, field=(0,1,2)))
            self.Nside = hp.npix2nside(self.hmap[0].size)
        else:
            cl, nm = self.readcambfits('camb_planck2013_r0_lensing.fits')
            self.hmap = np.array(hp.synfast(cl, self.Nside, new=True, verbose=False))
        
    def gentempmap(self):
        """Make template map"""

        # First, template is input map
        self.hmaptemp = self.hmap*1.0

        # T noise
        if self.Ttemptype == 'planck':
            fn = 'maps/mc_noise/143/ffp8_noise_143_full_map_mc_00000.fits'
            hmapn = 1e6 * np.array(hp.read_map(fn, field=0))
            hmapn = np.array(hp.ud_grade(hmapn, self.Nside, pess=False))
            hmapn[np.abs(hmapn)>1e10] = 0
            self.hmaptemp[0] += hmapn

        # QU noise
        if self.QUtemptype == 'planck':

            # Add Planck noise
            fn = 'maps/mc_noise/143/ffp8_noise_143_full_map_mc_00000.fits'
            hmapn = 1e6 * np.array(hp.read_map(fn, field=(1,2)))
            hmapn = np.array(hp.ud_grade(hmapn, self.Nside, pess=False))
            hmapn[np.abs(hmapn)>1e10] = 0
            self.hmaptemp[1:] += hmapn
            
        elif (self.QUtemptype == 'spt') | (self.QUtemptype == 's4'):

            # Add SPT noise, chi-by-eye to Henning SPT-500deg^2 paper N_l and
            # functional form in  http://users.physics.harvard.edu/~buza/20161220_chkS4/
            
            if self.QUtemptype == 'spt':
                sigmap = 9.0 # uK-arcmin, SPT
            elif self.QUtemptype == 's4':
                sigmap = 1.2 # uK-arcmin, CMB-S4
            lknee = 300.
            lexp = -1.8

            l = np.arange(8000)*1.0
            Nl = 4*np.pi / (41253.*60**2) * (1+(l/lknee)**(lexp)) * sigmap**2
            Nl[0] = 0

            # Get noise realization
            hmapQn = hp.synfast(Nl, self.Nside, new=True, verbose=False)
            hmapUn = hp.synfast(Nl, self.Nside, new=True, verbose=False)

            # Add
            self.hmaptemp[1] += hmapQn
            self.hmaptemp[2] += hmapUn

            self.sigmap = sigmap
            self.lknee = lknee
            self.lexp = lexp


    def readcambfits(self, fname):
        """Read in a CAMB generated fits file and return a numpy array of the table
        values. Returns numpy array of N_l x N_fields and a string array of field
        names. Returns C_l's in uK^2.""" 

        h=fits.open(fname)
        d=h[1].data
        nm=d.names

        nl=d[nm[0]].size
        nfields=np.size(nm)

        cl=np.zeros([nfields,nl])

        for k,val in enumerate(nm):
            cl[k]=d[val]

        # Convert to uK^2
        cl = cl*1e12

        return cl,nm

    def gettraj(self):
        """Get scan trajectory."""
        
        self.pixsize = 0.25 # degrees

        self.azthrow = 55.0
        self.radrift = 12.0 
        self.rathrow = self.azthrow + self.radrift
        self.decthrow = 5.0
        self.mapracen = 0
        self.mapdeccen = 57.5

        bsralim = np.array([self.mapracen-self.rathrow/2., self.mapracen+self.rathrow/2.])
        bsdeclim = np.array([self.mapdeccen-self.decthrow/2., self.mapdeccen+self.decthrow/2.])
        self.ralim, self.declim = self.reckon(bsralim, bsdeclim, self.r, self.theta+self.dk)

        mapdec = np.arange(45.0, 70.0, self.pixsize)
        mapra  = np.arange(-55, 55, self.pixsize/cosd(mapdec.mean()))
        self.mapra, self.mapdec = np.meshgrid(mapra, mapdec)

        self.mapind = np.where((self.mapra>=self.ralim[0]) & (self.mapra<=self.ralim[1])
                               & (self.mapdec>=self.declim[0]) & (self.mapdec<=self.declim[1]))

        ra = self.mapra[self.mapind]
        dec = self.mapdec[self.mapind]
        self.ra, self.dec = np.meshgrid(np.unique(ra),np.unique(dec))

        # Polarization angle of detector A in focal plane coordinates
        # w.r.t. north, same sense as FP theta. Detector B assumed orthogonal.
        self.alphafp = 0.0

        # Angle this pol vector makes w.r.t. radial vector
        self.chi = self.alphafp - self.theta

        # Angle the pol vector makes w.r.t. north when projected on sky
        az = self.azimuth(self.dec, self.ra, ..., ...)

    def gensig(self):
        """Gen signal"""
        
        ravec = np.ravel(self.ra)
        decvec = np.ravel(self.dec)
        siga = np.zeros_like(ravec)
        sigb = np.zeros_like(ravec)

        for k in range(len(ravec)):
            
            print('{:d} of {:d}'.format(k,len(ravec)))

            ra = ravec[k]
            dec = decvec[k]
            Tca, Qca, Uca, Tcb, Qcb, Ucb = self.convolve(ra, dec)
            chiA = self.alpha + self.dk
            siga[k] = Tca + Qca*cosd(2*chiA) + Uca*sind(2*chiA)
            sigb[k] = Tcb + Qcb*cosd(2*(chiA+90)) + Ucb*sind(2*(chiA+90))

        self.siga = np.reshape(siga, self.ra.shape)
        self.sigb = np.reshape(sigb, self.ra.shape)
        self.pairsum  = (self.siga + self.sigb)/2.0
        self.pairdiff = (self.siga - self.sigb)/2.0


    def addmapnoise(self, sigmadet=None):
        """Add noise to map"""
        if sigmadet is not None:
            self.sigmadet = sigmadet
            
        # Noise is in uK-arcmin, get pixsize in arcmin^2
        pixsize = (self.pixsize*60)**2
        sigma = self.sigmadet / np.sqrt(pixsize)
        nsum = np.random.randn(*self.pairdiff.shape)*sigma * 5
        ndiff = np.random.randn(*self.pairdiff.shape)*sigma
        self.pairdiff = self.pairdiff + ndiff
        self.pairsum = self.pairsum + nsum
        

    def filtermap(self, mapin=None):
        """Poly filter + ground subtract map"""

        if mapin is None:
            self.pairdiff = self.filtermap(self.pairdiff)
            self.pairsum  = self.filtermap(self.pairsum)
            return
        
        dra = self.ra[0,1]-self.ra[0,0]
        npixscan = np.round(self.azthrow/dra).astype(int)
        nscan = mapin.shape[1] - npixscan + 1
        x = np.linspace(-1,1,npixscan)

        mapout = np.ones((mapin.shape[0], mapin.shape[1], nscan))*np.nan
        gsub = np.ones((npixscan,mapin.shape[1],nscan))

        for k,val in enumerate(mapin):
            for j in range(nscan):
                s = j
                e = j+npixscan
                y = mapin[k,s:e]
                z = np.polyfit(x, y, 3)
                p = np.poly1d(z)
                mapout[k,s:e,j] = y - p(x)

        mapout = np.nanmean(mapout, 2)

        return mapout

    def gnomproj_xy2lonlat(self, xdeg, ydeg, lonc, latc):
        """All in degrees"""
        x = np.atleast_1d(xdeg*np.pi/180)
        y = np.atleast_1d(ydeg*np.pi/180)
        rho = np.sqrt(x**2+y**2)
        c = np.arctan(rho)*180/np.pi
        lon = lonc + np.arctan2(x*sind(c), rho*cosd(latc)*cosd(c) - y*sind(latc)*sind(c))*180/np.pi
        lat = np.arcsin(cosd(c)*sind(latc) + y*sind(c)*cosd(latc)/rho)*180/np.pi

        lon[rho==0] = lonc
        lat[rho==0] = latc

        return lon, lat

    def gnomproj_lonlat2xy(self, lon, lat, lonc, latc):
        """All in degrees"""
        cosc = sind(latc)*sind(lat) + cosd(latc)*cosd(lat)*cosd(lon-lonc)
        x = (cosd(lat)*sind(lon-lonc)/cosc)*180/np.pi
        y = (cosd(latc)*sind(lat) - sind(latc)*cosd(lat)*cosd(lon-lonc))/cosc * 180/np.pi
        return x, y

    def gnomproj(self, hmap, lonc, latc, xsize, ysize, reso, rot=0, Full=False):
        """Gnomonic projection from healpix map. 
        lonc, latc  = center of projection in degrees
        xsize,ysize = projection size in degrees
        reso = pixel size in degrees
        rot = projection rotation in degrees"""
        
        x = np.arange(-xsize/2.0, xsize/2.0+reso/2., reso)
        y = np.arange(-ysize/2.0, ysize/2.0+reso/2., reso)
        xx,yy = np.meshgrid(x,y)
        ra,dec = self.gnomproj_xy2lonlat(xx, yy, lonc, latc)

        if rot !=0:
            # Rotate projection
            az = self.azimuth(latc, lonc, dec, ra)
            d = self.dist(latc, lonc, dec, ra)
            ra, dec = self.reckon(lonc, latc, d, az + rot)

        # Interpolate map
        m = hp.get_interp_val(hmap, ra, dec, lonlat=True)

        if Full:
            return xx, yy, ra, dec, m
        else:
            return m


    def convolve(self, rac, decc):
        """Convolve healpix map with beam centered on ra dec. Assumes beams A
        and B use same grid."""

        phi = -(self.Ba.phi + np.pi/2)*180/np.pi # North=0, increasing to East, degrees
        theta = self.Ba.rr / 60.  # arcmin -> degrees

        ra_i, dec_i = self.reckon(rac, decc, theta, phi + self.dk)

        # Get healpix map values
        T_i = hp.get_interp_val(self.hmap[0], ra_i, dec_i, lonlat=True)
        if self.inclpol:
            Q_i = hp.get_interp_val(self.hmap[1], ra_i, dec_i, lonlat=True)
            U_i = hp.get_interp_val(self.hmap[2], ra_i, dec_i, lonlat=True)

        # Multiply and sum
        Tca = np.sum(T_i*self.Ba.mb)
        Tcb = np.sum(T_i*self.Bb.mb)

        if self.inclpol:
            Qca = np.sum(Q_i*self.Ba.mb)
            Uca = np.sum(U_i*self.Ba.mb)
            Qcb = np.sum(Q_i*self.Bb.mb)
            Ucb = np.sum(U_i*self.Bb.mb)
        else:
            Qca = 0
            Uca = 0
            Qcb = 0
            Ucb = 0

        return Tca, Qca, Uca, Tcb, Qcb, Ucb


    def reckon(self, lon0, lat0, theta, phi):
        "Reckon on unit sphere, all in degrees"""
        
        lat = np.arcsin(sind(lat0)*cosd(theta) +
                        cosd(lat0)*sind(theta)*cosd(-phi)) * 180/np.pi
        
        if (90-lat0)<1e-6:
            # Numerical precision issues
            dlon = (phi+180)*np.pi/180
        else:
            dlon = np.arctan2(sind(theta)*sind(-phi)*cosd(lat0),
                             cosd(theta) - sind(lat0)*sind(lat))

        lon = 180/np.pi * (np.mod(lon0*np.pi/180 - dlon + np.pi, 2*np.pi) - np.pi)

        return lon, lat

    def dist(self, lat1, lon1, lat2, lon2):
        """Great circle distance, lat/lon in degrees, returns in degrees."""
        dlon = lon2 - lon1
        num = np.sqrt( (cosd(lat2)*sind(dlon))**2 + (cosd(lat1)*sind(lat2)-sind(lat1)*cosd(lat2)*cosd(dlon))**2 )
        denom = sind(lat1)*sind(lat2) + cosd(lat1)*cosd(lat2)*cosd(dlon)
        d = np.arctan2(num, denom) * 180/np.pi
        return d

    def azimuth(self, lat1, lon1, lat2, lon2):
        """Initial azimuth bearing, measured east from north,
        connecting point1 -> point2. Everything in degrees."""
        dlon = lon2 - lon1
        num = sind(dlon)*cosd(lat2)
        denom = cosd(lat1)*sind(lat2) - sind(lat1)*cosd(lat2)*cosd(dlon)
        az = np.arctan2(num, denom) * 180/np.pi
        return az


    def prepcpm(self, beam=None):
        """Let's fit the pair diff map with the healpix map. """

        # Get regression template, which is the gnomonic projection of the input
        # healpix map around every pixel center
        ysum = np.ravel(self.pairsum)
        ydiff = np.ravel(self.pairdiff)
        ravec = np.ravel(self.ra)
        decvec = np.ravel(self.dec)
        npix = len(ysum)

        for k in range(npix):
            
            xs = 4.0 # T template x size in degrees
            ys = 4.0 # T template y size in degrees
            #res = 0.0404040404 # resolution in degrees (matches fwd sim beam)
            res = 0.1 # resolution in degrees             

            xx, yy, ra, dec, tempT = self.gnomproj(self.hmaptemp[0], ravec[k], decvec[k], 
                                                   xs, ys, res, rot=self.dk, Full=True)
            xx, yy, ra, dec, tempQ = self.gnomproj(self.hmaptemp[1], ravec[k], decvec[k], 
                                                   xs, ys, res, rot=self.dk, Full=True)
            xx, yy, ra, dec, tempU = self.gnomproj(self.hmaptemp[2], ravec[k], decvec[k], 
                                                   xs, ys, res, rot=self.dk, Full=True)

            # Construct pairdiff predictor from Q and U template maps
            chiA = self.alpha + self.dk
            siga = tempQ*cosd(2*chiA) + tempU*sind(2*chiA)
            sigb = tempQ*cosd(2*(chiA+90)) + tempU*sind(2*(chiA+90))
            pairdiff = (siga - sigb)/2.

            if k==0:
                # Initialize regressor
                #X = np.zeros( (npix, 2*len(np.ravel(tempT))) )
                X = np.zeros( (npix, len(np.ravel(tempT))) )
                
            # Construct regressor
            #X[k, :] = np.concatenate((np.ravel(tempT), np.ravel(pairdiff)))
            X[k, :] = np.ravel(tempT)

        self.xx = xx
        self.yy = yy
        self.X = X
        self.ysum = ysum
        self.ydiff = ydiff
        self.ravec = ravec
        self.decvec = decvec
        self.nT = tempT.size # Number of T regressors
        self.tempshape = tempT.shape

    def filterTtemplate(self):
        """Filter the T template"""
        self.X[:, 0:self.nT] = self.filtertemplate(self.X[:,0:self.nT])

    def filtertemplate(self, X):
        """Filter templates"""

        #This might take a while.
        sz = self.pairdiff.shape
        Xfilt = np.ones_like(X)
        for k in range(X.shape[1]):
            print('Filter template, {0} of {1}'.format(k,X.shape[1]))
            Xfilt[:,k] = np.ravel(self.filtermap(X[:,k].reshape(sz)))
        return Xfilt
            
    def cpm(self, cpmalpha=None, b=None, fittype=None):
        """Fit template to data"""

        if cpmalpha is not None:
            self.cpmalpha = cpmalpha

        if fittype is None:
            self.cpm(fittype='pairsum')
            self.cpm(fittype='pairdiff')
            return

        nT = self.nT
        XT = self.X[:,0:self.nT]
        #Xpol = self.X[:,nT:]

        if fittype == 'pairsum':
            # Fit pairsum
            y = self.ysum
            X = XT
            I = np.identity(X.shape[1])*self.cpmalpha

        elif fittype == 'pairdiff':
            # Fit pairdiff
            y = self.ydiff
            rr = np.sqrt(self.xx**2 + self.yy**2)
            beam = np.ravel(np.exp(-rr**2 / (2*(self.Ba.sigma/60.)**2)))
            beam = beam/np.nansum(beam)
            beamtile = beam[np.newaxis,:].repeat(y.size, 0)
            #Xdiff = np.sum(beamtile*Xpol,1)[:,np.newaxis]
            #Xdifffilt = self.filtertemplate(Xdiff)
            #X = np.concatenate((XT,Xdiff),1)
            X = XT

            #X = self.X
            #X = XT
            #y = y - Xdiff

            I = np.identity(X.shape[1])*self.cpmalpha
            beam = np.ravel(self.bTsum)
            beam = beam/np.max(beam)
            for j,val in enumerate(beam):
                I[j,j] = I[j,j]/np.abs(val)
            #I[-1,-1] = 0

        if b is None:

            # Non nan indices
            ind = np.where(np.isfinite(y))[0]
            yfit = y[ind]
            Xfit = X[ind,:]

            ################
            # Do regression, homebrew ridge, direct inversion
            b = np.linalg.inv(Xfit.T.dot(Xfit) + I).dot(Xfit.T).dot(yfit)

            ###############
            # Do regression, homebrew ridge, SVD technique

            #print('homebrew ridge, SVD')
            #U,S,V = np.linalg.svd(X, full_matrices=False)
            #lam = np.ones(S.size)*self.cpmalpha
            #lam[nT:] = 0
            #D = np.diag(S/(S**2+lam))
            #b = V.T.dot(D).dot(U.T).dot(ydiff)

        ##################
        # Get prediction

        # Only use T coefficients to predict
        bT = b[0:nT]
        XT = X[:,0:nT]

        # Get prediction
        if fittype == 'pairsum':
            self.bsum = b
            self.bTsum = np.reshape(bT, self.tempshape)
        elif fittype == 'pairdiff':
            self.bdiff = b
            self.bTdiff = np.reshape(bT, self.tempshape)
            self.T2Ppred = np.reshape(XT.dot(bT), self.ra.shape)


    def info(self):
        print('beam mismatch? {0}'.format(np.any(self.Ba.mb != self.Bb.mb)))
        print('dk = {0}'.format(self.dk))
        print('r = {0}'.format(self.r))
        print('theta = {0}'.format(self.theta))
        print('inclpol = {0}'.format(self.inclpol))
        print('cpmalpha = {:0.1e}'.format(self.cpmalpha))
        print('Ttemptype = {:s}'.format(self.Ttemptype))
        print('QUtemptype = {:s}'.format(self.QUtemptype))

    def save(self, prefix, i):
        """Strip out healpix maps and save as pickle"""
        x = dc(self)
        delattr(x,'hmap')
        delattr(x,'hmaptemp')
        fn = 'simdata/{:s}_{:04d}.npy'.format(prefix,i)
        np.save(fn, x)


