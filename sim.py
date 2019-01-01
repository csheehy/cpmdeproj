import numpy as np
import healpy as hp
import beam
from astropy.io import fits
from functools import partial
from scipy.interpolate import interp1d
from matplotlib.pyplot import *
from copy import deepcopy as dc
import decorr
import sys
import os
import map
ion()

def cosd(x):
    return np.cos(x*np.pi/180)
def sind(x):
    return np.sin(x*np.pi/180)
def tand(x):
    return np.tan(x*np.pi/180)

def readcambfits(fname):
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

def round(val, fac=1e-6):
    """Round"""
    return np.round(val/fac)*fac

def gensigmap(type='lenspdust', Nside=1024, lmax=None, rlz=0):

    if type == 'lens':
        cl, nm = readcambfits('spec/camb_planck2013_r0_lensing_lensfix.fits')
        fn = 'input_maps/camb_planck2013_r0_lensing_lensfix_n{:04d}_r{:04d}.fits'.format(Nside,rlz)

    elif type == 'lenspdust':
        cl, nm = readcambfits('spec/camb_planck2013_r0_lensing_lensfix.fits')
        mod = decorr.Model()
        l = np.arange(len(cl[0]))
        cld = mod.getdustcl(l,143,0.1,2)
        cl[2,:] = cl[2,:] + cld
        Ad = mod.fsky2A(0.1)[1]
        Adstr = '{:0.3f}'.format(Ad).replace('.','p')
        fn = 'input_maps/camb_planck2013_r0_lensing_lensfix_A{:s}_n{:04d}_r{:04d}.fits'.format(Adstr,Nside,rlz)
        
    elif type == 'lenspdust_noTE':
        cl, nm = readcambfits('spec/camb_planck2013_r0_lensing_lensfix.fits')
        mod = decorr.Model()
        l = np.arange(len(cl[0]))
        cld = mod.getdustcl(l,143,0.1,2)
        cl[2,:] = cl[2,:] + cld
        cl[3,:] = 0 # zero TE
        Ad = mod.fsky2A(0.1)[1]
        Adstr = '{:0.3f}'.format(Ad).replace('.','p')
        fn = 'input_maps/camb_planck2013_r0_noTE_lensing_lensfix_A{:s}_n{:04d}_r{:04d}.fits'.format(Adstr,Nside,rlz)

    res = hp.synfast(cl, Nside, new=True, verbose=False,
                     lmax=lmax, alm=True)
    hmap = np.array(res[0])
    alm0 = res[1]

    hp.write_map(fn, hmap)

    alm = dc(alm0)
    alm[2][:] = 0 # Zero B
    hmap = hp.alm2map(alm, Nside)
    hp.write_map(fn.replace('camb_planck2013','camb_planck2013_EnoB'), hmap)

    alm = dc(alm0)
    alm[1][:] = 0 # Zero E
    hmap = hp.alm2map(alm, Nside)
    hp.write_map(fn.replace('camb_planck2013','camb_planck2013_BnoE'), hmap)

def dist(lon1, lat1, lon2, lat2):
    """Great circle distance, lat/lon in degrees, returns in degrees."""
    dlon = lon2 - lon1
    num = np.sqrt( (cosd(lat2)*sind(dlon))**2 + (cosd(lat1)*sind(lat2)-sind(lat1)*cosd(lat2)*cosd(dlon))**2 )
    denom = sind(lat1)*sind(lat2) + cosd(lat1)*cosd(lat2)*cosd(dlon)
    d = np.arctan2(num, denom) * 180/np.pi
    return d

def azimuth(lon1, lat1, lon2, lat2):
    """Initial azimuth bearing, measured east from north,
    connecting point1 -> point2. Everything in degrees."""
    dlon = lon2 - lon1
    num = sind(dlon)*cosd(lat2)
    denom = cosd(lat1)*sind(lat2) - sind(lat1)*cosd(lat2)*cosd(dlon)
    az = np.arctan2(num, denom) * 180/np.pi
    return az


def reckon(lon0, lat0, theta, phi):
    "Reckon on unit sphere, all in degrees"""

    lat = np.arcsin(sind(lat0)*cosd(theta) +
                    cosd(lat0)*sind(theta)*cosd(-phi)) * 180/np.pi

    lon0 = np.atleast_1d(lon0)
    lat0 = np.atleast_1d(lat0)
    phi = np.atleast_1d(phi)
    theta = np.atleast_1d(theta)

    dlon = np.arctan2(sind(theta)*sind(-phi)*cosd(lat0),
                      cosd(theta) - sind(lat0)*sind(lat))

    # Numerical precision issues
    ind = np.where((90-lat0)<1e-6)[0]
    if len(ind) > 0:
        dlon[ind] = phi[ind] + 180*np.pi/180

    lon = 180/np.pi * (np.mod(lon0*np.pi/180 - dlon + np.pi, 2*np.pi) - np.pi)

    return lon, lat


def gnomproj_xy2lonlat(xdeg, ydeg, lonc, latc):
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

def gnomproj_lonlat2xy(lon, lat, lonc, latc):
    """All in degrees"""
    cosc = sind(latc)*sind(lat) + cosd(latc)*cosd(lat)*cosd(lon-lonc)
    x = (cosd(lat)*sind(lon-lonc)/cosc)*180/np.pi
    y = (cosd(latc)*sind(lat) - sind(latc)*cosd(lat)*cosd(lon-lonc))/cosc * 180/np.pi
    return x, y

def gnomproj(hmap, lonc, latc, xsize, ysize, reso, rot=0, Full=False):
    """Gnomonic projection from healpix map. 
    lonc, latc  = center of projection in degrees
    xsize,ysize = projection size in degrees
    reso = pixel size in degrees
    rot = projection rotation in degrees"""

    if np.size(reso) == 1:
        # Constant grid spacing
        x = np.arange(-xsize/2.0, xsize/2.0+reso/2., reso)
        y = np.arange(-ysize/2.0, ysize/2.0+reso/2., reso)
    else:
        # Two grid spacings
        x1 = np.arange(-xsize[0]/2.0, xsize[0]/2.0+reso[0]/2., reso[0])
        y1 = np.arange(-ysize[0]/2.0, ysize[0]/2.0+reso[0]/2., reso[0])
        x2 = np.arange(-xsize[1]/2.0, xsize[1]/2.0+reso[1]/2., reso[1])
        y2 = np.arange(-ysize[1]/2.0, ysize[1]/2.0+reso[1]/2., reso[1])
        indx = np.where(np.abs(x2)>(xsize[0]/2))
        indy = np.where(np.abs(y2)>(xsize[0]/2))
        x = np.unique(round(np.sort(np.concatenate((x1,x2[indx])))))
        y = np.unique(round(np.sort(np.concatenate((y1,y2[indy])))))
        
    xx,yy = np.meshgrid(x,y)
    ra,dec = gnomproj_xy2lonlat(xx, yy, lonc, latc)

    if rot !=0:
        # Rotate projection
        az = azimuth(lonc, latc, ra, dec)
        d = dist(lonc, latc, ra, dec)
        ra, dec = reckon(lonc, latc, d, az + rot)

    # Interpolate map
    m = hp.get_interp_val(hmap, ra, dec, lonlat=True)

    if Full:
        return xx, yy, ra, dec, m
    else:
        return m




class sim(object):

    def __init__(self, Ba, Bb, inputmap=None, r=0, theta=0, dk=0.0,
                 tempNside=512, lmax=None, sn='000', rlz=0):

        """Simulate given a beam object"""
        self.Ba = Ba
        self.Bb = Bb
        self.tempNside = tempNside
        self.lmax = lmax
        self.inputmap = inputmap
        self.r = r
        self.theta = theta
        self.dk = dk
        self.sn = sn
        self.rlz = rlz
        
        self.gettraj()

    def runsim(self, Ttemptype='noiseless', QUtemptype='noiseless', sigtype='sig'):
        """addtempnoise = False (none), 'planck' or 'spt'"""

        self.getsigmap(sigtype)

        self.Ttemptype = Ttemptype
        self.QUtemptype = QUtemptype
        self.sigtype = sigtype

        if sigtype == 'TnoP':
            self.gentempmap()
            self.gentemp()
        else:
            for k in ['tempxx','tempyy','nT','X']:
                setattr(self,k,0)

        self.gensig()

        #self.filtermap()

        return


    def getsigmap(self, sigtype):
        """Generate a healpix CMB map with TT, EE, and BB"""
        fn = self.inputmap.replace('rxxxx','r{:04d}'.format(self.rlz))

        if sigtype == 'EnoB':
            fn = fn.replace('camb_planck2013','camb_planck2013_EnoB')

        # Load
        self.hmap = np.array(hp.read_map('input_maps/'+fn, field=(0,1,2)))
        self.Nside = hp.npix2nside(self.hmap[0].size)

        # Band limit. Technically we should not band limit and just simulate
        # with beam rolloff, but the map should not have power beyond the
        # nyquist limit of the beam postage stamp pixelizaiton, which for the
        # case of our sidelobes is quite coarse (0.2 deg)
        alm = hp.map2alm(self.hmap)
        cl = hp.alm2cl(alm[0])
        fl = np.zeros_like(cl)
        l = np.arange(len(fl))
        l0 = 500
        l1 = 600
        fl[0:l0] = 1.0
        ll = np.arange(l1-l0)
        ll = ll * np.pi / np.max(ll)
        fl[l0:l1] = 0.5*(np.cos(ll) + 1)

        alm2 = [hp.almxfl(k, fl) for k in alm]
        self.hmap = np.array(hp.alm2map(alm2, self.Nside))
        del alm
        del alm2

        if sigtype == 'EnoB':
            # no T->P for EnoB sims
            self.hmap[0,:] = 0

    
    def gentempmap(self):
        """Make template map"""

        print('preparing template map...')
        sys.stdout.flush()

        # First, template is input map at lower res
        self.hmaptemp = np.array(hp.ud_grade(self.hmap, self.tempNside))

        ###################
        # T noise
        if self.Ttemptype == 'planck':
            fn = 'planckmaps/mc_noise/143/ffp8_noise_143_full_map_mc_{:05d}.fits'.format(self.rlz)
            hmapn = 1e6 * np.array(hp.read_map(fn, field=0))
            hmapn = np.array(hp.ud_grade(hmapn, self.tempNside, pess=False))
            hmapn[np.abs(hmapn)>1e10] = 0
        else:
            hmapn = 0

        # Add T noise
        self.hmaptemp[0] += hmapn
            
        ###################
        # Q/U noise
        if self.QUtemptype == 'planck':
            # Add Planck noise
            fn = 'planckmaps/mc_noise/143/ffp8_noise_143_full_map_mc_{:05d}.fits'.format(self.rlz)
            hmapn = 1e6 * np.array(hp.read_map(fn, field=(1,2)))

        elif (self.QUtemptype == 'spt'):
            fn = 'input_maps/SPT_noise_map_r{:04d}.fits'.format(self.rlz)
            hmapn = hp.read_map(fn, field=(1,2))
            

        elif (self.QUtemptype == 's4'):
            fn = 'input_maps/S4_noise_map_r{:04d}.fits'.format(self.rlz)
            hmapn = hp.read_map(fn, field=(1,2))

        elif self.QUtemptype == 'noiseless':            
            hmapn = np.zeros_like(self.hmaptemp[1:])

        else:
            raise ValueError('QU template type not recognized')
                
        # Add noise
        hmapn = np.array(hp.ud_grade(hmapn, self.tempNside, pess=False))
        hmapn[np.abs(hmapn)>1e10] = 0
        self.hmaptemp[1:] += hmapn

        # Smooth Q/U templates
        fwhm = 30.0 # fwhm in arcmin
        self.hmaptemp[1] = hp.smoothing(self.hmaptemp[1], fwhm=fwhm/60.*np.pi/180)
        self.hmaptemp[2] = hp.smoothing(self.hmaptemp[2], fwhm=fwhm/60.*np.pi/180)

        
    def gettraj(self):
        """Get scan trajectory."""
        
        self.elstep = 0.25 # degrees

        self.azthrow = 55.0
        self.radrift = 12.0 
        self.rathrow = self.azthrow + self.radrift
        self.decthrow = 5.0
        self.mapracen = 0
        self.mapdeccen = 57.5 # BK field
        #self.mapdeccen = 0 # Equator

        # Define boresight pointing, always the same.
        bsralim = np.array([self.mapracen-self.rathrow/2., self.mapracen+self.rathrow/2.])
        bsdeclim = np.array([self.mapdeccen-self.decthrow/2., self.mapdeccen+self.decthrow/2.])
        bsra = np.arange(bsralim[0], bsralim[1]+.001, self.elstep/cosd(self.mapdeccen))
        bsdec = np.arange(bsdeclim[0], bsdeclim[1]+.001, self.elstep)
        bsra, bsdec = np.meshgrid(bsra, bsdec)

        # Round
        self.bsra = np.ravel(round(bsra))
        self.bsdec = np.ravel(round(bsdec))
        
        # Calculate detector centroid pointing
        ra, dec = reckon(self.bsra, self.bsdec, self.r, self.theta+self.dk)
        self.ra = round(ra)
        self.dec = round(dec)

        # Polarization angle of detector A in focal plane coordinates
        # w.r.t. north, same sense as FP theta. Detector B assumed orthogonal.
        self.alphafp = 0.0

        # Angle this pol vector makes w.r.t. radial vector
        self.chi = self.alphafp - self.theta

        # Angle the pol vector makes w.r.t. north when projected on sky
        az = azimuth(self.ra, self.dec, self.bsra, self.bsdec)
        self.alpha = 180 + az + self.chi

        # Special case
        ind = np.where((self.ra == self.bsra) & (self.dec == self.bsdec))[0]
        if len(ind) > 0:
            self.alpha[ind] = self.alphafp + self.dk

        # Round
        self.alpha = round(self.alpha)

        ind = self.alpha<0
        self.alpha[ind] = self.alpha[ind] + 360

        ind = self.alpha >= 180
        self.alpha[ind] = self.alpha[ind] - 360


    def gensig(self):
        """Gen signal"""
        
        if self.sigtype != 'noi':

            siga = np.zeros_like(self.ra)
            sigb = np.zeros_like(self.ra)

            for k in range(len(self.ra)):

                print('Generating TOD element {:d} of {:d}'.format(k,len(self.ra)))
                sys.stdout.flush()

                ra = self.ra[k]
                dec = self.dec[k]
                rotang = self.alpha[k]
                Tca, Qca, Uca, Tcb, Qcb, Ucb = self.convolve(ra, dec, rotang, 
                                                             self.Ba.mb, self.Bb.mb, 
                                                             self.Ba.rr, self.Ba.phi)
                if hasattr(self.Ba,'sl'):
                    Tsa, Qsa, Usa, Tsb, Qsb, Usb = self.convolve(ra, dec, rotang, 
                                                                 self.Ba.sl, self.Bb.sl,
                                                                 self.Ba.rrsl, self.Ba.phisl)
                    Tca += Tsa
                    Qca += Qsa
                    Uca += Usa
                    Tcb += Tsb
                    Qcb += Qsb
                    Ucb += Usb

                chiA = self.alpha[k]
                chiB = chiA + 90
                siga[k] = Tca + Qca*cosd(2*chiA) + Uca*sind(2*chiA)
                sigb[k] = Tcb + Qcb*cosd(2*chiB) + Ucb*sind(2*chiB)

            self.siga = siga
            self.sigb = sigb

        else:

            sens = 0.03 # Total pairmap sensitivity in uK
            N = self.ra.size
            sig = sens * np.sqrt(N)
            self.siga = np.random.randn(N)*sig
            self.sigb = np.random.randn(N)*sig
            # Now add correlated part
            ncorr = np.random.randn(N)*sig*4
            self.siga += ncorr
            self.sigb += ncorr

        # Sum/diff
        self.pairsum  = (self.siga + self.sigb)/2.0
        self.pairdiff = (self.siga - self.sigb)/2.0
        

    def filtermap(self):
        """Poly filter + ground subtract map"""
        # Not yet implemented
        return 


    def convolve(self, rac, decc, rotang, ba, bb, r, phi_in):
        """Convolve healpix map with beam centered on ra dec. Assumes beams A
        and B use same grid."""

        phi = -(phi_in + np.pi/2 + np.pi)*180/np.pi # North=0, increasing to East, degrees
        theta = r / 60.  # arcmin -> degrees

        # Use alpha because FP "up" direction does not point north when
        # projected on sky.
        ra_i, dec_i = reckon(rac, decc, theta, phi + rotang)

        # Get healpix map values
        if self.sigtype != 'PnoT':
            T_i = hp.get_interp_val(self.hmap[0], ra_i, dec_i, lonlat=True)
        if self.sigtype != 'TnoP':
            Q_i = hp.get_interp_val(self.hmap[1], ra_i, dec_i, lonlat=True)
            U_i = hp.get_interp_val(self.hmap[2], ra_i, dec_i, lonlat=True)

        # Multiply and sum
        if self.sigtype != 'PnoT':
            Tca = np.sum(T_i*ba)
            Tcb = np.sum(T_i*bb)
        else:
            Tca = 0
            Tcb = 0

        if self.sigtype != 'TnoP':
            Qca = np.sum(Q_i*ba)
            Uca = np.sum(U_i*ba)
            Qcb = np.sum(Q_i*bb)
            Ucb = np.sum(U_i*bb)
        else:
            Qca = 0
            Uca = 0
            Qcb = 0
            Ucb = 0

        return Tca, Qca, Uca, Tcb, Qcb, Ucb


    def gentemp(self):
        """Let's fit the pair diff map with the healpix map. """

        print('Sampling off template map...')
        sys.stdout.flush()

        # Get regression template, which is the gnomonic projection of the input
        # healpix map around every pixel center
        nt = self.ra.size

        for k in range(nt):

            # Construct T template
            #xs = [4.0, 40.0] # x size in degrees
            #ys = [4.0, 40.0] # y size in degrees
            #res = [0.1, 0.25] # resolution in degrees             
            xs = 20.0
            ys = 20.0
            res = 0.1
            xx, yy, ra, dec, tempT = gnomproj(self.hmaptemp[0], self.ra[k], self.dec[k],
                                              xs, ys, res, rot=self.alpha[k], Full=True)
            tempT = np.ravel(tempT)
            
            if k == 0:
                XT = np.zeros((nt, tempT.size))
                
            XT[k] = tempT

        # Construct pairdiff predictor from Q and U template maps
        tempQ = hp.get_interp_val(self.hmaptemp[1], self.ra, self.dec, lonlat=True)
        tempU = hp.get_interp_val(self.hmaptemp[2], self.ra, self.dec, lonlat=True)
        chiA = self.alpha
        chiB = chiA + 90
        siga = tempQ*cosd(2*chiA) + tempU*sind(2*chiA)
        sigb = tempQ*cosd(2*chiB) + tempU*sind(2*chiB)
        pairdiff = (siga - sigb)/2.
        pairdiff = pairdiff.reshape(nt, 1)
        nQU = 1

        # Deriv map templates
        Tsm = hp.smoothing(self.hmaptemp[0], fwhm = 30.0/60. * np.pi/180)
        alm = hp.map2alm(Tsm)
        dum, dth, dph = hp.alm2map_der1(alm, self.tempNside)
        dth_alm = hp.map2alm(dth)
        dph_alm = hp.map2alm(dph)
        dth, dth2, dthdph = hp.alm2map_der1(dth_alm, self.tempNside)
        dum, dum, dph2 = hp.alm2map_der1(dph_alm, self.tempNside)

        # Sample off deriv map templates
        Tsm   = hp.get_interp_val(Tsm,   self.ra, self.dec, lonlat=True)
        dth   = hp.get_interp_val(dth,   self.ra, self.dec, lonlat=True)
        dph   = hp.get_interp_val(dph,   self.ra, self.dec, lonlat=True)
        dth2  = hp.get_interp_val(dth2,  self.ra, self.dec, lonlat=True)
        dph2  = hp.get_interp_val(dph2,  self.ra, self.dec, lonlat=True)
        dthdph = hp.get_interp_val(dthdph, self.ra, self.dec, lonlat=True)

        Xderiv = np.zeros((nt, 6))
        Xderiv[:,0] = Tsm # Gain
        Xderiv[:,1] = dth # Diff point x
        Xderiv[:,2] = dph # Diff point y
        Xderiv[:,3] = dph2 + dth2 # Diff bw
        Xderiv[:,4] = dph2 - dth2 # Diff plus ellip
        Xderiv[:,5] = 2*dthdph # Cross elip
        
        # Initialize regressor matrix
        self.X = np.concatenate( (XT, Xderiv, pairdiff), axis=1)
        self.nT = tempT.size # Number of T regressors
        self.tempxx = xx
        self.tempyy = yy


    def save(self, i):
        """Strip out healpix maps and save as pickle"""

        print('Saving...')

        tod = {}
        flds = ['alpha', 'alphafp', 'Ba', 'Bb', 'bsra', 'bsdec', 'chi', 'dec',
                'dk', 'Nside', 'tempNside', 'nT', 'pairdiff', 'pairsum', 'QUtemptype', 'r', 'ra',
                'siga', 'sigb','theta','Ttemptype','X', 'inputmap', 
                'tempxx', 'tempyy']
        for k in flds:
            tod[k] = getattr(self, k)
        pth = 'tod/{:s}/'.format(self.sn)
        try:
            os.makedirs(pth)
        except:
            print('{:s} exists, skipping mkdir'.format(pth))

        fn = '{:s}_r{:04d}_dk{:03d}_{:04d}.npy'.format(self.sigtype, self.rlz, np.int(self.dk), i)        
        fnout = '{:s}/{:s}'.format(pth,fn)
        np.save(fnout, tod)
        
        return fnout

