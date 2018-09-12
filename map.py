import numpy as np
from glob import glob
from scipy import sparse
from fast_histogram import histogram2d as h2d
from copy import deepcopy as dc
import sys

def cosd(x):
    return np.cos(x*np.pi/180)
def sind(x):
    return np.sin(x*np.pi/180)
def tand(x):
    return np.tan(x*np.pi/180)

#def h2d(*args, **kwargs):
#    x, dum, dum = histogram2d(*args, **kwargs)
#    return x

class pairmap(object):
    
    def __init__(self, fn0='TnoP_notempnoise_dk???_????.npy'):
        """Read in pairmaps, coadd into full map"""
        self.fn = np.sort(glob('tod/'+fn0))
        self.getinfo()
        self.getmapdefn()
        self.coadd()
        return

    def getinfo(self):
        """Get TOD info and load first TOD"""

        self.dk = []
        self.pairnum = []
        for k,fn in enumerate(self.fn):
            if k==0:
                # Load first TOD
                self.tod = np.load(fn).item()
            self.pairnum.append(self.fn2pairnum(fn))
            self.dk.append(self.fn2dk(fn))

        self.pairnum = np.array(self.pairnum)
        self.dk = np.array(self.dk)

        self.id = np.unique(self.pairnum)
        self.Npair = len(self.id)
        self.Npixtemp = self.tod['X'].shape[1]

        dku = np.unique(self.dk)
        self.dkstr = 'dk'
        for k in dku:
            self.dkstr += '{:03d}+'.format(k)
        self.dkstr = self.dkstr[0:-1]

    def fn2pairnum(self, fn):
        """Gets pair ID number from filename"""
        return np.int(fn[-8:-4])

    def fn2dk(self, fn):
        """Gets pair ID number from filename"""
        return np.int(fn[-12:-9])

    def getmapdefn(self):
        """Get map pixels"""

        # Define map pixels
        self.pixsize = 0.25

        self.dec1d = np.arange(45.0, 70.0+0.01, self.pixsize)
        #self.dec1d = np.arange(-13, 13+0.01, self.pixsize) # Equator
        self.decbe = np.append(self.dec1d-self.pixsize/2., self.dec1d[-1]+self.pixsize/2)

        self.dx = self.pixsize / cosd(self.dec1d.mean())
        self.ra1d  = np.arange(-55, 55+0.01, self.dx)
        self.rabe = np.append(self.ra1d-self.dx/2., self.ra1d[-1]+self.dx/2.)

        self.ra, self.dec = np.meshgrid(self.ra1d, self.dec1d)
        self.Npixmap = self.ra.size

        
    def coadd(self):
        """Coadd data into maps"""

        sz  = (self.dec1d.size, self.ra1d.size)
        szt  = (self.Npixtemp, self.dec1d.size, self.ra1d.size)
        ac0 = np.zeros(sz, dtype='float32')
        act0 = np.zeros(szt, dtype='float32')

        # Initialize
        print('Initializing ac matrices')
        ac = {}
        flds = ['wsum','wzsum','wwv', 'w','wz','wcz','wsz',
                'wcc','wss','wcs','wwccv','wwssv','wwcsv']
        for k in flds: 
            ac[k] = dc(ac0)
        
        flds = ['wct','wst','wt']
        for k in flds:
            ac[k] = dc(act0)

        # Bin each pair into pixel
        #bins = [self.decbe, self.rabe] # If using np.histogram2d
        bins = [len(self.decbe)-1, len(self.rabe)-1] # If using fast histogram2d
        rng = [(np.min(self.decbe),np.max(self.decbe)), (np.min(self.rabe),np.max(self.rabe))]

        for p,pn in enumerate(self.id):
            
            print('coadding pair {0} of {1}...'.format(p+1, len(self.id)))

            doind = np.where( (self.pairnum == pn) )[0]
            dofn = self.fn[doind]
            acc = dc(ac)

            for j,fn0 in enumerate(dofn):

                print('loading {:s}'.format(fn0))
                sys.stdout.flush()

                # Load data
                val = np.load(fn0).item()

                # Binning info
                x = val['ra']; y = val['dec']

                # Pairsum quantities
                z = val['pairsum']
                v = np.ones_like(z)
                w = 1.0/v

                acc['wsum'] += h2d(y, x, bins=bins, range=rng, weights=w)
                acc['wzsum'] += h2d(y, x, bins=bins, range=rng, weights=w*z)
                acc['wwv'] += h2d(y, x, bins=bins, range=rng, weights=w*w*v)

                # Pair diff quantities
                z = np.ravel(val['pairdiff'])
                v = np.ones_like(z)
                w = 1.0/v
                c = cosd(np.ravel(2*val['alpha']))
                s = sind(np.ravel(2*val['alpha']))

                acc['w'] += h2d(y, x, bins=bins, range=rng, weights=w)
                acc['wz'] += h2d(y, x, bins=bins, range=rng, weights=w*z)
                acc['wcz'] += h2d(y, x, bins=bins, range=rng, weights=w*c*z)
                acc['wsz'] += h2d(y, x, bins=bins, range=rng, weights=w*s*z)
                acc['wcc'] += h2d(y, x, bins=bins, range=rng, weights=w*c*c)
                acc['wss'] += h2d(y, x, bins=bins, range=rng, weights=w*s*s)
                acc['wcs'] += h2d(y, x, bins=bins, range=rng, weights=w*c*s)
                acc['wwccv'] += h2d(y, x, bins=bins, range=rng, weights=w*w*c*c*v)
                acc['wwssv'] += h2d(y, x, bins=bins, range=rng, weights=w*w*s*s*v)
                acc['wwcsv'] += h2d(y, x, bins=bins, range=rng, weights=w*w*c*s*v)

                # Template
                for m in range(self.Npixtemp):
                    t = val['X'][:, m]
                    acc['wct'][m] += h2d(y, x, bins=bins, range=rng, weights=w*c*t)
                    acc['wst'][m] += h2d(y, x, bins=bins, range=rng, weights=w*s*t)
                    acc['wt'][m]  += h2d(y, x, bins=bins, range=rng, weights=w*t)

            
            # Get filename and save
            print('saving...')
            sys.stdout.flush()
            ind = dofn[0].find('dk')
            fnout = dofn[0][0:ind] + self.dkstr + dofn[0][(ind+5):]
            fnout = fnout.replace('tod','pairmaps')
            fnout = fnout.replace('.npy','.npz')
            np.savez(fnout, w=acc['w'], wz=acc['wz'], wcz=acc['wcz'],
                     wsz=acc['wsz'], wcc=acc['wcc'], wss=acc['wss'],
                     wcs=acc['wcs'], wwccv=acc['wwccv'], wwssv=acc['wwssv'], 
                     wwcsv=acc['wwcsv'], wct=acc['wct'], wst=acc['wst'],
                     wt=acc['wt'], wsum=acc['wsum'], wzsum=acc['wzsum'],
                     wwv=acc['wwv'], pairnum=pn, ra=self.ra, dec=self.dec)

        return

class map(object):

    def __init__(self, fn0):

        if fn0.find('maps/') == -1:
            self.fn = np.sort(glob('pairmaps/'+fn0))
            self.makemap()
        else:
            self.load(fn0)

    def makemap(self):
        """Make TQU maps"""
        
        # Coadd over pairs
        print('making TQU maps...')

        self.acs = {}

        for j,fn in enumerate(self.fn):

            # Load
            print('loading {:s}'.format(fn))
            sys.stdout.flush()
            ac = np.load(fn)

            flds = ['wsum','wzsum','wwv', 'w','wz','wcz','wsz',
                    'wcc','wss','wcs','wwccv','wwssv','wwcsv']
            
            for k in flds:

                if j == 0:
                    self.acs[k] = ac[k]
                    self.ra = ac['ra']
                    self.dec = ac['dec']
                else:
                    self.acs[k] += ac[k]

            ac.close()


        # Compute TQU maps
        self.T = self.acs['wzsum'] / self.acs['wsum']
        self.Tvar = self.acs['wwv'] / self.acs['wsum']**2

        self.acs['wz'] = self.acs['wz'] / self.acs['w']        
        self.acs['wcz'] = self.acs['wcz'] / self.acs['w']
        self.acs['wsz'] = self.acs['wsz'] / self.acs['w']
        self.acs['wcc'] = self.acs['wcc'] / self.acs['w']
        self.acs['wss'] = self.acs['wss'] / self.acs['w']
        self.acs['wcs'] = self.acs['wcs'] / self.acs['w']
        self.acs['wwccv'] = self.acs['wwccv'] / self.acs['w']**2
        self.acs['wwssv'] = self.acs['wwssv'] / self.acs['w']**2
        self.acs['wwcsv'] = self.acs['wwcsv'] / self.acs['w']**2
        
        x = self.acs['wcc']
        y = self.acs['wss']
        z = self.acs['wcs']
        self.acs['n'] = 1.0 / (x*y - z**2)
        self.acs['n'][np.abs(self.acs['n'])>1e3] = np.nan
        self.acs['e'] = self.acs['n']*y
        self.acs['f'] = -self.acs['n']*z
        self.acs['g'] = self.acs['n']*x

        self.Q = self.acs['e']*self.acs['wcz'] + self.acs['f']*self.acs['wsz']
        self.U = self.acs['f']*self.acs['wcz'] + self.acs['g']*self.acs['wsz']
        self.Qvar = self.acs['e']**2 * self.acs['wwccv'] + \
                    self.acs['f']**2*self.acs['wwssv'] + \
                    2*self.acs['e']*self.acs['f']*self.acs['wwcsv']
        self.Uvar = self.acs['f']**2 * self.acs['wwccv'] + \
                    self.acs['g']**2*self.acs['wwssv'] + \
                    2*self.acs['f']*self.acs['g']*self.acs['wwcsv']
        self.QUcovar = self.acs['e'] * self.acs['f']*self.acs['wwccv'] + \
                       self.acs['e'] * self.acs['g']*self.acs['wwcsv'] + \
                       self.acs['f'] * self.acs['f']*self.acs['wwcsv'] + \
                       self.acs['f'] * self.acs['g']*self.acs['wwssv'] 
                       
        self.Tw = 1/self.Tvar
        self.Tw = self.Tw / np.nanmax(self.Tw)
        self.Qw = 1/self.Qvar
        self.Qw = self.Qw / np.nanmax(self.Qw)
        self.Uw = 1/self.Uvar
        self.Uw = self.Uw / np.nanmax(self.Uw)
        self.Pw = 1/(self.Uvar+self.Qvar)
        self.Pw = self.Pw / np.nanmax(self.Pw)
        self.pixsize = self.dec[1,0] - self.dec[0,0]


    def addmapnoise(self, depthQU = 1.0, depthT = 5):
        """Add noise to map.
        depthQU = Q and U map depth in uK-arcmin.
        depthT = T map depth in uK-arcmin.
        """

        print('adding noise to map...')

        # Noise is in uK-arcmin, get pixsize in arcmin^2
        pixsize = (self.pixsize*60)**2

        # Scale to pixel size
        sigmaT = depthT / np.sqrt(pixsize)
        sigmaQU = depthQU / np.sqrt(pixsize)

        # Get pol noise maps and add
        nT = np.random.randn(*self.T.shape)*sigmaT
        nQ = np.random.randn(*self.Q.shape)*sigmaQU
        nU = np.random.randn(*self.U.shape)*sigmaQU
        
        # Scale by weight
        nT /= np.sqrt(self.Tw)
        nQ /= np.sqrt(self.Qw)
        nU /= np.sqrt(self.Uw)


        # Set to zero
        nT[~np.isfinite(nT)] = 0
        nQ[~np.isfinite(nQ)] = 0
        nU[~np.isfinite(nU)] = 0
        
        # Add
        self.T += nT
        self.Q += nQ
        self.U += nU

    def save(self, ext=None):
        """Save the map"""

        if ext is not None:
            extt = '_'+ext
        else:
            extt = ''

        fnout = self.fn[0][0:-9] + extt + '.npz'
        fnout = fnout.replace('pairmaps','maps')
        np.savez(fnout, T=self.T, Q=self.Q, U=self.U, Tw=self.Tw, Qw=self.Qw,
                 Uw=self.Uw, Tvar=self.Tvar, Qvar=self.Qvar, Uvar=self.Uvar,
                 QUcovar=self.QUcovar, Pw=self.Pw, ra=self.ra, dec=self.dec,
                 pixsize=self.pixsize, fn=self.fn, acs=self.acs) 
        

    def load(self, fn):
        """Load the map"""
        x = np.load(fn)
        k = x.keys()
        for kk in k:
            setattr(self, kk, x[kk])
        x.close()



