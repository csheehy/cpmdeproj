import numpy as np
from glob import glob

def cosd(x):
    return np.cos(x*np.pi/180)
def sind(x):
    return np.sin(x*np.pi/180)
def tand(x):
    return np.tan(x*np.pi/180)

def h2d(*args, **kwargs):
    x, dum, dum = np.histogram2d(*args, **kwargs)
    return x

class map(object):
    
    def __init__(self, fn0='TnoP_noiseless_*.npy'):
        """Read in pairmaps, coadd into full map"""
        self.fn = np.sort(glob('pairmaps/'+fn0))
        self.loaddata()
        self.getmapdefn()
        self.coadd()
        return

    def loaddata(self):
        """Load data"""

        self.pairmaps = []
        self.pairnum = []
        for k,fn in enumerate(self.fn):
            print('loading {0} of {1}'.format(k+1,len(self.fn)))            
            self.pairmaps.append(np.load(fn).item())
            self.pairnum.append(self.fn2pairnum(fn))

        self.pairnum = np.array(self.pairnum)
        self.id = np.unique(self.pairnum)
        self.Npair = len(self.id)
        self.Npixtemp = self.pairmaps[0].X.shape[1]

    def fn2pairnum(self, fn):
        """Gets pair ID number from filename"""
        return np.int(fn[-8:-4])

    def getmapdefn(self):
        """Get map pixels"""

        # Define map pixels
        self.pixsize = 0.25

        self.dec1d = np.arange(45.0, 70.0+0.01, self.pixsize)
        self.decbe = np.append(self.dec1d-self.pixsize/2., self.dec1d[-1]+self.pixsize/2)

        self.dx = self.pixsize / cosd(self.dec1d.mean())
        self.ra1d  = np.arange(-55, 55+0.01, self.dx)
        self.rabe = np.append(self.ra1d-self.dx/2., self.ra1d[-1]+self.dx/2.)

        self.ra, self.dec = np.meshgrid(self.ra1d, self.dec1d)
        self.ravec = np.ravel(self.ra)
        self.decvec = np.ravel(self.dec)
        self.Npixmap = self.ra.size


    def addmapnoise(self, sigmadet=100):
        """Add noise to map"""

        # Noise is in uK-arcmin, get pixsize in arcmin^2
        pixsize = (self.pixsize*60)**2

        # Scale to pixel size
        self.sigmadet = sigmadet
        sigma = self.sigmadet / np.sqrt(pixsize)

        # Get noise maps and add to pairmaps
        for k,val in enumerate(self.pairmaps):
            nsum = np.random.randn(*val.pairsum.shape)*sigma * 5
            ndiff = np.random.randn(*val.pairdiff.shape)*sigma
            val.pairdiff = val.pairdiff + ndiff
            val.pairsum = val.pairsum + nsum

        
    def coadd(self):
        """Coadd data into maps"""

        # Initialize
        self.ac = {}
        sz = (self.Npair, self.dec1d.size, self.ra1d.size)
        self.ac['wsum'] = np.zeros(sz)
        self.ac['wzsum'] = np.zeros(sz)
        self.ac['wwv'] = np.zeros(sz)

        self.ac['w'] = np.zeros(sz)
        self.ac['wz'] = np.zeros(sz)
        self.ac['wcz'] = np.zeros(sz)
        self.ac['wsz'] = np.zeros(sz)
        self.ac['wcc'] = np.zeros(sz)
        self.ac['wss'] = np.zeros(sz)
        self.ac['wcs'] = np.zeros(sz)
        self.ac['wwccv'] = np.zeros(sz)
        self.ac['wwssv'] = np.zeros(sz)
        self.ac['wwcsv'] = np.zeros(sz)
        
        # Template
        sz = (self.Npair, self.Npixtemp, self.dec1d.size, self.ra1d.size)
        self.ac['wct'] = np.zeros(sz)
        self.ac['wst'] = np.zeros(sz)

        # Bin each pair into pixels
        for j,val in enumerate(self.pairmaps):
            
            k = np.where(self.id == self.pairnum[j])

            # Binning info
            x = val.ravec; y = val.decvec; bins = [self.decbe, self.rabe]
            
            # Pairsum quantities
            z = np.ravel(val.pairsum)
            v = np.ones_like(z)
            w = 1.0/v

            self.ac['wsum'][k] += h2d(y, x, bins=bins, weights=w)
            self.ac['wzsum'][k] += h2d(y, x, bins=bins, weights=w*z)
            self.ac['wwv'][k] += h2d(y, x, bins=bins, weights=w*w*v)

            # Pair diff quantities
            z = np.ravel(val.pairdiff)
            v = np.ones_like(z)
            w = 1.0/v
            c = cosd(np.ravel(2*val.alpha))
            s = sind(np.ravel(2*val.alpha))

            self.ac['w'][k] += h2d(y, x, bins=bins, weights=w)
            self.ac['wz'][k] += h2d(y, x, bins=bins, weights=w*z)
            self.ac['wcz'][k] += h2d(y, x, bins=bins, weights=w*c*z)
            self.ac['wsz'][k] += h2d(y, x, bins=bins, weights=w*s*z)
            self.ac['wcc'][k] += h2d(y, x, bins=bins, weights=w*c*c)
            self.ac['wss'][k] += h2d(y, x, bins=bins, weights=w*s*s)
            self.ac['wcs'][k] += h2d(y, x, bins=bins, weights=w*c*s)
            self.ac['wwccv'][k] += h2d(y, x, bins=bins, weights=w*w*c*c*v)
            self.ac['wwssv'][k] += h2d(y, x, bins=bins, weights=w*w*s*s*v)
            self.ac['wwcsv'][k] += h2d(y, x, bins=bins, weights=w*w*c*s*v)

            # Template
            for m in range(self.Npixtemp):
                t = val.X[:, m]
                self.ac['wct'][k,m] += h2d(y, x, bins=bins, weights=w*c*t)
                self.ac['wst'][k,m] += h2d(y, x, bins=bins, weights=w*s*t)

        # Coadd over pairs
        self.acs = {}
        for k in self.ac.keys():
            self.acs[k] = np.sum(self.ac[k], 0)

        # Compute TQU maps
        self.T = self.acs['wzsum'] / self.acs['wsum']
        self.Tvar = self.acs['wwv'] / self.acs['wsum']**2

        self.acs['z'] = self.acs['wz'] / self.acs['w']        
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
        self.acs['n'][np.abs(self.acs['n'])>1e10] = np.nan
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
