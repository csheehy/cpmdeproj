import numpy as np
from glob import glob
from scipy import sparse
from fast_histogram import histogram2d as h2d
from scipy.interpolate import griddata
#import fbpca
#from sklearn.linear_model import Ridge
from ridge import Ridge
from sim import dist
from copy import deepcopy as dc
import sys
import os
from matplotlib.pyplot import *

def cosd(x):
    return np.cos(x*np.pi/180)
def sind(x):
    return np.sin(x*np.pi/180)
def tand(x):
    return np.tan(x*np.pi/180)

#def h2d(*args, **kwargs):
#    x, dum, dum = histogram2d(*args, **kwargs)
#    return x

def addmaps(m1, m2):
    """Add maps"""
    
    m3 = dc(m1)
    fld = ['T','Q','U','b','Qpred','Upred','Qpred_cpm','Upred_cpm']
    for k in fld:
        setattr(m3,k, getattr(m1,k) + getattr(m2,k))
    return m3 

class pairmap(object):
    
    def __init__(self, fn0='000/TnoP_notempnoise_dk???_0001.npy', cpmalpha=1,
                 temptype='TR360+pol', dext='', cpmtype='perpix', dpdk='perdk'):
        """Read in pairmaps, coadd into full map"""
        self.sn = fn0[0:3]
        self.temptype = temptype
        self.cpmalpha = cpmalpha
        self.cpmtype = cpmtype
        self.dpdk = dpdk

        self.fn = np.sort(glob('tod/'+fn0))
        self.getinfo()
        self.getmapdefn()
        ac = self.coadd()
        self.save(ac, dext=dext)

    def getinfo(self):
        """Get TOD info and load first TOD"""

        dk = []
        for k,fn in enumerate(self.fn):
            if k==0:
                # Load first TOD
                self.tod = np.load(fn).item()
            dk.append(self.fn2dk(fn))
            
        self.Npixtemp = self.tod['X'].shape[1]
        self.dk = np.array(dk)
        self.udk = np.unique(self.dk)
        self.ndk = len(self.udk)
        if self.dpdk == 'perdk':
            self.dpdkset = [ [0],[45],[90],[135],[180],[225],[270],[315] ] 
        elif self.dpdk == 'alldk':
            self.dpdkset = [ [0, 45, 90, 135, 180, 225, 270, 315] ] 

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


    def deproj(self, ac):
        """Deproject"""

        # Number of dk angles

        # Number of T map coeffs
        Npixtemp = self.Npixtemp
        nT = self.nT
        ndk = self.ndk

        # Map sizes
        sz = ac['wz'][0].shape
        Npixmap = np.prod(np.array(sz))        

        # Initialize predictions
        ac['wctpol'] = np.zeros_like(ac['wz'])
        ac['wstpol'] = np.zeros_like(ac['wz'])
        ac['wczpred'] = np.zeros_like(ac['wz'])
        ac['wszpred'] = np.zeros_like(ac['wz'])
        ac['wczpred_cpm'] = np.zeros_like(ac['wz'])
        ac['wszpred_cpm'] = np.zeros_like(ac['wz'])
        if 'polsub' in self.temptype:
            ac['b'] = np.zeros((ndk,nT))
        else:
            ac['b'] = np.zeros((ndk,Npixtemp))
        ac['bt'] = np.zeros((ndk,nT))

        print('deprojecting...')
        sys.stdout.flush()

        # For regular fitting
        for dodks in self.dpdkset:
            
            dkind = [np.where(k==self.udk)[0][0] for k in dodks]

            y = np.ravel(ac['wz'][dkind])
            ysum = np.ravel(ac['wzsum'][dkind])
            fitindarr = np.where(ac['wz'][dkind] != 0)
            fitind = np.where(y != 0)[0]
            y = y[fitind]
            ysum = ysum[fitind]
            ra  = np.tile(np.ravel(self.ra),len(dkind))[fitind]
            dec = np.tile(np.ravel(self.dec),len(dkind))[fitind]
            pixid = np.tile(np.arange(self.ra.size),len(dkind))[fitind]

            # Init design matrices
            X =  np.zeros((len(y), Npixtemp))
            Xc = np.zeros((len(y), Npixtemp))
            Xs = np.zeros((len(y), Npixtemp))

            # Construct and pop for memory's sake
            for k in range(Npixtemp):
                X[:,k] =  np.ravel(ac['wt'][dkind,k,:,:])[fitind]
                Xc[:,k] = np.ravel(ac['wct'][dkind,k,:,:])[fitind]
                Xs[:,k] = np.ravel(ac['wst'][dkind,k,:,:])[fitind]


            #############
            # Fit wz
            if 'polsub' in self.temptype:
                # Subtract pol
                alph = np.ones(nT)*self.cpmalpha
                if Npixtemp > nT:
                    ysub = X[:,-1]*1.0
                else:
                    ysub = 0
                X = X[:,0:nT]
            else:
                # Fit pol (if present) along with T
                alph = np.ones(Npixtemp)*self.cpmalpha
                alph[nT:] = 0
                ysub = 0

            # Ridge regression pair sum
            r = Ridge(alpha=self.cpmalpha)
            r.fit(X[:,0:nT], ysum)
            bt = r.coef_*1.0

            # Ridge regression pair diff
            r = Ridge(alpha=alph)
            r.fit(X, y-ysub)
            b = r.coef_

            # Predict wcz, wsz
            ypredc = np.zeros_like(ac['wz'][dkind])
            ypreds = np.zeros_like(ac['wz'][dkind])
            t = range(nT)
            ypredc[fitindarr] = r.predict(Xc, t)
            ypreds[fitindarr] = r.predict(Xs, t)

            ##############
            # Now we predict pol on different pixels sets. Trying left/right
            # half. This will be a bit noisy, probably, but the idea is that the
            # cross spectrum of this prediction with the real map will yield an
            # unbiased measurement of T->P.
            medra = np.median(ra)
            ypredc_cpm = np.zeros_like(ypredc)
            ypreds_cpm = np.zeros_like(ypreds)

            if self.cpmtype == 'perpix':
                # Predict each pixel separately
                uid = np.unique(pixid)
            elif self.cpmtype == 'lr':
                # Predict left and right half separately
                uid = [0,1]

            for k in range(len(uid)):

                if np.mod(k,100) == 0:
                    print('CPM fitting pixel {0} of {1}'.format(k+1, len(uid)))
                    sys.stdout.flush()

                if uid == [0,1]:
                    # deproject left side w/ right side and vice versa
                    win = 3.0
                    if k==0:
                        f = np.where(ra<=(medra-win))[0] # fit
                        p = np.where(ra>medra)[0] # pred
                    elif k==1:
                        f = np.where(ra>(medra+win))[0]
                        p = np.where(ra<=medra)[0]
                else:
                    # Deproject each pixel separately with all other pixels
                    # outside certain radius (5 deg)
                    p = np.where(pixid == uid[k])[0]
                    d = dist(ra[p[0]], dec[p[0]], ra, dec)
                    f = np.where(d > 5.0)[0]

                r.fit(X[f,:], (y-ysub)[f])
                pp = tuple([g[p] for g in fitindarr])
                ypredc_cpm[pp] = r.predict(Xc[p], t)
                ypreds_cpm[pp] = r.predict(Xs[p], t)

            ac['b'][dkind] = b
            ac['bt'][dkind] = bt
            ac['wczpred'][dkind] = ypredc
            ac['wszpred'][dkind] = ypreds
            ac['wczpred_cpm'][dkind] = ypredc_cpm
            ac['wszpred_cpm'][dkind] = ypreds_cpm

            if 'pol' in self.temptype:
                wctpol = np.zeros_like(ac['wz'][dkind])
                wstpol = np.zeros_like(ac['wz'][dkind])
                wctpol[fitindarr] = Xc[:,-1]*1.0
                wstpol[fitindarr] = Xs[:,-1]*1.0
                ac['wctpol'][dkind] = wctpol
                ac['wstpol'][dkind] = wstpol
        
        return ac

        
    def coadd(self):
        """Coadd data into maps"""

        ndk = self.ndk
        sz  = (ndk, self.dec1d.size, self.ra1d.size)
        szt  = (ndk, self.Npixtemp, self.dec1d.size, self.ra1d.size)
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
        
        # Bin into pixels
        #bins = [self.decbe, self.rabe] # If using np.histogram2d
        bins = [len(self.decbe)-1, len(self.rabe)-1] # If using fast histogram2d
        rng = [(np.min(self.decbe),np.max(self.decbe)), (np.min(self.rabe),np.max(self.rabe))]
        
        for j,fn0 in enumerate(self.fn):

            # dk index
            d = np.where(self.udk == self.dk[j])[0]
            if len(d)>0:
                # Exists
                d = d[0]
            else:
                print('skipping {:s}, not in defined dkset'.format(fn0))
                continue

            print('loading {:s}'.format(fn0))
            sys.stdout.flush()
            
            # Load data
            val = dict(np.load(fn0).item())
            
            # Binning info
            x = val['ra']; y = val['dec']
            
            # Pairsum quantities
            z = val['pairsum']
            v = np.ones_like(z)
            w = 1.0/v
            
            ac['wsum'][d] += h2d(y, x, bins=bins, range=rng, weights=w)
            ac['wzsum'][d] += h2d(y, x, bins=bins, range=rng, weights=w*z)
            ac['wwv'][d] += h2d(y, x, bins=bins, range=rng, weights=w*w*v)
            
            # Pair diff quantities
            z = np.ravel(val['pairdiff'])
            v = np.ones_like(z)
            w = 1.0/v
            c = cosd(np.ravel(2*val['alpha']))
            s = sind(np.ravel(2*val['alpha']))
            
            ac['w'][d] += h2d(y, x, bins=bins, range=rng, weights=w)
            ac['wz'][d] += h2d(y, x, bins=bins, range=rng, weights=w*z)
            ac['wcz'][d] += h2d(y, x, bins=bins, range=rng, weights=w*c*z)
            ac['wsz'][d] += h2d(y, x, bins=bins, range=rng, weights=w*s*z)
            ac['wcc'][d] += h2d(y, x, bins=bins, range=rng, weights=w*c*c)
            ac['wss'][d] += h2d(y, x, bins=bins, range=rng, weights=w*s*s)
            ac['wcs'][d] += h2d(y, x, bins=bins, range=rng, weights=w*c*s)
            ac['wwccv'][d] += h2d(y, x, bins=bins, range=rng, weights=w*w*c*c*v)
            ac['wwssv'][d] += h2d(y, x, bins=bins, range=rng, weights=w*w*s*s*v)
            ac['wwcsv'][d] += h2d(y, x, bins=bins, range=rng, weights=w*w*c*s*v)

            if self.temptype != 'none':

                # Assemble template
                X = self.gettemplate(val, t=self.temptype)

                if j == 0:
                    print('Initializing act matrices')
                    szt  = (ndk, self.Npixtemp, self.dec1d.size, self.ra1d.size)
                    act0 = np.zeros(szt, dtype='float32')
                    flds = ['wct','wst','wt']
                    for k in flds:
                        ac[k] = dc(act0)

                for m in range(self.Npixtemp):
                    t = X[:, m]
                    ac['wct'][d][m] += h2d(y, x, bins=bins, range=rng, weights=w*c*t)
                    ac['wst'][d][m] += h2d(y, x, bins=bins, range=rng, weights=w*s*t)
                    ac['wt'][d][m]  += h2d(y, x, bins=bins, range=rng, weights=w*t)

            else:

                # Fill with zeros for T->P prediction
                for fld in ['wczpred','wszpred','wczpred_cpm','wszpred_cpm','wctpol','wstpol']:
                    ac[fld] = np.zeros_like(ac['wcz'])
                ac['b'] = [0]
                ac['bt'] = [0]
                self.nT = 0


        if self.temptype != 'none':
            # Deproject
            ac = self.deproj(ac)


        # Get rid of for memory's sake
        ac.pop('wt')
        ac.pop('wct')
        ac.pop('wst')

        ac['tempxx'] = val['tempxx']
        ac['tempyy'] = val['tempyy']

        return ac


    def gettemplate(self, v, t='TR360+pol'):
        """Tweak template. 
        t = :
        'TRxxx': Direct T map, include radii up to xxx in degrees (set to 999
                 for everything).
        'deriv': traditional derivative templates (for now only works fitting dk
                 angles individually)
        'pol': include pair diff 
        Join with '+', i.e. 'TR360+pol'
        """

        X = v['X']
        nT = v['nT']

        XT = X[:,0:nT]
        Xderiv = X[:,nT:(nT+6)] # dp1111

        if 'polself' not in t:
            Xpol = X[:,-1].reshape(XT.shape[0],1)
        else:
            # Load map
            ms = map(); ms.load('maps/{:s}/sig_r0000_dkxxx.npz'.format(self.sn))
            mn = map(); mn.load('maps/{:s}/noi_r0000_dkxxx.npz'.format(self.sn))
            m = addmaps(ms,mn)
            ra = np.ravel(m.ra); dec = np.ravel(m.dec); Q = np.ravel(m.Q); U = np.ravel(m.U)
            Q[~np.isfinite(Q)] = 0; U[~np.isfinite(U)] = 0
            Qi = griddata((ra,dec), Q, (v['ra'],v['dec']), method='cubic', fill_value=0)
            Ui = griddata((ra,dec), U, (v['ra'],v['dec']), method='cubic', fill_value=0)
            chiA = v['alpha']
            chiB = chiA + 90
            siga = Qi*cosd(2*chiA) + Ui*sind(2*chiA)
            sigb = Qi*cosd(2*chiB) + Ui*sind(2*chiB)
            Xpol = (siga - sigb)/2.
            Xpol = Xpol.reshape(XT.shape[0],1)

        tt = t.split('+')

        # Construct template
        Xret = []
        self.nT = 0
        for k, t in enumerate(tt):

            if t[0:2] == 'TR':
                # Direct T map
                r = np.float(t[2:])
                ind = np.where((np.abs(np.ravel(v['tempxx'])) <= r) &
                               (np.abs(np.ravel(v['tempyy'])) <= r))[0]
                Xret.append(XT[:,ind])
                self.nT += len(ind)
                v['tempxx'] = np.ravel(v['tempxx'])[ind]
                v['tempyy'] = np.ravel(v['tempyy'])[ind]
                
            elif t=='deriv':
                # Deriv based templates
                Xret.append(Xderiv)
                self.nT += 6

            elif (t == 'pol') | (t == 'polsub') | (t == 'polself'):
                Xret.append(Xpol)

        Xret = np.concatenate(Xret,axis=1)
        self.Npixtemp = Xret.shape[1]

        return Xret


    def save(self, ac, dext=''):
        """Save"""
        
        # Get filename and save
        fnout = self.fn[0]
        ind = fnout.find('dk')
        fnout = fnout[0:ind] + 'dkxxx' + fnout[(ind+5):]
        fnout = fnout.replace('tod','pairmaps')
        fnout = fnout.replace('.npy','.npz')
        dn,fn = os.path.split(fnout)
        dn = dn+dext
        if not os.path.isdir(dn):
            os.makedirs(dn)
        fnout = os.path.join(dn,fn)
        print('saving to {:s}'.format(fnout))
        sys.stdout.flush()
        np.savez_compressed(fnout, w=ac['w'], wz=ac['wz'], wcz=ac['wcz'],
                            wsz=ac['wsz'], wcc=ac['wcc'], wss=ac['wss'],
                            wcs=ac['wcs'], wwccv=ac['wwccv'], wwssv=ac['wwssv'], 
                            wwcsv=ac['wwcsv'], wsum=ac['wsum'],
                            wzsum=ac['wzsum'], wwv=ac['wwv'],
                            wczpred=ac['wczpred'], wszpred=ac['wszpred'], 
                            wczpred_cpm=ac['wczpred_cpm'], wszpred_cpm=ac['wszpred_cpm'], 
                            wctpol=ac['wctpol'], wstpol=ac['wstpol'],
                            b=ac['b'], bt=ac['bt'], cpmalpha=self.cpmalpha,
                            ra=self.ra, dec=self.dec, dpdkset=self.dpdkset,
                            nT=self.nT, tempxx=ac['tempxx'], tempyy=ac['tempyy'])



class map(object):

    def __init__(self, fn0=None):

        if fn0 is not None:
            self.fn = np.sort(glob('pairmaps/'+fn0))
            self.coaddpairmaps()
            self.makemap()

    def loadpairmap(self,fn):
        """Load pairmap"""
        print('loading {:s}'.format(fn))
        sys.stdout.flush()
        x = np.load(fn)
        ac = dict(x)
        x.close()
        return ac

    def coaddpairmaps(self):
        """Make TQU maps"""
        
        # Coadd over pairs
        print('making TQU maps...')

        self.acs = {}

        for j,fn in enumerate(self.fn):

            # Load
            ac = self.loadpairmap(fn)
            
            if j==0:
                self.nT = ac['nT']

            flds = ['wsum','wzsum','wwv', 'w','wz','wcz','wsz',
                    'wcc','wss','wcs','wwccv','wwssv','wwcsv',
                    'wctpol', 'wstpol', 'wczpred','wszpred',
                    'wczpred_cpm','wszpred_cpm']
            
            
            # Sum over first dimension
            for k in flds:
                ac[k] = np.nansum(ac[k],0)


            if j == 0:
                for k in flds:
                    self.acs[k] = dc(ac[k])
                self.ra = ac['ra']
                self.dec = ac['dec']
                self.b = [ac['b']*1.0]
                self.bt = [ac['bt']*1.0]
            else:
                for k in flds:
                    self.acs[k] += ac[k]
                self.b.append(ac['b'])
                self.bt.append(ac['bt'])


    def makemap(self):
        """Make TQU maps"""

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
        
        self.acs['wczpred'] = self.acs['wczpred'] / self.acs['w']
        self.acs['wszpred'] = self.acs['wszpred'] / self.acs['w']

        self.acs['wczpred_cpm'] = self.acs['wczpred_cpm'] / self.acs['w']
        self.acs['wszpred_cpm'] = self.acs['wszpred_cpm'] / self.acs['w']

        if 'wctpol' in self.acs.keys():
            self.acs['wctpol'] = self.acs['wctpol'] / self.acs['w']
            self.acs['wstpol'] = self.acs['wstpol'] / self.acs['w']


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


        self.Qpred = self.acs['e']*self.acs['wczpred'] + self.acs['f']*self.acs['wszpred']
        self.Upred = self.acs['f']*self.acs['wczpred'] + self.acs['g']*self.acs['wszpred']

        self.Qpred_cpm = self.acs['e']*self.acs['wczpred_cpm'] + self.acs['f']*self.acs['wszpred_cpm']
        self.Upred_cpm = self.acs['f']*self.acs['wczpred_cpm'] + self.acs['g']*self.acs['wszpred_cpm']

        if 'wctpol' in self.acs.keys():
            self.Qpol = self.acs['e']*self.acs['wctpol'] + self.acs['f']*self.acs['wstpol']
            self.Upol = self.acs['f']*self.acs['wctpol'] + self.acs['g']*self.acs['wstpol']                       

        self.Tw = 1/self.Tvar
        self.Tw = self.Tw / np.nanmax(self.Tw)
        self.Qw = 1/self.Qvar
        self.Qw = self.Qw / np.nanmax(self.Qw)
        self.Uw = 1/self.Uvar
        self.Uw = self.Uw / np.nanmax(self.Uw)
        self.Pw = 1/(self.Uvar+self.Qvar)
        self.Pw = self.Pw / np.nanmax(self.Pw)
        self.pixsize = self.dec[1,0] - self.dec[0,0]
        self.b = np.array(self.b)
        self.bt = np.array(self.bt)

        # If only 1 dk angle, get rid of one dim
        if len(self.T) == 1:
            self.pop(0)

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


    def filter(self, l=[20,200]):
        """Filter map"""
        print('filtering map to l={:d}-{:d}'.format(*l))
        
        sz = self.T.shape

        # Get angular frequency in radians
        reso = self.pixsize * np.pi/180
        ux = np.fft.fftshift(np.fft.fftfreq(sz[1], reso)) 
        uy = np.fft.fftshift(np.fft.fftfreq(sz[0], reso)) 

        # ux and uy
        ux, uy = np.meshgrid(ux,uy)

        # Convert radians^-1 to ell (ell = 2pi * rad^-1)
        lx = ux * 2 *np.pi
        ly = uy * 2 *np.pi
        lr = np.sqrt(lx**2 + ly**2)

        # Get map fts
        T = self.T*self.Tw
        Q = self.Q*self.Qw
        U = self.U*self.Uw

        indT = ~np.isfinite(T)
        indQ = ~np.isfinite(Q)
        indU = ~np.isfinite(U)

        T[~np.isfinite(T)] = 0
        Q[~np.isfinite(Q)] = 0
        U[~np.isfinite(U)] = 0

        Tft = np.fft.fftshift(np.fft.fft2(T))
        Qft = np.fft.fftshift(np.fft.fft2(Q))
        Uft = np.fft.fftshift(np.fft.fft2(U))

        ind = np.where( (lr<l[0]) | (lr>l[1]) )
        Tft[ind] = 0
        Qft[ind] = 0
        Uft[ind] = 0

        T = np.fft.ifft2(np.fft.ifftshift(Tft))
        Q = np.fft.ifft2(np.fft.ifftshift(Qft))
        U = np.fft.ifft2(np.fft.ifftshift(Uft))

        self.T = np.real(T) / self.Tw
        self.Q = np.real(Q) / self.Qw
        self.U = np.real(U) / self.Uw
     
        self.T[indT] = np.nan
        self.Q[indQ] = np.nan
        self.U[indU] = np.nan

        
    def pop(self,i):
        """Return map object with ith map only"""
        flds = ['T','Q','U','Tw','Qw','Uw','Pw',
                'Qpred','Upred','Qpred_cpm','Upred_cpm']
        for k in flds:
            setattr(self,k,getattr(self,k)[i])
        self.b = np.squeeze(self.b[:,i,:])

    def deproj(self):
        """Subtract T/Q/Upred fields"""
        self.Q -= self.Qpred
        self.U -= self.Upred

    def save(self, ext=None):
        """Save the map"""

        if ext is not None:
            extt = '_'+ext
        else:
            extt = ''

        fnout = self.fn[0][0:-9] + extt + '.npz'
        fnout = fnout.replace('pairmaps','maps')
        dn = os.path.dirname(fnout)
        if not os.path.isdir(dn):
            os.makedirs(dn)
        np.savez(fnout, T=self.T, Q=self.Q, U=self.U, Tw=self.Tw, Qw=self.Qw,
                 Uw=self.Uw, Tvar=self.Tvar, Qvar=self.Qvar, Uvar=self.Uvar,
                 QUcovar=self.QUcovar, Pw=self.Pw, ra=self.ra, dec=self.dec,
                 pixsize=self.pixsize, fn=self.fn, acs=self.acs, 
                 b=self.b, bt=self.bt, Qpred=self.Qpred, Upred=self.Upred,
                 Qpred_cpm=self.Qpred_cpm, Upred_cpm=self.Upred_cpm,
                 Qpol=self.Qpol, Upol=self.Upol)
        

    def load(self, fn):
        """Load the map"""
        x = np.load(fn)
        k = x.keys()
        for kk in k:
            if kk == 'acs':
                setattr(self, kk, x[kk].item())
            else:
                setattr(self, kk, x[kk])
        x.close()



