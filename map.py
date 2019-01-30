import numpy as np
from glob import glob
from fast_histogram import histogram2d as h2d
from scipy.interpolate import griddata
from ridge import Ridge
from sim import dist
from copy import deepcopy as dc
import sys
import os
import gc
import string
import random
from matplotlib.pyplot import *

def cosd(x):
    return np.cos(x*np.pi/180)
def sind(x):
    return np.sin(x*np.pi/180)
def tand(x):
    return np.tan(x*np.pi/180)

#def h2d(*args, **kwargs):
#    x, dum, dum = np.histogram2d(*args, **kwargs)
#    return x

def addmaps(m1, m2):
    """Add maps"""
    
    m3 = dc(m1)
    fld = ['T','Q','U','Qpred','Upred','Qpred_cpm','Upred_cpm','b','bcpm']
    for k in fld:
        setattr(m3,k, getattr(m1,k) + getattr(m2,k))

    fld = m1.acs.keys()
    for k in fld:
        m3.acs[k] += m2.acs[k]

    return m3 

class pairmap(object):
    
    def __init__(self, fn0='000/TnoP_notempnoise_dk???_0001.npz', cpmalpha=1,
                 dpt='deriv', cpmdpt='TR1.2+pol', dext='', cpmtype='lr',
                 dpdk='perdk', cpmdpdk='alldk'): 
        """Read in pairmaps, coadd into full map"""
        self.dpt = dpt
        self.dpdk = dpdk

        self.cpmdpt = cpmdpt
        self.cpmdpdk = cpmdpdk
        self.cpmalpha = cpmalpha
        self.cpmtype = cpmtype

        self.fn = np.sort(glob('tod/'+fn0))
        self.getinfo()
        self.getmapdefn()

        # Make save dir if it doesn't exist
        self.fnout = self.getfnout(dext)
        if os.path.isfile(self.fnout):
            print(self.fnout + ' exists, skipping')
            return

        # Binned design matrix filenames
        fn = self.tempfn(self.fnout)
        self.Xfn =  fn.replace('temp','X')
        self.Xsfn = fn.replace('temp','Xs')
        self.Xcfn = fn.replace('temp','Xc')
        
        # Coadd
        ac = self.coadd()
        self.save(ac, dext=dext)
        gc.collect()

    def getinfo(self):
        """Get TOD info and load first TOD"""

        dk = []
        for k,fn in enumerate(self.fn):
            if k==0:
                # Load first TOD
                self.tod = np.load(fn)
            dk.append(self.fn2dk(fn))
            
        self.dk = np.array(dk)
        self.udk = np.unique(self.dk)
        self.ndk = len(self.udk)

        perdk = [ [dk] for dk in self.udk ]
        alldk = [ self.udk ]

        if self.dpdk == 'perdk':
            self.dpdkset = perdk
        elif self.dpdk == 'alldk':
            self.dpdkset = alldk

        if self.cpmdpdk == 'perdk':
            self.cpmdpdkset = perdk
        elif self.cpmdpdk == 'alldk':
            self.cpmdpdkset = alldk

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

        #self.dec1d = np.arange(45.0, 70.0+0.01, self.pixsize) # BK
        #self.dec1d = np.arange(-13, 13+0.01, self.pixsize) # Equator
        self.dec1d = np.arange(20.0, 70.0+0.01, self.pixsize) # S4
        self.decbe = np.append(self.dec1d-self.pixsize/2., self.dec1d[-1]+self.pixsize/2)

        self.dx = self.pixsize / cosd(self.dec1d.mean())
        #self.ra1d  = np.arange(-55, 55+0.01, self.dx) # BK
        self.ra1d  = np.arange(-50, 50+0.01, self.dx) # S4
        self.rabe = np.append(self.ra1d-self.dx/2., self.ra1d[-1]+self.dx/2.)

        self.ra, self.dec = np.meshgrid(self.ra1d, self.dec1d)
        self.Npixmap = self.ra.size


    def coadd(self):
        """Coadd data into maps"""

        ndk = self.ndk
        sz  = (ndk, self.dec1d.size, self.ra1d.size)
        ac0 = np.zeros(sz, dtype='float32')
        
        # Initialize
        print('Initializing ac matrices')
        ac = {}
        flds = ['wsum','wzsum','wwv', 'w','wz','wcz','wsz',
                'wcc','wss','wcs','wwccv','wwssv','wwcsv']
        for k in flds: 
            ac[k] = dc(ac0)
        
        # Bin into pixels
        #bins = [self.decbe, self.rabe] # If using np.histogram2d
        bins = [len(self.decbe)-1, len(self.rabe)-1] # If using fast histogram2d
        rng = [(np.min(self.decbe),np.max(self.decbe)), (np.min(self.rabe),np.max(self.rabe))]
        
        # Initialize templates
        ac['wct'] = []
        ac['wst'] = []
        ac['wt'] = []

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
            val = np.load(fn0)

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

            gc.collect()

            fnt = self.tempfn(fn0)
            if j==0:
                fnti = fnt.replace('temp_','tempinfo_')
                fnti = fnti.replace('npy','npz')

                ti = np.load(fnti)
                ac['tempxx'] = ti['tempxx']
                ac['tempyy'] = ti['tempyy']
                self.nT = ti['nT']

            # Assemble template. A good N-dim sparse matrix package would be
            # nice right about now, but I cannot find one.
            if not os.path.isfile(self.Xfn):

                print('Binning act matrices')

                # Load template
                X = np.load(fnt, mmap_mode='r', allow_pickle=False)
                self.Npixtemp = X.shape[1]

                ind = np.where(ac['wz'][d] != 0 )
                szt = (self.Npixtemp, len(ind[0]))

                # Read in in chunks of 1000
                Nfac = 24000
                Nreads = np.int(np.ceil(self.Npixtemp*1.0 / Nfac))

                act = np.zeros(szt, dtype='float16')
                ast = np.zeros(szt, dtype='float16')
                at  = np.zeros(szt, dtype='float16')

                en = 0
                for l in range(Nreads):
                    st = en
                    en = np.min([st+Nfac, self.Npixtemp])

                    XX = X[:,st:en]*1.0
                    N = XX.shape[1]

                    for m in range(N):

                        if ((m+st) > (self.nT)) & ((m+st) <= (self.nT+5)):
                            # float16 can't exceed 2^16 = 65,000, otherwise inf, and
                            # T derivs have large values
                            fac = 1e-4
                        else:
                            fac = 1

                        z = h2d(y, x, bins=bins, range=rng, weights=w*c*XX[:,m])
                        act[m+st] += z[ind]*fac

                        z = h2d(y, x, bins=bins, range=rng, weights=w*s*XX[:,m])
                        ast[m+st] += z[ind]*fac

                        z = h2d(y, x, bins=bins, range=rng, weights=w*XX[:,m])
                        at[m+st] += z[ind]*fac


                    del XX
                    gc.collect()

                ac['wct'].append(act)
                ac['wst'].append(ast)
                ac['wt'].append(at)

                del X
                del act
                del ast
                del at
                val.close()
                gc.collect()

        

        if not os.path.isfile(self.Xfn):
            # Massage and save design matrices
            yy = np.ravel(ac['wz'])
            fitind = np.where(yy != 0)[0]
            yy = yy[fitind]

            X =  np.zeros((len(yy), self.Npixtemp), dtype='float16')
            Xc = np.zeros((len(yy), self.Npixtemp), dtype='float16')
            Xs = np.zeros((len(yy), self.Npixtemp), dtype='float16')

            for k in range(self.Npixtemp):
                X[:,k]  = np.concatenate([ac['wt'][j][k] for j in range(ndk)])
            np.save(self.Xfn, X, allow_pickle=False)
            del X
            del ac['wt']
            gc.collect()

            for k in range(self.Npixtemp):
                Xc[:,k] = np.concatenate([ac['wct'][j][k] for j in range(ndk)])
            np.save(self.Xcfn, Xc, allow_pickle=False)
            del Xc
            del ac['wct']
            gc.collect()

            for k in range(self.Npixtemp):
                Xs[:,k] = np.concatenate([ac['wst'][j][k] for j in range(ndk)])
            np.save(self.Xsfn, Xs, allow_pickle=False)
            del Xs
            del ac['wst']
            gc.collect()

            

        # Deproject
        ac = self.deproj(ac, self.dpdkset, self.dpt)
        ac = self.deprojcpm(ac, self.cpmdpdkset, self.cpmdpt)

        return ac


    def getdpind(self, t0, ac):
        """Get template indices"""

        ts = t0.split('+')
        dpind = []
        tind = []

        nT = self.nT

        startind = 0
        tempxx = np.ravel(ac['tempxx'])
        tempyy = np.ravel(ac['tempyy'])

        dpind = []
        tind = []
        for k, t in enumerate(ts):

            if t[0:2] == 'TR':

                # Direct T map
                r = np.float(t[2:])
                ind = np.where((np.abs(tempxx) <= r) &
                               (np.abs(tempyy) <= r))[0]
                dpind.append(ind)
                tind.append(np.ones(len(ind)))

            elif t=='deriv':
                # Deriv based templates
                ind = np.arange(nT,nT+6)
                dpind.append(ind)
                tind.append(np.ones(len(ind)))

            elif (t == 'pol'):
                ind = [nT+6]
                dpind.append(ind)
                tind.append(np.zeros(len(ind)))

        dpind = np.concatenate(dpind)
        tind = np.concatenate(tind).astype('bool')
        
        return dpind, tind


    def deproj(self, ac, dpdkset, t):
        """Deproject"""

        # What templates do we deproject with
        dpind, tind = self.getdpind(t, ac)

        # Map sizes
        sz = ac['wz'][0].shape

        # Initialize predictions
        ac['wzpred'] = np.zeros_like(ac['wz'])
        ac['wczpred'] = np.zeros_like(ac['wz'])
        ac['wszpred'] = np.zeros_like(ac['wz'])
        ac['b'] = []

        print('deprojecting...')
        sys.stdout.flush()

        # For regular fitting
        for dodks in dpdkset:
            
            dkind = [np.where(k==self.udk)[0][0] for k in dodks]

            y = np.ravel(ac['wz'][dkind])
            fitindarr = np.where(ac['wz'][dkind] != 0)
            fitind = np.where(y != 0)[0]
            y = y[fitind]

            # Load design matrices
            X  = np.load(self.Xfn,  mmap_mode='r', allow_pickle=False)
            Xc = np.load(self.Xcfn, mmap_mode='r', allow_pickle=False)
            Xs = np.load(self.Xsfn, mmap_mode='r', allow_pickle=False)

            # Least squares fit, pairdiff
            #b = np.linalg.lstsq(X[:,dpind].astype('float32'), y)[0]
            r = Ridge(alpha=0)
            r.fit(X[:,dpind].astype('float32'), y)
            b = r.coef_*1.0

            # Predict wcz, wsz
            ypred  = np.zeros_like(ac['wz'][dkind])
            ypredc = np.zeros_like(ac['wz'][dkind])
            ypreds = np.zeros_like(ac['wz'][dkind])
            ypred[fitindarr]  = X [:,dpind].dot(b)
            ypredc[fitindarr] = Xc[:,dpind].dot(b)
            ypreds[fitindarr] = Xs[:,dpind].dot(b)

            ac['b'].append(b)
            ac['wzpred'][dkind] = ypred
            ac['wczpred'][dkind] = ypredc
            ac['wszpred'][dkind] = ypreds

        ac['b'] = np.array(ac['b'])

        return ac

    def deprojcpm(self, ac, dpdkset, t):

        # What templates do we deproject with
        dpind, tind = self.getdpind(t, ac)

        ac['wczpred_cpm'] = np.zeros_like(ac['wz'])
        ac['wszpred_cpm'] = np.zeros_like(ac['wz'])
        ac['wctpol'] = np.zeros_like(ac['wz'])
        ac['wstpol'] = np.zeros_like(ac['wz'])
        ac['bcpm'] = []
        
        print('cpm deprojecting...')
        sys.stdout.flush()

        for dodks in dpdkset:
            
            dkind = [np.where(k==self.udk)[0][0] for k in dodks]

            y = np.ravel(ac['wz'][dkind])
            ypred = np.ravel(ac['wzpred'][dkind])
            fitindarr = np.where(ac['wz'][dkind] != 0)
            fitind = np.where(y != 0)[0]
            y = y[fitind]
            ypred = ypred[fitind]
            ra  = np.tile(np.ravel(self.ra),len(dkind))[fitind]
            dec = np.tile(np.ravel(self.dec),len(dkind))[fitind]
            pixid = np.tile(np.arange(self.ra.size),len(dkind))[fitind]

            # Load design matrices
            X  = np.load(self.Xfn,  mmap_mode='r', allow_pickle=False)
            Xc = np.load(self.Xcfn, mmap_mode='r', allow_pickle=False)
            Xs = np.load(self.Xsfn, mmap_mode='r', allow_pickle=False)                


            ##############
            # Now we predict pol on different pixels sets. Trying left/right
            # half. This will be a bit noisy, probably, but the idea is that the
            # cross spectrum of this prediction with the real map will yield an
            # unbiased measurement of T->P.
            medra = np.median(ra)
            ypredc_cpm = np.zeros_like(ac['wz'][dkind])
            ypreds_cpm = np.zeros_like(ac['wz'][dkind])

            if self.cpmtype == 'perpix':
                # Predict each pixel separately
                uid = np.unique(pixid)
                win = 18.0
            elif self.cpmtype == 'lr':
                # Predict left and right half separately
                uid = [0,1]
                win = 20.0
            elif self.cpmtype == 'col':
                # Predict each map column
                win = 18.0
                ura = np.unique(ra)
                uid = [pixid[ra == k] for k in ura]

            btemp = []
            for k in range(len(uid)):

                if np.mod(k,10) == 0:
                    print('CPM fitting pixel {0} of {1}'.format(k+1, len(uid)))
                    sys.stdout.flush()

                if uid == [0,1]:
                    # deproject left side w/ right side and vice versa
                    if k==0:
                        f = np.where(ra<=(medra-win))[0] # fit
                        p = np.where(ra>medra)[0] # pred
                    elif k==1:
                        f = np.where(ra>(medra+win))[0]
                        p = np.where(ra<=medra)[0]
                else:
                    # Deproject each pixel group separately with all other pixels
                    # outside certain distance
                    p = np.where(np.in1d(pixid, np.atleast_1d(uid[k])))[0]
                    d = np.zeros((len(p), len(ra)))
                    for j in range(len(p)):
                        d[j] = dist(ra[p[j]], dec[p[j]], ra, dec)
                    dmin = np.min(d, 0)
                    f = np.where(dmin > win)[0]

                # Can't fit too much or I get error "init_dgesdd failed init" in
                # SVD step of ridge
                if len(f) > 10000:
                    f = f[0:10000]


                #if ~np.all(tind):
                #    b = np.linalg.lstsq(np.atleast_2d(X[f,~tind]).T, (y-ypred)[f])[0]
                #    polpred = np.atleast_2d(X[f,~tind]).T.dot(b)
                #else:
                #    polpred = 0

                xx = np.zeros(len(tind))
                yy = np.zeros(len(tind))
                xx[tind] = ac['tempxx']
                yy[tind] = ac['tempyy']

                # Don't fit low r
                #noind = ((np.abs(xx) < 2) &
                #        (np.abs(yy) < 2))
                #X[:, noind] = 0 

                # Fit pol separately 
                #r = Ridge(alpha=self.cpmalpha)
                #r.fit(X[f][:,tind], (y-ypred)[f]-polpred)

                # Fit pol simultaneously
                #alpha = tind * self.cpmalpha
                r = Ridge(alpha=self.cpmalpha)
                r.fit(X[f][:,dpind].astype('float32'), (y-ypred)[f], zi = (~tind) )

                # Don't predict low r
                rind = ~((np.abs(xx) < 2.0) & (np.abs(yy) < 2.0))

                pp = tuple([g[p] for g in fitindarr])
                ypredc_cpm[pp] = r.predict(Xc[p][:,dpind], tind&rind)
                ypreds_cpm[pp] = r.predict(Xs[p][:,dpind], tind&rind)

                btemp.append(r.coef_*1.0)

            ac['wczpred_cpm'][dkind] = ypredc_cpm
            ac['wszpred_cpm'][dkind] = ypreds_cpm
            ac['bcpm'].append(btemp)

            if 'pol' in self.cpmdpt:
                wctpol = np.zeros_like(ac['wz'][dkind])
                wstpol = np.zeros_like(ac['wz'][dkind])
                polind = np.where(~tind)[0][0]
                wctpol[fitindarr] = Xc[:,dpind][:,polind]*1.0
                wstpol[fitindarr] = Xs[:,dpind][:,polind]*1.0
                ac['wctpol'][dkind] = wctpol
                ac['wstpol'][dkind] = wstpol
                
        return ac

    def randstring(self, size=6):
        """Generate random string of size size"""
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(size))


    def tempfn(self, fn0):
        """Return filename of ac structure containing template"""
        fn = fn0.replace('/sig','/temp')
        fn = fn.replace('/signoi','/temp')
        fn = fn.replace('/noi','/temp')
        fn = fn.replace('/EnoB','/temp')
        fn = fn.replace('/BnoE','/temp')
        fn = fn.replace('/TnoP','/temp')

        fn = fn.replace('npz','npy')
        fn = fn.replace('pairmaps/','/data/csheehy/cpmdeproj/')

        return fn


    def getfnout(self, dext):
        """Get filename for save"""

        fnout = self.fn[0]
        if self.udk.size > 1:
            ind = fnout.find('dk')
            fnout = fnout[0:ind] + 'dkxxx' + fnout[(ind+5):]
        fnout = fnout.replace('tod','pairmaps')
        fnout = fnout.replace('.npy','.npz')
        dn,fn = os.path.split(fnout)
        dn = dn+dext
        fnout = os.path.join(dn,fn)
        return fnout

    def save(self, ac, dext=''):
        """Save"""
        
        # Get filename and save
        fnout = self.getfnout(dext)
        dn = os.path.dirname(fnout)
        try:
            os.makedirs(dn)
        except:
            pass

        print('saving to {:s}'.format(fnout))
        sys.stdout.flush()
        np.savez_compressed(fnout, w=ac['w'], wz=ac['wz'], wcz=ac['wcz'],
                            wsz=ac['wsz'], wcc=ac['wcc'], wss=ac['wss'],
                            wcs=ac['wcs'], wwccv=ac['wwccv'], wwssv=ac['wwssv'], 
                            wwcsv=ac['wwcsv'], wsum=ac['wsum'],
                            wzsum=ac['wzsum'], wwv=ac['wwv'], wzpred=ac['wzpred'],
                            wczpred=ac['wczpred'], wszpred=ac['wszpred'], 
                            wczpred_cpm=ac['wczpred_cpm'], wszpred_cpm=ac['wszpred_cpm'], 
                            wctpol=ac['wctpol'], wstpol=ac['wstpol'],
                            b=ac['b'], bcpm=ac['bcpm'], cpmalpha=self.cpmalpha,
                            ra=self.ra, dec=self.dec, dpdkset=self.dpdkset,
                            cpmdpdkset=self.cpmdpdkset, dpt=self.dpt,
                            cpmdpt=self.cpmdpt, nT=self.nT, tempxx=ac['tempxx'],
                            tempyy=ac['tempyy'] )



class map(object):

    def __init__(self, fn0=None):

        if fn0 is not None:
            #self.fn = np.sort(glob('pairmaps/'+fn0))
            self.fn = np.sort(os.popen('ls pairmaps/'+fn0).read().split('\n'))
            self.fn = self.fn[self.fn != '']
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
                    'wctpol', 'wstpol', 'wzpred', 'wczpred','wszpred',
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
                self.bcpm = [ac['bcpm']*1.0]
            else:
                for k in flds:
                    self.acs[k] += ac[k]
                self.b.append(ac['b'])
                self.bcpm.append(ac['bcpm'])


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
        
        self.acs['wzpred'] = self.acs['wzpred'] / self.acs['w']
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
        
        bb = []
        for k in self.bcpm:
            b = np.mean(k, 1)
            bb.append(b)
        self.bcpm = np.array(bb)

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

    def getfnout(self, ext=None):
        """Get fn for save"""
        
        if ext is not None:
            extt = '_'+ext
        else:
            extt = ''

        fnout = self.fn[0][0:-9] + extt + '.npz'
        fnout = fnout.replace('pairmaps','maps')
        return fnout


    def save(self, ext=None):
        """Save the map"""

        fnout = self.getfnout(ext)
        dn = os.path.dirname(fnout)
        try:
            os.makedirs(dn)
        except:
            print('{:s} exists, skipping mkdir'.format(dn))

        np.savez(fnout, T=self.T, Q=self.Q, U=self.U, Tw=self.Tw, Qw=self.Qw,
                 Uw=self.Uw, Tvar=self.Tvar, Qvar=self.Qvar, Uvar=self.Uvar,
                 QUcovar=self.QUcovar, Pw=self.Pw, ra=self.ra, dec=self.dec,
                 pixsize=self.pixsize, fn=self.fn, acs=self.acs, 
                 b=self.b, bcpm=self.bcpm, Qpred=self.Qpred, Upred=self.Upred,
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



