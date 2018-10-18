import numpy as np
from glob import glob
from scipy import sparse
from fast_histogram import histogram2d as h2d
from sklearn.linear_model import Ridge
import fbpca
from copy import deepcopy as dc
import sys
import os

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
    
    def __init__(self, fn0='000/TnoP_notempnoise_dk???_0001.npy', cpmalpha=1,
                 cpmalphat=1):
        """Read in pairmaps, coadd into full map"""
        self.cpmalpha = cpmalpha
        self.cpmalphat = cpmalphat
        self.fn = np.sort(glob('tod/'+fn0))
        self.getinfo()
        self.getmapdefn()
        self.coadd()

        return

    def getinfo(self):
        """Get TOD info and load first TOD"""

        
        dk = []
        for k,fn in enumerate(self.fn):
            if k==0:
                # Load first TOD
                self.tod = np.load(fn).item()
            dk.append(self.fn2dk(fn))

        self.dk = np.array(dk)
        #self.dksets = [ [0,45,90,135], [180,225,270,315] ]
        #self.dksets = [ [0,45], [180,225] ]
        self.dksets = [ [0,45,90,135,180,225,270,315] ]

        self.Npixtemp = self.tod['X'].shape[1]
        self.nT = self.tod['nT']


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
        ndk = len(ac['wz'])

        # Number of T map coeffs
        Npixtemp = self.Npixtemp
        nT = self.nT
        

        # Initialize predictions
        ac['wzpred'] = np.zeros_like(ac['wz'])
        ac['wzsumpred'] = np.zeros_like(ac['wz'])
        ac['wczpred'] = np.zeros_like(ac['wz'])
        ac['wszpred'] = np.zeros_like(ac['wz'])
        ac['b'] = np.zeros((ndk,Npixtemp))
        ac['bt'] = np.zeros((ndk,nT))

        dkind = range(ndk)
        for i in dkind:

            print('deprojecting dk set {0} of {1}...'.format(i+1,ndk))
            sys.stdout.flush()

            #fitind = np.setxor1d(dkind, i)
            fitind = i  # Fit data and predicted data

            ydp = np.ravel(ac['wz'][fitind])
            fitinddp = np.where(ydp != 0)[0]
            ydp = ydp[fitinddp]

            ytdp = np.ravel(ac['wzsum'][fitind])
            fitindtdp = np.where(ytdp !=0 )
            ytdp = ytdp[fitindtdp]

            # Data to predict
            y = np.ravel(ac['wz'][i])
            predind = np.where(y != 0)[0]
            y = y[predind]

            yt = np.ravel(ac['wzsum'][i])
            predindt = np.where(yt !=0 )
            yt = yt[predindt]

            # Initialize design matrices
            Xdp = np.zeros((ydp.size, Npixtemp))
            X = np.zeros((y.size, Npixtemp))
            Xc = np.zeros((ac['wcz'][i].size, Npixtemp))
            Xs = np.zeros((ac['wcz'][i].size, Npixtemp))
            for k in range(Npixtemp):
                Xc[:,k] = np.ravel(ac['wct'][i,k,:,:])
                Xs[:,k] = np.ravel(ac['wst'][i,k,:,:])
                X[:,k] = np.ravel(ac['wt'][i,k,:,:])[predind]
                Xdp[:,k] = np.ravel(ac['wt'][fitind,k,:,:])[fitinddp]

            XT = X[:,0:nT]
            XTdp = Xdp[:,0:nT]
            
            ################
            # Fit T

            # Direct matrix inversion
            I = np.identity(nT)*self.cpmalphat
            bt = np.linalg.inv(XTdp.T.dot(XTdp) + I).dot(XTdp.T).dot(ytdp)
            ypredt  = XT.dot(bt[0:nT])
            
            # Built in sklearn solver
            #r = Ridge(alpha=self.cpmalpha)
            #r.fit(XTdp, ytdp)
            #ypredt = r.predict(XTdp)
            #bt = r.coef_

            # SVD decomp. using fast randomized SVD
            #k = np.min(XTdp.shape)-1
            #k = 1000
            #U,S,V = fbpca.pca(XTdp, k=k, n_iter=1)
            #D = np.diag(S/(S**2 + self.cpmalphat))
            #bt = V.T.dot(D).dot(U.T).dot(ytdp)
            #ypredt  = XT.dot(bt[0:nT])

            #############
            # Fit pol
            alph = np.ones(Npixtemp)*self.cpmalpha
            alph[nT:] = 0
            I = np.identity(Npixtemp)*alph
            b  = np.linalg.inv(Xdp.T.dot(Xdp) + I).dot(Xdp.T).dot(ydp)
            ypred = XT.dot(b[0:nT])

            # Built in sklearn solver
            #r.fit(XTdp, ydp)
            #b = r.coef_
            #ypred = r.predict(XT)

            # SVD decomp. using fast randomized SVD
            #D = np.diag(S/(S**2 + self.cpmalpha))
            #b = V.T.dot(D).dot(U.T).dot(ydp)
            #ypred  = XT.dot(b[0:nT])

            
            # Now stick prediction into map
            wzpred = np.zeros(ac['wcz'][i].size)
            wzsumpred = np.zeros(ac['wzsum'][i].size)

            wzpred[predind] = ypred
            wzsumpred[predindt] = ypredt

            wzpred = wzpred.reshape(ac['wcz'][i].shape)
            wzsumpred = wzsumpred.reshape(ac['wcz'][i].shape)

            wczpred = Xc[:,0:nT].dot(b[0:nT]).reshape(ac['wcz'][i].shape)
            wszpred = Xs[:,0:nT].dot(b[0:nT]).reshape(ac['wsz'][i].shape)

            ac['b'][i] = b
            ac['bt'][i] = bt
            ac['wzpred'][i] = wzpred
            ac['wzsumpred'][i] = wzsumpred
            ac['wczpred'][i] = wczpred
            ac['wszpred'][i] = wszpred

        # Get rid of templates for memory sake
        ac.pop('wt')
        ac.pop('wct')
        ac.pop('wst')

        return ac

        
    def coadd(self):
        """Coadd data into maps"""

        ndk = len(self.dksets)
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
            d = np.where([self.dk[j] in k for k in self.dksets])[0]
            if len(d)>0:
                # Exists
                d = d[0]
            else:
                print('skipping {:s}, not in defined dkset'.format(fn0))
                continue

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
            
            # Template
            for m in range(self.Npixtemp):
                t = val['X'][:, m]
                ac['wct'][d][m] += h2d(y, x, bins=bins, range=rng, weights=w*c*t)
                ac['wst'][d][m] += h2d(y, x, bins=bins, range=rng, weights=w*s*t)
                ac['wt'][d][m]  += h2d(y, x, bins=bins, range=rng, weights=w*t)
                


        # Deproject
        ac = self.deproj(ac)
                
        # Get filename and save
        fnout = self.fn[0]
        ind = fnout.find('dk')
        fnout = fnout[0:ind] + 'dkxxx' + fnout[(ind+5):]
        fnout = fnout.replace('tod','pairmaps')
        fnout = fnout.replace('.npy','.npz')
        dn = os.path.dirname(fnout)
        if not os.path.isdir(dn):
            os.makedirs(dn)
        print('saving to {:s}'.format(fnout))
        sys.stdout.flush()
        np.savez_compressed(fnout, w=ac['w'], wz=ac['wz'], wcz=ac['wcz'],
                            wsz=ac['wsz'], wcc=ac['wcc'], wss=ac['wss'],
                            wcs=ac['wcs'], wwccv=ac['wwccv'], wwssv=ac['wwssv'], 
                            wwcsv=ac['wwcsv'], wsum=ac['wsum'],
                            wzsum=ac['wzsum'], wwv=ac['wwv'],
                            b=ac['b'], bt=ac['bt'], wzpred=ac['wzpred'],
                            wzsumpred=ac['wzsumpred'], wczpred=ac['wczpred'],
                            wszpred=ac['wszpred'], cpmalpha=self.cpmalpha,
                            cpmalphat=self.cpmalphat, ra=self.ra,
                            dec=self.dec, dk=self.dksets, nT=self.nT)



class map(object):

    def __init__(self, fn0):

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
                
            flds = ['wsum','wzsum','wwv', 'w','wz','wcz','wsz',
                    'wcc','wss','wcs','wwccv','wwssv','wwcsv',
                    'wzpred', 'wzsumpred', 'wczpred','wszpred']
            
            
            # Sum over first dimension
            #for k in flds:
                #ac[k] = np.nansum(ac[k],0)
                #ac[k] = ac[k][0]

            #ac['b'] = np.nanmean(ac['b'],0)
            #ac['bt'] = np.nanmean(ac['bt'],0)


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
        self.Tpred = self.acs['wzsumpred'] / self.acs['wsum']

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
        m = dc(self)
        flds = ['T','Q','U','Tw','Qw','Uw','Pw','Qpred','Upred','Tpred']
        for k in flds:
            setattr(m,k,getattr(m,k)[i])
        m.b = np.squeeze(m.b[:,i,:])
        m.bt = np.squeeze(m.bt[:,i,:])
        return m

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
                 pixsize=self.pixsize, fn=self.fn, acs=self.acs) 
        

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



