import numpy as np
import beam
import healpy as hp
from scipy.signal import convolve2d
from matplotlib.pyplot import *

class aps(object):
    
    def __init__(self, m, m2=None, ext='', ext2=''):

        # Store map
        self.m = m
        self.m2 = m2
        
        # Get fourier plane weighting
        self.getFTweights()

        # Zero pad
        self.zeropad(ext,ext2)

        # Deal with NaNs
        self.makefinite()

        # Get FFT axes
        self.getfreqaxes()
        
        # Get FFTs
        self.getffts()

        # Get bins
        self.getbins()

        # Bin into bandpowers
        self.getps()


    def zeropad(self, ext, ext2):
        """Zero pad and make finite"""
        self.T = self.zeropadsub(self.m.T)
        self.Q = self.zeropadsub(getattr(self.m, 'Q'+ext))
        self.U = self.zeropadsub(getattr(self.m, 'U'+ext))

        self.Tw = self.zeropadsub(self.m.Tw)
        self.Pw = self.zeropadsub(self.m.Pw)

        if self.m2 is not None:
            self.T2 = self.zeropadsub(self.m2.T)
            self.Q2 = self.zeropadsub(getattr(self.m2, 'Q'+ext2))
            self.U2 = self.zeropadsub(getattr(self.m2, 'U'+ext2))

    def makefinite(self):
        """Deal"""

        ind = (np.isfinite(self.T)) & (np.isfinite(self.Tw))
        if self.m2 is not None:
            ind = (ind) & (np.isfinite(self.T2))
        self.Tw[~ind] = 0
        self.T[~ind] = 0
        if self.m2 is not None:
            self.T2[~ind] = 0

        ind = (np.isfinite(self.Q)) & (np.isfinite(self.Pw)) & (np.isfinite(self.U))
        if self.m2 is not None:
            ind = (ind) & (np.isfinite(self.Q2)) & (np.isfinite(self.U2))
        self.Pw[~ind] = 0
        self.Q[~ind] = 0
        self.U[~ind] = 0

        if self.m2 is not None:
            self.Q2[~ind] = 0
            self.U2[~ind] = 0


    def getfreqaxes(self):
        """Get frequency axes"""

        xs = len(self.T)

        # Get angular frequency in radians
        self.reso = self.m.pixsize * np.pi/180
        u = np.fft.fftshift(np.fft.fftfreq(xs, self.reso)) 

        # ux and uy
        self.ux, self.uy = np.meshgrid(u,u)
        self.del_u = u[1]-u[0]

        # Convert radians^-1 to ell (ell = 2pi * rad^-1)
        self.lx = self.ux * 2 *np.pi
        self.ly = self.uy * 2 *np.pi
        self.lr = np.sqrt(self.lx**2 + self.ly**2)

        return


    def zeropadsub(self, x):
        """Zero pad to make square"""

        # Next power of two
        n = np.max(np.array(x.shape))            
        pow = np.ceil(np.log2(n))
        sz = 2**pow

        # Padded array
        y = np.zeros((sz,sz))

        # Mid point
        x0 = np.round(sz/2.)

        # Size of original array
        szy = x.shape[0]
        szx = x.shape[1]

        # Start/end x and y
        sx = np.round(x0 - szx/2.)
        sy = np.round(x0 - szy/2.)
        ex = sx + szx
        ey = sy + szy

        # Pad
        y[sy:ey, sx:ex] = x

        # Store for extracting later
        self.sx = sx; self.ex = ex; self.sy = sy; self.ey = ey

        return y


    def getFTweights(self):
        """Get F-plane weighting"""

        self.Pw = 1/(self.m.Qvar + self.m.Uvar)
        self.Pw /= np.nanmax(self.Pw)
        self.Tw = self.m.Tw

        if self.m2 is not None:
            Pw2 = 1/(self.m2.Qvar + self.m2.Uvar)
            Pw2 /= np.nanmax(Pw2)
            self.Pw = np.sqrt(self.Pw*Pw2)
            self.Tw = np.sqrt(self.Tw*self.m2.Tw)


    def getffts(self):
        """Get FFTs"""

        self.Tft1 = self.getfftsub(self.T, self.Tw)
        self.Qft1 = self.getfftsub(self.Q, self.Pw)
        self.Uft1 = self.getfftsub(self.U, self.Pw)
        self.Eft1, self.Bft1 = self.qu2eb(self.Qft1, self.Uft1)
        #self.Eft1, self.Bft1 = self.qu2eb_pure(self.Q, self.U, self.Pw)

        if self.m2 is not None:
            self.Tft2 = self.getfftsub(self.T2, self.Tw)
            self.Qft2 = self.getfftsub(self.Q2, self.Pw)
            self.Uft2 = self.getfftsub(self.U2, self.Pw)
            self.Eft2, self.Bft2 = self.qu2eb(self.Qft2, self.Uft2)
            #self.Eft2, self.Bft2 = self.qu2eb_pure(self.Q2, self.U2, self.Pw)

    def getfftsub(self, x, w, applyfac=True):
        """Get FT sub"""
        if applyfac:
            fac1 = np.sqrt(np.prod(np.array(x.shape)) / np.nansum(w**2))
            fac2 = self.reso**2 * np.prod(np.array(x.shape))
        else:
            fac1 = 1
            fac2 = 1

        xft = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x*w))) * fac1 * fac2

        return xft


    def qu2eb(self, Qft, Uft):
        """Q/U Fourier planes to E/B Fourier planes"""
        chi = np.arctan2(self.uy, self.ux) - np.pi/2
        c = np.cos(2*chi); s = np.sin(2*chi)
        Eft = Qft*c + Uft*s
        Bft = Qft*s - Uft*c
        return Eft, Bft


    def qu2eb_pure(self, Q, U, w):
        """Kendrick Smith style pure-B estimator"""

        # Eft
        Qft = np.fft.fftshift(np.fft.fft2(Q*w))
        Uft = np.fft.fftshift(np.fft.fft2(U*w))

        chi = np.arctan2(self.uy, self.ux)# - np.pi/2
        c = np.cos(2*chi); s = np.sin(2*chi)

        Eft = Qft*c + Uft*s

        # Bft

        # Smooth window a bit
        ker = np.ones((7,7))
        wsm = convolve2d(w, ker, 'same')
        #wsm = w*1.0

        # Get window derivs
        warr = self.calc_window_derivs(wsm)
        
        # Get FFTs of Q/U times weight mask and its derivs
        fftarr = np.zeros((7,Q.shape[0],Q.shape[1]), dtype=complex)

        # FFT(Q*W)
        fftarr[0] = np.fft.fft2(Q*warr[0])
        # FFT(U*W)                       
        fftarr[1] = np.fft.fft2(U*warr[0])
        # FFT(Q*dW/dx)                   
        fftarr[2] = np.fft.fft2(Q*warr[1])
        # FFT(Q*dW/dy)                   
        fftarr[3] = np.fft.fft2(Q*warr[2])
        # FFT(U*dW/dy)                  
        fftarr[4] = np.fft.fft2(U*warr[2])
        # FFT(U*dW/dx)                   
        fftarr[5] = np.fft.fft2(U*warr[1])
        # FFT(Q*d2W/dx/dy + U*(d2W/dy2 - d2W/dx2))
        fftarr[6] = np.fft.fft2(2*Q*warr[5] + U*(warr[4]-warr[3]))

        for k in range(len(fftarr)):
            fftarr[k] = np.fft.fftshift(fftarr[k])
        
        # Add them up with right prefactors to get pure-B
        sarg = np.sin(chi)
        sarg2 = np.sin(2*chi)
        carg = np.cos(chi)
        carg2 = np.cos(2*chi)

        Bft = -sarg2*fftarr[0] + carg2*fftarr[1] - \
              2*np.complex(0,1)/self.lr * \
              (sarg*fftarr[2] + carg*fftarr[3] + \
               sarg*fftarr[4] - carg*fftarr[5]) + \
              fftarr[6]/self.lr**2

        # Get fac
        winfacB = np.nansum(warr[0]**2) / np.prod(np.array(Q.shape))
        winfacE = np.nansum(w**2) / np.prod(np.array(Q.shape))
        norm = np.prod(np.array(Q.shape)) * self.del_u**2
        Bft = Bft / np.sqrt(winfacB) / norm
        Eft = Eft / np.sqrt(winfacE) / norm

        return Eft, Bft


    def calc_window_derivs(self, w):
        """Get window derivatives"""
        dx = self.m.pixsize * np.pi/180
        dy = self.m.pixsize * np.pi/180

        dwdx1 = w - np.roll(w,1,axis=1)
        dwdx2 = np.roll(w,-1,axis=1) - w
        dwdx_temp = (dwdx1+dwdx2)/2
        dwdx = dwdx_temp/dx

        dwdy1 = w - np.roll(w,1,axis=0)
        dwdy2 = np.roll(w,-1,axis=0) - w
        dwdy_temp = (dwdy1+dwdy2)/2
        dwdy = dwdy_temp/dy

        d2wdx1 = dwdx_temp - np.roll(dwdx_temp,1,axis=1)
        d2wdx2 = np.roll(dwdx_temp,-1,axis=1) - dwdx_temp
        d2wdx = (d2wdx1 + d2wdx2)/(2*dx)

        d2wdy1 = dwdy_temp - np.roll(dwdy_temp,1,axis=0)
        d2wdy2 = np.roll(dwdy_temp,-1,axis=0) - dwdy_temp
        d2wdy = (d2wdy1 + d2wdy2)/(2*dy)

        d2dxdy1 = dwdy_temp - np.roll(dwdy_temp,1,axis=1)
        d2dxdy2 = np.roll(dwdy_temp,-1,axis=1) - dwdy_temp
        d2dxdy = (d2dxdy1 + d2dxdy2)/(2*dx)

        res = np.zeros((6,w.shape[0],w.shape[1]))
        res[0] = w
        res[1] = dwdx
        res[2] = dwdy
        res[3] = d2wdx
        res[4] = d2wdy
        res[5] = d2dxdy

        return res


    def eb2qu(self, Eft, Bft):
        """E/B F-planes to Q/U F-planes"""
        chi = -np.arctan2(self.uy, self.ux) + np.pi/2
        c = np.cos(chi); s = np.sin(chi)
        Qft = Eft*c + Bft*s
        Uft = Eft*s - Bft*c
        return Qft, Uft


    def getbins(self, lmin=20, lmax=500, bw=35):
        """Get bins"""
        self.le = np.arange(lmin, lmax+1, bw)
        self.l = (self.le[0:-1] + self.le[1:])/2.0


    def binfft(self, ft1, ft2, w=None):
        """Multiply complex map alms, take real component, and bin l(l+1) C_l / 2pi
        in radial bins. Then return D_l."""

        if w is None:
            w = np.ones_like(np.real(ft1))

        fac = self.lr*(self.lr+1)/(2*np.pi) * self.del_u**2 

        z = np.real(ft1 * np.conj(ft2)) * fac

        dl = []

        for k in range(len(self.le)-1):
            lo = self.le[k]
            hi = self.le[k+1]
            ind = np.where( (self.lr>=lo) & (self.lr<hi) )
            dl.append(np.nansum(z[ind]*w[ind])/np.nansum(w[ind]))
        dl = np.array(dl)

        return dl


    def getps(self):
        """Get power spectra"""

        self.dl = np.zeros((3, len(self.l)))
        self.dl[0] = self.binfft(self.Tft1, self.Tft1)
        self.dl[1] = self.binfft(self.Eft1, self.Eft1)
        self.dl[2] = self.binfft(self.Bft1, self.Bft1)

        if self.m2 is not None:
            self.dl = [self.dl]
            dl = np.zeros((3, len(self.l)))
            dl[0] = self.binfft(self.Tft2, self.Tft2)
            dl[1] = self.binfft(self.Eft2, self.Eft2)
            dl[2] = self.binfft(self.Bft2, self.Bft2)
            self.dl.append(dl)

            dl = np.zeros((3, len(self.l)))
            dl[0] = self.binfft(self.Tft1, self.Tft2)
            dl[1] = self.binfft(self.Eft1, self.Eft2)
            dl[2] = self.binfft(self.Bft1, self.Bft2)
            self.dl.append(dl)

            self.dl = np.array(self.dl)
