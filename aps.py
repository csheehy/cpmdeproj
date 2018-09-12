import numpy as np
import beam
import healpy as hp

class aps(object):
    
    def __init__(self, m, m2=None):

        # Store map
        self.m = m
        self.m2 = m2
        
        # Get fourier plane weighting
        self.getFTweights()

        # Zero pad
        self.zeropad()

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


    def zeropad(self):
        """Zero pad and make finite"""
        self.T = self.zeropadsub(self.m.T)
        self.Q = self.zeropadsub(self.m.Q)
        self.U = self.zeropadsub(self.m.U)

        self.Tw = self.zeropadsub(self.m.Tw)
        self.Pw = self.zeropadsub(self.m.Pw)

        if self.m2 is not None:
            self.T2 = self.zeropadsub(self.m2.T)
            self.Q2 = self.zeropadsub(self.m2.Q)
            self.U2 = self.zeropadsub(self.m2.U)

    def makefinite(self):
        """Deal"""

        ind = (np.isfinite(self.T)) & (np.isfinite(self.Tw))
        if self.m2 is not None:
            ind = (ind) & (np.isfinite(self.T2))
        self.Tw[~ind] = 0
        self.T[~ind] = 0

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

        if self.m2 is not None:
            self.Tft2 = self.getfftsub(self.T2, self.Tw)
            self.Qft2 = self.getfftsub(self.Q2, self.Pw)
            self.Uft2 = self.getfftsub(self.U2, self.Pw)
            self.Eft2, self.Bft2 = self.qu2eb(self.Qft2, self.Uft2)


    def getfftsub(self, x, w):
        """Get FT sub"""
        fac1 = np.sqrt(np.prod(np.array(x.shape)) / np.nansum(w**2))
        fac2 = self.reso**2 * np.prod(np.array(x.shape))

        xft = np.fft.fftshift(np.fft.ifft2(x*w)) * fac1 * fac2

        return xft


    def qu2eb(self, Qft, Uft):
        """Q/U Fourier planes to E/B Fourier planes"""
        chi = np.arctan2(self.uy, self.ux) - np.pi/2
        c = np.cos(2*chi); s = np.sin(2*chi)
        Eft = Qft*c + Uft*s
        Bft = Qft*s - Uft*c
        return Eft, Bft

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

        delu = self.ux[0,1]-self.ux[0,0]

        fac = self.lr*(self.lr+1)/(2*np.pi) * delu**2 

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

