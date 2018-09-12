import numpy as np
from math import factorial as fact

def bl(l, fwhm=30.0):
    """Get beam window function, fwhm in arcmin"""
    sigma = ((fwhm/60.)*np.pi/180) / 2.3548
    bl = np.exp(-0.5*l*(l+1)*sigma**2)
    return bl


class beam(object):
    
    def __init__(self):
        """Set up beam."""
        
        self.fwhm = 30.0 # Arcmin
        self.sigma = self.fwhm / 2.3548
        self.getmb()

        return 

    def getmb(self):
        """Have to figure out a main beam"""
        
        # Define x,y grid in arcmin
        npts = 100
        self.x = np.linspace(-4*self.fwhm,4*self.fwhm,npts)
        self.y = self.x
        self.xx, self.yy = np.meshgrid(self.x,self.y)
        self.rr = np.sqrt(self.xx**2 + self.yy**2)
        self.phi = np.arctan2(self.yy,self.xx)

        # Get Gaussian beam component
        self.g = np.exp(-self.rr**2/(2*self.sigma**2))
        
        # Now get some stupid zernike modes
        self.n = [1, 2, 3, 4]
        self.m = [[-1,1],
                  [-2,0,2],
                  [-3,3],
                  [-4,4]]
            
        # Random coefficients
        self.coeff = []
        for k,n in enumerate(self.n):
            self.coeff.append(np.random.randn(len(self.m[k])))

        self.z = np.zeros_like(self.rr)
        for k,n in enumerate(self.n):
            for j,m in enumerate(self.m[k]):
                self.z += self.coeff[k][j] * self.zernike(n,m,2*self.fwhm)

        # Normalize zernike beam
        self.z = self.z / np.max(self.z) / 10.0
        
        # Main beam
        self.mb = self.g*(1+self.z)
        self.mb = self.mb / np.nansum(self.mb)


    def zernike(self, n, m_in, rnorm=None):
        """Zernike mode"""
        
        m = np.abs(m_in)

        if not self.iseven(n-m):
            # n-m is not even, zernike mode is zero
            return np.zeros_like(self.rr)

        # Get Zernike coefficient R
        kmax = (n-m)/2  # summation limit

        # Normalize to max r along x axis by default
        if rnorm is None:
            rnorm = np.max(self.x)

        # rho coordinate
        rho = self.rr / rnorm

        R = 0
        for k in range(kmax+1):
            fac_num = (-1)**k * fact(n-k)
            fac_denom = fact(k) * fact((n+m)/2 - k) * fact((n-m)/2 - k)
            fac  = (fac_num/fac_denom) * rho**(n-2*k)
            R += fac

        R[rho>1] = 0

        # Big hack. What am I doing wrong?
        if self.iseven(n):
            R = -R

        # Now get zernike mode
        if m_in>=0:
            return R*np.cos(m*self.phi)
        else:
            return R*np.sin(m*self.phi)


    def iseven(self, x):
        """Return True if even, False if not"""
        if x/2.0 == np.round(x/2.0):
            return True
        else:
            return False

    def getquadrupole(self, x):
        """Return [x - rot90(x) + rot90(x,2) - rot90(x,3)]/4."""
        return (x - np.rot90(x) + np.rot90(x,2) - np.rot90(x,3))/4.
            
