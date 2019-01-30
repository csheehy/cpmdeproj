import numpy as np
from math import factorial as fact

def bl(l, fwhm=30.0):
    """Get beam window function, fwhm in arcmin"""
    sigma = ((fwhm/60.)*np.pi/180) / 2.3548
    bl = np.exp(-0.5*l*(l+1)*sigma**2)
    return bl


class beam(object):
    
    def __init__(self, fn=None):
        """Set up beam."""
        
        self.fwhm = 30.0 # Arcmin
        self.sigma = self.fwhm / 2.3548
        self.getmb(fn)

        return 

    def getmb(self, fn=None):
        """Have to figure out a main beam"""
        
        # Define x,y grid in arcmin
        npts = 100
        self.x = np.linspace(-4*self.fwhm,4*self.fwhm,npts)
        self.y = self.x
        if fn is not None:
            ac = dict(np.load('pairmaps/'+fn))
            xs = np.sqrt(ac['bt'][0].size)
            self.xx = ac['tempxx'].reshape(xs,xs) * 60
            self.yy = ac['tempyy'].reshape(xs,xs) * 60
            self.rr = np.sqrt(self.xx**2 + self.yy**2)
            self.phi = np.arctan2(self.yy,self.xx)
            self.mb = ac['bt'].mean(0).reshape(xs,xs)
        else:
            self.xx, self.yy = np.meshgrid(self.x,self.y)
            self.rr = np.sqrt(self.xx**2 + self.yy**2)
            self.phi = np.arctan2(self.yy,self.xx)

            # Get Gaussian beam component
            self.g = np.exp(-self.rr**2/(2*self.sigma**2))

            # Now get some stupid zernike modes
            self.n = [2, 5, 6, 7, 8]
            self.m = [\
                      np.arange(-2,2),
                      np.arange(-5,5),
                      np.arange(-6,6),
                      np.arange(-7,7),
                      np.arange(-8,8)]


            # Random coefficients
            self.coeff = []
            for k,n in enumerate(self.n):
                self.coeff.append(np.random.randn(len(self.m[k])))

            self.z = np.zeros_like(self.rr)
            rho = self.rr/(2*self.fwhm)
            phi = self.phi
            for k,n in enumerate(self.n):
                for j,m in enumerate(self.m[k]):
                    self.z += self.coeff[k][j] * self.zernike(rho, phi, n, m)

            # Normalize zernike beam
            self.z = self.z*self.g
            self.z = 0.2 * self.z / np.max(np.abs(self.z)) 

            # Main beam
            self.mb = self.g + self.z

        fac = np.nansum(self.mb)
        self.mb = self.mb / fac
        self.g = self.g / fac
        self.z = self.z / fac


    def getsl(self, fac):
        """Get a uniform quadrpolar sidelobe between 7 and 9 degrees"""
        
        x = np.linspace(-10*60, 10*60, 100) # arcmin
        y = np.linspace(-10*60, 10*60, 100) # arcmin

        xx,yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        phi = np.arctan2(yy,xx)
        z = self.zernike(rr/np.max(rr), phi, 2, 2)
        g = np.exp(-(rr-8*60)**2/(2*(0.5*60)**2))
        z = z*g
        z = z/np.max(z)

        # Normalize
        z = z * fac / np.sum(np.abs(z)) # 0.3% of power in sidelobe

        self.phisl = phi
        self.rrsl  = rr
        self.sl = z
        self.xsl = x
        self.ysl = y
        self.xxsl = xx
        self.yysl = yy

    def zernike(self, rho, phi, n, m_in, rnorm=None):
        """Zernike mode"""
        
        m = np.abs(m_in)

        if not self.iseven(n-m):
            # n-m is not even, zernike mode is zero
            return np.zeros_like(rho)

        # Get Zernike coefficient R
        kmax = (n-m)/2  # summation limit

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
            return R*np.cos(m*phi)
        else:
            return R*np.sin(m*phi)

            
    def iseven(self, x):
        """Return True if even, False if not"""
        if x/2.0 == np.round(x/2.0):
            return True
        else:
            return False

    def getquadrupole(self, x):
        """Return [x - rot90(x) + rot90(x,2) - rot90(x,3)]/4."""
        return (x - np.rot90(x) + np.rot90(x,2) - np.rot90(x,3))/4.
            
