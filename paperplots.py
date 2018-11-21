import numpy as np
import healpy as hp
from scipy.optimize import curve_fit
from matplotlib.pyplot import *
ion()

class loaddata(object):

    def __init__(self):
        # Load data

        # S4 large aperture noise
        self.S4noisemap = hp.read_map('input_maps/S4_noise_map_new.fits',field=(0,1,2))
        self.S4noisealm = hp.map2alm(self.S4noisemap)
        self.S4noisecl  = hp.alm2cl(self.S4noisealm)

        # Get beam
        self.b = dict(np.load('beams/beam_0000.npz'))


class paperplots(object):

    def __init__(self, d):
        """Plots are:
        plotbeam() - beam
        plotS4noi() - S4 noise Cl's
        """
        self.d = d


    def plotbeam(self):
        """Plot beam"""
        close(1)
        figure(1, figsize=(5,8))
        
        ba = self.d.b['Ba'].item().mb
        bb = self.d.b['Bb'].item().mb
        xx = self.d.b['Ba'].item().xx/60. # arcmin->deg
        yy = self.d.b['Ba'].item().yy/60. # arcmin->deg

        ba = ba
        bb = bb


        # Fit ellip gaussians
        fun = lambda x,p0,p1,p2,p3,p4: p0*np.exp(-((x[0] - p1)**2/(2*p2**2) + (x[1]-p3)**2/(2*p4**2)))
                                       

        x0 = (1.0, 0.0, .21, 0.0, .21)
        xy = (np.ravel(xx),np.ravel(yy))

        popt, pcov = curve_fit(fun, xy, np.ravel(ba))
        bafit = fun(xy, *popt).reshape(*ba.shape)


        popt, pcov = curve_fit(fun, xy, np.ravel(bb))
        bbfit = fun(xy, *popt).reshape(*ba.shape)

        normfac = np.max((ba+bb)/2.)

        # Diff beam
        subplot(2,1,1)
        imshow((ba-bb)/normfac, extent=(xx.min(),xx.max(),yy.max(),yy.min()), aspect='equal')
        c  = colorbar()
        ax = gca()
        ax.set_xticks(ax.get_xticks()[0::2])
        ax.set_yticks(ax.get_yticks()[0::2])

        ylabel('degrees');
        title('differential beam')

        subplot(2,1,2)
        imshow((ba-bafit - (bb-bbfit))/normfac, extent=(xx.min(),xx.max(),yy.max(),yy.min()), aspect='equal')
        c  = colorbar()
        ax = gca()
        ax.set_xticks(ax.get_xticks()[0::2])
        ax.set_yticks(ax.get_yticks()[0::2])
        xlabel('degrees');
        ylabel('degrees');
        title('after removing ellip. Gaussian')

        tight_layout()
        savefig('figs/beam.pdf', bbox_inches='tight')
        


    def plotS4noi(self):
        """Plot S4 noise Cl"""

        close(1)
        figure(1, figsize=(5,5))
        
        SPT = np.loadtxt('SPT_Nl_irreg.csv', delimiter=',').T

        cl = self.d.S4noisecl
        l = np.arange(cl[0].size)
        ll = np.arange(10000)

        loglog(SPT[0], SPT[1], '.k', label='SPTpol 500 deg$^2$')
        fun = lambda l, sigmap, lknee, lexp: np.log(4*np.pi / (41253.*60**2) * (1+(l/lknee)**lexp) * sigmap**2)
        #popt, pcov = curve_fit(fun, SPT[0], np.log(SPT[1]))
        #loglog(ll, np.exp(fun(ll,*popt)), 'k', label=r'SPT model (${:0.2f}\ \mu K\ arcmin, \ell_{{knee}}={:0.1f}, \expon.={:0.2f}$)'.format(*popt))
        sigmap = 9.4; lknee = 250.0; lexp = -1.8
        loglog(ll, np.exp(fun(ll,sigmap,lknee,lexp)),'k', label=r'SPT best fit ($9.4 \mu K\ arcmin$)')

        loglog(l,cl[0], label='S4 sim (rlz 1)', color='gray')
        sigmap = 1.2; lknee = 250.0; lexp = -1.8
        loglog(ll, np.exp(fun(ll,sigmap,lknee,lexp)),'k', label=r'S4 best fit ($1.2 \mu K\ arcmin$)')
        
        xlim(20,6000)
        ylim(1e-7,1e-3)
        grid('on')

        legend(loc='upper right')

        xlabel(r'Multipole $\ell$')
        ylabel(r'$C_{\ell}^{EE,noise} [\mu K^2]$')
        
        tight_layout()
        savefig('figs/S4Nl.pdf', bbox_inches='tight')
