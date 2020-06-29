import numpy as np
import healpy as hp
import map
from scipy.optimize import curve_fit
from matplotlib.pyplot import *
from matplotlib.ticker import MaxNLocator
import time
from sim import readcambfits
ion()


def addcbar(ax=None, w=0.02, fmt=None, sci=False, location=None):

    pause(0.01)
    show()

    if ax is None:
        ax = gca()
    p = ax.get_position()

    if location is None:
        cax = axes([p.x1, p.y0, w, p.height])
        c = colorbar(cax=cax, format=fmt)
    elif location == 'top':
        cax = axes([p.x0, p.y1, p.width, w])
        c = colorbar(cax=cax, format=fmt, orientation='horizontal', ticklocation='top')

    if sci & (fmt is None):
        c.formatter.set_powerlimits((0, 0))
        c.update_ticks()
    draw()

    return c
    
class loaddata(object):

    def __init__(self):
        # Load data

        # S4 large aperture noise
        #self.S4noisemap = hp.read_map('input_maps/S4_noise_map_new.fits',field=(0,1,2))
        #self.S4noisealm = hp.map2alm(self.S4noisemap)
        #self.S4noisecl  = hp.alm2cl(self.S4noisealm)

        # Get beam
        self.b = dict(np.load('beams/beam_v3_uqpsl_0p3pct_0000.npz'))

        # Load ac
        self.acderiv = np.load('pairmaps/002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk/TnoP_r0000_dkxxx_0000.npz')
        self.acTR = np.load('pairmaps/002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk/TnoP_r0000_dkxxx_0000.npz')
        self.acTRnosl = np.load('pairmaps/002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk/TnoPnosl_r0000_dkxxx_0000.npz')

        # Load maps
        rlz = 0

        suff = '002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk'

        self.mt = map.map()
        self.ms = map.map()
        self.me = map.map()
        self.mn = map.map()
        self.mtnosl = map.map()
        self.msnosl = map.map()
        self.menosl = map.map()

        self.mt.load('maps/'+suff+'/TnoP_r{:04d}_dkxxx.npz'.format(rlz))
        self.ms.load('maps/'+suff +'/sig_r{:04d}_dkxxx.npz'.format(rlz))
        self.me.load('maps/'+suff+'/EnoB_r{:04d}_dkxxx.npz'.format(rlz))
        self.mn.load('maps/'+suff +'/noi_r{:04d}_dkxxx.npz'.format(rlz))
        self.mtnosl.load('maps/'+suff+'/TnoPnosl_r{:04d}_dkxxx.npz'.format(rlz))
        self.msnosl.load('maps/'+suff +'/signosl_r{:04d}_dkxxx.npz'.format(rlz))
        self.menosl.load('maps/'+suff+'/EnoBnosl_r{:04d}_dkxxx.npz'.format(rlz))

        self.ms.Q -= self.mt.Q; self.ms.Qpred_cpm -= self.mt.Qpred_cpm; self.ms.Qpred -= self.mt.Qpred
        self.ms.U -= self.mt.U; self.ms.Upred_cpm -= self.mt.Upred_cpm; self.ms.Upred -= self.mt.Upred
        self.msnosl.Q -= self.mtnosl.Q; self.msnosl.Qpred_cpm -= self.mtnosl.Qpred_cpm; self.msnosl.Qpred -= self.mtnosl.Qpred
        self.msnosl.U -= self.mtnosl.U; self.msnosl.Upred_cpm -= self.mtnosl.Upred_cpm; self.msnosl.Upred -= self.mtnosl.Upred

        suff = '002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk'

        self.mtd = map.map()
        self.msd = map.map()
        self.med = map.map()
        self.mnd = map.map()
        self.mtnosld = map.map()
        self.msnosld = map.map()
        self.menosld = map.map()

        self.mtd.load('maps/'+suff+'/TnoP_r{:04d}_dkxxx.npz'.format(rlz))
        self.msd.load('maps/'+suff +'/sig_r{:04d}_dkxxx.npz'.format(rlz))
        self.med.load('maps/'+suff+'/EnoB_r{:04d}_dkxxx.npz'.format(rlz))
        self.mnd.load('maps/'+suff +'/noi_r{:04d}_dkxxx.npz'.format(rlz))
        self.mtnosld.load('maps/'+suff+'/TnoPnosl_r{:04d}_dkxxx.npz'.format(rlz))
        self.msnosld.load('maps/'+suff +'/signosl_r{:04d}_dkxxx.npz'.format(rlz))
        self.menosld.load('maps/'+suff+'/EnoBnosl_r{:04d}_dkxxx.npz'.format(rlz))

        self.msd.Q -= self.mtd.Q; self.msd.Qpred_cpm -= self.mtd.Qpred_cpm; self.msd.Qpred -= self.mtd.Qpred
        self.msd.U -= self.mtd.U; self.msd.Upred_cpm -= self.mtd.Upred_cpm; self.msd.Upred -= self.mtd.Upred
        self.msnosld.Q -= self.mtnosld.Q; self.msnosld.Qpred_cpm -= self.mtnosld.Qpred_cpm; self.msnosld.Qpred -= self.mtnosld.Qpred
        self.msnosld.U -= self.mtnosld.U; self.msnosld.Upred_cpm -= self.mtnosld.Upred_cpm; self.msnosld.Upred -= self.mtnosld.Upred


        for k in ['Q','U','Qpred','Upred','b','bcpm','Qpred_cpm','Upred_cpm']:
            setattr(self.mn,  k, getattr(self.mnd,k)/np.sqrt(10))
            setattr(self.mnd, k, getattr(self.mnd,k)/np.sqrt(10))

        self.msnd     = map.addmaps(self.msd,     self.mnd)
        self.msnnosld = map.addmaps(self.msnosld, self.mnd)

        # Load aps
        self.aTRdp    = dict(np.load('aps/002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk_dp.npz'))
        self.aTR      = dict(np.load('aps/002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk_nodp.npz'))
        self.aderiv   = dict(np.load('aps/002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk_nodp.npz'))
        self.aderivdp = dict(np.load('aps/002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk_dp.npz'))

        self.aTRdpnosl    = dict(np.load('aps/002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk_dpnosl.npz'))
        self.aTRnosl      = dict(np.load('aps/002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk_nodpnosl.npz'))
        self.aderivnosl   = dict(np.load('aps/002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk_nodpnosl.npz'))
        self.aderivdpnosl = dict(np.load('aps/002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk_dpnosl.npz'))

        # Correct spectra for beam
        bl = hp.gauss_beam(30.0/60.*np.pi/180)
        lbl = np.arange(len(bl))
        fac = np.interp(self.aTR['l'],lbl,bl**2)
        flds = self.aTR.keys()
        flds.remove('l')
        for f in flds:
            self.aTRdp[f] /= fac
            self.aTR[f] /= fac
            self.aderiv[f] /= fac
            self.aderivdp[f] /= fac

            self.aTRdpnosl[f] /= fac
            self.aTRnosl[f] /= fac
            self.aderivnosl[f] /= fac
            self.aderivdpnosl[f] /= fac


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
        figure(1, figsize=(8,8))
        
        ba = self.d.b['Ba'].item().mb*1.0
        bb = self.d.b['Bb'].item().mb*1.0
        xx = self.d.b['Ba'].item().xx/60. # arcmin->deg
        yy = self.d.b['Ba'].item().yy/60. # arcmin->deg

        # Normalize everything to peak
        fac = 1./ (ba+bb).max()
        ba = ba*fac
        bb = bb*fac
        diffbeam = ba - bb

        # Fit ellip gaussians
        #def fun(x, p0, p1, p2, p3, p4):
        #    return p0*np.exp(-((x[0] - p1)**2/(2*p2**2) + (x[1]-p3)**2/(2*p4**2)))

        def fun(x, p0, p1, p2, p3, p4, p5):
            sigmax = p2
            sigmay = p4
            N = p0
            dx = p1
            dy = p3
            theta = p5
            xx = x[0]
            yy = x[1]

            A = np.cos(theta)**2 / sigmax**2 + np.sin(theta)**2/sigmay**2
            B = 2 * (1/sigmax**2 - 1/sigmay**2)*np.sin(theta)*np.cos(theta)
            C = np.sin(theta)**2 / sigmax**2 + np.cos(theta)**2 / sigmay ** 2

            z = A*(xx-dx)**2 + B*(xx-dx)*(yy-dy) + C*(yy-dy)**2

            return  N*np.exp(-0.5*z)


        x0 = (6e-3, 0.001, .21, 0.001, .21, 0)
        xy = (np.ravel(xx),np.ravel(yy))

        popt, pcov = curve_fit(fun, xy, np.ravel(ba), x0, method='trf')
        bafit = fun(xy, *popt).reshape(*ba.shape)

        popt, pcov = curve_fit(fun, xy, np.ravel(bb), x0, method='lm')
        bbfit = fun(xy, *popt).reshape(*ba.shape)

        yfit = bafit - bbfit


        # linear templates
        sigma = 0.5/2.3548
        rr = np.sqrt(xx**2+yy**2)
        T1 = np.exp(-rr**2/(2*sigma**2))
        T2 = T1*xx/sigma**2
        T3 = T1*yy/sigma**2
        T4 = T1/sigma**4 * (xx**2 + yy**2 - 2*sigma**2)
        T5 = T1/sigma**4 * (xx**2 + yy**2)
        T6 = 2*xx*yy*T1 / sigma**4
        T = [T1,T2,T3,T4,T5,T6]

        # Fit
        y = np.ravel(diffbeam)
        X = []
        for k in T:
            X.append(np.ravel(k))
        X = np.array(X).T
        b = np.linalg.lstsq(X,y)[0]
        #yfit = X.dot(b).reshape(diffbeam.shape)

        ###########
        # Plot
        cl = np.array([-.1,.1])
        xl = (-1.2,1.2)

        subplots_adjust(wspace=0, hspace=0)

        subplot(2,2,1)
        imshow(diffbeam, extent=(xx.min(),xx.max(),yy.max(),yy.min()), aspect='equal')
        clim(*cl); xlim(*xl); ylim(*xl)
        ax = gca()
        grid('on')
        text(-1,.95,'differential beam')
        gca().set_xticklabels('')
        title('Input to sim')

        subplot(2,2,2)
        z = self.d.acTR['b'].reshape(24,24)
        fac = np.sum(np.max(diffbeam)) / np.max(np.abs(z))
        z *= fac
        imshow(z, extent=(-1.2,1.2,1.2,-1.2), aspect='equal')
        clim(*cl);xlim(*xl); ylim(*xl)
        grid('on')
        ax = gca()
        text(-1,.95,'Arbitrary fit param')
        gca().set_xticklabels('')
        gca().set_yticklabels('')
        title('Best fit')
        c = addcbar()
        c.set_ticks([-.05,0,0.05])


        cl = [-0.02,0.02]
        subplot(2,2,3)
        imshow(diffbeam-yfit, extent=(xx.min(),xx.max(),yy.max(),yy.min()), aspect='equal')
        clim(*cl); xlim(*xl); ylim(*xl)
        grid('on')
        ax = gca()
        text(-1,.95,'after removing ellip. Gaussian')

        subplot(2,2,4)
        z = self.d.acderiv['bcpm'].mean(1)[0,:-1].reshape(80,80)        
        fac = np.sum(np.max(diffbeam-yfit)) / np.max(np.abs(z))
        z *= fac
        imshow(z, extent=(-4,4,4,-4), aspect='equal')
        clim(*cl); xlim(*xl); ylim(*xl)
        grid('on')
        ax = gca()
        text(-1,.95,'CPM fit param after ellip. deproj.')
        gca().set_yticklabels('')
        c = addcbar()
        c.set_ticks([-0.01,0,0.01])

        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)
        ax.text(.02, 0.5, 'Degrees', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(.5, 0.025, 'Degrees', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')
 
        savefig('figs/beam.pdf', bbox_inches='tight', pad_inches=0)
        
        return diffbeam, yfit, xx, yy
        

    def plotsl(self):

        close(1)
        figure(1, figsize=(5,9))

        ba = self.d.b['Ba'].item().sl*1.0
        bb = self.d.b['Bb'].item().sl*1.0
        bamb = self.d.b['Ba'].item().mb*1.0
        bbmb = self.d.b['Bb'].item().mb*1.0
        xx = self.d.b['Ba'].item().xxsl/60. # arcmin->deg
        yy = self.d.b['Ba'].item().xxsl/60. # arcmin->deg
        xxmb = self.d.b['Ba'].item().xx/60. # arcmin->deg
        yymb = self.d.b['Ba'].item().xx/60. # arcmin->deg

        # Norm, 0.3% power in sidelobe relative to main beam
        z = ba-bb
        pairsum = bamb+bbmb
        pairsum = pairsum/np.max(pairsum)

        fac1 = 0.003 * np.sum(pairsum) / np.sum(np.abs(z))
        fac2 = ((xxmb[0,1] - xxmb[0,0]) / (xx[0,1] - xx[0,0]))**2
        z *= fac1 * fac2

        subplots_adjust(hspace=0,wspace=0)

        # Diff beam
        subplot(3,2,1)
        cl = [-1.5e-5,1.5e-5]
        imshow(z, extent=(xx.min(),xx.max(),yy.max(),yy.min()), aspect='equal')
        grid('on')
        title('input to sim')
        gca().set_xticklabels('')
        clim(*cl)
        c=addcbar(sci=True)
        c.set_ticks([-1e-5,0,1e-5])

        subplot(3,2,3)
        zfit = self.d.mt.bcpm.mean(0)[0,:-1].reshape(201,201)
        fac = np.sum(np.max(z)) / np.sum(np.max(zfit))
        zfit *= fac
        imshow(zfit, extent=(-10,10,10,-10), aspect='equal')
        grid('on')
        title('T->P only')
        clim(*cl)
        gca().set_xticklabels('')

        subplot(3,2,4)
        zfit = self.d.ms.bcpm.mean(0)[0,:-1].reshape(201,201)
        zfit *= fac
        imshow(zfit, extent=(-10,10,10,-10), aspect='equal')
        grid('on')
        title('sig')
        clim(*cl)
        gca().set_xticklabels('')
        gca().set_yticklabels('')
        c=addcbar(sci=True)
        c.set_ticks([-1e-5,0,1e-5])

        cl = [-1e-4,1e-4]
        subplot(3,2,5)
        zfit = self.d.mn.bcpm.mean(0)[0,:-1].reshape(201,201)
        zfit *= fac
        imshow(zfit, extent=(-10,10,10,-10), aspect='equal')
        clim(*cl)
        grid('on')
        title('noi')
        gca().set_xticks(gca().get_xticks()[0:-1])

        subplot(3,2,6)
        zfit = self.d.msn.bcpm.mean(0)[0,:-1].reshape(201,201)
        zfit *= fac
        imshow(zfit, extent=(-10,10,10,-10), aspect='equal')
        clim(*cl)
        grid('on')
        title('sig+noi')
        gca().set_yticklabels('')
        c=addcbar(sci=True)
        c.set_ticks([-1e-4,0,1e-4])

        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)
        ax.text(0.01, 0.5, 'degrees', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(0.5, 0.07, 'degrees', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')
 

        savefig('figs/sl.pdf', bbox_inches='tight', pad_inches=0)
        

    def plotac(self):
        
        close(1)
        figure(1, figsize=(14,10))

        cl = np.array([-5.,5.])

        ra = self.d.acTR['ra']
        dec = self.d.acTR['dec']
        ext = (ra.min(), ra.max(), -dec.max(), -dec.min())

        asp = (ra.max()-ra.min())/(dec.max()-dec.min())

        subplots_adjust(wspace=0, hspace=0)

        subplot(2,3,1)
        imshow(self.d.acderiv['wcz'].sum(0), extent=ext, aspect=asp)
        text(-40,-25,'raw leakage')
        clim(*cl)
        gca().set_xticklabels('')

        subplot(2,3,2)
        imshow((self.d.acderiv['wcz']-self.d.acderiv['wczpred']).sum(0), extent=ext, aspect=asp)
        text(-40,-25,'+ellip. dp')
        clim(*cl)
        gca().set_xticklabels('')
        gca().set_yticklabels('')

        subplot(2,3,3)
        im=imshow((self.d.acTR['wcz']-self.d.acTR['wczpred']).sum(0), extent=ext, aspect=asp)
        text(-40,-25,'+arb. dp')
        clim(*cl)
        gca().set_xticklabels('')
        gca().set_yticklabels('')
        c = addcbar()
        c.set_ticks([-4,-2,0,2,4])

        cl = np.array([-.3,.3])
        subplot(2,3,4)
        imshow((self.d.acTR['wcz']-self.d.acTR['wczpred']).sum(0), extent=ext, aspect=asp)
        text(-40,-25,'+arb. dp (color zoom)')
        clim(*cl)
        gca().set_yticks(gca().get_yticks()[1:])

        subplot(2,3,5)
        imshow((self.d.acTR['wcz']-self.d.acTRnosl['wcz']).sum(0), extent=ext, aspect=asp)
        text(-40,-25,'actual sidelobe leakage')
        clim(*cl)
        gca().set_yticklabels('')

        subplot(2,3,6)
        im=imshow((self.d.acTR['wczpred_cpm']).sum(0), extent=ext, aspect=asp)
        text(-40,-25,'CPM best fit pred.')
        clim(*cl)
        gca().set_yticklabels('')
        c = addcbar()
        c.set_ticks([-.2,-0.1,0,0.1,.2])

        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)
        ax.text(.07, 0.5, 'Dec. (deg)', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(.51, 0.06, 'R.A. (deg)', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')
 


        savefig('figs/ac.pdf', bbox_inches='tight', pad_inches=0)
      
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
        savefig('figs/S4Nl.pdf', bbox_inches='tight', pad_inches=0)
    
    def plotweights(self):
        """Plot weight mask"""

        close(1)
        figure(1, figsize=(4,6))

        m = self.d.mt

        ext = (m.ra.min(), m.ra.max(), -m.dec.max(), -m.dec.min())
        asp = (m.ra.max()-m.ra.min())/(m.dec.max()-m.dec.min())

        hmap = hp.read_map('input_maps/s4_nhits_01_fsky03bk_n0512.fits')
        #hmap = hp.read_map('input_maps/s4_nhits_04c_n0512.fits')
        ws4 = hp.get_interp_val(hmap, m.ra, -m.dec, lonlat=True)
        ws4 /= np.nanmax(ws4)

        wpol = (m.Qw + m.Uw)/2.
        wpol /= np.nanmax(wpol)

        wpol[~np.isfinite(wpol)] = 0
        ws4[~np.isfinite(ws4)] = 0

        subplots_adjust(wspace=0, hspace=0)

        subplot(2,1,1)
        imshow(wpol, extent=ext, aspect=asp, cmap='hot')
        clim([0,1])
        gca().set_xticklabels('')
        ylabel('Dec. (deg)')
        text(-45,-25,'This work',color='white')

        subplot(2,1,2)
        imshow(ws4, extent=ext, aspect=asp, cmap='hot')
        clim([0,1])
        xlabel('R.A. (deg)')
        ylabel('Dec. (deg)')
        text(-45,-25,'S4 r-forecasting',color='white')
        gca().set_yticks(gca().get_yticks()[0:-1])

        savefig('figs/weight.pdf', bbox_inches='tight', pad_inches=0)

        print('weight ratio sim/s4 = {:f}'.format(np.sum(wpol)/np.sum(ws4)))

    
    def plotmaps(self):
        """Plot maps"""
        
        
        close(1)
        figure(1, figsize = (11,8))

        w = self.d.mt.Qw
        self.plotmapseq(self.d.mt, w, 0, (-.4,.4), (-.1,.1), 'leakage')
        self.plotmapseq(self.d.ms, w, 1, (-2,2), (-.1,.1), 'signal')
        self.plotmapseq(self.d.mn, w, 2, (-.2,.2), (-.1,.1), 'noise')

        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)
        ax.text(.02, 0.5, 'Dec. (deg)', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(.51, 0.03, 'R.A. (deg)', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')
 

        savefig('figs/maps.pdf', bbox_inches='tight', pad_inches=0)

    def plotmapseq(self, m, w, s, c1, c2, yl):

        ext = (m.ra.min(), m.ra.max(), -m.dec.max(), -m.dec.min())
        asp = (m.ra.max()-m.ra.min())/(m.dec.max()-m.dec.min())


        ys = 3
        cloc = 'top'
        cf = 8
        ty = 5

        subplot(ys,4,s*4+1)
        imshow(m.Q*w, extent=ext, aspect=asp)
        clim(*c1)
        ylabel(yl, rotation=0, fontsize=12)
        gca().set_yticks([-70,-20])
        if s < (ys-1):
            gca().set_xticklabels('')
        else:
            gca().set_xticks([-50,50])
        if s in [0,1,2]:
            c = addcbar(location=cloc, w=0.01)
            c.set_ticks([c1[0],0,c1[1]])
            c.ax.tick_params(labelsize=cf)
        if s==0:
            title('Q map', y=ty, fontsize=12)


        subplot(ys,4,s*4+2)
        imshow(m.Qpred*w, extent=ext, aspect=asp)
        clim(*c1)
        gca().set_yticklabels('')
        if s < (ys-1):
            gca().set_xticklabels('')
        else:
            gca().set_xticks([-50,50])
        if s in [0,1,2]:
            c = addcbar(location=cloc, w=0.01)
            c.set_ticks([c1[0],0,c1[1]])
            c.ax.tick_params(labelsize=cf)
        if s==0:
            title('model fit', y=ty, fontsize=12)

        subplot(ys,4,s*4+3)
        imshow((m.Q-m.Qpred)*w, extent=ext, aspect=asp)
        clim(*c2)
        gca().set_yticklabels('')
        if s < (ys-1):
            gca().set_xticklabels('')
        else:
            gca().set_xticks([-50,50])
        if s in [0,1,2]:
            c = addcbar(location=cloc, w=0.01)
            c.set_ticks([c2[0],0,c2[1]])
            c.ax.tick_params(labelsize=cf)
        if s==0:
            title('difference', y=ty, fontsize=12)

        subplot(ys,4,s*4+4)
        imshow(m.Qpred_cpm*w, extent=ext, aspect=asp)
        clim(*c2)
        gca().set_yticklabels('')
        if s < (ys-1):
            gca().set_xticklabels('')
        else:
            gca().set_xticks([-50,50])
        if s in [0,1,2]:
            c = addcbar(location=cloc, w=0.01)
            c.set_ticks([c2[0],0,c2[1]])
            c.ax.tick_params(labelsize=cf)        
        if s==0:
            title('CPM pred.', y=ty, fontsize=12)


    def plotaps(self):
        
        close(1)
        figure(1, figsize = (10,12))

        l = self.d.aTR['l']
        

        s = self.d.aTRnosl['s']
        n = self.d.aTRnosl['n']
        t = self.d.aTRnosl['t']
        snt = self.d.aTRnosl['snt']

        sdp = self.d.aTRdpnosl['s']
        ndp = self.d.aTRdpnosl['n']
        tdp = self.d.aTRdpnosl['t']
        sntdp = self.d.aTRdpnosl['snt']
        
        sd = self.d.aderivnosl['s']
        nd = self.d.aderivnosl['n']
        td = self.d.aderivnosl['t']
        sntd = self.d.aderivnosl['snt']

        sddp = self.d.aderivdpnosl['s']
        nddp = self.d.aderivdpnosl['n']
        tddp = self.d.aderivdpnosl['t']
        sntddp = self.d.aderivdpnosl['snt']
        

        cl,nm = readcambfits('spec/camb_planck2013_r0p1.fits')
        r0p1 = cl[2]
        lr = np.arange(r0p1.size)
        r = 0.003
        


        for k,spec in enumerate([3,1,2]):
            subplot(2,3,k+1)
            errorbar(l, sd.mean(0)[spec], sd.std(0,ddof=1)[spec], fmt='g-', label='signal')
            errorbar(l, nd.mean(0)[spec], nd.std(0,ddof=1)[spec], fmt='c-', label='noise')
            errorbar(l, td.mean(0)[spec], td.std(0,ddof=1)[spec], fmt='k-', lw=2, label='leakage')
            xlim(0,350)
            if spec != 3:
                plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
                gca().set_yscale('log')
                ylim(1e-7,10)
            if k==2:
                legend(loc='lower right')

            errorbar(l, sddp.mean(0)[spec], sddp.std(0,ddof=1)[spec], fmt='g--')
            errorbar(l, nddp.mean(0)[spec], nddp.std(0,ddof=1)[spec], fmt='c--')
            errorbar(l, tddp.mean(0)[spec], tddp.std(0,ddof=1)[spec], fmt='k--', lw=2)
            plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
            xlim(0,350)
            if spec != 3:
                plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
                gca().set_yscale('log')
                ylim(1e-7,10)
            else:
                ylim(-50,100)
            gca().set_xticks([0,100,200,300])

            if spec==1:
                title('EE')
            if spec==2:
                title('BB')
            if spec==3:
                title('TE')

            if k==0:
                ylabel('ellip. Gauss \n deproj.', rotation=0, labelpad=20)

            subplot(2,3,k+1+3)
            errorbar(l, s.mean(0)[spec], s.std(0,ddof=1)[spec], fmt='g-', label='signal')
            errorbar(l, n.mean(0)[spec], n.std(0,ddof=1)[spec], fmt='c-', label='noise')
            errorbar(l, t.mean(0)[spec], t.std(0,ddof=1)[spec], fmt='k-', label='leakage', lw=2)
            plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
            xlim(0,350)
            if spec != 3:
                plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
                gca().set_yscale('log')
                ylim(1e-7,10)

            errorbar(l, sdp.mean(0)[spec], sdp.std(0,ddof=1)[spec], fmt='g--')
            errorbar(l, ndp.mean(0)[spec], ndp.std(0,ddof=1)[spec], fmt='c--')
            errorbar(l, tdp.mean(0)[spec], tdp.std(0,ddof=1)[spec], fmt='k--', lw=2)
            plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
            xlim(0,350)
            if spec != 3:
                plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), ':k',label='r={:0.0e}'.format(r))
                gca().set_yscale('log')
                ylim(1e-7,10)
            else:
                ylim(-50,100)
            gca().set_xticks([0,100,200,300])

            if k==0:
                ylabel('arbitrary \n deproj.', rotation=0, labelpad=20)

        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)
        ax.text(.02, 0.5, r'$\ell(\ell+1)C_{\ell}/2\pi\ [\mu K^2]$', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(.51, 0.03, r'Multipole $\ell$', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')
 
        savefig('figs/aps.pdf', bbox_inches='tight', pad_inches=0)


    def plotcpm(self):
        
        close(1)
        figure(1, figsize = (10,10))

        l = self.d.aTR['l']

        a = self.d.aTRdp['snt2_pred']
        anosl = self.d.aTRdpnosl['snt2_pred']
        aderiv = self.d.aderivdp['snt2_pred']
        aderivnosl = self.d.aderivdpnosl['snt2_pred']

        at = self.d.aTRdp['t']
        atnosl = self.d.aTRdpnosl['t']
        atderiv = self.d.aderivdp['t']
        atderivnosl = self.d.aderivdpnosl['t']

        atpt = self.d.aTRdp['t_predt']
        atptnosl = self.d.aTRdpnosl['t_predt']
        atptderiv = self.d.aderivdp['t_predt']
        atptderivnosl = self.d.aderivdpnosl['t_predt']

        cl,nm = readcambfits('spec/camb_planck2013_r0p1.fits')
        r0p1 = cl[2]
        lr = np.arange(r0p1.size)
        r = 0.003

        subplots_adjust(wspace=0, hspace=0)

        xl = [0,350]

        subplot(2,2,1)
        plot(l, atderivnosl.mean(0)[2], 'ok', color='gray',zorder=-100)
        errorbar(l, aderivnosl.mean(0)[2,2], np.std(aderivnosl,0,ddof=1)[2,2],  fmt='.k')
        plot(l, atptderivnosl.mean(0)[2,2], 'xk')
        plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), '--k')
        ylim(-.0002,.003)
        xlim(*xl)
        plot([0,350],[0,0],':k')
        gca().set_yticks([0,.001,.002,.003])
        gca().set_xticklabels([''])
        text(20,.0029,'ellip. deproj.\n4$^{\\circ}$ template\nno sidelobes', verticalalignment='top')


        subplot(2,2,2)
        plot(l, atderiv.mean(0)[2], 'ok', color='gray',zorder=-100)
        errorbar(l, aderiv.mean(0)[2,2], np.std(aderivnosl,0,ddof=1)[2,2],  fmt='.k')
        plot(l, atptderiv.mean(0)[2,2], 'xk')
        plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), '--k')
        ylim(-.0002,.003)
        xlim(*xl)
        plot([0,350],[0,0],':k')
        gca().set_yticks([0,.001,.002,.003])
        gca().set_xticklabels([''])
        gca().set_yticklabels([''])
        text(20,.0029,'ellip. deproj.\n4$^{\\circ}$ template\nuniform 8$^{\\circ}$ sidelobe', verticalalignment='top')


        subplot(2,2,3)
        plot(l, atnosl.mean(0)[2], 'ok', label='post-deprojection residual',color='gray',zorder=-100)
        errorbar(l, anosl.mean(0)[2,2], np.std(a,0,ddof=1)[2,2],  fmt='.k',  label='prediction')
        plot(l, atptnosl.mean(0)[2,2], 'xk', label='prediction expectation')
        plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), '--k',label='r={:0.3f}'.format(r))
        ylim(-.0004,.0004)
        xlim(*xl)
        plot([0,350],[0,0],':k')
        gca().set_yticks([-.0003,0,.0003])
        legend(loc='lower left')
        text(20,.00038,'arbitrary deproj.\n10$^{\\circ}$ template\nno sidelobes', verticalalignment='top')
        gca().set_xticks([0,50,100,150,200,250,300])

        subplot(2,2,4)
        plot(l, at.mean(0)[2], 'ok', color='gray',zorder=-100)
        errorbar(l, a.mean(0)[2,2], np.std(a,0,ddof=1)[2,2],  fmt='.k')
        plot(l, atpt.mean(0)[2,2], 'xk')
        plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), '--k')
        ylim(-.0004,.0004)
        xlim(*xl)
        plot([0,350],[0,0],':k')
        gca().set_yticks([-.0003,0,.0003])
        legend(loc='lower left')
        gca().set_yticklabels([''])
        text(20,.00038,'arbitrary deproj.\n10$^{\\circ}$ template\nuniform 8$^{\\circ}$ sidelobe', verticalalignment='top')
        gca().set_xticks([0,50,100,150,200,250,300,350])

        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)

        ax.text(0.02, 0.5, r'$\ell(\ell+1)C_{\ell}/2\pi\ [\mu K^2]$', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(.51, 0.03, r'Multipole $\ell$', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')

        savefig('figs/cpm.pdf', bbox_inches='tight', pad_inches=0)


    def rhohist(self):

        close(1)
        figure(1, figsize = (4,8))


        l = self.d.aTR['l'][0:4]

        a = self.d.aTRdp['snt2_pred'][:,2,2,0:4]
        anosl = self.d.aTRdpnosl['snt2_pred'][:,2,2,0:4]
        aderiv = self.d.aderivdpnosl['snt2_pred'][:,2,2,0:4]

        at = self.d.aTRdp['t_predt'][:,2,2,0:4]
        atnosl = self.d.aTRdpnosl['t_predt'][:,2,2,0:4]
        atderiv = self.d.aderivdpnosl['t_predt'][:,2,2,0:4]


        cl,nm = readcambfits('spec/camb_planck2013_r0p1.fits')
        r0p1 = cl[2]
        lr = np.arange(r0p1.size)
        r0p1 = r0p1*lr*(lr+1)/(2*np.pi)
        w = np.interp(l, lr, r0p1)
        
        rfac = (w*w).sum() / w.sum()
        
        rho = (a*w).sum(1) / w.sum()  / rfac * 0.1
        rhonosl = (anosl*w).sum(1) / w.sum() / rfac * 0.1
        rhoderiv = (aderiv*w).sum(1) / w.sum() / rfac * 0.1
 
        rhot = (at*w).sum(1) / w.sum() / rfac * 0.1
        rhotnosl = (atnosl*w).sum(1) / w.sum() / rfac * 0.1
        rhotderiv = (atderiv*w).sum(1) / w.sum() / rfac * 0.1

        print('rho ellip. = {:0.1e} +/- {:0.1e}'.format(rhoderiv.mean(), rhoderiv.std(ddof=1)))
        print('rhoT ellip. = {:0.1e} +/- {:0.1e}'.format(rhotderiv.mean(), rhotderiv.std(ddof=1)))

        print('rho nosl = {:0.1e} +/- {:0.1e}'.format(rhonosl.mean(), rhonosl.std(ddof=1)))
        print('rhoT nosl = {:0.1e} +/- {:0.1e}'.format(rhotnosl.mean(), rhotnosl.std(ddof=1)))

        print('rho = {:0.1e} +/- {:0.1e}'.format(rho.mean(), rho.std(ddof=1)))
        print('rhoT = {:0.1e} +/- {:0.1e}'.format(rhot.mean(), rhot.std(ddof=1)))

        

        rng = [-3e-3, 6e-3]
        nb = 10
        xt = [-3e-3,0,3e-3,6e-3]

        subplots_adjust(wspace=0, hspace=0)

        subplot(3,1,1)
        hist((rhoderiv-rhotderiv)+rhotderiv.mean(), range=rng, bins=nb, color='gray')
        ax = gca()
        ax.set_xticks(xt)
        ax.set_xticklabels('')
        ax.set_ylim(0,ax.get_ylim()[1]+1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plot([0,0],ax.get_ylim(),':k')
        plot([rhotderiv.mean(),rhotderiv.mean()],ax.get_ylim(),'--k',lw=3)
        text(0.02,0.95,'ellip. dp.\nr<4$^{\\circ}$ temp.\nno SL',verticalalignment='top',horizontalalignment='left', transform=ax.transAxes)

        subplot(3,1,2)
        hist(rhonosl, range=rng, bins=nb, color='gray')
        ax = gca()
        ax.set_xticks(xt)
        ax.set_xticklabels('')
        ax.set_ylim(0,ax.get_ylim()[1]+1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(ax.get_yticks()[0:-1])
        plot([0,0],ax.get_ylim(),':k')
        plot([rhotnosl.mean(),rhotnosl.mean()],ax.get_ylim(),'--k',lw=3)
        text(0.02,0.95,'arb. dp.\n10$^{\\circ}$ temp.\nno SL',verticalalignment='top',horizontalalignment='left', transform=ax.transAxes)


        subplot(3,1,3)
        hist((rho-rhot)+rhot.mean(), range=rng, bins=nb, color='gray')
        ax = gca()
        ax.set_xticks(xt)
        ax.set_ylim(0,ax.get_ylim()[1]+1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(ax.get_yticks()[0:-1])
        plot([0,0],ax.get_ylim(),':k')
        plot([rhot.mean(),rhot.mean()],ax.get_ylim(),'--k',lw=3)
        text(0.02,0.95,'arb. dp.\n10$^{\\circ}$ temp.\n8$^{\\circ}$ SL',verticalalignment='top',horizontalalignment='left', transform=ax.transAxes)


        ax = gcf().add_axes( [.49, .49, .02, .02] )
        ax.set_axis_off()
        ax.set_xlim(.49, .51)
        ax.set_ylim(.49, .51)

        ax.text(0.01, 0.5, 'N', fontsize=16,
                rotation='vertical', horizontalalignment='left', verticalalignment='center')
        ax.text(.51, 0.03, r'$\rho$', fontsize=16,
                rotation='horizontal', horizontalalignment='center', verticalalignment='bottom')

        savefig('figs/rhohist.pdf', bbox_inches='tight', pad_inches=0)
