import map
import fit
import aps
import numpy as np
from copy import deepcopy as dc
from matplotlib.pyplot import *
ion()

loaddata = False
dofilter = False

if loaddata:
    
    m1 = map.map('BnoE_dust_notempnoise_r0000_dk000+045_???[0,5].npz')
    m2 = map.map('BnoE_dust_notempnoise_r0000_dk180+225_???[0,5].npz')
    m1noT = map.map('BnoEnoT_dust_notempnoise_r0000_dk000+045_???[0,5].npz')
    m2noT = map.map('BnoEnoT_dust_notempnoise_r0000_dk180+225_???[0,5].npz')

    #m1 = map.map('maps/BnoE_dust_notempnoise_r0000_dk000+045.npz')
    #m2 = map.map('maps/BnoE_dust_notempnoise_r0000_dk180+225.npz')
    #m1noT = map.map('maps/BnoEnoT_dust_notempnoise_r0000_dk000+045.npz')
    #m2noT = map.map('maps/BnoEnoT_dust_notempnoise_r0000_dk180+225.npz')

    # Add noise
    m1n = dc(m1); m1n.addmapnoise()
    m2n = dc(m2); m2n.addmapnoise()

    m1noTn = dc(m1noT); m1noTn.addmapnoise()
    m2noTn = dc(m2noT); m2noTn.addmapnoise()

    if dofilter:
        # Filter maps
        lr = [0,200]
        m1.filter(lr)
        m2.filter(lr)
        m1noT.filter(lr)
        m2noT.filter(lr)

        m1n.filter(lr)
        m2n.filter(lr)
        m1noTn.filter(lr)
        m2noTn.filter(lr)


doconstruct = False

if doconstruct:
    #f1T0 = fit.fit(m1noT, 'T')
    f1Q0 = fit.fit(m1noT, 'Q')
    f1U0 = fit.fit(m1noT, 'U')
    f2Q0 = fit.fit(m2noT, 'Q')
    f2U0 = fit.fit(m2noT, 'U')


f1T = dc(f1T0)
f1Q = dc(f1Q0)
f1U = dc(f1U0)
f2Q = dc(f2Q0)
f2U = dc(f2U0)


domapdiff = False

if domapdiff:
    ind = np.intersect1d(f1Q.fitind, f2Q.fitind)
    ind1 = np.in1d(f1Q.fitind, ind)
    ind2 = np.in1d(f2Q.fitind, ind)
    f1Q.y = f1Q.y[ind1] - f2Q.y[ind2]
    f1U.y = f1U.y[ind1] - f2U.y[ind2]
    f1Q.X = f1Q.X[ind1,:] - f2Q.X[ind2,:]
    f1U.X = f1U.X[ind1,:] - f2U.X[ind2,:]
    f1U.w = 1/(1/f1U.w[ind1] + 1/f2U.w[ind2])
    f1Q.w = 1/(1/f1Q.w[ind1] + 1/f2Q.w[ind2])

    f1Q.fitind = ind; f1U.fitind = ind; f2Q.fitind = ind; f2U.fitind = ind

doregress = True

if doregress:
    
    m1pred = dc(m1)

    print('T regress')
    #f1T.regress(cpmalpha = 1e-3)
    #m1pred.T = f1T.zpred

    #lam = dc(f1T.b)
    #for n in range(len(m1.fn)):
    #    v = lam[289*n:(289*(n+1))]
    #    lam[289*n:(289*(n+1))] = v/v.max() 
    #lam[lam<.1] = .1

    xs = np.arange(-2, 2.01, 0.25)
    xx,yy = np.meshgrid(xs,xs)
    r = np.sqrt(xx**2 + yy**2)
    fwhm = 30.0 # arcmin
    sig = (fwhm/60.) / 2.3548
    g = np.exp(-r**2/(2*sig**2))

    lam = np.tile(np.ravel(g), f1T.Npair)

    print('Q regress')
    f1Q.regress(cpmalpha=1e-6)
    m1pred.Q = f1Q.zpred

    print('U regress')
    f1U.regress(cpmalpha=1e-6)
    m1pred.U = f1U.zpred

    m2pred = dc(m2)
    f2Q.regress(b=f1Q.b)
    m2pred.Q = f2Q.zpred
    f2U.regress(b=f1U.b)
    m2pred.U = f2U.zpred



dospec = True

if dospec:
    a11a = aps.aps(m1noT, m1pred)
    a22a = aps.aps(m2noT, m2pred)

    a11b = aps.aps(m1, m1pred)
    a22b = aps.aps(m2, m2pred)

    a11c = aps.aps(m1, m1noT)
    a22c = aps.aps(m2, m2noT)


doplot = True

if doplot:

    clf();
    
    l = a11a.l
    
    s = 2; # BB

    subplot(2,1,1)

    plot(l, a11c.dl[1][s], 'sc', label='temp1 x temp1')
    plot(l, a11b.dl[0][s], '.', label='real1 x real1')
    plot(l, a11b.dl[1][s], '.', label='pred1 x pred1')
    plot(l, a11c.dl[2][s], '.', label='real1 x temp1')
    plot(l, a11b.dl[2][s], 'xk', mew=2, label='real1 x pred1')
    plot(l, a11a.dl[2][s], 'xr', label='temp1 x pred1')

    gca().set_yscale('log')
    
    legend()

    
    subplot(2,1,2)

    plot(l, a22c.dl[1][s], 'sc', label='temp2 x temp2')
    plot(l, a22b.dl[0][s], '.', label='real2 x real2')
    plot(l, a22b.dl[1][s], '.', label='pred2 x pred2')
    plot(l, a22c.dl[2][s], '.', label='real2 x temp2')
    plot(l, a22b.dl[2][s], 'xk', mew=2, label='real2 x pred2')
    plot(l, a22a.dl[2][s], 'xr', label='temp2 x pred2')

    gca().set_yscale('log')
    
    legend()

    
