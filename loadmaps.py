import map
import aps
from matplotlib.pyplot import *
from sim import readcambfits
from copy import deepcopy as dc

doload = True
if doload:
    suff = '004_TR20.0+pol_alpha1_cpmlr_alldk'

    mt = map.map()
    ms = map.map()
    mn = map.map()
    msn = map.map()

    mt.load('maps/'+suff+'/TnoP_r0000_dkxxx.npz')
    ms.load('maps/'+suff+'/sig_r0000_dkxxx.npz')
    mn.load('maps/'+suff+'/noi_r0000_dkxxx.npz')
    msn.load('maps/'+suff+'/signoi_r0000_dkxxx.npz')
    #mtn = map.addmaps(mt,mn)

    mssub = dc(ms); mssub.Q -= ms.Qpol; mssub.U -= ms.Upol
    msnsub = dc(msn); msnsub.Q -= msn.Qpol; msnsub.U -= msn.Upol


att = aps.aps(mt)
a = aps.aps(msn,msnsub,ext='pred')
ac = aps.aps(msn,msnsub,ext='pred_cpm')

cl,nm = readcambfits('spec/camb_planck2013_r0p1.fits')
r0p1 = cl[2]
l = np.arange(r0p1.size)
r = 0.001

figure(2)
clf()
plot(att.l,att.dl[2],'.b',label='TnoP')
plot(a.l,a.dl[2][2],'-xb',label='sig x pred')
plot(a.l,ac.dl[2][2],'-xb',lw=2,label='sig x pred_cpm')

plot(l, l*(l+1)*r0p1*r/0.1/(2*np.pi), '--k',label='r={:0.0e}'.format(r))
plot([0,500],[0,0],':k')
xlim([0,500])
legend()
