import map
import aps
import healpy as hp
from matplotlib.pyplot import *
from sim import readcambfits
from copy import deepcopy as dc

doload = True
if doload:

    suff = '001_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk'

    mt = map.map()
    ms = map.map()
    mn = map.map()
    me = map.map()

    rlz = 1

    mt.load('maps/'+suff+'/TnoP_r{:04d}_dkxxx.npz'.format(rlz))
    ms.load('maps/'+suff +'/sig_r{:04d}_dkxxx.npz'.format(rlz))
    mn.load('maps/'+suff +'/noi_r{:04d}_dkxxx.npz'.format(rlz))
    me.load('maps/'+suff+'/EnoB_r{:04d}_dkxxx.npz'.format(rlz))

    for k in ['Q','U','Qpred','Upred','b','bcpm','Qpred_cpm','Upred_cpm']:
        setattr(mn,k,getattr(mn,k)/np.sqrt(10))

    mt.deproj()
    ms.deproj()
    mn.deproj()
    me.deproj()

    #hmap = hp.read_map('input_maps/camb_planck2013_r0_lensing_lensfix_A6p125_n1024_r0000.fits',field=(0,1,2))
    #hmap = hp.smoothing(hmap,fwhm=30.0/60*np.pi/180)
    #ms2 = dc(ms)
    #ms2.Q  -= hp.get_interp_val(hmap[1], ms.ra, ms.dec, lonlat=True)
    #ms2.U  -= hp.get_interp_val(hmap[2], ms.ra, ms.dec, lonlat=True)

    #hmap = hp.read_map('input_maps/camb_planck2013_EnoB_r0_lensing_lensfix_A6p125_n1024_r0000.fits',field=(0,1,2))
    #me2 = dc(me)
    #me2.Q  = hp.get_interp_val(hmap[1], ms.ra, ms.dec, lonlat=True)
    #me2.U  = hp.get_interp_val(hmap[2], ms.ra, ms.dec, lonlat=True)

    msn = map.addmaps(ms,mn)
    mtn = map.addmaps(mt,mn)
    men = map.addmaps(me,mn)

    ms2 = dc(ms)
    ms2.Q -= msn.Qpol
    ms2.U -= msn.Upol

    ms2.Q -= mt.Q
    ms2.U -= mt.U

    msn2 = dc(msn)
    msn2.Q -= msn.Qpol
    msn2.U -= msn.Upol

    me2 = dc(me)
    me2.Q -= me.Qpol
    me2.U -= me.Upol

    #msn2.Q -= mt.Q
    #msn2.U -= mt.U

    msnlat = dc(msn)
    msnlat.Q = msn.Qpol
    msnlat.U = msn.Upol
    
    #hmap = hp.read_map('input_maps/S4_noise_map_r{:04d}.fits'.format(rlz),field=(0,1,2))
    #hmap = hp.smoothing(hmap,fwhm=30.0/60*np.pi/180)
    #mnlat = dc(msn)
    #mnlat.Q = hp.get_interp_val(hmap[1], ms.ra, ms.dec, lonlat=True)
    #mnlat.U = hp.get_interp_val(hmap[2], ms.ra, ms.dec, lonlat=True)
    
    #mtn.Q -= mnlat.Q
    #mtn.U -= mnlat.U

att = aps.aps(mt)

a = aps.aps(msn2,msn,ext2='pred_cpm')
#a = aps.aps(mtn,msn,ext2='pred_cpm')
aa = aps.aps(mt,mt,ext2='pred_cpm')

a2 = aps.aps(msn,msn,ext2='pred_cpm',mb=me)
asat = aps.aps(msn, mb=me)
alat = aps.aps(msnlat, mb=me)

att2 = aps.aps(msn, mt, mb=me)
att3 = aps.aps(mt, msn2)

cl,nm = readcambfits('spec/camb_planck2013_r0p1.fits')
r0p1 = cl[2]
l = np.arange(r0p1.size)
r = 0.003


clf()
subplot(2,1,1)
plot(a.l,alat.dl[2],'-xy',label='<LAT>^2')
plot(a.l,asat.dl[2],'-xc',label='<SAT>^2')
plot(a.l,a.dl[0][2],'-xb',label='<SAT-LAT>^2')
plot(att.l,att.dl[2],'ok',label='<TnoP>^2')
plot(a.l,a.dl[1][2],'-xr',label='<pred>^2')
plot(att2.l,att2.dl[2][2],'-+',color='khaki',label='<SAT>x<TnoP>')
plot(att3.l,att3.dl[2][2],'-+y',label='<SAT-LAT>x<TnoP>')
plot(a.l,a2.dl[2][2],'-x',color='gray',label='<SAT>x<pred>')
plot(a.l,a.dl[2][2],'-xk',lw=2,label='<SAT-LAT>x<pred>')
plot(a.l,aa.dl[2][2],':k+',lw=1,label='<TnoP>x<pred_Tonly>')
plot(l, l*(l+1)*r0p1*r/0.1/(2*np.pi), '--k',label='r={:0.0e}'.format(r))
plot([0,500],[0,0],':k')
xlim([0,500])
legend()
xlabel('multipole')
ylabel('Dl^BB uK^2')

subplot(2,1,2)
plot(a.l,a.dl[1][2],'-xr',label='<pred>^2')
plot(att.l,att.dl[2],'ok',label='TnoP')
plot(att2.l,att2.dl[2][2],'-+',color='khaki',label='<SAT>x<TnoP>')
plot(att3.l,att3.dl[2][2],'-+y',label='<SAT-LAT>x<TnoP>')
plot(a.l,a2.dl[2][2],'-x',color='gray',label='<SAT>x<pred>')
plot(a.l,a.dl[2][2],'-xk',lw=2,label='<SAT-LAT>x<pred>')
plot(a.l,aa.dl[2][2],':k+',lw=1,label='<TnoP>x<pred_Tonly>')
plot(l, l*(l+1)*r0p1*r/0.1/(2*np.pi), '--k',label='r={:0.0e}'.format(r))

plot([0,500],[0,0],':k')
xlim([0,500])
xlabel('multipole')
ylabel('Dl^BB uK^2')
ylim(-2e-4,5e-4)
