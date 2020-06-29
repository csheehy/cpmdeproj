# Smooth HFI maps to common resolution
import healpy as hp
import numpy as np

mapdir = 'planckmaps/real/'
fnn = ['HFI_SkyMap_{:03d}_2048_R2.02_full.fits'.format(f) for f in [100,143,217,353]]
fwhm_arr = [9.651, 7.248, 4.990, 4.818] # arcmin

reso = 10 # output fwhm arcmin
Nside = 1024 # output Nside

for fn,fwhm in zip(fnn,fwhm_arr):
    fnin = mapdir + fn
    hmap = hp.read_map(fnin,field=(0,1,2))
    alm = hp.map2alm(hmap)
    lmax = 3*2048-1

    bl_in  = hp.gauss_beam(fwhm = fwhm/60. * np.pi/180, lmax=lmax)
    bl_out = hp.gauss_beam(fwhm = reso/60. * np.pi/180, lmax=lmax)
    fl = bl_out/bl_in

    # No aliasing
    lmax_out = 3*Nside-1
    fl[lmax_out:] = 0

    # Smooth alms
    almsm = tuple([hp.almxfl(k, fl) for k in alm])

    # alm -> map
    hmapsm = hp.alm2map(almsm, Nside)
    
    # write to fits
    fnout = fnin.replace('2048','{:d}_fwhm10arcmin'.format(Nside))
    hp.write_map(fnout, hmapsm)
