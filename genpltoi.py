from glob import glob
import numpy as np
from astropy.io import fits
import healpy as hp
from fast_histogram import histogram1d

# Load files
fnn = np.sort(glob('toi/HFI_TOI_*.fits'))

# We are going to histogram the TODs in each pixel of an healpix map
be = np.arange(0,360+1,5)
nbins = len(be) - 1

Nside = 1024
npix = hp.nside2npix(Nside)
#hmap = np.zeros((nbins,npix), dtype='float32')
# it crashed after writing nominal, so pick up from there
hmap = np.array(hp.read_map('toi/psi_hist_nominal.fits',field=range(nbins)))
fnn = fnn[466:]

for k,fn in enumerate(fnn):
    
    # Load data
    print('loading file {0} of {1}'.format(k+1,len(fnn)))
    x = fits.getdata(fn,2)

    # Let's just downsample to every 10th element for speed
    ds = 10
    th = x['THETA'][0::ds]
    ph = x['PHI'][0::ds]
    psi = x['PSI'][0::ds] * 180/np.pi

    # Get pixel for each TOD element
    pix = hp.ang2pix(Nside, th, ph)

    # Histogram Nhits on a per pixel basis
    upix = np.unique(pix)
    for j,pval in enumerate(upix):
        if np.mod(j,100) == 0:
            print ('histogramming pixel {0} of {1}...'.format(j,len(upix)-1))
        ind = np.where(pix == pval)
        #N,dum = np.histogram(psi[ind], be)
        N = histogram1d(psi[ind], range=[be[0],be[-1]], bins=len(be)-1)
        hmap[:,pval] += N

    # Save nominal mission if we're there. It's impossible to find OD
    # definitions of nominal vs full mission, but it's stated that the nominal
    # mission lasted "15.5 months" from beginning of survey 1 on OD91, so...
    OD = np.int(91 + 15.5 * 30)
    ind = fn.find('OD')
    OD0 = np.int(fn[(ind+2):(ind+6)])
    if OD == OD0:
        hp.write_map('toi/psi_hist_nominal.fits',hmap)

hp.write_map('toi/psi_hist_full.fits',hmap)
