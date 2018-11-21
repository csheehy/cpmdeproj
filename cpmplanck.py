import healpy as hp
import numpy as np
#from ridge import Ridge
from sklearn.linear_model import Ridge

doload = False

if doload:

    # Load all maps
    ff = np.array([100, 143, 217, 353])

    hmap = []
    for k,f in enumerate(ff):
        fn = 'planckmaps/real/HFI_SkyMap_{:03d}_1024_fwhm10arcmin_R2.02_full.fits'.format(f)
        hmap0 = hp.read_map(fn, field=(0,1,2))
        hmap.append(hmap0)
    hmap = np.array(hmap)
    del hmap0

    Ttemp = np.array(hp.read_map('toi/Ttemp_217.fits',field=arange(121)))
    
ffind = np.arange(len(ff))


Npix = hmap.shape[2]
Nside = hp.npix2nside(Npix)
pix = np.arange(Npix)
l,b = hp.pix2ang(Nside,pix,lonlat=True)

# Predict/fit this frequency index
p_freq = 2
f_freq = np.setxor1d(ffind,p_freq)

# Predict/fit these pixel indices
latcut = 0
f_pix = np.where(b>=latcut)[0]
p_pix = np.where(b<-latcut)[0]

# Get data
qu = 2
inclT = False
y = hmap[p_freq][qu, f_pix]
XPf = hmap[f_freq][:, qu, f_pix].T
if inclT:
    XTf = Ttemp[:,f_pix].T
    Xf = np.concatenate((XPf,XTf),axis=1)
else:
    Xf = XPf

# Fit
alpha = 0
r = Ridge(alpha=alpha)
r.fit(Xf, y)

# Predict
XPp = hmap[f_freq][:, qu, p_pix].T
if inclT:
    XTp = Ttemp[:,p_pix].T
    Xp = np.concatenate((XPp,XTp),axis=1)
else:
    Xp = XPp

# Initialize predicted map
hmappred = np.zeros(Npix)
hmappred[:] = np.nan
hmappred[p_pix] = r.predict(Xp)
hmappred[f_pix] = r.predict(Xf)


# Plot
mn = np.nanmin(hmappred)
mx = np.nanmax(hmappred)

close(1)
figure(1)
clf()
hp.mollview(hmappred,min=mn,max=mx,fig=1)

close(2)
figure(2)
clf()
hp.mollview(hmap[p_freq][qu,:],min=mn,max=mx,fig=2)

close(3)
figure(3)
clf()
hp.mollzoom(hmap[p_freq][qu,:] - hmappred,fig=3)

show()
