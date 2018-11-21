import healpy as hp
import sim

# We are going to histogram the TODs in each pixel of an healpix map
be = np.arange(0,360+1,5)
nbins = len(be) - 1
bc = (be[0:-1] + be[1:])/2.

doload = True
if doload:
    Nside = 1024
    m = np.array(hp.ud_grade(hp.read_map('toi/psi_hist_nominal.fits',field=range(nbins)),Nside))
    T = np.array(hp.ud_grade(hp.read_map('planckmaps/real/HFI_SkyMap_217_2048_R2.02_full.fits'),Nside))

reso = hp.nside2resol(Nside, arcmin=True) / 60.
xsize = reso * 10

# test proj
gp = np.ravel(sim.gnomproj(T, 0, 0, xsize, xsize, reso))
ntemp = gp.size

# Initialize
Npix = T.size
Ttemp = np.zeros((gp.size,Npix))
pix = np.arange(Npix)
l,b = hp.pix2ang(Nside, pix, lonlat=True)

for k in range(Npix):

    if np.mod(k,1000) == 0:
        print('pix {:d} of {:d}'.format(k,Npix))

    gp = np.zeros(ntemp)
    for j,rot in enumerate(bc):
        Nobs = m[j,k]
        if Nobs > 0:
            gp += Nobs*np.ravel(sim.gnomproj(T,l[k],b[k],xsize,xsize,reso,rot=rot))
    Ttemp[:,k] = gp / m[:,k].sum()

Ttemp[~np.isfinite(Ttemp)] = 0
hp.write_map('toi/Ttemp_217.fits',Ttemp)




