import healpy as hp
import numpy as np

# Add SPT noise, chi-by-eye to Henning SPT-500deg^2 paper N_l and
# functional form in  http://users.physics.harvard.edu/~buza/20161220_chkS4/

#sigmap = 9.0 # uK-arcmin, SPT
sigmap = 1.2 # uK-arcmin, CMB-S4

lknee = 300.
lexp = -1.8

l = np.arange(8000)*1.0
Nl = 4*np.pi / (41253.*60**2) * (1+(l/lknee)**(lexp)) * sigmap**2
Nl[0] = 0

# Get noise realization
Nside = 256
hmapTn = hp.synfast(Nl, Nside, new=True, verbose=False)
hmapQn = hp.synfast(Nl, Nside, new=True, verbose=False)
hmapUn = hp.synfast(Nl, Nside, new=True, verbose=False)

hp.write_map('input_maps/S4_noise_map.fits',[hmapTn,hmapQn,hmapUn])


