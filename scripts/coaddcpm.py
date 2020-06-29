import sim
import numpy as np
from scipy import sparse


loaddata = True
if loaddata:
    x = []
    for k in range(50):
        print(k)                         
        x.append(np.load('simdata/TnoP_noiseless_{:04d}.npy'.format(k)).item())

Npair = len(x)

ra = np.unique(x[0].mapra)
dec = np.unique(x[0].mapdec)
z=np.ones((Npair,dec.size,ra.size))*np.nan

for k in range(Npair):
    y = np.ones_like(x[k].mapra)*np.nan
    y[x[k].mapind] = np.ravel(x[k].pairdiff)
    z[k] = y

w = np.isfinite(z).astype(float)

Npixmap = x[0].mapra.size
Npixtemp = x[0].X.shape[1]
X = sparse.lil_matrix((Npixmap, Npixtemp*Npair))

ravec = np.ravel(x[0].mapra)
decvec = np.ravel(x[0].mapdec)

for k in range(Npixmap):

    print('{0} of {1}'.format(k,Npixmap))
    
    ra0 = ravec[k]
    dec0 = decvec[k]

    indra = np.where(ra==ra0)[0][0]
    inddec = np.where(dec==dec0)[0][0]
    wvec = w[:,inddec,indra]*1.0
    wsum = wvec.sum()
    if wsum > 0:
        wvec /= wvec.sum()

    for j,val in enumerate(wvec):
        if val==0:
            continue
        s = Npixtemp*j
        e = Npixtemp*(j+1)
        idx = np.where((np.ravel(x[j].ra==ra0)) & (np.ravel(x[j].dec==dec0)))[0]
        X[k,s:e] = val * x[j].X[idx, :]
        
y = np.ravel(np.nansum(z*w,0)/np.nansum(w,0))




