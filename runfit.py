import map
import fit
import numpy as np
from copy import deepcopy as dc

loaddata = False

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

dofit = True

if dofit:
    
    m1pred = dc(m1)

    f = fit.fit(m1, 'T')
    f.regress(cpmalpha = 1e5)
    lam = dc(f.b)
    lam[lam<100] = 100

    m1pred.T = f.zpred
    f = fit.fit(m1, 'Q')
    f.regress(cpmalpha=(1/lam)*1e10)
    m1pred.Q = f.zpred
    f = fit.fit(m1, 'U')
    f.regress(cpmalpha=(1/lam)*1e10)
    m1pred.U = f.zpred


    

