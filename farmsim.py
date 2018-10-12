import farmit
import numpy as np

i = np.arange(256)

x = np.linspace(-7, 7, 16)
y = np.linspace(-7, 7, 16)
xx, yy = np.meshgrid(x, y)

r = np.ravel(np.sqrt(xx**2+yy**2))
theta = np.ravel(np.arctan2(yy,xx)*180/np.pi)

inputmap = ['camb_planck2013_r0_lensing_lensfix_n0256_r0000.fits']
beamfile = np.array(['beams/beam_{:04d}.npz'.format(j) for j in i])
#beamfile = ['beams/beam_udp_0000.npz']
sn = ['004']
rlz = [0]
Ttt = ['noiseless']
QUtt = [None]

#######################
f = farmit.farmit('runsim.py', 
                  args={'theta':theta, 'r':r, 'i':i, 'inputmap':inputmap, 
                        'beamfile':beamfile, 'rlz':rlz, 'sn':sn,
                        'Ttt':Ttt, 'QUtt':QUtt}, 
                  reqs={'N':2})

f.writejobfiles()
f.runjobs()

