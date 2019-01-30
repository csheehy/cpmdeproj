import farmit
import numpy as np

i = np.arange(256)

x = np.linspace(-14, 14, 16)
y = np.linspace(-14, 14, 16)
xx, yy = np.meshgrid(x, y)

r = np.ravel(np.sqrt(xx**2+yy**2))
theta = np.ravel(np.arctan2(yy,xx)*180/np.pi)

i = i[0::10]
r = r[0::10]
theta = theta[0::10]

#i = np.atleast_1d(i[0])
#r = np.atleast_1d(r[0])
#theta = np.atleast_1d(theta[0])

inputmap = ['camb_planck2013_r0_lensing_lensfix_A6p125_n1024_rxxxx.fits']
beamfile = np.array(['beams/beam_v3_uqpsl_0p3pct_{:04d}.npz'.format(j) for j in i])
tempNside = [1024]

sn = ['001']


Ttt = ['planck']
QUtt = ['s4']


#######################
#rlzarr = np.arange(10)
rlzarr = [1]

for rlz in rlzarr:
    f = farmit.farmit('runsimpairmap.py', 
                      args={'theta':theta, 'r':r, 'i':i, 'inputmap':inputmap, 
                            'beamfile':beamfile, 'sn':sn, 'rlz':[rlz],
                            'Ttt':Ttt, 'QUtt':QUtt, 'tempNside':tempNside}, 
                      reqs={'N':1, 'mode':'bynode','notgroup':'[gen3,gen4,gen5,gen7]'})

    f.writejobfiles()
    f.runjobs()
