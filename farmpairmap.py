import farmit
import numpy as np

i = np.arange(256)

sn = ['004']
rlz = [0]

f = farmit.farmit('runpairmap.py', 
                  args={'i':i, 'sn':sn, 'rlz':rlz},
                  reqs={'N':1,'mode':'bynode'})

f.writejobfiles()
f.runjobs()

