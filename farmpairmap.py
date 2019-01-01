import farmit
import numpy as np
import time

#i = np.arange(256)
i = np.arange(0,256,10)
#i = [0]

sn = ['006']
rlz = [0]

f = farmit.farmit('runpairmap.py', 
                  args={'i':i, 'sn':sn, 'rlz':rlz},
                  #reqs={'N':1, 'mode':'bynode','notgroup':'[gen3,gen7]'})
                  reqs={'N':1, 'mode':'bynode'})

f.writejobfiles()
f.runjobs()

