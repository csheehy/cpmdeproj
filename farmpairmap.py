import farmit
import numpy as np

i = np.arange(256)

sn = ['000']
rlz = [0]
cpmalpha = [1.0]
cpmalphat = [1.0]


#######################
f = farmit.farmit('runpairmap.py', 
                  args={'i':i, 'sn':sn, 'rlz':rlz, 'cpmalpha':cpmalpha, 'cpmalphat':cpmalphat},
                  reqs={'N':1,'mode':'bynode'})

f.writejobfiles()
f.runjobs()

