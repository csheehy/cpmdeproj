import map
import numpy as np


#cpmalpha = [10]
#N = len(cpmalpha)
#cpmdpt   = ['TR10.0+pol']*N
#dpt      = ['TR1.2']*N
#dpdk     = ['alldk']*N
#cpmdpdk  = ['alldk']*N
#cpmtype = ['lr']*N

cpmalpha = [1]
N = len(cpmalpha)
cpmdpt   = ['TR4.0']*N
dpt      = ['deriv']*N
dpdk     = ['perdk']*N
cpmdpdk  = ['alldk']*N
cpmtype = ['lr']*N

for t,ct,d,cd,c,a in zip(dpt,cpmdpt,dpdk,cpmdpdk,cpmtype,cpmalpha):

    for st in ['EnoB','TnoP','sig','noi']:

        if t == 'none':
            dext = ''
        else:
            dext = '_'+t+'++'+ct
            dext += '_alpha'+np.str(a)
            dext += '_cpm'+c
            dext += '_'+d+'++'+cd
        
        dir = '001' + dext + '/'
        #m = map.map(dir+st+'_*.npz')
        m = map.map(dir+st+'_r0001_dkxxx_???0.npz') # Every tenth detector
        #m = map.map(dir+st+'_r0000_dkxxx_0{0??,1[0-3]?}.npz')
        m.save()

