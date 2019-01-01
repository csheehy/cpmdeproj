import map
import numpy as np

#dpt      = ['deriv',  'TR1.2',  'deriv',  'deriv',       'deriv',  'deriv']
#cpmdpt   = ['TR1.2',  'TR10.0', 'TR10.0', 'TR10.0+pol',  'TR20.0', 'TR20.0+pol']
#cpmalpha = [0,        0,        0,        0,             0,        0]
#dpdk     = ['perdk',  'alldk',  'perdk',  'perdk',       'perdk',  'perdk']
#cpmdpdk  = ['alldk',  'alldk',  'alldk',  'alldk',       'alldk',  'alldk']
#cpmtype = ['lr']*len(dpt)

cpmalpha = [1]
N = len(cpmalpha)
cpmdpt   = ['TR10.0+pol']*N
dpt      = ['TR1.2']*N
dpdk     = ['alldk']*N
cpmdpdk  = ['alldk']*N
cpmtype = ['lr']*N

for t,ct,d,cd,c,a in zip(dpt,cpmdpt,dpdk,cpmdpdk,cpmtype,cpmalpha):

    for st in ['EnoB','TnoP','sig','noi']:
    #for st in ['TnoP']:

        if t == 'none':
            dext = ''
        else:
            dext = '_'+t+'++'+ct
            dext += '_alpha'+np.str(a)
            dext += '_cpm'+c
            dext += '_'+d+'++'+cd
        
        dir = '006' + dext + '/'
        #m = map.map(dir+st+'_*.npz')
        m = map.map(dir+st+'*_???0.npz') # Every tenth detector
        m.save()

