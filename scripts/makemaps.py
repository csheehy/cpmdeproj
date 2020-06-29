import map
import numpy as np


dpt      = ['deriv',      'TR1.2']
cpmdpt   = ['TR4.0+pol',  'TR10.0+pol']
cpmalpha = [1,             10]
dpdk     = ['perdk',      'alldk']
cpmdpdk  = ['alldk',      'alldk']
cpmtype  = ['lr',         'lr']

sn = '002'
rlzz = [6,7]

for rlz in rlzz:

    for t,ct,d,cd,c,a in zip(dpt,cpmdpt,dpdk,cpmdpdk,cpmtype,cpmalpha):

        for st in ['EnoB','TnoP','sig','noi','EnoBnosl','TnoPnosl','signosl']:

            if t == 'none':
                dext = ''
            else:
                dext = '_'+t+'++'+ct
                dext += '_alpha'+np.str(a)
                dext += '_cpm'+c
                dext += '_'+d+'++'+cd

            dir = sn + dext + '/'
            m = map.map(dir+st+'_r{:04d}_dkxxx_???0.npz'.format(rlz)) 
            m.save()

