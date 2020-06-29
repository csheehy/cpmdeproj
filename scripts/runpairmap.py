import argparse
import numpy as np
import map

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", dest="i", type=int, default=0)
parser.add_argument("--sn", dest="sn", type=str, default='xxx')
parser.add_argument("--rlz", dest="rlz", type=int, default=0)

o = parser.parse_args()

#dpt      = ['deriv',  'TR1.2',  'deriv',  'deriv',       'deriv',  'deriv']
#cpmdpt   = ['TR1.2',  'TR10.0', 'TR10.0', 'TR10.0+pol',  'TR20.0', 'TR20.0+pol']
#cpmalpha = [0,        0,        0,        0,             0,        0]
#dpdk     = ['perdk',  'alldk',  'perdk',  'perdk',       'perdk',  'perdk']
#cpmdpdk  = ['alldk',  'alldk',  'alldk',  'alldk',       'alldk',  'alldk']

cpmalpha = [1]
N = len(cpmalpha)
cpmdpt   = ['TR9.0+pol']*N
dpt      = ['TR1.2']*N
dpdk     = ['alldk']*N
cpmdpdk  = ['alldk']*N
cpmtype = ['lr']*N

for t,ct,d,cd,c,a in zip(dpt,cpmdpt,dpdk,cpmdpdk,cpmtype,cpmalpha):

    for st in ['sig','TnoP','EnoB','noi']:

        if t == 'none':
            dext = ''
        else:
            dext = '_'+t+'++'+ct
            dext += '_alpha'+np.str(a)
            dext += '_cpm'+c
            dext += '_'+d+'++'+cd

        map.pairmap('{:s}/{:s}_r{:04d}_dk???_{:04d}.npy'.format(o.sn, st, o.rlz, o.i),
                    cpmalpha=a, dpt=t, cpmdpt=ct, dpdk=d, cpmdpdk=cd, cpmtype=c,
                    dext=dext)

