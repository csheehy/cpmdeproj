import argparse
import numpy as np
import map

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", dest="i", type=int, default=0)
parser.add_argument("--sn", dest="sn", type=str, default='xxx')
parser.add_argument("--rlz", dest="rlz", type=int, default=0)

o = parser.parse_args()

tt =   ['deriv', 'TR2.0', 'TR2.0', 'TR2.0+pol', 'TR2.0+pol', 'TR6.0', 'TR6.0+pol', 'TR10.0', 'TR10.0+pol', 'TR20.0', 'TR20.0+pol', 'TR20.0+pol']
dpdk = ['perdk', 'perdk', 'alldk', 'perdk',     'perdk',     'perdk', 'perdk',     'perdk',  'perdk',      'perdk',  'perdk',       'alldk']
cpm  = ['lr',    'lr',    'lr',    'lr',        'perpix',    'lr',    'lr',        'lr',     'lr',         'lr',     'lr',         'lr']


alpha = [1,0]


for a in alpha:

    for st in ['sig','noi','TnoP','signoi']:    

        for t,d,c in zip(tt,dpdk,cpm):

            if t == 'none':
                dext = ''
            else:
                dext = '_'+t
                dext += '_alpha'+np.str(a)
                dext += '_cpm'+c
                dext += '_'+d

            map.pairmap('{:s}/{:s}_r{:04d}_dk???_{:04d}.npy'.format(o.sn, st, o.rlz, o.i),
                        cpmalpha=a, temptype=t, dext=dext, cpmtype=c, dpdk=d)

