import beam
import map
import sim
import argparse
import numpy as np
import cPickle as cP
import os

def dosim(ba, bb, r, theta, dk, inputmap, rlz, sn, i, Ttt, QUtt, tempNside):

    s = sim.sim(ba, bb, r=r, theta=theta, dk=dk, inputmap=inputmap,
                rlz=rlz, sn=sn, i=i, tempNside=tempNside)

    s.runsim(sigtype='TnoP', Ttemptype=Ttt, QUtemptype=QUtt)
    s.runsim(sigtype='sig', Ttemptype=Ttt, QUtemptype=QUtt)
    s.runsim(sigtype='noi', Ttemptype=Ttt, QUtemptype=QUtt)
    s.runsim(sigtype='EnoB', Ttemptype=Ttt, QUtemptype=QUtt)
    s.runsim(sigtype='TnoPnosl', Ttemptype=Ttt, QUtemptype=QUtt)
    s.runsim(sigtype='signosl', Ttemptype=Ttt, QUtemptype=QUtt)
    s.runsim(sigtype='EnoBnosl', Ttemptype=Ttt, QUtemptype=QUtt)
    
    return s

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-r", dest="r", type=float, default=0)
parser.add_argument("--theta", dest="theta", type=float, default=0)
parser.add_argument("-i", dest="i", type=int, default=0)
parser.add_argument("--inputmap", dest="inputmap", type=str, default='camb_planck2013_r0_lensing_lensfix_A6p125_n1024_rxxxx.fits')
parser.add_argument("--beamfile", dest="beamfile", type=str, default='beams/beam_v3_uqpsl_0p3pct_0000.npz')
parser.add_argument("--sn", dest="sn", type=str, default='xxx')
parser.add_argument("--rlz", dest="rlz", type=int, default=0)
parser.add_argument("--Ttt", dest="Ttt", type=str, default='planck')
parser.add_argument("--QUtt", dest="QUtt", type=str, default='s4')
parser.add_argument("--tempNside", dest="tempNside", type=int, default=1024)
parser.add_argument("--dk", dest="dk", type=float, default=0)

o = parser.parse_args()

x = np.load(o.beamfile)
ba = x['Ba'].item()
bb = x['Bb'].item()
x.close()


#######
# Sim
dks = np.arange(0,360.,45)
for dk in dks:
    s = dosim(ba, bb, o.r, o.theta, dk, o.inputmap, o.rlz, o.sn, o.i, o.Ttt, o.QUtt, o.tempNside)


######
# Pairmaps

dpt      = ['deriv',      'TR1.2']
cpmdpt   = ['TR4.0+pol',  'TR10.0+pol']
cpmalpha = [1,             10]
dpdk     = ['perdk',      'alldk']
cpmdpdk  = ['alldk',      'alldk']
cpmtype  = ['lr',         'lr']


for t,ct,d,cd,c,a in zip(dpt,cpmdpt,dpdk,cpmdpdk,cpmtype,cpmalpha):

    for st in ['TnoP','sig','EnoB','noi', 'TnoPnosl','signosl','EnoBnosl']:

        if t == 'none':
            dext = ''
        else:
            dext = '_'+t+'++'+ct
            dext += '_alpha'+np.str(a)
            dext += '_cpm'+c
            dext += '_'+d+'++'+cd

        m = map.pairmap('{:s}/{:s}_r{:04d}_dk???_{:04d}.npz'.format(o.sn, st, o.rlz, o.i),
                        cpmalpha=a, dpt=t, cpmdpt=ct, dpdk=d, cpmdpdk=cd, cpmtype=c,
                        dext=dext)


######
# Delete template. Sad but these files are ginormous. Delete TnoP because
# missing TnoP file triggers template regeneration.
for st in ['TnoP', 'temp']:
    fnout = s.getfnout(st, o.i)
    if st=='temp':
        fnout = fnout.replace('npz','npy')
    fnout = fnout.replace('dk{:03d}'.format(np.int(dks[-1])), 'dk???')
    cmd = 'rm -f ' + fnout
    print('executing command "' + cmd + '"')
    os.system(cmd)

#####
# Delete saved design matrices. These are *really* huge.
fnn = [m.Xfn, m.Xcfn, m.Xsfn]
for fn in fnn:
    cmd = 'rm -f ' + fn
    print('executing command "' + cmd + '"')
    os.system(cmd)

