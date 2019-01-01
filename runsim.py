import beam
import map
import sim
import argparse
import numpy as np
import cPickle as cP

def dosim(ba, bb, r, theta, dk, inputmap, rlz, sn, i, Ttt, QUtt):

    s = sim.sim(ba, bb, r=r, theta=theta, dk=dk, inputmap=inputmap,
                rlz=rlz, sn=sn)

    s.runsim(sigtype='TnoP', Ttemptype=Ttt, QUtemptype=QUtt)
    fnt = s.save(i)
    s.runsim(sigtype='sig', Ttemptype=Ttt, QUtemptype=QUtt)
    fns = s.save(i)
    s.runsim(sigtype='noi', Ttemptype=Ttt, QUtemptype=QUtt)
    fnn = s.save(i)
    s.runsim(sigtype='EnoB', Ttemptype=Ttt, QUtemptype=QUtt)
    fnn = s.save(i)

    ## Add s+n
    #xs = np.load(fns).item()
    #xn = np.load(fnn).item()
    #for k in ['siga','sigb','pairsum','pairdiff']:
    #    xs[k] += xn[k]
    #fns = fns.replace('sig' ,'signoi')
    #np.save(fns, xs)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-r", dest="r", type=float, default=0)
parser.add_argument("--theta", dest="theta", type=float, default=0)
parser.add_argument("-i", dest="i", type=int, default=0)
parser.add_argument("--inputmap", dest="inputmap", type=str, default='camb_planck2013_r0_lensing_lensfix_A6p125_n1024_rxxxx.fits')
parser.add_argument("--beamfile", dest="beamfile", type=str, default='beams/beam_v2_0000.npz')
parser.add_argument("--rlz", dest="rlz", type=int, default=0)
parser.add_argument("--sn", dest="sn", type=str, default='xxx')
parser.add_argument("--Ttt", dest="Ttt", type=str, default='planck')
parser.add_argument("--QUtt", dest="QUtt", type=str, default='s4')


o = parser.parse_args()

if o.beamfile is not None:
    x = np.load(o.beamfile)
    ba = x['Ba'].item()
    bb = x['Bb'].item()
    x.close()
    
    # Substitue gaussian beam
    #ba.mb = ba.g/ba.g.sum()
    #bb.mb = bb.g/bb.g.sum()

    # Substitute delta function for pre-smoothed map (very inefficient)
    #ba.mb = 1.0
    #bb.mb = 1.0
    #ba.rr = 0.0; ba.phi = 0.0
    #bb.rr = 0.0; bb.phi = 0.0

else:
    ba = beam.beam()
    bb = beam.beam()

dks = np.arange(0,360.,45)
for dk in dks:
    dosim(ba, bb, o.r, o.theta, dk, o.inputmap, o.rlz, o.sn, o.i, o.Ttt, o.QUtt)

