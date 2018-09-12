import beam
import sim
import argparse
import numpy as np
import cPickle as cP

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-r", dest="r", type=float, default=0)
parser.add_argument("--theta", dest="theta", type=float, default=0)
parser.add_argument("-i", dest="i", type=int, default=0)
parser.add_argument("--prefix", dest="prefix", type=str, default='test')
parser.add_argument("--inclpol", dest="inclpol", type=str2bool, default=False)
parser.add_argument("--inputmap", dest="inputmap", type=str, default='camb_planck2013_r0_lensing_lensfix_A6p125_n0256_r0000.fits')
parser.add_argument("--beamfile", dest="beamfile", type=str, default=None)

o = parser.parse_args()

if o.beamfile is not None:
    x = np.load(o.beamfile)
    ba = x['Ba'].item()
    bb = x['Bb'].item()
else:
    ba = beam.beam()
    bb = beam.beam()


s = sim.sim(ba, bb, r=o.r, theta=o.theta, dk=0.0, inputmap=o.inputmap)
s.runsim(inclpol=o.inclpol)
s.save(o.prefix+'_dk000', o.i)

s = sim.sim(ba, bb, r=o.r, theta=o.theta, dk=45.0, inputmap=o.inputmap)
s.runsim(inclpol=o.inclpol)
s.save(o.prefix+'_dk045', o.i)

s = sim.sim(ba, bb, r=o.r, theta=o.theta, dk=90.0, inputmap=o.inputmap)
s.runsim(inclpol=o.inclpol)
s.save(o.prefix+'_dk180', o.i)

s = sim.sim(ba, bb, r=o.r, theta=o.theta, dk=135.0, inputmap=o.inputmap)
s.runsim(inclpol=o.inclpol)
s.save(o.prefix+'_dk225', o.i)

