import beam
import sim
import argparse
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-r", dest="r", type=float)
parser.add_argument("--theta", dest="theta", type=float)
parser.add_argument("-i", dest="i", type=int)
o = parser.parse_args()

inclpol = True

if not inclpol:
    ba = beam.beam()
    bb = beam.beam()
    prefix = 'TnoP_notempnoise'
else:
    x = np.load('pairmaps/TnoP_notempnoise_dk000_{:04d}.npy'.format(o.i)).item()
    ba = x.Ba
    bb = x.Bb
    prefix = 'TwithP_notempnoise'


s = sim.sim(ba, bb, r=o.r, theta=o.theta, dk=0.0)
s.runsim(inclpol=inclpol)
s.save(prefix+'_dk000', o.i)

s = sim.sim(ba, bb, r=o.r, theta=o.theta, dk=45.0)
s.runsim(inclpol=inclpol)
s.save(prefix+'_dk045', o.i)

