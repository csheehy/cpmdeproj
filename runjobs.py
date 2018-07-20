import beam
import sim
import argparse
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-r", dest="r", type=float)
parser.add_argument("--theta", dest="theta", type=float)
parser.add_argument("-i", dest="i", type=int)
o = parser.parse_args()

#ba = beam.beam()
#bb = beam.beam()
x = np.load('simdata/TnoP_noiseless_{:04d}.npy'.format(o.i)).item()
ba = x.Ba
bb = x.Bb


s = sim.sim(ba, bb, r=o.r, theta=o.theta)
s.runsim(inclpol=True)
s.save('TwithP_noiseless', o.i)

