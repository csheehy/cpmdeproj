import beam
import numpy as np

for k in range(256):
    ba = beam.beam()
    #ba.getsl()
    bb = beam.beam()
    #bb.getsl()
    #bb.sl[:] = 0.0 # Set to zero
    
    fn = 'beams/beam_v2_{:04d}.npz'.format(k)
    np.savez(fn, Ba=ba, Bb=bb)
