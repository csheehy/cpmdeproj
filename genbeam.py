import beam
import numpy as np

for k in range(256):

    ba = beam.beam()
    ba.getsl(0.003)

    bb = beam.beam()
    bb.getsl(0.003)
    bb.sl[:] = 0.0 # Set to zero
    
    fn = 'beams/beam_v3_uqpsl_0p3pct_{:04d}.npz'.format(k)
    np.savez(fn, Ba=ba, Bb=bb)
