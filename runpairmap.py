import argparse
import numpy as np
import map

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", dest="i", type=int, default=0)
parser.add_argument("--sn", dest="sn", type=str, default='xxx')
parser.add_argument("--rlz", dest="rlz", type=int, default=0)
parser.add_argument("--cpmalpha", dest="cpmalpha", type=float, default=1)
parser.add_argument("--cpmalphat", dest="cpmalphat", type=float, default=1)

o = parser.parse_args()

map.pairmap('{:s}/TnoP_r{:04d}_dk???_{:04d}.npy'.format(o.sn, o.rlz, o.i), cpmalpha=o.cpmalpha, cpmalphat=o.cpmalphat)
map.pairmap('{:s}/sig_r{:04d}_dk???_{:04d}.npy'.format(o.sn, o.rlz, o.i), cpmalpha=o.cpmalpha, cpmalphat=o.cpmalphat)
map.pairmap('{:s}/noi_r{:04d}_dk???_{:04d}.npy'.format(o.sn, o.rlz, o.i), cpmalpha=o.cpmalpha, cpmalphat=o.cpmalphat)
