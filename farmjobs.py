import farmit
import numpy as np

i = np.arange(256)

x = np.linspace(-7, 7, 16)
y = np.linspace(-7, 7, 16)
xx, yy = np.meshgrid(x, y)

r = np.ravel(np.sqrt(xx**2+yy**2))
theta = np.ravel(np.arctan2(yy,xx)*180/np.pi)
f = farmit.farmit('runjobs.py', args={'theta':theta, 'r':r, 'i':i},
                  reqs={'N':4})
f.writejobfiles()
f.runjobs()

