import numpy as np

th = 7.5 # spin axis precession amplitude (degrees)
n = 1 # anti-clockwise motion around sun
w = 2*np.pi / 0.5 # pulsation (radians/year)
phi = 340 # phase (deg)

# Years 1,2 - surveys 1-4, ph = 340 deg
# Years 3,4 - surveys 5-8, ph = 250

# Time axis
t = np.arange(0,2,0.001) # years

# Ecliptic
lon =  th * np.sin( (-1)**n * w * t + phi*np.pi/180)
lat = -th * np.cos( (-1)**n * w * t + phi*np.pi/180)
