import aps
import map
from copy import deepcopy as dc
from matplotlib.pyplot import *

def plotaps(ms0, mn0, mt0, msn0):
    
    # Copy
    ms = dc(ms0)
    mn = dc(mn0)
    mt = dc(mt0)
    msn = dc(msn0)

    # S+N
    #msn = map.addmaps(ms,mn)

    # Get undeproj aps
    ass = aps.aps(ms)
    an = aps.aps(mn)
    asn = aps.aps(msn)
    at = aps.aps(mt)
    
    # Now deproject
    ms.deproj()
    mn.deproj()
    mt.deproj()
    msn.deproj()
    
    # Get post deproj aps
    assdp = aps.aps(ms)
    andp = aps.aps(mn)
    asndp = aps.aps(msn)
    atdp = aps.aps(mt)

    # Now plot
    clf()
    for k in [1,2]:
        
        l = ass.l

        subplot(1,2,k)
        plot(l, asn.dl[k], '.-r', label='s+n')
        plot(l, ass.dl[k], '.-b', label='sig')
        plot(l, an.dl[k], '.-c', label='noi')
        plot(l, at.dl[k], '.-g', label='TnoP')

        plot(l, asndp.dl[k], '+--r')
        plot(l, assdp.dl[k], '+--b')
        plot(l, andp.dl[k], '+--c')
        plot(l, atdp.dl[k], '+--g')
        
        if k == 1:
            title('EE (solid/dash = before/after deproj.)')
        if k==2:
            title('BB (solid/dash = before/after deproj.)')

        gca().set_yscale('log')
        xlabel('ell')
        grid('on')
        ylim(1e-10,10)
        if k==1:
            ylabel('l(l+1)/2pi Cl (uK^2)')
        if k==2:
            legend(loc='upper left')

        
