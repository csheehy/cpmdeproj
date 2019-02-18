import aps
import map
from copy import deepcopy as dc
from matplotlib.pyplot import *
from sim import readcambfits

suff = '002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk'
#suff = '002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk'
rlz = 0

mt = map.map()
ms = map.map()
mn = map.map()
me = map.map()

mt.load('maps/'+suff+'/TnoPnosl_r{:04d}_dkxxx.npz'.format(rlz))
ms.load('maps/'+suff +'/signosl_r{:04d}_dkxxx.npz'.format(rlz))
mn.load('maps/'+suff +'/noi_r{:04d}_dkxxx.npz'.format(rlz))
me.load('maps/'+suff+'/EnoBnosl_r{:04d}_dkxxx.npz'.format(rlz))


for k in ['Q','U','Qpred','Upred','b','bcpm','Qpred_cpm','Upred_cpm']:
    setattr(mn,k,getattr(mn,k)/np.sqrt(10))

ms.Q -= mt.Q
ms.U -= mt.U

# S+N
msn = map.addmaps(ms,mn)


# Get undeproj aps
ass = aps.aps(ms, mb=me)
an = aps.aps(mn)
asn = aps.aps(msn, mb=me)
at = aps.aps(mt)

# Now deproject
ms.deproj()
mn.deproj()
mt.deproj()
msn.deproj()
me.deproj()

ms.Q += mt.Qpred
ms.U += mt.Upred
msn.Q += mt.Qpred
msn.U += mt.Upred



# Get post deproj aps
assdp = aps.aps(ms, mb=me)
andp = aps.aps(mn)
asndp = aps.aps(msn, mb=me)
atdp = aps.aps(mt)

cl,nm = readcambfits('spec/camb_planck2013_r0p1.fits')
r0p1 = cl[2]
lr = np.arange(r0p1.size)
r = 0.003

# Now plot
clf()
for k,s in enumerate([3,1,2]):

    l = ass.l

    subplot(1,3,k+1)
    plot(l, asn.dl[s], '.-r', label='s+n')
    plot(l, ass.dl[s], '.-b', label='sig')
    plot(l, an.dl[s], '.-c', label='noi')
    plot(l, at.dl[s], '.-g', label='TnoP')

    plot(l, asndp.dl[s], '+--r')
    plot(l, assdp.dl[s], '+--b')
    plot(l, andp.dl[s], '+--c')
    plot(l, atdp.dl[s], '+--g')
    
    xl = xlim()

    plot(lr, lr*(lr+1)*r0p1*r/0.1/(2*np.pi), '--k',label='r={:0.0e}'.format(r))
    xlim(xl)

    if s == 1:
        title('EE')
    if s==2:
        title('BB')
    if s==3:
        title('TE')

    if s != 3:
        gca().set_yscale('log')
        ylim(1e-8,10)

    xlabel('ell')
    grid('on')

    if k==0:
        ylabel('l(l+1)/2pi Cl (uK^2)')
    if k==2:
        legend(loc='upper left')

show()        
