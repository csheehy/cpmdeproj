import map
import aps
import numpy as np
from glob import glob
from copy import deepcopy as dc

suffs = ['002_TR1.2++TR10.0+pol_alpha10_cpmlr_alldk++alldk',
         '002_deriv++TR4.0+pol_alpha1_cpmlr_perdk++alldk']

sls = ['', 'nosl']

for suff in suffs:
    for sl in sls:

        fnn = glob('maps/'+suff+'/noi_r????_dkxxx.npz')

        mt = map.map()
        ms = map.map()
        mn = map.map()
        me = map.map()

        a = {}
        adp = {}
        flds = ['snt_t','snt2_t', 'snt_pred', 'snt2_pred',
                't_predt','t','s','n','snt','sn'] 
        for k in flds:
            a[k]   = []
            adp[k] = []

        for fn in fnn:

            mn.load(fn)

            fn = fn.replace('noi','noi'+sl)
            ms.load(fn.replace('noi','sig'))
            mt.load(fn.replace('noi','TnoP'))
            me.load(fn.replace('noi','EnoB'))
            for k in ['Q','U','Qpred','Upred','b','bcpm','Qpred_cpm','Upred_cpm']:
                setattr(mn,k,getattr(mn,k)/np.sqrt(10))
            ms.Q -= mt.Q; ms.U -= mt.U
            ms.Qpred -= mt.Qpred; ms.Upred -= mt.Upred
            ms.Qpred_cpm -= mt.Qpred_cpm; ms.Upred_cpm -= mt.Upred_cpm

            msn  = map.addmaps(ms,mn)
            msnt = map.addmaps(msn,mt)

            msnt2 = dc(msnt)
            msnt2.Q -= msnt.Qpol
            msnt2.U -= msnt.Upol

            a['snt_t'].append(aps.aps(msnt,  mt).dl) # SAT x TnoP
            a['snt2_t'].append(aps.aps(msnt2, mt).dl) # (SAT-LAT) x TnoP
            a['snt_pred'].append(aps.aps(msnt, msnt, ext2='pred_cpm', mb=me).dl) # SAT x pred
            a['snt2_pred'].append(aps.aps(msnt2, msnt, ext2='pred_cpm').dl) # (SAT-LAT) x pred
            a['t_predt'].append(aps.aps(mt, mt, ext='pred_cpm').dl) # TnoP x pred_Tonly

            a['t'].append(aps.aps(mt).dl)
            a['s'].append(aps.aps(ms, mb=me).dl)
            a['n'].append(aps.aps(mn).dl)
            a['snt'].append(aps.aps(msnt, mb=me).dl)
            a['sn'].append(aps.aps(msn, mb=me).dl)


            mt.deproj()
            ms.deproj()
            mn.deproj()
            me.deproj()

            msn  = map.addmaps(ms,mn)
            msnt = map.addmaps(msn,mt)

            msnt2 = dc(msnt)
            msnt2.Q -= msnt.Qpol
            msnt2.U -= msnt.Upol

            adp['snt_t'].append(aps.aps(msnt,  mt).dl) # SAT x TnoP
            adp['snt2_t'].append(aps.aps(msnt2, mt).dl) # (SAT-LAT) x TnoP
            adp['snt_pred'].append(aps.aps(msnt, msnt, ext2='pred_cpm', mb=me).dl) # SAT x pred
            adp['snt2_pred'].append(aps.aps(msnt2, msnt, ext2='pred_cpm').dl) # (SAT-LAT) x pred
            adp['t_predt'].append(aps.aps(mt, mt, ext='pred_cpm').dl) # TnoP x pred_Tonly

            adp['t'].append(aps.aps(mt).dl)
            adp['s'].append(aps.aps(ms, mb=me).dl)
            adp['n'].append(aps.aps(mn).dl)
            adp['snt'].append(aps.aps(msnt, mb=me).dl)
            adp['sn'].append(aps.aps(msn, mb=me).dl)

        for k in a.keys():
            a[k]   = np.array(a[k])
            adp[k] = np.array(adp[k])

        aa = aps.aps(mt)
        a['l'] = aa.l
        adp['l'] = aa.l

        np.savez('aps/{:s}_nodp{:s}.npz'.format(suff, sl), **a)
        np.savez('aps/{:s}_dp{:s}.npz'.format(suff, sl), **adp)




