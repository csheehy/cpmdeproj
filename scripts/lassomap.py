import map
import aps
from sim import gnomproj
from ridge import Ridge
from sklearn.linear_model import Lasso

doload = True

if doload:

    suff = '003_TR1.2++TR10.0+pol_alpha1_cpmlr_alldk++alldk'

    mt = map.map()
    ms = map.map()
    mn = map.map()
    me = map.map()

    mt.load('maps/'+suff+'/TnoP_r0000_dkxxx.npz')
    ms.load('maps/'+suff+'/sig_r0000_dkxxx.npz')
    mn.load('maps/'+suff+'/noi_r0000_dkxxx.npz')
    me.load('maps/'+suff+'/EnoB_r0000_dkxxx.npz')

    for k in ['Q','U','Qpred','Upred','b','bcpm','Qpred_cpm','Upred_cpm']:
        setattr(mn,k,getattr(mn,k)/np.sqrt(10))    
    for k in mn.acs.keys():
        mn.acs[k] /= np.sqrt(10)

    mt.deproj()
    ms.deproj()
    mn.deproj()
    me.deproj()

    msn = map.addmaps(ms,mn)
    mtn = map.addmaps(mt,mn)
    men = map.addmaps(me,mn)

    msn2 = dc(msn)
    msn2.Q -= msn.Qpol
    msn2.U -= msn.Upol

    msnlat = dc(msn)
    msnlat.Q = msn.Qpol
    msnlat.U = msn.Upol


    # Construct regressor
    hmap = hp.read_map('input_maps/camb_planck2013_r0_lensing_lensfix_A6p125_n1024_r0000.fits',field=0)
    hmapn = hp.read_map('planckmaps/mc_noise/143/ffp8_noise_143_full_map_mc_00000.fits',field=0)
    hmap = hp.ud_grade(hmap,512) + hp.ud_grade(hmapn,512)
    del hmapn

    ytc = np.ravel(mt.acs['wcz'])
    ind = np.where(np.isfinite(ytc))
    indarr = np.where(np.isfinite(mt.acs['wcz']))
    ytc = ytc[ind]
    ysc = np.ravel(ms.acs['wcz'])[ind]
    ysnc = np.ravel(msn.acs['wcz'])[ind]
    ypredc = np.ravel(msn.acs['wczpred'])[ind]
    ytpredc = np.ravel(mt.acs['wczpred'])[ind]
    ylatc = np.ravel(ms.acs['wctpol'])[ind] # s not sn since we improperly
                                           # divided sn by sqrt(10) above

    yts = np.ravel(mt.acs['wsz'])[ind]
    yss = np.ravel(ms.acs['wsz'])[ind]
    ysns = np.ravel(msn.acs['wsz'])[ind]
    ypreds = np.ravel(msn.acs['wszpred'])[ind]
    ytpredcs = np.ravel(mt.acs['wszpred'])[ind]
    ylats = np.ravel(ms.acs['wstpol'])[ind] # s not sn since we improperly
                                           # divided sn by sqrt(10) above

    ra = np.ravel(mt.ra)[ind]
    dec = np.ravel(mt.dec)[ind]

    xs = 20.0 # deg
    ys = 20.0 # deg
    reso = 0.2 # deg
    xx,yy,dum,dum,gp = gnomproj(hmap, ra[0], dec[0], xs, ys, reso, Full=True)

    X = np.zeros((ytc.size, gp.size), dtype='float32')

    for k in range(len(ytc)):
        print(k)
        X[k] = np.ravel(gnomproj(hmap, ra[k], dec[k], xs, ys, reso, Full=False))


dofit = True

if dofit:

    bc = []
    bs = []

    wczpred = np.zeros_like(mt.acs['wcz'])
    wszpred = np.zeros_like(mt.acs['wcz'])

    #r = Ridge(alpha=1,normalize=True)
    r = Lasso(alpha=1e-4,normalize=True)
    w = np.ravel(mn.acs['w'])[ind]

    ww = np.tile(np.atleast_2d(w).T, X.shape[1])

    yc = (ysnc-ypredc-ylatc)*1.0
    ys = (ysns-ypreds-ylats)*1.0

    rind = np.where( (np.abs(np.ravel(xx))>2) & (np.abs(np.ravel(yy))>2))[0]
    ra = np.ravel(mt.ra)[ind]
    dec = np.ravel(mt.dec)[ind]

    #############
    noind = np.where( (np.abs(np.ravel(xx))<2) & (np.abs(np.ravel(yy))<2))[0]
    X[:,noind] = 0

    f = np.where(ra<-5)[0]
    p = np.where(ra>0)
    pp = tuple([g[p] for g in indarr])

    r.fit((X*ww)[f], (yc*w)[f])
    bc.append(r.coef_*1.0)
    wczpred[pp] = r.predict(X[p])#, rind)
    r.fit((X*ww)[f], (ys*w)[f])
    bs.append(r.coef_*1.0)
    wszpred[pp] = r.predict(X[p])#, rind)


    f = np.where(ra>5)[0]
    p = np.where(ra<=0)
    pp = tuple([g[p] for g in indarr])
    r.fit((X*ww)[f], (yc*w)[f])
    bc.append(r.coef_*1.0)
    wczpred[pp] = r.predict(X[p])#, rind)
    r.fit((X*ww)[f], (ys*w)[f])
    bs.append(r.coef_*1.0)
    wszpred[pp] = r.predict(X[p])#, rind)

    bc = np.array(bc)
    bs = np.array(bs)


Qpred = ms.acs['e']*wczpred + ms.acs['f']*wszpred
Upred = ms.acs['f']*wczpred + ms.acs['g']*wszpred

mpred = dc(msn)
mpred.Q = Qpred
mpred.U = Upred

msn2 = dc(msn)
msn2.Q -= msnlat.Q
msn2.U -= msnlat.U

a = aps.aps(mt, mpred)
asn = aps.aps(msn2, mpred)
att = aps.aps(mt)

clf()
plot(att.l,att.dl[2],'ok',label='TnoP')
plot(a.l, a.dl[2][2],'.b-',label='TnoP x pred')
plot(asn.l, asn.dl[2][2],'.r-',label='(SAT-LAT) x pred')
plot([0,500],[0,0],'k:')
legend()


