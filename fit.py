import sim
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg
from sklearn.utils.extmath import randomized_svd

class fit(object):
    
    def __init__(self, prefix='TnoP_noiseless'):
        # Do nothing
        self.loaddata(prefix=prefix)
        self.coadddata()
        self.setupregress()

        return

    def loaddata(self, prefix='TnoP_noiseless'):
        """Load data"""

        self.pairmaps = []
        for k in range(10):
            print(k)                         
            self.pairmaps.append(np.load('simdata/{:s}_{:04d}.npy'.format(prefix,k)).item())

        self.Npair = len(self.pairmaps)

        self.ra = np.unique(self.pairmaps[0].mapra)
        self.dec = np.unique(self.pairmaps[0].mapdec)
        self.ravec = np.ravel(self.pairmaps[0].mapra)
        self.decvec = np.ravel(self.pairmaps[0].mapdec)
        self.Npixmap = self.pairmaps[0].mapra.size
        self.Npixtemp = self.pairmaps[0].X.shape[1]

    def coadddata(self):
        """Coadd data into maps"""

        self.z = np.ones((self.Npair, self.dec.size, self.ra.size))*np.nan

        for k in range(self.Npair):
            y = np.ones_like(self.pairmaps[k].mapra)*np.nan
            y[self.pairmaps[k].mapind] = np.ravel(self.pairmaps[k].pairdiff)
            self.z[k] = y

        self.w = np.isfinite(self.z).astype(float)
        self.wsum = np.nansum(self.w, 0)
        self.zmean = np.nansum(self.z*self.w, 0) / self.wsum

    def setupregress(self):
        """Set up for regression"""

        y = np.ravel(self.zmean) 

        self.fitind = np.where(np.isfinite(y))[0]
        Nfit = len(self.fitind)
        self.X = sparse.lil_matrix((Nfit, self.Npixtemp*self.Npair))
        self.y = y[self.fitind]

        for k,val in enumerate(self.fitind):

            print('{0} of {1}'.format(k,Nfit))

            ra0 = self.ravec[val]
            dec0 = self.decvec[val]

            indra = np.where(self.ra==ra0)[0][0]
            inddec = np.where(self.dec==dec0)[0][0]
            wvec = self.w[:,inddec,indra]*1.0
            wsum = wvec.sum()
            if wsum > 0:
                wvec /= wvec.sum()

            for j,valw in enumerate(wvec):
                if valw==0:
                    continue
                s = self.Npixtemp*j
                e = self.Npixtemp*(j+1)
                idx = np.where((np.ravel(self.pairmaps[j].ra==ra0)) & (np.ravel(self.pairmaps[j].dec==dec0)))[0]
                self.X[k,s:e] = valw * self.pairmaps[j].X[idx, :]

        # Set up weights
        self.wvec = np.ravel(self.wsum)[self.fitind]
        self.y = self.y*self.wvec
        for k in range(self.X.shape[0]):
            self.X[k] = self.X[k] * self.wvec[k]


    def regress(self, cpmalpha=1e7, k=500):
        """Fit"""


        # To csc
        self.X = self.X.tocsc()

        if k is None:
            k = np.min(self.X.shape)-1

        #I = sparse.identity(self.X.shape[1])*cpmalpha
        #I = sparse.identity(self.X.shape[1])*cpmalpha
        #I = I.tocsc()
        #self.b = slinalg.inv(self.X.T.dot(self.X) + I).dot(self.X.T).dot(self.y)

        #X = self.X.toarray()
        #U,S,V = np.linalg.svd(X, full_matrices=False)
        #lam = np.ones(S.size)*cpmalpha
        #D = np.diag(S/(S**2+lam))
        #self.b = V.T.dot(D).dot(U.T).dot(self.y)

        #U,S,V = slinalg.svds(self.X, k=k)
        U,S,V = randomized_svd(self.X, n_components=k, n_iter=1)
        self.U = U
        self.S = S
        self.V = V

        D = np.diag(S/(S**2 + cpmalpha))
        self.b = V.T.dot(D).dot(U.T).dot(self.y)

        yy = self.X.dot(self.b)
        ypred = np.ones_like(self.ravec)*np.nan

        # Undo weights
        yy /= self.wvec

        # Back to 2D map
        ypred[self.fitind] = yy
        self.zpred = ypred.reshape(self.zmean.shape)

