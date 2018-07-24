import sim
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg
from sklearn.utils.extmath import randomized_svd

class fit(object):
    
    def __init__(self, map):
        # Do nothing
        self.m = map
        
        self.getmapinfo()
        self.setupregress()

        return

    def getmapinfo(self):
        """Get up some info"""

        self.ra = np.unique(self.m.ra)
        self.dec = np.unique(self.m.dec)
        self.ravec = np.ravel(self.m.ra)
        self.decvec = np.ravel(self.m.dec)


    def setupregress(self):
        """Set up for regression"""

        y = np.ravel(self.m.acs['wcz']) 

        self.fitind = np.where((np.isfinite(y)) & (y!=0))[0]
        Nfit = len(self.fitind)
        self.X = sparse.lil_matrix((Nfit, self.m.Npixtemp*self.m.Npair))
        self.y = y[self.fitind]

        for k,val in enumerate(self.fitind):

            print('{0} of {1}'.format(k,Nfit))

            ra0 = self.ravec[val]
            dec0 = self.decvec[val]

            indra = np.where(self.ra==ra0)[0][0]
            inddec = np.where(self.dec==dec0)[0][0]

            for j in range(self.m.Npair):
                s = self.m.Npixtemp*j
                e = self.m.Npixtemp*(j+1)
                self.X[k,s:e] = self.m.ac['wct'][j,:,inddec,indra]


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

        # Back to 2D map
        ypred[self.fitind] = yy
        self.zpred = ypred.reshape(self.m.T.shape)

