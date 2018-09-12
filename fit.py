import sim
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg
from sklearn.utils.extmath import randomized_svd
import fbpca
import sys

#from matplotlib.pyplot import *
#ion()

class fit(object):
    
    def __init__(self, map, type='Q'):
        # Do nothing
        self.m = map
        
        self.getmapinfo()
        self.setupregress(type)

        return

    def getmapinfo(self):
        """Get up some info"""

        self.ra = np.unique(self.m.ra)
        self.dec = np.unique(self.m.dec)
        self.ravec = np.ravel(self.m.ra)
        self.decvec = np.ravel(self.m.dec)
        self.Npair = len(self.m.fn)

    def setupregress(self, type='Q'):
        """Set up for regression"""

        if type == 'Q':
            y = np.ravel(self.m.Q)
            w = np.ravel(self.m.Qw)
            
        if type == 'U':
            y = np.ravel(self.m.U)
            w = np.ravel(self.m.Uw)
            
        if type == 'wz':
            y = np.ravel(self.m.acs['wz'])
            w = np.ravel(self.m.acs['w'])

        if type == 'T':
            y = np.ravel(self.m.T)
            w = np.ravel(self.m.Tw)


        self.fitind = np.where((np.isfinite(y)) & (y!=0))[0]
        Nfit = len(self.fitind)
        self.y = y[self.fitind]
        self.w = w[self.fitind]
        self.w = self.w / np.nanmax(self.w)


        for j in range(self.Npair):

            print('pair {0} of {1}'.format(j+1, self.Npair))

            fn = self.m.fn[j]
            print('loading {:s}...'.format(fn))
            sys.stdout.flush()
            ac = np.load(fn)

            if j==0:
                self.Npixtemp = len(ac['wct'])
                self.X = np.zeros((Nfit, self.Npixtemp*self.Npair), dtype='float32')


            if type == 'Q':
                a0 = ac['wct']
                a1 = self.m.acs['e']

                b0 = ac['wst']
                b1 = self.m.acs['f']

                c = self.m.acs['w']

            if type == 'U':
                a0 = ac['wct']
                a1 = self.m.acs['f']

                b0 = ac['wst']
                b1 = self.m.acs['g']

                c = self.m.acs['w']

            if type == 'wz':
                a0 = ac['wt']
                a1 = 1

                b0 = np.zeros(self.Npixtemp)
                b1 = 0

                c = self.m.acs['w']

            if type == 'T':
                a0 = ac['wt']
                a1 = 1
                
                b0 = np.zeros(self.Npixtemp)
                b1 = 0

                c = self.m.acs['w'] # should really be acs['wsum'], but this is
                                     # appropriate for 'wt' used above. 


            zz = np.zeros((self.Npixtemp, len(self.dec), len(self.ra)))

            for k in range(self.Npixtemp):
                z0 = (a0[k]*a1+b0[k]*b1)/c
                z0[~np.isfinite(z0)] = 0
                zz[k] = z0

            ac.close()

            for k,val in enumerate(self.fitind):

                ra0 = self.ravec[val]
                dec0 = self.decvec[val]

                indra = np.where(self.ra==ra0)[0][0]
                inddec = np.where(self.dec==dec0)[0][0]

                s = self.Npixtemp*j
                e = self.Npixtemp*(j+1)
                self.X[k,s:e] = zz[:,inddec,indra]
        

    def regress(self, cpmalpha=1e4, k=5000, b=None):
        """Fit"""

        print('Regressing, this could take a while...')
        sys.stdout.flush()

        if k is None:
            k = np.min(self.X.shape)-1

        self.k = k
        self.cpmalpha = cpmalpha

        if b is None:
            # Regress
            if np.size(cpmalpha) == 1:
                I = np.identity(self.X.shape[1], dtype='float32')*self.cpmalpha
            else:
                I = np.diag(cpmalpha)
            self.b = np.linalg.inv(self.X.T.dot(self.X) + I).dot(self.X.T).dot(self.y)
        else:
            # Just predict using provided coefficients
            self.b = b

        #X = self.X.toarray()
        #U,S,V = np.linalg.svd(X, full_matrices=False)
        #lam = np.ones(S.size)*cpmalpha
        #D = np.diag(S/(S**2+lam))
        #self.b = V.T.dot(D).dot(U.T).dot(self.y)

        #U,S,V = slinalg.svds(self.X, k=k)
        #U,S,V = randomized_svd(self.X*self.w[:,np.newaxis], n_components=k, n_iter=1)
        #U,S,V = randomized_svd(self.X, n_components=k, n_iter=1)
        #U,S,V = fbpca.pca(self.X*self.w[:,np.newaxis], k=k, n_iter=1)

        #self.U = U
        #self.S = S
        #self.V = V

        #D = np.diag(S/(S**2 + cpmalpha))
        #self.b = V.T.dot(D).dot(U.T).dot(self.y*self.w)

        yy = (self.X*self.w[:,np.newaxis]).dot(self.b)
        ypred = np.ones_like(self.ravec)*np.nan

        # Back to 2D map
        ypred[self.fitind] = yy / self.w
        self.zpred = ypred.reshape(self.m.T.shape)

