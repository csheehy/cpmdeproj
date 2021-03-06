import numpy as np
from sklearn.linear_model import Ridge as skRidge

class Ridge:
    
    def __init__(self, alpha=1, fit_intercept=True, normalize=True):
        """Ridge regression, alpha can be scalar or array."""
        if not (fit_intercept & normalize):
            raise Exception('For now must fit intercept and normalize.')
        
        self.alpha = np.atleast_1d(alpha*1.0)

        if len(np.unique(self.alpha)) == 1:
            self.alpha = self.alpha[0]

        return
    
    def fit(self, X, y, zi=None):
        """Fit"""
        
        # Standardize
        X, y, Xnorm, Xm, ym = self.standardize(X, y)
    
        if zi is not None:
            # Effectively zero alpha for this particular index
            fac = 1e12
            X[:,zi] *= fac

        if np.size(self.alpha) == 1:
            # SVD
            if self.alpha>0:
                U,S,V = np.linalg.svd(X, full_matrices=False)
                D = np.diag(S/(S**2 + self.alpha))
                b = V.T.dot(D).dot(U.T).dot(y)
            else:
                b = np.linalg.lstsq(X, y)[0]

            self.coef_ = b
        else:
            # Direct inverse. This is lame. Want a way to not have to invert
            # with multiple alpha, but this does not seem to work with the SVD
            # method above.
            I = np.diag(self.alpha).astype('float32')
            self.coef_  = np.linalg.inv(X.T.dot(X) + I).dot(X.T).dot(y)
        
        # Rescale coefficients
        self.coef_ /= Xnorm

        if zi is not None:
            self.coef_[zi] *= fac

        # Unscale design matrix and uncenter data
        X = X*Xnorm + Xm
        y += ym
        
        if zi is not None:
            X[:,zi] /= fac

        # Set intercept
        self.intercept_ = ym - Xm.dot(self.coef_.T)


    def predict(self, X, i=None):
        """Predict given design matrix and previous fit. Optionally provide
        index array of regressors (X 2nd dim) to use in prediction."""

        if i is None:
            i = range(X.shape[1])
        return X[:,i].dot(self.coef_[i].T) #+ self.intercept_


    def standardize(self, X, y):
        """Standardize data"""

        # Mean of data
        #ym = np.mean(y)
        ym = 0

        # Mean of regressors
        #Xm = X.mean(0)
        Xm = np.zeros_like(X.shape[1])

        # L2 norm
        Xnorm = np.sqrt(np.sum((X-Xm)**2, 0))
        Xnorm[Xnorm==0] = 1.0

        # Standardize
        X = (X-Xm) / Xnorm
        y -= ym
                
        return X, y, Xnorm, Xm, ym
