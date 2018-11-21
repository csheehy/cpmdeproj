import numpy as np

class Ridge:
    
    def __init__(self, alpha=1, fit_intercept=True, normalize=True):
        """Ridge regression, alpha can be scalar or array."""
        if not (fit_intercept & normalize):
            raise Exception('For now must fit intercept and normalize.')
        
        self.alpha = alpha
        return
    
    def fit(self, X, y):
        """Fit"""
        
        # Standardize
        X, y, Xnorm, Xm, ym = self.standardize(X, y)

        if np.size(self.alpha) == 1:
            I = np.identity(X.shape[1]).astype('float32')*self.alpha
        else:
            I = np.diag(self.alpha).astype('float32')

        # Direct inverse
        if np.any(self.alpha):
            self.coef_  = np.linalg.inv(X.T.dot(X) + I).dot(X.T).dot(y)
        else:
            self.coef_ = np.linalg.lstsq(X, y)[0]
        
        # Rescale coefficients
        self.coef_ /= Xnorm

        # Unscale design matrix and uncenter data
        X = X*Xnorm + Xm
        y += ym
        
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
