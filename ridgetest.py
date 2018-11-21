from sklearn.linear_model import Ridge
#from ridge import Ridge

# Data
#x = np.linspace(0,1,100)
#y = 2 + 0.5*x + 2*x**2 - x**3
#n = 0.2*np.random.randn(len(y))

r = Ridge(alpha=0, fit_intercept=True, normalize=True)

# Design matrix
npoly = 95
X = np.zeros((len(y),npoly))
for k in range(npoly):
    X[:,k] = x**k

# Fit
r.fit(X,y)

cs = r.coef_
ps = r.predict(X)

r.fit(X,n)
cn = r.coef_
pn = r.predict(X)

r.fit(X,y+n)
csn = r.coef_
psn = r.predict(X)

I = np.identity(npoly)*1
Xm = X.mean(0)
Xstd = np.sqrt(np.sum((X-Xm)**2,0))
Xstd[Xstd==0] = 1
Xfit = (X-Xm)/Xstd
b  = np.linalg.inv(Xfit.T.dot(Xfit) + I).dot(Xfit.T).dot(y+n - (y+n).mean())
b = b/Xstd
intercept = (y+n).mean() - Xm.dot(b.T)
ypred = X.dot(b) + intercept

clf();
plot(x,y+n,'.b',label='sig+noi data')
plot(x,y,'.r',label='sig only data')
plot(x,n,'.c',label='noi only data')

plot(x,psn,'b',label='sig+noi fit')
plot(x,ps,'r',label='sig fit')
plot(x,pn,'c',label='noi fit')

#plot(x,ps+pn - psn, '--', label='s+n-sn')

legend(loc='upper left')
