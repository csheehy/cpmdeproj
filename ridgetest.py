from sklearn.linear_model import Ridge

# Data
x = np.linspace(0,1,1000)
y = 2 + 0.5*x + 2*x**2 - x**3
n = 0.2*np.random.randn(len(y))

r = Ridge(alpha=1, fit_intercept=False, normalize=True)

# Design matrix
X = np.zeros((len(y),4))
X[:,0] = 1
X[:,1] = x
X[:,2] = x**2
X[:,3] = x**3

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


print(cs + cn - csn)

clf();
plot(x,y+n)
plot(x,y)
plot(x,ps,'k',label='sig')
plot(x,pn,'c',label='noi')
plot(x,psn,'m',label='sig+noi')
plot(x,ps+pn - psn, '--', label='s+n-sn')
legend()
