import numpy as np
from scipy import integrate, stats
from scipy.special import expit
from scipy.stats import norm
# Toy example
Q=np.array([.5,.3,.2]); P=np.array([.1,.3,.6]); m=np.array([4,2,.5])
print('naive',Q@m,'theta',P@m,'w',P/Q)
pi=.2; p=pi*P/(pi*P+(1-pi)*Q); o=p/(1-p); print('p',p.round(3),'odds',o.round(4),'SN',(Q*o)@m/(Q@o))
p2=p**2; o2=p2/(1-p2); est=(Q*o2)@m/(Q@o2); print('odds p^2',o2.round(4),'SN',est,'bias',est/(P@m)-1)
P4=np.array([.1,.3,.5,.1]); thac=(P4[:3]@m)/0.9; print('thac',thac,'interval',0.9*thac,0.9*thac+0.1*100, 'shrunk', 0.9*thac, thac)
wt=np.minimum(P/Q,2); print('trim',(Q*wt)@m/(Q@wt), 'unnorm',(Q*wt)@m)
phi=np.minimum(1,2/(P/Q)); print('Ephi',P@phi,'cov',P@(m*phi)-(P@m)*(P@phi))
# measurement example
mu=lambda x: expit(-3-0.8*x)
I=lambda f: integrate.quad(f,-15,15,limit=400)[0]
th=I(lambda x: mu(x)*norm.pdf(x-1))
for c in (0.3,-0.3):
    thX=I(lambda x: mu(x+c)*norm.pdf(x-1)); print('c',c,'theta',th,'thetaX',thX,'bias%',100*(thX/th-1),'AUC',norm.cdf((1+c)/np.sqrt(2)))
print('AUC no error',norm.cdf(1/np.sqrt(2)))
# Kish counterexample
rng=np.random.default_rng(0)
for n0,R in ((1000,20000),(100000,2000),(10**7,40)):
    r=[]
    for _ in range(R):
        x=rng.random(n0); w=0.5/np.sqrt(x); r.append(w.sum()**2/(w**2).sum()/n0)
    r=np.array(r); print(n0,'median',np.median(r).round(3),'p1,p99',np.quantile(r,[.01,.99]).round(3))
# sensitivity rare cond equivalence; lower bound
for L in (1.5,3,5):
    for pp in np.linspace(0.01,0.5,50):
        c=(1-L*pp)/(1-pp); assert (c>=1/L-1e-12)==(pp<=1/(1+L)+1e-12)
print('rare-outcome equivalence ok')
# 10% claim
print('P(one |z|>=3 in 40)',1-(1-2*norm.sf(3))**40)
