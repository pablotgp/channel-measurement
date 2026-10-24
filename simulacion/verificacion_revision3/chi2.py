import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gen import Params, draw_clients, s_star, p_hist, lin_D
from scipy.special import expit
from truth import wquantile, wmean
def pD_V(V,V5,p):  # nu=0: integrate H only
    ph=p_hist(V); n=len(V)
    return (1-ph)*expit(lin_D(V,V5,np.zeros(n),np.zeros(n),p))+ph*expit(lin_D(V,V5,np.ones(n),np.zeros(n),p))
for lam in (0.5,1.5,3.0):
    p=Params(lam=lam); out=[]
    for seed in (11,12,13,14):
        c=draw_clients(1_500_000,p,np.random.default_rng(seed)); V,V5,D,m=c['V'],c['V5'],c['D'],c['m']
        s=s_star(V,p); cs=wquantile(s[D],0.10,m[D].astype(float)); CC=~D
        pD=pD_V(V,V5,p); w=np.where(s<=cs,pD/(1-pD),0.0); w=w/wmean(w,CC,m)
        out.append(round(wmean(w**2,CC,m)-1,2))
    print('lam',lam,'chi2 by seed',out)
