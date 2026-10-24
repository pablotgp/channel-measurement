import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gen import Params, draw_clients, g_and_pD
from truth import population, wquantile
from scenarios import SCEN
# rare outcome
for nm in ['base','confunde_sin_C','confusion_U']:
    p=SCEN[nm][0]; c=draw_clients(600_000,p,np.random.default_rng(1)); CC=~c['D']
    g,_=g_and_pD(c['V'][CC],c['V5'][CC],p); m=c['m'][CC]
    for q in (0.999,0.9999):
        gq=wquantile(g,q,m.astype(float)); print(nm,q,'g_q',round(gq,4),'Lambda_max',round(1/gq-1,2))
    print(nm,'max g',g.max(),'Lambda',1/g.max()-1)
# lambda calibration
for lam in (0.3,0.5,0.75,1.0,1.5,2.0,3.0):
    r=population(Params(lam=lam),n=300_000,seed=11)
    print('lam',lam,'chi2',round(r['chi2'],2),'auc',round(r['auc'],4))
