"""¿Se invierte de verdad el signo del recorte con pesos estimados?

Por réplica (escenario base, semillas nuevas 120000+): efecto del recorte en p99, p95 y p90
  - con pesos estimados ŵ y la respuesta verdadera g (sin ruido de Y)   -> 'est_g'
  - con pesos estimados ŵ y Y                                           -> 'est_Y'
  - con pesos verdaderos w y g                                         -> 'ver_g'
Reporta media, error de Monte Carlo y z frente a 0.
"""
import json, sys
import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from scenarios import SCEN
from gen import sample_calls, g_and_pD, s_star
from truth import features
from est import LGB_CLF

p, _, n = SCEN["base"]
cs = json.load(open("c_star.json"))["base"]

def one(seed):
    d = sample_calls(n, p, cs, np.random.default_rng(seed))
    Z, Y, V, V5 = d["Z"], d["Y"], d["V"], d["V5"]; cc = Z == 0
    g, pD = g_and_pD(V, V5, p)
    w = np.where(s_star(V, p) <= cs, pD / (1 - pD), 0.0)
    X = features(V, V5); ph = np.zeros(len(Z))
    for tr, te in GroupKFold(5).split(X, Z, d["client"]):
        ph[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
    ph = np.clip(ph, 1e-6, 1 - 1e-6); o = (ph / (1 - ph))[cc]
    wc, gc, Yc = w[cc], g[cc], Y[cc]
    out = {}
    for q in (99, 95, 90):
        def eff(a, resp):
            t = np.quantile(a, q / 100); at = np.minimum(a, t)
            return np.sum(at * resp) / np.sum(at) - np.sum(a * resp) / np.sum(a)
        out[q] = (eff(o, gc), eff(o, Yc), eff(wc, gc))
    return out

R = int(sys.argv[1])
res = Parallel(n_jobs=2)(delayed(one)(120_000 + r) for r in range(R))
for q in (99, 95, 90):
    A = np.array([r[q] for r in res]) * 1e5
    for j, nom in enumerate(("ŵ con g (sin ruido de Y)", "ŵ con Y", "w con g")):
        x = A[:, j]; m, s = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        print(f"p{q} {nom:24s}: {m:+6.2f} ± {s:.2f} milésimas   z vs 0 = {m/s:+.1f}   réplicas con signo +: {np.mean(x>0)*100:.0f} %")
