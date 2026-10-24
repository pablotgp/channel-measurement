"""Contraste no tautológico del lema de calibración.

Se estiman por separado, con datos, (a) el modelo 'cola frente a CC' (Z) y (b) el modelo de canal (D),
cada uno con y sin U, y se comparan los cuantiles de max(λ,1/λ) en la región R = {s* <= c*}.
"""
import json
import numpy as np
import lightgbm as lgb
from scenarios import SCEN
from gen import draw_clients, s_star
from truth import features

p, _, _ = SCEN["confusion_U"]
cs = json.load(open("c_star.json"))["confusion_U"]
c = draw_clients(300_000, p, np.random.default_rng(11))
V, V5, U, D, m = c["V"], c["V5"], c["U"], c["D"], c["m"]
R = s_star(V, p) <= cs
S = D & R
X = features(V, V5); XU = np.c_[X, U]
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, verbose=-1, n_jobs=2)
half = np.random.default_rng(5).random(len(D)) < 0.5

def odds_out(Xm, y, sub):
    o = np.full(len(y), np.nan)
    for tr, te in ((half & sub, ~half & sub), (~half & sub, half & sub)):
        pr = lgb.LGBMClassifier(**LGB).fit(Xm[tr], y[tr], sample_weight=m[tr]).predict_proba(Xm[te])[:, 1]
        pr = np.clip(pr, 1e-6, 1 - 1e-6); o[te] = pr / (1 - pr)
    return o

def wq(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum(); return x[o][np.searchsorted(cw, q)]

pool = S | ~D                                   # muestra combinada: cola (Z=1) y canal básico (Z=0)
lz = odds_out(XU, S.astype(int), pool) / odds_out(X, S.astype(int), pool)
ld = odds_out(XU, D.astype(int), np.ones(len(D), bool)) / odds_out(X, D.astype(int), np.ones(len(D), bool))
sel = R & pool
for nom, l in (("modelo cola frente a CC", lz), ("modelo de canal", ld)):
    s = np.maximum(l, 1 / l)[sel]
    print(f"{nom:24s}: " + "  ".join(f"q{int(q*100)} = {wq(s, m[sel].astype(float), q):.2f}" for q in (0.5, 0.9, 0.99)))
