"""¿El signo del recorte depende de usar pesos verdaderos o estimados?

En las mismas réplicas del escenario base: recorte con w verdadero (oráculo) frente a
recorte con ô estimado (LightGBM con cross-fitting). Diferencias pareadas frente a la
predicción de la Proposición 4.
"""
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scenarios import SCEN
from gen import sample_calls, g_and_pD, s_star
from truth import features
from est import LGB_CLF

p, _, n = SCEN["base"]
cs = json.load(open("c_star.json"))["base"]
pred = json.load(open("predicciones.json"))["base"]
out = {q: {"oraculo": [], "estimado": []} for q in (99, 95, 90)}
for r in range(60):
    d = sample_calls(n, p, cs, np.random.default_rng(50_000 + r))
    Z, Y, V, V5 = d["Z"], d["Y"], d["V"], d["V5"]; cc = Z == 0
    _, pD = g_and_pD(V, V5, p)
    w = np.where(s_star(V, p) <= cs, pD / (1 - pD), 0.0)
    X = features(V, V5); ph = np.zeros(len(Z))
    for tr, te in GroupKFold(5).split(X, Z, d["client"]):
        ph[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
    ph = np.clip(ph, 1e-6, 1 - 1e-6); o = ph / (1 - ph)
    hj = lambda a: np.sum(a[cc] * Y[cc]) / np.sum(a[cc])
    for q in out:
        for nom, a in (("oraculo", w), ("estimado", o)):
            tau = np.quantile(a[cc], q / 100)
            out[q][nom].append(hj(np.minimum(a, tau)) - hj(a))
for q in out:
    pq = pred[f"lim_trim{q}"] - pred["lim"]
    for nom in ("oraculo", "estimado"):
        x = np.array(out[q][nom]); m, s = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        print(f"p{q} {nom:8s}: diferencia media {m:+.6f} ± {s:.6f}  predicción {pq:+.6f}  z={(m-pq)/s:+.2f}")
