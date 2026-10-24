"""Respuesta media por deciles de peso, con pesos verdaderos y estimados (escenario base).

El signo del recorte lo decide cómo varía la respuesta con el peso (Prop. 4 del paper 1,
Prop. 3 del paper 2). Los deciles se forman por masa de la medida relevante:
  - pesos verdaderos w: deciles de w bajo P (llamadas del grupo objetivo);
  - pesos estimados ŵ: deciles de ŵ bajo P̂ (donantes ponderados por ŵ).
Recortar quita peso al decil superior; si su respuesta está por encima de la media, la
estimación baja, y si está por debajo, sube.
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


def deciles(key, resp, wt):
    o = np.argsort(key); cw = np.cumsum(wt[o]) / wt.sum(); dec = np.minimum((cw * 10).astype(int), 9)
    return np.array([np.sum((resp[o] * wt[o])[dec == k]) / np.sum(wt[o][dec == k]) for k in range(10)])


d = sample_calls(300_000, p, cs, np.random.default_rng(5))
Z, V, V5 = d["Z"], d["V"], d["V5"]; S = Z == 1
g, pD = g_and_pD(V, V5, p)
w = np.where(s_star(V, p) <= cs, pD / (1 - pD), 0.0)
dw = deciles(w[S], g[S], np.ones(S.sum()))
print("Pesos verdaderos, deciles de w bajo P (%):", np.round(dw * 100, 2), f" media {g[S].mean()*100:.2f}")
print("Correlaciones en el grupo objetivo: (w, V1) %.2f  (g, V1) %.2f  (w, g) %.2f" % (
    np.corrcoef(w[S], V[S, 0])[0, 1], np.corrcoef(g[S], V[S, 0])[0, 1], np.corrcoef(w[S], g[S])[0, 1]))

res, medias, fuera = [], [], []
for seed in range(5):
    d = sample_calls(n, p, cs, np.random.default_rng(9000 + seed))
    Z, V, V5 = d["Z"], d["V"], d["V5"]; cc = Z == 0
    g, pD = g_and_pD(V, V5, p)
    X = features(V, V5); ph = np.zeros(len(Z))
    for tr, te in GroupKFold(5).split(X, Z, d["client"]):
        ph[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
    ph = np.clip(ph, 1e-6, 1 - 1e-6); o = (ph / (1 - ph))[cc]
    res.append(deciles(o, g[cc], o)); medias.append(np.sum(o * g[cc]) / o.sum())
    wv = np.where(s_star(V, p) <= cs, pD / (1 - pD), 0.0)[cc]
    fuera.append(o[wv == 0].sum() / o.sum())
print("Pesos estimados, deciles de ŵ bajo P̂ (%), media de 5 réplicas:", np.round(np.mean(res, 0) * 100, 2),
      f" media {np.mean(medias)*100:.2f}")
print(f"Masa de P̂ en donantes fuera del grupo objetivo (w = 0): {np.mean(fuera):.3f}")
