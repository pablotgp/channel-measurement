"""Diagnóstico en la simulación: ¿por qué se invierte el signo del recorte con ŵ?

En réplicas del escenario base: w verdadero, ŵ (LightGBM con cross-fitting), g(V) verdadero.
1) Efecto predicho con la fórmula A usando g (sin ruido de Y) frente al observado con Y.
2) Media de g bajo P_hat por deciles de ŵ (el signo lo decide su monotonía).
3) Composición del decil superior de ŵ: ¿tienen w alto, o w bajo/cero con error positivo?
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
pred_A, obs, oracle = [], [], []
dec_g, top_info = [], []
for r in range(30):
    d = sample_calls(n, p, cs, np.random.default_rng(80_000 + r))
    Z, Y, V, V5 = d["Z"], d["Y"], d["V"], d["V5"]; cc = Z == 0
    g, pD = g_and_pD(V, V5, p)
    w = np.where(s_star(V, p) <= cs, pD / (1 - pD), 0.0)
    X = features(V, V5); ph = np.zeros(len(Z))
    for tr, te in GroupKFold(5).split(X, Z, d["client"]):
        ph[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
    ph = np.clip(ph, 1e-6, 1 - 1e-6); o = ph / (1 - ph)
    oc, wc, gc, Yc = o[cc], w[cc], g[cc], Y[cc]
    tau = np.quantile(oc, 0.90)
    phi = np.minimum(1, tau / oc); P = oc / oc.sum()
    pred_A.append((np.sum(P * gc * phi) - np.sum(P * gc) * np.sum(P * phi)) / np.sum(P * phi))
    hj = lambda a: np.sum(a * Yc) / np.sum(a)
    obs.append(hj(np.minimum(oc, tau)) - hj(oc))
    tw = np.quantile(wc, 0.90); hjg = lambda a: np.sum(a * gc) / np.sum(a)
    oracle.append(hjg(np.minimum(wc, tw)) - hjg(wc))
    # deciles de ŵ bajo P_hat
    qs = np.quantile(oc, np.linspace(0, 1, 11)); b = np.clip(np.digitize(oc, qs[1:-1]), 0, 9)
    dec_g.append([np.sum(P[b == k] * gc[b == k]) / np.sum(P[b == k]) for k in range(10)])
    top = oc >= tau
    wn = wc / wc.mean(); on = oc / oc.mean()
    top_info.append([np.mean(wc[top] == 0), np.mean(gc[top]), np.mean(gc[(wn > np.quantile(wn, 0.9))]),
                     np.sum(P[top] * gc[top]) / np.sum(P[top]), np.sum(P * gc)])
print(f"Efecto recorte p90 (Hájek, ŵ): observado con Y {np.mean(obs)*1e5:+.2f}e-5 | fórmula A con g {np.mean(pred_A)*1e5:+.2f}e-5")
print(f"Efecto recorte p90 con w verdadero (con g): {np.mean(oracle)*1e5:+.2f}e-5")
print("Media de g bajo P_hat por decil de ŵ (1..10):", np.round(np.mean(dec_g, 0) * 100, 3))
t = np.mean(top_info, 0)
print(f"Decil superior de ŵ: fracción con w=0 {t[0]:.3f}; g medio {t[1]*100:.3f}% (vs decil superior de w verdadero {t[2]*100:.3f}%)")
print(f"Bajo P_hat: g medio en decil sup. de ŵ {t[3]*100:.3f}% frente a g medio global {t[4]*100:.3f}%")
