"""Ilustración con datos públicos: NSW (Dehejia-Wahba) frente a CPS-1 (LaLonde 1986).

Grupo objetivo: tratados de NSW (S=1). Canal donante: CPS-1 (D=0).
Estimando: theta = E[Y(0) | tratados], ganancias de 1978.
Referencia experimental: media de re78 en el grupo de control aleatorizado de NSW.

Los ingresos de CPS están censurados por arriba (top-coding) y los de NSW no.
Es un caso real de una variable medida de forma distinta en cada fuente.

Uso: python ilustracion/lalonde.py  -> imprime resultados y guarda ilustracion/resultados_lalonde.json
"""
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
nsw = pd.read_stata(os.path.join(HERE, "datos", "nsw_mixtape.dta"))
cps = pd.read_stata(os.path.join(HERE, "datos", "cps_mixtape.dta"))
TOP = {v: float(cps[v].max()) for v in ("re74", "re75", "re78")}


def features(d, cap=False, extra=True):
    X = d[["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]].astype(float).copy()
    if cap:
        for v in ("re74", "re75"):
            X[v] = np.minimum(X[v], TOP[v])
    if extra:
        X["u74"] = (X.re74 == 0).astype(float); X["u75"] = (X.re75 == 0).astype(float)
        X["age2"] = X.age ** 2; X["educ2"] = X.educ ** 2
    return X


def fit_predict(kind, Xtr, ytr, Xte, clf=True, seed=0):
    if kind == "logit":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)) if clf else make_pipeline(StandardScaler(), LinearRegression())
    else:
        P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m = lgb.LGBMClassifier(**P) if clf else lgb.LGBMRegressor(**P)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1] if clf else m.predict(Xte)


def estimate(kind, cap_X=False, cap_Y=False, K=5, seed=0):
    t = nsw[nsw.treat == 1]
    X = pd.concat([features(t, cap_X), features(cps, cap_X)], ignore_index=True)
    Z = np.r_[np.ones(len(t)), np.zeros(len(cps))]
    Y = np.r_[t.re78.values, np.minimum(cps.re78.values, TOP["re78"]) if cap_Y else cps.re78.values]
    p = np.zeros(len(Z)); m = np.zeros(len(Z))
    for tr, te in StratifiedKFold(K, shuffle=True, random_state=seed).split(X, Z):
        p[te] = fit_predict(kind, X.iloc[tr], Z[tr], X.iloc[te], True, seed)
        c = tr[Z[tr] == 0]
        m[te] = fit_predict(kind, X.iloc[c], Y[c], X.iloc[te], False, seed)
    p = np.clip(p, 1e-6, 1 - 1e-6); o = p / (1 - p)
    D0 = Z == 0; S = Z == 1
    w = o[D0] / o[D0].mean() * 1.0
    hajek = np.sum(o[D0] * Y[D0]) / np.sum(o[D0])
    A = np.sum(o[D0] * (Y[D0] - m[D0])) / np.sum(o[D0])
    dr = A + m[S].mean()
    pi = S.mean()
    phi = np.zeros(len(Z))
    phi[S] = (m[S] - m[S].mean()) / pi
    phi[D0] = o[D0] * (Y[D0] - m[D0] - A) / (np.sum(o[D0]) / len(Z))
    se = np.sqrt(np.mean(phi ** 2) / len(Z))
    neff = o[D0].sum() ** 2 / np.sum(o[D0] ** 2)
    return dict(clasificador=kind, cap_X=cap_X, cap_Y=cap_Y, hajek=hajek, dr=dr, se_dr=se,
                masa_p_099=float(np.mean(p[S] > 0.99)), masa_p_0999=float(np.mean(p[S] > 0.999)),
                neff=float(neff), neff_n0=float(neff / D0.sum()),
                peso_max_frac=float(o[D0].max() / o[D0].sum()))


if __name__ == "__main__":
    t, c0 = nsw[nsw.treat == 1], nsw[nsw.treat == 0]
    bench = dict(theta=float(c0.re78.mean()), se=float(c0.re78.std() / np.sqrt(len(c0))),
                 theta_cap=float(np.minimum(c0.re78, TOP["re78"]).mean()),
                 media_tratados=float(t.re78.mean()), media_cps=float(cps.re78.mean()))
    medicion = {v: dict(tope_cps=TOP[v], frac_cps_en_tope=float((cps[v] >= TOP[v] - 1).mean()),
                        frac_tratados_sobre_tope=float((t[v] > TOP[v] + 1).mean()),
                        n_tratados_sobre_tope=int((t[v] > TOP[v] + 1).sum()))
                for v in TOP}
    res = []
    for kind in ("logit", "lgbm"):
        for cap_X, cap_Y in ((False, False), (True, False), (True, True)):
            r = [estimate(kind, cap_X, cap_Y, seed=s) for s in range(5)]  # 5 particiones del cross-fitting
            agg = {k: (float(np.median([x[k] for x in r])) if isinstance(r[0][k], float) else r[0][k]) for k in r[0]}
            res.append(agg)
    out = dict(referencia=bench, medicion=medicion, estimaciones=res)
    json.dump(out, open(os.path.join(HERE, "resultados_lalonde.json"), "w"), indent=1)
    print(f"Referencia experimental theta = {bench['theta']:.0f} (EE {bench['se']:.0f}); con tope {bench['theta_cap']:.0f}")
    print(f"Media tratados {bench['media_tratados']:.0f}; media CPS {bench['media_cps']:.0f}")
    for v, d in medicion.items():
        print(v, d)
    for r in res:
        print(f"{r['clasificador']:5s} capX={r['cap_X']!s:5s} capY={r['cap_Y']!s:5s} Hajek {r['hajek']:7.0f}  DR {r['dr']:7.0f} (EE {r['se_dr']:5.0f})"
              f"  masa p>0.99 {r['masa_p_099']:.3f}  p>0.999 {r['masa_p_0999']:.3f}  neff {r['neff']:6.1f}  peso max {r['peso_max_frac']:.3f}")
