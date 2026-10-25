"""p̂ de los participantes del NSW con ingresos de 1974 por encima del tope de la CPS.

Para cada clasificador y cada una de las 5 particiones del cross-fitting usadas en lalonde.py,
guarda p̂ de esos participantes y su cociente de odds frente a las odds a priori pi/(1-pi).
Uso: python ilustracion/lalonde_tope.py -> ilustracion/resultados_lalonde_tope.json
"""
import json, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from lalonde import nsw, cps, features, fit_predict, TOP, HERE

t = nsw[nsw.treat == 1].reset_index(drop=True)
X = pd.concat([features(t), features(cps)], ignore_index=True)
Z = np.r_[np.ones(len(t)), np.zeros(len(cps))]
above = np.where(t.re74.values > TOP["re74"])[0]
pi = Z.mean(); prior = pi / (1 - pi)
out = dict(pi=pi, tope_re74=TOP["re74"], re74_sobre_tope=[float(v) for v in t.re74.values[above]], resultados={})
for kind in ("logit", "lgbm"):
    rows = []
    for seed in range(5):
        p = np.zeros(len(Z))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, Z):
            p[te] = fit_predict(kind, X.iloc[tr], Z[tr], X.iloc[te], True, seed)
        ph = p[above]
        rows.append(dict(p=[float(v) for v in ph], odds_rel=[float(v / (1 - v) / prior) for v in ph],
                         p_max_target=float(p[Z == 1].max())))
    out["resultados"][kind] = rows
    print(kind, [[round(v, 3) for v in r["p"]] for r in rows])
json.dump(out, open(os.path.join(HERE, "resultados_lalonde_tope.json"), "w"), indent=1)
print("pi", round(pi, 4), "re74", out["re74_sobre_tope"])
