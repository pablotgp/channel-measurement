"""Comprobación NO registrada: sesgo predicho para B con mu ajustada con LightGBM en lugar de splines.

Misma fórmula que nhanes_prediccion.py (asignación base, coef_bmi por defecto).
Uso: python ilustracion/nhanes_mu_lgbm.py -> ilustracion/prediccion_nhanes_mu_lgbm.json
"""
import json, os
import numpy as np
import nhanes_comun as c

df = c.cargar()
pi = c.prob_canal(df).values
cols = c.NUM_V + c.CAT_V + ["bmi"]
out = {}
for y in ("sbp", "waist"):
    mu = c.regresion("bmi", learner="lgbm").fit(df[cols], df[y])
    Xs = df[cols].copy(); Xs["bmi"] = df.bmi_self
    d = mu.predict(Xs) - mu.predict(df[cols])
    out[y] = dict(sesgo_predicho_B_mu_lgbm=float(np.sum(pi * d) / pi.sum()))
json.dump(out, open(os.path.join(c.HERE, "prediccion_nhanes_mu_lgbm.json"), "w"), indent=1)
print(out)
