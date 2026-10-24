"""Ilustración semisintética con NHANES: medición distinta del IMC en dos «canales».

Cada réplica asigna el canal con prob_canal (nhanes_comun.py); el grupo objetivo es el canal 1.
theta = media del resultado en el grupo objetivo (se observa: es una prueba con verdad conocida).
Estimador doblemente robusto con cross-fitting (5 pliegues), clasificador y regresión con splines.
Especificaciones A-D: ver nhanes_prediccion.py. Las predicciones se registraron antes (prediccion_nhanes.json).
"""
import json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import nhanes_comun as c

df = c.cargar()
PI = c.prob_canal(df).values
SPECS = {"A": ("bmi", "bmi"), "B": ("bmi_self", "bmi"), "C": (None, None), "D": ("bmi_self", "bmi_self")}


def matriz(S, col_obj, col_don):
    X = df[c.NUM_V + c.CAT_V].copy()
    if col_obj:
        X["b"] = np.where(S, df[col_obj], df[col_don])
    return X


def una(seed):
    rng = np.random.default_rng([seed, 7])
    S = rng.random(len(df)) < PI
    out = dict(seed=seed, n1=int(S.sum()))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(df, S))
    for k, (co, cd) in SPECS.items():
        X = matriz(S, co, cd); bcol = "b" if co else None
        p = np.zeros(len(df)); m = {y: np.zeros(len(df)) for y in ("sbp", "waist")}
        for tr, te in folds:
            p[te] = c.clasificador(bcol).fit(X.iloc[tr], S[tr]).predict_proba(X.iloc[te])[:, 1]
            don = tr[~S[tr]]
            for y in m:
                m[y][te] = c.regresion(bcol).fit(X.iloc[don], df[y].values[don]).predict(X.iloc[te])
        p = np.clip(p, 1e-6, 1 - 1e-6); o = p / (1 - p); D0 = ~S
        r = dict(auc=roc_auc_score(S, p), masa_099=float(np.mean(p[S] > 0.99)),
                 neff_n0=float(o[D0].sum() ** 2 / np.sum(o[D0] ** 2) / D0.sum()))
        for y in m:
            Y = df[y].values
            A = np.sum(o[D0] * (Y[D0] - m[y][D0])) / np.sum(o[D0])
            r[f"dr_{y}"] = A + m[y][S].mean()
            r[f"hajek_{y}"] = np.sum(o[D0] * Y[D0]) / np.sum(o[D0])
            r[f"theta_{y}"] = Y[S].mean()
        out[k] = r
    # Detector más flexible (LightGBM) solo para A y B: ¿distingue la medición distinta?
    if seed < 40:
        for k in ("A", "B"):
            X = matriz(S, *SPECS[k]); p = np.zeros(len(df))
            for tr, te in folds:
                g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1, random_state=seed)
                p[te] = g.fit(X.iloc[tr], S[tr]).predict_proba(X.iloc[te])[:, 1]
            out[k]["auc_lgbm"] = roc_auc_score(S, p)
            out[k]["masa_099_lgbm"] = float(np.mean(p[S] > 0.99))
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    res = Parallel(n_jobs=-1)(delayed(una)(s) for s in range(R))
    json.dump(res, open(os.path.join(c.HERE, f"res_nhanes_{R}.json"), "w"))
    pred = json.load(open(os.path.join(c.HERE, "prediccion_nhanes.json")))["valores"]
    filas = []
    for k in SPECS:
        for y in ("sbp", "waist"):
            b = np.array([r[k][f"dr_{y}"] - r[k][f"theta_{y}"] for r in res])
            bh = np.array([r[k][f"hajek_{y}"] - r[k][f"theta_{y}"] for r in res])
            filas.append(dict(spec=k, y=y, sesgo_dr=b.mean(), mcse=b.std(ddof=1) / np.sqrt(R), de=b.std(ddof=1),
                              sesgo_hajek=bh.mean(), pred_B=pred[y]["sesgo_predicho_B"] if k == "B" else None))
        a = [r[k]["auc"] for r in res]; mm = [r[k]["masa_099"] for r in res]; ne = [r[k]["neff_n0"] for r in res]
        print(f"{k}: AUC {np.mean(a):.3f}  masa p>0,99 {np.mean(mm):.4f}  neff/n0 {np.mean(ne):.3f}"
              + (f"  AUC LightGBM {np.mean([r[k]['auc_lgbm'] for r in res if 'auc_lgbm' in r[k]]):.3f}"
                 f"  masa LightGBM {np.mean([r[k]['masa_099_lgbm'] for r in res if 'auc_lgbm' in r[k]]):.4f}" if k in 'AB' else ""))
    t = pd.DataFrame(filas); print(t.round(3).to_string(index=False))
    t.to_csv(os.path.join(c.HERE, f"resumen_nhanes_{R}.csv"), index=False)
