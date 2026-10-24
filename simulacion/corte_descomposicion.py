"""Análisis exploratorio (no registrado de antemano): ¿la mejora del corte conocido viene
de imponer el corte o de usar las llamadas premium de fuera de la cola?

Variante intermedia: el clasificador estándar (cola frente a donantes, sin datos nuevos)
con el corte impuesto a mano, ô = 1{s(X) <= c}·p̂/(1 - p̂). Mismas réplicas y mismo m̂.
Uso: python corte_descomposicion.py <escenario> <R> <semilla_inicial>
"""
import os, sys, json
import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from scenarios import SCEN
from gen import s_star, g_and_pD
from truth import features
from est import LGB_CLF, LGB_REG
from corte_conocido import sample_all, metrics


def one(name, seed, cs, K=5):
    p, _, n = SCEN[name]
    d = sample_all(n, p, cs, np.random.default_rng(seed))
    Z, Y, client = d["Z"], d["Y"], d["client"]; cc = Z == 0
    X = features(d["V"], d["V5"]); g, _ = g_and_pD(d["V"], d["V5"], p)
    ph = np.zeros(len(Z)); mh = np.zeros(len(Z))
    for tr, te in GroupKFold(K).split(X, Z, client):
        ph[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
        trc = tr[cc[tr]]
        mh[te] = lgb.LGBMRegressor(**LGB_REG).fit(X[trc], Y[trc]).predict(X[te])
    ph = np.clip(ph, 1e-6, 1 - 1e-6)
    o = np.where(s_star(d["V"], p) <= cs, ph / (1 - ph), 0.0)
    return {"std_con_corte": metrics(o, Y, mh, Z, client, g)}


if __name__ == "__main__":
    name, R, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cs = json.load(open("c_star.json"))[name]
    from corte_descomposicion import one as one_mod
    res = Parallel(n_jobs=int(os.environ.get("N_JOBS", "2")))(delayed(one_mod)(name, seed0 + r, cs) for r in range(R))
    json.dump(res, open(f"res_corte_descomp_{name}_{R}.json", "w"))
