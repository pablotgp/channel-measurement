"""Bootstrap por cliente del estimador doblemente robusto (DR), re-estimando los modelos auxiliares.

Por cada réplica: se generan los datos, se calcula el DR y su intervalo de Wald (error típico
agrupado), y B remuestreos por cliente con reemplazo; en cada remuestreo se repite el estimador
completo (LightGBM + cross-fitting de 5 bloques). Las copias de un mismo cliente comparten grupo,
para que nunca queden en bloques distintos del cross-fitting.

Uso: python bootstrap.py <escenario> <R> <B> <semilla_inicial>
"""
import os, sys, json, time
import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from scenarios import SCEN
from gen import sample_calls
from truth import features
from est import LGB_CLF, LGB_REG, _hajek_dr


def dr(X, Z, Y, groups, K=5):
    cc = Z == 0
    p = np.zeros(len(Z)); mh = np.zeros(len(Z))
    for tr, te in GroupKFold(K).split(X, Z, groups):
        p[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
        trc = tr[cc[tr]]
        mh[te] = lgb.LGBMRegressor(**LGB_REG).fit(X[trc], Y[trc]).predict(X[te])
    p = np.clip(p, 1e-6, 1 - 1e-6); o = p / (1 - p)
    _, est, se = _hajek_dr(o, Y, mh, Z, groups)
    return est, se


def one(name, seed, B, cs):
    p, useC, n = SCEN[name]
    rng = np.random.default_rng(seed)
    d = sample_calls(n, p, cs, rng)
    X = features(d["V"], d["V5"]); Z, Y, cl = d["Z"], d["Y"], d["client"]
    est, se = dr(X, Z, Y, cl)
    uniq = np.unique(cl)
    order = np.argsort(cl, kind="stable"); cl_sorted = cl[order]
    starts = np.searchsorted(cl_sorted, uniq, "left"); ends = np.searchsorted(cl_sorted, uniq, "right")
    boot = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        g = np.repeat(np.arange(len(pick)), ends[pick] - starts[pick])   # copia a la que pertenece cada fila
        boot[b] = dr(X[idx], Z[idx], Y[idx], uniq[pick][g])[0]
    return dict(est=float(est), se=float(se), boot=boot.tolist())


if __name__ == "__main__":
    name, R, B, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    cs = json.load(open("c_star.json"))[name]
    t = time.time()
    res = Parallel(n_jobs=int(os.environ.get("N_JOBS", "2")))(delayed(one)(name, seed0 + r, B, cs) for r in range(R))
    json.dump(res, open(f"boot_{name}_{R}x{B}.json", "w"))
    print(f"{name}: {R} réplicas × {B} remuestreos en {time.time()-t:.0f} s")
