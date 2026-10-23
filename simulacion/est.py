"""Estimadores de la sección 7: Hájek y doblemente robusto (DR) con cross-fitting por cliente."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from truth import features

LGB_CLF = dict(n_estimators=150, learning_rate=0.08, num_leaves=15, min_child_samples=80,
               subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=1)
LGB_REG = dict(LGB_CLF, objective="poisson")


def _se_cluster(phi, client, n):
    s = np.bincount(np.unique(client, return_inverse=True)[1], weights=phi)
    return np.sqrt(np.sum(s ** 2)) / n


def _hajek_dr(o, Y, mh, Z, client):
    """Hájek y DR normalizado con su error típico agrupado por cliente (función de influencia)."""
    cc = Z == 0; n = len(Z); pi = Z.mean()
    B = np.sum(o[cc]) / n
    hj = np.sum(o[cc] * Y[cc]) / np.sum(o[cc])
    A = np.sum(o[cc] * (Y[cc] - mh[cc])) / n
    dr = A / B + mh[Z == 1].mean()
    phi = np.zeros(n)
    phi[cc] = o[cc] * (Y[cc] - mh[cc] - A / B) / B
    phi[~cc] = (mh[~cc] - mh[~cc].mean()) / pi
    return hj, dr, _se_cluster(phi, client, n)


def estimate(d, use_C, spec=False, K=5):
    Z, Y, client = d["Z"], d["Y"], d["client"]
    X = features(d["V"], d["V5"])
    if use_C:
        X = np.c_[X, d["C"]]                       # NaN = no registrado (LightGBM lo trata como ausente)
    cc = Z == 0
    p = np.zeros(len(Z)); mh = np.zeros(len(Z))
    for tr, te in GroupKFold(K).split(X, Z, client):
        clf = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
        trc = tr[cc[tr]]
        reg = lgb.LGBMRegressor(**LGB_REG).fit(X[trc], Y[trc])
        mh[te] = reg.predict(X[te])
    p = np.clip(p, 1e-6, 1 - 1e-6); o = p / (1 - p)

    r = {}
    r["hajek"], r["dr"], r["dr_se"] = _hajek_dr(o, Y, mh, Z, client)
    oc = o[cc]
    r["ess_ratio"] = oc.sum() ** 2 / (oc ** 2).sum() / cc.sum()
    r["eps_hat"] = np.mean(p[Z == 1] > 0.99)
    r["auc"] = roc_auc_score(Z, p)
    r["n0"] = int(cc.sum()); r["n1"] = int((Z == 1).sum())
    for q in (99, 95, 90):
        tau = np.quantile(oc, q / 100)
        ot = np.minimum(o, tau)
        r[f"hajek_trim{q}"], r[f"dr_trim{q}"], r[f"dr_trim{q}_se"] = _hajek_dr(ot, Y, mh, Z, client)

    if spec:   # especificaciones erróneas: logística lineal y resultado lineal en V1
        Xl = features(d["V"], d["V5"])
        ol = np.zeros(len(Z)); ml = np.zeros(len(Z))
        for tr, te in GroupKFold(K).split(Xl, Z, client):
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xl[tr], Z[tr])
            pl = lr.predict_proba(Xl[te])[:, 1]; ol[te] = pl / (1 - pl)
            trc = tr[cc[tr]]
            A = np.c_[np.ones(len(trc)), d["V"][trc, 0]]
            coef = np.linalg.lstsq(A, Y[trc], rcond=None)[0]
            ml[te] = coef[0] + coef[1] * d["V"][te, 0]
        r["hajek_lin"], r["dr_linw_goodm"], r["dr_linw_goodm_se"] = _hajek_dr(ol, Y, mh, Z, client)
        _, r["dr_goodw_linm"], r["dr_goodw_linm_se"] = _hajek_dr(o, Y, ml, Z, client)
        _, r["dr_both_bad"], r["dr_both_bad_se"] = _hajek_dr(ol, Y, ml, Z, client)
        r["reg_lin"] = ml[Z == 1].mean()
    return r
