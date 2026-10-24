"""Estimador con corte conocido frente al estándar, en las mismas réplicas.

Si S = D·1{s(X) <= c} con s conocida, dP/dQ(x) ∝ 1{s(x) <= c}·e(x)/(1 - e(x)), con
e(x) = P(D = 1 | x). En vez del clasificador «cola frente a donantes», se estima e con
todas las llamadas (cola, canal básico y premium fuera de la cola) y el corte se pone a mano.

Para cada réplica se genera exactamente el mismo conjunto de datos que gen.sample_calls
(mismo orden de números aleatorios) y además las llamadas premium fuera de la cola, que la
simulación original descartaba. Los dos estimadores comparten el modelo de resultado m̂.

Uso: python corte_conocido.py <escenario> <R> <semilla_inicial>
"""
import os, sys, json, time
import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from scenarios import SCEN
from gen import draw_clients, s_star, lin_Y, g_and_pD
from truth import features
from est import LGB_CLF, LGB_REG, _hajek_dr

QS = (99, 95, 90)


def sample_all(n, p, cs, rng):
    """Igual que gen.sample_calls, más las llamadas premium fuera del grupo objetivo."""
    c = draw_clients(n, p, rng)
    inR = s_star(c["V"], p) <= cs
    S = c["D"] & inR
    keep = S | ~c["D"]
    idx = np.repeat(np.flatnonzero(keep), c["m"][keep])
    Z = S[idx].astype(int)
    V, V5, H, U, e = c["V"][idx], c["V5"][idx], c["H"][idx], c["U"][idx], c["e"][idx]
    Y = np.zeros(len(idx)); cc = Z == 0
    for k in (1, 2):
        pr = expit(lin_Y(V[cc], k) + p.delta * H[cc] + p.nu_y * U[cc] + e[cc])
        Y[cc] += rng.random(cc.sum()) < pr
    out = c["D"] & ~inR                                   # premium fuera de la cola (sin usar rng)
    idx1 = np.repeat(np.flatnonzero(out), c["m"][out])
    C = np.where((Z == 1) & H, c["Cval"][idx], np.nan)            # como gen.sample_calls
    C_1 = np.where(c["H"][idx1], c["Cval"][idx1], np.nan)         # premium: se registra si hay historial
    return dict(client=idx, Z=Z, V=V, V5=V5, Y=Y, C=C, client1=idx1, V_1=c["V"][idx1], V5_1=c["V5"][idx1], C_1=C_1)


def metrics(o, Y, mh, Z, client, g):
    cc = Z == 0
    r = {}
    r["hajek"], r["dr"], r["dr_se"] = _hajek_dr(o, Y, mh, Z, client)
    oc = o[cc]
    r["ess_ratio"] = oc.sum() ** 2 / (oc ** 2).sum() / cc.sum()
    hj = lambda a, resp: np.sum(a * resp) / np.sum(a)
    for q in QS:
        tau = np.quantile(oc, q / 100); ot = np.minimum(o, tau)
        r[f"hajek_trim{q}"], r[f"dr_trim{q}"], r[f"dr_trim{q}_se"] = _hajek_dr(ot, Y, mh, Z, client)
        r[f"eff_g{q}"] = hj(np.minimum(oc, tau), g[cc]) - hj(oc, g[cc])    # recorte sin ruido de Y
    return r


def one(name, seed, cs, K=5):
    p, useC, n = SCEN[name]
    d = sample_all(n, p, cs, np.random.default_rng(seed))
    Z, Y, client = d["Z"], d["Y"], d["client"]; cc = Z == 0
    X = features(d["V"], d["V5"])
    if useC:
        X = np.c_[X, d["C"]]                       # NaN = no registrado, como en est.estimate
    g, _ = g_and_pD(d["V"], d["V5"], p)
    # estándar: idéntico a est.estimate
    ph = np.zeros(len(Z)); mh = np.zeros(len(Z))
    for tr, te in GroupKFold(K).split(X, Z, client):
        ph[te] = lgb.LGBMClassifier(**LGB_CLF).fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
        trc = tr[cc[tr]]
        mh[te] = lgb.LGBMRegressor(**LGB_REG).fit(X[trc], Y[trc]).predict(X[te])
    ph = np.clip(ph, 1e-6, 1 - 1e-6); o_std = ph / (1 - ph)
    # corte conocido: propensión de canal con todas las llamadas
    X1 = features(d["V_1"], d["V5_1"])
    if useC:
        X1 = np.c_[X1, d["C_1"]]
    XA = np.r_[X, X1]; DA = np.r_[Z, np.ones(len(X1), int)]; GA = np.r_[client, d["client1"]]
    eh = np.zeros(len(XA))
    for tr, te in GroupKFold(K).split(XA, DA, GA):
        eh[te] = lgb.LGBMClassifier(**LGB_CLF).fit(XA[tr], DA[tr]).predict_proba(XA[te])[:, 1]
    eh = np.clip(eh[:len(Z)], 1e-6, 1 - 1e-6)
    o_cut = np.where(s_star(d["V"], p) <= cs, eh / (1 - eh), 0.0)
    r = {"std": metrics(o_std, Y, mh, Z, client, g), "corte": metrics(o_cut, Y, mh, Z, client, g),
         "n_premium_fuera": int(len(X1))}
    r["std"]["eps_hat"] = float(np.mean(ph[Z == 1] > 0.99))       # masa sin soporte (clasificador seguro)
    r["corte"]["eps_hat"] = float(np.mean(eh[Z == 1] > 0.99))     # análogo: ê ≈ 1 en el grupo objetivo
    return r


if __name__ == "__main__":
    name, R, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cs = json.load(open("c_star.json"))[name]
    t = time.time()
    from corte_conocido import one as one_mod      # por referencia: loky no puede reconstruir funciones de __main__
    res = Parallel(n_jobs=int(os.environ.get("N_JOBS", "2")))(delayed(one_mod)(name, seed0 + r, cs) for r in range(R))
    json.dump(res, open(f"res_corte_{name}_{R}.json", "w"))
    print(f"{name}: {R} réplicas en {time.time()-t:.0f} s")


def _fit_corte(X, Z, Y, client, XA, DA, GA, nR, inR, wt=None, wtA=None, K=5):
    """DR con corte conocido; wt/wtA = pesos de cliente (bootstrap bayesiano) o None."""
    cc = Z == 0
    mh = np.zeros(len(Z)); eh = np.zeros(len(XA))
    for tr, te in GroupKFold(K).split(X, Z, client):
        trc = tr[cc[tr]]
        mh[te] = lgb.LGBMRegressor(**LGB_REG).fit(X[trc], Y[trc], sample_weight=None if wt is None else wt[trc]).predict(X[te])
    for tr, te in GroupKFold(K).split(XA, DA, GA):
        eh[te] = lgb.LGBMClassifier(**LGB_CLF).fit(XA[tr], DA[tr], sample_weight=None if wtA is None else wtA[tr]).predict_proba(XA[te])[:, 1]
    eh = np.clip(eh[:nR], 1e-6, 1 - 1e-6)
    o = np.where(inR, eh / (1 - eh), 0.0)
    if wt is None:
        _, est, se = _hajek_dr(o, Y, mh, Z, client)
        return est, se
    a = wt[cc] * o[cc]
    return np.sum(a * (Y[cc] - mh[cc])) / np.sum(a) + np.sum(wt[~cc] * mh[~cc]) / np.sum(wt[~cc]), None


def one_corte_bayes(name, seed, B, cs):
    """Bootstrap bayesiano (pesos Exp(1) por cliente) del estimador con corte conocido."""
    p, _, n = SCEN[name]
    d = sample_all(n, p, cs, np.random.default_rng(seed))
    Z, Y, client = d["Z"], d["Y"], d["client"]
    X = features(d["V"], d["V5"]); X1 = features(d["V_1"], d["V5_1"])
    XA = np.r_[X, X1]; DA = np.r_[Z, np.ones(len(X1), int)]; GA = np.r_[client, d["client1"]]
    inR = s_star(d["V"], p) <= cs
    est, se = _fit_corte(X, Z, Y, client, XA, DA, GA, len(Z), inR)
    brng = np.random.default_rng([seed, 2])
    boot = np.empty(B)
    for b in range(B):
        wc = brng.exponential(1.0, n); wc /= wc.mean()          # un peso por cliente de la población
        boot[b] = _fit_corte(X, Z, Y, client, XA, DA, GA, len(Z), inR, wt=wc[client], wtA=wc[GA])[0]
    return dict(est=float(est), se=float(se), boot=boot.tolist())
