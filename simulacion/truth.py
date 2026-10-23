"""Verdad y predicciones analíticas por escenario, calculadas en una población grande.

Todas las medias se ponderan por el número de llamadas m de cada cliente, de modo
que coinciden con los estimandos a nivel de llamada.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from gen import Params, draw_clients, s_star, mu0, g_and_pD


def wmean(x, mask, m):
    return np.sum(x[mask] * m[mask]) / np.sum(m[mask])


def wquantile(x, q, wt):
    o = np.argsort(x); cw = np.cumsum(wt[o]) / wt.sum()
    return x[o][np.searchsorted(cw, q)]


def features(V, V5):
    return np.c_[V, V5 == 1, V5 == 2].astype(float)


def population(p: Params, n=1_500_000, seed=0):
    rng = np.random.default_rng(seed)
    c = draw_clients(n, p, rng)
    V, V5, H, U, D, m = c["V"], c["V5"], c["H"], c["U"], c["D"], c["m"]
    s = s_star(V, p)
    c_star = wquantile(s[D], 0.10, m[D])
    S = D & (s <= c_star)
    CC = ~D
    mu = mu0(V, H, U, p)
    g, pD = g_and_pD(V, V5, p)

    out = dict(c_star=c_star, P_D1=wmean(D.astype(float), np.ones(n, bool), m),
               frac_S_in_D=wmean(S.astype(float), D, m),
               pi=np.sum(m[S]) / np.sum(m[S | CC]),
               n_calls_per_client=m.mean())
    out["theta"] = wmean(mu, S, m)                       # verdad
    out["theta_V"] = wmean(g, S, m)                      # límite sin C
    out["kappa"] = wmean(H.astype(float), S, m)
    out["theta_C"] = wmean(g, S & ~H, m)                 # límite con C
    out["theta_ac_real"] = wmean(mu, S & ~H, m)          # verdad de la parte sin dato
    out["tasa_CC"] = wmean(g, CC, m)

    # dP_V/dQ_V (normalizado para que E_Q[w] = 1)
    w = np.where(s <= c_star, pD / (1 - pD), 0.0)
    w = w / wmean(w, CC, m)
    out["check_EQw_g"] = wmean(w * g, CC, m)             # debe ser theta_V
    out["chi2"] = wmean(w ** 2, CC, m) - 1
    out["ess_ratio"] = 1 / (1 + out["chi2"])
    lab = np.r_[np.ones(S.sum()), np.zeros(CC.sum())]
    out["auc"] = roc_auc_score(lab, np.r_[w[S], w[CC]], sample_weight=np.r_[m[S], m[CC]])

    # recorte en cuantiles de w sobre las llamadas del canal básico
    for q in (0.99, 0.95, 0.90):
        tau = wquantile(w[CC], q, m[CC]); wt = np.minimum(w, tau)
        th_t = wmean(wt * g, CC, m) / wmean(wt, CC, m)
        phi = np.minimum(1, tau / np.where(w > 0, w, np.inf))
        cov = wmean(g * phi, S, m) - out["theta_V"] * wmean(phi, S, m)
        out[f"theta_trim{int(q*100)}"] = th_t
        out[f"trimcov{int(q*100)}"] = out["theta_V"] + cov / wmean(phi, S, m)

    # especificaciones erróneas: clasificador logístico lineal y modelo de resultado lineal en V1
    sub = rng.random(n) < min(1.0, 400_000 / n)
    idx = np.flatnonzero((S | CC) & sub)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(features(V[idx], V5[idx]), S[idx], sample_weight=m[idx])
    pl = lr.predict_proba(features(V, V5))[:, 1]; ol = pl / (1 - pl)
    A = np.c_[np.ones(CC.sum()), V[CC, 0]]
    coef = np.linalg.lstsq(A * np.sqrt(m[CC])[:, None], g[CC] * np.sqrt(m[CC]), rcond=None)[0]
    mbar = coef[0] + coef[1] * V[:, 0]
    out["lim_hajek_lin"] = wmean(ol * g, CC, m) / wmean(ol, CC, m)
    out["lim_dr_both_bad"] = wmean(ol * (g - mbar), CC, m) / wmean(ol, CC, m) + wmean(mbar, S, m)
    out["lim_reg_lin"] = wmean(mbar, S, m)               # solo modelo de resultado malo
    return out


if __name__ == "__main__":
    import sys, json
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    r = population(Params(lam=lam), n=600_000)
    print(json.dumps({k: round(float(v), 6) for k, v in r.items()}, indent=1))
