"""Comprobación del Teorema 5(c): medición distinta por canal con soporte.

Variable real X* (p. ej. saldo). Donante: X* ~ Q* = N(0,1); grupo objetivo: X* ~ P* = N(1,1).
Respuesta en el canal básico: mu*(x*) = expit(-3 - 0.8 x*).
Medición: el canal básico registra X = X* (phi0 = identidad); el premium registra X = X* + c (phi1).
Predicción del teorema: el estimador converge a E_{P*}[ mu*( phi0^{-1}(phi1(X*)) ) ] = E_{P*}[ mu*(X* + c) ],
no a theta = E_{P*}[ mu*(X*) ]. El soporte se cumple (normales) y no hay p = 1.
"""
import numpy as np
from scipy import integrate, stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

mu_star = lambda x: expit(-3 - 0.8 * x)
theta = integrate.quad(lambda x: mu_star(x) * stats.norm.pdf(x - 1), -12, 12)[0]
rng = np.random.default_rng(3)
n0, n1 = 400_000, 100_000
for c in (0.0, 0.3, -0.3):
    pred = integrate.quad(lambda x: mu_star(x + c) * stats.norm.pdf(x - 1), -12, 12)[0]
    est = []
    for r in range(20):
        x0 = rng.standard_normal(n0); y0 = (rng.random(n0) < mu_star(x0)).astype(float)
        x1 = rng.normal(1, 1, n1) + c                     # medido con el instrumento premium
        X = np.r_[x0, x1][:, None]; Z = np.r_[np.zeros(n0), np.ones(n1)]
        p = LogisticRegression(C=1e6).fit(X, Z).predict_proba(x0[:, None])[:, 1]
        o = p / (1 - p)
        est.append(np.sum(o * y0) / np.sum(o))
    est = np.array(est); se = est.std(ddof=1) / np.sqrt(len(est))
    print(f"c={c:+.1f}: theta={theta:.5f}  predicción={pred:.5f}  estimador={est.mean():.5f} ± {se:.5f}  "
          f"z(vs predicción)={(est.mean()-pred)/se:+.2f}  sesgo={est.mean()-theta:+.5f}  AUC~igual, p<1 en todo punto")
