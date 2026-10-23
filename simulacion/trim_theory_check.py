"""Comprobaciones numéricas para la Proposición 4b (recorte con pesos estimados).

(A) Fórmula exacta: para un peso fijo w_hat, el efecto del recorte sobre el Hájek
    normalizado es Cov_{P_hat}(m0, phi(w_hat)) / E_{P_hat}[phi(w_hat)], con
    dP_hat/dQ = w_hat / E_Q[w_hat]. Se comprueba por cuadratura en el ejemplo gaussiano.
(B) Ruido multiplicativo independiente log-normal: el signo se conserva (Efron 1965).
(C) Ruido que depende de X (no solo de w): el signo puede invertirse.
"""
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(5)
N = 4_000_000
# Ejemplo gaussiano: Q = N(0,1) en (x1, x2); w depende solo de x1; m0 depende de x1 y x2.
x1 = rng.standard_normal(N); x2 = rng.standard_normal(N)
mu = 1.0
w = np.exp(mu * x1 - mu**2 / 2)                 # dP/dQ verdadero, E_Q[w] = 1
m0 = expit(-3.5 - 0.8 * x1 + 0.6 * x2)          # decreciente en w (por x1)
theta = np.mean(w * m0)


def hajek(wt):
    return np.sum(wt * m0) / np.sum(wt)


def trim_effect(wh, q):
    tau = np.quantile(wh, q)
    return hajek(np.minimum(wh, tau)) - hajek(wh), tau


def formula(wh, tau):
    ph = wh / wh.sum()                         # P_hat (pesos normalizados sobre la muestra de Q)
    phi = np.minimum(1, tau / wh)
    Ephi = np.sum(ph * phi)
    cov = np.sum(ph * m0 * phi) - np.sum(ph * m0) * Ephi
    return cov / Ephi


print(f"theta = {theta:.5f}")
for nombre, wh in [
    ("w verdadero", w),
    ("ruido lognormal indep. (sd 0.5)", w * np.exp(0.5 * rng.standard_normal(N) - 0.125)),
    ("ruido lognormal indep. (sd 1.0)", w * np.exp(1.0 * rng.standard_normal(N) - 0.5)),
    ("encogimiento monótono w^0.6", w ** 0.6),
    ("error que depende de x2 (+0.8*x2)", w * np.exp(0.8 * x2)),
    ("error que depende de x2 (-0.8*x2)", w * np.exp(-0.8 * x2)),
]:
    eff, tau = trim_effect(wh, 0.90)
    print(f"{nombre:36s}: efecto recorte p90 = {eff:+.6f} | fórmula A = {formula(wh, tau):+.6f}")
