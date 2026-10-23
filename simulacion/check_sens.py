"""Comprobación de las cotas de sensibilidad (modelo marginal de Tan, cotas de Dorn y Guo).

Para un perfil x fijo, con Y discreta de ley q_k = P(Y = y_k | X = x, D = 0), el problema es
    max / min  sum_k q_k * lambda_k * y_k
    s.a.       sum_k q_k * lambda_k = 1,  1/L <= lambda_k <= L.
Se resuelve por programación lineal (fuerza bruta) y se compara con:
  - la fórmula cerrada para Y binaria;
  - el atajo para resultados raros: max = L * E[Y|x] y min = E[Y|x] / L si P(Y>0|x) <= 1/(1+L).
"""
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(8)


def lp(q, y, L, sentido):
    c = -(q * y) if sentido == "max" else (q * y)
    res = linprog(c, A_eq=[q], b_eq=[1.0], bounds=[(1 / L, L)] * len(q), method="highs")
    return (q * y) @ res.x


def binaria(m, L, sentido):
    a = 1 / (1 + L)
    if sentido == "max":
        t = min(m, a)
    else:
        t = max(0.0, m + a - 1)
    return L * t + (m - t) / L


err_bin, err_raro, n_raro = 0.0, 0.0, 0
for _ in range(2000):
    L = rng.uniform(1, 6)
    m = rng.uniform(0, 1)
    q = np.array([1 - m, m]); y = np.array([0.0, 1.0])
    for s in ("max", "min"):
        err_bin = max(err_bin, abs(lp(q, y, L, s) - binaria(m, L, s)))
    # Y de conteo {0,1,2} con P(Y>0) pequeña: atajo de resultados raros
    p0 = rng.uniform(1 - 1 / (1 + L), 1.0)
    r = rng.uniform(0, 1); q3 = np.array([p0, (1 - p0) * r, (1 - p0) * (1 - r)]); y3 = np.array([0.0, 1.0, 2.0])
    EY = q3 @ y3
    err_raro = max(err_raro, abs(lp(q3, y3, L, "max") - L * EY), abs(lp(q3, y3, L, "min") - EY / L))
    n_raro += 1
print(f"Fórmula binaria vs programación lineal: error máximo {err_bin:.2e} (2000 casos)")
print(f"Atajo de resultados raros vs programación lineal: error máximo {err_raro:.2e} ({n_raro} casos)")

# Contraejemplo: si el resultado no es raro, el atajo deja de valer
L, m = 3.0, 0.5
print(f"No raro (m=0.5, L=3): máximo exacto {binaria(m, L, 'max'):.4f} frente a L*m = {L*m:.4f}")
