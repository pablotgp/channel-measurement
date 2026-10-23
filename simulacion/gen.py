"""Generador de la simulación (sección 9 del borrador).

Unidad = cliente; cada cliente hace entre 1 y 5 llamadas. Covariables, historial,
canal y U son del cliente; el resultado se sortea por llamada con un efecto
aleatorio de cliente e ~ N(0, sd_e^2).

Todas las cantidades "verdaderas" (mu0, g, P(D=1|V), w) se calculan de forma
analítica integrando U y e con cuadratura de Gauss-Hermite.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit

_x, _w = np.polynomial.hermite_e.hermegauss(12)   # nodos para N(0,1)
GH_X, GH_W = _x, _w / _w.sum()

SIGMA = np.array([[0.3 ** abs(i - j) for j in range(4)] for i in range(4)])
CHOL = np.linalg.cholesky(SIGMA)


@dataclass(frozen=True)
class Params:
    lam: float = 1.0      # solapamiento: escala del índice de canal en V
    zeta: float = 1.0     # efecto del historial real H en el canal
    delta: float = 0.0    # efecto del historial real H en la compra (confusión vía H)
    nu: float = 0.0       # efecto de U (no observada) en el canal
    nu_y: float = 0.0     # efecto de U en la compra
    alpha0: float = -1.4  # intercepto del canal
    sd_e: float = 0.5     # efecto aleatorio de cliente en la compra
    uplift: float = 1.0   # efecto del canal premium en la compra (log-odds)


# ---------- piezas del modelo -------------------------------------------------
def draw_clients(n, p: Params, rng):
    V = rng.standard_normal((n, 4)) @ CHOL.T
    lo = np.c_[np.zeros(n), 0.6 * V[:, 0], -0.6 * V[:, 0]]
    pr = np.exp(lo); pr /= pr.sum(1, keepdims=True)
    V5 = (rng.random(n)[:, None] > pr.cumsum(1)).sum(1)          # 0,1,2
    H = rng.random(n) < p_hist(V)
    Cval = np.where(H, 0.5 * V[:, 0] + rng.standard_normal(n), np.nan)
    U = rng.standard_normal(n)
    D = rng.random(n) < expit(lin_D(V, V5, H, U, p))
    e = p.sd_e * rng.standard_normal(n)
    m = rng.integers(1, 6, n)                                      # llamadas por cliente
    return dict(V=V, V5=V5, H=H, Cval=Cval, U=U, D=D, e=e, m=m)


def p_hist(V):
    return expit(-0.4 + 0.8 * V[:, 0] + 0.3 * V[:, 2])


def lin_D(V, V5, H, U, p):
    idx = 1.0 * V[:, 0] + 0.5 * V[:, 1] - 0.5 * V[:, 2] + 0.3 * (V5 == 1)
    return p.alpha0 + p.lam * idx + p.zeta * H + p.nu * U


def lin_Y(V, k):
    """Índices lineales de los productos comunes k=1,2 (sin canal, H, U ni e)."""
    if k == 1:
        return -4.6 + 0.6 * V[:, 0] + 0.2 * V[:, 1] + 0.3 * V[:, 3]
    return -5.0 + 0.3 * V[:, 0] + 0.4 * V[:, 2] - 0.2 * V[:, 3]


def lin_Y3(V):
    """Producto exclusivo del canal premium (solo existe Y3(1))."""
    return -3.8 + 0.5 * V[:, 0]


def _Ee_expit(z, sd_e):
    """E_e[expit(z + e)], e ~ N(0, sd_e^2), por cuadratura; z de cualquier forma."""
    return (expit(z[..., None] + sd_e * GH_X) * GH_W).sum(-1)


def mu0(V, H, U, p):
    """E[Y(0) | V, H, U] integrando e: compra esperada en el canal básico (productos 1 y 2)."""
    out = 0
    for k in (1, 2):
        out = out + _Ee_expit(lin_Y(V, k) + p.delta * H + p.nu_y * U, p.sd_e)
    return out


def s_star(V, p):
    """Índice verdadero de propensión de compra en el canal premium (función fija de V)."""
    out = _Ee_expit(lin_Y3(V), p.sd_e)
    for k in (1, 2):
        out = out + _Ee_expit(lin_Y(V, k) + p.uplift, p.sd_e)
    return out


def g_and_pD(V, V5, p, chunk=200_000):
    """g(v) = E[Y | V=v, D=0] y P(D=1 | V=v), integrando H (binaria) y U (cuadratura)."""
    n = len(V); g = np.empty(n); pD = np.empty(n)
    for s in range(0, n, chunk):
        Vc, V5c = V[s:s + chunk], V5[s:s + chunk]; nc = len(Vc)
        ph = p_hist(Vc)
        num = np.zeros(nc); den = np.zeros(nc); pd1 = np.zeros(nc)
        for h, wh in ((0, 1 - ph), (1, ph)):
            Hc = np.full(nc, h)
            for u, wu in zip(GH_X, GH_W):
                Uc = np.full(nc, u)
                pd = expit(lin_D(Vc, V5c, Hc, Uc, p))
                m = mu0(Vc, Hc, Uc, p)
                wgt = wh * wu
                num += wgt * (1 - pd) * m
                den += wgt * (1 - pd)
                pd1 += wgt * pd
        g[s:s + chunk] = num / den; pD[s:s + chunk] = pd1
    return g, pD


def sample_calls(n_clients, p: Params, c_star, rng):
    """Una réplica: llamadas del grupo objetivo (Z=1) y del canal básico (Z=0).

    Devuelve el resultado Y (compras en el canal básico) solo para Z=0; C registrado
    solo si D=1 y H=1.
    """
    c = draw_clients(n_clients, p, rng)
    S = c["D"] & (s_star(c["V"], p) <= c_star)
    keep = S | ~c["D"]
    idx = np.repeat(np.flatnonzero(keep), c["m"][keep])          # expandir a llamadas
    Z = S[idx].astype(int)
    V, V5, H, U, e = c["V"][idx], c["V5"][idx], c["H"][idx], c["U"][idx], c["e"][idx]
    C = np.where((Z == 1) & H, c["Cval"][idx], np.nan)
    Y = np.zeros(len(idx))
    cc = Z == 0
    for k in (1, 2):
        pr = expit(lin_Y(V[cc], k) + p.delta * H[cc] + p.nu_y * U[cc] + e[cc])
        Y[cc] += rng.random(cc.sum()) < pr
    return dict(client=idx, Z=Z, V=V, V5=V5, C=C, Y=Y, H=H)
