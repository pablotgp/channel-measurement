"""Auditoría independiente de las cantidades 'verdaderas' de la simulación.

No usa mu0 ni g calculados por cuadratura para la verdad: sortea los resultados
potenciales con Bernoulli (como en las réplicas) en una población nueva (otra
semilla) y compara con lo que dio truth.population.
"""
import json
import numpy as np
from scipy.special import expit
from scenarios import SCEN
from gen import draw_clients, s_star, lin_Y, g_and_pD

pred = json.load(open("predicciones.json"))
cstar = json.load(open("c_star.json"))
rng = np.random.default_rng(424242)
N = 3_000_000

for name in ("base", "confunde_sin_C", "solap_bajo", "confusion_U"):
    p, _, _ = SCEN[name]
    c = draw_clients(N, p, rng)
    V, V5, H, U, D, e, m = c["V"], c["V5"], c["H"], c["U"], c["D"], c["e"], c["m"]
    s = s_star(V, p); S = D & (s <= cstar[name]); CC = ~D
    # (0) la regla fija deja ~10 % de las llamadas premium en la cola
    frac = m[S].sum() / m[D].sum()
    # (a) theta por sorteo directo de Y(0) para cada llamada de la cola
    idx = np.repeat(np.flatnonzero(S), m[S])
    y0 = np.zeros(len(idx))
    for k in (1, 2):
        y0 += rng.random(len(idx)) < expit(lin_Y(V[idx], k) + p.delta * H[idx] + p.nu_y * U[idx] + e[idx])
    th_mc, th_se = y0.mean(), y0.std() / np.sqrt(len(idx))
    # (b) identidad del Teorema 1 con Y sorteada en CC: E_Q[w Y] / E_Q[w] = E_P[g]
    jdx = np.repeat(np.flatnonzero(CC), m[CC])
    y = np.zeros(len(jdx))
    for k in (1, 2):
        y += rng.random(len(jdx)) < expit(lin_Y(V[jdx], k) + p.delta * H[jdx] + p.nu_y * U[jdx] + e[jdx])
    _, pD = g_and_pD(V[jdx], V5[jdx], p)
    w = np.where(s[jdx] <= cstar[name], pD / (1 - pD), 0.0)
    lim_mc = np.sum(w * y) / np.sum(w)
    # error de Monte Carlo aproximado de un cociente ponderado
    lim_se = np.sqrt(np.sum(w**2 * (y - lim_mc) ** 2)) / np.sum(w)
    # (c) tasa media en CC
    tasa_mc = y.mean()
    P = pred[name]
    print(f"\n{name}: fracción cola/premium = {frac:.4f} (objetivo 0,10)")
    print(f"  (a) theta sorteado {th_mc:.5f} ± {th_se:.5f}  vs verdad registrada {P['theta']:.5f}  z={(th_mc-P['theta'])/th_se:+.2f}")
    print(f"  (b) E_Q[wY]/E_Q[w] sorteado {lim_mc:.5f} ± {lim_se:.5f} vs límite registrado {pred[name]['lim'] if not SCEN[name][1] else 'n/a'}  z={(lim_mc-P['lim'])/lim_se:+.2f}")
    print(f"  (c) tasa CC sorteada {tasa_mc:.5f} vs registrada {P['tasa_CC']:.5f}  z={(tasa_mc-P['tasa_CC'])/(y.std()/np.sqrt(len(y))):+.2f}")
