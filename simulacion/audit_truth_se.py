"""Error de Monte Carlo de la 'verdad' y de las predicciones (población finita).

1) En la población original (semilla 11), error típico agrupado por cliente de theta y del
   límite theta_V = media de g en la cola.
2) Recalcula ambos en una población independiente (semilla 99) para ver su estabilidad.
"""
import json
import numpy as np
from scenarios import SCEN
from gen import draw_clients, s_star, mu0, g_and_pD

cstar = json.load(open("c_star.json"))
pred = json.load(open("predicciones.json"))
out = {}
for name in ("base", "confunde_sin_C", "solap_medio", "solap_bajo", "confusion_U"):
    p, _, _ = SCEN[name]
    res = []
    for seed in (11, 99):
        rng = np.random.default_rng(seed)
        c = draw_clients(1_500_000, p, rng)
        V, V5, H, U, D, m = c["V"], c["V5"], c["H"], c["U"], c["D"], c["m"]
        S = D & (s_star(V, p) <= cstar[name])
        mu = mu0(V[S], H[S], U[S], p); g, _ = g_and_pD(V[S], V5[S], p); ms = m[S]
        th = np.sum(ms * mu) / ms.sum(); lim = np.sum(ms * g) / ms.sum()
        se_th = np.sqrt(np.sum(ms**2 * (mu - th) ** 2)) / ms.sum()
        se_lim = np.sqrt(np.sum(ms**2 * (g - lim) ** 2)) / ms.sum()
        res.append((th, se_th, lim, se_lim))
    (t1, s1, l1, sl1), (t2, s2, l2, sl2) = res
    print(f"{name:15s} theta: {t1*100:.4f} (EE {s1*100:.4f}) | otra población {t2*100:.4f}  "
          f"|| límite: {l1*100:.4f} (EE {sl1*100:.4f}) | otra {l2*100:.4f}   [registrado θ={pred[name]['theta']*100:.4f}, lim={pred[name]['lim']*100:.4f}]")
    out[name] = dict(se_theta=s1, se_lim=sl1)
json.dump(out, open("se_verdad.json", "w"), indent=1)
