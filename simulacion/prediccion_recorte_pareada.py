"""Predicción del efecto del recorte calculada como diferencia pareada en la población.

Problema detectado en el repaso del 24-10-2026: en predicciones.json, el efecto predicho del
recorte se obtenía como lim_trim - theta, donde lim_trim se calcula sobre las llamadas
donantes de la población y theta sobre las del grupo objetivo. Son dos subconjuntos finitos
distintos, y la diferencia entre E_Q[w g] y E_P[g] en la población (unas 0,9 milésimas de
punto) contamina un efecto que solo mide entre 0,4 y 20 milésimas.

Aquí se calcula de dos formas pareadas, sin ese error:
  (a) Hájek recortado - Hájek sin recortar, ambos sobre los donantes de la población;
  (b) la fórmula de la Proposición 4, Cov_P(g, phi)/E_P[phi], sobre el grupo objetivo.
Se usan dos poblaciones independientes (semillas 11, la original, y 0).
"""
import json
from truth import population
from scenarios import SCEN

out = {}
for name in ("base", "solap_bajo"):
    p, _, _ = SCEN[name]
    for seed in (11, 0):
        r = population(p, n=1_500_000, seed=seed)
        for q in (99, 95, 90):
            out[f"{name}_s{seed}_p{q}"] = dict(
                original=(r[f"theta_trim{q}"] - r["theta"]) * 1e5,
                pareada_donantes=(r[f"theta_trim{q}"] - r["check_EQw_g"]) * 1e5,
                formula_prop4=(r[f"trimcov{q}"] - r["theta_V"]) * 1e5)
        print(f"{name}, población semilla {seed}: E_Q[w g] - theta = {(r['check_EQw_g'] - r['theta'])*1e5:+.2f} milésimas")
for k, v in out.items():
    print(f"{k:20s} original {v['original']:+7.2f}   pareada {v['pareada_donantes']:+7.2f}   fórmula {v['formula_prop4']:+7.2f}")
json.dump(out, open("prediccion_recorte_pareada.json", "w"), indent=1)
