"""Compara el estimador con corte conocido con el estándar, en las mismas réplicas.

Uso: python analyze_corte.py   (lee res_corte_<escenario>_200.json que existan)
"""
import json, os
import numpy as np

pred = json.load(open("predicciones.json"))
print("Predicción registrada:", json.load(open("prediccion_corte_conocido.json"))["registrada"])
for name in ("base", "n_10000", "n_pequeno"):
    f = f"res_corte_{name}_200.json"
    if not os.path.exists(f):
        continue
    R = json.load(open(f)); th = pred[name]["theta"]; n = len(R)
    print(f"\n== {name} ({n} réplicas), θ = {th*100:.3f} %")
    for m in ("std", "corte"):
        c = lambda k: np.array([r[m][k] for r in R])
        h, d, se = c("hajek"), c("dr"), c("dr_se")
        z = lambda x: (x.mean() - th) / (x.std(ddof=1) / np.sqrt(n))
        cov = np.mean(np.abs(d - th) <= 1.96 * se)
        print(f"  {m:6s} Hájek {h.mean()*100:.3f} (z {z(h):+.1f})  DR {d.mean()*100:.3f} (z {z(d):+.1f})  "
              f"DE DR {d.std(ddof=1)*100:.3f}  RECM {np.sqrt(np.mean((d-th)**2))*100:.3f}  "
              f"EE/DE {se.mean()/d.std(ddof=1):.2f}  cobertura {cov*100:.1f} % (± {np.sqrt(cov*(1-cov)/n)*100:.1f})  "
              f"n_eff/n0 {c('ess_ratio').mean():.3f}")
        effs = []
        for q in (99, 95, 90):
            e = c(f"eff_g{q}") * 1e5
            effs.append(f"p{q} {e.mean():+.2f} ± {e.std(ddof=1)/np.sqrt(n):.2f} ({np.mean(e > 0)*100:.0f} % > 0)")
        print("         recorte del Hájek con g (milésimas): " + "; ".join(effs))
    dd = np.array([r["corte"]["dr"] - r["std"]["dr"] for r in R]) * 100
    print(f"  diferencia pareada DR corte − estándar: {dd.mean():+.4f} ± {dd.std(ddof=1)/np.sqrt(n):.4f} pp")
