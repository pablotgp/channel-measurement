"""Compara el estimador con corte conocido con el estándar, en las mismas réplicas.

Uso: python analyze_corte.py   (lee los res_corte_<escenario>_200.json que existan)
"""
import json, os
import numpy as np

ESCENARIOS = ("base", "n_10000", "n_pequeno", "con_C", "confunde_sin_C", "confunde_con_C",
              "solap_medio", "solap_bajo", "confusion_U")
# Límite predicho del DR cuando difiere del del Hájek (Corolario 2, prediccion_DR_sin_soporte.json)
DR_PRED = {"con_C": 0.009575447347237531, "confunde_con_C": 0.013139808831507744}


def resumen(name, R):
    pred = json.load(open("predicciones.json"))[name]
    th, lh = pred["theta"], pred["lim"]; ld = DR_PRED.get(name, lh); n = len(R)
    out = [f"== {name} ({n} réplicas), θ = {th*100:.3f} %, límite predicho Hájek {lh*100:.3f} %, DR {ld*100:.3f} %"]
    for m in ("std", "corte"):
        c = lambda k: np.array([r[m][k] for r in R])
        h, d, se = c("hajek"), c("dr"), c("dr_se")
        z = lambda x, l: (x.mean() - l) / (x.std(ddof=1) / np.sqrt(n))
        cov = np.mean(np.abs(d - th) <= 1.96 * se)
        eps = f"  ε̂ {c('eps_hat').mean():.3f}" if "eps_hat" in R[0][m] else ""
        out.append(f"  {m:6s} Hájek {h.mean()*100:.3f} (z {z(h, lh):+.1f})  DR {d.mean()*100:.3f} (z {z(d, ld):+.1f})  "
                   f"DE DR {d.std(ddof=1)*100:.3f}  EE/DE {se.mean()/d.std(ddof=1):.2f}  "
                   f"cobertura {cov*100:.1f} % (± {np.sqrt(cov*(1-cov)/n)*100:.1f})  n_eff/n0 {c('ess_ratio').mean():.3f}{eps}")
        effs = []
        for q in (99, 95, 90):
            e = c(f"eff_g{q}") * 1e5
            effs.append(f"p{q} {e.mean():+.2f} ± {e.std(ddof=1)/np.sqrt(n):.2f}")
        out.append("         recorte del Hájek con g (milésimas): " + "; ".join(effs))
    dd = np.array([r["corte"]["dr"] - r["std"]["dr"] for r in R]) * 100
    out.append(f"  diferencia pareada DR corte − estándar: {dd.mean():+.4f} ± {dd.std(ddof=1)/np.sqrt(n):.4f} pp")
    return "\n".join(out)


if __name__ == "__main__":
    print("Predicciones registradas:", json.load(open("prediccion_corte_conocido.json"))["registrada"],
          "y", json.load(open("prediccion_corte_escenarios.json"))["registrada"])
    for name in ESCENARIOS:
        f = f"res_corte_{name}_200.json"
        if os.path.exists(f):
            print("\n" + resumen(name, json.load(open(f))))
