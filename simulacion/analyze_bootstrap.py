"""Compara el bootstrap por cliente con el intervalo de Wald en las mismas réplicas."""
import json, glob
import numpy as np

pred = json.load(open("prediccion_bootstrap.json"))
th = json.load(open("predicciones.json"))["n_10000"]["theta"]
res = []
for f in sorted(glob.glob("boot_n_10000_parte*.json")):
    res += json.load(open(f))
R = len(res)
est = np.array([r["est"] for r in res]); se = np.array([r["se"] for r in res])
boot = np.array([r["boot"] for r in res])
se_b = boot.std(axis=1, ddof=1)
lo_p, hi_p = np.quantile(boot, 0.025, axis=1), np.quantile(boot, 0.975, axis=1)
cov = {
    "Wald (EE agrupado)": np.abs(est - th) <= 1.96 * se,
    "Wald con EE bootstrap": np.abs(est - th) <= 1.96 * se_b,
    "Percentil bootstrap": (lo_p <= th) & (th <= hi_p),
}
mc = lambda c: np.sqrt(c.mean() * (1 - c.mean()) / R)
print(f"Réplicas: {R}; remuestreos por réplica: {boot.shape[1]}; θ = {th*100:.3f} %")
print(f"DE real del DR en estas réplicas: {est.std(ddof=1)*100:.3f} pp (en las 200 del escenario: 0,294)")
print(f"EE medio de Wald: {se.mean()*100:.3f} pp;  EE medio bootstrap: {se_b.mean()*100:.3f} pp")
print(f"Media de las réplicas bootstrap menos la estimación (sesgo bootstrap medio): {(boot.mean(axis=1) - est).mean()*100:+.3f} pp")
for k, c in cov.items():
    print(f"  {k:24s}: cobertura {c.mean()*100:.1f} % ± {mc(c)*100:.1f}")
w, b = cov["Wald (EE agrupado)"], cov["Wald con EE bootstrap"]
print(f"Réplicas que cubre el bootstrap y no Wald: {np.sum(b & ~w)}; al revés: {np.sum(w & ~b)}")
lo_miss = np.mean(th < lo_p); hi_miss = np.mean(th > hi_p)
print(f"Percentil: θ queda por debajo del intervalo en {lo_miss*100:.1f} % y por encima en {hi_miss*100:.1f} %")
