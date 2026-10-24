"""Compara el bootstrap por cliente con el intervalo de Wald en las mismas réplicas.

Uso: python analyze_bootstrap.py [escenario] [carpeta]
  escenario: n_10000 (por defecto) o base
  carpeta:   donde están los ficheros boot_<escenario>_parte*.json (por defecto, la actual)
"""
import json, glob, os, sys
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else "n_10000"
folder = sys.argv[2] if len(sys.argv) > 2 else "."
PRED = {"n_10000": "prediccion_bootstrap.json", "base": "prediccion_bootstrap_base.json",
        "base_bayes": "prediccion_bootstrap_bayesiano.json", "n_10000_bayes": "prediccion_bootstrap_bayesiano.json"}
base_name = name.replace("_bayes", "")

th = json.load(open("predicciones.json"))[base_name]["theta"]
files = sorted(glob.glob(os.path.join(folder, f"boot_{name}_parte*.json")))
if not files:
    sys.exit(f"No hay ficheros boot_{name}_parte*.json en {folder}")
res = []
for f in files:
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
try:
    sd200 = np.std([r["dr"] for r in json.load(open(f"res_{base_name}_200.json"))], ddof=1)
    sd200_txt = f"{sd200*100:.3f}"
except FileNotFoundError:
    sd200_txt = "no disponible"
print(f"Escenario: {name}; ficheros: {len(files)}")
print(f"Réplicas: {R}; remuestreos por réplica: {boot.shape[1]}; θ = {th*100:.3f} %")
print(f"DE real del DR en estas réplicas: {est.std(ddof=1)*100:.3f} pp (en las 200 del escenario: {sd200_txt})")
print(f"EE medio de Wald: {se.mean()*100:.3f} pp;  EE medio bootstrap: {se_b.mean()*100:.3f} pp")
print(f"Media de las réplicas bootstrap menos la estimación (sesgo bootstrap medio): {(boot.mean(axis=1) - est).mean()*100:+.3f} pp")
for k, c in cov.items():
    print(f"  {k:24s}: cobertura {c.mean()*100:.1f} % ± {mc(c)*100:.1f}")
w, b = cov["Wald (EE agrupado)"], cov["Wald con EE bootstrap"]
print(f"Réplicas que cubre el bootstrap y no Wald: {np.sum(b & ~w)}; al revés: {np.sum(w & ~b)}")
lo_miss = np.mean(th < lo_p); hi_miss = np.mean(th > hi_p)
print(f"Percentil: θ queda por debajo del intervalo en {lo_miss*100:.1f} % y por encima en {hi_miss*100:.1f} %")
if name in PRED and os.path.exists(PRED[name]):
    print("\nPredicción registrada antes de ejecutar:\n  " + json.load(open(PRED[name]))["prediccion"])
