"""Escenarios de la simulación y cálculo de predicciones (antes de simular)."""
import json
from gen import Params

BASE = Params(lam=0.5)
# nombre: (parámetros, usa C, n_clientes por réplica)
N_BASE = 30_000
SCEN = {
    "base":            (BASE, False, N_BASE),
    "con_C":           (BASE, True, N_BASE),
    "confunde_sin_C":  (Params(lam=0.5, delta=1.2), False, N_BASE),
    "confunde_con_C":  (Params(lam=0.5, delta=1.2), True, N_BASE),
    "solap_medio":     (Params(lam=1.5), False, N_BASE),
    "solap_bajo":      (Params(lam=3.0), False, N_BASE),
    "confusion_U":     (Params(lam=0.5, nu=1.0, nu_y=1.0), False, N_BASE),
    "n_pequeno":       (BASE, False, N_BASE // 10),
    "n_10000":         (BASE, False, 10_000),
}

if __name__ == "__main__":
    from truth import population
    pops = {}
    pred = {}
    for name, (p, useC, n) in SCEN.items():
        if p not in pops:
            pops[p] = population(p, n=1_500_000, seed=11)
        r = pops[p]
        pr = dict(theta=r["theta"], chi2=r["chi2"], ess_ratio=r["ess_ratio"], auc=r["auc"],
                  pi=r["pi"], kappa=r["kappa"], tasa_CC=r["tasa_CC"])
        pr["lim"] = r["theta_C"] if useC else r["theta_V"]          # límite de Hájek y DR
        pr["eps"] = r["kappa"] if useC else 0.0
        for q in (99, 95, 90):
            pr[f"lim_trim{q}"] = r[f"theta_trim{q}"]                # Hájek recortado (sin C)
        pr["lim_hajek_lin"] = r["lim_hajek_lin"]
        pr["lim_dr_both_bad"] = r["lim_dr_both_bad"]
        pr["lim_reg_lin"] = r["lim_reg_lin"]
        pr["theta_ac_real"] = r["theta_ac_real"]
        pred[name] = pr
        print(name, {k: round(v, 6) for k, v in pr.items()})
    json.dump(pred, open("predicciones.json", "w"), indent=1)
