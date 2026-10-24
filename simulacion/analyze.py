"""Compara los resultados de la simulación con las predicciones registradas antes de simular."""
import json, glob, sys
import numpy as np

pred = json.load(open("predicciones.json"))
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # réplicas: python analyze.py [R]


def load(name):
    return json.load(open(f"res_{name}_{R}.json"))


def col(res, k):
    return np.array([r[k] for r in res])


def mc(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


rows_main, rows_trim, rows_spec = [], [], []
for name in pred:
    try:
        res = load(name)
    except FileNotFoundError:
        continue
    pr = pred[name]; th = pr["theta"]; lim = pr["lim"]
    for est in ("hajek", "dr"):
        x = col(res, est); m, s = mc(x)
        row = dict(escenario=name, estimador=est, theta=th, prediccion=lim, media=m, mcse=s,
                   z=(m - lim) / s, sesgo=m - th, sd=x.std(ddof=1), rmse=np.sqrt(np.mean((x - th) ** 2)))
        if est == "dr":
            se = col(res, "dr_se")
            row["cobertura"] = np.mean(np.abs(x - th) <= 1.96 * se)
            row["se_medio/sd"] = se.mean() / x.std(ddof=1)
        rows_main.append(row)
    rows_main[-1].update(ess_est=col(res, "ess_ratio").mean(), ess_teo=pr["ess_ratio"],
                         eps_est=col(res, "eps_hat").mean(), eps_teo=pr["eps"],
                         auc_est=col(res, "auc").mean(), chi2_teo=pr["chi2"])
    # recorte: diferencias pareadas (recortado - sin recortar)
    if not name.startswith("confunde_con") and name != "con_C":
        for q in (99, 95, 90):
            dh = col(res, f"hajek_trim{q}") - col(res, "hajek")
            dd = col(res, f"dr_trim{q}") - col(res, "dr")
            ph = pr[f"lim_trim{q}"] - pr["lim"]
            mh, sh = mc(dh); md, sd_ = mc(dd)
            cov = np.mean(np.abs(col(res, f"dr_trim{q}") - th) <= 1.96 * col(res, f"dr_trim{q}_se"))
            rows_trim.append(dict(escenario=name, q=q, pred_hajek=ph, hajek=mh, hajek_mcse=sh,
                                  z=(mh - ph) / sh, dr=md, dr_mcse=sd_, cobertura_dr=cov))
    if "hajek_lin" in res[0]:
        for k, p_ in (("hajek_lin", pr["lim_hajek_lin"]), ("dr_linw_goodm", lim), ("dr_goodw_linm", lim),
                      ("dr_both_bad", pr["lim_dr_both_bad"]), ("reg_lin", pr["lim_reg_lin"]), ("dr", lim)):
            x = col(res, k); m, s = mc(x)
            r = dict(escenario=name, estimador=k, prediccion=p_, media=m, mcse=s, z=(m - p_) / s, sesgo=m - th)
            if k + "_se" in res[0]:
                r["cobertura"] = np.mean(np.abs(x - th) <= 1.96 * col(res, k + "_se"))
            rows_spec.append(r)


def show(rows, title):
    print("\n==", title)
    for r in rows:
        print("  " + "  ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


show(rows_main, "Principales")
show(rows_trim, "Recorte (diferencias pareadas)")
show(rows_spec, "Especificación")
json.dump(dict(main=rows_main, trim=rows_trim, spec=rows_spec),
          open("resumen.json", "w"), indent=1, default=float)
