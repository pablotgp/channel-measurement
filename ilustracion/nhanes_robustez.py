"""Robustez de la ilustración NHANES. Un solo comando:

    python ilustracion/nhanes_robustez.py

Tareas (200 réplicas cada una):
  debil  : el IMC influye menos en la asignación al canal (coeficiente 0,4 en vez de 0,8)
  fuerte : el IMC influye más (1,2)
  lgbm   : asignación base (0,8), pero clasificador y regresión con LightGBM en vez de splines

Antes de ejecutar cada tarea se calcula y guarda la predicción del Teorema 5(3) (prediccion_<tarea>.json),
con la hora, antes de que exista ninguna estimación. Es reanudable: cada réplica se guarda en
resultados_robustez/<tarea>/rep_<semilla>.json; si se corta, se relanza el mismo comando.
Al terminar crea resultados_robustez.zip con todo, para enviarlo.
"""
import os, sys, json, glob, zipfile, datetime, time

if __name__ == "__main__" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import nhanes_comun as c

R = 200
TAREAS = {"debil": dict(coef_bmi=0.4, learner="spline"),
          "fuerte": dict(coef_bmi=1.2, learner="spline"),
          "lgbm": dict(coef_bmi=0.8, learner="spline_pred_lgbm_est")}
SPECS = {"A": ("bmi", "bmi"), "B": ("bmi_self", "bmi"), "C": (None, None), "D": ("bmi_self", "bmi_self")}
OUT = os.path.join(HERE, "resultados_robustez")
N_JOBS = int(os.environ.get("N_JOBS", max(1, (os.cpu_count() or 2) - 1)))


def learner_est(t):
    return "lgbm" if TAREAS[t]["learner"].endswith("lgbm_est") else "spline"


def prediccion(df, tarea):
    pi = c.prob_canal(df, coef_bmi=TAREAS[tarea]["coef_bmi"]).values
    cols = c.NUM_V + c.CAT_V + ["bmi"]
    val = {}
    for y in ("sbp", "waist"):
        mu = c.regresion("bmi").fit(df[cols], df[y])  # mu siempre con splines y toda la población
        Xs = df[cols].copy(); Xs["bmi"] = df.bmi_self
        d = mu.predict(Xs) - mu.predict(df[cols])
        val[y] = dict(theta_esperado=float(np.sum(pi * df[y]) / pi.sum()),
                      sesgo_predicho_B=float(np.sum(pi * d) / pi.sum()))
    out = dict(fecha=datetime.datetime.now().isoformat(timespec="seconds"), tarea=tarea, **TAREAS[tarea],
               predicciones=["B: sesgo del DR compatible con sesgo_predicho_B (|z|<3 o dentro del 25 %), negativo.",
                             "A: sesgo pequeño frente al de B (menos del 20 % de B en la cintura).",
                             "Masa de p̂ > 0,99 en el objetivo inferior al 1 % en A y en B.",
                             "C: sesgo negativo y mayor en valor absoluto que B en la cintura.",
                             "D: sesgo en la cintura menor en valor absoluto que el de B."],
               valores=val)
    return out


def una(df, PI, seed, learner):
    rng = np.random.default_rng([seed, 7])
    S = rng.random(len(df)) < PI
    res = dict(seed=seed, n1=int(S.sum()))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(df, S))
    for k, (co, cd) in SPECS.items():
        X = df[c.NUM_V + c.CAT_V].copy()
        if co:
            X["b"] = np.where(S, df[co], df[cd])
        bcol = "b" if co else None
        p = np.zeros(len(df)); m = {y: np.zeros(len(df)) for y in ("sbp", "waist")}
        for tr, te in folds:
            p[te] = c.clasificador(bcol, learner).fit(X.iloc[tr], S[tr]).predict_proba(X.iloc[te])[:, 1]
            don = tr[~S[tr]]
            for y in m:
                m[y][te] = c.regresion(bcol, learner).fit(X.iloc[don], df[y].values[don]).predict(X.iloc[te])
        p = np.clip(p, 1e-6, 1 - 1e-6); o = p / (1 - p); D0 = ~S
        r = dict(auc=float(roc_auc_score(S, p)), masa_099=float(np.mean(p[S] > 0.99)),
                 neff_n0=float(o[D0].sum() ** 2 / np.sum(o[D0] ** 2) / D0.sum()))
        for y in m:
            Y = df[y].values
            A = np.sum(o[D0] * (Y[D0] - m[y][D0])) / np.sum(o[D0])
            r[f"dr_{y}"] = float(A + m[y][S].mean())
            r[f"hajek_{y}"] = float(np.sum(o[D0] * Y[D0]) / np.sum(o[D0]))
            r[f"theta_{y}"] = float(Y[S].mean())
        res[k] = r
    return res


def trabajo(tarea, seed):
    f = os.path.join(OUT, tarea, f"rep_{seed}.json")
    if os.path.exists(f):
        return
    df = c.cargar()
    PI = c.prob_canal(df, coef_bmi=TAREAS[tarea]["coef_bmi"]).values
    r = una(df, PI, seed, learner_est(tarea))
    tmp = f + ".tmp"; json.dump(r, open(tmp, "w")); os.replace(tmp, f)


def resumen(tarea):
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, tarea, "rep_*.json")))]
    pred = json.load(open(os.path.join(OUT, f"prediccion_{tarea}.json")))["valores"]
    lineas = [f"== {tarea} ({len(reps)} réplicas) {TAREAS[tarea]}"]
    for k in SPECS:
        a = np.mean([r[k]["auc"] for r in reps]); mm = np.mean([r[k]["masa_099"] for r in reps])
        ne = np.mean([r[k]["neff_n0"] for r in reps])
        partes = [f"{k}: AUC {a:.3f} masa p>0,99 {mm:.4f} neff/n0 {ne:.3f}"]
        for y in ("sbp", "waist"):
            b = np.array([r[k][f"dr_{y}"] - r[k][f"theta_{y}"] for r in reps])
            s = f"{y} sesgo DR {b.mean():+.3f} (EMC {b.std(ddof=1)/np.sqrt(len(b)):.3f})"
            if k == "B":
                s += f" pred {pred[y]['sesgo_predicho_B']:+.3f}"
            partes.append(s)
        lineas.append("  ".join(partes))
    return "\n".join(lineas)


if __name__ == "__main__":
    tareas = sys.argv[1:] or list(TAREAS)
    df = c.cargar()
    for t in tareas:
        os.makedirs(os.path.join(OUT, t), exist_ok=True)
        fp = os.path.join(OUT, f"prediccion_{t}.json")
        if not os.path.exists(fp):
            json.dump(prediccion(df, t), open(fp, "w"), indent=1, ensure_ascii=False)
            print(f"[{t}] predicción guardada antes de ejecutar: {fp}", flush=True)
    for t in tareas:
        t0 = time.time()
        pend = [s for s in range(R) if not os.path.exists(os.path.join(OUT, t, f"rep_{s}.json"))]
        print(f"[{t}] {len(pend)} réplicas pendientes, {N_JOBS} procesos", flush=True)
        Parallel(n_jobs=N_JOBS, verbose=5)(delayed(trabajo)(t, s) for s in pend)
        print(f"[{t}] terminado en {(time.time()-t0)/60:.1f} min", flush=True)
    texto = "\n\n".join(resumen(t) for t in tareas)
    open(os.path.join(OUT, "resumen.txt"), "w").write(texto)
    print(texto)
    z = os.path.join(HERE, "resultados_robustez.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(os.path.join(OUT, "**", "*.*"), recursive=True):
            zf.write(f, os.path.relpath(f, HERE))
    print(f"\nListo. Envía este fichero: {z}")
