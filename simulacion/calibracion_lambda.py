"""Calibración de Λ (modelo de sensibilidad marginal) frente a las covariables observadas.

1) Lema: en la región R = {s*(V) <= c*}, el factor oculto λ(X,U) del modelo "cola frente a CC"
   coincide con el cociente de odds del modelo de canal: odds(D=1|X,U) / odds(D=1|X).
   Se comprueba calculando λ de las dos formas, de manera exacta.
2) λ exacto de U en el escenario de confusión no observada (cuantiles de max(λ, 1/λ) en R).
3) Estimador de benchmark: para cada variable j, LightGBM del canal con y sin j (o con y sin U),
   entrenado en una mitad y evaluado en la otra; λ_j = odds_con / odds_sin en R.
"""
import json
import numpy as np
import lightgbm as lgb
from scipy.special import expit
from scenarios import SCEN
from gen import draw_clients, s_star, lin_D, p_hist, g_and_pD, GH_X, GH_W
from truth import features

QS = (0.50, 0.90, 0.99)
p, _, _ = SCEN["confusion_U"]
cs = json.load(open("c_star.json"))["confusion_U"]
rng = np.random.default_rng(20261024)


def wq(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(cw, q)]


def pD_given_VU(V, V5, U):
    """P(D=1 | V, U), integrando H con P(H|V) (H es independiente de U dado V)."""
    ph = p_hist(V)
    return (1 - ph) * expit(lin_D(V, V5, np.zeros(len(V)), U, p)) + ph * expit(lin_D(V, V5, np.ones(len(V)), U, p))


# ---------- 1) y 2): cálculo exacto en una población grande
c = draw_clients(400_000, p, rng)
V, V5, U, D, m = c["V"], c["V5"], c["U"], c["D"], c["m"]
R = s_star(V, p) <= cs
_, pD_V = g_and_pD(V[R], V5[R], p)                      # P(D=1 | V), integrando H y U
pD_VU = pD_given_VU(V[R], V5[R], U[R])
lam_canal = (pD_VU / (1 - pD_VU)) / (pD_V / (1 - pD_V))
# Modelo "cola frente a CC" (con π arbitrario): odds(Z=1|x) ∝ P(S=1|x)/P(D=0|x) = 1{R}·pD/(1-pD)
pi = 0.2
odds_cola_VU = pi / (1 - pi) * pD_VU / (1 - pD_VU)
odds_cola_V = pi / (1 - pi) * pD_V / (1 - pD_V)
lam_cola = odds_cola_VU / odds_cola_V
print(f"(1) Lema: máx |λ_cola − λ_canal| / λ_canal = {np.max(np.abs(lam_cola - lam_canal) / lam_canal):.1e}")
# comprobación de que E[λ | X] = 1 bajo el donante (D = 0) en R: E[λ·1{D=0}|X]/P(D=0|X)
lam_sym = np.maximum(lam_canal, 1 / lam_canal); w = m[R].astype(float)
exact = {q: wq(lam_sym, w, q) for q in QS}
print("(2) λ exacto de U en R:  " + "  ".join(f"q{int(q*100)} = {exact[q]:.2f}" for q in QS))

# ---------- 3) benchmarks estimados con el modelo de canal
n_fit = 200_000
c2 = draw_clients(n_fit, p, np.random.default_rng(7))
V2, V52, U2, D2, m2 = c2["V"], c2["V5"], c2["U"], c2["D"], c2["m"]
R2 = s_star(V2, p) <= cs
X_full = features(V2, V52)                                # V1..V4, V5==1, V5==2
names = ["V1", "V2", "V3", "V4", "V5"]
cols = {"V1": [0], "V2": [1], "V3": [2], "V4": [3], "V5": [4, 5]}
half = np.random.default_rng(3).random(n_fit) < 0.5
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, verbose=-1, n_jobs=2)


def odds_out(X):
    """Odds de D=1 fuera de muestra (entrena en una mitad, predice en la otra)."""
    o = np.zeros(len(X))
    for tr, te in ((half, ~half), (~half, half)):
        pr = lgb.LGBMClassifier(**LGB).fit(X[tr], D2[tr], sample_weight=m2[tr]).predict_proba(X[te])[:, 1]
        pr = np.clip(pr, 1e-6, 1 - 1e-6); o[te] = pr / (1 - pr)
    return o


o_full = odds_out(X_full)
res = {}
o_U = odds_out(np.c_[X_full, U2])                          # validación: "benchmark" de U
lam = o_U / o_full; s = np.maximum(lam, 1 / lam)[R2]
res["U (estimado)"] = {q: wq(s, m2[R2].astype(float), q) for q in QS}
for nm in names:
    keep = [k for k in range(X_full.shape[1]) if k not in cols[nm]]
    lam = o_full / odds_out(X_full[:, keep]); s = np.maximum(lam, 1 / lam)[R2]
    res[nm] = {q: wq(s, m2[R2].astype(float), q) for q in QS}
print("(3) Benchmarks en R, cuantiles de max(λ, 1/λ):")
print("    U (exacto)    " + "  ".join(f"q{int(q*100)} = {exact[q]:.2f}" for q in QS))
for k, v in res.items():
    print(f"    {k:13s} " + "  ".join(f"q{int(q*100)} = {v[q]:.2f}" for q in QS))
json.dump({"exacto_U": exact, "benchmarks": res}, open("calibracion_lambda_resultado.json", "w"), indent=1)
