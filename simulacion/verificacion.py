"""Verificación numérica de los resultados del borrador en ejemplos continuos.

Ejemplo A (desplazamiento gaussiano): Q = N(0,1), P = N(mu,1).
  w(x) = exp(mu x - mu^2/2), m0(x) = sigmoid(a + b x).
  Verdades por cuadratura numérica.
"""
import numpy as np
from scipy import integrate, stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

rng = np.random.default_rng(2026)
mu, a, b = 1.0, -3.5, -0.8
w = lambda x: np.exp(mu * x - mu**2 / 2)
m0 = lambda x: expit(a + b * x)
fQ = lambda x: stats.norm.pdf(x)
fP = lambda x: stats.norm.pdf(x - mu)
I = lambda f: integrate.quad(f, -12, 12, limit=400)[0]

theta = I(lambda x: m0(x) * fP(x))
print(f"theta (cuadratura) = {theta:.6f}")
res = {}

def check(nombre, obtenido, esperado, tol):
    ok = abs(obtenido - esperado) <= tol
    res[nombre] = ok
    print(f"[{'OK ' if ok else 'FALLO'}] {nombre}: obtenido {obtenido:.6f}  esperado {esperado:.6f}  (tol {tol:g})")

# ---------- Teorema 1: Hájek con w verdadero
n0 = 2_000_000
X0 = rng.normal(size=n0); Y0 = (rng.random(n0) < m0(X0)).astype(float)
W0 = w(X0)
check("T1 Hájek con w verdadero", (W0 * Y0).sum() / W0.sum(), theta, 4e-4)
check("T1 forma no normalizada", (W0 * Y0).mean(), theta, 5e-4)
check("T1 E_Q[w] = 1", W0.mean(), 1.0, 1e-2)

# ---------- Teorema 2: clasificador logístico (modelo correcto) e invariancia en pi
for n1 in [200_000, 1_000_000]:
    X1 = rng.normal(mu, 1, size=n1)
    Xp = np.r_[X0[:600_000], X1]; Z = np.r_[np.zeros(600_000), np.ones(n1)]
    clf = LogisticRegression(C=1e6).fit(Xp[:, None], Z)
    pz = clf.predict_proba(X0[:600_000, None])[:, 1]; o = pz / (1 - pz)
    check(f"T2 odds logísticos, pi={n1/(600_000+n1):.2f}",
          (o * Y0[:600_000]).sum() / o.sum(), theta, 1e-3)
# odds = (pi/(1-pi)) w : comprobar la constante
pi = n1 / (600_000 + n1)
xs = np.array([-1.0, 0.0, 1.5])
po = clf.predict_proba(xs[:, None])[:, 1]
check("T2 odds/w = pi/(1-pi)", np.mean(po / (1 - po) / w(xs)), pi / (1 - pi), 5e-3)

# Clasificador no paramétrico con cross-fitting (como LightGBM)
X1 = rng.normal(mu, 1, size=200_000)
Xp = np.r_[X0[:400_000], X1]; Z = np.r_[np.zeros(400_000), np.ones(200_000)]
Yp = np.r_[Y0[:400_000], np.full(200_000, np.nan)]
oo = np.zeros(len(Xp))
for tr, te in KFold(5, shuffle=True, random_state=0).split(Xp):
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(Xp[tr, None], Z[tr])
    pp = np.clip(hgb.predict_proba(Xp[te, None])[:, 1], 1e-6, 1 - 1e-6); oo[te] = pp / (1 - pp)
c = Z == 0
check("T2 boosting con cross-fitting", (oo[c] * Yp[c]).sum() / oo[c].sum(), theta, 2e-3)

# Calibración: p~ = p^2 mismo orden, pesos sesgados
pz = clf.predict_proba(X0[:600_000, None])[:, 1]; pt = pz**2; ot = pt / (1 - pt)
sesgo_cal = (ot * Y0[:600_000]).sum() / ot.sum() - theta
print(f"      calibración: p~=p^2 da sesgo {sesgo_cal:+.5f} (debe ser claramente distinto de 0)")
res["T2 mal calibrado sesga"] = abs(sesgo_cal) > 5e-3

# ---------- Proposición 1: doble robustez (cuadratura)
def psi(wbar, mbar):
    return I(lambda x: wbar(x) * (m0(x) - mbar(x)) * fQ(x)) + I(lambda x: mbar(x) * fP(x))
mbad = lambda x: 0.03 + 0 * x; wbad = lambda x: 1 + 0 * x
check("P1 pesos malos, modelo bueno", psi(wbad, m0), theta, 1e-9)
check("P1 pesos buenos, modelo malo", psi(w, mbad), theta, 1e-9)
check("P1 ambos malos = fórmula producto", psi(wbad, mbad) - theta,
      I(lambda x: (wbad(x) - w(x)) * (m0(x) - mbad(x)) * fQ(x)), 1e-9)

# ---------- Proposición 4: recorte (signo positivo: b<0 y w creciente => g decreciente)
for q in [0.99, 0.95, 0.90]:
    tau = np.quantile(W0, q)
    wt = lambda x: np.minimum(w(x), tau)
    th_tau = I(lambda x: wt(x) * m0(x) * fQ(x)) / I(lambda x: wt(x) * fQ(x))
    ph = lambda x: np.minimum(1, tau / w(x))
    Ephi = I(lambda x: ph(x) * fP(x))
    cov = I(lambda x: m0(x) * ph(x) * fP(x)) - theta * Ephi
    check(f"P4 fórmula covarianza (tau=p{int(q*100)})", th_tau - theta, cov / Ephi, 1e-9)
    Wt = np.minimum(W0, tau)
    check(f"P4 Monte Carlo recortado (tau=p{int(q*100)})", (Wt * Y0).sum() / Wt.sum(), th_tau, 4e-4)
    res[f"P4 signo positivo p{int(q*100)}"] = th_tau > theta
    un = I(lambda x: wt(x) * m0(x) * fQ(x))
    res[f"P4 no normalizado <= theta p{int(q*100)}"] = un <= theta
    print(f"      tau=p{int(q*100)}: sesgo {th_tau-theta:+.5f} (>0: {th_tau>theta}), no normalizado {un:.5f} <= {theta:.5f}: {un<=theta}")

# Signo contrario si g creciente (b>0)
m0b = lambda x: expit(a + 0.8 * x)
thb = I(lambda x: m0b(x) * fP(x)); tau = np.quantile(W0, 0.95)
thb_t = I(lambda x: np.minimum(w(x), tau) * m0b(x) * fQ(x)) / I(lambda x: np.minimum(w(x), tau) * fQ(x))
res["P4 signo negativo si g creciente"] = thb_t < thb
print(f"      g creciente: sesgo {thb_t-thb:+.5f} (<0: {thb_t<thb})")

# ---------- Corolario 2: DR recortado
tau = np.quantile(W0, 0.95); wt = lambda x: np.minimum(w(x), tau)
mbar = lambda x: np.where(w(x) > tau, m0(x), 0.03)   # bien en la zona recortada, mal fuera
mbar2 = lambda x: np.where(w(x) > tau, 0.03, m0(x))  # mal en la zona recortada
for nom, mb in [("m bien en zona recortada", mbar), ("m mal en zona recortada", mbar2)]:
    un = psi(wt, mb) - theta
    form = -I(lambda x: (w(x) > tau) * (w(x) - tau) * (m0(x) - mb(x)) * fQ(x))
    check(f"C2 no normalizado, {nom}", un, form, 1e-9)
    cN = I(lambda x: wt(x) * fQ(x))
    nor = I(lambda x: wt(x) * (m0(x) - mb(x)) * fQ(x)) / cN + I(lambda x: mb(x) * fP(x)) - theta
    print(f"      normalizado, {nom}: sesgo {nor:+.6f}  (el corolario NO cubre esta versión)")

# ---------- Varianza (función de influencia) y Kish
pi = 0.2; n = 20_000; R = 3000
sig2 = lambda x: m0(x) * (1 - m0(x))
V = I(lambda x: (m0(x) - theta) ** 2 * fP(x)) / pi + I(lambda x: w(x) ** 2 * sig2(x) * fQ(x)) / (1 - pi)
ests = []
for r in range(R):
    Z = rng.random(n) < pi
    X = np.where(Z, rng.normal(mu, 1, n), rng.normal(0, 1, n))
    Y = (rng.random(n) < m0(X)).astype(float)
    o = pi / (1 - pi) * w(X[~Z])
    ests.append((o * (Y[~Z] - m0(X[~Z]))).sum() / o.sum() + m0(X[Z]).mean())
ests = np.array(ests)
check("T4 varianza asintótica n·Var", n * ests.var(), V, 0.08 * V)
z = (ests - theta) / np.sqrt(V / n)
check("T4 cobertura Wald 95 % (oráculo)", np.mean(np.abs(z) < 1.96), 0.95, 0.012)
Ew2 = I(lambda x: w(x) ** 2 * fQ(x))
check("Kish n_eff/n0 -> 1/E_Q[w^2]", W0.sum() ** 2 / (W0**2).sum() / n0, 1 / Ew2, 5e-3)
check("chi2 gaussiano = e^{mu^2}-1", Ew2 - 1, np.exp(mu**2) - 1, 1e-6)

# ---------- Proposición 5: contraejemplo, Q=U(0,1), P=Beta(1/2,1), w = x^{-1/2}/2
print("      contraejemplo chi2 = inf: n_eff/n0 al crecer n0")
ratios = []
for k in [3, 4, 5, 6, 7]:
    u = rng.random(10**k); ww = 0.5 / np.sqrt(u)
    ratios.append(ww.sum() ** 2 / (ww**2).sum() / 10**k)
    print(f"        n0=1e{k}: n_eff/n0 = {ratios[-1]:.4f}")
res["P5 n_eff/n0 decrece si chi2=inf"] = all(np.diff(ratios) < 0)
check("P5 integra 1", integrate.quad(lambda x: 0.5 / np.sqrt(x), 0, 1)[0], 1.0, 1e-8)

# ---------- Sección 5: variables de canal (población grande, binado fino en V)
print("\nSección 5: variables de canal")
def canal(delta, zeta=1.5, N=6_000_000):
    V = rng.normal(size=N)
    H = rng.random(N) < expit(-0.5 + 0.7 * V)          # C* != vacío (historial real)
    D = rng.random(N) < expit(0.2 + 0.8 * V + zeta * H)
    mu_star = expit(-3.2 - 0.5 * V + delta * H)          # E[Y(0)|X*]
    Y0 = (rng.random(N) < mu_star).astype(float)
    q = np.quantile(V[D], 0.10); S = D & (V <= q)        # grupo objetivo por regla fija en V
    cc = ~D
    theta = mu_star[S].mean()                            # verdad
    bins = np.quantile(V[cc], np.linspace(0, 1, 401)); bins[0], bins[-1] = -np.inf, np.inf
    bc = np.clip(np.digitize(V, bins) - 1, 0, 399)
    g_hat = np.bincount(bc[cc], weights=Y0[cc], minlength=400) / np.bincount(bc[cc], minlength=400)
    g_true = np.bincount(bc[cc], weights=mu_star[cc], minlength=400) / np.bincount(bc[cc], minlength=400)
    sin_C = g_hat[bc[S]].mean()                          # estimador reponderado sin C
    sesgo_pred = -(mu_star[S] - g_true[bc[S]]).mean()    # -∫(mu*-g)dP*
    kappa = H[S].mean()
    con_C = g_hat[bc[S & ~H]].mean()                     # con C: solo la parte C=vacío, comparada con g
    pred_con_C = g_true[bc[S & ~H]].mean()               # E_P*[g(V) | C*=vacío]
    theta_ac_real = mu_star[S & ~H].mean()
    return theta, sin_C, sesgo_pred, kappa, con_C, pred_con_C, theta_ac_real

for delta in [0.0, 1.2]:
    th, sinC, sp, kap, conC, pconC, thac = canal(delta)
    print(f"  delta={delta}: theta={th:.5f}  sin C={sinC:.5f}  kappa={kap:.3f}")
    if delta == 0:
        check("P3 sin C, delta=0: insesgado", sinC, th, 3e-4)
        check("Teo3 con C, delta=0: estima theta_ac", conC, thac, 3e-4)
    else:
        check("P3 sin C, delta!=0: sesgo = -∫(mu*-g)dP*", sinC - th, sp, 3e-4)
        check("S5 con C, delta!=0: converge a E[g|C*=vacío]", conC, pconC, 3e-4)
        print(f"      con C: {conC:.5f}; theta_ac real (parte sin dato) {thac:.5f}; "
              f"sobreestima esa parte: {conC > thac}")
        res["S5 con C sobreestima la parte sin dato"] = conC > thac

print("\nRESUMEN:", sum(res.values()), "de", len(res), "comprobaciones correctas")
for k, v in res.items():
    if not v: print("  FALLA:", k)
