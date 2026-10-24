"""Predicciones registradas ANTES de ejecutar los estimadores en NHANES (nhanes_ilustracion.py).

Teorema 5(3): si el grupo objetivo registra el IMC autodeclarado y los donantes el medido, el
estimador converge a theta_X = E_S[ mu(V, IMC_autodeclarado) ], con mu(V, b) = E[Y | V, IMC medido = b].
La asignación al canal depende solo de (V, IMC medido), así que E[Y | V, IMC, D=0] = mu y la
intercambiabilidad latente se cumple. Promediando sobre asignaciones con probabilidad pi_i:
   sesgo predicho = sum_i pi_i [mu(V_i, autodecl_i) - mu(V_i, medido_i)] / sum_i pi_i.
mu se ajusta con toda la población (splines + ridge), no con las réplicas.
"""
import json, os, datetime
import numpy as np
import nhanes_comun as c

df = c.cargar()
pi = c.prob_canal(df).values
pred = {}
for y in ("sbp", "waist"):
    mu = c.regresion("bmi").fit(df[c.NUM_V + c.CAT_V + ["bmi"]], df[y])
    Xs = df[c.NUM_V + c.CAT_V + ["bmi"]].copy(); Xs["bmi"] = df.bmi_self
    d = mu.predict(Xs) - mu.predict(df[c.NUM_V + c.CAT_V + ["bmi"]])
    theta = float(np.sum(pi * df[y]) / pi.sum())
    pred[y] = dict(theta_esperado=theta, sesgo_predicho_B=float(np.sum(pi * d) / pi.sum()))

out = dict(
    fecha=datetime.datetime.now().isoformat(timespec="seconds"),
    poblacion=len(df), replicas=200,
    especificaciones=dict(A="IMC medido en los dos grupos (oráculo)",
                          B="IMC autodeclarado en el objetivo, medido en los donantes (medición distinta)",
                          C="sin IMC (solo variables comunes V)",
                          D="IMC autodeclarado en los dos grupos (mismo instrumento, con error)"),
    predicciones=[
        "P1. A: sesgo medio del DR compatible con 0 (|z| < 3) en los dos resultados.",
        "P2. B: sesgo medio del DR compatible con el sesgo predicho por el Teorema 5(3) (|z| < 3, tolerando un error de "
        "modelo de hasta el 25 % del sesgo predicho), negativo en los dos resultados.",
        "P3. Indetectable: el AUC del clasificador en B difiere del de A en menos de 0,02, y la masa de p̂ > 0,99 en el "
        "grupo objetivo es inferior al 1 % en A y en B.",
        "P4. C: sesgo negativo en los dos resultados, y para la cintura mayor en valor absoluto que el de B.",
        "D se registra como exploratorio, sin predicción.",
        "No evaluamos la cobertura: las réplicas comparten la población y solo varía la asignación, así que el EE de "
        "Wald (que incluye el muestreo de la población) no es comparable con la dispersión entre réplicas."],
    valores=pred)
json.dump(out, open(os.path.join(c.HERE, "prediccion_nhanes.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out["valores"], indent=1))
