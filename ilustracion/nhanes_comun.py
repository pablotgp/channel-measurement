"""Datos y piezas comunes de la ilustración semisintética con NHANES 2017-2018.

Población: los adultos (20 años o más) con IMC medido y autodeclarado válidos, tensión sistólica,
cintura, nivel educativo y ratio de renta (n = 4034). No usamos los pesos de encuesta: tratamos
la muestra como la población.

IMC autodeclarado = peso declarado (lb) / altura declarada (in)^2, convertido a kg/m^2 y
redondeado a 0,1, la misma precisión con la que NHANES publica el IMC medido. Sin ese redondeo,
los dos IMC vivirían en conjuntos de valores distintos y el soporte se rompería artificialmente.
"""
import os
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "datos", "nhanes")

NUM_V = ["age", "pir"]
CAT_V = ["sex", "race", "educ"]


def cargar():
    r = lambda f: pd.read_sas(os.path.join(D, f + ".xpt"))
    df = (r("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]]
          .merge(r("BMX_J")[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN")
          .merge(r("WHQ_J")[["SEQN", "WHD010", "WHD020"]], on="SEQN")
          .merge(r("BPX_J")[["SEQN", "BPXSY1", "BPXSY2", "BPXSY3"]], on="SEQN"))
    df = df[(df.RIDAGEYR >= 20) & (df.WHD010 < 100) & (df.WHD020 < 1000)].copy()
    df["bmi_self"] = np.round(df.WHD020 * 0.45359237 / (df.WHD010 * 0.0254) ** 2, 1)
    df["sbp"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    df = df[df.DMDEDUC2 <= 5]
    df = df.dropna(subset=["BMXBMI", "bmi_self", "sbp", "BMXWAIST", "INDFMPIR"])
    out = pd.DataFrame(dict(age=df.RIDAGEYR.values, pir=df.INDFMPIR.values, sex=df.RIAGENDR.values,
                            race=df.RIDRETH3.values, educ=df.DMDEDUC2.values,
                            bmi=df.BMXBMI.values, bmi_self=df.bmi_self.values,
                            sbp=df.sbp.values, waist=df.BMXWAIST.values))
    return out.reset_index(drop=True)


def z(x):
    return (x - x.mean()) / x.std()


# Regla de asignación al «canal objetivo» (p. ej., atención telefónica). Depende solo de
# covariables latentes (las verdaderas), así que la intercambiabilidad latente se cumple por construcción.
COEF = dict(bmi=0.8, age=0.5, pir=-0.4)


def prob_canal(df, a=-1.0, coef_bmi=None):
    cb = COEF["bmi"] if coef_bmi is None else coef_bmi
    return expit(a + cb * z(df.bmi) + COEF["age"] * z(df.age) + COEF["pir"] * z(df.pir))


def disenio(bmi_col):
    num = NUM_V + ([bmi_col] if bmi_col else [])
    return ColumnTransformer([
        ("spl", SplineTransformer(n_knots=5, degree=3), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_V)])


LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)


def _lgb_X(bmi_col):
    num = NUM_V + ([bmi_col] if bmi_col else [])
    return ColumnTransformer([("num", "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_V)])


def clasificador(bmi_col, learner="spline"):
    if learner == "lgbm":
        import lightgbm as lgb
        return make_pipeline(_lgb_X(bmi_col), lgb.LGBMClassifier(**LGB))
    return make_pipeline(disenio(bmi_col), LogisticRegression(C=1.0, max_iter=5000))


def regresion(bmi_col, learner="spline"):
    if learner == "lgbm":
        import lightgbm as lgb
        return make_pipeline(_lgb_X(bmi_col), lgb.LGBMRegressor(**LGB))
    return make_pipeline(disenio(bmi_col), Ridge(alpha=1.0))
