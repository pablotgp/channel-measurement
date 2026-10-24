# Evaluación contrafactual del cambio de canal

Material de trabajo de dos artículos en preparación sobre cómo estimar qué pasaría si un grupo de clientes fuese atendido por otro canal, sin un experimento aleatorizado:

1. **Cuando los canales miden distinto: evaluación contrafactual del cambio de canal mediante cambio de medida.** Es el artículo principal. Identificación como cambio de medida (Radon–Nikodym), soporte vía clasificador (descomposición de Lebesgue), variables registradas solo en un canal o medidas con instrumentos distintos, sensibilidad a la confusión no observada y simulación con predicciones registradas antes de ejecutarla.
2. **Recortar pesos estimados: signo, ruido e inferencia.** Es una nota complementaria. Sesgo exacto del recorte de pesos, conservación o inversión de su signo con pesos estimados y comportamiento del estimador doblemente robusto recortado.

Todo el material es sintético o teórico. **No contiene datos reales de ninguna entidad.**

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `borradores/` | Exportaciones de los dos borradores: el paper 1 en markdown y PDF, y el paper 2 en PDF. Los originales se editan en Claude Docs; aquí hay una copia fechada. |
| `guias/` | Guías de lectura que explican cada resultado paso a paso, con fuente LaTeX (`*.tex`, preámbulo común en `preambulo.tex`) y PDF compilado (`guias/pdf/`). |
| `simulacion/` | Código de la simulación, predicciones registradas, resultados brutos de cada escenario y scripts de comprobación y auditoría. |

## La simulación

Ficheros principales:

- `gen.py`: generador de datos (clientes, llamadas, canal, historial, compras).
- `truth.py`: verdad y límites teóricos calculados de forma analítica en una población grande.
- `scenarios.py`: escenarios. Ejecutado como script, calcula y guarda las predicciones en `predicciones.json`.
- `est.py`: estimadores de Hájek y doblemente robusto, con *cross-fitting* por cliente (LightGBM).
- `run.py`: ejecuta las réplicas de un escenario y guarda `res_<escenario>_<R>.json`.
- `analyze.py`: compara los resultados con las predicciones.

Predicciones registradas antes de ejecutar:

- `predicciones.json`: todos los escenarios de la sección 9 del paper 1.
- `prediccion_DR_sin_soporte.json`: predicción del Corolario 3, registrada antes del escenario «confunde_con_C».
- `prediccion_n_10000.json`: escenario intermedio de 10 000 clientes.

Comprobaciones: `verificacion.py` contrasta los resultados teóricos en ejemplos continuos. `audit.py` y `audit_truth_se.py` comprueban las verdades por una vía independiente. `check_trim.py`, `diag_trim_sim.py`, `trim_theory_check.py` y `sign_test.py` tratan del recorte con pesos estimados; la salida de este último está en `sign_test_resultado_60_replicas.txt`. `check_medicion.py` contrasta el Teorema 5 y `check_sens.py` la Proposición 7. `calibracion_lambda.py` calibra Λ frente a las covariables observadas (Lema 3), con su predicción registrada en `prediccion_calibracion_lambda.json` y su resultado en `calibracion_lambda_resultado.json`; `calibracion_lema_check.py` contrasta el lema estimando por separado los dos modelos, y su salida está en `calibracion_lema_check_resultado.txt`.

`res_base_40.json` es la prueba piloto inicial (40 réplicas, semilla 500). Notas sobre los resultados del escenario de solapamiento medio: `res_solap_medio_200.json` y `res_solap_medio_ORIGINAL.json` son la tanda original (semillas 1000–1199) y `res_solap_medio_rep7000.json` es la réplica con semillas nuevas (7000–7199).

### Reproducir

```bash
cd simulacion
pip install -r ../requirements.txt
python scenarios.py            # predicciones (unos 2 minutos)
python run.py base 200 1000    # un escenario: nombre, réplicas, semilla inicial
python analyze.py              # comparación con las predicciones
```

Con 2 núcleos, cada escenario de 30 000 clientes y 200 réplicas tarda unos 8–10 minutos.

Por defecto se usan 2 procesos en paralelo. Para usar todos los núcleos de tu máquina, define `N_JOBS` (`-1` = todos):

```bash
N_JOBS=-1 python run.py base 200 1000
N_JOBS=-1 python bootstrap.py n_10000 80 60 3000   # escenario, réplicas, remuestreos, semilla inicial
```

Los resultados no dependen de `N_JOBS`: cada réplica usa su propia semilla y LightGBM se ejecuta con un solo hilo dentro de cada proceso. Las semillas usadas son: 1000 para los escenarios de la tabla principal, 7000 para la réplica del solapamiento medio y 3000 para el escenario de 10 000 clientes.

## Estado

Los borradores están en español, para fijar notación y resultados; la versión final irá en inglés y LaTeX. La lista de tareas pendientes está al final del paper 1.

Última actualización: 24 de octubre de 2026.
