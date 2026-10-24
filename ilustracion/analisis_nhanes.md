# NHANES 2017-2018: resultados (200 réplicas) frente a las predicciones registradas

Las predicciones están en prediccion_nhanes.json, registradas en el commit 60b7217 antes de ejecutar.
Sesgos del estimador doblemente robusto (DR): media sobre réplicas de (estimación − theta), con su error de Monte Carlo (EMC).

| Esp. | Tensión sistólica (mmHg) | Cintura (cm) | AUC logit | AUC LightGBM | masa p̂>0,99 | n_eff/n0 |
|---|---|---|---|---|---|---|
| A: IMC medido en los dos | −0,11 (EMC 0,06) | −0,18 (0,02) | 0,743 | 0,724 | 0 | 0,41 |
| B: autodeclarado en el objetivo, medido en los donantes | −0,48 (0,06); predicho −0,47 | −2,56 (0,02); predicho −2,44 | 0,715 | 0,704 | 0 | 0,50 |
| C: sin IMC | −1,78 (0,05) | −10,70 (0,04) | 0,653 | — | 0 | 0,74 |
| D: autodeclarado en los dos | −0,47 (0,06) | −1,33 (0,03) | 0,737 | — | 0 | 0,44 |

## Predicciones
- P1 (A sin sesgo): se cumple en la tensión (z = −1,7). Falla en la cintura: −0,18 cm, z = −8,3. El sesgo es pequeño (7 % del de B) y procede de los modelos (aditivos y regularizados), no de la medición. Hájek: −0,73.
- P2 (B ≈ predicción del Teorema 5): se cumple. Tensión: z = −0,3. Cintura: diferencia de −0,12 cm, dentro de la tolerancia registrada del 25 % (0,61). Descontando el sesgo de modelo de A, B − A = −2,37 frente a −2,44.
- P3 (indetectable): la masa de p̂ cerca de 1 es 0 en A y en B, así que esa parte se cumple. La parte del AUC falla tal como se escribió: baja 0,028 con logit y 0,020 con LightGBM, más que el umbral de 0,02. En la práctica esa bajada no se puede observar, porque A no está disponible. Además, n_eff/n0 *mejora* con la medición distinta (0,41 → 0,50): los diagnósticos de solapamiento ven unos pesos mejores justo cuando el estimador está sesgado.
- P4 (C, sesgo negativo y mayor que el de B en la cintura): se cumple (−10,7 frente a −2,6).
- D (exploratorio): medir los dos grupos con el mismo instrumento con error sesga la mitad que medirlos distinto en la cintura (−1,33 frente a −2,56), e igual en la tensión.

# Robustez (ejecutada en el PC de Pablo, 200 réplicas por tarea)

Las predicciones (resultados_robustez/prediccion_*.json) se registraron en el commit a989734, antes de ejecutar. Las que llegaron del PC son idénticas.

| Tarea | A cintura | B cintura (pred.) | B tensión (pred.) | C cintura | D cintura | n_eff/n0 A → B |
|---|---|---|---|---|---|---|
| Débil (coef. IMC 0,4) | −0,08 | −2,26 (−2,19) | −0,42 (−0,41) | −5,69 | −0,61 | 0,61 → 0,67 |
| Base (0,8), splines | −0,18 | −2,56 (−2,44) | −0,48 (−0,47) | −10,70 | −1,33 | 0,41 → 0,50 |
| Fuerte (1,2) | −0,46 | −2,89 (−2,61) | −0,56 (−0,51) | −14,44 | −2,20 | 0,26 → 0,34 |
| Base (0,8), LightGBM | −0,01 | −2,42 (−2,44) | −0,20 (−0,47) | −10,38 | −1,23 | 0,26 → 0,35 |

Contraste: se cumplen 14 de las 15 predicciones registradas.
- B ≈ predicción del Teorema 5: se cumple en 5 de 6 casos. Falla la tensión con LightGBM: −0,20 frente a −0,47 (z = 4,0). Interpretación, sin contrastar: la tensión depende poco del IMC y, con un resultado tan ruidoso, LightGBM aplana esa dependencia, así que traslada solo una parte del error de medición. La predicción usa mu ajustada con splines.
- A pequeño frente a B (menos del 20 %): se cumple en los tres casos (3 %, 16 % y 0 %). Con LightGBM el sesgo de A desaparece (−0,006). Esto confirma que el −0,18 de la ejecución base venía de los modelos de splines.
- Masa de p̂ > 0,99 nula: se cumple en todos los casos.
- C negativo y mayor que B en la cintura: se cumple en los tres casos.
- |D| < |B| en la cintura: se cumple en los tres casos.
- No registrado, pero consistente en los cuatro diseños: n_eff/n0 mejora con la medición distinta.
