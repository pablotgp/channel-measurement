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
