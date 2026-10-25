# EPI-Maching-Learning

Módulo de **EPI-LAB** para **análisis de predicción con aprendizaje automático** y **Super Learner**, pensado para epidemiología clínica. Inspirado en el módulo de *machine learning* del curso *Epidemiological Methods* (Ehsan Karim), en el taller de DataLab y en la serie de modelos de predicción del *BMJ*.

Todo se calcula en el navegador (Web Worker): los datos no salen de tu computadora. Sitio estático, sin instalación.

## Qué incluye

| Paso | Contenido |
|---|---|
| 1 · Conceptos | Sobreajuste y sesgo-varianza con un **laboratorio interactivo**, esquemas de validación, fugas de información, tabla de algoritmos, Super Learner y métricas |
| 2 · Datos | CSV/Excel; ejemplos: **UCI/cateterismo de arteria pulmonar** (mortalidad y estancia), ERC, lesión renal aguda y PHQ-9 |
| 3 · Pregunta | Desenlace binario o continuo; predictores; codificación en indicadores; advertencias de eventos por variable; **tamaño mínimo para regresión (Riley et al.) y semáforo de algoritmos según pacientes, eventos y columnas** |
| 4 · Algoritmos y validación | Biblioteca con hiperparámetros y variantes: media, regresión logística/lineal, **lasso/ridge/red elástica**, k vecinos, **árbol CART**, **bosque aleatorio**, **boosting de gradiente**, red neuronal; validación cruzada (repetida) u hold-out; pesos NNLS o log-verosimilitud |
| 5 · Resultados | Tabla comparativa (AUC con IC, Brier, log-loss, O/E, RMSE, R²…), pesos del **Super Learner**, **Super Learner con validación cruzada anidada** (rendimiento honesto), ROC, calibración, observado vs predicho |
| 6 · Interpretación | Importancia por permutación, **dependencia parcial + ICE**, curva de decisión, predicción en pacientes nuevos, **simplificación del modelo** (variables con caída del AUC > 0,01; evaluación anidada, variables forzadas, reajuste de todo) |
| 3 · Pregunta (adicional) | **Tabla 1** (total y por desenlace, con p) y **regresión cruda/ajustada** por predictor, exportables |
| 7 · Reporte | Texto de métodos y resultados (ES/EN), tablas Word/CSV, predicciones fuera de muestra, **código Python (scikit-learn, sin figuras)** y **calculadora de riesgo descargable (HTML autónomo)** y **borrador de manuscrito ES/EN** (tablas y figuras en inglés) |

Las notas junto a cada algoritmo orientan los hiperparámetros, y la sección de validación recomienda pliegues (V) y repeticiones según los datos. Complementa a EPI-PREDIC: aquí se comparan algoritmos; allí se desarrolla y valida el modelo de regresión.

La imputación (mediana/moda), la estandarización y la elección de λ ocurren **dentro de cada pliegue** para evitar fugas de información.

## Validación de los motores

```bash
node tests/validate.mjs
python3 tools/make_reference.py   # regenera tests/reference.json (scikit-learn, SciPy)
```

* Árbol CART, boosting de gradiente (clasificación y regresión) y k vecinos: iguales a scikit-learn (diferencia < 1e-9).
* Regresión penalizada (lasso, ridge; logística y lineal): igual a scikit-learn a 1e-9. NNLS: igual a `scipy.optimize.nnls`.
* Bosque aleatorio: AUC dentro de 0.01–0.03 del de scikit-learn; red neuronal: comprobaciones de rendimiento.
* Super Learner: pesos en el símplex, riesgo de validación cruzada ≤ mejor algoritmo, importancia y dependencia parcial recuperan la estructura simulada.

## Advertencias

* Los algoritmos flexibles necesitan muchos más datos que la regresión; con pocos eventos la regresión penalizada suele ganar.
* No equilibre artificialmente las clases; evalúe calibración y utilidad, no solo AUC.
* Importancia y dependencia parcial describen al modelo, no efectos causales (véase EPI-CAUSAL); la validación externa se hace en EPI-PREDIC.
* Datos de ejemplo de ERC/LRA/PHQ-9 simulados; el ejemplo RHC procede de Connors et al., *JAMA* 1996 (datos públicos de Vanderbilt), reducido con `tools/make_rhc.py`.

## Uso local

```bash
python3 -m http.server 8000   # luego http://localhost:8000
```

## Licencia y cita

© Daniel Fernández-Guzmán. Licencia **CC BY-NC-ND 4.0**. Usar la aplicación y publicar los resultados está permitido citando EPI-Maching-Learning (ver `CITATION.cff`).
