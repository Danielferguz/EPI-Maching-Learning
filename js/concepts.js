import { fitOLS } from './models.js?v=2026.10.25-1';
import { rng } from './dist.js?v=2026.10.25-1';
import { labPlot } from './mlplots.js?v=2026.10.25-1';
import { figWrap, bindFigs } from './figs.js?v=2026.10.25-1';

export const CONCEPTS_HTML = `
<div class="card prose">
  <h2>Aprendizaje automático para predecir: lo esencial</h2>
  <div class="howto"><div><b>Qué es</b>Un algoritmo aprende, a partir de ejemplos con desenlace conocido, una regla que predice el desenlace en pacientes nuevos. Aquí «aprendizaje supervisado»: clasificación (Sí/No) o regresión (valor numérico).</div><div><b>Qué no es</b>Predecir no es explicar ni estimar efectos causales. Un predictor muy «importante» no es necesariamente una causa ni un blanco terapéutico (para causalidad use EPI-CAUSAL).</div><div><b>Qué se mide</b>El rendimiento <i>en datos que el modelo no vio</i>: discriminación (AUC), calibración (¿el 20 % predicho ocurre en ~20 %?), error (RMSE, Brier) y utilidad.</div></div>
  <h3>1. Sobreajuste y la compensación sesgo-varianza</h3>
  <p>Un modelo muy flexible reproduce el ruido de la muestra de entrenamiento: parece excelente en los mismos datos (rendimiento <b>aparente</b>) y falla en datos nuevos. Un modelo demasiado simple es estable pero ignora estructura real. El objetivo es la complejidad que minimiza el error en datos nuevos. Pruebe el laboratorio:</p>
  <div class="form"><label>Pacientes de entrenamiento (n) <input id="lab-n" type="number" min="15" max="400" value="40"></label><label>Ruido (DE) <input id="lab-sd" type="number" step="0.1" min="0.1" max="2" value="0.5"></label><label>Grado del polinomio <input id="lab-deg" type="range" min="1" max="12" value="3"> <b id="lab-degv">3</b></label><label>Semilla <input id="lab-seed" type="number" value="7"></label></div>
  <div id="lab-out" class="plots single"></div>
  <p class="sub">Lo que se ve: la línea roja es el ajuste, la punteada la relación verdadera. A la derecha, el error de entrenamiento siempre baja al aumentar el grado, pero el error en datos nuevos tiene forma de U. Con menos pacientes o más ruido el grado óptimo baja.</p>
  <h3>2. Cómo evaluar sin engañarse</h3>
  <table class="data"><thead><tr><th>Esquema</th><th>Cómo</th><th>Cuándo</th></tr></thead><tbody>
  <tr><td><b>Validación cruzada (V pliegues)</b></td><td>Se entrena con V−1 pliegues y se predice el pliegue restante; se repite V veces</td><td>Esquema por defecto; usa todos los datos. V = 5–10; repítala si n es pequeño</td></tr>
  <tr><td><b>Hold-out</b></td><td>Se reserva una fracción como prueba que se usa una sola vez</td><td>n grande o cuando se necesita una evaluación independiente final</td></tr>
  <tr><td><b>Bootstrap</b></td><td>Corrección del optimismo repitiendo todo el proceso en remuestras (véase EPI-PREDIC)</td><td>Modelos de regresión para pocas variables</td></tr>
  <tr><td><b>Validación externa</b></td><td>Otro centro u otro periodo</td><td>Imprescindible antes de usar el modelo en la práctica</td></tr></tbody></table>
  <p><b>Fugas de información:</b> cualquier paso que use el desenlace o los datos de prueba (imputar con la media global, estandarizar con todos los datos, elegir variables mirando todo el conjunto) infla el rendimiento. Aquí la imputación, la estandarización y el ajuste de λ se hacen <i>dentro</i> de cada pliegue.</p>
  <h3>3. Los algoritmos de esta herramienta</h3>
  <table class="data"><thead><tr><th>Algoritmo</th><th>Idea</th><th>Fortalezas</th><th>Debilidades</th><th>Hiperparámetros clave</th></tr></thead><tbody>
  <tr><td>Regresión logística / lineal</td><td>Suma ponderada de las variables</td><td>Interpretable, estable, pocos datos</td><td>No capta no linealidades ni interacciones si no se especifican</td><td>—</td></tr>
  <tr><td>Lasso / ridge / red elástica</td><td>Regresión con penalización que contrae los coeficientes</td><td>Reduce el sobreajuste; lasso selecciona variables</td><td>Sigue siendo lineal</td><td>α (mezcla), λ (CV)</td></tr>
  <tr><td>k vecinos</td><td>Promedia los k pacientes más parecidos</td><td>Sin supuestos</td><td>Sensible a escalas y a variables irrelevantes; lento con n grande</td><td>k</td></tr>
  <tr><td>Árbol CART</td><td>Reglas «si-entonces» por particiones sucesivas</td><td>Fácil de explicar</td><td>Inestable, rendimiento limitado</td><td>profundidad, tamaño de hoja, cp</td></tr>
  <tr><td>Bosque aleatorio</td><td>Promedio de muchos árboles decorrelacionados</td><td>Robusto, capta interacciones, poco ajuste</td><td>Menos interpretable; probabilidades algo «tímidas»</td><td>n.º de árboles, mtry, hoja mínima</td></tr>
  <tr><td>Boosting de gradiente</td><td>Árboles pequeños secuenciales que corrigen errores</td><td>Muy buen rendimiento tabular</td><td>Sobreajusta si se abusa; más hiperparámetros</td><td>rondas, tasa de aprendizaje, profundidad</td></tr>
  <tr><td>Red neuronal (1 capa)</td><td>Combinaciones no lineales de las variables</td><td>Flexible</td><td>Necesita muchos datos y estandarización; inestable</td><td>neuronas, decay (L2), épocas</td></tr></tbody></table>
  <p class="sub">Con datos tabulares clínicos, la regresión penalizada suele rendir tanto como los algoritmos complejos salvo que haya interacciones fuertes y muchos datos; los modelos flexibles pueden requerir &gt; 10 veces más eventos por variable (van der Ploeg et al., 2014; Riley, 2019).</p>
  <h3>4. Super Learner</h3>
  <ol><li>Se define una <b>biblioteca</b> de algoritmos (idealmente variados).</li><li>Con validación cruzada se obtiene, para cada paciente, la predicción de <i>cada</i> algoritmo cuando ese paciente no estaba en el entrenamiento.</li><li>Se halla la combinación convexa de predicciones (pesos ≥ 0 que suman 1) que mejor predice el desenlace (mínimos cuadrados no negativos o máxima verosimilitud).</li><li>Se reajustan todos los algoritmos con todos los datos y se combinan con esos pesos.</li></ol>
  <p>Propiedad (oráculo): asintóticamente rinde tan bien como el mejor algoritmo de la biblioteca (o su mejor combinación). El Super Learner «discreto» es solo el ganador de la validación cruzada. Para estimar su rendimiento sin optimismo se usa la <b>validación cruzada anidada</b> (cvSuperLearner). Es el motor de TMLE en EPI-CAUSAL.</p>
  <h3>5. Métricas</h3>
  <ul><li><b>AUC:</b> probabilidad de que un paciente con evento tenga mayor riesgo predicho que uno sin evento (0.5 azar, 1 perfecto).</li><li><b>Brier / error cuadrático medio:</b> promedio de (predicho − observado)²; menor es mejor. El <b>Brier escalado</b> lo compara con predecir siempre la prevalencia.</li><li><b>Log-loss:</b> penaliza fuertemente las predicciones muy seguras y erróneas.</li><li><b>Calibración:</b> pendiente (ideal 1), intercepto, curva y observados/esperados (O/E).</li><li><b>RMSE, MAE, R²:</b> para desenlaces numéricos.</li></ul>
  <p><b>Desbalance de clases:</b> con eventos raros no «equilibre» artificialmente (sobremuestreo, SMOTE): estropea la calibración. Use métricas de probabilidad (Brier, calibración, curva de decisión) y umbrales clínicos.</p>
</div>`;

export function initLab() {
  const run = () => {
    const n = +document.getElementById('lab-n').value, sd = +document.getElementById('lab-sd').value, deg = +document.getElementById('lab-deg').value, R = rng(+document.getElementById('lab-seed').value || 1);
    document.getElementById('lab-degv').textContent = deg;
    const f = x => Math.sin(2.4 * x) + 0.4 * x, gen = m => { const x = Float64Array.from({ length: m }, () => 2 * R.u() - 1), y = Float64Array.from(x, v => f(v) + sd * R.norm()); return { x, y }; };
    const tr = gen(n), te = gen(600), des = (x, d) => Array.from(x, v => Float64Array.from({ length: d + 1 }, (_, k) => v ** k)), errs = [];
    const fits = []; for (let d = 1; d <= 12; d++) { try { const m = fitOLS(des(tr.x, d), tr.y), pr = (x) => des(x, d).map(r => r.reduce((s, a, k) => s + a * m.beta[k], 0)), rm = (x, y) => Math.sqrt(pr(x).reduce((s, p, i) => s + (p - y[i]) ** 2, 0) / y.length); errs.push({ train: rm(tr.x, tr.y), test: rm(te.x, te.y) }); fits.push(m); } catch { errs.push({ train: NaN, test: NaN }); fits.push(null); } }
    const m = fits[deg - 1], grid = Array.from({ length: 120 }, (_, k) => -1 + 2 * k / 119), curve = m ? grid.map(x => [x, m.beta.reduce((s, b, k) => s + b * x ** k, 0)]) : [];
    const out = document.getElementById('lab-out'); out.innerHTML = figWrap(labPlot({ x: tr.x, y: tr.y, curve, truth: grid.map(x => [x, f(x)]), deg, errs: errs.map(e => ({ train: isFinite(e.train) ? e.train : 0, test: isFinite(e.test) ? e.test : 3 })) }), 'laboratorio_sobreajuste'); bindFigs(out);
  };
  ['lab-n', 'lab-sd', 'lab-deg', 'lab-seed'].forEach(id => document.getElementById(id).addEventListener('input', run)); run();
}
