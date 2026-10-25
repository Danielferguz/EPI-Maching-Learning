// Orientación sobre qué algoritmos son razonables según el tamaño de muestra, los eventos y los predictores, y notas de hiperparámetros / validación.
// Los umbrales son reglas prácticas conservadoras (van der Ploeg 2014; Riley 2019-2020; Steyerberg 2019): orientan, no sustituyen el juicio.
import { qnorm, dnorm } from './dist.js?v=2026.10.25-1';

const ln = Math.log, exp = Math.exp;
// ---- criterios de tamaño de muestra de Riley et al. (desarrollo, desenlace binario; mismos cálculos que pmsampsize y EPI-PREDIC)
const nShrink = (p, r2, S) => p / ((S - 1) * ln(1 - r2 / S));
const maxR2 = prev => 1 - exp(2 * (prev * ln(prev) + (1 - prev) * ln(1 - prev)));
function cstatToR2cs(C, prev) {
  const mu = Math.SQRT2 * qnorm(C), b = mu, a = ln(prev / (1 - prev)) - mu * mu / 2, lo = -9, hi = 9 + mu, N = 4000, h = (hi - lo) / N; let l1 = 0;
  for (let i = 0; i <= N; i++) { const x = lo + i * h, w = (i === 0 || i === N ? 0.5 : 1) * h, eta = a + b * x; l1 += w * (prev * dnorm(x - mu) * -Math.log1p(exp(-eta)) + (1 - prev) * dnorm(x) * -Math.log1p(exp(eta))); }
  return 1 - exp(2 * (prev * ln(prev) + (1 - prev) * ln(1 - prev) - l1));
}
// p = número de parámetros candidatos (columnas); devuelve el n mínimo para un modelo de regresión con esa C esperada
export function rileyBinary({ cstat = 0.75, prev, p, shrink = 0.9, delta = 0.05, margin = 0.05 }) {
  const mx = maxR2(prev), r2 = cstatToR2cs(cstat, prev), S2 = r2 / (r2 + delta * mx);
  const n1 = Math.ceil(nShrink(p, r2, shrink)), n2 = Math.ceil(nShrink(p, r2, S2)), n3 = Math.ceil((qnorm(0.975) / margin) ** 2 * prev * (1 - prev)), n = Math.max(n1, n2, n3);
  return { n, events: Math.ceil(n * prev), r2cs: r2, crit: [n1, n2, n3] };
}

// ---- semáforo de algoritmos. eff = eventos (o, en regresión, n/3: «eventos equivalentes»); p = columnas; n = pacientes
// shortcut: umbrales heurísticos fijos; mejorar con simulaciones propias del problema si se necesita rigor
export function adviseLearners({ n, events, p, fam }) {
  const eff = fam === 'binary' ? events : n / 3, epp = eff / Math.max(1, p), bin = fam === 'binary', unit = bin ? 'eventos' : 'pacientes/3';
  const r = (st, why) => ({ st, why });
  const small = `${eff.toFixed(0)} ${unit}, ${epp.toFixed(1)} por columna`;
  return {
    mean: r('ok', 'Línea base: todo algoritmo debe superarla.'),
    glm: epp >= 10 ? r('ok', `Suficiente (${small}).`) : epp >= 5 ? r('warn', `Justo (${small}); preferible la versión penalizada.`) : r('avoid', `Pocos datos (${small}): coeficientes inestables y sobreajuste; use regresión penalizada o reduzca predictores.`),
    enet: epp >= 3 ? r('ok', `Recomendada: tolera pocos datos (${small}).`) : r('warn', `Muy pocos datos (${small}); lasso fuerte y reduzca predictores.`),
    tree: eff >= 200 ? r('ok', `Interpretable, pero inestable (${small}); no esperar el mejor rendimiento.`) : eff >= 80 ? r('warn', `Inestable con ${small}.`) : r('avoid', `Demasiado inestable con ${small}.`),
    knn: n >= 1000 && p <= 15 ? r('ok', `n grande y pocas columnas (n = ${n}, p = ${p}).`) : n >= 300 && p <= 30 ? r('warn', `Sensible a la dimensión (n = ${n}, p = ${p}).`) : r('avoid', `Se degrada con pocas observaciones o muchas columnas (n = ${n}, p = ${p}).`),
    rf: eff >= 200 && epp >= 15 ? r('ok', `Datos suficientes (${small}).`) : eff >= 80 && epp >= 5 ? r('warn', `Rendimiento y calibración inestables (${small}); compare con la regresión.`) : r('avoid', `Pocos datos (${small}): sobreajuste probable.`),
    gbm: eff >= 300 && epp >= 20 ? r('ok', `Datos suficientes (${small}); ajuste tasa y rondas.`) : eff >= 100 && epp >= 8 ? r('warn', `Riesgo de sobreajuste (${small}); tasa baja y pocas rondas.`) : r('avoid', `Pocos datos (${small}): mejor regresión penalizada.`),
    nnet: eff >= 500 && epp >= 50 ? r('ok', `Datos suficientes (${small}).`) : eff >= 150 && epp >= 20 ? r('warn', `Exigente en datos (${small}); use decay alto.`) : r('avoid', `Pocos datos (${small}): sobreajuste casi seguro.`),
  };
}
export const ST = { ok: ['●', 'Razonable', '#2E7D32'], warn: ['▲', 'Con cautela', '#B7791F'], avoid: ['✖', 'Mejor no', '#B3261E'] };

// ---- notas de hiperparámetros (valores de partida razonables y cuándo moverlos)
export const HPNOTE = {
  mean: 'Sin hiperparámetros.',
  glm: 'Sin hiperparámetros. Necesita ≥ 10 eventos por columna; con menos, use la versión penalizada.',
  enet: 'α: 1 = lasso (selecciona variables), 0 = ridge (reduce todo); 0,5 suele ir bien con predictores correlacionados. Pliegues internos 5–10. Regla λ: «min» da el mejor ajuste; «1se» da modelos más simples y estables (recomendada con pocos eventos).',
  knn: 'k: empiece cerca de √n y pruebe 10–50; k pequeño = más varianza, k grande = más sesgo. Requiere pocas columnas.',
  tree: 'Profundidad 3–5; hoja mínima ≈ 1–5 % de n (≥ 20); cp 0,001–0,01 (mayor = árbol más corto). Los árboles aislados son inestables: úselos para interpretar, no para ganar.',
  rf: 'Árboles: 300–1000 (más no daña). mtry 0 = √p (clasificación) o p/3 (regresión); pruebe ½ y 2× el valor por defecto. Hoja mínima 5–20 (mayor con pocos eventos o ruido).',
  gbm: 'Tasa de aprendizaje 0,01–0,1 con más rondas (tasa × rondas ≈ 5–20); profundidad 2–4; hoja mínima ≥ 10; submuestra 0,5–0,8. Con pocos eventos: tasa 0,01–0,02, profundidad 1–2.',
  nnet: 'Neuronas 3–10; decay 0,01–0,1 (mayor con pocos datos); épocas 200–500; tasa 0,01–0,02. Suele rendir menos que regresión penalizada en datos tabulares pequeños.',
};

// ---- recomendación de pliegues (V) y repeticiones de la validación cruzada según el tamaño efectivo
export function adviseCV({ n, events, fam }) {
  const eff = fam === 'binary' ? events : n / 3;
  const V = eff < 100 || n < 300 ? 5 : 10, reps = eff < 100 ? '10 (el máximo de la herramienta)' : eff < 300 ? '5–10' : n < 2000 ? '3–5' : '1–3';
  return { V, reps, text: `Con ${n} pacientes${fam === 'binary' ? ` y ${events} eventos` : ''}: use <b>V = ${V}</b> pliegues (estratificados por el evento) y <b>${reps}</b> repeticiones. Más pliegues reducen el sesgo pero aumentan la varianza y el tiempo; las repeticiones estabilizan la estimación cuando hay pocos datos. Para el Super Learner anidado, 5 pliegues externos y 5 internos suelen bastar. Si ajusta hiperparámetros o selecciona variables, la evaluación debe ser anidada o externa para no ser optimista.` };
}
