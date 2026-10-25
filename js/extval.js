// Validación externa de un modelo ya entrenado (sin reajustarlo) y su recalibración (Binuya et al. 2022):
// calibración en general (CITL), pendiente de calibración y comparación honesta de estrategias de recalibración por validación cruzada.
import { metrics } from './sl.js?v=2026.10.25-5';
import { fitGLM, fitOLS } from './models.js?v=2026.10.25-5';

const clamp = p => Math.min(1 - 1e-9, Math.max(1e-9, p)), logit = p => Math.log(clamp(p) / (1 - clamp(p))), expit = z => 1 / (1 + Math.exp(-z));
const sum = a => a.reduce((s, v) => s + v, 0), mean = a => sum(a) / a.length;

// calibración de las predicciones p frente a y: binario → pendiente, CITL (intercepto con pendiente fija en 1) y O/E; continuo → pendiente, CITL (media y − media predicha)
export function calibration(p, y, fam) {
  if (fam === 'binary') {
    const lp = Array.from(p, logit); let slope = NaN, slopeSE = NaN, citl = NaN, citlSE = NaN;
    try { const f = fitGLM(lp.map(v => Float64Array.of(1, v)), y, 'binomial'); slope = f.beta[1]; slopeSE = f.se[1]; } catch { /* separación */ }
    try { const f = fitGLM(lp.map(() => Float64Array.of(1)), y, 'binomial', { offset: Float64Array.from(lp) }); citl = f.beta[0]; citlSE = f.se[0]; } catch { /* separación */ }
    return { slope, slopeSE, citl, citlSE, oe: sum(y) / sum(p) };
  }
  let slope = NaN, slopeSE = NaN; try { const f = fitOLS(Array.from(p, v => Float64Array.of(1, v)), y); slope = f.beta[1]; slopeSE = f.se[1]; } catch { /* sin variación */ }
  return { slope, slopeSE, citl: mean(y) - mean(p), citlSE: NaN };
}
// parámetros de recalibración: 'citl' (solo nivel) o 'slope' (nivel y escala). Binario: p' = expit(a + b·logit p); continuo: ŷ' = a + b·ŷ
export function fitRecal(p, y, fam, kind) {
  if (kind === 'orig') return { a: 0, b: 1 };
  if (fam === 'binary') {
    const lp = Array.from(p, logit);
    if (kind === 'citl') { const f = fitGLM(lp.map(() => Float64Array.of(1)), y, 'binomial', { offset: Float64Array.from(lp) }); return { a: f.beta[0], b: 1 }; }
    const f = fitGLM(lp.map(v => Float64Array.of(1, v)), y, 'binomial'); return { a: f.beta[0], b: f.beta[1] };
  }
  if (kind === 'citl') return { a: mean(y) - mean(p), b: 1 };
  const f = fitOLS(Array.from(p, v => Float64Array.of(1, v)), y); return { a: f.beta[0], b: f.beta[1] };
}
export const applyRecal = (p, r, fam) => Array.from(p, v => (fam === 'binary' ? expit(r.a + r.b * logit(v)) : r.a + r.b * v));

// pliegues estratificados por evento (binario) con semilla
function foldsOf(n, K, y, seed) {
  let s = seed >>> 0; const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296), f = new Int32Array(n), idx = Array.from({ length: n }, (_, i) => i), groups = y ? [idx.filter(i => y[i] === 1), idx.filter(i => y[i] !== 1)] : [idx];
  for (const g of groups) { g.sort(() => u() - 0.5); g.forEach((i, k) => (f[i] = k % K)); }
  return f;
}
// estrategias: original, recalibración en general, recalibración con pendiente; rendimiento por validación cruzada (K pliegues × reps) en la base de validación
export function cvRecal(p, y, fam, { K = 5, reps = 3, seed = 7, kinds = ['orig', 'citl', 'slope'] } = {}) {
  const out = {};
  for (const kind of kinds) {
    const acc = [];
    for (let r = 0; r < reps; r++) {
      const f = foldsOf(y.length, K, fam === 'binary' ? y : null, seed + 101 * r), P = new Array(y.length);
      for (let k = 0; k < K; k++) {
        const tr = [], te = []; f.forEach((g, i) => (g === k ? te : tr).push(i));
        const rc = fitRecal(tr.map(i => p[i]), Float64Array.from(tr, i => y[i]), fam, kind), q = applyRecal(te.map(i => p[i]), rc, fam); te.forEach((i, j) => (P[i] = q[j]));
      }
      const m = metrics(P, y, fam), c = calibration(P, y, fam); acc.push({ ...m, slope: c.slope, citl: c.citl, oe: fam === 'binary' ? c.oe : NaN });
    }
    const keys = Object.keys(acc[0]); out[kind] = Object.fromEntries(keys.map(k => [k, mean(acc.map(a => a[k]))]));
  }
  return out;
}
// la estrategia más sencilla con pendiente 0.9–1.1 e intercepto cercano a 0; si ninguna, la de menor Brier (binario) o RMSE (continuo)
export function pickRecal(fam, res, order = ['orig', 'citl', 'slope']) {
  const ok = r => r && r.slope >= 0.9 && r.slope <= 1.1 && Math.abs(r.citl) < (fam === 'binary' ? 0.1 : 0.1 * (r.rmse || 1)), first = order.find(k => ok(res[k]));
  if (first) return { id: first, why: 'la más sencilla con pendiente entre 0.9 y 1.1 e intercepto cercano a 0' };
  const key = fam === 'binary' ? 'brier' : 'rmse', c = [...order].sort((a, b) => res[a][key] - res[b][key])[0];
  return { id: c, why: `ninguna alcanza una calibración ideal; es la de menor ${fam === 'binary' ? 'Brier' : 'RMSE'}` };
}
