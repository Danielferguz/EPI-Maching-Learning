// Validación cruzada, métricas, Super Learner (van der Laan, Polley y Hubbard, 2007), Super Learner con validación cruzada anidada,
// importancia por permutación y dependencia parcial.
import { fitPipeline, predictPipeline, libraryLabel } from './learners.js?v=2026.10.25-6';
import { rng } from './dist.js?v=2026.10.25-6';
import { symInv, mv } from './linalg.js?v=2026.10.25-6';

export const tick = () => new Promise(r => setTimeout(r, 0));
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;

// ---------------------------------------------------------------- métricas
export function aucOf(score, y) {
  const idx = Array.from(score.keys()).sort((a, b) => score[a] - score[b]); let r = 0, n1 = 0;
  for (let k = 0; k < idx.length;) { let j = k; while (j + 1 < idx.length && score[idx[j + 1]] === score[idx[k]]) j++; const rr = (k + j) / 2 + 1; for (let m = k; m <= j; m++) if (y[idx[m]]) { r += rr; n1++; } k = j + 1; }
  const n0 = idx.length - n1; return (r - n1 * (n1 + 1) / 2) / (n1 * n0);
}
export function metricsBinary(p, y) {
  const n = y.length, e = sum(y), eps = 1e-9;
  let ll = 0, br = 0; for (let i = 0; i < n; i++) { const q = Math.min(1 - eps, Math.max(eps, p[i])); ll -= y[i] ? Math.log(q) : Math.log(1 - q); br += (p[i] - y[i]) ** 2; }
  const prev = e / n, auc = aucOf(p, y), n1 = e, n0 = n - e, q1 = auc / (2 - auc), q2 = 2 * auc * auc / (1 + auc), se = Math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0));
  return { auc, aucLo: Math.max(0, auc - 1.96 * se), aucHi: Math.min(1, auc + 1.96 * se), brier: br / n, brierScaled: 1 - (br / n) / (prev * (1 - prev)), logloss: ll / n, oe: e / sum(p), prev };
}
export function metricsCont(pred, y) {
  const n = y.length, my = mean(y), sse = sum(Array.from(y, (v, i) => (v - pred[i]) ** 2)), sst = sum(Array.from(y, v => (v - my) ** 2));
  return { rmse: Math.sqrt(sse / n), mae: sum(Array.from(y, (v, i) => Math.abs(v - pred[i]))) / n, r2: 1 - sse / sst, mse: sse / n };
}
export const metrics = (p, y, fam) => (fam === 'binary' ? metricsBinary(p, y) : metricsCont(p, y));
// riesgo usado por el Super Learner: error cuadrático medio (equivale a Brier en binarios)
export const risk = (p, y) => mean(Array.from(y, (v, i) => (v - p[i]) ** 2));

// ---------------------------------------------------------------- pliegues
export function makeFolds(y, V, { strat = true, seed = 1 } = {}) {
  const n = y.length, R = rng(seed), fold = new Int32Array(n), ord = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(R.u() * (i + 1)); [ord[i], ord[j]] = [ord[j], ord[i]]; }
  if (strat) { const bin = new Set(Array.from(y)).size === 2; if (bin) ord.sort((a, b) => y[a] - y[b]); }
  ord.forEach((i, k) => (fold[i] = k % V)); return fold;
}
const rowsOf = (X, ix) => ix.map(i => X[i]);

// ---------------------------------------------------------------- predicciones de validación cruzada para cada algoritmo
export async function cvPredict(lib, X, y, fam, { V = 10, seed = 1, fold = null, onProgress = null, label = 'Validación cruzada' } = {}) {
  const n = y.length, K = lib.length, f = fold || makeFolds(y, V, { seed }), Vn = Math.max(...f) + 1, Z = Array.from({ length: K }, () => new Float64Array(n)), time = new Float64Array(K);
  for (let v = 0; v < Vn; v++) {
    const tr = [], te = []; for (let i = 0; i < n; i++) (f[i] === v ? te : tr).push(i);
    const Xtr = rowsOf(X, tr), ytr = Float64Array.from(tr, i => y[i]), Xte = rowsOf(X, te);
    for (let k = 0; k < K; k++) { const t0 = performance.now(); const pl = fitPipeline(lib[k], Xtr, ytr, fam, { seed: seed + 7 * v + k }), p = predictPipeline(pl, Xte); te.forEach((i, q) => (Z[k][i] = p[q])); time[k] += performance.now() - t0; await tick(); }
    onProgress?.({ stage: label, done: v + 1, total: Vn });
  }
  return { Z, fold: f, time };
}

// ---------------------------------------------------------------- mínimos cuadrados no negativos (Lawson y Hanson, 1974)
export function nnls(A, b, maxit = 500) { // A: n × K (arreglo de filas), b: n
  const n = A.length, K = A[0].length, AtA = Array.from({ length: K }, (_, i) => Float64Array.from({ length: K }, (_, j) => { let s = 0; for (let r = 0; r < n; r++) s += A[r][i] * A[r][j]; return s; })), Atb = Float64Array.from({ length: K }, (_, i) => { let s = 0; for (let r = 0; r < n; r++) s += A[r][i] * b[r]; return s; });
  const x = new Float64Array(K), P = new Set(); const grad = () => Float64Array.from({ length: K }, (_, i) => { let s = Atb[i]; for (let j = 0; j < K; j++) s -= AtA[i][j] * x[j]; return s; });
  const solveP = () => { const idx = [...P], M = idx.map(i => Float64Array.from(idx, j => AtA[i][j] + (i === j ? 1e-12 : 0))), Mi = symInv(M); if (!Mi) return null; const s = mv(Mi, idx.map(i => Atb[i])), z = new Float64Array(K); idx.forEach((i, q) => (z[i] = s[q])); return z; };
  for (let it = 0; it < maxit; it++) {
    const w = grad(); let t = -1, mx = 1e-10; for (let i = 0; i < K; i++) if (!P.has(i) && w[i] > mx) { mx = w[i]; t = i; }
    if (t < 0) break; P.add(t);
    for (let inner = 0; inner < 100; inner++) {
      const z = solveP(); if (!z) { P.delete(t); break; }
      if ([...P].every(i => z[i] > 1e-14)) { x.set(z); break; }
      let alpha = Infinity; for (const i of P) if (z[i] <= 1e-14) alpha = Math.min(alpha, x[i] / (x[i] - z[i]));
      for (let i = 0; i < K; i++) x[i] += alpha * (z[i] - x[i]);
      for (const i of [...P]) if (x[i] < 1e-12) { x[i] = 0; P.delete(i); }
    }
  }
  return x;
}
// pesos en el símplex que maximizan la log-verosimilitud binomial de la mezcla (NNloglik de SuperLearner)
export function simplexLogLik(Z, y, iters = 400) { // Z: K × n
  const K = Z.length, n = y.length; let a = new Float64Array(K).fill(1 / K); const eps = 1e-6;
  const ll = w => { let s = 0; for (let i = 0; i < n; i++) { let p = 0; for (let k = 0; k < K; k++) p += w[k] * Z[k][i]; p = Math.min(1 - eps, Math.max(eps, p)); s += y[i] ? Math.log(p) : Math.log(1 - p); } return s; };
  let cur = ll(a), step = 1;
  for (let it = 0; it < iters; it++) {
    const g = new Float64Array(K); for (let i = 0; i < n; i++) { let p = 0; for (let k = 0; k < K; k++) p += a[k] * Z[k][i]; p = Math.min(1 - eps, Math.max(eps, p)); const d = y[i] ? 1 / p : -1 / (1 - p); for (let k = 0; k < K; k++) g[k] += d * Z[k][i]; }
    // ascenso exponenciado (se mantiene en el símplex)
    let improved = false; for (let h = 0; h < 25; h++) { const b = Float64Array.from(a, (v, k) => v * Math.exp(step * g[k] / n)); const s = sum(b); for (let k = 0; k < K; k++) b[k] /= s; const l = ll(b); if (l >= cur - 1e-14) { improved = l > cur + 1e-12; a = b; cur = l; step *= 1.3; break; } step /= 2; }
    if (!improved && step < 1e-8) break;
  }
  return a;
}

// ---------------------------------------------------------------- Super Learner
// method: 'nnls' (mínimos cuadrados no negativos, pesos normalizados a 1) | 'loglik' (máxima verosimilitud binomial)
export async function superLearner(lib, X, y, fam, { V = 10, seed = 1, method = 'nnls', onProgress = null, cv = null } = {}) {
  const K = lib.length, n = y.length, c = cv || await cvPredict(lib, X, y, fam, { V, seed, onProgress, label: 'Super Learner · validación cruzada' }), Z = c.Z;
  const cvRisk = Z.map(z => risk(z, y)), discrete = cvRisk.indexOf(Math.min(...cvRisk));
  let w;
  if (method === 'loglik' && fam === 'binary') w = simplexLogLik(Z, y);
  else { const A = Array.from({ length: n }, (_, i) => Float64Array.from(Z, z => z[i])); w = nnls(A, y); const s = sum(w); w = s > 0 ? Float64Array.from(w, v => v / s) : Float64Array.from({ length: K }, (_, k) => (k === discrete ? 1 : 0)); }
  const ens = Float64Array.from({ length: n }, (_, i) => { let s = 0; for (let k = 0; k < K; k++) s += w[k] * Z[k][i]; return s; });
  const fits = []; for (let k = 0; k < K; k++) { fits.push(fitPipeline(lib[k], X, y, fam, { seed: seed + 100 + k })); onProgress?.({ stage: 'Ajuste final de cada algoritmo', done: k + 1, total: K }); await tick(); }
  const predictEach = Xn => fits.map(f => predictPipeline(f, Xn));
  const predict = Xn => { const P = predictEach(Xn), o = new Float64Array(Xn.length); for (let k = 0; k < K; k++) if (w[k] > 0) for (let i = 0; i < o.length; i++) o[i] += w[k] * P[k][i]; return o; };
  return { lib, fam, Z, fold: c.fold, cvRisk, weights: w, discrete, ensCV: ens, ensRisk: risk(ens, y), predict, predictEach, time: c.time, method, fits };
}

// Super Learner con validación cruzada externa: estima honestamente el rendimiento del conjunto (y del «discreto») y compara con cada algoritmo
export async function cvSuperLearner(lib, X, y, fam, { Vouter = 5, Vinner = 5, seed = 1, method = 'nnls', onProgress = null } = {}) {
  const n = y.length, K = lib.length, f = makeFolds(y, Vouter, { seed: seed + 999 }), out = { ens: new Float64Array(n), disc: new Float64Array(n), each: Array.from({ length: K }, () => new Float64Array(n)), folds: f, weights: [] };
  for (let v = 0; v < Vouter; v++) {
    const tr = [], te = []; for (let i = 0; i < n; i++) (f[i] === v ? te : tr).push(i);
    const Xtr = rowsOf(X, tr), ytr = Float64Array.from(tr, i => y[i]), Xte = rowsOf(X, te);
    const sl = await superLearner(lib, Xtr, ytr, fam, { V: Vinner, seed: seed + 31 * v, method, onProgress: p => onProgress?.({ stage: `Super Learner anidado (pliegue externo ${v + 1}/${Vouter})`, done: v + p.done / p.total * (p.stage.startsWith('Ajuste') ? 1 : 0.6), total: Vouter }) });
    const P = sl.predictEach(Xte); out.weights.push(Array.from(sl.weights));
    te.forEach((i, q) => { out.disc[i] = P[sl.discrete][q]; let s = 0; for (let k = 0; k < K; k++) { out.each[k][i] = P[k][q]; s += sl.weights[k] * P[k][q]; } out.ens[i] = s; });
    onProgress?.({ stage: 'Super Learner anidado', done: v + 1, total: Vouter });
  }
  return out;
}

// ---------------------------------------------------------------- interpretación
// importancia por permutación: caída de rendimiento al barajar una variable (o todas las columnas de una variable categórica a la vez)
export function permutationImportance(predict, X, y, fam, groups, { R = 5, seed = 1 } = {}) {
  const base = fam === 'binary' ? aucOf(predict(X), y) : -Math.sqrt(risk(predict(X), y)), rg = rng(seed), n = X.length;
  return groups.map(g => {
    const drops = [];
    for (let r = 0; r < R; r++) {
      const perm = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rg.u() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      const Xp = X.map((x, i) => { const c = Float64Array.from(x); for (const j of g.cols) c[j] = X[perm[i]][j]; return c; });
      const sc = fam === 'binary' ? aucOf(predict(Xp), y) : -Math.sqrt(risk(predict(Xp), y)); drops.push(base - sc);
    }
    return { name: g.name, drop: mean(drops), sd: Math.sqrt(sum(drops.map(d => (d - mean(drops)) ** 2)) / Math.max(1, drops.length - 1)) };
  }).sort((a, b) => b.drop - a.drop);
}
// dependencia parcial (y curvas ICE) para una variable: g = {name, cols, kind:'num'|'cat', levels?, dummyCols?}
export function partialDependence(predict, X, g, { grid = 20, maxRows = 300, ice = 25, seed = 1 } = {}) {
  const R = rng(seed), n = X.length, pick = Array.from({ length: Math.min(n, maxRows) }, () => Math.floor(R.u() * n)), Xs = pick.map(i => X[i]);
  let xs;
  if (g.kind === 'num') { const v = Array.from(X, x => x[g.cols[0]]).filter(Number.isFinite).sort((a, b) => a - b); xs = Array.from({ length: grid }, (_, k) => v[Math.floor((v.length - 1) * (0.02 + 0.96 * k / (grid - 1)))]); xs = [...new Set(xs)]; }
  else xs = g.levels.map((_, k) => k);
  const iceRows = Array.from({ length: Math.min(ice, Xs.length) }, () => []), pd = xs.map(val => {
    const Xv = Xs.map(x => { const c = Float64Array.from(x); if (g.kind === 'num') c[g.cols[0]] = val; else g.cols.forEach((j, q) => (c[j] = val === q + 1 ? 1 : 0)); return c; });
    const p = predict(Xv); for (let i = 0; i < iceRows.length; i++) iceRows[i].push(p[i]); return mean(p);
  });
  return { x: xs, pd, ice: iceRows };
}
export { libraryLabel };
