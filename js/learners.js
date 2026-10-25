// Algoritmos de aprendizaje automático implementados en JavaScript (sin dependencias): media, GLM, ridge/lasso/red elástica,
// k vecinos, árbol CART, bosque aleatorio, boosting de gradiente y red neuronal de una capa.
// Interfaz común:  fit(X, y, family, params, ctx) → modelo;  predict(modelo, X) → Float64Array
//   X: arreglo de filas Float64Array (sin NaN); y: 0/1 (family = 'binary') o continuo (family = 'continuous').
import { fitGLM, fitOLS } from './models.js?v=2026.10.25-5';
import { enet, cvEnet, standardize } from './glmnet.js?v=2026.10.25-5';
import { rng } from './dist.js?v=2026.10.25-5';

const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const expit = e => 1 / (1 + Math.exp(-e));
const mean = a => sum(a) / a.length;

// ---------------------------------------------------------------- árbol de regresión CART (también para probabilidades: media de 0/1)
// opts: maxDepth, minNode (hoja mínima), minSplit, mtry (variables candidatas por nodo), cp (mejora mínima relativa a la SC total), R (generador)
export function growTree(X, y, idx, { maxDepth = 30, minNode = 5, minSplit = 10, mtry = null, cp = 0, R = null } = {}) {
  const p = X[0].length, feat = [], thr = [], left = [], right = [], val = [], cnt = [];
  let S = 0, SS = 0; for (const i of idx) { S += y[i]; SS += y[i] * y[i]; }
  const sst = Math.max(SS - S * S / idx.length, 1e-12), minGain = cp * sst, order = Array.from({ length: p }, (_, j) => j), buf = new Float64Array(idx.length);
  const add = (v, n) => { feat.push(-1); thr.push(0); left.push(-1); right.push(-1); val.push(v); cnt.push(n); return feat.length - 1; };
  const build = (ids, depth) => {
    const n = ids.length; let s = 0; for (let k = 0; k < n; k++) s += y[ids[k]];
    const node = add(s / n, n);
    if (depth >= maxDepth || n < minSplit || n < 2 * minNode) return node;
    // variables candidatas
    let cand = order; if (mtry && mtry < p) { const a = order.slice(); for (let k = 0; k < mtry; k++) { const j = k + Math.floor(R.u() * (p - k)); [a[k], a[j]] = [a[j], a[k]]; } cand = a.slice(0, mtry); }
    let best = { gain: minGain, f: -1, t: 0 };
    const base = s * s / n;
    for (const f of cand) {
      let mn = Infinity, mx = -Infinity; for (let k = 0; k < n; k++) { const v = X[ids[k]][f]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx <= mn) continue;
      if (mn === 0 && mx === 1) { // variable binaria: un solo punto de corte
        let nl = 0, sl = 0; for (let k = 0; k < n; k++) if (X[ids[k]][f] === 0) { nl++; sl += y[ids[k]]; }
        const nr = n - nl; if (nl < minNode || nr < minNode) continue; const g = sl * sl / nl + (s - sl) * (s - sl) / nr - base; if (g > best.gain) best = { gain: g, f, t: 0.5 };
        continue;
      }
      const ord = Uint32Array.from(ids).sort((a, b) => X[a][f] - X[b][f]); let sl = 0;
      for (let k = 0; k < n - 1; k++) {
        sl += y[ord[k]]; const nl = k + 1, nr = n - nl;
        if (nl < minNode) continue; if (nr < minNode) break;
        const v0 = X[ord[k]][f], v1 = X[ord[k + 1]][f]; if (v1 <= v0) continue;
        const g = sl * sl / nl + (s - sl) * (s - sl) / nr - base; if (g > best.gain) best = { gain: g, f, t: (v0 + v1) / 2 };
      }
    }
    if (best.f < 0) return node;
    const L = [], Rr = []; for (let k = 0; k < n; k++) (X[ids[k]][best.f] <= best.t ? L : Rr).push(ids[k]);
    feat[node] = best.f; thr[node] = best.t; left[node] = build(L, depth + 1); right[node] = build(Rr, depth + 1);
    return node;
  };
  build(Int32Array.from(idx), 0);
  return { feat: Int32Array.from(feat), thr: Float64Array.from(thr), left: Int32Array.from(left), right: Int32Array.from(right), val: Float64Array.from(val), cnt: Int32Array.from(cnt) };
}
export const leafOf = (t, x) => { let k = 0; while (t.feat[k] >= 0) k = x[t.feat[k]] <= t.thr[k] ? t.left[k] : t.right[k]; return k; };
const treePredict = (t, X) => Float64Array.from(X, x => t.val[leafOf(t, x)]);

// ---------------------------------------------------------------- definición de los algoritmos
const clamp01 = a => { for (let i = 0; i < a.length; i++) a[i] = Math.min(1 - 1e-6, Math.max(1e-6, a[i])); return a; };
const dropConst = X => { const p = X[0].length, keep = []; for (let j = 0; j < p; j++) { let a = X[0][j], c = false; for (let i = 1; i < X.length; i++) if (X[i][j] !== a) { c = true; break; } if (c) keep.push(j); } return keep; };
const sub = (X, keep) => X.map(x => Float64Array.from(keep, j => x[j]));

export const LEARNERS = {
  mean: {
    name: 'Media (referencia)', desc: 'Predice siempre la media (prevalencia). Sirve de línea base: todo algoritmo debe superarla.', params: {},
    fit: (X, y) => ({ m: mean(y) }), predict: (m, X) => new Float64Array(X.length).fill(m.m),
  },
  glm: {
    name: 'Regresión logística / lineal', desc: 'Modelo lineal generalizado sin penalización (todas las variables).', params: {},
    fit(X, y, fam) {
      const keep = dropConst(X), Z = sub(X, keep).map(x => Float64Array.of(1, ...x));
      try { const f = fam === 'binary' ? fitGLM(Z, y, 'binomial', { ridge: 1e-8 }) : fitOLS(Z, y); return { keep, beta: f.beta, fam }; }
      catch { const f = fitGLM(Z, y, fam === 'binary' ? 'binomial' : 'gaussian', { ridge: 1e-2 }); return { keep, beta: f.beta, fam }; }
    },
    predict(m, X) { const o = Float64Array.from(X, x => { let e = m.beta[0]; for (let k = 0; k < m.keep.length; k++) e += m.beta[k + 1] * x[m.keep[k]]; return e; }); return m.fam === 'binary' ? clamp01(o.map(expit)) : o; },
  },
  enet: {
    name: 'Regresión penalizada (glmnet)', desc: 'Lasso (α = 1), ridge (α = 0) o red elástica; λ elegido por validación cruzada interna.', params: { alpha: 1, nfolds: 5, rule: 'min' },
    fit(X, y, fam, P, ctx) { const f = fam === 'binary' ? 'binomial' : 'gaussian', cv = cvEnet(X, y, { family: f, alpha: P.alpha, nfolds: P.nfolds, nlambda: 30, seed: ctx?.seed || 1 }), lam = P.rule === '1se' ? cv.lambda1se : cv.lambdaMin, m = enet(X, y, { family: f, alpha: P.alpha, lambda: lam }); return { ...m, fam, lam }; },
    predict(m, X) { const o = Float64Array.from(X, x => { let e = m.intercept; for (let j = 0; j < x.length; j++) e += m.beta[j] * x[j]; return e; }); return m.fam === 'binary' ? clamp01(o.map(expit)) : o; },
  },
  knn: {
    name: 'k vecinos más cercanos', desc: 'Promedia el desenlace de los k pacientes más parecidos (variables estandarizadas).', params: { k: 15 },
    fit(X, y, fam, P) { const st = standardize(X); return { Z: st.Z, y, mean: st.mean, sd: st.sd, k: Math.min(P.k, X.length) }; },
    predict(m, X) {
      const p = m.mean.length, n = m.Z.length, k = m.k;
      return Float64Array.from(X, x => {
        const z = new Float64Array(p); for (let j = 0; j < p; j++) z[j] = (x[j] - m.mean[j]) / m.sd[j];
        const dk = new Float64Array(k).fill(Infinity), ik = new Int32Array(k);
        for (let i = 0; i < n; i++) { let d = 0; const r = m.Z[i]; for (let j = 0; j < p; j++) { const t = z[j] - r[j]; d += t * t; if (d >= dk[k - 1]) break; } if (d < dk[k - 1]) { let q = k - 1; while (q > 0 && dk[q - 1] > d) { dk[q] = dk[q - 1]; ik[q] = ik[q - 1]; q--; } dk[q] = d; ik[q] = i; } }
        let s = 0; for (let q = 0; q < k; q++) s += m.y[ik[q]]; return s / k;
      });
    },
  },
  tree: {
    name: 'Árbol de decisión (CART)', desc: 'Particiones binarias sucesivas; fácil de leer pero inestable.', params: { maxDepth: 5, minNode: 20, cp: 0.002 },
    fit: (X, y, fam, P) => ({ t: growTree(X, y, Array.from(y.keys()), { maxDepth: P.maxDepth, minNode: P.minNode, minSplit: 2 * P.minNode, cp: P.cp }), fam }),
    predict: (m, X) => (m.fam === 'binary' ? clamp01(treePredict(m.t, X)) : treePredict(m.t, X)),
  },
  rf: {
    name: 'Bosque aleatorio', desc: 'Promedia muchos árboles entrenados en muestras bootstrap y con variables al azar.', params: { ntree: 300, mtry: 0, minNode: 5 },
    fit(X, y, fam, P, ctx) {
      const n = X.length, p = X[0].length, R = rng(ctx?.seed || 1), mtry = P.mtry > 0 ? P.mtry : Math.max(1, fam === 'binary' ? Math.round(Math.sqrt(p)) : Math.round(p / 3)), trees = [];
      for (let b = 0; b < P.ntree; b++) { const idx = Int32Array.from({ length: n }, () => Math.floor(R.u() * n)); trees.push(growTree(X, y, idx, { maxDepth: 40, minNode: P.minNode, minSplit: Math.max(2, 2 * P.minNode - 2), mtry, R })); }
      return { trees, fam };
    },
    predict(m, X) { const o = new Float64Array(X.length); for (const t of m.trees) for (let i = 0; i < X.length; i++) o[i] += t.val[leafOf(t, X[i])]; for (let i = 0; i < o.length; i++) o[i] /= m.trees.length; return m.fam === 'binary' ? clamp01(o) : o; },
  },
  gbm: {
    name: 'Boosting de gradiente', desc: 'Árboles pequeños añadidos uno a uno, cada uno corrige los errores del anterior (como gbm/xgboost).', params: { nrounds: 200, lr: 0.05, depth: 3, minNode: 10, subsample: 0.7 },
    fit(X, y, fam, P, ctx) {
      const n = X.length, R = rng(ctx?.seed || 1), bin = fam === 'binary', F = new Float64Array(n), trees = [], m0 = mean(y), F0 = bin ? Math.log(m0 / (1 - m0)) : m0;
      F.fill(F0); const r = new Float64Array(n), h = new Float64Array(n);
      for (let b = 0; b < P.nrounds; b++) {
        for (let i = 0; i < n; i++) { if (bin) { const p = expit(F[i]); r[i] = y[i] - p; h[i] = p * (1 - p); } else { r[i] = y[i] - F[i]; h[i] = 1; } }
        const ids = P.subsample < 1 ? Array.from({ length: n }, (_, i) => i).filter(() => R.u() < P.subsample) : Array.from({ length: n }, (_, i) => i);
        const t = growTree(X, r, ids, { maxDepth: P.depth, minNode: P.minNode, minSplit: 2 * P.minNode });
        // valores de hoja de Newton: Σr / Σh
        const num = new Float64Array(t.val.length), den = new Float64Array(t.val.length); for (const i of ids) { const k = leafOf(t, X[i]); num[k] += r[i]; den[k] += h[i]; }
        for (let k = 0; k < t.val.length; k++) if (t.feat[k] < 0) t.val[k] = den[k] > 1e-12 ? num[k] / den[k] : 0;
        for (let i = 0; i < n; i++) F[i] += P.lr * t.val[leafOf(t, X[i])];
        trees.push(t);
      }
      return { trees, F0, lr: P.lr, fam };
    },
    predict(m, X) { const o = Float64Array.from(X, x => { let f = m.F0; for (const t of m.trees) f += m.lr * t.val[leafOf(t, x)]; return f; }); return m.fam === 'binary' ? clamp01(o.map(expit)) : o; },
  },
  nnet: {
    name: 'Red neuronal (1 capa oculta)', desc: 'Perceptrón con una capa oculta (tanh), regularización L2 y entrenamiento con Adam.', params: { size: 8, decay: 0.01, epochs: 250, lr: 0.02 },
    fit(X, y, fam, P, ctx) {
      const st = standardize(X), Z = st.Z, n = Z.length, p = Z[0].length, H = P.size, R = rng(ctx?.seed || 1), bin = fam === 'binary';
      const ym = bin ? 0 : mean(y), ys = bin ? 1 : (Math.sqrt(sum(Array.from(y, v => (v - mean(y)) ** 2)) / n) || 1), t = Float64Array.from(y, v => (v - ym) / ys);
      const W1 = Array.from({ length: H }, () => Float64Array.from({ length: p }, () => R.norm() * Math.sqrt(1 / p))), b1 = new Float64Array(H), W2 = Float64Array.from({ length: H }, () => R.norm() * Math.sqrt(1 / H)); let b2 = bin ? Math.log(mean(y) / (1 - mean(y))) : 0;
      const params = [...W1, b1, W2], mom = params.map(a => new Float64Array(a.length)), vel = params.map(a => new Float64Array(a.length)), mb = [0, 0], g1 = W1.map(() => new Float64Array(p)), gb1 = new Float64Array(H), gW2 = new Float64Array(H), A = Array.from({ length: n }, () => new Float64Array(H));
      let step = 0;
      for (let e = 0; e < P.epochs; e++) {
        let gb2 = 0; gb1.fill(0); gW2.fill(0); g1.forEach(g => g.fill(0));
        for (let i = 0; i < n; i++) {
          const a = A[i]; let o = b2; for (let k = 0; k < H; k++) { let s = b1[k]; const w = W1[k], z = Z[i]; for (let j = 0; j < p; j++) s += w[j] * z[j]; a[k] = Math.tanh(s); o += W2[k] * a[k]; }
          const d = (bin ? expit(o) - t[i] : o - t[i]) / n;
          gb2 += d; for (let k = 0; k < H; k++) { gW2[k] += d * a[k]; const dh = d * W2[k] * (1 - a[k] * a[k]); gb1[k] += dh; const g = g1[k], z = Z[i]; for (let j = 0; j < p; j++) g[j] += dh * z[j]; }
        }
        step++; const b1c = 1 - 0.9 ** step, b2c = 1 - 0.999 ** step, upd = (w, g, m, v, l2) => { for (let q = 0; q < w.length; q++) { const gg = g[q] + (l2 ? P.decay * w[q] : 0); m[q] = 0.9 * m[q] + 0.1 * gg; v[q] = 0.999 * v[q] + 0.001 * gg * gg; w[q] -= P.lr * (m[q] / b1c) / (Math.sqrt(v[q] / b2c) + 1e-8); } };
        for (let k = 0; k < H; k++) upd(W1[k], g1[k], mom[k], vel[k], true); upd(b1, gb1, mom[H], vel[H], false); upd(W2, gW2, mom[H + 1], vel[H + 1], true);
        mb[0] = 0.9 * mb[0] + 0.1 * gb2; mb[1] = 0.999 * mb[1] + 0.001 * gb2 * gb2; b2 -= P.lr * (mb[0] / b1c) / (Math.sqrt(mb[1] / b2c) + 1e-8);
      }
      return { st, W1, b1, W2, b2, ym, ys, bin, H };
    },
    predict(m, X) {
      const p = m.st.mean.length, o = Float64Array.from(X, x => { let out = m.b2; for (let k = 0; k < m.H; k++) { let s = m.b1[k]; const w = m.W1[k]; for (let j = 0; j < p; j++) s += w[j] * (x[j] - m.st.mean[j]) / m.st.sd[j]; out += m.W2[k] * Math.tanh(s); } return m.bin ? expit(out) : out * m.ys + m.ym; });
      return m.bin ? clamp01(o) : o;
    },
  },
};

// biblioteca: lista de {id, learner, params, label}
export const defaultLibrary = (names = ['mean', 'glm', 'enet', 'rf', 'gbm']) => names.map(id => ({ id, learner: id, params: { ...LEARNERS[id].params }, label: LEARNERS[id].name }));
// nombres en inglés para figuras y manuscritos
export const EN_NAME = { mean: 'Mean (reference)', glm: 'Logistic / linear regression', enet: 'Penalised regression (elastic net)', knn: 'k-nearest neighbours', tree: 'Decision tree (CART)', rf: 'Random forest', gbm: 'Gradient boosting', nnet: 'Neural network (1 hidden layer)' };
export function libraryLabelEn(c) { const d = LEARNERS[c.learner].params, diff = Object.keys(d).filter(k => c.params[k] !== d[k]).map(k => `${k}=${c.params[k]}`), base = c.label === LEARNERS[c.learner].name ? EN_NAME[c.learner] : String(c.label).replace('(variante)', '(variant)'); return base + (diff.length ? ` (${diff.join(', ')})` : ''); }
export function libraryLabel(c) { const d = LEARNERS[c.learner].params; const diff = Object.keys(d).filter(k => c.params[k] !== d[k]).map(k => `${k}=${c.params[k]}`); return c.label + (diff.length ? ` (${diff.join(', ')})` : ''); }

// imputación por la mediana del entrenamiento (se aplica dentro de cada pliegue para evitar fugas de información)
export function fitImputer(X) {
  const p = X[0].length, med = new Float64Array(p);
  for (let j = 0; j < p; j++) { const v = []; for (const x of X) if (!Number.isNaN(x[j])) v.push(x[j]); v.sort((a, b) => a - b); med[j] = v.length ? v[Math.floor(v.length / 2)] : 0; }
  const f = X2 => X2.map(x => { let bad = false; for (let j = 0; j < p; j++) if (Number.isNaN(x[j])) { bad = true; break; } if (!bad) return x; const r = Float64Array.from(x); for (let j = 0; j < p; j++) if (Number.isNaN(r[j])) r[j] = med[j]; return r; });
  f.med = med; return f; // la mediana se expone para exportar el modelo
}
export function fitPipeline(c, X, y, fam, ctx) {
  const L = LEARNERS[c.learner], imp = fitImputer(X), Xi = imp(X); // siempre se ajusta: un dato perdido solo en el pliegue de prueba también se imputa
  return { model: L.fit(Xi, y, fam, c.params, ctx), L, imp };
}
export const predictPipeline = (pl, X) => pl.L.predict(pl.model, pl.imp ? pl.imp(X) : X);
