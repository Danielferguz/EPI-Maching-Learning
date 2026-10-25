// Simplificación del modelo por importancia por permutación: se conservan las variables cuya permutación reduce el AUC en más de `thr`
// (en regresión, `thr` × DE del desenlace de aumento del RMSE). La importancia se mide en pliegues de PRUEBA (el Super Learner se ajusta
// sin ellos) y se promedia; la evaluación anidada repite todo el proceso de selección dentro de cada pliegue externo (sin optimismo).
import { makeFolds, superLearner, permutationImportance, metrics } from './sl.js?v=2026.10.25-4';

const sum = a => { let s = 0; for (const v of a) s += v; return s; }, mean = a => (a.length ? sum(a) / a.length : NaN);
const sdOf = a => (a.length > 1 ? Math.sqrt(sum(a.map(v => (v - mean(a)) ** 2)) / (a.length - 1)) : 0);
const part = (f, v, n) => { const tr = [], te = []; for (let i = 0; i < n; i++) (f[i] === v ? te : tr).push(i); return [tr, te]; };

export async function selectVars(lib, X, y, fam, groups, { Vsel = 3, Vcv = 5, R = 5, thr = 0.01, force = [], seed = 1, method = 'nnls' }, onStep = () => {}) {
  const n = y.length, f = makeFolds(y, Vsel, { seed: seed + 555 }), acc = groups.map(() => []);
  const cut = fam === 'binary' ? thr : thr * sdOf(Array.from(y));
  for (let v = 0; v < Vsel; v++) {
    const [tr, te] = part(f, v, n), sl = await superLearner(lib, tr.map(i => X[i]), Float64Array.from(tr, i => y[i]), fam, { V: Vcv, seed: seed + v, method });
    const imp = permutationImportance(sl.predict, te.map(i => X[i]), Float64Array.from(te, i => y[i]), fam, groups.map(g => ({ name: g.name, cols: g.cols })), { R, seed: seed + v });
    for (const i of imp) acc[groups.findIndex(g => g.name === i.name)].push(i.drop);
    onStep();
  }
  const rows = groups.map((g, k) => ({ name: g.name, drop: mean(acc[k]), sd: sdOf(acc[k]), keep: mean(acc[k]) > cut || force.includes(g.name), forced: force.includes(g.name) && !(mean(acc[k]) > cut) }));
  if (!rows.some(r => r.keep)) { const top = rows.reduce((a, r) => (r.drop > a.drop ? r : a), rows[0]); top.keep = true; top.top1 = true; } // nunca un modelo vacío
  return { rows, cut };
}

// p: {X, y, fam, lib, groups:[{name, cols}], thr, R, Vsel, Vcv, nested, Vo, force, seed, method}
export async function simplify(p, emit = () => {}) {
  const { X, y, fam, lib, groups, thr = 0.01, R = 5, Vsel = 3, Vcv = 5, nested = true, Vo = 5, force = [], seed = 1, method = 'nnls' } = p, n = y.length;
  const o = { Vsel, Vcv, R, thr, force, seed, method }, total = nested ? Vo * (Vsel + 2) + Vsel : Vsel; let done = 0;
  const step = stage => () => emit({ stage, done: ++done, total });
  const colsOf = names => groups.filter(g => names.includes(g.name)).flatMap(g => g.cols).sort((a, b) => a - b), sub = (rows, cols) => rows.map(x => Float64Array.from(cols, j => x[j]));
  let honest = null; const freq = groups.map(() => 0);
  if (nested) {
    const f = makeFolds(y, Vo, { seed: seed + 999 }), full = new Float64Array(n), red = new Float64Array(n), sizes = [];
    for (let v = 0; v < Vo; v++) {
      const [tr, te] = part(f, v, n), Xtr = tr.map(i => X[i]), ytr = Float64Array.from(tr, i => y[i]), Xte = te.map(i => X[i]);
      const sel = await selectVars(lib, Xtr, ytr, fam, groups, { ...o, seed: seed + 31 * v }, step(`Selección anidada (pliegue externo ${v + 1}/${Vo})`)), kept = sel.rows.filter(r => r.keep).map(r => r.name), cols = colsOf(kept);
      sel.rows.forEach((r, k) => { if (r.keep) freq[k]++; }); sizes.push(kept.length);
      const slF = await superLearner(lib, Xtr, ytr, fam, { V: Vcv, seed: seed + v, method }); step('Modelo completo')(); slF.predict(Xte).forEach((q, i) => (full[te[i]] = q));
      const slR = await superLearner(lib, sub(Xtr, cols), ytr, fam, { V: Vcv, seed: seed + v, method }); step('Modelo reducido')(); slR.predict(sub(Xte, cols)).forEach((q, i) => (red[te[i]] = q));
    }
    honest = { full: metrics(full, y, fam), reduced: metrics(red, y, fam), sizes, Vo };
  }
  const fin = await selectVars(lib, X, y, fam, groups, o, step('Selección final con todos los datos'));
  const rows = fin.rows.map((r, k) => ({ ...r, freq: nested ? freq[k] / Vo : null })).sort((a, b) => b.drop - a.drop);
  return { rows, kept: rows.filter(r => r.keep).map(r => r.name), thr, cut: fin.cut, honest, nested };
}
