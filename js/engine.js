// Motor de ejecución: se usa dentro de un Web Worker (js/worker.js) o, si no hay soporte, en el hilo principal.
import { cvPredict, superLearner, cvSuperLearner, metrics, makeFolds, permutationImportance, partialDependence, risk } from './sl.js?v=2026.10.25-5';
import { libraryLabel } from './learners.js?v=2026.10.25-5';
import { simplify } from './simplify.js?v=2026.10.25-5';
import { serializeModel } from './calc.js?v=2026.10.25-5';

const S = { sl: null, evalX: null, evalY: null, fam: null, Xtr: null, labels: [] };
const avg = arr => { const o = {}; for (const k of Object.keys(arr[0])) o[k] = arr.reduce((s, m) => s + m[k], 0) / arr.length; return o; };

export async function handle(cmd, p, emit = () => {}) {
  if (cmd === 'fit') {
    const { X, y, fam, lib, scheme, method, cvsl, seed, labels } = p, n = y.length; let Xtr = X, ytr = y, Xte = null, yte = null, te = null;
    if (scheme.type === 'holdout') {
      const K = Math.max(2, Math.round(1 / scheme.testFrac)), f = makeFolds(y, K, { seed }), tr = [], tt = []; for (let i = 0; i < n; i++) (f[i] === 0 ? tt : tr).push(i);
      Xtr = tr.map(i => X[i]); ytr = Float64Array.from(tr, i => y[i]); Xte = tt.map(i => X[i]); yte = Float64Array.from(tt, i => y[i]); te = tt;
    }
    const reps = scheme.type === 'cv' ? Math.max(1, scheme.repeats || 1) : 1, V = scheme.V, all = [];
    let first = null;
    for (let r = 0; r < reps; r++) {
      const c = await cvPredict(lib, Xtr, ytr, fam, { V, seed: seed + 1000 * r, onProgress: pr => emit({ stage: `${pr.stage}${reps > 1 ? ` (repetición ${r + 1}/${reps})` : ''}`, done: pr.done, total: pr.total }) });
      all.push(c.Z.map(z => metrics(z, ytr, fam))); if (r === 0) first = c;
    }
    const sl = await superLearner(lib, Xtr, ytr, fam, { V, seed, method, cv: first, onProgress: pr => emit(pr) });
    const perLearner = lib.map((_, k) => avg(all.map(a => a[k]))), ensCV = metrics(sl.ensCV, ytr, fam), discCV = metrics(first.Z[sl.discrete], ytr, fam);
    let honest = null;
    if (cvsl?.on) { const h = await cvSuperLearner(lib, Xtr, ytr, fam, { Vouter: cvsl.Vouter, Vinner: cvsl.Vinner, seed, method, onProgress: pr => emit(pr) }); honest = { ens: metrics(h.ens, ytr, fam), disc: metrics(h.disc, ytr, fam), each: h.each.map(z => metrics(z, ytr, fam)), ensPred: h.ens, weights: h.weights }; }
    let test = null;
    if (Xte) { const P = sl.predictEach(Xte), ens = sl.predict(Xte); test = { each: P.map(z => metrics(z, yte, fam)), ens: metrics(ens, yte, fam), disc: metrics(P[sl.discrete], yte, fam), ensPred: ens, y: yte, idx: te }; }
    S.sl = sl; S.fam = fam; S.Xtr = Xtr; S.labels = lib.map(libraryLabel); S.evalX = Xte || Xtr; S.evalY = yte || ytr; S.hold = !!Xte;
    return { labels: lib.map(libraryLabel), weights: sl.weights, cvRisk: sl.cvRisk, discrete: sl.discrete, perLearner, ensCV, discCV, honest, test, ensOOF: sl.ensCV, y: ytr, Z: first.Z, time: first.time, n: ytr.length, nTest: Xte ? yte.length : 0, oofEach: first.Z };
  }
  if (cmd === 'importance') { if (!S.sl) throw new Error('Ajuste primero los modelos.'); return permutationImportance(S.sl.predict, S.evalX, S.evalY, S.fam, p.groups, { R: p.R || 5, seed: p.seed || 1 }); }
  if (cmd === 'pdp') { if (!S.sl) throw new Error('Ajuste primero los modelos.'); return p.groups.map(g => ({ name: g.name, kind: g.kind, levels: g.levels, ...partialDependence(S.sl.predict, S.evalX, g, { seed: p.seed || 1 }) })); }
  if (cmd === 'predict') { if (!S.sl) throw new Error('Ajuste primero los modelos.'); return { ens: S.sl.predict(p.X), each: S.sl.predictEach(p.X) }; }
  if (cmd === 'simplify') return simplify(p, emit);
  if (cmd === 'export') { if (!S.sl) throw new Error('Ajuste primero los modelos.'); return serializeModel(S.sl, S.labels, S.Xtr); }
  throw new Error('Comando desconocido: ' + cmd);
}
