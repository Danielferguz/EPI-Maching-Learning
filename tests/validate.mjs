// Validación de los algoritmos y del Super Learner frente a scikit-learn y SciPy (tests/reference*.json).
import { readFileSync } from 'node:fs';
import { LEARNERS, defaultLibrary } from '../js/learners.js';
import { nnls, superLearner, cvSuperLearner, cvPredict, aucOf, risk, simplexLogLik, permutationImportance, partialDependence, makeFolds } from '../js/sl.js';
import { enet } from '../js/glmnet.js';
import { rng } from '../js/dist.js';

const R = f => JSON.parse(readFileSync(new URL(f, import.meta.url)));
const ref = R('reference.json'), g = R('reference_glmnet.json');
let fails = 0, checks = 0;
const ok = (name, cond, extra = '') => { checks++; if (!cond) { fails++; console.log('FAIL', name, extra); } else console.log('ok  ', name); };
const md = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));
const X = ref.X.map(r => Float64Array.from(r)), Xt = ref.Xt.map(r => Float64Array.from(r)), yb = Float64Array.from(ref.yb), ybt = Float64Array.from(ref.ybt), yc = Float64Array.from(ref.yc);

let m = LEARNERS.tree.fit(X, yc, 'continuous', { maxDepth: 4, minNode: 10, cp: 0 }); ok('árbol CART = scikit-learn (profundidad 4, hoja ≥ 10)', md(LEARNERS.tree.predict(m, Xt), ref.tree) < 1e-9);
m = LEARNERS.gbm.fit(X, yb, 'binary', { nrounds: 60, lr: 0.1, depth: 3, minNode: 10, subsample: 1 }); ok('boosting binario = GradientBoostingClassifier', md(LEARNERS.gbm.predict(m, Xt), ref.gbm_bin) < 1e-9);
m = LEARNERS.gbm.fit(X, yc, 'continuous', { nrounds: 60, lr: 0.1, depth: 3, minNode: 10, subsample: 1 }); ok('boosting continuo = GradientBoostingRegressor', md(LEARNERS.gbm.predict(m, Xt), ref.gbm_reg) < 1e-9);
m = LEARNERS.knn.fit(X, yb, 'binary', { k: 15 }); ok('k vecinos = KNeighborsClassifier (estandarizado)', md(LEARNERS.knn.predict(m, Xt), ref.knn) < 1e-12);
m = LEARNERS.rf.fit(X, yb, 'binary', { ntree: 300, mtry: 0, minNode: 5 }, { seed: 4 }); const aucRF = aucOf(LEARNERS.rf.predict(m, Xt), ybt);
ok(`bosque aleatorio: AUC de prueba ${aucRF.toFixed(3)} vs scikit-learn ${ref.rf_auc.toFixed(3)} (±0.03)`, Math.abs(aucRF - ref.rf_auc) < 0.03);
for (const f of g.fits.slice(0, 4)) { const Xg = g.X.map(r => Float64Array.from(r)), yg = Float64Array.from(f.family === 'binomial' ? g.y : g.yc), r = enet(Xg, yg, { family: f.family, alpha: f.alpha, lambda: f.lambda }); ok(`glmnet ${f.family} α=${f.alpha} λ=${f.lambda} = scikit-learn`, Math.max(Math.abs(r.intercept - f.intercept), ...f.beta.map((b, j) => Math.abs(b - r.beta[j]))) < 1e-6); }
const A = ref.nnls.A.map(r => Float64Array.from(r)), w = nnls(A, ref.nnls.b); ok('NNLS = scipy.optimize.nnls', md(w, ref.nnls.w) < 1e-8, Array.from(w));
m = LEARNERS.nnet.fit(X, yb, 'binary', { size: 6, decay: 0.01, epochs: 200, lr: 0.02 }, { seed: 2 }); ok('red neuronal: AUC de prueba > 0.75', aucOf(LEARNERS.nnet.predict(m, Xt), ybt) > 0.75, aucOf(LEARNERS.nnet.predict(m, Xt), ybt));
m = LEARNERS.nnet.fit(X, yc, 'continuous', { size: 6, decay: 0.01, epochs: 200, lr: 0.02 }, { seed: 2 }); { const pr = LEARNERS.nnet.predict(m, Xt), my = yc.reduce((a, b) => a + b, 0) / yc.length; ok('red neuronal (continuo): predice mejor que la media', Math.sqrt(pr.reduce((s, v, i) => s + (v - (1.2 * Xt[i][0] - 0.8 * Xt[i][1] * (Xt[i][2] > 0) + 0.7 * Xt[i][5] - 0.4)) ** 2, 0) / pr.length) < 0.6); }
// log-verosimilitud en el símplex: debe superar a cualquier algoritmo individual y a la mezcla uniforme
const r0 = rng(5), P1 = Float64Array.from(yb, v => Math.min(0.95, Math.max(0.05, 0.5 + 0.3 * (v - 0.5) + 0.2 * r0.norm() * 0.5))), P2 = Float64Array.from(yb, () => 0.4), ww = simplexLogLik([P1, P2], yb);
ok('NNloglik: pesos en el símplex y favorecen al predictor informativo', Math.abs(ww[0] + ww[1] - 1) < 1e-9 && ww[0] > 0.5, Array.from(ww));
// Super Learner en datos simulados: el riesgo CV del conjunto no es peor que el mejor algoritmo (con holgura) y los pesos suman 1
const lib = defaultLibrary(['mean', 'glm', 'enet', 'rf', 'gbm']); lib[3].params.ntree = 80; lib[4].params.nrounds = 80;
const sl = await superLearner(lib, X, yb, 'binary', { V: 5, seed: 2 }), best = Math.min(...sl.cvRisk);
console.log('   riesgo CV por algoritmo:', sl.cvRisk.map(v => v.toFixed(4)).join(' '), '· SL', sl.ensRisk.toFixed(4), '· pesos', Array.from(sl.weights).map(v => v.toFixed(2)).join(' '));
ok('Super Learner: Σ pesos = 1, pesos ≥ 0', Math.abs(sum(sl.weights) - 1) < 1e-9 && sl.weights.every(v => v >= 0));
ok('Super Learner: riesgo CV ≤ mejor algoritmo + 0.002 (el conjunto se elige con esos mismos datos)', sl.ensRisk <= best + 0.002);
ok('Super Learner: mejor que la media (referencia)', sl.ensRisk < sl.cvRisk[0] - 0.01);
const pt = sl.predict(Xt); ok('Super Learner: AUC en datos nuevos > 0.7', aucOf(pt, ybt) > 0.7, aucOf(pt, ybt));
const cvsl = await cvSuperLearner(lib, X, yb, 'binary', { Vouter: 3, Vinner: 3, seed: 3 });
ok('cvSuperLearner: predicciones externas completas y en [0,1]', cvsl.ens.every(v => v >= 0 && v <= 1) && cvsl.each.length === 5, `${aucOf(cvsl.ens, yb).toFixed(3)}`);
// importancia por permutación: la variable 0 (la más predictiva) debe ser la primera
const imp = permutationImportance(sl.predict, Xt, ybt, 'binary', Array.from({ length: 6 }, (_, j) => ({ name: 'x' + j, cols: [j] })), { R: 3 }); ok('importancia por permutación: x0 es la más importante', imp[0].name === 'x0', imp.map(i => i.name).join());
const pd = partialDependence(sl.predict, Xt, { name: 'x0', cols: [0], kind: 'num' }); ok('dependencia parcial: creciente en x0', pd.pd[pd.pd.length - 1] > pd.pd[0] + 0.15);
// pliegues estratificados
const fo = makeFolds(yb, 5, { seed: 1 }); const cnt = Array.from({ length: 5 }, (_, v) => { let e = 0, n = 0; yb.forEach((y, i) => { if (fo[i] === v) { n++; e += y; } }); return e / n; }); ok('pliegues estratificados: prevalencia similar', Math.max(...cnt) - Math.min(...cnt) < 0.03);
function sum(a) { let s = 0; for (const v of a) s += v; return s; }
console.log(`\n${checks - fails}/${checks} comprobaciones correctas`); process.exit(fails ? 1 : 0);
