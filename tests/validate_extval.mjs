// Validación externa y recalibración (js/extval.js): predicciones con intercepto desplazado y pendiente excesiva → la recalibración con pendiente las corrige.
import { calibration, fitRecal, applyRecal, cvRecal, pickRecal } from '../js/extval.js';
let fails = 0, checks = 0; const ok = (n, c, x = '') => { checks++; if (!c) { fails++; console.log('FAIL', n, x); } else console.log('ok  ', n); };
let s = 12345; const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296), nrm = () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
const n = 4000, x = Array.from({ length: n }, nrm), expit = z => 1 / (1 + Math.exp(-z)), y = Float64Array.from(x, v => (u() < expit(-1 + 0.8 * v) ? 1 : 0));
// el modelo cree: logit = 0.6 − 1.2 + 1.2·x (nivel +0.6 demasiado alto, escala 1.5 veces mayor)
const p = Float64Array.from(x, v => expit(0.6 - 1 + 1.2 * v)), c = calibration(p, y, 'binary');
ok('pendiente de calibración ≈ 0.8/1.2 = 0.67', Math.abs(c.slope - 0.667) < 0.08, c.slope);
ok('el modelo sobreestima el riesgo (CITL < 0 y O/E < 1)', c.citl < -0.2 && c.oe < 0.9, JSON.stringify(c));
const r = fitRecal(p, y, 'binary', 'slope'), q = applyRecal(p, r, 'binary'), c2 = calibration(q, y, 'binary');
ok('recalibración con pendiente: pendiente ≈ 1 y CITL ≈ 0 en los mismos datos', Math.abs(c2.slope - 1) < 0.02 && Math.abs(c2.citl) < 0.02, JSON.stringify(c2));
const cv = cvRecal(Array.from(p), y, 'binary', { K: 5, reps: 2 });
ok('validación cruzada: original mal calibrada; con pendiente bien calibrada', cv.orig.slope < 0.8 && cv.slope.slope > 0.9 && cv.slope.slope < 1.1 && Math.abs(cv.slope.citl) < 0.1, JSON.stringify(cv));
ok('la recalibración casi no cambia el AUC (transformación monótona por pliegue) y baja el Brier', Math.abs(cv.slope.auc - cv.orig.auc) < 0.005 && cv.slope.brier < cv.orig.brier, `${cv.slope.auc} ${cv.orig.auc}`);
ok('la recalibración en general corrige el nivel pero no la escala', Math.abs(cv.citl.citl) < 0.1 && cv.citl.slope < 0.8);
ok('sugiere la recalibración con pendiente', pickRecal('binary', cv).id === 'slope');
// continuo: y = 2 + 0.5·x; predicción = 1 + 1.0·x → pendiente 0.5, nivel desplazado
const yc = Float64Array.from(x, v => 2 + 0.5 * v + 0.3 * nrm()), pc = Float64Array.from(x, v => 1 + v), cc = calibration(pc, yc, 'continuous');
ok('continuo: pendiente ≈ 0.5 y CITL = media(y) − media(pred)', Math.abs(cc.slope - 0.5) < 0.03 && Math.abs(cc.citl - (yc.reduce((a, b) => a + b) - pc.reduce((a, b) => a + b)) / n) < 1e-9, JSON.stringify(cc));
const cvc = cvRecal(Array.from(pc), yc, 'continuous', { K: 5, reps: 1 });
ok('continuo: la recalibración reduce el RMSE', cvc.slope.rmse < cvc.orig.rmse * 0.5, `${cvc.slope.rmse} ${cvc.orig.rmse}`);
console.log(`\n${checks - fails}/${checks} comprobaciones correctas`); process.exit(fails ? 1 : 0);
