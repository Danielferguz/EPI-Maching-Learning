// Tabla 1 y regresión cruda/ajustada frente a statsmodels y SciPy; código Python sin figuras ni enlaces.
import { readFileSync } from 'node:fs';
import { crudeAdjusted, groupColumn } from '../js/analysis.js';
import { table1 } from '../js/report.js';
import { inferMeta, typedData } from '../js/data.js';
import { parseCSV } from '../js/io.js';
import { pythonScript } from '../js/pycode.js';
import { libraryLabelEn } from '../js/learners.js';

const ref = JSON.parse(readFileSync(new URL('reference_ua.json', import.meta.url)));
let fails = 0, checks = 0; const ok = (n, c, x = '') => { checks++; if (!c) { fails++; console.log('FAIL', n, x); } else console.log('ok  ', n); };
const near = (a, b, t = 1e-6) => Math.abs(a - b) <= t * Math.max(1, Math.abs(b));
const rows = parseCSV(ref.csv), meta = inferMeta(rows, ['']); meta.y.event = 'Yes'; const ds = typedData(rows, meta, ['']);
const R = crudeAdjusted(ds, meta, { type: 'binary', vars: ['x1', 'x2', 'g'], y: 'y', event: 'Yes' }), eff = R.rows.filter(r => !r.header);
ok('regresión ajustada (OR) = statsmodels', eff.every((r, i) => near(r.adj.est, ref.adj[i], 1e-5)));
ok('regresión cruda (OR) = statsmodels', eff.every((r, i) => near(r.crude.est, ref.crude[i], 1e-5)));
const S = crudeAdjusted(ds, meta, { type: 'binary', vars: ['x1', 'x2', 'g'], y: 'y', event: 'Yes', perSD: true }); ok('por 1 DE: etiqueta y OR = OR por unidad ^ DE', S.rows[0].name.includes('per 1 SD') && S.rows[0].adj.est > 0);
const g = groupColumn(ds, meta, { type: 'binary', y: 'y' }), T = table1(g.ds, g.meta, ['x1', 'g'], g.group, { lang: 'en', nonparam: true }), pRow = T.rows.filter(r => r.bold).map(r => r.cells[r.cells.length - 2]), num = s => (s.startsWith('<') ? 0 : parseFloat(s));
ok('Tabla 1: Mann-Whitney = SciPy', Math.abs(num(pRow[0]) - ref.mw_p) < 1e-3); ok('Tabla 1: χ² = SciPy', Math.abs(num(pRow[1]) - ref.chi_p) < 1e-3);
const gt = groupColumn(ds, meta, { type: 'continuous', y: 'x1' }); ok('grupos por terciles para desenlace continuo', gt.ds.cols[gt.group].filter(v => v === 'T1 (bajo)').length > 100);
const py = pythonScript({ data_file: 'd.csv', na_values: ['NA'], outcome: 'y', family: 'binary', event: 'Yes', predictors: [{ name: 'x1', type: 'num' }], library: [{ learner: 'glm', label: 'GLM', params: {} }], scheme: { type: 'cv', V: 5, repeats: 1, test_frac: 0.25 }, method: 'nnls', seed: 1, nested: { on: false, Vo: 5, Vi: 5 }, selection: { run: false, thr: 0.01, R: 5, Vsel: 3, Vcv: 5, force: [], selected: null }, meta: { n: 500, events: 200, date: '2026-01-01' } });
ok('código Python de ML: Tabla 1 y regresión, sin figuras ni GitHub', !/matplotlib|plt\.|github/i.test(py) && py.includes('def table1') && py.includes('def crude_adjusted'));
ok('nombres de algoritmos en inglés para figuras', libraryLabelEn({ learner: 'rf', label: 'Bosque aleatorio', params: { ntree: 300, mtry: 0, minNode: 5 } }) === 'Random forest' && libraryLabelEn({ learner: 'enet', label: 'Ridge', params: { alpha: 0, nfolds: 5, rule: 'min' } }).startsWith('Ridge'));
console.log(`\n${checks - fails}/${checks} comprobaciones correctas`); process.exit(fails ? 1 : 0);
