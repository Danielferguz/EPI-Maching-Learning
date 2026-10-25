// EPI-Maching-Learning · interfaz: datos, pregunta, algoritmos y validación, resultados, interpretación y reporte.
import { readFile, download, toCSV } from './io.js?v=2026.10.25-6';
import { inferMeta, typedData, NA_DEFAULT, missCount, levelsOf, isMiss } from './data.js?v=2026.10.25-6';
import { mean, sd } from './tests.js?v=2026.10.25-6';
import { LEARNERS, defaultLibrary, libraryLabel, libraryLabelEn } from './learners.js?v=2026.10.25-6';
import { call } from './runner.js?v=2026.10.25-6';
import { hbarPlot, rocMulti, calibrationPlot, scatterCalib, dcaPlot, pdpPlot } from './mlplots.js?v=2026.10.25-6';
import { calibGroups, lowess, calibSlope, dcaBinary, rocPts } from './evalutil.js?v=2026.10.25-6';
import { figWrap, bindFigs } from './figs.js?v=2026.10.25-6';
import { wordDoc, matrixHTML, matrixCSV, table1, table1HTML } from './report.js?v=2026.10.25-6';
import { groupColumn, crudeAdjusted, uniAdjTable } from './analysis.js?v=2026.10.25-6';
import { CONCEPTS_HTML, initLab } from './concepts.js?v=2026.10.25-6';
import { HELP_HTML } from './help.js?v=2026.10.25-6';
import { practiceHTML, practiceHint, practiceFor } from './practice.js?v=2026.10.25-6';
import { initRoadmap, initStageBar } from './roadmap.js?v=2026.10.25-6';
import { calibration, fitRecal, applyRecal, cvRecal, pickRecal } from './extval.js?v=2026.10.25-6';
import { metrics } from './sl.js?v=2026.10.25-6';
import { adviseLearners, adviseCV, rileyBinary, ST, HPNOTE } from './advice.js?v=2026.10.25-6';
import { pythonScript } from './pycode.js?v=2026.10.25-6';
import { calcHTML, calcPredict } from './calc.js?v=2026.10.25-6';
import { buildManuscript } from './manuscript.js?v=2026.10.25-6';

const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const f2 = (x, d = 3) => (isFinite(x) ? x.toFixed(d) : '–'), pct = x => (isFinite(x) ? (100 * x).toFixed(1) + '%' : '–');
const opt = (v, l, sel) => `<option value="${esc(v)}"${sel ? ' selected' : ''}>${esc(l ?? v)}</option>`;
const errBox = e => `<div class="err">${esc(e.message || e)}</div>`;
const table = (head, rows) => `<div class="tablewrap"><table class="data"><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.c.map((c, k) => `<td${k ? ' class="num"' : ''}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const state = { raw: [], meta: {}, ds: null, fname: '', spec: null, lib: [], res: null, prep: null };

// ---------------- ejemplos ----------------
const EXAMPLES = [
  { file: 'erc_descenso_rapido_manuscrito.csv', title: 'Manuscrito · descenso rápido de la TFGe (Super Learner, datos simulados)', design: 'Cohorte simulada · 800 pacientes · 2 centros', use: 'Ejemplo para redactar un manuscrito (ES/EN): Tabla 1, regresión cruda y ajustada, algoritmos y Super Learner, importancia, curva de decisión y calculadora. Inspirado en un estudio de un índice inflamatorio-nutricional; los datos son inventados.', meta: { 'Rapid eGFR decline': { event: 'Yes' } }, drop: ['Neutrophils (x10^3/uL)', 'Lymphocytes (x10^3/uL)', 'Monocytes (x10^3/uL)'], y: 'Rapid eGFR decline' },
  { file: 'rhc_ml.csv', title: 'UCI · mortalidad (cateterismo de arteria pulmonar)', design: 'Cohorte · 5735 pacientes · 55 variables', use: 'Clasificación con muchas variables (Taller de Machine Learning). ~2/3 mueren: calibración y desbalance moderado.', meta: { Death: { event: 'Yes' } }, drop: ['Length of stay (days)'], y: 'Death' },
  { file: 'rhc_ml.csv', title: 'UCI · duración de la estancia (días)', design: 'Cohorte · 5735 pacientes', use: 'Regresión con desenlace continuo asimétrico (como en el taller).', meta: {}, drop: ['Death'], y: 'Length of stay (days)' },
  { file: 'erc_prediccion_desarrollo.csv', title: 'ERC · falla renal a 2 años', design: 'Cohorte · 1800 pacientes', use: 'Clasificación con datos perdidos y relaciones no lineales (TFGe).', meta: { 'Falla renal a 2 años': { event: 'Sí' } }, drop: ['ID', 'Seguimiento (años)', 'Falla renal'], y: 'Falla renal a 2 años' },
  { file: 'lesion_renal_aguda.csv', title: 'Lesión renal aguda poscirugía cardiaca', design: 'Cohorte · 520 pacientes', use: 'Pocos eventos: observe cómo los algoritmos flexibles pierden frente a la regresión.', meta: { 'LRA (KDIGO)': { event: 'Sí' } }, drop: ['ID', 'Bypass prolongado (> 120 min)'], y: 'LRA (KDIGO)' },
  { file: 'hemodialisis_depresion.csv', title: 'Hemodiálisis · puntaje PHQ-9', design: 'Transversal · 260 pacientes', use: 'Regresión con n pequeño.', meta: {}, drop: ['ID', 'Depresión', 'Grupo de edad'], y: 'Puntaje PHQ-9' },
];
const naCodes = () => [...NA_DEFAULT, ...$('#na-codes').value.split(',').map(s => s.trim()).filter(Boolean)];

// ---------------- navegación ----------------
function showTab(t) { $$('.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === t)); $$('.tab').forEach(s => s.classList.toggle('active', s.id === 'tab-' + t)); if (t === 'algoritmos') { renderLib(); estimate(); } if (t === 'pregunta') renderQuestion(); stageMark?.(t); window.scrollTo({ top: 0 }); }
let stageMark = null;
const goTool = (tab, sel) => { showTab(tab); if (sel) setTimeout(() => { const el = $(sel); if (el && !el.closest('[hidden]')) el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 60); };
const rm = initRoadmap($('#roadmap'), { app: 'm', go: goTool });
stageMark = initStageBar($('#stagebar'), st => { showTab('ruta'); rm.show(st); });
$$('.tabs button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
document.addEventListener('click', e => { const g = e.target.closest('[data-go]'); if (g) showTab(g.dataset.go); });
const gate = () => { $$('.need-data').forEach(e => (e.hidden = !!state.ds)); $$('.has-data').forEach(e => (e.hidden = !state.ds)); $$('.need-res').forEach(e => (e.hidden = !!state.res)); };
$('#concepts').innerHTML = CONCEPTS_HTML; initLab(); $('#help').innerHTML = HELP_HTML;

// ---------------- datos ----------------
$('#examples').innerHTML = EXAMPLES.map((x, k) => `<button class="ghost" data-ex="${k}"><em>${esc(x.design)}</em><b>${esc(x.title)}</b><small>${esc(x.use)}</small></button>`).join('');
$$('[data-ex]').forEach(b => b.addEventListener('click', async () => {
  const x = EXAMPLES[+b.dataset.ex], txt = await (await fetch('examples/' + x.file)).text(); await loadFile(new File([txt], x.file));
  for (const [v, o] of Object.entries(x.meta || {})) if (state.meta[v]) Object.assign(state.meta[v], o);
  for (const v of x.drop || []) if (state.meta[v]) state.meta[v].type = 'skip';
  rebuild(); state.isExample = true; state.spec = { y: x.y, vars: null }; renderVarTable(); $$('[data-ex]').forEach(e => e.classList.toggle('active', e === b));
}));
$('#file').addEventListener('change', e => e.target.files[0] && loadFile(e.target.files[0]));
const drop = $('#drop'); ['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); })); ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); })); drop.addEventListener('drop', e => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));
async function loadFile(file) {
  try {
    const { sheets } = await readFile(file), name = Object.keys(sheets)[0]; state.isExample = false; state.fname = file.name.replace(/\.[^.]+$/, ''); state.raw = sheets[name]; if (!state.raw.length) throw new Error('El archivo no tiene filas.');
    state.meta = inferMeta(state.raw, naCodes()); state.res = null; state.spec = null; rebuild(); $('#res-out').innerHTML = ''; $('#int-out').innerHTML = ''; $('#rep-out').innerHTML = '';
    $('#load-msg').innerHTML = `<p class="pill">✓ ${esc(state.fname)}: ${state.raw.length} filas × ${Object.keys(state.meta).length} columnas</p>`; renderVarTable(); gate();
  } catch (e) { $('#load-msg').innerHTML = errBox(e); }
}
function rebuild() { state.ds = typedData(state.raw, Object.fromEntries(Object.entries(state.meta).filter(([, m]) => m.type !== 'skip')), naCodes()); }
$('#na-codes').addEventListener('change', () => { if (state.raw.length) { state.meta = inferMeta(state.raw, naCodes()); rebuild(); renderVarTable(); } });
const usable = (...t) => Object.values(state.meta).filter(m => m.type !== 'skip' && (!t.length || t.includes(m.type))).map(m => m.name);
function renderVarTable() {
  if (!state.ds) return; const T = $('#vartable');
  T.innerHTML = `<thead><tr><th>Variable</th><th>Tipo</th><th>Referencia</th><th>Evento</th><th class="num">Perdidos</th><th>Resumen</th></tr></thead><tbody>${Object.values(state.meta).map(m => {
    const col = state.ds.cols[m.name], miss = col ? missCount(col) : 0, isCat = m.type === 'cat'; let s = '';
    if (m.type === 'num' && col) { const v = Array.from(col).filter(isFinite); s = `media ${f2(mean(v), 2)} ± ${f2(sd(v), 2)}; rango ${f2(Math.min(...v), 2)}–${f2(Math.max(...v), 2)}`; } else if (isCat && col) { const c = {}; col.forEach(x => x != null && (c[x] = (c[x] || 0) + 1)); s = levelsOf(m).slice(0, 6).map(l => `${l} (${c[l] || 0})`).join(' · ') + (m.levels.length > 6 ? ' …' : ''); }
    return `<tr data-v="${esc(m.name)}"><td>${esc(m.name)}</td><td><select data-k="type">${opt('num', 'Cuantitativa', m.type === 'num')}${opt('cat', 'Categórica', m.type === 'cat')}${opt('id', 'Identificador', m.type === 'id')}${opt('skip', 'Excluir', m.type === 'skip')}</select></td><td>${isCat ? `<select data-k="ref">${m.levels.map(l => opt(l, l, l === m.ref)).join('')}</select>` : ''}</td><td>${isCat ? `<select data-k="event">${m.levels.map(l => opt(l, l, l === m.event)).join('')}</select>` : ''}</td><td class="num">${m.type === 'skip' ? '' : `${miss} (${pct(miss / state.raw.length)})`}</td><td class="sub" style="white-space:normal">${esc(s)}</td></tr>`;
  }).join('')}</tbody>`;
  $$('select', T).forEach(s => s.addEventListener('change', () => { const v = s.closest('tr').dataset.v, k = s.dataset.k, m = state.meta[v]; if (k === 'type' && s.value === 'num' && !m.allNum) { alert(`«${v}» contiene valores no numéricos.`); s.value = m.type; return; } m[k] = s.value; rebuild(); if (k === 'type') renderVarTable(); }));
}

// ---------------- pregunta ----------------
function renderQuestion() {
  if (!state.ds) return; const cand = usable('cat', 'num'), sp = state.spec || {};
  $('#q-y').innerHTML = cand.map(v => opt(v, v, v === sp.y)).join(''); updateY(sp.vars);
}
function updateY(vars = null) {
  const y = $('#q-y').value, m = state.meta[y]; if (!m) return; const bin = m.type === 'cat';
  $('#q-fam').value = bin ? 'Clasificación (probabilidad del evento)' : 'Regresión (valor numérico)'; $('#q-ev-w').hidden = !bin;
  $('#q-ev').innerHTML = (m.levels || []).map(l => opt(l, l, l === m.event)).join('');
  const others = usable('cat', 'num').filter(v => v !== y), keep = vars ? new Set(vars) : new Set(others);
  $('#q-vars').innerHTML = others.map(v => `<label class="pv"><input type="checkbox" data-pv="${esc(v)}"${keep.has(v) ? ' checked' : ''}> ${esc(v)}${state.meta[v].type === 'cat' ? ' <span class="badge cat">categ.</span>' : ''}</label>`).join('');
  $$('#q-vars input').forEach(i => i.addEventListener('change', qInfo)); qInfo();
}
$('#q-y').addEventListener('change', () => updateY()); $('#q-all').addEventListener('click', () => { $$('#q-vars input').forEach(i => (i.checked = true)); qInfo(); }); $('#q-none').addEventListener('click', () => { $$('#q-vars input').forEach(i => (i.checked = false)); qInfo(); });
function readSpec() { const y = $('#q-y').value, m = state.meta[y]; return { y, event: m.type === 'cat' ? $('#q-ev').value : null, fam: m.type === 'cat' ? 'binary' : 'continuous', vars: $$('#q-vars input:checked').map(i => i.dataset.pv) }; }
// matriz de diseño: numéricas tal cual; categóricas en indicadores (sin la referencia); NaN se conserva (se imputa dentro de cada pliegue)
export function buildMatrix(ds, meta, spec) {
  const rows = []; for (let i = 0; i < ds.n; i++) { const v = ds.cols[spec.y][i]; if (!isMiss(v)) rows.push(i); }
  const names = [], groups = [], cols = [];
  for (const v of spec.vars) {
    const m = meta[v], col = ds.cols[v];
    if (m.type === 'num') { groups.push({ name: v, cols: [names.length], kind: 'num' }); names.push(v); cols.push(rows.map(i => col[i])); }
    else { const lv = levelsOf(m), ix = []; lv.slice(1).forEach(l => { ix.push(names.length); names.push(`${v}: ${l}`); cols.push(rows.map(i => (isMiss(col[i]) ? NaN : String(col[i]) === l ? 1 : 0))); }); groups.push({ name: v, cols: ix, kind: 'cat', levels: lv }); }
  }
  const X = rows.map((_, r) => Float64Array.from(cols, c => c[r])), my = meta[spec.y];
  const y = Float64Array.from(rows, i => (spec.fam === 'binary' ? (String(ds.cols[spec.y][i]) === String(spec.event) ? 1 : 0) : ds.cols[spec.y][i]));
  return { X, y, names, groups, rows };
}
// orientación según n, eventos y columnas (semáforo de algoritmos, tamaño mínimo de Riley y notas de validación)
function adviceNow() {
  const s = readSpecSafe(); if (!s || !s.vars.length || !state.ds) return null;
  const b = buildMatrix(state.ds, state.meta, s), n = b.X.length, p = b.names.length, ev = s.fam === 'binary' ? b.y.reduce((a, c) => a + c, 0) : null;
  let riley = null, rileyErr = ''; if (ev != null) try { riley = rileyBinary({ cstat: +$('#q-cexp').value || 0.75, prev: ev / n, p }); } catch (e) { rileyErr = e.message; }
  return { n, p, events: ev, fam: s.fam, riley, rileyErr, learners: adviseLearners({ n, events: ev ?? 0, p, fam: s.fam }), cv: adviseCV({ n, events: ev ?? 0, fam: s.fam }) };
}
function renderAdvice() {
  const A = adviceNow(), box = $('#q-advice'); $('#q-adv-wrap').hidden = !A; if (!A) { box.innerHTML = ''; return; }
  let riley = '';
  if (A.fam === 'binary') { const r = A.riley; riley = r ? (() => { const okN = A.n >= r.n; return `<p><b>Regresión (criterios de Riley):</b> con ${A.p} columnas candidatas, AUC esperado ${f2(+$('#q-cexp').value || 0.75, 2)} y prevalencia ${pct(A.events / A.n)}, el mínimo es ≈ <b>${r.n} pacientes (${r.events} eventos)</b>. Tiene ${A.n} (${A.events}): <b style="color:${okN ? '#2E7D32' : '#B3261E'}">${okN ? 'suficiente' : 'insuficiente'}</b>${okN ? '' : ': reduzca predictores, use regresión penalizada y evite algoritmos flexibles'}. Los algoritmos flexibles necesitan bastante más.</p>`; })() : `<p class="sub">${esc(A.rileyErr)}</p>`; }
  else riley = `<p><b>Regresión:</b> como mínimo ≈ 10–15 pacientes por columna candidata; tiene ${f2(A.n / A.p, 1)} (n = ${A.n}, ${A.p} columnas). Los algoritmos flexibles necesitan bastante más.</p>`;
  state.advice = A;
  box.innerHTML = riley + table(['Algoritmo', 'Valoración', 'Motivo'], Object.entries(LEARNERS).map(([id, L]) => { const a = A.learners[id]; return { c: [esc(L.name), `<b style="color:${ST[a.st][2]}">${ST[a.st][0]} ${ST[a.st][1]}</b>`, esc(a.why)] }; })).replace(/ class="num"/g, '');
}
$('#q-cexp').addEventListener('input', renderAdvice);
// Tabla 1 y regresión cruda/ajustada (se recalculan al cambiar el desenlace o los predictores)
let descTimer;
const descSpec = () => { const s = readSpec(); return { type: s.fam === 'binary' ? 'binary' : 'continuous', y: s.y, event: s.event, vars: s.vars }; };
const t1Now = (lang, np) => { const sp = descSpec(), g = groupColumn(state.ds, state.meta, sp); return { ...table1(g.ds, g.meta, sp.vars, g.group, { lang, nonparam: np }), group: g.label }; };
const uaNow = lang => { const sp = descSpec(), R = crudeAdjusted(state.ds, state.meta, { type: sp.type, vars: sp.vars, y: sp.y, event: sp.event }); return { R, T: uniAdjTable(R, lang) }; };
const t1For = (sp0, vars, lang, np) => { const sp = { type: sp0.fam === 'binary' ? 'binary' : 'continuous', y: sp0.y, event: sp0.event }, g = groupColumn(state.ds, state.meta, sp); return { ...table1(g.ds, g.meta, vars, g.group, { lang, nonparam: np }), group: g.label }; };
const uaFor = (sp0, lang) => { const sp = { type: sp0.fam === 'binary' ? 'binary' : 'continuous', y: sp0.y, event: sp0.event }, R = crudeAdjusted(state.ds, state.meta, { ...sp, vars: sp0.vars, perSD: !!$('#ua-sd')?.checked }); return { R, T: uniAdjTable(R, lang) }; };
const tHTML = T => `<div class="tablewrap"><table class="data t1"><thead><tr>${T.head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${T.rows.map((r, i) => `<tr>${r.map((c, k) => `<td${T.bold[i] && k === 0 ? ' class="b"' : ''}>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
function doDesc() {
  const box = $('#q-desc'); if (!state.ds || !$('#q-y').value || !$$('#q-vars input:checked').length) { box.hidden = true; return; } box.hidden = false;
  try { const T = t1Now($('#t1-lang').value, $('#t1-np').checked); $('#t1-out').innerHTML = `<p class="sub">Grupos: ${esc(T.group)}</p>${table1HTML(T)}`; } catch (e) { $('#t1-out').innerHTML = errBox(e); }
  try { const { R, T } = uaNow($('#t1-lang').value); state.ua = R; $('#ua-out').innerHTML = `<p class="sub">${R.n} pacientes con datos completos${R.events != null ? ` · ${R.events} eventos` : ''}${R.dropped ? ` (${R.dropped} excluidos por datos perdidos)` : ''}${R.adjOk ? '' : ' · el modelo ajustado no convergió (colinealidad o separación)'}.</p>${tHTML(T)}`; } catch (e) { $('#ua-out').innerHTML = errBox(e); }
}
['#t1-np', '#t1-lang', '#ua-sd'].forEach(id => $(id).addEventListener('change', doDesc));
$('#t1-doc').addEventListener('click', () => { const T = t1Now($('#t1-lang').value, $('#t1-np').checked); download('tabla1.doc', wordDoc('Table 1', matrixHTML({ head: T.head, body: T.rows.map(r => ({ c: r.cells, b: r.bold })) }), T.note), 'application/msword'); });
$('#t1-csv').addEventListener('click', () => { const T = t1Now($('#t1-lang').value, $('#t1-np').checked); download('tabla1.csv', matrixCSV({ head: T.head, body: T.rows.map(r => ({ c: r.cells })) }), 'text/csv'); });
$('#ua-doc').addEventListener('click', () => { const { T } = uaNow($('#t1-lang').value); download('regresion_cruda_ajustada.doc', wordDoc('Table 2', matrixHTML({ head: T.head, body: T.rows.map((r, i) => ({ c: r, b: T.bold[i] })) })), 'application/msword'); });
$('#ua-csv').addEventListener('click', () => { const { T } = uaNow($('#t1-lang').value); download('regresion_cruda_ajustada.csv', matrixCSV({ head: T.head, body: T.rows.map(r => ({ c: r })) }), 'text/csv'); });
function qInfo() {
  const s = readSpec(); if (!s.vars.length) { $('#q-info').textContent = 'Marque al menos un predictor.'; $('#q-warn').innerHTML = ''; $('#q-adv-wrap').hidden = true; $('#q-desc').hidden = true; return; }
  const b = buildMatrix(state.ds, state.meta, s), ev = s.fam === 'binary' ? b.y.reduce((a, c) => a + c, 0) : null, nm = b.X.reduce((a, x) => a + (x.some(Number.isNaN) ? 1 : 0), 0);
  $('#q-info').textContent = `${b.X.length} pacientes · ${s.vars.length} variables → ${b.names.length} columnas${ev != null ? ` · ${ev} eventos (${pct(ev / b.X.length)})` : ''} · ${nm} filas con algún dato perdido (se imputan)`;
  const w = []; if (ev != null && ev / b.names.length < 10) w.push(`Hay ${f2(ev / b.names.length, 1)} eventos por columna: riesgo alto de sobreajuste para los algoritmos flexibles (van der Ploeg et al. sugieren > 100 eventos por variable para bosques y redes). Prefiera regresión penalizada y compare con la referencia simple.`);
  if (ev != null && Math.min(ev, b.X.length - ev) < 30) w.push('Muy pocos pacientes en una de las categorías del desenlace.');
  if (b.X.length < 200) w.push('Con menos de 200 pacientes los resultados serán inestables; repita la validación cruzada.');
  $('#q-warn').innerHTML = w.map(t => `<div class="warn">${esc(t)}</div>`).join(''); renderAdvice(); clearTimeout(descTimer); descTimer = setTimeout(doDesc, 300);
}

// ---------------- biblioteca ----------------
const PD = { enet: { alpha: ['α (1 = lasso, 0 = ridge)', 0.1], nfolds: ['Pliegues internos', 1], rule: ['Regla de λ', ['min', '1se']] }, knn: { k: ['k vecinos', 1] }, tree: { maxDepth: ['Profundidad máx.', 1], minNode: ['Hoja mínima', 1], cp: ['cp (mejora mínima)', 0.001] }, rf: { ntree: ['N.º de árboles', 50], mtry: ['mtry (0 = automático)', 1], minNode: ['Hoja mínima', 1] }, gbm: { nrounds: ['Rondas', 10], lr: ['Tasa de aprendizaje', 0.01], depth: ['Profundidad', 1], minNode: ['Hoja mínima', 1], subsample: ['Submuestra', 0.1] }, nnet: { size: ['Neuronas', 1], decay: ['Decay (L2)', 0.005], epochs: ['Épocas', 50], lr: ['Tasa', 0.005] } };
const PRESETS = { fast: ['mean', 'glm', 'enet', 'rf', 'gbm'], full: ['mean', 'glm', 'enet', 'knn', 'tree', 'rf', 'gbm', 'nnet'], simple: ['mean', 'glm', 'enet'] };
function setPreset(k) { state.lib = PRESETS[k].map(id => ({ learner: id, params: { ...LEARNERS[id].params }, label: LEARNERS[id].name, on: true })); if (k === 'full') { state.lib.splice(3, 0, { learner: 'enet', params: { ...LEARNERS.enet.params, alpha: 0 }, label: 'Ridge', on: true }); } if (k === 'fast') state.lib.splice(3, 0, { learner: 'enet', params: { ...LEARNERS.enet.params, alpha: 0 }, label: 'Ridge', on: false }); renderLib(); }
$$('[data-preset]').forEach(b => b.addEventListener('click', () => setPreset(b.dataset.preset)));
function renderLib() {
  if (!state.lib.length) setPreset('fast'); const AD = adviceNow()?.learners;
  $('#lib').innerHTML = `<div class="tablewrap"><table class="data"><thead><tr><th></th><th>Algoritmo</th><th>Hiperparámetros</th><th></th></tr></thead><tbody>${state.lib.map((c, i) => `<tr><td><input type="checkbox" data-on="${i}"${c.on ? ' checked' : ''}></td><td style="white-space:normal;min-width:220px"><b>${esc(c.label)}</b>${AD ? ` <span class="flag" style="color:${ST[AD[c.learner].st][2]}" title="${esc(AD[c.learner].why)}">${ST[AD[c.learner].st][0]} ${ST[AD[c.learner].st][1]}</span>` : ''}<br><span class="sub">${esc(LEARNERS[c.learner].desc)}${AD ? ' ' + esc(AD[c.learner].why) : ''}</span><details class="sub"><summary>Qué valores usar</summary>${esc(HPNOTE[c.learner])}</details></td><td>${Object.entries(PD[c.learner] || {}).map(([k, d]) => Array.isArray(d[1]) ? `<label class="inline sub">${d[0]} <select data-pi="${i}|${k}">${d[1].map(o => opt(o, o, c.params[k] === o)).join('')}</select></label>` : `<label class="inline sub">${d[0]} <input data-pi="${i}|${k}" type="number" step="${d[1]}" value="${c.params[k]}" style="width:78px"></label>`).join(' ') || '<span class="sub">sin hiperparámetros</span>'}</td><td><button class="x" data-dup="${i}" title="Duplicar como variante">⧉</button><button class="x" data-del="${i}" title="Quitar">×</button></td></tr>`).join('')}</tbody></table></div><div class="row gap wrap mt"><span class="sub">Añadir:</span>${Object.keys(LEARNERS).map(id => `<button class="ghost sm" data-add="${id}">${esc(LEARNERS[id].name)}</button>`).join('')}</div>`;
  $$('[data-on]').forEach(e => e.addEventListener('change', () => { state.lib[+e.dataset.on].on = e.checked; estimate(); }));
  $$('[data-pi]').forEach(e => e.addEventListener('input', () => { const [i, k] = e.dataset.pi.split('|'); state.lib[+i].params[k] = e.tagName === 'SELECT' ? e.value : +e.value; }));
  $$('[data-dup]').forEach(e => e.addEventListener('click', () => { const c = state.lib[+e.dataset.dup]; state.lib.splice(+e.dataset.dup + 1, 0, { ...c, params: { ...c.params }, label: c.label + ' (variante)' }); renderLib(); }));
  $$('[data-del]').forEach(e => e.addEventListener('click', () => { state.lib.splice(+e.dataset.del, 1); renderLib(); }));
  $$('[data-add]').forEach(e => e.addEventListener('click', () => { const id = e.dataset.add; state.lib.push({ learner: id, params: { ...LEARNERS[id].params }, label: LEARNERS[id].name, on: true }); renderLib(); }));
}
const sch = () => ({ type: $('#v-type').value, V: +$('#v-V').value, repeats: +$('#v-rep').value || 1, testFrac: +$('#v-test').value });
function estimate() {
  if (!state.ds) return; const t = $('#v-type').value; $$('.cvonly').forEach(e => (e.hidden = t !== 'cv')); $$('.hoonly').forEach(e => (e.hidden = t !== 'holdout'));
  let fam = 'binary'; try { fam = readSpec().fam; } catch { /* sin especificar */ } $$('.binonly').forEach(e => (e.hidden = fam !== 'binary'));
  const s = readSpecSafe(), n = s ? buildMatrix(state.ds, state.meta, s).X.length : 0, K = state.lib.filter(c => c.on).length, V = +$('#v-V').value, cv = $('#v-cvsl').checked && t === 'cv';
  const AV = adviceNow(); $('#v-notes').innerHTML = AV ? `<p>${AV.cv.text}</p>` : '';
  $('#v-est').innerHTML = `<p>Se ajustarán ≈ <b>${K * V * (t === 'cv' ? +$('#v-rep').value || 1 : 1) + K + (cv ? (+$('#v-vo').value) * (K * (+$('#v-vi').value) + K) : 0)}</b> modelos${n > 3000 && state.lib.some(c => c.on && ['rf', 'gbm', 'nnet', 'knn'].includes(c.learner)) ? '. Con n &gt; 3000 los bosques, el boosting y las redes pueden tardar varios minutos: empiece con el preset rápido y menos pliegues.' : '.'}</p>`;
}
const readSpecSafe = () => { try { return $('#q-y').value ? readSpec() : state.spec && state.spec.y ? { ...state.spec, fam: state.meta[state.spec.y].type === 'cat' ? 'binary' : 'continuous', vars: state.spec.vars || usable('cat', 'num').filter(v => v !== state.spec.y), event: state.meta[state.spec.y].event } : null; } catch { return null; } };
['#v-type', '#v-V', '#v-rep', '#v-cvsl', '#v-vo', '#v-vi'].forEach(s => $(s).addEventListener('input', estimate));

// ---------------- ejecución ----------------
$('#run').addEventListener('click', async () => {
  const btn = $('#run'), prog = $('#prog'), msg = $('#run-msg');
  try {
    if (!$('#q-y').value) renderQuestion(); const spec = readSpec(); if (!spec.vars.length) throw new Error('Marque al menos un predictor en el paso 3.');
    const lib = state.lib.filter(c => c.on).map(c => ({ learner: c.learner, params: c.params, label: c.label })); if (!lib.length) throw new Error('Seleccione al menos un algoritmo.');
    const b = buildMatrix(state.ds, state.meta, spec); if (spec.fam === 'binary' && (Math.min(...[b.y.reduce((a, c) => a + c, 0), b.y.length - b.y.reduce((a, c) => a + c, 0)]) < 10)) throw new Error('El desenlace tiene menos de 10 pacientes en una categoría.');
    btn.disabled = true; prog.hidden = false; prog.value = 0; msg.textContent = 'Iniciando…';
    const scheme = sch(), t0 = performance.now(), method = $('#v-method').value === 'loglik' && spec.fam === 'binary' ? 'loglik' : 'nnls', cvsl = { on: $('#v-cvsl').checked && scheme.type === 'cv', Vouter: +$('#v-vo').value, Vinner: +$('#v-vi').value };
    const res = await call('fit', { X: b.X, y: b.y, fam: spec.fam, lib, scheme, method, cvsl, seed: +$('#v-seed').value || 1 }, p => { prog.value = Math.min(1, p.done / p.total); msg.textContent = `${p.stage}: ${Math.floor(p.done)} de ${p.total}`; });
    state.res = { ...res, spec, scheme, method, cvslOn: cvsl.on, cvsl, seed: +$('#v-seed').value || 1, lib, b, secs: (performance.now() - t0) / 1000, simp: state.pendingSimp || null, advice: adviceNow() }; state.pendingSimp = null; state.imp = null; state.pdp = null; state.exportM = null;
    prog.hidden = true; msg.textContent = `Listo en ${state.res.secs.toFixed(1)} s.`; gate(); renderResults(); $('#int-out').innerHTML = ''; showTab('resultados'); renderInterp(); renderReport();
  } catch (e) { prog.hidden = true; msg.innerHTML = ''; $('#v-est').insertAdjacentHTML('beforeend', errBox(e)); console.error(e); }
  btn.disabled = false;
});

// ---------------- resultados ----------------
function metricRow(label, m, fam, w, extra = '') {
  return fam === 'binary' ? [label, `${f2(m.auc)} (${f2(m.aucLo)}–${f2(m.aucHi)})`, f2(m.brier, 4), f2(m.brierScaled), f2(m.logloss), f2(m.oe, 2), w == null ? '' : f2(w, 3), extra] : [label, f2(m.rmse), f2(m.mae), f2(m.r2), w == null ? '' : f2(w, 3), extra];
}
function renderResults() {
  const R = state.res, LB = R.lib.map(libraryLabelEn), fam = R.spec.fam, bin = fam === 'binary', K = R.labels.length, head = bin ? ['Algoritmo', 'AUC (IC 95 %)', 'Brier', 'Brier escalado', 'Log-loss', 'O/E', 'Peso SL', 'Tiempo (s)'] : ['Algoritmo', 'RMSE', 'MAE', 'R²', 'Peso SL', 'Tiempo (s)'];
  const key = bin ? 'auc' : 'rmse', better = (a, b) => (bin ? a > b : a < b);
  const rows = R.labels.map((l, k) => ({ c: metricRow(esc(l), R.perLearner[k], fam, R.weights[k], f2(R.time[k] / 1000, 1)), m: R.perLearner[k] }));
  const bestIdx = rows.reduce((bi, r, k) => (better(r.m[key], rows[bi].m[key]) ? k : bi), 0); rows[bestIdx].cls = 'best';
  const body = [...rows, { c: metricRow('<b>Super Learner (conjunto)</b> <span class="flag">optimista</span>', R.ensCV, fam, null) , cls: 'sl' }, { c: metricRow('Super Learner discreto (' + esc(R.labels[R.discrete]) + ')', R.discCV, fam, null) }];
  if (R.honest) body.push({ c: metricRow('<b>Super Learner · validación cruzada anidada</b>', R.honest.ens, fam, null), cls: 'sl' }, { c: metricRow('Super Learner discreto · anidada', R.honest.disc, fam, null) });
  const cmp = [...LB.map((l, k) => ({ label: l, value: R.honest ? R.honest.each[k][key] : R.perLearner[k][key] })), ...(R.honest ? [{ label: 'Super Learner (nested CV)', value: R.honest.ens[key], color: '#8B2E3B' }] : [{ label: 'Super Learner (apparent CV)', value: R.ensCV[key], color: '#8B2E3B' }])];
  const lo = bin ? 0.5 : 0, charts = [];
  charts.push(figWrap(hbarPlot(cmp, { title: bin ? 'AUC by algorithm' : 'RMSE by algorithm', xlab: bin ? 'AUC (cross-validation)' : 'RMSE (cross-validation)', xmin: bin ? 0.5 : 0, fmt: v => v.toFixed(3) }), 'comparacion_algoritmos'));
  charts.push(figWrap(hbarPlot(LB.map((l, k) => ({ label: l, value: R.weights[k] })), { title: 'Super Learner weights', xlab: 'Weight (sum to 1)', xmin: 0, xmax: 1, fmt: v => v.toFixed(2) }), 'pesos_super_learner'));
  const predBest = R.honest ? R.honest.ensPred : R.ensOOF;
  if (bin) {
    const curves = [...R.Z.map((z, k) => ({ label: LB[k], pts: rocPts(z, R.y) })), { label: 'Super Learner' + (R.honest ? ' (nested CV)' : ''), pts: rocPts(predBest, R.y), color: '#8B2E3B', bold: true }];
    charts.push(figWrap(rocMulti(curves), 'roc_algoritmos'));
    const g = calibGroups(predBest, R.y, 10), cal = calibSlope(predBest, R.y), curve = lowess(predBest, R.y, 2 / 3, 40).map(([a, b]) => [a, Math.min(1, Math.max(0, b))]);
    charts.push(figWrap(calibrationPlot({ groups: g, curve, pred: predBest, y: R.y, title: 'Super Learner calibration' + (R.honest ? ' (nested CV)' : ''), note: `Slope ${f2(cal.slope, 2)} · intercept ${f2(cal.intercept, 2)}` }), 'calibracion_super_learner'));
  } else charts.push(figWrap(scatterCalib({ pred: Array.from(predBest), y: Array.from(R.y), curve: lowess(predBest, R.y, 2 / 3, 40), title: 'Super Learner: observed vs predicted' + (R.honest ? ' (nested CV)' : '') }), 'observado_vs_predicho'));
  const hold = R.test ? `<h3>Conjunto de prueba (n = ${R.nTest}) · tocado una sola vez</h3>${table(head.slice(0, bin ? 6 : 4), [...R.labels.map((l, k) => ({ c: metricRow(esc(l), R.test.each[k], fam, null).slice(0, bin ? 6 : 4) })), { c: metricRow('<b>Super Learner</b>', R.test.ens, fam, null).slice(0, bin ? 6 : 4), cls: 'sl' }, { c: metricRow('Super Learner discreto', R.test.disc, fam, null).slice(0, bin ? 6 : 4) }])}` : '';
  const sch_ = R.scheme.type === 'cv' ? `validación cruzada de ${R.scheme.V} pliegues${R.scheme.repeats > 1 ? ` repetida ${R.scheme.repeats} veces` : ''}` : `entrenamiento/prueba (${pct(1 - R.scheme.testFrac)} / ${pct(R.scheme.testFrac)}) con validación cruzada de ${R.scheme.V} pliegues en el entrenamiento`;
  $('#res-out').innerHTML = `<div class="card"><h2>Comparación de algoritmos</h2><div class="howto"><div><b>Qué se ve</b>Rendimiento de cada algoritmo en pacientes que no usó para aprender (${sch_}; n = ${R.n}), y del Super Learner que los combina.</div><div><b>Para qué</b>Elegir el algoritmo (o el conjunto) que mejor predice en datos nuevos y comprobar si la complejidad aporta algo frente a la regresión simple.</div><div><b>Cómo interpretar</b>Diferencias pequeñas (p. ej., AUC &lt; 0.01–0.02) suelen ser ruido. La fila «Super Learner (conjunto)» es optimista porque los pesos se ajustaron con esas mismas predicciones: la fila anidada es la honesta.</div></div>${table(head, body)}<p class="sub">Resaltado: mejor algoritmo individual por ${bin ? 'AUC' : 'RMSE'}. ${bin ? 'Brier escalado = 1 − Brier/(prev·(1−prev)); O/E = observados/esperados.' : ''} Peso: contribución en el Super Learner (${R.method === 'loglik' ? 'máxima verosimilitud binomial' : 'mínimos cuadrados no negativos'}).</p>${hold}${bin ? '<div class="howto"><div><b>Barras por algoritmo</b>El AUC (o RMSE) de cada algoritmo con predicciones fuera de muestra, más el Super Learner.</div><div><b>ROC</b>Sensibilidad frente a 1 − especificidad; más cerca de la esquina superior izquierda = mejor discriminación.</div><div><b>Calibración</b>Riesgo observado frente al predicho: la diagonal es lo ideal; por encima, el modelo subestima; por debajo, sobrestima. Pesos: cuánto aporta cada algoritmo al conjunto.</div></div>' : '<div class="howto"><div><b>Barras por algoritmo</b>El RMSE de cada algoritmo con predicciones fuera de muestra (menor = mejor).</div><div><b>Observado vs predicho</b>Los puntos deberían seguir la diagonal; una curva que se aparta indica sesgo en parte del rango.</div><div><b>Pesos</b>Cuánto aporta cada algoritmo al Super Learner.</div></div>'}<div class="plots">${charts.join('')}</div></div><div class="stepnav"><button class="ghost" data-go="algoritmos">← Algoritmos</button><button class="primary" data-go="interpretacion">Siguiente: interpretación →</button></div>`;
  bindFigs($('#res-out')); renderXv();
}

// ---------------- validación externa y recalibración ----------------
const xv = { raw: null, name: '' };
function renderXv() {
  const R = state.res; if (!R) { $('#xv-out').innerHTML = ''; return; } xv.raw = null;
  $('#xv-out').innerHTML = `<div class="card" id="xv-card"><h2>Validación externa y recalibración</h2><div class="howto"><div><b>Qué se hace</b>Se aplica el Super Learner ya entrenado (sin reajustarlo) a pacientes de otro centro, otro periodo u otra población, y se mide su discriminación y calibración.</div><div><b>Para qué</b>La validación cruzada solo corrige el sobreajuste; la externa muestra si el modelo se transporta. Si discrimina bien pero se calibra mal, se puede recalibrar (nivel de riesgo y/o escala) en lugar de rehacerlo.</div><div><b>Cómo interpretar</b>Pendiente &lt; 0.9: predicciones demasiado extremas (sobreajuste); intercepto (CITL) ≠ 0 o O/E ≠ 1: el riesgo medio no coincide. La recalibración se compara por validación cruzada dentro de la base de validación.</div></div>
    <div class="drop slim"><input type="file" id="xv-file" accept=".csv,.txt,.tsv,.xlsx,.xls"><label for="xv-file"><b>Cargar la base de validación (CSV/Excel)</b> con las mismas columnas, o <a href="#" id="xv-ex">usar la base de práctica de este ejemplo</a></label></div><div id="xv-practice"></div>
    <p class="sub" id="xv-info"></p><div class="row gap wrap"><button class="primary" id="xv-run">Evaluar</button><span id="xv-msg" class="muted"></span></div><div id="xv-res"></div></div>`;
  const load = async file => { try { const { sheets } = await readFile(file), raw = sheets[Object.keys(sheets)[0]]; if (!raw.length) throw new Error('El archivo no tiene filas.'); xv.raw = raw; xv.name = file.name; $('#xv-info').innerHTML = `<span class="pill">✓ ${esc(file.name)}: ${raw.length} filas × ${Object.keys(raw[0]).length} columnas</span>`; } catch (e) { $('#xv-info').innerHTML = errBox(e); } };
  $('#xv-file').addEventListener('change', e => e.target.files[0] && load(e.target.files[0]));
  $('#xv-practice').innerHTML = practiceHTML(state.fname);
  const loadPractice = async file => { try { await load(new File([await (await fetch('examples/' + file)).text()], file)); } catch (err) { $('#xv-info').innerHTML = errBox(err); } };
  $('#xv-ex').addEventListener('click', e => { e.preventDefault(); const c = practiceFor(state.fname); if (c) loadPractice(c.file); else $('#xv-info').innerHTML = errBox('El ejemplo de desarrollo cargado no tiene base de práctica emparejada: elija una de la lista (tras cargar su ejemplo en el paso 2 y entrenar) o cargue su propia base.'); });
  $('#xv-practice').addEventListener('click', e => { const b = e.target.closest('[data-practice]'); if (b) loadPractice(b.dataset.practice); });
  $('#xv-run').addEventListener('click', runXv);
}
async function runXv() {
  const msg = $('#xv-msg'), out = $('#xv-res'); out.innerHTML = ''; msg.textContent = 'Evaluando…';
  try {
    const R = state.res, fam = R.spec.fam, bin = fam === 'binary'; if (!xv.raw) throw new Error('Cargue primero la base de validación.');
    const miss = [...R.spec.vars, R.spec.y].filter(v => !(v in xv.raw[0])); if (miss.length) throw new Error(`La base de validación no contiene: ${miss.join(', ')}.${practiceHint(state.fname)}`);
    const active = Object.fromEntries(Object.entries(state.meta).filter(([k, m]) => m.type !== 'skip' && k in xv.raw[0])), ds = typedData(xv.raw, active, naCodes()), b = buildMatrix(ds, state.meta, R.spec);
    if (b.y.length < 30) throw new Error('Muy pocos pacientes con desenlace en la base de validación.');
    if (bin && (b.y.reduce((a, c) => a + c, 0) < 5 || b.y.reduce((a, c) => a + c, 0) > b.y.length - 5)) throw new Error('Se necesitan al menos 5 pacientes con y 5 sin el evento.');
    const o = await call('predict', { X: b.X }), p = Float64Array.from(o.ens), y = b.y, m = metrics(p, y, fam), c = calibration(p, y, fam), ev = bin ? y.reduce((a, v) => a + v, 0) : null;
    const devP = R.honest ? R.honest.ensPred : R.ensOOF, devM = R.honest ? R.honest.ens : R.ensCV, devC = calibration(devP, R.y, fam), z = 1.96, ci = (e, se, d = 3) => `${f2(e, d)} (${f2(e - z * se, d)} a ${f2(e + z * se, d)})`;
    const head = ['Medida', 'Validación externa', 'Desarrollo (validación cruzada)', 'Ideal'];
    const rows = bin ? [{ c: ['Estadístico C (AUC)', `${f2(m.auc)} (${f2(m.aucLo)} a ${f2(m.aucHi)})`, f2(devM.auc), ''] }, { c: ['Pendiente de calibración', ci(c.slope, c.slopeSE), f2(devC.slope), '1'] }, { c: ['Calibración en general (CITL)', ci(c.citl, c.citlSE), f2(devC.citl), '0'] }, { c: ['Observados/esperados (O/E)', f2(c.oe), f2(devM.oe), '1'] }, { c: ['Brier (escalado)', `${f2(m.brier, 4)} (${f2(m.brierScaled)})`, `${f2(devM.brier, 4)} (${f2(devM.brierScaled)})`, ''] }]
      : [{ c: ['RMSE', f2(m.rmse), f2(devM.rmse), ''] }, { c: ['MAE', f2(m.mae), f2(devM.mae), ''] }, { c: ['R²', f2(m.r2), f2(devM.r2), '1'] }, { c: ['Pendiente de calibración', ci(c.slope, c.slopeSE), f2(devC.slope), '1'] }, { c: ['CITL (media observada − predicha)', f2(c.citl), f2(devC.citl), '0'] }];
    const plots = bin ? figWrap(calibrationPlot({ groups: calibGroups(p, y, 10), curve: lowess(p, y, 2 / 3, 40).map(([a, q]) => [a, Math.min(1, Math.max(0, q))]), pred: p, y, title: 'External validation: calibration', note: `Slope ${f2(c.slope, 2)} · CITL ${f2(c.citl, 2)} · O/E ${f2(c.oe, 2)}` }), 'calibracion_validacion_externa') + figWrap(rocMulti([{ label: 'Super Learner', pts: rocPts(p, y), color: '#8B2E3B', bold: true }], 'ROC curve (external validation)'), 'roc_validacion_externa')
      : figWrap(scatterCalib({ pred: Array.from(p), y: Array.from(y), curve: lowess(p, y, 2 / 3, 40), title: 'External validation: observed vs predicted' }), 'observado_vs_predicho_validacion');
    const cv = cvRecal(Array.from(p), y, fam), pick = pickRecal(fam, cv), name = { orig: 'Modelo original (sin recalibrar)', citl: 'Recalibración en general (solo intercepto)', slope: 'Recalibración con intercepto y pendiente' };
    const rh = bin ? ['Estrategia', 'C (AUC)', 'Pendiente', 'CITL', 'O/E', 'Brier'] : ['Estrategia', 'R²', 'Pendiente', 'CITL', 'RMSE'], rr = Object.keys(cv).map(k => ({ cls: k === pick.id ? 'best' : '', c: [name[k], ...(bin ? [f2(cv[k].auc), f2(cv[k].slope), f2(cv[k].citl), f2(cv[k].oe), f2(cv[k].brier, 4)] : [f2(cv[k].r2), f2(cv[k].slope), f2(cv[k].citl), f2(cv[k].rmse)])] }));
    const rc = fitRecal(p, y, fam, pick.id === 'orig' ? 'slope' : pick.id), eq = bin ? `riesgo recalibrado = expit(${f2(rc.a, 3)} + ${f2(rc.b, 3)} × logit(riesgo del Super Learner))` : `valor recalibrado = ${f2(rc.a, 3)} + ${f2(rc.b, 3)} × valor del Super Learner`;
    out.innerHTML = `${bin && ev < 100 ? `<p class="warn">La base de validación tiene ${ev} eventos: con menos de ≈ 100 eventos, las medidas de calibración son muy imprecisas.</p>` : ''}<p class="sub">${b.y.length} pacientes${bin ? ` (${ev} eventos)` : ''} de «${esc(xv.name)}».</p>${table(head, rows)}<div class="plots">${plots}</div>
      <h3>Recalibración</h3><div class="explain">Se ajusta una recta (en la escala logit del riesgo si es binario) entre lo predicho y lo observado: el <b>intercepto</b> corrige el nivel medio del riesgo y la <b>pendiente</b> la dispersión de las predicciones. Rendimiento por validación cruzada (5 pliegues × 3 repeticiones) dentro de la base de validación:</div>${table(rh, rr)}
      <p><b>Sugerencia:</b> ${esc(name[pick.id])} — ${esc(pick.why)}.${pick.id === 'orig' ? ' El modelo original ya está bien calibrado en este entorno.' : ''}</p><p class="sub">Con todos los pacientes de la validación: ${esc(eq)}. Esta corrección debe validarse en datos independientes; no cambia el orden de los pacientes (el AUC no cambia).</p>
      <div class="row gap wrap"><button class="ghost" id="xv-dl">Descargar predicciones (original y recalibradas)</button></div>`;
    bindFigs(out);
    $('#xv-dl').addEventListener('click', () => { const r1 = fitRecal(p, y, fam, 'citl'), r2 = fitRecal(p, y, fam, 'slope'), a1 = applyRecal(p, r1, fam), a2 = applyRecal(p, r2, fam), cols = ['fila', 'observado', 'predicho', 'recalibrado_intercepto', 'recalibrado_intercepto_pendiente']; download('validacion_externa_predicciones.csv', toCSV(Array.from(p, (v, i) => Object.fromEntries(cols.map((k, j) => [k, [b.rows[i] + 1, y[i], v, a1[i], a2[i]][j]]))), cols), 'text/csv'); });
    msg.textContent = '';
  } catch (e) { msg.textContent = ''; out.innerHTML = errBox(e); console.error(e); }
}

// ---------------- interpretación ----------------
function renderInterp() {
  const R = state.res; if (!R) return;
  $('#int-out').innerHTML = `<div class="card"><h2>¿Qué variables importan y cómo?</h2><div class="howto"><div><b>Importancia por permutación</b>Se baraja una variable y se mide cuánto empeora la predicción (AUC o RMSE). Es agnóstica al algoritmo y agrupa las columnas de una variable categórica.</div><div><b>Dependencia parcial (PDP) e ICE</b>Muestra cómo cambia la predicción media (línea roja) y la de pacientes individuales (grises) al variar una variable, manteniendo las demás como están.</div><div><b>Precauciones</b>Importancia ≠ causalidad. Con variables muy correlacionadas la importancia se reparte. Se calcula con ${R.test ? 'el conjunto de prueba' : 'los datos de entrenamiento (valores relativos, algo optimistas)'}.</div></div><div class="row gap mt"><button class="primary" id="i-run">Calcular importancia y dependencia parcial</button><span id="i-msg" class="muted"></span></div><div id="i-res"></div></div>${R.spec.fam === 'binary' ? `<div class="card"><h2>Curva de decisión</h2><div class="row gap"><label class="inline">Umbral máximo <input id="d-max" type="number" step="0.05" value="0.8" style="width:80px"></label><button class="ghost" id="d-run">Calcular</button></div><div id="d-out"></div></div>` : ''}<div class="card" id="simp-card"><h2>Simplificar el modelo (variables que realmente aportan)</h2><div class="howto"><div><b>Qué se hace</b>Se mide cuánto cae el ${R.spec.fam === 'binary' ? 'AUC' : 'rendimiento'} al barajar cada variable, en pacientes que el modelo no vio, y se conservan solo las que superan el umbral.</div><div><b>Para qué</b>Un modelo con menos predictores es más fácil de recoger en la práctica, se sobreajusta menos y necesita menos pacientes.</div><div><b>Precauciones</b>Elegir variables con todos los datos y evaluar después con los mismos datos da un AUC optimista; la evaluación anidada repite la selección dentro de cada pliegue. Las variables muy correlacionadas se reparten la importancia: revise las que descarta.</div></div>${R.scheme.type === 'holdout' ? '<p class="warn">Con entrenamiento/prueba la simplificación usaría el conjunto de prueba. Use validación cruzada para esta función.</p>' : `<div class="form"><label>Umbral de caída ${R.spec.fam === 'binary' ? 'del AUC' : '(× DE del desenlace, RMSE)'} <input id="s-thr" type="number" step="0.005" min="0" value="0.01"></label><label>Repeticiones de la permutación <input id="s-R" type="number" min="1" max="20" value="5"></label><label>Pliegues de selección <input id="s-vs" type="number" min="2" max="10" value="3"></label><label class="inline wide"><input type="checkbox" id="s-nested" checked> Evaluación anidada (recomendada; tarda varias veces más)</label></div><p class="sub">Conservar siempre (variables clínicas obligatorias):</p><div class="predvars">${R.spec.vars.map(v => `<label class="pv"><input type="checkbox" data-force="${esc(v)}"> ${esc(v)}</label>`).join('')}</div><div class="row gap mt"><button class="primary" id="s-run">Calcular selección</button><progress id="s-prog" max="1" value="0" hidden></progress><span id="s-msg" class="muted"></span></div><div id="s-res"></div>`}</div><div class="card" id="calc-card"><h2>Calculadora de riesgo (variables del modelo)</h2><div class="howto"><div><b>Qué se hace</b>Introduzca los valores de las variables del modelo de un paciente nuevo y obtenga la predicción del Super Learner (y de cada algoritmo).</div><div><b>Para qué</b>Usar el modelo sin volver a entrenarlo; puede descargarlo como página HTML autónoma en «Reporte».</div><div><b>Precaución</b>Herramienta de investigación sin validación externa. Los campos vacíos se imputan con la mediana del desarrollo y reducen la precisión.</div></div><div class="form" id="cf-fields">${R.b.groups.map((g, i) => (g.kind === 'num' ? `<label>${esc(g.name)}<input type="number" step="any" data-cf="${i}" placeholder="${(() => { const v = R.b.X.map(x => x[g.cols[0]]).filter(Number.isFinite); return v.length ? `rango ${(+Math.min(...v).toPrecision(3))}–${(+Math.max(...v).toPrecision(3))}` : ''; })()}"></label>` : `<label>${esc(g.name)}<select data-cf="${i}"><option value="">(desconocido)</option>${g.levels.map(l => `<option>${esc(l)}</option>`).join('')}</select></label>`)).join('')}</div><div class="row gap mt"><button class="primary" id="cf-go">Estimar</button><span id="cf-msg" class="muted"></span></div><div id="cf-out"></div></div><div class="card"><h2>Predecir en pacientes nuevos</h2><p class="sub">Cargue un CSV con las mismas columnas de predictores (el desenlace no es necesario). Se descargan las predicciones del Super Learner y de cada algoritmo. Las predicciones solo valen mientras esta sesión del navegador siga abierta.</p><div class="drop slim"><input type="file" id="new-file" accept=".csv,.txt,.tsv,.xlsx,.xls"><label for="new-file"><b>Cargar base nueva</b></label></div><div id="new-msg"></div></div>`;
  $('#cf-go').addEventListener('click', calcNow); $('#i-run').addEventListener('click', runInterp); $('#s-run')?.addEventListener('click', runSimplify); $('#d-run')?.addEventListener('click', runDCA); $('#new-file').addEventListener('change', e => e.target.files[0] && predictNew(e.target.files[0]));
}
async function runInterp() {
  const R = state.res, msg = $('#i-msg'); msg.textContent = 'Calculando…';
  try {
    const groups = R.b.groups, imp = await call('importance', { groups: groups.map(g => ({ name: g.name, cols: g.cols })), R: 5, seed: 1 }), top = imp.slice(0, 6).map(i => groups.find(g => g.name === i.name));
    const pdp = await call('pdp', { groups: top.map(g => ({ name: g.name, cols: g.kind === 'cat' ? g.cols : g.cols, kind: g.kind, levels: g.levels })) });
    const bin = R.spec.fam === 'binary', unit = bin ? 'Drop in AUC' : 'Increase in RMSE';
    $('#i-res').innerHTML = `<div class="howto"><div><b>Importancia</b>Cuánto empeora la predicción al barajar cada variable (barras) con su variación entre repeticiones (línea). Variables con caída cercana a 0 aportan poco.</div><div><b>Dependencia parcial</b>La línea roja es la predicción media al variar la variable; las grises, pacientes individuales (ICE). Curvas grises muy distintas indican interacciones.</div><div><b>Precaución</b>Importancia y dependencia describen al modelo, no efectos causales; con variables correlacionadas la importancia se reparte.</div></div><div class="plots">${figWrap(hbarPlot(imp.slice(0, 20).map(i => ({ label: i.name, value: Math.max(0, i.drop), lo: Math.max(0, i.drop - i.sd), hi: i.drop + i.sd })), { title: 'Permutation importance', xlab: unit, xmin: 0, fmt: v => v.toFixed(3) }), 'importancia_permutacion')}${pdp.map(p => figWrap(pdpPlot(p, { ylab: bin ? 'Predicted probability' : 'Predicted value' }), `pdp_${p.name}`)).join('')}</div>`; bindFigs($('#i-res'));
    state.imp = imp; msg.textContent = '';
  } catch (e) { msg.innerHTML = errBox(e); }
}
async function runDCA() {
  const R = state.res, p = R.honest ? R.honest.ensPred : R.ensOOF, mx = +$('#d-max').value || 0.8, th = Array.from({ length: 40 }, (_, k) => 0.02 + (mx - 0.02) * k / 39), rows = dcaBinary(p, R.y, th);
  $('#d-out').innerHTML = `<div class="plots single">${figWrap(dcaPlot(rows, { title: 'Super Learner decision curve' }), 'dca_super_learner')}</div><p class="sub">Predicciones fuera de muestra (${R.honest ? 'validación cruzada anidada' : 'validación cruzada'}). Beneficio neto = VP/n − FP/n·pt/(1−pt).</p>`; bindFigs($('#d-out'));
}
async function runSimplify() {
  const R = state.res, bin = R.spec.fam === 'binary', btn = $('#s-run'), prog = $('#s-prog'), msg = $('#s-msg'); btn.disabled = true; prog.hidden = false; prog.value = 0;
  try {
    const nested = $('#s-nested').checked, force = $$('[data-force]:checked').map(i => i.dataset.force), thr = +$('#s-thr').value || 0.01, Vo = Math.min(5, Math.max(3, +$('#v-vo').value || 5));
    const out = await call('simplify', { X: R.b.X, y: R.b.y, fam: R.spec.fam, lib: R.lib, groups: R.b.groups.map(g => ({ name: g.name, cols: g.cols })), thr, R: +$('#s-R').value || 5, Vsel: +$('#s-vs').value || 3, Vcv: Math.min(5, R.scheme.V), nested, Vo, force, seed: R.seed, method: R.method }, p => { prog.value = Math.min(1, p.done / p.total); msg.textContent = `${p.stage}: ${p.done} de ${p.total}`; });
    state.lastSimp = { ...out, allVars: R.spec.vars.slice(), force, R: +$('#s-R').value || 5, Vsel: +$('#s-vs').value || 3, Vo };
    const f3 = v => f2(v, 3), mt = m => (bin ? [`${f2(m.auc)} (${f2(m.aucLo)}–${f2(m.aucHi)})`, f2(m.brier, 4), f2(m.oe, 2)] : [f2(m.rmse), f2(m.mae), f2(m.r2)]);
    const rows = out.rows.map(r => ({ c: [esc(r.name), `${f3(r.drop)} ± ${f3(r.sd)}`, out.nested ? pct(r.freq) : '–', r.keep ? `<b>Conservar</b>${r.forced ? ' (forzada)' : ''}${r.top1 ? ' (ninguna superó el umbral; se deja la mejor)' : ''}` : 'Descartar'], cls: r.keep ? 'best' : '' }));
    const hon = out.honest ? `<h3>Rendimiento honesto (la selección se repite dentro de cada pliegue externo)</h3>${table(['Modelo', ...(bin ? ['AUC (IC 95 %)', 'Brier', 'O/E'] : ['RMSE', 'MAE', 'R²']), 'Variables'], [{ c: ['Completo', ...mt(out.honest.full), String(R.spec.vars.length)] }, { c: ['Simplificado', ...mt(out.honest.reduced), `${out.kept.length} (por pliegue: ${out.honest.sizes.join(', ')})`], cls: 'sl' }])}` : '<p class="warn">Sin evaluación anidada: el rendimiento del modelo simplificado que verá tras reajustar será optimista.</p>';
    $('#s-res').innerHTML = `<h3>Importancia por permutación en pliegues de prueba</h3>${table(['Variable', bin ? 'Caída del AUC (media ± DE)' : 'Aumento del RMSE (media ± DE)', 'Conservada en pliegues externos', 'Decisión'], rows)}<p class="sub">Umbral: ${bin ? `caída del AUC > ${out.thr}` : `aumento del RMSE > ${out.thr} × DE del desenlace`}. Se conservan <b>${out.kept.length} de ${R.spec.vars.length}</b> variables: ${esc(out.kept.join(', '))}.</p>${hon}<div class="row gap mt"><button class="primary" id="s-apply">Reajustar todo con las ${out.kept.length} variables seleccionadas</button><span class="sub">Recalcula modelos, resultados, reporte, código Python y calculadora.</span></div>`;
    $('#s-apply').addEventListener('click', applySimplify); msg.textContent = '';
  } catch (e) { msg.innerHTML = errBox(e); console.error(e); }
  prog.hidden = true; btn.disabled = false;
}
function applySimplify() {
  const S = state.lastSimp, R = state.res; if (!S) return;
  state.pendingSimp = { kept: S.kept, allVars: S.allVars, thr: S.thr, nested: S.nested, honest: S.honest, R: S.R, Vsel: S.Vsel, force: S.force };
  state.spec = { ...R.spec, vars: S.kept }; renderQuestion(); showTab('algoritmos'); $('#run').click();
}
async function calcNow() {
  const R = state.res, bin = R.spec.fam === 'binary', msg = $('#cf-msg'); msg.textContent = 'Calculando…';
  try {
    state.exportM ||= await call('export', {}); const x = new Float64Array(R.b.names.length).fill(NaN);
    $$('[data-cf]').forEach(el => { const g = R.b.groups[+el.dataset.cf]; if (g.kind === 'num') { const v = el.value.trim(); x[g.cols[0]] = v === '' ? NaN : Number(v); } else if (el.value === '') g.cols.forEach(j => (x[j] = NaN)); else g.cols.forEach((j, q) => (x[j] = g.levels[q + 1] === el.value ? 1 : 0)); });
    const r = calcPredict(state.exportM, x), f = v => (bin ? (100 * v).toFixed(1) + ' %' : v.toFixed(2));
    $('#cf-out').innerHTML = `<div class="kpis"><div class="kpi"><b>${f(r.ens)}</b><span>${bin ? 'riesgo predicho (Super Learner)' : 'valor predicho (Super Learner)'}</span></div></div><details><summary class="sub">Predicción de cada algoritmo</summary>${table(['Algoritmo', 'Peso', 'Predicción'], state.exportM.learners.map((L, k) => ({ c: [esc(L.label), f2(L.w, 3), f(r.each[k])] })))}</details>`; msg.textContent = '';
  } catch (e) { msg.innerHTML = errBox(e); }
}
async function predictNew(file) {
  const msg = $('#new-msg'); try {
    const { sheets } = await readFile(file), raw = sheets[Object.keys(sheets)[0]], R = state.res, miss = R.spec.vars.filter(v => !(v in raw[0])); if (miss.length) throw new Error(`Faltan columnas: ${miss.join(', ')}`);
    const active = Object.fromEntries(Object.entries(state.meta).filter(([k, m]) => m.type !== 'skip' && k in raw[0])), ds = typedData(raw, active, naCodes()), spec = { ...R.spec, y: '__none__' }; ds.cols.__none__ = new Array(ds.n).fill('x'); const meta = { ...state.meta, __none__: { name: '__none__', type: 'cat', levels: ['x'], ref: 'x', event: 'x' } };
    const b = buildMatrix(ds, meta, { ...spec, fam: 'continuous' }); const out = await call('predict', { X: b.X }), cols = ['fila', 'Super Learner', ...R.labels];
    const rows = b.X.map((_, i) => Object.fromEntries(cols.map((c, k) => [c, k === 0 ? b.rows[i] + 1 : k === 1 ? out.ens[i] : out.each[k - 2][i]])));
    download('predicciones_nuevos.csv', toCSV(rows, cols), 'text/csv'); msg.innerHTML = `<p class="pill">✓ ${rows.length} predicciones descargadas.</p>`;
  } catch (e) { msg.innerHTML = errBox(e); }
}

// ---------------- reporte ----------------
function methodsText(lang) {
  const R = state.res, es = lang === 'es', bin = R.spec.fam === 'binary', sp = R.spec, nest = R.honest, sc = R.scheme, nev = bin ? R.y.reduce((a, c) => a + c, 0) : null;
  const L = (a, b) => (es ? a : b), libTxt = R.lib.map(libraryLabel).join('; ');
  const metr = bin ? L('el área bajo la curva ROC (AUC, con IC 95 % de Hanley y McNeil), la puntuación de Brier (y su versión escalada), el log-loss, la pendiente e intercepto de calibración y el cociente observados/esperados (O/E)', 'the area under the ROC curve (AUC, 95% CI by Hanley and McNeil), the Brier score (and scaled Brier), log-loss, calibration slope and intercept, and the observed/expected ratio (O/E)') : L('la raíz del error cuadrático medio (RMSE), el error absoluto medio (MAE) y el R²', 'root mean squared error (RMSE), mean absolute error (MAE) and R²');
  const val = sc.type === 'cv' ? L(`validación cruzada estratificada de ${sc.V} pliegues${sc.repeats > 1 ? `, repetida ${sc.repeats} veces con distintas particiones (se promediaron las métricas)` : ''}`, `${sc.V}-fold stratified cross-validation${sc.repeats > 1 ? `, repeated ${sc.repeats} times with different partitions (metrics were averaged)` : ''}`) : L(`una partición entrenamiento/prueba (${pct(sc.testFrac)} de prueba, utilizada una sola vez) con validación cruzada de ${sc.V} pliegues en el entrenamiento`, `a train/test split (${pct(sc.testFrac)} test set, used once) with ${sc.V}-fold cross-validation in the training set`);
  const best = R.labels.map((l, k) => [l, R.perLearner[k]]).sort((a, b) => (bin ? b[1].auc - a[1].auc : a[1].rmse - b[1].rmse))[0];
  const res = bin ? `${L('El mejor algoritmo individual fue', 'The best single algorithm was')} ${best[0]} (AUC ${f2(best[1].auc)}); ${nest ? `${L('el Super Learner con validación cruzada anidada alcanzó', 'the nested cross-validated Super Learner achieved')} AUC ${f2(nest.ens.auc)} (${f2(nest.ens.aucLo)}–${f2(nest.ens.aucHi)}), Brier ${f2(nest.ens.brier, 4)}` : ''}.` : `${L('El mejor algoritmo individual fue', 'The best single algorithm was')} ${best[0]} (RMSE ${f2(best[1].rmse)}, R² ${f2(best[1].r2)}); ${nest ? `${L('el Super Learner anidado', 'the nested Super Learner')}: RMSE ${f2(nest.ens.rmse)}, R² ${f2(nest.ens.r2)}` : ''}.`;
  const ss = R.advice?.riley && bin ? L(` Para una regresión con ${R.advice.p} parámetros candidatos, los criterios de Riley et al. con un AUC esperado de ${f2(+$('#q-cexp').value || 0.75, 2)} requerían ≈ ${R.advice.riley.n} pacientes (${R.advice.riley.events} eventos); la muestra tuvo ${R.n} (${nev}).`, ` For a regression model with ${R.advice.p} candidate parameters, the criteria of Riley et al. with an expected AUC of ${f2(+$('#q-cexp').value || 0.75, 2)} required ≈ ${R.advice.riley.n} participants (${R.advice.riley.events} events); the sample had ${R.n} (${nev}).`) : '';
  const sm = R.simp ? L(`\n\nSelección de variables: se partió de ${R.simp.allVars.length} predictores. La importancia de cada variable se calculó por permutación (${R.simp.R} repeticiones; caída del ${bin ? 'AUC' : 'rendimiento'} al barajar la variable, agrupando los indicadores de las categóricas) en pliegues de prueba (${R.simp.Vsel} pliegues, Super Learner ajustado sin ellos) y se promedió; se conservaron las variables con una caída > ${R.simp.thr}${R.simp.force.length ? ` y se forzó la inclusión de ${R.simp.force.join(', ')}` : ''}, resultando ${R.simp.kept.length} predictores (${R.simp.kept.join(', ')}). ${R.simp.honest ? `El rendimiento se evaluó repitiendo todo el proceso de selección dentro de cada pliegue externo (${R.simp.honest.Vo} pliegues): ${bin ? `AUC ${f2(R.simp.honest.full.auc)} con todas las variables y ${f2(R.simp.honest.reduced.auc)} con el modelo simplificado` : `RMSE ${f2(R.simp.honest.full.rmse)} y ${f2(R.simp.honest.reduced.rmse)}`}.` : 'No se realizó evaluación anidada de la selección, por lo que el rendimiento del modelo simplificado es optimista.'} El modelo se reajustó con las variables seleccionadas.`, `\n\nVariable selection: we started from ${R.simp.allVars.length} predictors. Variable importance was computed by permutation (${R.simp.R} repetitions; drop in ${bin ? 'AUC' : 'performance'} after shuffling the variable, grouping the indicators of categorical variables) in held-out folds (${R.simp.Vsel} folds, Super Learner fitted without them) and averaged; variables with a drop > ${R.simp.thr} were kept${R.simp.force.length ? ` and ${R.simp.force.join(', ')} were forced in` : ''}, leaving ${R.simp.kept.length} predictors (${R.simp.kept.join(', ')}). ${R.simp.honest ? `Performance was assessed by repeating the whole selection process within each outer fold (${R.simp.honest.Vo} folds): ${bin ? `AUC ${f2(R.simp.honest.full.auc)} with all variables and ${f2(R.simp.honest.reduced.auc)} with the simplified model` : `RMSE ${f2(R.simp.honest.full.rmse)} and ${f2(R.simp.honest.reduced.rmse)}`}.` : 'Selection was not evaluated with nested resampling, so the performance of the simplified model is optimistic.'} The model was refitted with the selected variables.`) : '';
  return L(
    `Se desarrolló un modelo de predicción de ${sp.y}${bin ? ` (evento: ${sp.event})` : ''} con ${R.b.names.length} columnas derivadas de ${sp.vars.length} predictores en ${R.n} pacientes${bin ? ` (${nev} eventos, ${pct(nev / R.n)})` : ''}. Las variables categóricas se codificaron en indicadores (categoría de referencia omitida); los datos perdidos de los predictores se imputaron con la mediana del conjunto de entrenamiento dentro de cada pliegue, y las variables se estandarizaron dentro del algoritmo cuando era necesario, para evitar fugas de información. Las características basales se describieron en el total y según el desenlace con mediana (rango intercuartílico) o n (%) y se compararon con la U de Mann-Whitney, la χ² de Pearson o la exacta de Fisher; la asociación de cada predictor se estimó con regresión cruda y ajustada.${ss}\n\nSe compararon los algoritmos: ${libTxt}. La regresión penalizada se ajustó por descenso de coordenadas con λ elegido por validación cruzada interna; los árboles, bosques y boosting son árboles de regresión CART (en clasificación se promedia el desenlace 0/1 para obtener probabilidades); la red neuronal tiene una capa oculta con activación tanh, regularización L2 y optimizador Adam. El Super Learner combinó las predicciones fuera de muestra de los algoritmos con pesos no negativos que suman 1 (${R.method === 'loglik' ? 'máxima verosimilitud binomial' : 'mínimos cuadrados no negativos'}) elegidos por validación cruzada de ${sc.V} pliegues. El rendimiento se evaluó con ${val}${nest ? `, y el del Super Learner con validación cruzada anidada (${R.cvsl.Vouter} pliegues externos, ${R.cvsl.Vinner} internos), que evita el optimismo de usar las mismas predicciones para elegir los pesos y evaluarlos` : ''}, usando ${metr}. ${bin ? 'La utilidad clínica se resumió con curvas de decisión (beneficio neto). ' : ''}La importancia de las variables se estimó por permutación (caída del ${bin ? 'AUC' : 'rendimiento'}) y su efecto con gráficos de dependencia parcial e ICE.${sm}\n\n${res}\n\nLos métodos corresponden a las siguientes implementaciones de Python: regresión logística/lineal (sklearn.linear_model.LogisticRegression, LinearRegression), regresión penalizada (LogisticRegressionCV, ElasticNetCV), árbol de decisión (sklearn.tree.DecisionTreeRegressor), bosque aleatorio (RandomForestRegressor), boosting de gradiente (GradientBoostingClassifier, GradientBoostingRegressor), k vecinos (KNeighborsClassifier, KNeighborsRegressor), red neuronal (MLPClassifier, MLPRegressor), pesos del Super Learner (scipy.optimize.nnls), métricas (sklearn.metrics.roc_auc_score, brier_score_loss, log_loss) y comparaciones de grupos (scipy.stats.mannwhitneyu, chi2_contingency, fisher_exact; statsmodels para la regresión cruda y ajustada). Se proporciona un script equivalente como material suplementario.`,
    `A prediction model for ${sp.y}${bin ? ` (event: ${sp.event})` : ''} was developed using ${R.b.names.length} columns derived from ${sp.vars.length} predictors in ${R.n} patients${bin ? ` (${nev} events, ${pct(nev / R.n)})` : ''}. Categorical variables were one-hot encoded (reference category omitted); missing predictor values were imputed with the training-set median within each fold, and variables were standardised within the algorithm when required, to avoid information leakage. Baseline characteristics were described overall and by outcome as median (interquartile range) or n (%) and compared with the Mann–Whitney U test, Pearson's χ² or Fisher's exact test; the association of each predictor was estimated with crude and adjusted regression.${ss}\n\nThe following algorithms were compared: ${libTxt}. Penalised regression was fitted by coordinate descent with λ chosen by internal cross-validation; trees, random forests and boosting use CART regression trees (for classification the 0/1 outcome is averaged to obtain probabilities); the neural network has one hidden layer with tanh activation, L2 regularisation and the Adam optimiser. The Super Learner combined the out-of-sample predictions of the algorithms with non-negative weights summing to 1 (${R.method === 'loglik' ? 'binomial likelihood' : 'non-negative least squares'}) chosen by ${sc.V}-fold cross-validation. Performance was assessed with ${val}${nest ? `, and that of the Super Learner with nested cross-validation (${R.cvsl.Vouter} outer, ${R.cvsl.Vinner} inner folds), which avoids the optimism of choosing and evaluating the weights on the same predictions` : ''}, using ${metr}. ${bin ? 'Clinical utility was summarised with decision curves (net benefit). ' : ''}Variable importance was estimated by permutation (drop in ${bin ? 'AUC' : 'performance'}) and variable effects with partial dependence and ICE plots.${sm}\n\n${res}\n\nThe methods correspond to the following Python implementations: logistic/linear regression (sklearn.linear_model.LogisticRegression, LinearRegression), penalised regression (LogisticRegressionCV, ElasticNetCV), decision tree (sklearn.tree.DecisionTreeRegressor), random forest (RandomForestRegressor), gradient boosting (GradientBoostingClassifier, GradientBoostingRegressor), k-nearest neighbours (KNeighborsClassifier, KNeighborsRegressor), neural network (MLPClassifier, MLPRegressor), Super Learner weights (scipy.optimize.nnls), metrics (sklearn.metrics.roc_auc_score, brier_score_loss, log_loss) and group comparisons (scipy.stats.mannwhitneyu, chi2_contingency, fisher_exact; statsmodels for crude and adjusted regression). An equivalent script is provided as supplementary material.`);
}
// configuración del script Python (sin datos de pacientes)
function pyConfig() {
  const R = state.res, sp = R.spec, bin = sp.fam === 'binary', sc = R.scheme, S = R.simp, vars = S ? S.allVars : sp.vars;
  return { data_file: `${state.fname}.csv`, na_values: naCodes(), outcome: sp.y, family: sp.fam, event: sp.event, predictors: vars.map(v => { const m = state.meta[v]; return m.type === 'num' ? { name: v, type: 'num' } : { name: v, type: 'cat', levels: levelsOf(m) }; }),
    library: R.lib.map(c => ({ learner: c.learner, label: libraryLabelEn(c), params: c.params })), scheme: { type: sc.type, V: sc.V, repeats: sc.repeats, test_frac: sc.testFrac }, method: R.method, seed: R.seed,
    nested: { on: !!R.cvslOn, Vo: R.cvsl.Vouter, Vi: R.cvsl.Vinner }, selection: { run: false, thr: S?.thr ?? 0.01, R: S?.R ?? 5, Vsel: S?.Vsel ?? 3, Vcv: Math.min(5, sc.V), force: S?.force ?? [], selected: S ? S.kept : null },
    meta: { n: R.n, events: bin ? R.y.reduce((a, c) => a + c, 0) : null, date: new Date().toISOString().slice(0, 10) } };
}
async function exportCalc() {
  const R = state.res, m = $('#r-msg'); m.textContent = 'Preparando…';
  try {
    const M = await call('export', {}), bin = R.spec.fam === 'binary', rg = v => +Number(v).toPrecision(3);
    const groups = R.b.groups.map(g => { if (g.kind !== 'num') return { name: g.name, kind: 'cat', cols: g.cols, levels: g.levels }; const v = R.b.X.map(x => x[g.cols[0]]).filter(Number.isFinite); return { name: g.name, kind: 'num', cols: g.cols, min: rg(Math.min(...v)), max: rg(Math.max(...v)) }; });
    const html = calcHTML(M, { title: `Calculadora · ${R.spec.y}`, outcome: R.spec.y, event: R.spec.event, fam: R.spec.fam, groups, ncols: R.b.names.length, n: R.n, events: bin ? R.y.reduce((a, c) => a + c, 0) : null, auc: bin ? f2((R.honest ? R.honest.ens : R.ensCV).auc) : '', date: new Date().toISOString().slice(0, 10), notes: R.simp ? `Modelo simplificado: ${R.simp.kept.length} de ${R.simp.allVars.length} predictores.` : '' });
    download(`calculadora_${R.spec.y.replace(/\W+/g, '_')}.html`, html, 'text/html'); m.innerHTML = `<p class="pill">✓ Calculadora descargada (${(html.length / 1e6).toFixed(1)} MB). Funciona sin conexión.</p>`;
  } catch (e) { m.innerHTML = errBox(e); }
}
function renderReport() {
  const R = state.res; if (!R) return; const bin = R.spec.fam === 'binary';
  const mat = { head: bin ? ['Algoritmo', 'AUC', 'IC inf.', 'IC sup.', 'Brier', 'Log-loss', 'Peso'] : ['Algoritmo', 'RMSE', 'MAE', 'R²', 'Peso'], body: R.labels.map((l, k) => ({ c: bin ? [l, f2(R.perLearner[k].auc), f2(R.perLearner[k].aucLo), f2(R.perLearner[k].aucHi), f2(R.perLearner[k].brier, 4), f2(R.perLearner[k].logloss), f2(R.weights[k])] : [l, f2(R.perLearner[k].rmse), f2(R.perLearner[k].mae), f2(R.perLearner[k].r2), f2(R.weights[k])] })) };
  $('#rep-out').innerHTML = `<div class="card"><h2>Texto de métodos y resultados</h2><div class="row gap"><button class="ghost" data-lang="es">Español</button><button class="ghost" data-lang="en">English</button><span class="grow"></span><button class="ghost" id="r-copy">Copiar</button></div><textarea id="methods" rows="14"></textarea></div><div class="card"><h2>Tabla y datos</h2><div class="row gap wrap"><button class="ghost" id="r-doc">Tabla Word (.doc)</button><button class="ghost" id="r-csv">Tabla CSV</button><button class="ghost" id="r-pred">Predicciones fuera de muestra (CSV)</button><button class="ghost" id="r-py">Código Python (.py, sin figuras)</button><button class="ghost" id="r-calc">Calculadora de riesgo (HTML)</button></div><div id="r-msg"></div><p class="sub">El código Python reproduce el análisis con scikit-learn y SciPy (ver advertencias en la cabecera del archivo). La calculadora es una página autónoma con el modelo final: no contiene datos de pacientes, pero sí el modelo ajustado con ellos; es una herramienta de investigación, no validada externamente.</p>${table(mat.head, mat.body)}</div>`;
  let lang = 'es'; const set = () => ($('#methods').value = methodsText(lang)); set(); $$('[data-lang]').forEach(b => b.addEventListener('click', () => { lang = b.dataset.lang; set(); }));
  $('#rep-out').insertAdjacentHTML('beforeend', `<div class="card"><h2>Borrador de manuscrito (Español / English)</h2><div class="howto"><div><b>Qué se ve</b>Un borrador con título, resumen, objetivo, métodos y resultados (Tabla 1, regresión cruda y ajustada, comparación de algoritmos y Super Learner) con figuras en inglés, redactado con los números de su análisis.</div><div><b>Para qué</b>Partir de una estructura de revista (TRIPOD+AI) con la estadística descrita con las funciones equivalentes de Python.</div><div><b>Cómo usarlo</b>Revise y complete introducción, discusión y referencias. Si los datos son reales, desmarque «simulados». Para Word: abra el .doc descargado o copie desde el navegador.</div></div><div class="form"><label>Título (opcional) <input id="ms-title" placeholder="se genera automáticamente"></label><label>Población (para el título) <input id="ms-pop" placeholder="p. ej., adults with CKD stage 3-4"></label><label>Nombre del desenlace <input id="ms-out" placeholder="${esc(R.spec.y)}"></label><label class="inline"><input type="checkbox" id="ms-sim"${state.isExample ? ' checked' : ''}> Los datos son simulados (ejemplo)</label></div><div class="row gap wrap"><button class="primary" id="ms-es">Generar en español</button><button class="primary" id="ms-en">Generate in English</button><button class="ghost" id="ms-dl" disabled>Descargar HTML</button><button class="ghost" id="ms-doc" disabled>Descargar .doc</button><span id="ms-msg" class="muted"></span></div><iframe id="ms-view" style="width:100%;height:520px;border:1px solid var(--line);border-radius:8px;margin-top:10px" hidden></iframe></div>`);
  let ms = null; const gen = async lang => { const msg = $('#ms-msg'); msg.textContent = 'Generando…'; try { const vars = R.simp ? R.simp.allVars : R.spec.vars, sp0 = { ...R.spec, vars }; state.isExample = $('#ms-sim').checked;
    ms = await buildManuscript({ lang, title: $('#ms-title').value.trim(), pop: $('#ms-pop').value.trim(), outcome: $('#ms-out').value.trim(), state: { res: R, imp: state.imp, isExample: state.isExample, nTotal: state.raw.length }, t1: t1For(sp0, vars, lang, true), ua: uaFor(R.spec, lang) }); const fr = $('#ms-view'); fr.hidden = false; fr.srcdoc = ms.html; $('#ms-dl').disabled = $('#ms-doc').disabled = false; msg.textContent = ''; } catch (e) { msg.textContent = e.message; console.error(e); } };
  $('#ms-es').addEventListener('click', () => gen('es')); $('#ms-en').addEventListener('click', () => gen('en')); $('#ms-dl').addEventListener('click', () => ms && download('manuscrito.html', ms.html, 'text/html')); $('#ms-doc').addEventListener('click', () => ms && download('manuscrito.doc', ms.html, 'application/msword'));
  $('#r-py').addEventListener('click', () => download('replicar_analisis.py', pythonScript(pyConfig()), 'text/x-python')); $('#r-calc').addEventListener('click', exportCalc);
  $('#r-copy').addEventListener('click', () => navigator.clipboard?.writeText($('#methods').value));
  $('#r-doc').addEventListener('click', () => download('comparacion_algoritmos.doc', wordDoc('Comparación de algoritmos', matrixHTML(mat)), 'application/msword')); $('#r-csv').addEventListener('click', () => download('comparacion_algoritmos.csv', matrixCSV(mat), 'text/csv'));
  $('#r-pred').addEventListener('click', () => { const cols = ['fila', 'observado', ...R.labels, 'Super Learner']; const rows = Array.from(R.y, (y, i) => Object.fromEntries(cols.map((c, k) => [c, k === 0 ? R.b.rows[i] + 1 : k === 1 ? y : k <= R.labels.length + 1 ? R.Z[k - 2][i] : (R.honest ? R.honest.ensPred : R.ensOOF)[i]]))); download('predicciones_fuera_de_muestra.csv', toCSV(rows, cols), 'text/csv'); });
}
$('#fig-width').addEventListener('change', () => ($('#fig-mm').disabled = $('#fig-width').value !== 'custom')); $('#fig-mm').disabled = true;
gate(); window.__epi = { state, buildMatrix };
