// Calculadora de riesgo descargable: serializa el Super Learner final y genera una página HTML autónoma (sin conexión, sin datos de pacientes).
import { fitImputer } from './learners.js?v=2026.10.25-6';

const rnd = (a, d = 6) => Array.from(a, v => +Number(v).toPrecision(d));
const tree = t => ({ f: Array.from(t.feat), t: rnd(t.thr, 6), l: Array.from(t.left), r: Array.from(t.right), v: rnd(t.val, 5) });
// sl: resultado de superLearner (con fits y weights); Xtr: matriz de entrenamiento (para las medianas de imputación)
export function serializeModel(sl, labels, Xtr) {
  const medians = Array.from(fitImputer(Xtr).med);
  const learners = [];
  sl.fits.forEach((f, k) => {
    const w = sl.weights[k]; if (!(w > 0)) return; const id = sl.lib[k].learner, m = f.model;
    let mm;
    if (id === 'mean') mm = { m: m.m };
    else if (id === 'glm') mm = { keep: Array.from(m.keep), beta: rnd(m.beta, 8), fam: m.fam };
    else if (id === 'enet') mm = { intercept: m.intercept, beta: rnd(m.beta, 8), fam: m.fam };
    else if (id === 'tree') mm = { t: tree(m.t), fam: m.fam };
    else if (id === 'rf') mm = { trees: m.trees.map(tree), fam: m.fam };
    else if (id === 'gbm') mm = { trees: m.trees.map(tree), F0: m.F0, lr: m.lr, fam: m.fam };
    else if (id === 'nnet') mm = { mean: rnd(m.st.mean, 8), sd: rnd(m.st.sd, 8), W1: m.W1.map(a => rnd(a, 8)), b1: rnd(m.b1, 8), W2: rnd(m.W2, 8), b2: m.b2, ym: m.ym, ys: m.ys, bin: m.bin };
    else throw new Error(`«${labels[k]}» (${id}) guarda datos de pacientes y no se puede exportar. Quítelo de la biblioteca (o use el código Python) para generar la calculadora.`);
    learners.push({ id, label: labels[k], w, m: mm });
  });
  return { learners, med: medians };
}

// Predicción autónoma (se copia tal cual dentro del HTML); x: Float64Array con NaN = desconocido
export function calcPredict(M, x) {
  const expit = e => 1 / (1 + Math.exp(-e)), cl = a => Math.min(1 - 1e-6, Math.max(1e-6, a));
  const leaf = (t, z) => { let k = 0; while (t.f[k] >= 0) k = z[t.f[k]] <= t.t[k] ? t.l[k] : t.r[k]; return t.v[k]; };
  const z = Array.from(x, (v, j) => (Number.isNaN(v) ? M.med[j] : v)); let ens = 0; const each = [];
  for (const L of M.learners) {
    const m = L.m, id = L.id; let p;
    if (id === 'mean') p = m.m;
    else if (id === 'glm') { let e = m.beta[0]; m.keep.forEach((j, k) => (e += m.beta[k + 1] * z[j])); p = m.fam === 'binary' ? cl(expit(e)) : e; }
    else if (id === 'enet') { let e = m.intercept; for (let j = 0; j < z.length; j++) e += m.beta[j] * z[j]; p = m.fam === 'binary' ? cl(expit(e)) : e; }
    else if (id === 'tree') { p = leaf(m.t, z); if (m.fam === 'binary') p = cl(p); }
    else if (id === 'rf') { p = 0; for (const t of m.trees) p += leaf(t, z); p /= m.trees.length; if (m.fam === 'binary') p = cl(p); }
    else if (id === 'gbm') { let f = m.F0; for (const t of m.trees) f += m.lr * leaf(t, z); p = m.fam === 'binary' ? cl(expit(f)) : f; }
    else { let o = m.b2; for (let k = 0; k < m.W2.length; k++) { let s = m.b1[k]; for (let j = 0; j < z.length; j++) s += m.W1[k][j] * (z[j] - m.mean[j]) / m.sd[j]; o += m.W2[k] * Math.tanh(s); } p = m.bin ? cl(expit(o)) : o * m.ys + m.ym; }
    each.push(p); ens += L.w * p;
  }
  return { ens, each };
}

const E = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
// info: {title, outcome, event, fam, groups:[{name, kind, cols, levels?, min?, max?}], ncols, n, events, auc, date, notes}
export function calcHTML(M, info) {
  const bin = info.fam === 'binary', payload = JSON.stringify({ M, groups: info.groups, ncols: info.ncols, fam: info.fam, outcome: info.outcome, event: info.event }).replace(/</g, '\\u003c');
  const fields = info.groups.map((g, i) => g.kind === 'num'
    ? `<label>${E(g.name)}<input type="number" step="any" data-i="${i}" placeholder="${g.min != null ? `rango observado ${E(g.min)}–${E(g.max)}` : ''}"></label>`
    : `<label>${E(g.name)}<select data-i="${i}"><option value="">(desconocido)</option>${g.levels.map(l => `<option>${E(l)}</option>`).join('')}</select></label>`).join('');
  return `<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${E(info.title)}</title>
<style>:root{--bg:#fff;--fg:#1f2933;--mu:#6b7785;--line:#d9dee4;--ac:#2E7F90;--card:#f6f8fa}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:760px;margin:0 auto;padding:20px 16px 40px}h1{font-size:1.3rem;margin:.2em 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
label{display:flex;flex-direction:column;font-size:.85rem;color:var(--mu);gap:3px}input,select{font:inherit;padding:7px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{font:inherit;padding:8px 16px;border:0;border-radius:6px;background:var(--ac);color:#fff;cursor:pointer}button.g{background:transparent;color:var(--ac);border:1px solid var(--ac)}
.big{font-size:2.2rem;font-weight:700;color:var(--ac)}.mu{color:var(--mu);font-size:.85rem}.warn{border-left:4px solid #B3261E;background:rgba(179,38,30,.08);padding:8px 12px;border-radius:6px;font-size:.85rem}table{border-collapse:collapse;width:100%;font-size:.85rem}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}</style></head><body><main>
<h1>${E(info.title)}</h1><p class="mu">Calculadora de ${bin ? 'riesgo' : 'predicción'} de «${E(info.outcome)}»${bin ? ` (evento: ${E(info.event)})` : ''}. Modelo desarrollado con EPI-Maching-Learning (EPI-LAB) el ${E(info.date)} en ${info.n} pacientes${bin ? ` (${info.events} eventos)` : ''}. Funciona sin conexión: los cálculos se hacen en este navegador.</p>
<div class="warn"><b>Herramienta de investigación.</b> El modelo no ha sido validado externamente y no debe usarse para decidir la atención de un paciente. Los valores desconocidos se imputan con la mediana del conjunto de desarrollo, lo que reduce la precisión.${info.auc ? ` Rendimiento interno (validación cruzada): AUC ${E(info.auc)}.` : ''}</div>
<div class="card"><div class="grid" id="f">${fields}</div><p><button id="go">Calcular</button> <button class="g" id="rs">Borrar</button></p></div>
<div class="card" id="out" hidden><div class="mu">${bin ? 'Riesgo predicho' : 'Valor predicho'}</div><div class="big" id="big"></div><p class="mu" id="sub"></p><details><summary class="mu">Predicción de cada algoritmo</summary><table id="tb"></table></details></div>
<p class="mu">${E(info.notes || '')}</p>
<script type="application/json" id="model">${payload}</script>
<script>
const calcPredict = ${calcPredict.toString()};
const D = JSON.parse(document.getElementById('model').textContent), $ = s => document.querySelector(s);
const val = () => { const x = new Float64Array(D.ncols).fill(NaN); document.querySelectorAll('[data-i]').forEach(el => { const g = D.groups[+el.dataset.i]; if (g.kind === 'num') { const v = el.value.trim(); x[g.cols[0]] = v === '' ? NaN : Number(v); } else if (el.value === '') g.cols.forEach(j => (x[j] = NaN)); else g.cols.forEach((j, q) => (x[j] = g.levels[q + 1] === el.value ? 1 : 0)); }); return x; };
$('#go').onclick = () => { const r = calcPredict(D.M, val()), bin = D.fam === 'binary', f = v => (bin ? (100 * v).toFixed(1) + ' %' : v.toFixed(2));
  $('#out').hidden = false; $('#big').textContent = f(r.ens); $('#sub').textContent = bin ? 'Probabilidad estimada del evento por el Super Learner (combinación ponderada de los algoritmos).' : 'Valor estimado por el Super Learner.';
  $('#tb').innerHTML = '<tr><th>Algoritmo</th><th>Peso</th><th>Predicción</th></tr>' + D.M.learners.map((L, k) => '<tr><td>' + L.label.replace(/</g, '&lt;') + '</td><td>' + L.w.toFixed(3) + '</td><td>' + f(r.each[k]) + '</td></tr>').join(''); };
$('#rs').onclick = () => { document.querySelectorAll('[data-i]').forEach(el => (el.value = '')); $('#out').hidden = true; };
</script></main></body></html>`;
}
