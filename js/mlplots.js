// Gráficos del módulo de aprendizaje automático (SVG): comparación de algoritmos, ROC superpuestas, pesos, importancia, dependencia parcial y laboratorio de sobreajuste.
import { frame, path, esc, PAL, BLUE, RED, GREY, STYLE } from './pplots.js?v=2026.10.25-5';
export { calibrationPlot, scatterCalib, rocPlot, rocPoints, dcaPlot } from './pplots.js?v=2026.10.25-5';

const svgOpen = (W, H) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" class="plot" role="img">${STYLE}`;
// barras horizontales con etiquetas; rows: [{label, value, lo?, hi?, color?, note?}]
export function hbarPlot(rows, { title = '', xlab = '', xmin = null, xmax = null, fmt = v => v.toFixed(3), ref = null } = {}) {
  const lw = Math.min(260, 6.6 * Math.max(...rows.map(r => r.label.length), 8)), rowH = 24, W = lw + 360, H = 46 + rows.length * rowH + 34, x0 = lw + 14, x1 = W - 80;
  const vals = rows.flatMap(r => [r.value, r.lo ?? r.value, r.hi ?? r.value]), lo = xmin ?? Math.min(0, ...vals), hi = xmax ?? Math.max(...vals) * 1.08, sx = v => x0 + (v - lo) / ((hi - lo) || 1) * (x1 - x0), base = lo > 0 ? lo : 0;
  let s = svgOpen(W, H) + (title ? `<text x="8" y="18" class="h">${esc(title)}</text>` : '');
  rows.forEach((r, i) => {
    const y = 34 + i * rowH, c = r.color || PAL[0];
    s += `<text x="${lw + 6}" y="${y + 13}" text-anchor="end" class="s">${esc(r.label)}</text><rect x="${sx(base)}" y="${y + 2}" width="${Math.max(0, sx(r.value) - sx(base))}" height="${rowH - 8}" fill="${c}" opacity=".85"/>`;
    if (r.lo != null) s += `<line x1="${sx(r.lo)}" x2="${sx(r.hi)}" y1="${y + rowH / 2 - 1}" y2="${y + rowH / 2 - 1}" stroke="currentColor"/>`;
    s += `<text x="${Math.max(sx(r.value), sx(r.hi ?? r.value)) + 5}" y="${y + 13}" class="s">${fmt(r.value)}${r.note ? ' ' + esc(r.note) : ''}</text>`;
  });
  const yb = 34 + rows.length * rowH + 4; s += `<line x1="${x0}" x2="${x1}" y1="${yb}" y2="${yb}" stroke="currentColor"/>`;
  if (ref != null) s += `<line x1="${sx(ref)}" x2="${sx(ref)}" y1="30" y2="${yb}" stroke="${GREY}" stroke-dasharray="4 3"/>`;
  s += `<text x="${(x0 + x1) / 2}" y="${H - 8}" text-anchor="middle">${esc(xlab)}</text>`; return s + '</svg>';
}
export function rocMulti(curves, title = 'ROC curves (cross-validated predictions)') {
  const F = frame({ W: 460, H: 420, xdom: [0, 1], ydom: [0, 1], xlab: '1 − Specificity', ylab: 'Sensitivity', title }); let s = F.s;
  s += `<line x1="${F.sx(0)}" y1="${F.sy(0)}" x2="${F.sx(1)}" y2="${F.sy(1)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  curves.forEach((c, k) => { const col = c.color || PAL[k % PAL.length]; s += `<path d="${path(c.pts, F.sx, F.sy)}" fill="none" stroke="${col}" stroke-width="${c.bold ? 3 : 1.5}"/><rect x="${F.sx(0.45)}" y="${F.sy(0.28) + k * 15}" width="9" height="9" fill="${col}"/><text x="${F.sx(0.45) + 14}" y="${F.sy(0.28) + 9 + k * 15}" class="s">${esc(c.label)}</text>`; });
  return s + '</svg>';
}
// dependencia parcial con curvas ICE; pd = {name, kind, x[], pd[], ice[][], levels?}
export function pdpPlot(pd, { ylab = 'Prediction', title = null } = {}) {
  const isNum = pd.kind === 'num', ys = [...pd.pd, ...pd.ice.flat()], lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.05 || 0.01;
  if (isNum) {
    const F = frame({ W: 380, H: 300, xdom: [Math.min(...pd.x), Math.max(...pd.x)], ydom: [lo - pad, hi + pad], xlab: pd.name, ylab, title: title ?? `Partial dependence · ${pd.name}`, ml: 56 }); let s = F.s;
    pd.ice.forEach(l => (s += `<path d="${path(pd.x.map((x, i) => [x, l[i]]), F.sx, F.sy)}" fill="none" stroke="${GREY}" stroke-width=".7" opacity=".45"/>`));
    s += `<path d="${path(pd.x.map((x, i) => [x, pd.pd[i]]), F.sx, F.sy)}" fill="none" stroke="${RED}" stroke-width="2.6"/>`; return s + '</svg>';
  }
  const L = pd.levels, W = 380, H = 300, m = { l: 56, r: 12, t: 30, b: 70 }, bw = (W - m.l - m.r) / L.length, sy = v => H - m.b - (v - (lo - pad)) / (hi + pad - (lo - pad)) * (H - m.b - m.t);
  let s = svgOpen(W, H) + `<text x="8" y="18" class="h">${esc(title ?? `Partial dependence · ${pd.name}`)}</text><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="none" stroke="currentColor" stroke-width=".7"/>`;
  pd.pd.forEach((v, i) => { const x = m.l + i * bw + bw * 0.15; s += `<rect x="${x}" y="${sy(v)}" width="${bw * 0.7}" height="${H - m.b - sy(v)}" fill="${BLUE}" opacity=".85"/><text x="${x + bw * 0.35}" y="${H - m.b + 12}" class="s" transform="rotate(35 ${x + bw * 0.35} ${H - m.b + 12})">${esc(String(L[i]).slice(0, 16))}</text>`; });
  [lo, (lo + hi) / 2, hi].forEach(v => (s += `<text x="${m.l - 5}" y="${sy(v) + 4}" text-anchor="end" class="s">${(+v.toPrecision(3))}</text>`)); return s + '</svg>';
}
// laboratorio de sobreajuste: izquierda datos y ajuste; derecha error de entrenamiento y de prueba según el grado
export function labPlot({ x, y, curve, truth, deg, errs }) {
  const W = 760, H = 330, F1 = frame({ W: 380, H, xdom: [-1, 1], ydom: [-2.2, 2.2], xlab: 'x', ylab: 'y', title: `Polinomio de grado ${deg}` }); let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" class="plot" role="img">${STYLE}` + F1.s.replace(/^<svg[^>]*>/, '').replace(/<style>.*?<\/style>/, '').replace(/<\/svg>$/, '');
  s += `<path d="${path(truth, F1.sx, F1.sy)}" fill="none" stroke="${GREY}" stroke-width="2" stroke-dasharray="5 3"/>`;
  x.forEach((v, i) => (s += `<circle cx="${F1.sx(v)}" cy="${F1.sy(y[i])}" r="3.2" fill="${BLUE}" fill-opacity=".8"/>`));
  s += `<path d="${path(curve.filter(p => Math.abs(p[1]) < 5), F1.sx, F1.sy)}" fill="none" stroke="${RED}" stroke-width="2.4"/>`;
  const mx = Math.max(...errs.flatMap(e => [e.train, Math.min(e.test, 3)])) * 1.05, F2 = frame({ W: 380, H, xdom: [1, errs.length], ydom: [0, mx], xlab: 'Grado del polinomio (complejidad)', ylab: 'RMSE', title: 'Error de entrenamiento y de prueba' });
  const body2 = F2.s.replace(/^<svg[^>]*>/, '').replace(/<style>.*?<\/style>/, '').replace(/<\/svg>$/, ''); s += `<g transform="translate(380 0)">${body2}`;
  s += `<path d="${path(errs.map((e, i) => [i + 1, e.train]), F2.sx, F2.sy)}" fill="none" stroke="${BLUE}" stroke-width="2.2"/><path d="${path(errs.map((e, i) => [i + 1, Math.min(e.test, mx)]), F2.sx, F2.sy)}" fill="none" stroke="${RED}" stroke-width="2.2"/>`;
  s += `<line x1="${F2.sx(deg)}" x2="${F2.sx(deg)}" y1="${F2.m.t}" y2="${H - F2.m.b}" stroke="${GREY}" stroke-dasharray="3 3"/><rect x="${F2.sx(1) + 10}" y="${F2.m.t + 8}" width="9" height="9" fill="${BLUE}"/><text x="${F2.sx(1) + 24}" y="${F2.m.t + 16}" class="s">Entrenamiento</text><rect x="${F2.sx(1) + 10}" y="${F2.m.t + 24}" width="9" height="9" fill="${RED}"/><text x="${F2.sx(1) + 24}" y="${F2.m.t + 32}" class="s">Prueba (datos nuevos)</text></g>`;
  return s + '</svg>';
}
