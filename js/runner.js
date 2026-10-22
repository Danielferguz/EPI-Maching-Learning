// Puente con el motor: Web Worker si está disponible; si no, el mismo código en el hilo principal.
let worker = null, fallback = null, seq = 0; const pending = new Map();
function ensure() {
  if (worker || fallback) return;
  try {
    worker = new Worker(new URL('./worker.js?v=2026.10.22-1', import.meta.url), { type: 'module' });
    worker.onmessage = e => { const { id, progress, result, error } = e.data, p = pending.get(id); if (!p) return; if (progress) p.on?.(progress); else { pending.delete(id); error ? p.rej(new Error(error)) : p.res(result); } };
    worker.onerror = ev => { console.warn('Worker no disponible, se usa el hilo principal', ev.message); worker = null; import('./engine.js?v=2026.10.22-1').then(m => (fallback = m)); };
  } catch { import('./engine.js?v=2026.10.22-1').then(m => (fallback = m)); }
}
export async function call(cmd, payload, onProgress) {
  ensure();
  if (!worker && !fallback) fallback = await import('./engine.js?v=2026.10.22-1');
  if (fallback) return fallback.handle(cmd, payload, onProgress);
  const id = ++seq; return new Promise((res, rej) => { pending.set(id, { res, rej, on: onProgress }); worker.postMessage({ id, cmd, payload }); });
}
