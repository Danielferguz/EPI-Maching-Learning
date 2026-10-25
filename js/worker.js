// Web Worker: ejecuta el motor fuera del hilo de la interfaz.
import { handle } from './engine.js?v=2026.10.25-6';
self.onmessage = async e => {
  const { id, cmd, payload } = e.data;
  try { const out = await handle(cmd, payload, m => self.postMessage({ id, progress: m })); self.postMessage({ id, result: out }); }
  catch (err) { self.postMessage({ id, error: err.message || String(err) }); }
};
