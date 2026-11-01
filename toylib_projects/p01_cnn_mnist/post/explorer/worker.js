// Runs the forward pass off the main thread, and loads the weights only here.
importScripts("cnn.js", "weights.js");
const weights = CNN.decodeWeights(self.EXPLORER_WEIGHTS);

// The strongest channels of a stage, scaled to bytes, as build_data.py exports them.
function strongest(t, count) {
  const totals = new Float32Array(t.c);
  for (let i = 0; i < t.data.length; i++) totals[i % t.c] += t.data[i];
  const order = [...totals.keys()].sort((a, b) => totals[b] - totals[a]).slice(0, count);
  let top = 1e-6;
  for (const ch of order) for (let p = ch; p < t.data.length; p += t.c) top = Math.max(top, t.data[p]);
  const out = new Uint8Array(t.h * t.w * count);
  for (let p = 0; p < t.h * t.w; p++) order.forEach((ch, i) => (out[p * count + i] = (t.data[p * t.c + ch] / top) * 255));
  return out;
}

self.onmessage = ({ data: { image, size, id, counts } }) => {
  const t0 = performance.now();
  const { acts, logits, cam } = CNN.forward({ data: image, h: size, w: size, c: 3 }, weights);
  const maps = {};
  for (const [stage, count] of Object.entries(counts)) maps[stage] = strongest(acts[stage], count);
  self.postMessage({ id, logits, cam, maps, ms: performance.now() - t0 });
};
self.postMessage({ ready: true });
