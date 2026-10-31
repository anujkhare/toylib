// Checks cnn.js against the logits JAX computed: node parity_check.js
const fs = require("fs");
const path = require("path");
global.atob = (s) => Buffer.from(s, "base64").toString("latin1");
global.window = global;
require("./cnn.js");
eval(fs.readFileSync(path.join(__dirname, "data.js"), "utf8"));
const D = window.EXPLORER_DATA;
const snapshot = D.snapshots.find((s) => s.epoch === D.parity.epoch);
const weights = CNN.decodeWeights(snapshot.weights);
const raw = Buffer.from(D.images, "base64");
let worst = 0;
for (let n = 0; n < D.parity.count; n++) {
  const image = new Float32Array(32 * 32 * 3);
  for (let i = 0; i < image.length; i++) {
    const c = i % 3;
    image[i] = (raw[n * 32 * 32 * 3 + i] / 255 - D.mean[c]) / D.std[c];
  }
  const t0 = Date.now();
  const out = CNN.forward({ data: image, h: 32, w: 32, c: 3 }, weights);
  const ms = Date.now() - t0;
  out.logits.forEach((l, k) => (worst = Math.max(worst, Math.abs(l - D.parity.logits[n][k]))));
  if (n === D.parity.count - 1) console.log("last forward pass took", ms, "ms");
}
console.log("largest logit difference from JAX:", worst.toFixed(5));
process.exit(worst < 1e-2 ? 0 : 1);
