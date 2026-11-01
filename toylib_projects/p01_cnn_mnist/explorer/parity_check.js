// Checks cnn.js against the logits JAX computed, and times it: node parity_check.js
const fs = require("fs");
const path = require("path");
global.atob = (s) => Buffer.from(s, "base64").toString("latin1");
global.window = global.self = global;
require("./cnn.js");
eval(fs.readFileSync(path.join(__dirname, "data.js"), "utf8"));
eval(fs.readFileSync(path.join(__dirname, "weights.js"), "utf8"));
const D = window.EXPLORER_DATA;
const weights = CNN.decodeWeights(self.EXPLORER_WEIGHTS);
const raw = Buffer.from(D.images, "base64");
const SIZE = D.size, PIXELS = SIZE * SIZE * 3;
let worst = 0, total = 0;
for (let n = 0; n < D.parity.length; n++) {
  const image = new Float32Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    const c = i % 3;
    image[i] = (raw[n * PIXELS + i] / 255 - D.mean[c]) / D.std[c];
  }
  const t0 = Date.now();
  const out = CNN.forward({ data: image, h: SIZE, w: SIZE, c: 3 }, weights);
  const ms = Date.now() - t0;
  if (n > 0) total += ms;
  out.logits.forEach((l, k) => (worst = Math.max(worst, Math.abs(l - D.parity[n][k]))));
}
console.log("mean forward pass, after the first:", Math.round(total / (D.parity.length - 1)), "ms");
console.log("largest logit difference from JAX:", worst.toFixed(5));
process.exit(worst < 1e-2 ? 0 : 1);
