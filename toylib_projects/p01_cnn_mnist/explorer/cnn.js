// The post 1 CNN's forward pass in plain JavaScript, on Float32Arrays in HWC order.
// Every intermediate tensor is returned so the page can draw it.
(function (root) {
  const SHAPES = { conv1: [3, 32], conv2: [32, 64], conv3: [64, 64] };

  function decode(base64) {
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    return new Float32Array(bytes.buffer);
  }

  function decodeWeights(w) {
    const out = {};
    for (const key of Object.keys(w)) out[key] = decode(w[key]);
    return out;
  }

  // A tensor is { data, h, w, c }, with data[(y * w + x) * c + channel].
  function conv3x3(x, weights, bias, outChannels) {
    const { data, h, w, c } = x;
    const out = new Float32Array(h * w * outChannels);
    for (let y = 0; y < h; y++) {
      for (let px = 0; px < w; px++) {
        const o = (y * w + px) * outChannels;
        for (let ky = -1; ky <= 1; ky++) {
          const sy = y + ky;
          if (sy < 0 || sy >= h) continue; // SAME padding: outside counts as zero.
          for (let kx = -1; kx <= 1; kx++) {
            const sx = px + kx;
            if (sx < 0 || sx >= w) continue;
            const s = (sy * w + sx) * c;
            // Weights are HWIO: (kernel row, kernel column, input, output).
            const k = ((ky + 1) * 3 + (kx + 1)) * c * outChannels;
            for (let i = 0; i < c; i++) {
              const v = data[s + i];
              if (v === 0) continue;
              const row = k + i * outChannels;
              for (let oc = 0; oc < outChannels; oc++) out[o + oc] += v * weights[row + oc];
            }
          }
        }
        for (let oc = 0; oc < outChannels; oc++) out[o + oc] += bias[oc];
      }
    }
    return { data: out, h, w, c: outChannels };
  }

  function relu(x) {
    const out = new Float32Array(x.data.length);
    for (let i = 0; i < out.length; i++) out[i] = x.data[i] > 0 ? x.data[i] : 0;
    return { ...x, data: out };
  }

  function maxPool(x) {
    const { data, h, w, c } = x;
    const oh = h >> 1;
    const ow = w >> 1;
    const out = new Float32Array(oh * ow * c);
    for (let y = 0; y < oh; y++) {
      for (let px = 0; px < ow; px++) {
        for (let i = 0; i < c; i++) {
          const a = data[((2 * y) * w + 2 * px) * c + i];
          const b = data[((2 * y) * w + 2 * px + 1) * c + i];
          const d = data[((2 * y + 1) * w + 2 * px) * c + i];
          const e = data[((2 * y + 1) * w + 2 * px + 1) * c + i];
          out[(y * ow + px) * c + i] = Math.max(a, b, d, e);
        }
      }
    }
    return { data: out, h: oh, w: ow, c };
  }

  function forward(image, weights) {
    const conv1Raw = conv3x3(image, weights.conv1_w, weights.conv1_b, 32);
    const conv1 = relu(conv1Raw);
    const pool1 = maxPool(conv1);
    const conv2Raw = conv3x3(pool1, weights.conv2_w, weights.conv2_b, 64);
    const conv2 = relu(conv2Raw);
    const pool2 = maxPool(conv2);
    const conv3Raw = conv3x3(pool2, weights.conv3_w, weights.conv3_b, 64);
    const conv3 = relu(conv3Raw);

    const logits = new Float32Array(10);
    for (let k = 0; k < 10; k++) logits[k] = weights.head_b[k];
    const flat = conv3.data;
    for (let i = 0; i < flat.length; i++) {
      const v = flat[i];
      if (v === 0) continue;
      const row = i * 10;
      for (let k = 0; k < 10; k++) logits[k] += v * weights.head_w[row + k];
    }
    return { conv1Raw, conv1, pool1, conv2Raw, conv2, pool2, conv3Raw, conv3, logits };
  }

  // How much each of the 8x8 positions adds to one class's score. The sum of
  // these plus the head's bias is exactly that class's logit.
  function evidence(conv3, weights, classIndex) {
    const out = new Float32Array(conv3.h * conv3.w);
    const { c } = conv3;
    for (let p = 0; p < out.length; p++) {
      let sum = 0;
      for (let i = 0; i < c; i++) {
        const v = conv3.data[p * c + i];
        if (v !== 0) sum += v * weights.head_w[(p * c + i) * 10 + classIndex];
      }
      out[p] = sum;
    }
    return out;
  }

  function softmax(logits) {
    const top = Math.max(...logits);
    const exp = Array.from(logits, (l) => Math.exp(l - top));
    const total = exp.reduce((a, b) => a + b, 0);
    return exp.map((e) => e / total);
  }

  const api = { decode, decodeWeights, conv3x3, relu, maxPool, forward, evidence, softmax, SHAPES };
  if (typeof module !== "undefined") module.exports = api;
  root.CNN = api;
})(typeof window !== "undefined" ? window : globalThis);
