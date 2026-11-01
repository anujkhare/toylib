// The post's ResNet-18-shaped model in plain JavaScript, on Float32Arrays in HWC order.
// Each stage's activations are returned so the page can draw them.
(function (root) {
  const NUM_GROUPS = 8, EPS = 1e-5;

  function decode(base64) {
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    return new Float32Array(bytes.buffer);
  }

  // A tensor is { data, h, w, c }, with data[(y * w + x) * c + channel].
  // SAME padding, matching jax.lax.conv_general_dilated. Weights are HWIO.
  function conv(x, weights, kernel, stride, outChannels) {
    const { data, h, w, c } = x;
    const oh = Math.ceil(h / stride), ow = Math.ceil(w / stride);
    const padY = Math.floor(Math.max((oh - 1) * stride + kernel - h, 0) / 2);
    const padX = Math.floor(Math.max((ow - 1) * stride + kernel - w, 0) / 2);
    const out = new Float32Array(oh * ow * outChannels);
    for (let y = 0; y < oh; y++) {
      for (let px = 0; px < ow; px++) {
        const o = (y * ow + px) * outChannels;
        for (let ky = 0; ky < kernel; ky++) {
          const sy = y * stride - padY + ky;
          if (sy < 0 || sy >= h) continue; // Outside the image counts as zero.
          for (let kx = 0; kx < kernel; kx++) {
            const sx = px * stride - padX + kx;
            if (sx < 0 || sx >= w) continue;
            const s = (sy * w + sx) * c;
            const k = (ky * kernel + kx) * c * outChannels;
            for (let i = 0; i < c; i++) {
              const v = data[s + i];
              if (v === 0) continue;
              const row = k + i * outChannels;
              for (let oc = 0; oc < outChannels; oc++) out[o + oc] += v * weights[row + oc];
            }
          }
        }
      }
    }
    return { data: out, h: oh, w: ow, c: outChannels };
  }

  // Normalizes per group over height, width and the group's channels, one image at a time.
  function groupNorm(x, scale, bias) {
    const { data, c } = x, perGroup = c / NUM_GROUPS, pixels = data.length / c;
    const out = new Float32Array(data.length);
    for (let g = 0; g < NUM_GROUPS; g++) {
      const first = g * perGroup;
      let sum = 0, squares = 0;
      for (let p = 0; p < pixels; p++) {
        for (let i = first; i < first + perGroup; i++) {
          const v = data[p * c + i];
          sum += v; squares += v * v;
        }
      }
      const n = pixels * perGroup, mean = sum / n, inverse = 1 / Math.sqrt(squares / n - mean * mean + EPS);
      for (let p = 0; p < pixels; p++) {
        for (let i = first; i < first + perGroup; i++) {
          out[p * c + i] = (data[p * c + i] - mean) * inverse * scale[i] + bias[i];
        }
      }
    }
    return { ...x, data: out };
  }

  function relu(x) {
    const out = new Float32Array(x.data.length);
    for (let i = 0; i < out.length; i++) out[i] = x.data[i] > 0 ? x.data[i] : 0;
    return { ...x, data: out };
  }

  function add(a, b) {
    const out = new Float32Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
    return { ...a, data: out };
  }

  function block(x, w, name, outChannels, stride) {
    let y = relu(groupNorm(conv(x, w[name + ".conv1"], 3, stride, outChannels), w[name + ".norm1.scale"], w[name + ".norm1.bias"]));
    y = groupNorm(conv(y, w[name + ".conv2"], 3, 1, outChannels), w[name + ".norm2.scale"], w[name + ".norm2.bias"]);
    const project = w[name + ".project"];
    return relu(add(y, project ? conv(x, project, 1, stride, outChannels) : x));
  }

  function forward(image, w) {
    const acts = {};
    let x = relu(groupNorm(conv(image, w.stem, 7, 2, 32), w["stem_norm.scale"], w["stem_norm.bias"]));
    acts.stem = x;
    for (let i = 0; i < 8; i++) {
      const stage = i >> 1;
      x = block(x, w, `block${i}`, 32 << stage, stage > 0 && i % 2 === 0 ? 2 : 1);
      acts[`block${i}`] = x;
    }
    const logits = new Float32Array(10);
    for (let k = 0; k < 10; k++) {
      let sum = w["head.bias"][k];
      for (let i = 0; i < x.c; i++) {
        let mean = 0;
        for (let p = i; p < x.data.length; p += x.c) mean += x.data[p];
        sum += (mean / (x.h * x.w)) * w["head.weights"][i * 10 + k];
      }
      logits[k] = sum;
    }
    // What each of the 10x10 positions adds to each class, 100 values per class.
    const cam = new Float32Array(x.h * x.w * 10);
    for (let p = 0; p < x.h * x.w; p++) {
      for (let i = 0; i < x.c; i++) {
        const v = x.data[p * x.c + i];
        if (v !== 0) for (let k = 0; k < 10; k++) cam[k * x.h * x.w + p] += v * w["head.weights"][i * 10 + k];
      }
    }
    return { acts, logits, cam };
  }

  function softmax(logits) {
    const top = Math.max(...logits);
    const exp = Array.from(logits, (l) => Math.exp(l - top));
    const total = exp.reduce((a, b) => a + b, 0);
    return exp.map((e) => e / total);
  }

  function decodeWeights(bundle) {
    const all = decode(bundle.data), out = {};
    for (const [name, offset, length] of bundle.index) out[name] = all.subarray(offset, offset + length);
    return out;
  }

  const api = { decode, decodeWeights, conv, groupNorm, relu, forward, softmax };
  if (typeof module !== "undefined") module.exports = api;
  root.CNN = api;
})(typeof self !== "undefined" ? self : globalThis);
