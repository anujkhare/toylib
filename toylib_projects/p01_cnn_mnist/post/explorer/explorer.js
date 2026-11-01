// The post's interactive views. Each <div class="explorer" data-view="..."> becomes one
// view, and all of them share one photo. The data loads when a reader nears the first
// view after the photo bar, and the weights load in a worker only for a live run.
(function () {
  "use strict";
  const BASE = document.currentScript.src.replace(/[^/]*$/, "");
  const hosts = {};
  document.querySelectorAll(".explorer[data-view]").forEach((host) => {
    hosts[host.dataset.view] = host;
    host.append(el("p", { className: "ex-loading", textContent: "Loading the photos" }));
  });

  function el(tag, props = {}, ...kids) {
    const e = Object.assign(document.createElement(tag), props);
    e.append(...kids);
    return e;
  }
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fmtPct = (x, digits = 1) => (100 * x).toFixed(digits) + "%";
  const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

  // Colours for data, fixed in both themes because they encode values.
  const DIVERGING = [[16,66,129],[42,120,214],[134,182,239],[205,226,251],[240,239,236],[249,208,207],[236,138,137],[227,73,72],[156,44,43]];
  const SEQUENTIAL = [[252,252,251],[183,211,246],[109,167,236],[42,120,214],[13,54,107]];
  function ramp(stops, t) {
    const f = clamp(t, 0, 1) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
    return [0, 1, 2].map((k) => stops[i][k] + (stops[i + 1][k] - stops[i][k]) * u);
  }
  const diverging = (v, scale) => ramp(DIVERGING, 0.5 + 0.5 * v / (scale || 1));
  const sequential = (v) => ramp(SEQUENTIAL, v);

  function paint(canvas, w, h, color) {
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const [r, g, b] = color(x, y), o = (y * w + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function start() {
    const D = window.EXPLORER_DATA;
    const SIZE = D.size, PIXELS = SIZE * SIZE * 3, GRID = 10, CLASSES = D.classNames.length;
    const NUM_IMAGES = D.labels.length;
    const IMAGES = bytes(D.images), CAMS = new Float32Array(bytes(D.cams).buffer);
    const STAGE_NAMES = Object.keys(D.activations);
    const STAGE_MAPS = Object.fromEntries(STAGE_NAMES.map((s) => [s, bytes(D.activations[s].maps)]));
    const COUNTS = Object.fromEntries(STAGE_NAMES.map((s) => [s, D.activations[s].count]));
    const STAGE_LABEL = { stem: "stem", block1: "stage 1", block3: "stage 2", block5: "stage 3", block7: "stage 4" };
    const paintRGB = (canvas, rgb, offset = 0) =>
      paint(canvas, SIZE, SIZE, (x, y) => { const o = offset + (y * SIZE + x) * 3; return [rgb[o], rgb[o + 1], rgb[o + 2]]; });
    const thumb = (n) => { const c = el("canvas"); paintRGB(c, IMAGES, n * PIXELS); return c; };

    // Saved photos and live passes fill the same state, and every view reads it.
    const S = { index: 0, rgb: null, logits: null, probs: null, cam: null, maps: null, custom: false, edited: false, classIndex: null, top: 0 };
    const updaters = [];
    const render = () => updaters.forEach((u) => u());
    const setStatus = (text) => { if (statusLine) statusLine.textContent = text; };
    let statusLine = null;

    function loadSaved(n) {
      Object.assign(S, { index: n, custom: false, edited: false, classIndex: null });
      S.rgb = IMAGES.subarray(n * PIXELS, (n + 1) * PIXELS);
      S.logits = D.logits[n]; S.probs = D.probabilities[n];
      S.cam = CAMS.subarray(n * CLASSES * GRID * GRID, (n + 1) * CLASSES * GRID * GRID);
      S.maps = Object.fromEntries(STAGE_NAMES.map((s) => {
        const a = D.activations[s], each = a.side * a.side * a.count;
        return [s, STAGE_MAPS[s].subarray(n * each, (n + 1) * each)];
      }));
      setStatus("");
      render();
    }

    // A live pass on a worker that holds the weights. Only the newest request matters.
    let worker = null, requestId = 0, pending = null;
    function runLive(rgb) {
      const id = ++requestId;
      const send = () => {
        const image = new Float32Array(PIXELS);
        for (let i = 0; i < PIXELS; i++) image[i] = (rgb[i] / 255 - D.mean[i % 3]) / D.std[i % 3];
        worker.postMessage({ image, size: SIZE, id, counts: COUNTS }, [image.buffer]);
      };
      setStatus(worker && worker.ready ? "Running the network" : "Loading 11 MB of weights, then running the network");
      if (worker && worker.ready) { send(); return; }
      pending = send;
      if (worker) return;
      worker = new Worker(BASE + "worker.js");
      worker.onerror = () => setStatus("The network could not start in this browser. Saved photos still work.");
      worker.onmessage = ({ data }) => {
        if (data.ready) { worker.ready = true; if (pending) pending(); pending = null; return; }
        if (data.id !== requestId) return;
        S.logits = Array.from(data.logits);
        const top = Math.max(...S.logits), exp = S.logits.map((l) => Math.exp(l - top)), total = exp.reduce((a, b) => a + b, 0);
        S.probs = exp.map((e) => e / total); S.cam = data.cam; S.maps = data.maps;
        setStatus(`Ran on this photo in ${(data.ms / 1000).toFixed(2)} s`);
        render();
      };
    }

    function photoButton(n) {
      const name = D.classNames[D.labels[n]], said = D.classNames[D.predictions[n]];
      const b = el("button", { className: "ex-thumb", type: "button", title: name === said ? name : `${name}, predicted ${said}` }, thumb(n));
      b.setAttribute("aria-label", b.title);
      b.dataset.n = n;
      b.onclick = () => loadSaved(n);
      return b;
    }

    function buildPhoto(host) {
      const photo = el("canvas", { className: "ex-tile", width: SIZE, height: SIZE });
      const verdict = el("div", { className: "ex-verdict" }), truth = el("div", { className: "ex-truth" });
      const change = el("button", { className: "ex-btn", type: "button", textContent: "Change photo" });
      change.setAttribute("aria-expanded", "false");
      const input = el("input", { type: "file", accept: "image/*", id: "exUpload" });
      const upload = el("label", { className: "ex-upload ex-btn" }, input, el("span", { textContent: "Upload" }));
      statusLine = el("span", { className: "ex-cap" });
      statusLine.setAttribute("role", "status");
      const gallery = el("div", { className: "ex-gallery" });
      const indices = [...Array(NUM_IMAGES).keys()];
      gallery.append(...indices.filter((n) => n % 2 === 0 && D.predictions[n] === D.labels[n]).map(photoButton));
      const drawer = el("div", { className: "ex-drawer", hidden: true }, gallery);
      const toggle = () => { drawer.hidden = !drawer.hidden; change.setAttribute("aria-expanded", !drawer.hidden); };
      change.onclick = toggle;
      gallery.addEventListener("click", toggle);
      input.onchange = (e) => {
        const file = e.target.files[0]; if (!file) return;
        const url = URL.createObjectURL(file), img = new Image();
        img.onload = () => {
          // Same as the test photos: the central square, at 160x160.
          const c = el("canvas", { width: SIZE, height: SIZE }), ctx = c.getContext("2d", { willReadFrequently: true });
          const side = Math.min(img.width, img.height);
          ctx.imageSmoothingQuality = "high";
          ctx.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, SIZE, SIZE);
          const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data, rgb = new Uint8Array(PIXELS);
          for (let i = 0; i < SIZE * SIZE; i++) for (let k = 0; k < 3; k++) rgb[i * 3 + k] = rgba[i * 4 + k];
          Object.assign(S, { rgb, custom: true, edited: false, classIndex: null });
          paintRGB(photo, rgb); runLive(rgb);
          URL.revokeObjectURL(url);
        };
        img.onerror = () => setStatus("That file could not be read as an image. Try a JPEG or PNG.");
        img.src = url;
      };
      host.replaceChildren(
        el("div", { className: "ex-bar" }, photo, el("div", {}, verdict, truth), el("div", { className: "ex-actions" }, change, upload), statusLine),
        drawer);
      updaters.push(() => {
        paintRGB(photo, S.rgb);
        S.top = S.probs.indexOf(Math.max(...S.probs));
        verdict.textContent = `${D.classNames[S.top]}, ${fmtPct(S.probs[S.top], 0)}`;
        const label = D.labels[S.index];
        truth.textContent = S.custom ? "your photo" : `labelled ${D.classNames[label]}${S.top !== label ? ", so this is wrong" : ""}`;
        document.querySelectorAll(".ex-thumb").forEach((b) => b.setAttribute("aria-pressed", !S.custom && +b.dataset.n === S.index));
      });
    }

    const PRESETS = {
      "vertical edges": [-1, 0, 1, -2, 0, 2, -1, 0, 1],
      "horizontal edges": [-1, -2, -1, 0, 0, 0, 1, 2, 1],
      "blur": [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
      "sharpen": [0, -1, 0, -1, 5, -1, 0, -1, 0],
    };
    function buildConvolution(host) {
      let values = [...PRESETS["vertical edges"]];
      const gray = new Float32Array(SIZE * SIZE), out = new Float32Array(SIZE * SIZE);
      const before = el("canvas", { className: "ex-tile ex-square" }), after = el("canvas", { className: "ex-tile ex-square" });
      const presets = el("div", { className: "ex-chips" }), kernel = el("div", { className: "ex-kernel" });
      const inputs = values.map((_, i) => {
        const input = el("input", { type: "number", step: "0.1", id: `exKernel${i}` });
        input.setAttribute("aria-label", `Filter weight ${i + 1}`);
        input.oninput = () => { values[i] = parseFloat(input.value) || 0; marks(); compute(); };
        kernel.append(input);
        return input;
      });
      const sync = () => inputs.forEach((input, i) => (input.value = values[i]));
      const marks = () => [...presets.children].forEach((b) => b.setAttribute("aria-pressed", PRESETS[b.textContent].every((v, i) => v === values[i])));
      Object.keys(PRESETS).forEach((name) => {
        const b = el("button", { className: "ex-chip", type: "button", textContent: name });
        b.onclick = () => { values = [...PRESETS[name]]; sync(); marks(); compute(); };
        presets.append(b);
      });
      function compute() {
        for (let i = 0; i < gray.length; i++) gray[i] = (0.299 * S.rgb[i * 3] + 0.587 * S.rgb[i * 3 + 1] + 0.114 * S.rgb[i * 3 + 2]) / 127.5 - 1;
        let top = 1e-6;
        for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
          let sum = 0;
          for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
            const sy = y + ky - 1, sx = x + kx - 1;
            if (sy >= 0 && sx >= 0 && sy < SIZE && sx < SIZE) sum += values[ky * 3 + kx] * gray[sy * SIZE + sx];
          }
          out[y * SIZE + x] = sum; top = Math.max(top, Math.abs(sum));
        }
        paint(before, SIZE, SIZE, (x, y) => { const v = (gray[y * SIZE + x] + 1) * 127.5; return [v, v, v]; });
        paint(after, SIZE, SIZE, (x, y) => diverging(out[y * SIZE + x], top));
      }
      const col = (label, node, extra = "") => el("div", { className: `ex-col ${extra}` }, el("span", { className: "ex-cap", textContent: label }), node);
      host.replaceChildren(presets,
        el("div", { className: "ex-row ex-pair" }, col("your photo, in grey", before), col("filter", kernel, "ex-wide"), col("output: blue negative, red positive", after)));
      sync(); marks();
      updaters.push(compute);
    }

    function buildStages(host) {
      let stage = "block3", selected = 0;
      const tabs = el("div", { className: "ex-chips" }), grid = el("div", { className: "ex-channels" });
      const big = el("canvas", { className: "ex-tile ex-big" }), cap = el("span", { className: "ex-cap" });
      STAGE_NAMES.forEach((name) => {
        const side = D.activations[name].side;
        const b = el("button", { className: "ex-chip", type: "button", textContent: `${STAGE_LABEL[name]}, ${side}×${side}` });
        b.dataset.stage = name;
        b.onclick = () => { stage = name; selected = 0; update(); };
        tabs.append(b);
      });
      const draw = (canvas, ch) => {
        const a = D.activations[stage], maps = S.maps[stage];
        paint(canvas, a.side, a.side, (x, y) => sequential(maps[(y * a.side + x) * a.count + ch] / 255));
      };
      const tiles = Array.from({ length: 16 }, (_, i) => {
        const c = el("canvas"), b = el("button", { type: "button" }, c);
        b.setAttribute("aria-label", `Show channel ${i + 1} larger`);
        b.onclick = () => { selected = i; update(); };
        grid.append(b);
        return { b, c };
      });
      function update() {
        const a = D.activations[stage];
        [...tabs.children].forEach((b) => b.setAttribute("aria-pressed", b.dataset.stage === stage));
        tiles.forEach(({ b, c }, i) => {
          b.hidden = i >= a.count;
          if (i < a.count) { draw(c, i); b.setAttribute("aria-pressed", i === selected); }
        });
        draw(big, selected);
        cap.textContent = `${a.count} strongest of ${a.channels} channels`;
      }
      host.replaceChildren(tabs, el("div", { className: "ex-row" }, grid, el("div", { className: "ex-col", style: "flex:0 0 180px" }, cap, big)));
      updaters.push(update);
    }

    function buildEvidence(host) {
      const tabs = el("div", { className: "ex-chips" });
      D.classNames.forEach((_, i) => {
        const b = el("button", { className: "ex-chip", type: "button" });
        b.onclick = () => { S.classIndex = i; update(); };
        tabs.append(b);
      });
      const canvas = el("canvas", { className: "ex-tile ex-square ex-paint", width: SIZE, height: SIZE });
      const overlay = el("canvas", { className: "ex-tile ex-square", width: 320, height: 320 });
      const ledger = el("p", { className: "ex-sum" });
      const undo = el("button", { className: "ex-btn", type: "button", textContent: "Undo covering", hidden: true });
      const photo = el("canvas"), heat = el("canvas");
      function update() {
        const cls = S.classIndex ?? S.top, cam = S.cam.subarray(cls * GRID * GRID, (cls + 1) * GRID * GRID);
        [...tabs.children].forEach((b, i) => {
          b.textContent = `${D.classNames[i]} ${S.logits[i].toFixed(1)}`;
          b.setAttribute("aria-pressed", i === cls);
        });
        paintRGB(canvas, S.rgb); paintRGB(photo, S.rgb);
        let top = 1e-6, sum = 0;
        for (const v of cam) { top = Math.max(top, Math.abs(v)); sum += v; }
        paint(heat, GRID, GRID, (x, y) => diverging(cam[y * GRID + x], top));
        const ctx = overlay.getContext("2d");
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
        ctx.globalAlpha = 1; ctx.drawImage(photo, 0, 0, overlay.width, overlay.height);
        ctx.globalAlpha = 0.62; ctx.drawImage(heat, 0, 0, overlay.width, overlay.height); ctx.globalAlpha = 1;
        const mean = sum / cam.length, bias = S.logits[cls] - mean;
        ledger.textContent = `${D.classNames[cls]}: mean of 100 positions ${mean.toFixed(2)}  +  bias ${bias.toFixed(2)}  =  score ${S.logits[cls].toFixed(2)}`;
        undo.hidden = !S.edited || S.custom;
      }
      const stroke = (e) => {
        const b = canvas.getBoundingClientRect();
        const px = Math.floor((e.clientX - b.left) / b.width * SIZE), py = Math.floor((e.clientY - b.top) / b.height * SIZE);
        if (!S.edited) { S.rgb = S.rgb.slice(); S.edited = true; }
        // Covered pixels take the training set's mean colour, which is zero to the network.
        for (let y = py - 11; y <= py + 11; y++) for (let x = px - 11; x <= px + 11; x++)
          if (x >= 0 && y >= 0 && x < SIZE && y < SIZE) for (let k = 0; k < 3; k++) S.rgb[(y * SIZE + x) * 3 + k] = D.mean[k] * 255;
        paintRGB(canvas, S.rgb);
      };
      let down = false;
      canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); down = true; stroke(e); });
      canvas.addEventListener("pointermove", (e) => { if (down) stroke(e); });
      const finish = () => { if (down) { down = false; runLive(S.rgb); } };
      canvas.addEventListener("pointerup", finish);
      canvas.addEventListener("pointercancel", finish);
      undo.onclick = () => { if (!S.custom) { requestId++; loadSaved(S.index); } };
      const col = (label, node) => el("div", { className: "ex-col" }, el("span", { className: "ex-cap", textContent: label }), node);
      host.replaceChildren(tabs, el("div", { className: "ex-row ex-pair" }, col("drag to cover part of the photo", canvas), col("evidence for this class", overlay)), ledger, el("div", {}, undo));
      updaters.push(update);
    }

    function buildMistakes(host) {
      const grid = el("div", { className: "ex-confusion" }), M = D.confusion;
      let top = 0;
      M.forEach((row, i) => row.forEach((v, j) => { if (i !== j) top = Math.max(top, v); }));
      grid.append(el("span"));
      D.classNames.forEach((n) => grid.append(el("span", { className: "ex-label ex-top", textContent: n })));
      M.forEach((row, i) => {
        grid.append(el("span", { className: "ex-label", textContent: D.classNames[i] }));
        row.forEach((v, j) => {
          const t = i === j ? 0 : v / (top * 1.15), fill = sequential(t);
          grid.append(el("span", { className: "ex-cell", textContent: i === j ? "" : v, style: `background:${i === j ? "transparent" : `rgb(${fill.join(",")})`};color:${t > 0.5 ? "#fff" : "#15161a"}` }));
        });
      });
      const wrong = [...Array(NUM_IMAGES).keys()].filter((n) => D.predictions[n] !== D.labels[n]);
      host.replaceChildren(
        el("span", { className: "ex-cap", textContent: "true class (rows) against prediction, on all 3,925 test photos" }),
        el("div", { className: "ex-scroll" }, grid),
        el("span", { className: "ex-cap", textContent: "open a mistake, then see its evidence above" }),
        el("div", { className: "ex-gallery" }, ...wrong.map(photoButton)));
    }

    const BUILDERS = { photo: buildPhoto, convolution: buildConvolution, stages: buildStages, evidence: buildEvidence, mistakes: buildMistakes };
    Object.entries(hosts).forEach(([name, host]) => BUILDERS[name](host));
    loadSaved(0);
  }

  let loading = false;
  function loadData() {
    if (loading) return;
    loading = true;
    document.head.append(el("script", { src: BASE + "data.js", onload: start, onerror: () => {
      Object.values(hosts).forEach((host) => host.replaceChildren(el("p", { className: "ex-loading", textContent: "The interactive figures could not load." })));
    } }));
  }
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { observer.disconnect(); loadData(); }
  }, { rootMargin: "800px" });
  Object.entries(hosts).filter(([name]) => name !== "photo").forEach(([, host]) => observer.observe(host));
})();
