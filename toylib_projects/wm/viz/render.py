"""Render Episode → self-contained HTML.

Output is a single HTML string with everything inlined as base64 data-URIs:
  - animated WebP of the gameplay frames (auto-loops)
  - matplotlib PNG of state-over-time
  - JS-driven scrubber that shows per-frame metadata in a side panel

The same HTML works inline in Jupyter/Colab (wrap in IPython.display.HTML) or
as a standalone .html file you can double-click open.
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .loader import Episode

_ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT")


@dataclass
class RenderOptions:
    fps: int = 30  # playback frame rate
    downsample: int = 2  # take every Nth raw frame; at 60Hz native, 2→30Hz playback
    upscale: int = 2  # pixel-double frames for visibility (210×160 → 420×320)
    webp_quality: int = 70  # 0..100; per-frame WebP quality
    chart_dpi: int = 96
    chart_width_in: float = 9.0
    chart_height_in: float = 5.5


# ────────────────────────────────────────────────────────────────────────────
# Encoders
# ────────────────────────────────────────────────────────────────────────────


def _encode_frame_uris(frames: np.ndarray, opts: RenderOptions) -> list[str]:
    """Encode each frame as a base64 WebP data URI for JS-controlled canvas playback."""
    assert frames.ndim == 4 and frames.dtype == np.uint8
    h, w = frames.shape[1:3]
    dw, dh = w * opts.upscale, h * opts.upscale
    uris = []
    for f in frames:
        img = Image.fromarray(f)
        if opts.upscale != 1:
            img = img.resize((dw, dh), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=opts.webp_quality, method=2)
        uris.append(_b64(buf.getvalue(), "image/webp"))
    return uris


def _encode_state_chart(ep: Episode, opts: RenderOptions) -> bytes:
    """Stacked time-series chart of all state variables → PNG bytes."""
    L = ep.length
    t = np.arange(L)
    s = ep.states

    fig, axes = plt.subplots(
        4, 1,
        figsize=(opts.chart_width_in, opts.chart_height_in),
        dpi=opts.chart_dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 1, 1]},
    )

    # Paddle x vs ball x.
    axes[0].plot(t, s["paddle_x"], label="paddle_x", color="tab:blue", linewidth=1.0)
    axes[0].plot(t, s["ball_x"], label="ball_x", color="tab:orange", linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("x")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Ball y (inverted: 0 = top of screen, increasing = falling).
    axes[1].plot(t, s["ball_y"], color="tab:green", linewidth=0.8)
    axes[1].set_ylabel("ball_y")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3)

    # Score.
    axes[2].plot(t, s["score"], color="tab:red", linewidth=1.0)
    axes[2].set_ylabel("score")
    axes[2].grid(alpha=0.3)

    # Lives — step-like.
    axes[3].plot(t, s["lives"], color="tab:purple", linewidth=1.0, drawstyle="steps-post")
    axes[3].set_ylabel("lives")
    axes[3].set_xlabel("frame index")
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"{ep.name}   L={L}", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=opts.chart_dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ────────────────────────────────────────────────────────────────────────────


_PAGE_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }
h1 { margin: 0 0 8px 0; font-size: 18px; }
.subtitle { color: #666; font-size: 13px; margin-bottom: 16px; }
.viewer { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.viewer canvas { image-rendering: pixelated; border: 1px solid #ddd; display: block; }
.controls { margin-top: 8px; display: flex; align-items: center; gap: 8px; }
.controls button { font-size: 16px; background: none; border: 1px solid #ccc;
                   border-radius: 4px; cursor: pointer; padding: 3px 12px; line-height: 1.4; }
.controls button:hover { background: #f0f0f0; }
.panel { font-family: ui-monospace, Menlo, monospace; font-size: 13px;
         background: #f7f7f8; padding: 12px 16px; border-radius: 6px;
         border: 1px solid #e2e2e2; min-width: 220px; }
.panel .k { color: #888; display: inline-block; width: 70px; }
.panel .v { color: #111; font-weight: 600; }
.scrubber { margin-top: 16px; max-width: 640px; }
.scrubber input { width: 100%; }
.scrubber .label { font-family: ui-monospace, Menlo, monospace;
                   font-size: 12px; color: #555; margin-top: 4px; }
.chart { margin-top: 24px; }
.chart img { max-width: 100%; border: 1px solid #eee; }
.nav { margin-bottom: 12px; font-size: 13px; }
.nav a { color: #06c; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
"""

_VIEWER_TEMPLATE = """
<div class="wm-viz" id="{id}">
  <h1>{title}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="viewer">
    <div>
      <canvas id="{id}-canvas" width="{webp_w}" height="{webp_h}"></canvas>
      <div class="controls">
        <button id="{id}-play" title="Play / Pause">&#9646;&#9646;</button>
      </div>
    </div>
    <div class="panel">
      <div><span class="k">frame</span> <span class="v" id="{id}-f">0</span></div>
      <div><span class="k">action</span> <span class="v" id="{id}-a">—</span></div>
      <div><span class="k">paddle_x</span> <span class="v" id="{id}-px">—</span></div>
      <div><span class="k">ball_x</span> <span class="v" id="{id}-bx">—</span></div>
      <div><span class="k">ball_y</span> <span class="v" id="{id}-by">—</span></div>
      <div><span class="k">score</span> <span class="v" id="{id}-sc">—</span></div>
      <div><span class="k">lives</span> <span class="v" id="{id}-lv">—</span></div>
      <div><span class="k">bricks</span> <span class="v" id="{id}-br">—</span></div>
    </div>
  </div>
  <div class="scrubber">
    <input type="range" id="{id}-slider" min="0" max="{n_frames_minus_one}" value="0" step="1" />
    <div class="label" id="{id}-slabel">frame 0 / {last_frame_idx}</div>
  </div>
  <div class="chart"><img src="{chart_uri}" /></div>
</div>
<script>
(function() {{
  var id = {id_json};
  var fps = {fps};
  var root = document.getElementById(id);
  var canvas = root.querySelector('#' + id + '-canvas');
  var ctx = canvas.getContext('2d');
  var slider = root.querySelector('#' + id + '-slider');
  var playBtn = root.querySelector('#' + id + '-play');
  var slabel = root.querySelector('#' + id + '-slabel');
  var data = {data_json};
  var N = data.frame_uris.length;
  var msPerFrame = 1000 / fps;

  var imgs = data.frame_uris.map(function(uri) {{
    var img = new Image();
    img.src = uri;
    return img;
  }});

  var cur = 0;
  var playing = true;
  var rafId = null;
  var lastTs = null;
  var acc = 0;

  function showFrame(i) {{
    cur = i;
    slider.value = i;
    slabel.textContent = 'frame ' + data.frame_indices[i] + ' / ' + data.frame_indices[N - 1];
    root.querySelector('#' + id + '-f').textContent  = data.frame_indices[i];
    root.querySelector('#' + id + '-a').textContent  = data.action_names[data.actions[i]];
    root.querySelector('#' + id + '-px').textContent = data.paddle_x[i].toFixed(0);
    root.querySelector('#' + id + '-bx').textContent = data.ball_x[i].toFixed(0);
    root.querySelector('#' + id + '-by').textContent = data.ball_y[i].toFixed(0);
    root.querySelector('#' + id + '-sc').textContent = data.score[i];
    root.querySelector('#' + id + '-lv').textContent = data.lives[i];
    root.querySelector('#' + id + '-br').textContent = data.bricks_remaining[i];
    if (imgs[i].complete) {{
      ctx.drawImage(imgs[i], 0, 0);
    }} else {{
      imgs[i].onload = function() {{ ctx.drawImage(imgs[i], 0, 0); }};
    }}
  }}

  function tick(ts) {{
    if (!playing) return;
    if (lastTs === null) lastTs = ts;
    acc += ts - lastTs;
    lastTs = ts;
    var advanced = false;
    while (acc >= msPerFrame) {{
      acc -= msPerFrame;
      cur = (cur + 1) % N;
      advanced = true;
    }}
    if (advanced) showFrame(cur);
    rafId = requestAnimationFrame(tick);
  }}

  function pause() {{
    playing = false;
    playBtn.innerHTML = '&#9654;';
    if (rafId !== null) {{ cancelAnimationFrame(rafId); rafId = null; }}
  }}

  function play() {{
    playing = true;
    playBtn.innerHTML = '&#9646;&#9646;';
    lastTs = null;
    acc = 0;
    rafId = requestAnimationFrame(tick);
  }}

  playBtn.addEventListener('click', function() {{
    if (playing) pause(); else play();
  }});

  slider.addEventListener('mousedown', function() {{
    if (playing) pause();
  }});

  slider.addEventListener('input', function(e) {{
    showFrame(parseInt(e.target.value));
  }});

  showFrame(0);
  rafId = requestAnimationFrame(tick);
}})();
</script>
"""

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>
{nav}
{body}
</body></html>
"""


def _format_subtitle(ep: Episode) -> str:
    s = ep.states
    return (
        f"shard: {ep.shard_path.name} &middot; "
        f"mode: {ep.mode} &middot; difficulty: {ep.difficulty} &middot; "
        f"length: {ep.length} &middot; "
        f"score: {int(s['score'][-1])} &middot; "
        f"lives: {int(s['lives'][0])} → {int(s['lives'][-1])}"
    )


def build_episode_viewer(
    ep: Episode, opts: RenderOptions | None = None, dom_id: str | None = None
) -> str:
    """Return an HTML *fragment* (no <html>/<body>) for one episode.

    Suitable for inlining alongside other content. Use ``build_episode_page``
    for a full standalone document.
    """
    opts = opts or RenderOptions()
    dom_id = dom_id or f"viz-{abs(hash(ep.stem)) % 10**9}"

    frames_ds = ep.frames[:: opts.downsample]
    n = len(frames_ds)
    chart_bytes = _encode_state_chart(ep, opts)
    frame_uris = _encode_frame_uris(frames_ds, opts)

    canvas_h = ep.frames.shape[1] * opts.upscale
    canvas_w = ep.frames.shape[2] * opts.upscale

    # Original episode frame indices for each display frame (for the panel label).
    frame_indices = list(range(0, ep.length, opts.downsample))[:n]

    data = {
        "frame_uris": frame_uris,
        "frame_indices": frame_indices,
        "action_names": list(_ACTION_NAMES),
        "actions": ep.actions[:: opts.downsample][:n].astype(int).tolist(),
        **{k: ep.states[k][:: opts.downsample][:n].tolist()
           for k in ("paddle_x", "ball_x", "ball_y", "score", "lives", "bricks_remaining")},
    }

    return _VIEWER_TEMPLATE.format(
        id=dom_id,
        title=html.escape(f"{ep.shard_path.stem} / {ep.name}"),
        subtitle=_format_subtitle(ep),
        webp_w=canvas_w,
        webp_h=canvas_h,
        chart_uri=_b64(chart_bytes, "image/png"),
        n_frames_minus_one=n - 1,
        last_frame_idx=frame_indices[-1],
        fps=opts.fps,
        data_json=json.dumps(data),
        id_json=json.dumps(dom_id),
    )


def build_episode_page(
    ep: Episode,
    opts: RenderOptions | None = None,
    nav_html: str = "",
) -> str:
    """Return a full standalone HTML document for one episode."""
    return _PAGE_TEMPLATE.format(
        title=html.escape(f"{ep.name} — {ep.shard_path.name}"),
        css=_PAGE_CSS,
        nav=nav_html,
        body=build_episode_viewer(ep, opts),
    )


# ────────────────────────────────────────────────────────────────────────────
# Shard index page
# ────────────────────────────────────────────────────────────────────────────


def build_shard_index_page(rows: list[dict], title: str = "Episode index") -> str:
    """Build an index HTML listing all episodes.

    Each row dict should contain: name, length, score, lives_start, lives_end,
    and href (relative link to the per-episode page).
    """
    table_rows = "".join(
        f"<tr><td><a href='{html.escape(r['href'])}'>{html.escape(r['name'])}</a></td>"
        f"<td style='text-align:right'>{r.get('mode', '')}</td>"
        f"<td style='text-align:right'>{r.get('difficulty', '')}</td>"
        f"<td style='text-align:right'>{r['length']}</td>"
        f"<td style='text-align:right'>{r['score']}</td>"
        f"<td style='text-align:right'>{r['lives_start']} → {r['lives_end']}</td></tr>"
        for r in rows
    )
    table = (
        "<table style='border-collapse:collapse;font-family:ui-monospace,monospace;font-size:13px'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px 12px;border-bottom:1px solid #ddd'>episode</th>"
        "<th style='text-align:right;padding:4px 12px;border-bottom:1px solid #ddd'>mode</th>"
        "<th style='text-align:right;padding:4px 12px;border-bottom:1px solid #ddd'>diff</th>"
        "<th style='text-align:right;padding:4px 12px;border-bottom:1px solid #ddd'>length</th>"
        "<th style='text-align:right;padding:4px 12px;border-bottom:1px solid #ddd'>score</th>"
        "<th style='text-align:right;padding:4px 12px;border-bottom:1px solid #ddd'>lives</th>"
        "</tr></thead><tbody>"
        f"{table_rows}"
        "</tbody></table>"
    )
    body = f"<h1>{html.escape(title)}</h1><p class='subtitle'>{len(rows)} episodes</p>{table}"
    return _PAGE_TEMPLATE.format(title=html.escape(title), css=_PAGE_CSS, nav="", body=body)
