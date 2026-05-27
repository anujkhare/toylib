# Walkthrough Authoring Guidelines

This document tells a future Claude session (or human author) exactly how to write a learning walkthrough for one implementation track in `docs/walkthroughs/`. Read this before writing any walkthrough.

---

## 1. What a walkthrough is — and is not

| | Design doc (`docs/designs/`) | Walkthrough (`docs/walkthroughs/`) |
|---|---|---|
| **Primary reader** | Builder deciding *how* to build it | Learner building it for the first time |
| **Voice** | Deliberative — weighs options, records decisions | Guiding — commits to one path, explains it |
| **Coverage** | Complete — all trade-offs, all open questions | Selective — one end-to-end path |
| **Alternatives** | Central | Sidebar |
| **Entry point** | Before implementation | Alongside implementation |

A walkthrough is not a tutorial (it leads to real, tested code), not a reference (you read it once, start to finish), and not a design doc (it never hedges). Its job is to take a reader from *"I understand the problem"* to *"I have a working, tested implementation"* through a structured narrative.

The design doc is the source of record for *what* to build and *why* certain decisions were made. The walkthrough is derived from it, but written for a completely different audience — it stands entirely on its own. The reader of a walkthrough never needs to consult the design doc. The walkthrough may go deeper on explanations, add visualizations, and cover implementation detail that the design doc omits entirely.

---

## 2. Assumed reader profile

Before writing, state explicitly what the reader knows. Default assumption for this project:

**Knows:**
- Python, NumPy; basic JAX/Flax (can write a simple training loop)
- Backpropagation, gradient descent, convnets
- What an autoencoder is; what attention is

**Does not know:**
- The specific technique being introduced (flow matching, DiT, VAE, RoPE, etc.)
- The specific architecture choices made for this project
- The Breakout domain in any detail

Do not re-explain what the reader already knows. Do not assume knowledge they don't have.

---

## 3. Document structure (canonical section order)

Every walkthrough follows this structure. Use these exact heading names.

```
# Track Ax: <Short Title>

## Overview
## Prerequisites
## The Problem We're Solving
## Core Concept: <Name of the key technique>
## Architecture
  ### <Component 1>
  ### <Component 2>
  ...
## Training
## Implementation Milestones
  ### Milestone 1: <Name>
  ### Milestone 2: <Name>
  ...
## Further Reading
```

### What goes in each section

**Overview** (5–10 lines): One paragraph on what this track builds and why it exists in the project sequence. End with a diagram of the full pipeline at a glance.

**Prerequisites** (bullet list): What background knowledge is assumed. Link to a prior walkthrough if this track builds on one.

**The Problem We're Solving**: Motivate with the *pain* before offering the solution. Show what breaks if you try the naïve approach. The reader should feel the problem pressure before seeing the architecture. No equations yet.

**Core Concept**: Introduce the key technique — flow matching, KL regularization, temporal attention, etc. — on a toy example *first*. Use 1D or 2D data. Show a runnable code snippet or a clear diagram. Then generalize to the actual setting. Equations come here, but always annotated with shapes.

**Architecture** (one subsection per major component): For each component follow this micro-structure:
1. *Purpose* — one sentence on what this component does
2. *Intuition* — a diagram or visual analogy before the math
3. *Math* — equations with shapes annotated on every tensor
4. *Shape trace* — explicit step-by-step tensor flow through this component
5. *Interface* — the function/class signature the reader will implement (see §6)

**Training**: Loss derivation, optimizer setup, what to log and what healthy training looks like. Include expected loss curves or ranges where known.

**Implementation Milestones**: See §6.

**Further Reading**: Papers, sections of the design doc, reference repos. One line of annotation per link.

---

## 4. Content rules

### The one-path rule
Commit to one approach. Write *"we use X"*, not *"we could use X or Y"*. If an alternative is worth knowing, put it in a short `> **Alternative:** ...` blockquote at the end of the section with enough explanation to understand it without going elsewhere. The main text never hedges.

### Intuition before formalism
Always motivate with a diagram or concrete example before introducing an equation. If the reader can't picture what an equation is computing, the equation is dead weight.

### Narrative arc
Each section should leave the reader with a mild sense of *inevitability* — that the introduced mechanism is the natural answer to the pressure established in the previous section. The walkthrough is a story, not a catalog.

### Name confusions explicitly
When two things are easy to mix up (e.g., flow matching "velocity" vs. diffusion "score"), say so directly: *"This is easy to confuse with X — the difference is..."*

### Shape annotations on every tensor
Every tensor that appears in an equation or a code snippet must have its shape written beside it. Use the format `(B, T, D)` consistently. Define dimension names once at the start of the Architecture section.

### "Watch out" callouts
Mark common implementation pitfalls explicitly:
```
> **Watch out:** JAX's `jnp.take` silently clips out-of-bounds indices.
> Use `mode="fill"` or add an assertion.
```
Use these sparingly — one or two per major component, for genuinely tricky things.

---

## 5. Visualization guidelines

Visualizations are not decoration. Each one should answer a specific question the reader has at that point in the walkthrough. A well-placed visualization can replace two pages of prose.

### When to visualize
- At the start of the Architecture section: a full-pipeline diagram showing every major component and the tensor shapes flowing between them
- Whenever a data transformation changes shape, format, or meaning (patchify, latent encoding, attention masking)
- Whenever an architectural component has a non-obvious information flow (e.g. AdaLN-zero, cross-attention, RoPE)
- Whenever a concept benefits from seeing it on actual data (positional encoding patterns, flow interpolation paths, attention maps on real frames)

### Target standard: interactive HTML/JS

The primary visualization format for architecture diagrams and data-flow diagrams is **self-contained interactive HTML** — JavaScript-driven, richly annotated, and embeddable in the walkthrough as a standalone file in `docs/walkthroughs/assets/<track>/`. These are not placeholders or sketches: they are polished, reusable assets built to a consistent visual style.

A shared **visualization library** (`docs/walkthroughs/assets/vizlib/`) will define the design system — color palette, typography, node/edge styling, animation conventions — so all diagrams across all tracks look like they belong to the same document. This library is developed and refined as a separate effort; see the note at the end of this section.

**What interactive diagrams should do:**
- Annotate every edge with tensor shapes, using a consistent format: `(B, N, D)`
- Reveal detail on hover: clicking a component shows its purpose, math, and shape trace without leaving the diagram
- Animate data flow to show how information moves through the forward pass step by step
- Highlight which components are frozen vs. trainable (relevant from A2 onwards)
- Use a consistent color language: e.g., one color for spatial operations, another for temporal, another for conditioning mechanisms

### Fallback formats (for drafts or inline prose)

While the visualization library is being developed, or for quick shape traces embedded in prose, use these fallback formats:

**ASCII shape traces** for any component's tensor flow — portable, always renderable:
```
input:  (B, H, W, 3)       ← raw frame, 3 RGB channels
  → patchify (p=4):  (B, N, p²·3)   N = (H/p)·(W/p) = 256
  → linear proj:     (B, N, D)       D = model dim
  → + pos embed:     (B, N, D)
  → transformer ×L:  (B, N, D)
  → unpatchify:      (B, H, W, 3)
```

**Self-contained Python snippets** (runnable in a notebook) for concepts that require actual data to understand — flow interpolation, positional encoding patterns, attention distributions. Keep under 25 lines; include a comment describing what the reader should observe.

### Note on the visualization library

The interactive visualization library is being designed as a separate effort and will be documented in `docs/walkthroughs/assets/vizlib/README.md` once ready. When authoring a walkthrough before the library exists: mark diagram slots with a `<!-- VIZ: description of what this diagram should show -->` comment so they can be filled in later without restructuring the walkthrough. Do not block walkthrough authoring on the library being finished.

---

## 6. Milestone guidelines

Milestones divide the track into discrete, independently verifiable units of progress. The walkthrough defines *what* each milestone produces and *how to know it's working*. Translating milestones into concrete interfaces, stubs, and test code is a separate downstream step, not part of the walkthrough.

### Structure of a milestone

```markdown
### Milestone N: <Component name>

**What to build:** One sentence describing the deliverable.

**Success criteria:** What "done" looks like — observable behavior, not test code.
Describe what the reader should be able to see, run, or measure. Be concrete.
```

**Success criteria** should be written at the level of human observation, not code assertions. Examples of good criteria:
- *"Patchified output has shape `(B, N, D)` and reconstructing it exactly recovers the original image."*
- *"Loss decreases monotonically when training on a single batch for 50 steps."*
- *"Sampling with `guidance_scale=3.0` produces images that more clearly satisfy the caption than `guidance_scale=1.0`."*
- *"Varying the action sequence while holding the first frame fixed produces visibly different continuations."*

### Granularity
A milestone should be completable in one focused session (roughly 1–3 hours). If it feels too large, split it. If two milestones are always implemented together, merge them.

**Signs a milestone is too large:**
- The success criteria describes more than one independently verifiable thing
- "What to build" needs more than one sentence
- It's hard to check progress without completing the whole thing

---

## 7. Tone and style

- **First person plural ("we")** — the author and reader are building together. *"We patchify the input into..."* not *"The input is patchified into..."*
- **Active voice** — *"we compute the loss"* not *"the loss is computed"*
- **State confusions and tricky parts directly** — *"This is counterintuitive: we parameterize the velocity field, not the noise."*
- **Never apologize for simplifications** — *"For simplicity, we use..."* is fine. *"Unfortunately we have to oversimplify here..."* implies the reader is being shortchanged.
- **Callouts for important asides** — use `> **Note:**`, `> **Watch out:**`, `> **Alternative:**` as block-quote prefixes. Keep each to 2–3 sentences.
- **Concrete before abstract** — always introduce a specific example before the general case.
- **Equations have prose** — every equation block should be preceded by a sentence explaining in words what it computes, and followed by a sentence explaining what it means.

---

## 8. File naming and placement

```
docs/walkthroughs/
├── a1_pixel_dit.md
├── a2_latent_dit.md
├── a3_text_to_video.md
└── a4_action_conditioned.md
```

File names use the track identifier prefix (`a1_`, `a2_`, etc.) followed by a short slug. No spaces.

Each walkthrough is a single Markdown file. External assets go in `docs/walkthroughs/assets/`:

```
docs/walkthroughs/assets/
├── vizlib/          — shared JS/CSS design system (color palette, node styles, animations)
├── a1/              — interactive HTML diagrams for track A1
├── a2/
├── a3/
└── a4/
```

Interactive diagrams are self-contained HTML files (all JS/CSS inlined or loaded from `vizlib/`) so they render correctly when the Markdown is viewed on GitHub or served locally.

---

## 9. Pre-writing checklist

Before drafting a walkthrough, answer these questions. The answers go into the walkthrough, not this file.

- [ ] What does the reader have at the *start* of this track? (prior milestones, trained models, data)
- [ ] What does the reader have at the *end*? (what runs, what is tested, what can be visualized)
- [ ] What is the single most important concept introduced in this track?
- [ ] What is the single most common mistake when implementing it?
- [ ] What does "it's working" look like before the final property test? (what can you see or measure informally)
- [ ] What are the 5–8 milestones, in order? For each: one sentence on what to build, one sentence on how to verify it.
- [ ] Which reference paper or repo section is closest to what we're building?
