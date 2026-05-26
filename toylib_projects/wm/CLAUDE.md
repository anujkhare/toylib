# AI Collaboration Instructions — `wm`

## Purpose

This is a **learning project**. The primary objective (stated in `docs/plan.md`) is to implement and understand every component from scratch — every loss, tokenizer, and conditioning mechanism. The goal is not to produce working code as fast as possible; it is to build genuine understanding through doing.

## What you should and should not implement

### Do NOT implement core model code

The following are off-limits for AI implementation. When asked about these areas, explain concepts, discuss design trade-offs, review code the user has written, or point to relevant papers and reference repos — but do not write the implementation.

- **Model architectures:** DiT blocks, VAE encoder/decoder, attention layers, AdaLN-zero conditioning, patchify/unpatchify
- **Training mechanics:** flow-matching / rectified-flow loss, DDPM ε-prediction, EMA weight updates, gradient clipping, mixed-precision setup
- **Samplers:** Euler sampler, DDPM sampler, CFG guidance
- **Conditioning mechanisms:** text cross-attention, per-frame action AdaLN modulation, first-frame latent concatenation
- **VAE specifics:** KL loss, reparameterization trick, latent channel design
- **Temporal attention:** temporal RoPE, factorized spatial/temporal blocks, inflation from 2D weights

If the user pastes a partial implementation and asks for a review or debug help, you may point out bugs — but do not rewrite working or placeholder code on their behalf.

### You are free to help with

- **Infrastructure and tooling:** datagen scripts, HDF5 storage, data loaders, visualization tools, CLI interfaces, shell scripts
- **Tests:** test scaffolding, assertions, fixtures for datagen and viz modules
- **Design discussions:** architecture trade-offs, explaining paper techniques, comparing options, reviewing design decisions
- **Concept explanations:** walk through how flow matching works, what AdaLN-zero does, why KL regularization is used — at whatever depth is useful
- **Debugging non-model code:** errors in datagen, viz, or infra code are fair game
- **Eval tooling:** physics-grounded metrics, position extraction, plotting, FID/FVD measurement scaffolding
- **Reading and summarizing papers or reference repos** when the user wants to understand something before implementing it

## How to handle ambiguous requests

If the user asks you to "add" or "implement" something and it is unclear whether it falls in the off-limits category, ask first. Lean toward explaining rather than implementing when in doubt.

If the user explicitly says "just write it for me this time" or "I already understand this, help me move faster here" — defer to them; they know their own learning goals. Do not enforce these rules more strictly than the user wants in the moment.
