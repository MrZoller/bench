# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[semver](https://semver.org/).

## [Unreleased]

Everything below is on `main` and has never been published — the deploy workflow is written and
waits on GitHub Pages being enabled for the repository ([#40](https://github.com/MrZoller/bench/issues/40)).
This section becomes 0.1.0 on the first deploy.

### Added

- **An engine that computes rather than approximates.** `src/engine/` is pure — no React — so it
  can be pinned to published reference values, and it is: llama.cpp's published 4.58 GiB for
  Llama 3.1 8B at Q4_K_M, DeepSeek's stated KV footprint, and measured throughput on a DGX Spark
  and an EPYC 9654 at opposite ends of the hardware range. Capacity, decode speed and
  time-to-first-token are reported as three separate answers rather than collapsed into "will it
  fit".
- **KV cache that dispatches on attention family.** MLA caches one compressed latent per layer, so
  DeepSeek-family models cost a fraction of what the naive `2 × layers × kv_heads × head_dim`
  formula predicts; sliding-window layers stop growing past their window, roughly halving
  gpt-oss-120b's cache at 128K.
- **A derived catalog.** 17 models built from each repo's own `config.json` and safetensors index
  by `npm run catalog`, plus 25 curated devices. Parameter counts are summed from the index rather
  than recalled — except on the three models whose index includes a Multi-Token Prediction module
  inference never loads, which carry the vendor's published total and a written reason. The
  generator refuses to write a partial catalog rather than silently shrinking the product.
- **Four surfaces, one page.** The Bench (pick a model and hardware, drag usage, watch the memory
  budget fill and overflow), the verdict layer under it grading seven workload archetypes at the
  prompt each really sends, the Envelope (context × concurrency as a feasibility region), and the
  Matrix (every model against every shipping device).
- **A scenario in the URL**, so a configuration can be shared as a link.
- **A weekly catalog refresh** that regenerates from Hugging Face, verifies, and opens a pull
  request only when a figure actually moved — not when the timestamp did.

### Notes on accuracy

- **Every cache width is derived from the runtime's own source**, never from its name. llama.cpp's
  `q8_0` carries a 2-byte scale per 32-element block (34/32 bytes per element); MLX's carries an
  fp16 scale _and_ bias per group of 64, so 8.5 bits (17/16). Nominal widths would understate both,
  in the direction that reports a long-context configuration fitting when it does not.
- Every figure derived from a **stand-in weight format is marked on screen**. MLX has no catalogued
  native quantization, so another format of the same nominal width stands in
  ([#18](https://github.com/MrZoller/bench/issues/18)) — a documented approximation rather than an
  invisible one. The alternative was restricting Apple silicon to BF16, which makes a headline case
  unusable. The same marker covers any cache precision added without an established width; none
  currently needs it.
- **Pre-release hardware is labelled.** The one rumoured device row — the M5 Ultra Mac Studio — is
  press-rumour grade and says so, and the Matrix leaves it out entirely.
- Throughput is a roofline calibrated to two anchors and asserted within ±30%. It is a band, not a
  promise.

### Accessibility

- Reflow at a 320px viewport, and text scaled to **200%** without loss of content — tested in a
  real browser at four widths and under a font wider than any the app will resolve, because the
  first version of that suite passed on macOS and failed on CI over font metrics alone. The claim
  is 200%, where 1.4.4 stops; past roughly 250% the layout does break, and that is recorded rather
  than fixed.
- Hit targets meet WCAG 2.5.8. On a coarse pointer the Matrix grid and the three disclosure
  toggles are held to a stricter 44px, while a mouse keeps the dense 28px grid. The toggles key on
  `any-pointer: coarse`, so a touchscreen laptop is not treated as a mouse-only device.
- The Envelope's canvas and the memory bar each carry their figures as a table as well, and every
  graded state — the workload verdicts, the Envelope's regions, the Matrix's "will not run" — is
  named in words beside its colour. The Matrix's continuous ramp is the one thing colour alone
  carries, and it varies lightness rather than hue for that reason; each cell's exact figure is in
  its accessible name.
- Reduced motion is respected in the stylesheet, and separately in the one scroll JavaScript asks
  for, which the stylesheet cannot reach.

[unreleased]: https://github.com/MrZoller/bench/commits/main
