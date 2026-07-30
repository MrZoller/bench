# bench

Work out which open-weight LLMs run on your hardware, and how comfortably — across discrete
GPUs, unified-memory machines, and CPU+RAM.

**Read [docs/ROADMAP.md](docs/ROADMAP.md) first.** It carries the current phase, the settled
decisions, and the handful of derivations that were wrong on the first attempt and are silent
when they break. It is the handoff document between sessions.

## Commands

- Test: `npm test` (`test:watch`, `test:coverage` also exist)
- Browser tests: `npm run test:e2e` (Playwright; builds and serves automatically)
- Run: `npm run dev`
- Lint/format: `npm run lint` / `npm run format` (`format:check` is the CI gate)
- Build: `npm run build`

CI runs **lint → format:check → test → build → test:e2e**; run them before pushing.

## Stack

- React 19 + TypeScript (strict) + Vite + Tailwind v4 + Zustand + Canvas 2D, Vitest, Playwright.
  Imports use the `@/` alias for `src/`.

## Architecture

- `src/engine/` — pure math, **no React imports**. Memory footprint, placement, roofline
  throughput. This is the product's whole value; everything else renders it.
- `src/data/` — `quants.ts` (curated), `devices.json` (curated), `models.generated.json` (built
  from Hugging Face by `scripts/build-catalog.ts`).
- `src/components/` — the Bench (hero), Envelope, Matrix.

## Conventions

- Conventional commits: `type(scope): summary`
- **`e2e/` is for what jsdom structurally cannot answer**, not for a second copy of the unit suite.
  Layout, scrolling, `@media (pointer: coarse)`, and canvas actually painting. Everything else stays
  in Vitest, where it runs in a second. The rule exists because the gap already shipped a bug: a
  scroll anchor on a `display: contents` element generates no principal box, so `scrollIntoView`
  returned early in every real browser while jsdom — which has no `scrollIntoView` at all — passed it.
- **The engine's reference tests are the spec, not scaffolding.** They assert against llama.cpp's
  published file sizes, DeepSeek's stated KV footprint, and measured throughput on a DGX Spark
  and an EPYC 9654. If one fails, the model is wrong — do not widen the band to make it pass.
- **Never hardcode a model list from memory.** The landscape moves faster than any training
  cutoff. Architectures come from each repo's `config.json`, parameter counts from its
  safetensors index. Unknown architecture is a loud failure, never a guess.
- Every `devices.json` row carries a `source` URL and a `status` (`shipping`/`announced`/
  `rumored`). Pre-release specs must stay visibly labelled in the UI.
- **`devices.json` row order is display order.** Nothing sorts it: both the Hardware picker and the
  Matrix render the file as written. Rows are grouped by `class` — `discrete-gpu`, `unified-soc`,
  `cpu-ram` — a vendor's rows are contiguous within a class, and a vendor's product lines are
  contiguous within that, newest generation first. `catalog.test.ts` enforces the first two;
  `$comment-order` in the file states the rest and the blank lines are its record.
- Load the `dataviz` skill before writing any chart, meter, heatmap, or palette code.
