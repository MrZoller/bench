# bench

Work out which open-weight LLMs run on your hardware, and how comfortably — across discrete
GPUs, unified-memory machines, and CPU+RAM.

## Commands

- Test: `npm test` (`test:watch`, `test:coverage` also exist)
- Run: `npm run dev`
- Lint/format: `npm run lint` / `npm run format` (`format:check` is the CI gate)
- Build: `npm run build`

CI runs **lint → format:check → test → build**; run them before pushing.

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
- **The engine's reference tests are the spec, not scaffolding.** They assert against llama.cpp's
  published file sizes, DeepSeek's stated KV footprint, and measured throughput on a DGX Spark
  and an EPYC 9654. If one fails, the model is wrong — do not widen the band to make it pass.
- **Never hardcode a model list from memory.** The landscape moves faster than any training
  cutoff. Architectures come from each repo's `config.json`, parameter counts from its
  safetensors index. Unknown architecture is a loud failure, never a guess.
- Every `devices.json` row carries a `source` URL and a `status` (`shipping`/`announced`/
  `rumored`). Pre-release specs must stay visibly labelled in the UI.
- Load the `dataviz` skill before writing any chart, meter, heatmap, or palette code.
