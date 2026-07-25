# Roadmap

Where `bench` stands, what's left, and the decisions that would be expensive to re-derive.

This file exists because the working plan lived outside the repo and would have been lost.
Keep it current: it is the handoff document between sessions, and the place to look before
re-litigating a settled question.

## The thesis

Most VRAM calculators sort hardware by one number and apply one KV-cache formula to every
model. Both shortcuts break on the machines and models people actually care about. `bench`
computes rather than approximates, and reports **capacity, decode speed, and time-to-first-token
as three separate answers** rather than collapsing them into "will it fit".

Three things are the moat, in order:

1. **KV cache dispatches on attention family.** MLA (DeepSeek) caches one compressed latent per
   layer — no factor of two, no head multiplier, ~70 KB/token where the naive formula predicts
   several times that. Sliding-window layers (gpt-oss, Gemma) stop growing past their window,
   halving gpt-oss-120b's KV at 128K.
2. **The catalog is derived, never typed.** Architectures come from each repo's `config.json`,
   parameter counts from its safetensors index.
3. **The answer is a decision, not a number.** "3.2 tok/s" means nothing; "unusable for a coding
   agent, fine for overnight batch" is what people want. Not built yet — see phase 5, Verdict +
   explain layers.

## Status

| Phase                              | State                                                | Notes                                                                                                   |
| ---------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1. Scaffold                        | **done**                                             | React 19 + TS strict + Vite + Tailwind v4 + Zustand. CI: lint → format:check → test → build             |
| 2. Engine                          | **done**                                             | `src/engine/`, pure, no React. Pinned to published measurements at both ends of the hardware range      |
| 3. Catalogs                        | on [PR #1](https://github.com/MrZoller/bench/pull/1) | Complete, not yet on `main` — 17 models derived from HF, 25 devices curated. `npm run catalog` needs it |
| 4. Design tokens + the Bench       | **next**                                             | Hero surface. Load the `dataviz` skill before any chart/meter/palette code                              |
| 5. Verdict + explain layers        | pending                                              | Workload archetypes; the total-vs-active-params teaching moment                                         |
| 6. Envelope + Matrix surfaces      | pending                                              | Context × concurrency feasibility field; model × device heatmap                                         |
| 7. URL state, responsive, a11y     | pending                                              | Full config in the querystring so a link reproduces a scenario                                          |
| 8. Weekly catalog refresh + deploy | pending                                              | Scheduled `build-catalog` → PR on diff; static deploy to a zoller.ai subdomain                          |

## Decisions already made

Settled, with reasons. Reopen only with new information.

- **Hero surface is the Bench** — direct manipulation: pick a model and hardware, drag usage
  sliders, watch a stacked memory budget fill and overflow. Envelope and Matrix are secondary.
- **Build-time data, not runtime.** Keeps the site static and offline-safe; freshness comes from
  a scheduled regeneration job, not a fetch on page load.
- **Visual identity** is a dark instrument-panel chassis — a sibling to `~/code/wavefront`, not a
  clone. Different accent hue. Tokens in `src/design/tokens.ts`, mirrored as CSS custom properties.
- **Multi-GPU is modelled as a homogeneous rig** (`{device, count}`) with tensor-parallel
  sharding and an interconnect penalty. Heterogeneous mixes are out of scope.
- **Pricing is out of scope for v1.** Cloud $/Mtok versus local amortised cost is a different
  tool wearing the same chassis.

## Things that took real work to get right

Each of these was wrong first, and each is silent when it breaks. Do not "simplify" them without
reading the test that guards them.

**Engine**

- **Active weight bytes must use the dense/expert split**, not the model's blended bits-per-weight.
  gpt-oss active params are ~half BF16 dense tensors read in full every step; charging them the
  4.47 bpw blended rate understates bytes-per-token ~2×, straight onto decode throughput.
- **`bandwidthEfficiency` and `CLASS_BANDWIDTH_UTILIZATION` are two knobs fitted to two data
  points, and only their product is observable.** The split between "what the runtime achieves"
  and "what the memory subsystem allows" is _not identifiable_ from current data — it is a
  defensible physical story, not a measurement. It becomes testable when a second CPU-capable
  runtime or CPU device is added. Re-derive then; don't assume.
- **Compute precision comes from the quant's `computeDtype` gated on the runtime's
  `nativeLowPrecision`**, never from storage bit width. llama.cpp dequantizes every GGUF to fp16,
  so it cannot reach a Blackwell card's FP4 rate. Inferring from `bpw` overstated prefill 8×.
- **Prefill attention respects sliding windows** too, not just KV. Treating every gpt-oss layer as
  full attention overstates the attention term ~2× at long prompts.
- The two calibration anchors are **DGX Spark on gpt-oss-20b** (2,053 tok/s prefill, 49.7 tok/s
  decode) and **EPYC 9654 on DeepSeek-671B Q8** (~6 tok/s). They pin opposite ends of the roofline;
  a model calibrated only for discrete GPUs fails one of them.
- **Do not retune the constants to re-centre an anchor after fixing a bug.** Correcting the
  per-token basis moved Spark decode from ~10% under to ~19% over and Spark prefill from ~6% under
  to ~10% over, while EPYC stayed within 1% — proof the old fit was partly absorbing those errors.
  The knobs were left alone deliberately. Re-centring right after removing what a fudge factor was
  masking is how the next error gets hidden. All three sit inside the ±30% band the tests assert.

**Catalog**

- **`safetensors.total` counts tensor _elements_, not parameters.** True for FP8, false for MXFP4:
  gpt-oss-120b's `U8` count is exactly 33/32 of logical expert params, the extra being one scale
  byte per 32-value block. The ratio guard is tight to 0.5% on purpose — a loose band would also
  admit a uniformly-int8 model and silently discard its entire dense half.
- **Multi-Token Prediction modules inflate reported totals** (DeepSeek V3/R1 by ~13B, GLM-4.5-Air
  by ~4B) and inference never loads them. Detected via `num_nextn_predict_layers` and _refused_,
  not estimated; the seed list carries the published figure with a written reason.
- **`activeParams` excludes the input embedding unconditionally, and that is the _published_
  convention, not the physical one.** It is what reconciles every derived figure with its
  vendor's, and it is the wrong basis for decode. The engine reads `activeDenseParams`:
  - the embedding is subtracted only when **untied**. A tied table _is_ the output projection —
    a full vocab matmul every step — so subtracting it understates Gemma 3 12B by 5%.
  - **tied-ness comes from the absence of an `lm_head.weight` tensor**, never from
    `config.tie_word_embeddings`. That key is undefined on both Gemma 3 repos despite them being
    tied; trusting it drops a 1.0B table decode does read.
  - **non-text towers** (Gemma 3's vision encoder, ~0.42B) stay in `totalParams` and are excluded
    per token. The tensor classifier tests non-language prefixes _first_, against the name with a
    leading `model.` stripped — newer transformers exports nest the tower as
    `model.vision_tower.*`, which the `model.` language prefix would otherwise swallow silently.
- **Prefill additionally excludes the output projection.** Logits are computed for the positions
  that need them — one — not every prompt token. Charging it per token overstated gpt-oss-20b
  prefill 16%.
- **MoE layer selection has two conventions** and transformers implements each with a specific
  phase: DeepSeek `i >= first_k_dense_replace && i % moe_layer_freq == 0`, Qwen
  `(i + 1) % decoder_sparse_step == 0`. Conflating them overcounts by a whole layer whenever the
  layer count isn't a multiple of the step.
- **Gated repos** (`meta-llama/*`, `google/gemma-*`) return 401 unauthenticated, so those seeds
  point at open mirrors. `HF_TOKEN` allows the originals.

## Open questions

- ~~Codex connector coverage is unconfirmed.~~ **Confirmed working.** Both PRs were reviewed by
  `chatgpt-codex-connector`; PR #1 drew four P2 findings, all valid, all fixed in `6ffa766`. The
  reviews arrived roughly 40 minutes after the pushes, which is long enough to look like absence —
  don't conclude the connector is missing from a quiet first half-hour.
- **`main` is unprotected.** Rulesets need GitHub Pro on a private repo, so the "PRs only, all
  threads resolved" rule is convention rather than enforcement. Re-run the ruleset POST if the
  repo goes public.
- **Device specs need a verification pass before publishing.** Bandwidth is the number that
  governs everything and the one vendors bury. The `rumored` row (M5 Ultra) is press-rumour grade
  and must stay visibly labelled in the UI.
- **Final subdomain** on zoller.ai.

## Verification

```
npm test && npm run lint && npm run format:check && npm run build
npm run catalog -- --dry-run    # re-derive the model catalog without writing
```

The engine's reference tests are the spec. If one fails, the model is wrong — do not widen the
band to make it pass.
