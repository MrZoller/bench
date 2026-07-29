# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[semver](https://semver.org/).

## [Unreleased]

The first post-release bug sweep: six issues from an end-to-end review, all closed. Not one was a
wrong sum — a sweep of the whole catalog cross-product had already found zero non-finite,
negative or self-contradictory outputs, and a second found zero disagreements across thirteen
cross-surface invariants. Heavily-reviewed code pushes its bugs to the edges, and that is where
these were: in the curated data, in the periphery nothing exercises, in a latent sibling of a fixed
bug, and on the one accessibility axis with no spec behind it.

### Added

- **The model catalog covers the field again: 17 rows to 35, and a mechanism so it stays that way**
  ([#77](https://github.com/MrZoller/bench/issues/77)). `SEEDS` was hand-reviewed once, when the
  catalog was built, and never again — so the weekly refresh kept every _figure_ seven days old while
  the _list_ fell a year behind, and its newest entry was GLM-4.5-Air. Added: the top of the range
  (Kimi K2 at 1.03T, the first 1T-class row), the 480B class a 512 GB Mac is bought for (Qwen3 Coder
  480B-A35B), MLA at sizes people own hardware for (GLM 4.7 Flash at 30B-A3B, Mistral Small 4 119B,
  where the family was previously 671B models only), six publishers with no row at all (Microsoft,
  IBM, Cohere, ByteDance, MiniMax, Moonshot), the current head of six families whose stale sibling was
  the only one a user could pick (Llama 3.3 70B, Qwen3 4B/30B-A3B/235B-A22B 2507, DeepSeek V3.1, GLM
  4.7), and four rows under 4.5B where the small end had exactly one. And because no amount of
  re-deriving figures can notice a model that was never listed, every refresh now ends by asking the
  hub what the field is downloading and printing whatever the seed list neither carries nor has
  written down a reason for.
- **`NOT_SEEDED`: why the most-downloaded model in the world is not in the catalog**
  ([#77](https://github.com/MrZoller/bench/issues/77)). Nine families are refused rather than
  catalogued wrong, and until now that decision lived in a session transcript. It is data now — repo
  id to reason, checked against the live `config.json` — which is both the written record and what
  keeps the weekly candidate report from naming the same nine every Monday. The headline is that
  hybrid linear attention is no longer an exotic corner: the entire current Qwen generation is 8 to 16
  attending layers out of 32 to 64, so the most-downloaded current models on the hub cannot be priced
  until the third `AttentionCore` kind from #76 exists.

- **The hardware catalog covers the machines the audience owns: 25 rows to 43**
  ([#78](https://github.com/MrZoller/bench/issues/78)). The accuracy work had been thorough and the
  coverage had not. The cheapest catalogued GPU was the RTX 5080 at $999, AMD appeared only as
  datacenter Instinct parts, Intel not at all, and every Apple row was a maxed configuration — so most
  "will it run" questions had no row to ask about rather than an incomplete answer. Added: the
  sub-$1000 NVIDIA tier (5070 Ti, 5070, 5060 Ti 16GB, 4060 Ti 16GB, 3060 12GB), consumer Radeon
  (9070 XT, 7900 XTX, 7900 XT), Intel as a vendor for the first time (Arc B580, Arc Pro B60,
  A770 16GB, and Xeon 6980P at 844.8 GB/s, which stops `cpu-ram` reading as an AMD-only technique),
  five Apple rows from a 16 GiB MacBook Air up to the 192 GiB M2 Ultra Studio, and the MI325X. Every
  row carries its own source, and each vendor's compute derivation — Intel's INT8-to-FP16 halving,
  AMD's matrix rate against the vector one published beside it, NVIDIA's single sparse "AI TOPS"
  headline over 8 on Blackwell and 4 on Ada — is written down next to the rows that use it, with the
  headline figure each worked example starts from.

### Fixed

- **Three architectures the generator would have catalogued at 2x, 13x and a constant**
  ([#77](https://github.com/MrZoller/bench/issues/77)). Each reads as an ordinary config by every
  guard #76 added, and each is a current, high-traffic model. Gemma 4 declares `attention_k_eq_v` —
  keys and values are one tensor, so the `2 *` in the GQA term is exactly twice what the stack holds —
  and gives its global layers a second KV shape (4 heads x 512 against the windowed layers' 16 x 256),
  while the E-series shares the cache of 18 of its 42 layers. NVIDIA's Nemotron Super states its
  layers one at a time in `block_configs`, 31 of 80 with no attention at all, _and_
  `num_key_value_heads: null` because the grouping is per block — together 2560 KiB/token against 196,
  which is 320 GiB of imaginary cache at 128K. And DeepSeek V4 carries the same sparse-attention
  indexer V3.2-Exp is refused for, on a config with no `kv_lora_rank`, so it took the GQA branch where
  that guard was never asked — deriving a 128-token trailing window on every layer of a
  million-token-context model, a cache that does not grow at all. All three are refused with the
  evidence; the indexer guard now runs in front of both branches rather than inside one.
- **A safetensors index stored in LFS read as three lines of pointer text**
  ([#77](https://github.com/MrZoller/bench/issues/77)). `fetchTensorMap` used `/raw/`, which answers
  200 with `version https://git-lfs.github.com/spec/v1` for anything large — and a trillion-parameter
  model's index is large: Kimi K2's is 12 MB of 105,000 tensors. `JSON.parse` then failed on the
  letter `v`, indistinguishably from a corrupt repo, for precisely the models whose size is the whole
  question. It reads through `/resolve/` now, as the shard reads always did.
- **A dense model with zeroed MoE keys derived a NaN active-parameter count**
  ([#77](https://github.com/MrZoller/bench/issues/77)). `granite-4.0-micro` states
  `num_local_experts: 0` and `num_experts_per_tok: 0`, which satisfies the partial-config guard and
  then computes `(0 / 0) * 0`. `JSON.stringify` writes that into the committed catalog as `null`, and
  the loader checks `activeDenseParams` rather than `activeParams`, so it would have shipped. Zero
  experts is not an ambiguity — it is a dense model saying so.

- **A device id named a product that does not exist**
  ([#78](https://github.com/MrZoller/bench/issues/78)). `rtx-a6000-ada` fused the Ampere RTX A6000
  with the Ada RTX 6000 Ada Generation, while every spec on the row was the Ada card's. It is
  `rtx-6000-ada` now, and the old id is kept as an alias because `url.ts` puts `deviceId` into every
  shared scenario link: without one the old link still opens and silently shows the default machine's
  numbers under the sender's URL.
- **Below 32 GiB, an Apple row would have promised a ceiling under its own default**
  ([#78](https://github.com/MrZoller/bench/issues/78)). The `max(8 GiB, 1/16 of RAM)` reserve from
  #53 and Metal's recommended working set cross over there — on a 24 GiB machine both land on 16 GiB —
  and all six rows that existed sat far above the crossover, so the rule looked universal. Rows at or
  under it state no raiseable ceiling rather than one that promises less than the machine already
  gives, and both rules are now swept across every Apple row rather than spot-checked.

- **Strix Halo was charged the bandwidth gap twice**
  ([#51](https://github.com/MrZoller/bench/issues/51)). It was the only device carrying a
  _measured_ bandwidth — 213 GB/s against AMD's 256 rating — and the engine then applied
  `bandwidthEfficiency` and `CLASS_BANDWIDTH_UTILIZATION` on top, the constants that exist to model
  that very gap. Every Strix Halo throughput figure read 16.8% low against the treatment the other
  24 devices get — removing the double discount raises each one by 20.2% — on the surface whose
  whole purpose is ranking hardware against hardware. The catalog is theoretical peak throughout now, as the roadmap says it should be, and
  the measured field is deleted rather than deprecated — a field is an invitation, and its docblock
  was accepting.
- **Every Apple row claimed 100% of its RAM could be wired to the GPU**
  ([#53](https://github.com/MrZoller/bench/issues/53)). All six were tunable with no stated
  maximum, so the app offered the owner of a 96 GiB Mac Studio a 95.5 GiB configuration.
  `iogpu.wired_limit_mb` will _accept_ that value; what loads is bounded by what macOS needs to
  keep running, and the distance between the two is the whole subject of the field. Each row now
  reserves `max(8 GiB, 1/16 of RAM)` with the reason written down, and the catalog refuses a
  raiseable ceiling that does not say how far it raises.
- **The weekly refresh bot could open a pull request that stated no change at all**
  ([#54](https://github.com/MrZoller/bench/issues/54)). `changed` was decided by whole-document
  string equality while the summary was built from set differences, so any reordering satisfied one
  with nothing for the other to say — and that summary is interpolated into both the commit message
  and the pull request body. Both answers come from the same evidence now, and three further routes
  to a misleading summary closed with it.
- **The cached prefix was not clamped to each cell**
  ([#55](https://github.com/MrZoller/bench/issues/55)), where the prompt beside it was. Prefill
  charges every new token for attending over the resident prefix, so a 2K column was timed against
  a 49K session, and a prefix past a model's own limit took one Matrix cell from 16 s to 273 s.
  Latent rather than live — nothing in the app supplies a prefix to either grid — but both are
  exported engine API.
- **`toDevice` cast three fields into narrow types without checking any of them**
  ([#56](https://github.com/MrZoller/bench/issues/56)), on the hand-edited catalog of the two,
  while `toModel` below it validated the generated one. No committed row was ever wrong — but
  injecting the typos showed what one would buy: a misspelled `class` makes a device report as
  driven by nothing at all, and a misspelled compute dtype makes the RTX 5090 report a time to
  first token of `Infinity`.

### Accessibility

- **The Matrix is one tab stop instead of 408**
  ([#52](https://github.com/MrZoller/bench/issues/52)). Every cell is a button with a full-sentence
  accessible name, and the grid sits above the Usage controls — so 422 presses of Tab stood between
  the top of the page and the context slider that drives every figure on it, and a screen-reader
  user heard 408 sentences on the way. The ARIA grid pattern applies now: one tab stop, arrows to
  move between cells, Home and End for the ends of a row, Control with either for the ends of the
  grid. **422 becomes 15.**

  This was the one accessibility axis with no spec behind it, which is why a gap here outlived the
  four that have one: touch targets, reflow, pointer queries and contrast all have tokens and
  tests, and nothing was looking at focus order.

## [0.1.0] — 2026-07-28

First public release. Live at <https://mrzoller.github.io/bench/>.

The repository went public to publish: GitHub Pages is not available on a private repo without a
paid plan. That also restored the branch ruleset, which had been convention rather than enforcement
for the whole build.

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

[unreleased]: https://github.com/MrZoller/bench/compare/4de0fa6...HEAD
[0.1.0]: https://github.com/MrZoller/bench/commits/main
