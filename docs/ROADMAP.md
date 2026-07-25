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
   agent, fine for overnight batch" is what people want. Built: seven workload archetypes, each
   graded at the prompt it really sends. The hard part turned out not to be the thresholds but
   making every verdict state the bar it missed — see **Verdicts**, below.

## Status

Seven of eight phases are on `main` as of 25 July 2026. What remains is the weekly refresh job,
deployment, and the follow-up list in [issues #12–#20](https://github.com/MrZoller/bench/issues).

| Phase                              | State             | Notes                                                                                              |
| ---------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------- |
| 1. Scaffold                        | **done**          | React 19 + TS strict + Vite + Tailwind v4 + Zustand. CI: lint → format:check → test → build        |
| 2. Engine                          | **done**          | `src/engine/`, pure, no React. Pinned to published measurements at both ends of the hardware range |
| 3. Catalogs                        | **done** (#1)     | 17 models derived from HF, 25 devices curated. `npm run catalog` regenerates it.                   |
| 4. Design tokens + the Bench       | **done** (#5)     | Hero surface. Load the `dataviz` skill before any chart/meter/palette code                         |
| 5. Verdict + explain layers        | **done** (#4)     | Seven workload archetypes. See **Verdicts**, below                                                 |
| 6. Envelope + Matrix surfaces      | **done** (#7, #8) | Context × concurrency feasibility field; model × device heatmap                                    |
| 7. URL state, responsive, a11y     | **mostly** (#6)   | Querystring round-trips a scenario. No browser-level test pass — #19; URL defects in #15, #16      |
| 8. Weekly catalog refresh + deploy | **next**          | Scheduled `build-catalog` → PR on diff; static deploy to a zoller.ai subdomain                     |

**Correctness debt is tracked as issues, not here.** Ten are open. #9 and #10, which graded a
configuration as working when it is not, are fixed — together with #11, which printed a figure
measured at a different scenario from the one its sentence described. Filed as three bugs, one
class; written up under **Verdicts** below. What remains is labelling (#13, #20), UI state (#12,
#15, #16), test and performance debt (#17, #19), the MLX question (#18), and two engine bugs: the
layer-split spill fraction (#14), and prefill having no notion of a cached prefix (#23).

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
- **A PR merges on green CI, with outstanding review findings triaged, replied to, and filed as
  issues — not fixed first.** Set on 25 July 2026, after 74 resolved findings and no end in sight;
  the stack finished on 97. Every push drew a fresh review, including on the fixes from minutes
  earlier, so under "merge only when the reviewer comes back clean" there is no reachable fixed
  point. This is not a decision to ignore reviews: fix root causes, then merge, then file the rest.
  See #9–#20 for what that produced in practice. **This overrides the global "a PR is unfinished
  until the latest review is clean" rule, for this repo.**
- **Expect a review to name a subset of a class.** Three times now the finding named N instances
  and the same defect was live in N+2 places — a missing-cause audit raised for three verdict tiers
  was also true of chat and rag; an Envelope fix had a mirror-image omission one branch over; and
  the long-context "grade the job you admit" fix was separately true of serving, agent and rag
  (#9–#11), filed as three unrelated bugs and in fact one. Fixing only what is named is the most
  common way a round here fails to converge.
- **The seven archetypes are not a ladder, and completion may outrank chat.** The ordering is real
  but it is _only_ about latency budgets, and it does not survive contact with capacity. Completion
  sends 512 tokens where chat sends 1,024, so at 128 users on a small card the chat cache spills
  while completion's stays resident — serving 128 autocompletes genuinely is easier than 128
  conversations. Capping
  completion at chat's grade would restore the appearance of a ladder by reporting a failure that
  is not happening. The claim was dropped instead; the thresholds stay independent.

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
- **KV shards only as far as the model allows, and `placement` and `speed` must agree on how
  far.** Weights divide to any degree; GQA divides by attention head, so 4 KV heads on 8 cards
  replicate across each pair and per-card KV is a quarter, not an eighth. MLA has no head axis at
  all — vLLM keeps the whole latent on every rank. `kvShards()` is the one answer and both modules
  call it; when only `placement` knew, the memory panel said each card held the entire DeepSeek
  cache while the speed panel priced one eighth of it.
- **A layer split is not a speedup, and a layer count is not a KV divisor.** llama.cpp's default
  multi-device layout runs whole layers in sequence for one token, so a single stream sees one
  card's bandwidth and one card's FLOPS however many cards there are — that rig buys capacity,
  not speed, and modelling it as aggregate credited eight cards with ~4.9x. And on a hybrid model
  the layers are not interchangeable: Gemma's full-attention layers cache ~128x what its sliding
  ones do at 128K, so the busiest card is found by _sizing_ an assignment, not by dividing.
- **Offloaded weights read at the slower of host RAM and the bus to the host** —
  `min(hostBandwidth, device.hostLinkBytesPerSec)`. `interconnect` is the _device-to-device_ link
  `tpEfficiency` models and is not this: an H100 SXM talks to its neighbours over NVLink and to
  the host over PCIe 5.0. Modelling only host RAM made every spilled configuration on a PCIe 4.0
  card 2.5× too fast, on both decode and TTFT.
- **Prefill scales with concurrency; decode amortises it.** `estimateDecode` batched from the
  start and `estimatePrefill` ignored `usage.concurrency` entirely, so a 32-user configuration was
  graded on one user's time-to-first-token. The asymmetry is the whole point and is easy to get
  backwards: decode is memory-bound, so the weights are read once per step however many users are
  waiting and the tenth is nearly free. Prefill is compute-bound and one long prompt already
  saturates the units — serving `n` prompts is `n` times the arithmetic and the scheduler only
  chooses who waits for it. Two consequences worth stating: attention is evaluated at one
  sequence's length and _then_ scaled (sixteen users sending 2K each is sixteen quadratics over 2K,
  not one over 32K), and the offload streaming term is charged **once** — sized at the batch-wide
  expert union, but not multiplied by it, because the batch shares the weights it pulls across the
  bus. `prefillTokensPerSec` is machine-wide as a result, which is what keeps the published
  single-prompt anchors comparable with a concurrent estimate. Every sentence quoting it therefore
  has to say whose rate it is: the RAG verdict divides the batch back out, because the wait printed
  beside it is one document's; Telemetry and batch label theirs as aggregate instead. (#11.)
- **A rule the UI enforces is not a rule the engine has.** `quantApplies` kept unloadable
  model/runtime pairings out of the picker, so the app looked correct while `planPlacement`
  returned capacity and throughput for checkpoints that cannot be opened — AWQ under llama.cpp, a
  GGUF K-quant under vLLM. Every caller reaching the engine directly (Matrix, Envelope, anything
  importing `evaluate`) walked past it. The same gap produced the Matrix's P1: its quant
  substitution asked `quantApplies` without the runtime _and_ fell back to a hardcoded Q4_K_M, so
  under vLLM every dense row that fell back was scored at a format vLLM cannot read. Validate at
  the boundary.
- **Sharding needs a link, and refusing a rig means refusing its arithmetic too.** `canShard` keys
  on `interconnect`, not device class — a DGX Spark is `unified-soc` with a real ConnectX, a Mac
  Studio is the same class with nothing between chassis. Every divisor and multiplier that read
  `rig.count` now goes through `effectiveDeviceCount`, because the first attempt at this set the
  `unsupported` message and left the split running: eight Mac Studios were still reported as
  holding an eighth of the model each, and `achievedBandwidth` still summed eight cards over an
  interconnect that does not exist. A refusal that returns arithmetic for the impossible
  configuration is not a refusal.
- **Prefill attention is causal and respects sliding windows.** These are decoder-only models, so
  a full-attention layer computes `N * (N + 1) / 2` query-key pairs, not `N^2` — charging the
  square nearly doubles the attention term at long prompts and moves the point where the tile
  claims attention dominates. Sliding layers are causal too: a triangle while the window fills,
  then a band. The two corrections compound to about 3.7x on gpt-oss at a 16K prompt.

  Correcting this moved the DGX Spark prefill anchor from ~10% over to ~19% over. Per the rule
  below, the constants were **not** retuned to pull it back — a roofline that matches an anchor
  because it was fitted to it has stopped being evidence of anything.

- The two calibration anchors are **DGX Spark on gpt-oss-20b** (2,053 tok/s prefill, 49.7 tok/s
  decode) and **EPYC 9654 on DeepSeek-671B Q8** (~6 tok/s). They pin opposite ends of the roofline;
  a model calibrated only for discrete GPUs fails one of them.
- **Do not retune the constants to re-centre an anchor after fixing a bug.** Correcting the
  per-token basis moved Spark decode from ~10% under to ~19% over and Spark prefill from ~6% under
  to ~10% over, while EPYC stayed within 1% — proof the old fit was partly absorbing those errors.
  The knobs were left alone deliberately. Re-centring right after removing what a fudge factor was
  masking is how the next error gets hidden. All three sit inside the ±30% band the tests assert.

**Verdicts**

- **Grade a tier on the measurement its own sentence quotes, and on the scenario it recommends.**
  The long-context tight tier admits a machine holding 64K and was timing it on the archetype's full
  128K request — a prompt that rig has nowhere to put, and prefill is quadratic, so the impossible
  request routinely failed the tier that had just admitted it on capacity. The half-fix was worse
  than the bug: pointing the _reasons_ at the window the machine holds while the _predicate_ still
  read the reduced job meant a rig holding 160K was graded on its 64K measurement while the sentence
  beside it reported 1046s against a 600s bar. One value in both.

  That was one instance of a class, and three more were live in the same file (#9, #10, #11 — fixed
  together, since fixing them apart is how the first one took two attempts):

  - **Serving had no latency term at all.** Capacity and decode were the whole grade, so a
    deployment where every user waits minutes for a first token read as healthy: Llama 3.1 8B
    Q4_K_M on an EPYC 9654 at four users fits, decodes ~40 tok/s each, and takes ~165s to read the
    four 2K prompts. The gap _grew_ when `estimatePrefill` learned about concurrency — the estimate
    became right and nothing on this path read it. Bars are 10s and 30s, looser than chat's 2s and
    5s because a shared deployment queues by design.
  - **The agent tiers recommended a session and measured a turn.** Both read a rate taken at the
    archetype's 16K turn while their capacity bars endorsed the rig for a 32K or 64K session. 8B at
    BF16 on one 4090 under vLLM: 49.7 tok/s at the turn, ~8.6 once its own 64K session is resident
    and the weights spill to make room — below even the tight tier's 15. Fixed the long-context way,
    with one `agentMeasured` at whichever session the rig can hold, read by both tiers and every
    sentence. The consequence is deliberate: a rig that holds 64K has its _tight_ tier timed at 64K
    too. The reduced figure is for machines that cannot hold more, not a lenient reading for
    machines that can — and splitting it puts the grade and the sentence back on different
    measurements the moment `good` fails and `tight` holds.

    The first attempt at this reproduced the defect inside its own fix, which is worth knowing
    about: like every archetype it floors the evaluation at the configured context, so above 64K
    the tier's _bar_ and the _evaluated_ session are different numbers — and the new sentences
    printed the bar. At a 128K slider the row read "10 tok/s with a 64K session in the cache", a
    grade taken at twice the session it named, and the 64K it claimed would have been `tight`.
    Caught in review, not by the suite. If a sentence names a scenario, that name has to come from
    the same expression the estimate was called with.

    What the session buys is on the decode axis, and only there. `estimatePrefill` derives its
    linear and attention work from `promptTokens` alone, so the context reaches it through the
    placement or not at all: on the 4090 above, turn and session TTFT differ by 1.5% — the
    streaming term — and on a resident rig they are identical to the digit. The agent latency bars
    are therefore a turn's prompt pass priced on the session's placement, which is right for an
    agent whose prefix is cached and optimistic for one that resends its history. Fixing that needs
    an estimator that can attend new tokens against a cached prefix
    ([#23](https://github.com/MrZoller/bench/issues/23)) rather than anything in this file. Raised
    by Codex as P1 on PR #22; the mechanism is exactly as reported.

  - **The RAG sentence printed a machine-wide rate beside one document's time.** See the
    `prefillTokensPerSec` note under **Engine**. Two figures in one sentence have to divide into
    each other, and at eight users these were off by eight.

  The lesson generalises past this file: a predicate and its sentence are one claim, and the
  scenario is part of the measurement, not context around it.

- **A tight verdict must name the bar it missed.** Once the fail-level branches are exhausted it is
  easy to fall through to a positive fallback, which prints healthy figures beside a downgrade and
  explains nothing — "139 tok/s over 40K of context, 4.0s per turn" is three good numbers and no
  reason. The five tiers with more than one `good` bar now state whichever they miss, through one
  shared builder; five hand-written copies of that sentence is how two of them came to disagree.
  Completion's 0.4s bar is described rather than printed — at sub-second scale a limit and a near
  miss are a tenth apart and read as the same magnitude side by side. Chat's 2s and rag's 5s are
  far enough from their measurements to state plainly, and do.
- **`impossible` and `headroomBytes <= 0` are different claims.** The second means the fully
  resident placement is over budget; only the first means capacity is genuinely gone. Conflating
  them told users that one more concurrent user had "nowhere to go" when a partial offload still
  admits another, more slowly — and told a Mac to spill weights on a machine with no tier to spill
  to. `impossible` is computed once in `planPlacement`; the budget bar and Telemetry both take
  `canOffload` from the Bench, so two panels a few pixels apart cannot describe one placement
  differently — which they did.

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

Correctness follow-ups live in [issues #12–#20](https://github.com/MrZoller/bench/issues). This
section is for the questions those issues cannot settle.

- **MLX has no native quantization entries** ([#18](https://github.com/MrZoller/bench/issues/18)).
  GGUF K-quants stand in _by width_ — Q4_K_M's 4.83 bpw against MLX's ~4.5 — so every figure for an
  Apple-silicon configuration derives from a format MLX does not load. Invented data, documented as
  a modelling choice. The alternative, BF16 and INT8 only, makes Apple silicon unusable in a tool
  where it is a headline case. Now entangled with the `weightFormats` check, whose MLX list
  includes those K-quants precisely so the substitution keeps working.
- ~~Codex connector coverage is unconfirmed.~~ **Confirmed working**, and now well characterised.
  Reviews arrive roughly 40 minutes after a push, which is long enough to look like absence — don't
  conclude the connector is missing from a quiet first half-hour. Two further traps: it signals
  "no findings" with a 👍 _reaction_ rather than a comment, and that reaction survives later pushes,
  so merge-readiness needs the reaction's `created_at` to postdate the head commit. Zero unresolved
  threads right after a push usually means the review has not posted yet.
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
