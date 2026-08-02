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

**All eight phases are done, and the site is live** at
<https://mrzoller.github.io/bench/> as of 28 July 2026. The repository went public to get there —
Pages is not available on a private repo without a paid plan — which also restored the branch
ruleset that had been convention rather than enforcement since the start.

What remains is a naming decision, not work: the site serves from the Pages project URL, and a
zoller.ai subdomain is one repository variable away. See **Deployment**, below.

| Phase                              | State             | Notes                                                                                                |
| ---------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------- |
| 1. Scaffold                        | **done**          | React 19 + TS strict + Vite + Tailwind v4 + Zustand. CI: lint → format:check → test → build          |
| 2. Engine                          | **done**          | `src/engine/`, pure, no React. Pinned to published measurements at both ends of the hardware range   |
| 3. Catalogs                        | **done** (#1)     | 35 models derived from HF (#77), 43 devices curated (#78). `npm run catalog` regenerates the models. |
| 4. Design tokens + the Bench       | **done** (#5)     | Hero surface. Load the `dataviz` skill before any chart/meter/palette code                           |
| 5. Verdict + explain layers        | **done** (#4)     | Seven workload archetypes. See **Verdicts**, below                                                   |
| 6. Envelope + Matrix surfaces      | **done** (#7, #8) | Context × concurrency feasibility field; model × device heatmap                                      |
| 7. URL state, responsive, a11y     | **done** (#6)     | Querystring round-trips a scenario. Browser pass in `e2e/` (#19); reflow and hit targets (#35, #29)  |
| 8. Weekly catalog refresh + deploy | **done**          | Refresh opens a PR on a _substantive_ diff. Deployed to Pages, 28 July 2026 — see below              |

**Correctness debt is tracked as issues, not here.** #9 and #10, which graded a configuration as
working when it is not, are fixed — together with #11, which printed a figure measured at a
different scenario from the one its sentence described. Filed as three bugs, one class; written up
under **Verdicts** below. Both engine bugs are fixed: the layer-split spill fraction (#14) and
prefill having no notion of a cached prefix (#23); see **Engine** below. The browser-level test gap
(#19) is closed, and with it the legend overflow (#34) that only a browser could falsify; see
**Tests** below. Reflow at 200% text (#35) and the coarse-pointer targets (#29) are fixed, both by
sweeping the class rather than the named instance. The labelling (#13) and clipboard (#15) bugs
turned out to have been fixed in passing by #25 and #26 and were closed on the evidence.

### The post-release sweep, 29–31 July 2026

Eighteen issues filed against the live site, worked as one pass and merged as eighteen pull requests.
Most are written up in the sections below where they change a derivation; what follows is what the
_shape_ of the sweep taught, because that part is not visible from any single diff.

**Every issue named a subset of its class, and the ones that hurt were the ones I swept badly.** This
file already recorded that pattern three times; the sweep added five more. The Apple GPU-bin defect
was filed against one row and was live on four. The pointer-reclaim fix on the Matrix readout landed
on the cell handler and left both heading handlers with the identical bug. A candidate-report fix
taught the failed-fetch path to admit failure and left the capped-walk path publishing an
ordinary-looking table. In each case the reviewer named instance _n_ and the class was _n + 2_, and
in each case the second round was more expensive than sweeping would have been.

**A review round costs less than it looks and converges slower than it looks.** Codex found real
defects on every round of every PR; across the sweep exactly one premise was false, and even that one
had a true half. But three PRs took four and five rounds, and rounds three onward were almost
entirely defects in the _previous round's fix_ — a gate that fixed a dead control and created a trap
door, a reservation that fixed the panel and moved the page. `ROADMAP.md`'s existing rule (merge on
green CI, triage and file the rest) is what stops that; six issues were filed under it rather than
patched, and each carries its measurement so the next session does not re-derive it.

**Catalog growth is a load-bearing input to the test suite, and nothing was watching it.** The Matrix
is models × devices, so #78 and #77 together took it from 408 cells to 1,470 and the unit suite from
42s to about fourteen minutes — 3.6× the cells for 20× the wall clock, because `userEvent` slows
superlinearly with tree size. Two separate pull requests failed CI on a per-test timeout for changes
that touched no component, and each was diagnosed from scratch. The timeout has now been raised twice
and must not be raised a third time; see #101.

**The grid is _shipping_ devices, not catalog rows, and every count of it has to say so.** The first
draft of this paragraph multiplied 35 models by all 43 rows and reported 1,505 — but `Matrix.tsx`
filters on `status === 'shipping'`, so the rumoured M5 Ultra has never been a column and the true
products are 17 × 24 = 408, 17 × 42 = 714 and 35 × 42 = 1,470. The error is invisible at a glance
because both series grow the same way, and it is the same shape as every other figure in this file
that was derived from the wrong denominator: the number came from the catalog, and what the reader
needed came from the surface.

The same three numbers were live in two other places, which is the usual shape: the issue body of
#101, corrected in a comment there, and the timeout rationale in `vite.config.ts` — the one place a
maintainer is sent to read them, and the one this file tells them not to raise again. That copy is
corrected on the #101 branch rather than here, because it sits three lines from the timeout that
change also lowers the pressure on; until that merges the repository holds both readings, and this
paragraph is the record of which one is right.

**Codex needs a nudge here, and the ROADMAP's "~40 minutes" reads as patience when it is absence.**
Every clean verdict in the sweep arrived within about two minutes of an `@codex review` comment, and
several PRs sat 30 to 60 minutes with no reaction and no verdict before being nudged. The last one
bounds it: #107 sat **27 hours** on a green, thread-free head with no verdict of any kind, then
answered a nudge in two and a half minutes. And the connector was provably healthy the whole time —
it cleared #106 three minutes _before_ that push and #104 and #105 two hours before, so the silence
was specific to one push rather than a backlog. A silent push is not a queue position, and waiting
longer does not help. Treat a quiet half-hour as a skipped push rather than a slow one — one nudge per
push, per `watch-pr`.

The corollary is a detection rule, because the skipped push is invisible from the surface a reader
naturally checks. #107 carried a `+1` reaction from the connector and nothing else: no 👀, no review,
no comment. A reaction is not a verdict and 👀 is the only one that means a run is in flight, so
"the bot has touched this PR" and "the bot has reviewed this head" are separate questions. Ask the
second one, against both the reviews API and the issue comments, and compare the SHA as a prefix —
the no-findings form abbreviates it to ten characters.

**The tooling gaps were worktree-shaped.** Running agents in git worktrees needs a symlinked
`node_modules`, and `.gitignore` listed `node_modules/` with a trailing slash — which matches a
directory and not a symlink of that name, so `git add -A` staged the link. Worse, each worktree
carries its own `tsconfig.json`, so `npm run lint` reported 936 parse errors across the real `src/`
tree and `vitest` collected 22 copies of `App.test.tsx` and reported 44 failures belonging to another
checkout. Both are fixed in `.gitignore`, `eslint.config.js` and `vite.config.ts`; the general lesson
is that a nested checkout is invisible to a human and not to a tool that walks the tree.

**Cleaning up afterwards needs GitHub, because neither git command for "is this branch merged" works
against a squash merge.** The repo squash-merges, so a merged branch's tip is not an ancestor of
`main` and `git merge-base --is-ancestor` reports every one of them unmerged. `git cherry` fails the
same way for a subtler reason, and _conditionally_, which is worse: the squash commit's patch is the
_union_ of the branch's commits, so on a branch of two or more no individual commit has a matching
patch-id and every one reports `+`. A one-commit branch is the exception — its patch and the squash's
are the same patch, so `git cherry` correctly reports `-`. That is the trap rather than the escape: a
rule verified on the one-commit case and then applied to the rest looks like it works. Both failing
answers read as "unmerged work you are about to delete", which is the answer that stops a cleanup. The
test that actually holds, on branches of any length, is matching the local tip against the merged PR's
`headRefOid` — GitHub squashed
exactly that commit, so identity there is proof and nothing local can substitute for it.

MLX's 8-bit KV cache (#33) is **derived rather than marked**: `mlx-lm`'s source states the group
size and the scale-plus-bias dtypes, so the width is 8.5 bits and the catalog says so (#38). That
forced the contract question the marker always carried (#45) — the field asks whether a width is
_established_ now, not whether it is nominal.

## What's next — the v2 candidates

Four features, filed under the
[`v2 — guided mode` milestone](https://github.com/MrZoller/bench/milestone/1). The design for each
lives in its issue — the approach, the seams it builds on, and the traps already named — because
this file records settled decisions and the issues are where a feature's design lives until it
ships. What belongs here is the direction and the sequencing, so a session starting cold finds it
without walking the issue list.

The common thread: the moat today is correctness, and correctness is invisible to a first-time
visitor — every alternative _looks_ the same at a glance. Each of these either makes the accuracy
legible or removes the expertise the current entry point assumes.

**Launch commands shipped first, as sequenced.** What it settled is below, under **Launch
commands**; the other three are still as described here.

| Feature                                                         | Issue                                                | Why it is on the list                                                                                                                                                                 |
| --------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Launch commands — emit the runnable command for the placement   | [#136](https://github.com/MrZoller/bench/issues/136) | The engine already computes the layer split; competitors never did, so they structurally cannot print one. Surface it, then format it                                                 |
| Detect — one-click "what can my machine run"                    | [#137](https://github.com/MrZoller/bench/issues/137) | The 43-row picker assumes knowledge most visitors don't have. WebGPU narrows it — sometimes to a handful, on a redacting browser only to a vendor — and a guess stays visibly a guess |
| Recommend — rank the catalog for a device + workload            | [#138](https://github.com/MrZoller/bench/issues/138) | The question people arrive with. The Matrix holds the answer as 1,470 cells; this returns the decision                                                                                |
| Calibrate — predicted-vs-measured, submitted via issue template | [#139](https://github.com/MrZoller/bench/issues/139) | Two anchors become a measured lattice, and a public "we said 44, users measured 41" record is the argument no competitor can copy                                                     |

**The sequencing is part of the record.** Launch commands first — still the smallest lift, but not
the pure formatter the first draft of this section claimed, and the two prerequisites are worth
naming so #136 is not scoped off a false premise. `planPlacement` computes the layer assignment and
then discards it: its bins keep byte loads rather than layer counts, and only the busiest survives
onto `Placement` — so the split has to be surfaced before a formatter can read it, an engine change
that records what is already computed rather than new modelling. And a command may only name a
checkpoint that exists: the catalog carries the source repo id and no per-quant artifact, so a
runnable command for a non-native selection (Q4_K_M under llama.cpp, AWQ under vLLM) needs an
artifact resolver, or the feature is restricted to selections with a known artifact — which of
those is #136's first decision. A placeholder is honest only for the local file path, which is the
user's business either way; for the artifact itself it is a research assignment wearing a command's
clothes, and naming the source checkpoint instead would be the substituted-format failure (#18) in
copy-pasteable form. Calibrate still follows launch, and launching is necessary rather than
sufficient: a server up is not a measurement, so the command family has to include the benchmark
form — the scenario's prompt length, generation length and concurrency in the flags of the selected
runtime's own client — before a submitted number describes the priced workload rather than the
tool's defaults. One client per runtime, because `llama-bench` loads GGUF and cannot measure a vLLM
or MLX placement: each runtime's family carries its own, or calibration says out loud that it
starts at llama.cpp. That form is cheap once the assignment is surfaced, and it is the piece
calibrate actually pastes back.
Detect and recommend ship together as one guided-mode project, because
each is half of the three-click landing path — detect the machine, ask what the reader wants to do,
answer. Calibrate last: it is the slow-burn moat, and it compounds from the day the other three
make submitting easy.

### Launch commands, and what building them settled

**The artifact question resolved to the restrictive option, and the rule is smaller and better than
either arm the issue offered.** #136's first decision was an artifact resolver against "restrict to
selections with a known native artifact", and the answer turned out to be a one-line predicate: the
catalog can name **exactly one checkpoint per model — its own repo, at its own checkpoint format**.
`nativeQuant` is `quantization_config.quant_method` as the generator read it, and its absence means
the repo ships unquantized, which is the `bf16` row. So `artifactFor(model, quant)` is
`(model.nativeQuant ?? 'bf16') === quant.id ? model.id : undefined`, which covers all 35 rows rather
than the 9 the issue estimated — every model is nameable at exactly one format.

Everything else is a _conversion somebody else published_: a Q4_K_M GGUF, an AWQ pack, an
`mlx-community` port, in repos this catalog has never seen. A resolver for those is a new curation
surface with a freshness problem and no derivation behind it, which is the one thing this project
does not do. So vLLM and MLX refuse for a non-native selection and say what would have to exist.
**llama.cpp is unaffected and that is not a loophole**: `-m` takes a path on the reader's own disk,
which is the one case the roadmap already called an honest placeholder, so the GGUF family emits for
every selection while naming no checkpoint at all.

An unrecognised `quant_method` makes every format on that model unnameable rather than making the
wrong one nameable, which is the direction the whole module fails in.

**Four flags were not what a from-memory implementation would have written, and each was found by
reading upstream rather than recalling it.** This is the "a command is a claim, and flags drift"
trap paying out immediately rather than in six months:

- **llama.cpp's `-ngl` counts the output tensor.** `n_gpu_layers` defaults to `n_layer_all + 1` and
  `i_gpu_start = max(n_layer_all + 1 - ngl, 0)`, so a fully-resident 48-layer model wants `-ngl 49`
  and `-ngl 48` leaves the output tensor on the host.
- **`-c` is the whole cache, divided among the slots.** `n_ctx_seq` is `n_ctx / n_seq_max` unless the
  KV buffer is unified, and passing `-np` explicitly is what turns unification off — so eight users
  at 64K is `-c 524288`. Passing the per-user figure gives each slot an eighth of it.
- **Ollama's Modelfile has no `num_gpu` parameter.** The documented list is `num_ctx`, `num_predict`
  and the sampler knobs. So the layer split this whole feature exists to print is the one thing that
  surface cannot be told, and the template says so rather than inventing a line. The issue's own
  sketch said "the `Modelfile`/`num_gpu`/`num_ctx` equivalents".
- **`mlx_lm.server` has no KV quantization flag.** `--kv-bits` is `mlx_lm.generate`'s. An
  8-bit-cache scenario therefore cannot be _served_ at the precision it was priced at, and the note
  says the served cache is fp16 and roughly twice the size shown.

**The trap #136 names most sharply was reached from the far side, by the one flag the feature
exists for.** `-ngl` was right from the start — it reads `assignment.residentLayers` and never a
fraction — and `-ts` was wrong, which nothing in the issue's framing predicts because `-ts` is not a
flag the issue mentions. It is the payoff of surfacing the assignment: llama.cpp's default split is
proportional to device _memory_, so on identical cards it is an equal number of layers, which is the
wrong split for a model whose layers cache different amounts.

The defect is that **`-ts` does not distribute the model — it distributes the `-ngl` window.**
llama.cpp puts the last `ngl` layers on GPUs and splits _those_ by these proportions, so the two
flags are read together, and the first version gave them counts from two different scopes: resident
for `-ngl`, assigned for `-ts`. llama.cpp then re-derives a per-device split that is neither.
gpt-oss-120b at Q8_0, 128K over 8 users on four 4090s packs 9,9,9,9 layers and keeps 3,3,4,4 of them
resident — so the emitted pair asked llama.cpp to spread fourteen layers _evenly_ over cards bench
had sized for three and four. That is an OOM on load, in copy-pasteable form, which is exactly what
the issue's `-ngl` trap warns about, one flag over.

Two things about how it was caught are worth keeping. **The "only when uneven" gate was wrong in the
same way and hid it**: reading assigned counts, it suppressed the flag entirely on that rig, so the
one case that mattered emitted nothing at all — equal counts of _unequal_ layers is precisely where
the split is needed. And **the test suite had a spilled case and a sharded case and never their
product**, which is where the defect lived; the issue's own verification note asks for "a spilled
one, a sharded one" and the conjunction is what was missing. Both are now one test, mutation-checked.

**And `llama-bench -d` answers #139's sharpest trap for free.** It runs the test at a stated context
depth, which is exactly the resident-prefix state `estimatePrefill` charges an agent turn against —
so the measurement form reproduces the priced workload rather than a standalone prompt. That was the
piece calibrate was said to need before it could ask anyone for numbers, and it exists.

One thing to watch that is not a bug yet: **`RuntimeSpec.preallocFraction` is 0.9 and vLLM's own
`gpu_memory_utilization` default has moved to 0.92.** The emitter states `--gpu-memory-utilization
0.9` rather than leaving it out, so the command reproduces what the panel priced — but the catalog
figure and upstream's default are no longer the same number, and whether the engine should follow is
a separate question from whether the command should state it.

Deliberately absent, and not forgotten: cloud pricing stays out of scope per the settled decision
below, and fine-tuning memory (LoRA/QLoRA) is a second engine rather than a feature — real demand,
weak incumbents, and the wrong thing to attempt before guided mode ships. It is the v3-scale bet.

## Deployment

Two workflows, and the interesting decisions are in what each refuses to do.

**`catalog-refresh.yml`** regenerates the catalog every Monday and opens a pull request rather
than pushing to `main` — a model whose KV heads changed overnight is exactly the case a human
should see, and it is indistinguishable, to the job, from Hugging Face returning plausible
nonsense. Three things about it are easy to get wrong and are already wrong once elsewhere:

- **`git diff --quiet` is the wrong question.** `build-catalog.ts` stamps `generatedAt` on every
  write, so the file differs after every run whether or not a figure moved. Wired to that, the job
  would open an empty pull request every week for the rest of the project's life — and people who
  stop reading a bot that is right one week in fifty also stop reading it the week it matters.
  `scripts/catalog-diff.ts` compares `models` and `failures` only, and is unit-tested in both
  directions.
- **No `--allow-partial` on a schedule.** The generator refuses a partial write by design; a
  scheduled job is exactly where a 503 on five of seventeen seeds would silently delete 29% of the
  product. A red run is the intended outcome of a bad fetch.
- **The whole gate runs inside the refresh job, before the PR is opened.** GitHub deliberately does
  not trigger workflows on a push made with `GITHUB_TOKEN`, so the pull request it opens gets no CI
  of its own. Verifying in the same job is what stops a new model with an attention shape the
  engine cannot price arriving in a green-looking PR.

**`deploy.yml`** publishes `dist/` to GitHub Pages on every push to `main`, and does. The first
deploy ran on 28 July 2026.

Getting there needed a decision rather than code. Pages on a private repo requires a paid plan —
`POST /repos/MrZoller/bench/pages` returned `422 Your current plan does not support GitHub Pages
for this repository`, which is a hard block and not a settings toggle. The repo went public
instead, which also lifted the ruleset limitation recorded below. The workflow itself needed no
change.

Its preflight job stays, and is still worth having: it checks whether Pages exists and skips the
deploy with a notice rather than failing red. A fork with no Pages gets a green run and an
explanation instead of a broken-looking one.

Two settings are repository variables rather than committed values, because they describe where
the site is served rather than what it is, and both fail _quietly_ when wrong:

| Variable              | Default | What it is for                                                                               |
| --------------------- | ------- | -------------------------------------------------------------------------------------------- |
| `PAGES_BASE_PATH`     | `/`     | Vite's `base`. A Pages _project_ site serves from `/bench/`; a custom domain serves from `/` |
| `PAGES_CUSTOM_DOMAIN` | unset   | Written to `dist/CNAME` each deploy, since Pages drops the domain otherwise                  |

`PAGES_BASE_PATH` is set to `/bench/` today, because a Pages _project_ site serves from the repo
name. Attaching a custom domain means setting `PAGES_CUSTOM_DOMAIN` **and** returning
`PAGES_BASE_PATH` to `/` — changing one without the other is the failure mode this pair exists to
make visible, since a wrong `base` produces a blank page with 404s in the console rather than a
build error. Verified after the first deploy: the served HTML references `/bench/assets/…` and both
assets return 200.

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
- **A qualifier attaches to a claim, not to a subject** ([#160](https://github.com/MrZoller/bench/issues/160)).
  The narrow Matrix readout shows a tapped spilled cell as "spills 66% to host RAM" and carries no
  host-RAM caveat, and it stays that way. `HOST_RAM_UNCHECKED` exists because `planPlacement` sizes a
  spill with no host-RAM input at all, so any surface saying a spilled configuration **loads** is
  promising something the engine never checked — and the test is the verb, not the topic. The wide
  form promises ("runs only by spilling…") and carries the constant; the narrow form states the spill
  and promises nothing. Both channels that do promise are width-independent — the `title` and every
  cell's `aria-label` are the same `full` string at 320px as at 1440 — and the colour is not a third
  promise either, since `measureValue` scores a spilled fit at 0 under `fit`, the dimmest step.

  Granting it was costed rather than argued: at 200% text on a 320px viewport the rem floor doubles
  while the line width halves, so the full sentence needs a 10rem reservation and even the
  inline-clause register needs 7.5rem — **120px of permanently reserved blank space on a phone
  against 80px today**, on every page view whether or not the grid holds a spilled cell. And the
  brief register does not survive the move: "if the host has room for **them**" points at the
  _weights_ the Envelope's count line names one clause earlier, and this sentence names none, so
  granting the caveat properly would need a third spelling of one fact — the drift the two-register
  pair exists to prevent.

  **The finding's true half is recorded rather than repaired**: the legend line carrying the
  panel-level caveat is non-sticky, so it is off screen at exactly the moment the sticky readout
  exists for. The answer to that is the caption and the legend at panel level plus the two per-cell
  channels above — not a reservation levied on every narrow reader for a claim the narrow line is
  not making. Reopen only with a shorter register that reads correctly beside "spills 66% to host
  RAM", since the reservation is what the wording costs.

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
- **The spill fraction is the rig's, not the busiest device's.** `layerSplitBusiest` returned one
  device's load, so `offloadFraction` was a per-device ratio — and both speed estimators multiply it
  by the _whole model's_ active weights. Under a layer split the cards hold different amounts, so
  the cards that are still resident were billed host-bus time for an overflow at the cards that are
  not. Gemma 3 12B at 128K over five cards puts two of its eight full-attention layers on each of
  three cards and one on each of the other two, and almost no weights on the heavy three — so the
  busiest card holds 4% of the model: a 0.25 GiB overflow read as 87% of every weight streaming,
  against a true 11%. Every bin's load is kept now and the bytes that actually spill are summed. The
  uniform case — tensor parallelism, one device — is unchanged by construction, since `n` identical
  overflows over `n` identical shards give back the same ratio.

  `impossible` moved with it, and dragged a sentence along: it now asks _every_ device whether cache
  and activations alone are over, because the busiest card by _combined_ load is not necessarily the
  one holding the most cache. That broke an implication two panels were relying on — it used to be
  true by construction that an impossible offloadable rig had the busiest device's cache over the
  ceiling, so BudgetBar rebuilt the figure from `kvBytesPerDevice`. On Gemma 3 12B over three 4090s
  it then read "the cache and overhead alone need 19.1 GiB" under a header reading 23.0 GiB, and
  disproved the refusal beside it. The refusing floor is carried on `Placement` as
  `floorBytesPerDevice` instead, so the predicate and its sentence read one value. (#14.)

- **A cached prefix is a declaration, not a default.** `estimatePrefill` took its linear _and_ its
  attention work from `promptTokens`, so it could only express a standalone request — a prompt
  attending over itself. Every multi-turn archetype describes something else: `n` new tokens
  attending against a `P`-token prefix already resident. `attentionPairs` now takes the prefix and
  charges `n * P + causal(n)` on a full-attention layer, with sliding layers capping the prefix at
  their window on the same dispatch they already make.

  Two things about it are easy to get backwards. **It makes prefill slower, not faster** — a prefix
  cache saves _re-reading_ the prefix, which this function never charged for, and it does not make
  the new tokens cheaper to attend. And **it is opt-in per archetype** (`Workload.prefixIsCached`,
  declared on the agent alone) rather than derived from `contextTokens - promptTokens`: deriving it
  would have moved every archetype at once, including the single-prompt scenarios the calibration
  anchors are measured at. The prefix-0 path is bit-identical to the expression it replaced, and a
  test asserts that as an identity rather than inside the ±30% band, which would have absorbed the
  mistake silently.

  **The prefix is the session minus what the turn needs**, and every sentence has to say so. It took
  two review rounds to state it correctly. `agentSession` is the whole window, so the first draft
  printed the window itself — claiming a cache holding 80K of working set in a 64K window. The
  second subtracted the turn but not its answer, spending the entire budget on prefix and prompt and
  leaving the 512-token reply nowhere; the tell was at the boundary, where `cachedPrefix(needs(id))`
  returned 512 and claimed the room to answer as cached history for a scenario with no history at
  all. It goes through `needs(id)` now, which is the same boundary `fits` tests, because a limit
  stated twice is a limit that will disagree with itself — a comment this file already carried about
  a different copy of the same number.

  The consequence is a real regrade: 8B at Q4_K_M on one 5090 goes from 6.0s to 14s on an agent turn
  against the 47.5K resident in a 64K session — about seven times the query-key pairs — which is the
  difference between clearing the 10s bar and not. The threshold did not move; the estimate started
  describing what an agent does.

  **The capacity bars were not part of it**, and the review's second claim — that a rig holding
  exactly 64K is admitted for a request needing another 512 — does not hold. The session constants
  are windows, not prompts, so they already include generation; the agent's prompt-level bar
  (`fits`, at 16,896) does carry the allowance; and once the prefix is 47.5K the occupancy closes on
  65,536 exactly. Long-context is the deliberate contrast: its bars _do_ add the allowance, because
  its 131,072 is a `typicalPromptTokens` and needs room to answer on top. Same numeral, different
  kind of quantity — and widening the session bars would have forced `holdsFullSession` to advertise
  a "64.5K session".

  **Chat is the honest second candidate and is deliberately not declared.** It is back-and-forth by
  its own description and re-reads nothing under the same caching. Deferred rather than dismissed:
  at an 8K context a 1K turn against 7K resident is ~15x the pairs against a 2s bar, so it regrades
  chat on slow rigs and wants its own evidence. (#23.)

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

**Tests**

- **`e2e/` covers what jsdom structurally cannot, and nothing else.** Layout, scrolling,
  `@media (pointer: coarse)`, and canvas actually painting — everything else stays in Vitest, where
  it runs in a second. The rule is not tidiness: the gap shipped a bug. The Matrix's click-to-scroll
  was anchored on a `display: contents` element, which generates no principal box, so
  `scrollIntoView` returned early in every real browser — and jsdom has no `scrollIntoView` at all,
  so the guarded call passed every test. Caught in review; the replacement was believed correct and
  had never been observed working. All three scroll specs now fail if the anchor is put back.
- **A spec that measures the wrong element is worse than no spec**, and this suite produced three
  of them on the first run: a region-wide button locator that caught the measure toggles instead of
  the grid cells, `getByLabel('Model')` matching the Matrix's own section name, and
  `getByRole('button', { selected })`, which Playwright rejects outright. Each looked like an app
  bug for a few minutes. Mutation-check anything asserting geometry.
- **The touch project is emulation, so it asserts the emulation first.**
  `matchMedia('(pointer: coarse)').matches` is checked in its own test before any size is measured,
  or a change in how Playwright emulates a device silently moves every other assertion onto the
  mouse branch, where they all pass.
- **`vite preview` binds `localhost`, which is `::1` on an IPv6 host** — so the config passes
  `--host 127.0.0.1` to match the URL Playwright probes. Without it the run dies on a `webServer`
  timeout that says nothing about why.
- **Pin the state a conditional defect needs, and assert you reached it before measuring.** The
  Matrix legend's overflow (#34) needed three keys at once, two of them conditional, and on a fresh
  page only one renders — so a spec written against `/` passes with the bug intact. The scenario is
  in the querystring (`?r=mlx&q=q5_k_m`) and the three keys are asserted in their own test. Deleting
  the query params leaves all four geometry assertions green and fails only that one, which is what
  it is for.
- **Measure the viewport the defect actually needs.** The same spec was first written at 390px,
  where there is no overflow at all: the prose keys wrap their own text and the panel's padding
  absorbs the rest. It appears at 360 and escapes to the document at 320. Four assertions passed
  against unfixed markup before the widths were probed rather than assumed.
- **And the fix that closes the filed issue is rarely the whole defect.** The ramp is `flex-1`, so
  its flex basis is 0 and it is the only item in the row that yields. On the filed markup that put
  it at **zero width at every viewport from 320 to 1024px** — the legend's entire subject missing
  on a laptop, while the prose about the exceptions sat at full size, and nothing reported it.
  `flex-wrap` alone does not fix that: a zero-basis item still takes only the free space left on
  its own line, so it survives wherever a line breaks early (139.8px at 390, 373.8 at 640) and
  collapses wherever the keys nearly fill one (69.8px at 320, **13.6px at 1024**). A floor on the
  ramp group is the other half — `min-width` is resolved into the hypothetical main size, which is
  what both line-breaking and shrinking are measured against, so the ramp claims a width or takes a
  line of its own. The two halves are separately mutation-checked, and the desktop layout is
  unchanged by either.
- **A `rem` floor is a floor the viewport cannot argue with.** The obvious `min-w-48` fixes the
  ramp and quietly reopens the overflow: browser text scaling grows the root font size without
  shrinking the viewport, so at 320px with a 24px root the 12rem floor alone took the document to
  343/320 — the sideways scroll the wrap was added to remove, returning in the one setting a reader
  most needs it gone. `min-w-[min(12rem,100%)]` yields instead, and is identical at the default
  root. Worth checking on any `min-w-`/`w-` in rem that a narrow layout depends on.
- **Reflow at 200% text is a different test from reflow at 320px, and the page passed one while
  failing the other by 89px.** WCAG 1.4.10 asks about a narrow viewport at the default text size,
  which this app already satisfied; 1.4.4 asks about text scaled to 200%, which browsers do by
  growing the root font size and leaving the viewport alone. So every rem-derived width and every
  `whitespace-nowrap` line grows and nothing gives them more room. Fixed in #35, and the shape of
  the fix is the point: the filed instance was one of four identical `whitespace-nowrap` panel
  headers, now one `PanelCount` that protects the numeral pair — "12 of 425" broken across a line
  reads as two unrelated numbers — and lets the noun after it wrap like the prose it is.

  **The second offender was not a nowrap at all, and would not have been found by fixing the first
  one.** A non-wrapping flex row's min-content is the _sum_ of its children, and the segmented KV
  control sets the width of its grid column — so four options at a 32px root widened three `w-full`
  sliders in a panel the control is not part of, and it was the _sliders_ that left the viewport.
  `flex-wrap` makes the floor the widest single option instead. Two separate mechanisms, one
  symptom; the probe that proved the first fix is what found the second, which is the general
  lesson worth keeping.

  **The third offender was padding, and only CI could see it.** Both fixes above passed locally
  with 18px to spare and failed on the Linux runner by 4px, on markup neither run had changed. The
  cause is that the app's font stack — `ui-sans-serif, system-ui, -apple-system, 'Segoe UI', …` —
  resolves to SF on a Mac and to fontconfig's default sans on a runner, which is wider. **The
  overflow was real, not an artefact**: the page genuinely scrolled sideways for anyone whose
  system sans is wider than SF, which is most Linux users and any Windows machine not reaching
  Segoe UI. Measuring on one machine's typography is measuring the machine.

  The lever was padding, because at a 32px root the shell consumed **146 of 320px — 46% of the
  viewport** before any content was laid out. `p-4`/`p-5` are rem-derived, so they grow with the
  text while the viewport does not. Every one is now `p-[min(1rem,4vw)]` / `p-[min(1.25rem,5vw)]`,
  which is identical at the default root and yields only when the root font has outgrown the
  screen — the same shape as the `min-w-[min(12rem,100%)]` fix in #34, and the general form of that
  lesson: **a rem length in a layout a narrow viewport depends on wants a viewport term beside it.**

  `e2e/reflow.spec.ts` now runs every scenario twice, once at the host's own fonts and once at
  `'Courier New', monospace` — deliberately wider than any UI sans, and present or metric-aliased
  on all three platforms. That is what makes the verdict portable rather than a description of the
  machine that ran it, and reverting the padding fix now fails on a Mac. Two details that cost a
  round each: Verdana is **not** wide enough to reproduce the CI failure locally, so a spec written
  at Verdana would have shipped the same green-here-red-there result again; and the font has to be
  set through `--font-sans`, because `body` sets `font-family: var(--font-sans)` and an inline
  `style.fontFamily` on `<html>` loses to it silently. The spec asserts the stress font really is
  wider than the host's before trusting any of it.

  It holds at 200% only. Past a 40px root the page is **not** clean — long single words like
  "Unsupported" and the slider labels start escaping — and that is recorded rather than fixed,
  because 1.4.4 stops at 200% and "the bar is met" is a different claim from "the layout is
  unbreakable".

- **Absolute pixel type does not scale, and the precondition meant to catch it only checked a
  heading.** Both Envelope axes were `text-[10px]`, so at 200% every other figure doubled while
  they stayed put — a 1.4.4 failure outright, and worse than the numbers suggest, since the labels
  became _relatively_ half the size on the surface a low-vision reader had just asked to enlarge.
  They are `text-[0.625rem]` now, identical at the default root. Two neighbours went with them, and
  all three are one shape — **a length derived from a glyph width, written in pixels**: the
  Envelope's `MIN_COLUMN_PX` (the column those labels sit inside) and the Matrix's `headerHeight`
  (a character count times a pixel constant, sizing a `text-xs` label whose rotation clips when the
  row is short — since #64 it is `headerBand`, and both of its lengths are `rem`). The general form
  is the thing to keep: a length measured from text belongs in the same units as the text. (#42, #44.)

- **A rotation costs two lengths, and `headerHeight` only ever charged for one** ([#64](https://github.com/MrZoller/bench/issues/64)).
  `sin(45)` and `cos(45)` are the same number, so the Matrix's device labels leaned as far sideways
  as they stood tall — 246px of reserved band at every viewport, and 142px of the same quantity
  leaking out of a scroll container the grid otherwise fitted _exactly_ at both 1440 and 1024. The
  grid got a scrollbar it did not need, the default view hid the last four device names, and the one
  it cut off first was the 40-character label the 246px had been calculated from. Both numbers now
  come from one expression, which is the actual repair; the rest is what the repair had to get right.

  **The obvious label fix reintroduces the bug the rotation exists to prevent.** The filed
  suggestion is to strip the parenthetical — `(12-ch DDR5-4800)`, `(512 GB)` — since it is already
  in the tooltip and every cell's `aria-label`. Unconditionally, that collapses the three Mac Studio
  M3 Ultra rows into one string three columns wide, which is precisely the "a header that cannot
  distinguish its own columns is worse than none" failure the 45-degree labels were introduced to
  fix. The qualifier is dropped only where the rest of the name is already unique across the
  _rendered set_, so a catalog addition that collides with an existing stem lengthens both labels
  instead of quietly making one ambiguous. 40 characters becomes 25, and the band 161px.

  **No trailing lane works, and two were built before one did.** The filed suggestion is to reserve
  `cos(45) × longest` to the right of the grid. As `padding-right` it is non-negotiable, so at 1024 —
  min-content 857px inside a 934px panel — a 141px lane forces 65px of scrolling onto a grid that
  fits. As a yielding grid track, `minmax(min-content, 1fr) minmax(0, lean)`, it takes only the free
  space that happens to exist, which is not a quantity anyone controls: at 960px of viewport the
  container is 870px, the grid 857px — it fits — and `scrollWidth` was 920px anyway, with
  "Threadripper PRO 7995WX" painted 50px outside the visible right edge. That version shipped to
  review with both of its own geometry assertions sitting above the 948–1009px window it broke. The
  lesson is the one this file keeps writing down in other words: **a reservation whose size is "the
  space left over" has not reserved anything.**

  **Leaning the labels the other way has no width dependence at all** — the issue's third lever, and
  the one that holds. Anchored `right-1/2` and turned `+45deg` about `origin-bottom-right`, each label
  ends at its own column and runs up-and-_left_ over the model-name column, which is in flow and
  inside the scroll container at every width. `scrollWidth == clientWidth` at 1440, 1280, 1060 and
  1024, and `scrollWidth == the grid's own width` at 960, 390 and 320; the grid also gets back the
  141px the lane was taking off its width at 1440. Text that ascends left-to-right has to lean right,
  so this is geometry rather than taste: the direction and the anchor are one choice.

  **What that buys has to be reserved, because overflow to the left is worse than overflow to the
  right.** Right-side overflow is at least reachable by panning; left-side overflow is not scrollable
  at all — the name is gone at every scroll position. So the model column carries an explicit
  `min-width` of the same `lean` the band is derived from: 8px of real effect today (the longest model
  name asks 133px, the lean 141px), inert at 1440 where auto table layout hands the column 338px, and
  binding at 390 and under 200% text. `matrix-header.spec.ts` asserts the contract — the reservation
  covers the furthest any label actually leans — at the two widths where it binds, since at 1440 the
  slack would make it pass without testing anything.

  **And the sweep is the interesting half.** The defect class is not "a rotated label" — it is _any_
  scroll container whose scrollable area is enlarged by something out of flow: a rotation, an
  absolute label, a ring, a shadow. All three of the app's `overflow-x-auto` containers can do it,
  so `matrix-header.spec.ts` now checks every one of them: a container may only scroll as far as its
  **in-flow** content reaches. That sweep was wrong twice, in ways worth keeping. First it measured
  each child's `scrollWidth`, which counts the rotated labels because they are descendants of the
  table — green against the filed defect with 142px of overflow in front of it. Then it compared
  in-flow content against `clientWidth` instead of against `scrollWidth`, which is trivially true of
  everything that scrolls at all: green again, at 390px, against the same defect. Both versions read
  as coverage. The rule has to name the two quantities it is actually about, and out-of-flow boxes
  have to be excluded by hand.

- **The Envelope's axis titles are stacked rather than rotated, and the cost of rotating is smaller
  than it first looks — a fraction of a column, not a column.** Both axes were bare number strips:
  the gutter ran 1…128, the strip under the plot ran 2K…128K, and the only thing separating "128
  users" from "128K tokens" was a `K` in the smallest and faintest type on the page — while the
  hidden table, the caption and the canvas `aria-label` all named both quantities. The picture was
  the one representation that did not say, and its y axis runs bottom-up, which was stated only in a
  source comment, so a reader assuming top-to-bottom read the default field as "128 users at 2K is
  the comfortable cell" when it is 1 user at 2K.

  Width on this surface is genuinely scarce — `MIN_COLUMN_REM` floors every column and the plot
  already scrolls inside its own box at 320px, where it measures 364px of content in a 230px
  viewport — but **the first version of this note priced a rotated title at that whole ~110px
  label and that is wrong.** `writing-mode: vertical-rl` costs one line box of horizontal space, and
  costs the plot's _content_ width nothing at all; only its scroll viewport narrows. (`rotate-90` is
  the expensive one: transforms do not affect layout, so the element still reserves its unrotated
  box. That is the implementation the ~110px figure described, and not the one anyone would reach
  for.) Recorded because a later session reading the old wording as a hard constraint would decline a
  rotated title on the Matrix — whose axes are still untitled — on a number that is wrong for the CSS
  it would actually write. **The real reason to stack is legibility**: 12px type on its side is the
  least readable ink on a surface whose entire complaint was that its meaning rode on the least
  readable ink, and the ↑ has to read as up. Vertical space is the axis this panel has spare.

  The x title sits _outside_ the scroller so it stays centred on what the reader can see rather than
  on the scrolled content. Both titles are `aria-hidden`, like the tick strips, because the canvas
  `aria-label` is the textual equivalent and a visible title in the accessible tree has a screen
  reader hear each axis named twice. `e2e/envelope-axes.spec.ts` holds the geometry at 320px — the
  title above the canvas, the plot's width accounted for by the tick gutter and the row gap alone —
  and it is red with the titles deleted, checked. Its three box comparisons measure a **`Range` over
  the glyphs, not the paragraph**: both titles are block children of full-width containers, so their
  element rects are their containers' rects for any alignment and for text that overflows, and the
  centring and in-panel assertions were all true with `text-center` deleted before that substitution.
  Same lesson as the `getClientRects().length === 1` note below. (#81.)

- **One name per setting, and `SETTING_LABELS` is where it lives.** The same #81 fix found the Usage
  sliders saying "Context per sequence" and "Concurrent users" while the Envelope's table caption
  said "context length" and its row-header column said "Users" — two settings under three spellings,
  in one panel, with the field's own axes naming neither. Same failure `kvLabel` exists to prevent,
  one level up, and the same shape the "expect a subset of a class" rule predicts: the issue named
  the two missing titles, and more hand-written copies were live. The constant is keyed by `Config`
  field and `satisfies Record<keyof Config, string>`, so the keying is a claim the compiler checks
  rather than a comment — including the four setup labels, whose second reader is the Matrix's
  `sr-only` "Model" row axis, agreeing with the control today by coincidence.

  **The line is whether a surface names a setting or says something in a sentence**, and prose stays
  prose. "Currently at 32K context and 1 user" is a sentence about a cell; so is the Envelope's
  subhead, "How much room is left — context against concurrent users", which is also the section's
  `aria-labelledby` target. Substituting the labels there was tried and reverted: it renders "How
  much room is left Context per sequence against Concurrent users", announced verbatim every time
  the landmark is, which is worse English than the drift it prevents and out of register with the
  Matrix's sibling subhead. Axis titles, captions and column headers name settings; headings and
  hints talk. (#81.)

- **Simulating text zoom by setting the root font size is not text zoom, and a test built on it
  reports layouts nobody can reach.** Widening the reflow sweep to the `sm` and `lg` boundaries —
  640 and 1024 — produced six red tests and three plausible-looking layout bugs, all artifacts.
  Tailwind v4's breakpoints are `rem`, and **`rem` inside a media query resolves against the
  browser's _initial_ root font size, not an author-set one** — measured, not assumed: with
  `documentElement.style.fontSize` at 32px, `(min-width: 40rem)` still matches at 640px. So the
  simulation grows the text and leaves every breakpoint where it was, and the page lands in states
  real zoom never produces: three columns crushed into 213px each at a width that, for an actual
  reader at 200%, is a single stacked column.

  **The fix was to stop simulating.** `--blink-settings=defaultFontSize=32` changes the browser
  default — the thing a reader actually changes — so the breakpoints move with the text: `sm`
  becomes 1280px and `lg` 2048px, both verified. The `reflow` Playwright project launches with it
  and the spec asserts both that the root is 32px and that 640px is now _below_ `sm`, so a
  Blink-internal switch silently ceasing to work fails loudly instead of quietly re-testing at
  100%. The sweep then covers 320/640/1280/1920 honestly.

  Worth recording that the first attempt concluded the opposite. `--default-font-size` — a
  plausible flag name that does not exist — was tried, had no effect, and became a written claim
  that faithful emulation was impossible, with a carefully-reasoned model built around the
  limitation. The model was correct and the reason for it was false. A negative result about a
  tool is worth one more minute of checking than a positive one, because nothing later contradicts
  it. (#41.)

- **A grid of buttons is one tab stop, not four hundred** ([#52](https://github.com/MrZoller/bench/issues/52)).
  The Matrix is 408 cells, each a `<button>` with a full-sentence `aria-label`, and it sits _above_
  the Usage panel in DOM order — so reaching the context slider that drives every figure on the page
  took 422 presses of Tab, and a screen-reader user heard 408 sentences on the way. The ARIA grid
  pattern is what that is for: `role="grid"`, a roving `tabIndex` so exactly one cell is in the tab
  sequence, arrows to move between cells, Home/End and Ctrl+Home/End for the ends. 422 becomes 15.

  **Why this one survived when the other four axes did not**: touch targets, reflow at 200%,
  coarse-pointer queries and palette contrast all have tokens and specs behind them, and focus order
  had nothing looking at it. An axis with no spec is not an axis anyone is checking.

  A skip link was the cheaper alternative and is deliberately absent: past the roving index it saves
  a single keypress, and it never addressed the screen-reader traversal at all — which was the
  larger half of the problem.

  **The counting splits across both suites, deliberately.** The tab _sequence_ is a DOM property —
  `tabindex="-1"` is reachable by script and never by Tab — so `App.test.tsx` asserts it in a
  second. Whether pressing Tab actually lands where the sequence says is something jsdom cannot
  answer at all: it implements no sequential focus navigation, so a Tab keydown moves nothing and
  `document.activeElement` stays put. That half is `e2e/matrix-grid.spec.ts`. Both were checked
  against a reinjected defect, and the two that fail in the browser are exactly the two jsdom
  cannot see.

  Left alone at the time: **the Usage controls stayed below the two grids in DOM order.** They are the
  primary input and there was a real argument for moving them, but that read as a layout decision
  about what the page leads with rather than a keyboard-reachability bug, and the fix above already
  took the walk from 422 presses to 15.

  **That deferral was resolved the other way in #66, and the reason is worth recording because it is
  not the one this paragraph anticipated.** The argument that carried it was not distance and not
  keypresses: it was **discoverability**. A reader who never scrolls to the bottom never learns the
  controls exist, so the page presents figures they chose as fixed properties of the model and the
  hardware — which makes it a claim about what the numbers _are_, not a preference about what the page
  leads with. Measured before the move, at 1440x900: the context slider sat at y=3004 and the memory
  bar it fills at y=602, so the control was 2,402px below its own output. The pixel count is the
  evidence; the misrepresentation is the reason.

  Usage now sits directly under Setup — the landmark's own name since #74, where it stopped being
  "Configuration" and became an `sr-only` heading, which is what `App.test.tsx` and
  `usage-placement.spec.ts` both locate it by. Envelope and Matrix are the last two _output_ panels,
  and deliberately not "the terminal panels": on any MoE selection `Bench.tsx` renders the
  architecture aside after `<Matrix>`, which is a conditional panel and the reason
  `matrix-readout.spec.ts` measures section and document heights rather than a following sibling.

  The roving tab index is untouched and still does the work this entry is about, though not in the
  figure this paragraph first quoted. "15 presses either way" was the walk to the Usage panel, and #66
  moved Usage to the top of the page, so that quantity no longer describes anything: the grid costs
  **one** press to cross whichever end of the page it sits at, and what is worth counting now is the
  page. Two counts, from the two suites, and the gap between them is worth stating exactly:

  - `App.test.tsx` — **26** DOM tab-stop candidates over the whole document, against **1,495**
    without the roving index (26 − 1 + 1,470). That subtrahend is the grid, so it moves with the
    catalog: it was 714 before #77 doubled the model list, and a counterfactual quoting the old one
    is a wrong expected value for whoever reinjects the defect.
  - `e2e/matrix-grid.spec.ts` — **23** real browser stops, inside `<main>`.

  Three apart, from two unrelated causes and not one. **Two** are the KV radio group, which offers Tab
  only its checked member and a `querySelectorAll` all three. The **third** is scope rather than
  semantics: the masthead's copy-link button is a real stop that the browser walk excludes, because it
  sits outside `<main>` and that walk ends when focus leaves. Both hold a ceiling of forty,
  deliberately the same number, since a bound that fires in one channel and not the other is a bug
  report about the wrong file.

  **The cost is real and was accepted rather than solved.** At 390px both panels stack, so nine
  controls — about 620px — now precede the first figure, where before there were none. #66 named a
  sticky summary strip as the mitigation and deprioritised it, and the phone spec asserts the honest
  property — the controls are inside the first two screens — rather than a landing-screen claim the
  layout does not make. The slider and the bar _do_ come within one viewport of each other here, 683px
  apart on an 844px screen, which is the laptop test's property holding on a phone; what is not true is
  that either of them is on the first screen. Three source comments that asserted the old geometry
  moved with it, on the reasoning that a comment describing the old layout is how the next reader
  concludes the fix was a mistake.

  **And the spec did not assert the property this paragraph credited it with.** It measured
  `panel.top < 2 x viewport` — the panel _starts_ inside two screens — which stays green with the last
  three sliders below the fold, and below the fold is precisely where growth above Usage would put
  them one control at a time. A one-sided bound on a box whose height is the subject is the same shape
  as the `getClientRects().length === 1` note below: an assertion that cannot fail for the reason it
  was written. It reads the panel's bottom edge now, at 1,401px against the 1,688px bar, so the margin
  is 287px of measurement rather than of slack.

- **A focus indicator is a mark of its own, never a colour swap and never a channel a resting state
  already uses** ([#67](https://github.com/MrZoller/bench/issues/67)). The four primary selects —
  Model, Hardware, Quantization, Runtime — removed the outline and replaced it with a 1px border
  colour change measuring **1.95:1 against the unfocused edge**, where SC 2.4.13 asks for 3:1 at a
  2px minimum. `--color-control-border` had been raised to `#646d88` specifically so a control's
  edge cleared 3:1 _before_ focus; the focused state never got the same treatment, so the most
  important controls on the page were the only ones whose indicator you could not see.

  Two more instances came out of the sweep, both of them listed in the issue as already correct:

  - The budget legend drew `focus:ring-1` — 1px, half the minimum, and with the outline suppressed
    it is the whole indicator rather than a decoration on top of one.
  - The Matrix marked its selected square with `ring-2 ring-[accent] ring-offset-1` and lit
    `focus:ring-2 ring-[accent]` on focus: **the same channel, width and colour**, so focusing the
    marked square changed nothing whatsoever. A 1:1 change contrast, which is the select's 1.95:1
    in its most extreme form, and reachable in one click — clicking a cell makes it both the
    selection and the roving tab stop, so the marked square is exactly where Tab lands coming back
    to the grid. Selection is drawn inside the cell now, and focus stays outside it.

  **Moving a mark onto the heatmap changes what it has to contrast against, and the accent was never
  validated there.** Separating the two channels put the selection mark inside the cell, which took
  it off `--color-surface` — where `tokens.ts` measures the accent at 7.14:1 — and onto the ramp,
  where a single-tone accent frame measures **2.00, 1.48, 1.06, 1.38, 2.04, 3.07 and 4.52:1** across
  the seven steps of `sequential`. That is below the 3:1 non-text minimum on **304 of the grid's 408
  squares**, including the default selection at 1.38:1 on `#3987e5`; on `#6da7ec` the two colours sit
  0.022 apart in relative luminance, so the mark was a pure hue difference at 1.06:1 — #67's own
  failure mode, re-shipped as the resting state. The first attempt at this fix did exactly that, and
  the only reason it looked measured is that the browser spec scored the mark against the panel
  behind the cell instead of the fill on top of it.

  So the selected square wears **two tones**: 2px of accent bounded by `--color-surface` on both
  sides — the 2px `border-spacing` outside, a 1px separator inside — which is the dataviz surface
  ring, and the same trick `Envelope.tsx` already used for its "you are here" mark on the same ramp
  ("A ring, not a filled dot: the cell's own colour has to stay readable underneath it"). The
  invariant is _not_ that either tone clears 3:1 everywhere; it is that **one of them always does**:
  the separator on the five light steps (14.26 down to 3.50:1), the accent on the two dark ones where
  the separator disappears into the fill (3.07 and 4.52:1). Worst case 3.07:1, zero squares under the
  bar. Anything drawn on a cell in future — a value label, a comparison marker — inherits this
  obligation and none of the existing measurements, because `--color-accent` is validated against
  `surface` and nothing in `tokens.ts` says a word about the ramp. The separator rides the
  `--tw-shadow` slot rather than a second inset ring because Tailwind composes one box-shadow chain in
  a fixed order — `inset-shadow, inset-ring, ring-offset, ring, shadow` — and only the last slot
  paints _under_ the accent, which is what keeps the accent 2px wide rather than 1px. Spelling a
  bracketed utility out in prose is worth avoiding for its own small reason: Tailwind scans comments
  and Markdown as source, so an example in a sentence compiles to a rule of dead CSS.

  **The mechanism differs by control and the bar does not**, which is the decision rather than an
  oversight. The selects use `outline` where their neighbours use `ring`, because a ring is a
  `box-shadow` and a native `menulist` select is painted by the platform in WebKit — the fix that
  matched the rest of the app would have shipped nothing at all on Safari, and Chromium cannot tell
  you that. What is uniform is 2px and 3:1 against whatever the mark is drawn on, measured per
  surface rather than assumed from the token.

  **Split across both suites, for the reason #52's counting was.** Which indicator a control
  _declares_ is a class-list property, so `App.test.tsx` sweeps all 400-odd focusable elements in a
  second and pins the declared width, colour, and the channel collision — and, because the fill is an
  inline style and the tones are token names, it also does the ramp arithmetic above over every fill
  the grid paints. Whether an indicator _paints_ 2px at 3:1 needs a real stylesheet and a real focus
  ring, and jsdom has neither: that is `e2e/focus-indicators.spec.ts`, which walks the tab sequence
  with Tab rather than `focus()` — the UA ring is `:focus-visible`-gated and a scripted focus does not
  reliably satisfy the heuristic, so a sweep driven by `focus()` reports every slider on the page as
  painting nothing. Two things in that reader are easy to get wrong and both have now been wrong
  once: it exempts `outline-style: auto` from the thickness check, because Chromium reports 1px for a
  dual-tone ring it paints at 2px (asserted to still match something, or the exemption quietly
  becomes the rule), and it scores each layer against the colour that layer actually covers — an
  `inset` shadow over the element's own fill, everything else over the first opaque ancestor. Scoring
  every mark against the ancestor is what certified a 1.06:1 mark as 7.14:1.

- **`pointer: coarse` does not mean "this user can touch the screen".** It describes the _primary_
  pointing device, so a touchscreen laptop, a Surface, or an iPad with a keyboard case reports
  `fine` — and the disclosure toggles dropped back to 16px for someone who can still put a thumb on
  the glass. They use `any-pointer: coarse` now; the Matrix grid deliberately does not, because
  widening it there buys 44px rows on every laptop that merely has a touchscreen, multiplied across
  hundreds of cells, while the toggles cost 28px once per panel and are the accessibility
  affordance.

  **The asymmetry is asserted against the stylesheet rather than the layout**, which is a limitation
  worth recording. A true hybrid cannot be emulated: Playwright's `hasTouch` makes Chromium report a
  touch-_only_ device — both queries true, `any-pointer: fine` absent — and `Emulation.setEmulatedMedia`
  over CDP with explicit pointer features is silently ignored. Both measured. So
  `e2e/hybrid-targets.spec.ts` reads the shipped CSSOM and checks the query, the selector and the
  declaration, which is falsifiable and covers the failure an arbitrary Tailwind variant really has:
  compiling to nothing. It has to descend through `CSSGroupingRule` to do it — v4 nests utilities
  inside `@layer`, and a walk that only recursed through `CSSMediaRule` found nothing and passed.
  (#43.)

- **The numeral pair's nowrap cannot be falsified by geometry, and the spec says so.** "67 of 408"
  is short enough that it never breaks on its own at any root size from 32px to 64px — measured,
  not assumed — so `getClientRects().length === 1` is true with the class and true without it. The
  spec asserts the computed style instead. Worth stating because writing the geometry version is
  the obvious move and it would pass against markup with the protection deleted.

- **A probe that filters on a field the type does not have reports a clean sweep over zero cases.**
  Verifying that a serving failure at one user cannot recover at more meant sweeping the catalog, and
  the sweep filtered candidate quants with `runtime.weightFormats.includes(quant.format)`. `QuantSpec`
  has no `format` — the id _is_ the format — so `includes(undefined)` was false for every combination
  and the probe reported perfect monotonicity over **nothing**. It printed the sample count only
  because that had been added on a whim; asserting the property directly would have written a
  confident falsehood into a commit message. Every sweep in this repo now asserts it matched a
  plausible number of cases _before_ trusting its result, which is the same fail-open shape the
  touch-target exemption list is guarded against below.

- **`aria-describedby` is an IDREF _list_, and resolving it as one id fails in the direction that
  passes** ([#74](https://github.com/MrZoller/bench/issues/74)). Specs handed the whole attribute to
  `getElementById`, which returns `null` the moment a control carries two ids — so appending a second
  description to a picker would have reported a control that _has_ a note as having none, and every
  "no note runs past two lines" budget in `config-notes.spec.ts` is satisfied by zero lines. The
  resolvers split on whitespace and sum across the elements now. This is the third instance of one
  class in this file — with the `quant.format` probe above and the touch-target exemption list below —
  and the class is worth naming: **a resolver that fails silently makes a spec pass on broken
  markup.** All three failed by finding nothing and reading nothing as compliance, so the general
  guard is the same one: assert the sweep found something before believing what it says about it. Each
  affected spec now also asserts the note's _text_, because a line count alone is one-sided and the
  curated note was in fact dropped from that control once before, taking the 3090's NVLink caveat
  with it.

- **`getBoundingClientRect().top` is the border-box edge, so a padding comparison must add the border.**
  Asserting that an `sr-only` heading takes no grid track meant measuring the first control against
  the panel's content edge, and `panel.top + paddingTop` is short by the border width — which reads as
  the control sitting _lower_ than it does, making the assertion pass with slack it had not earned.
  `borderTopWidth` rides along with `paddingTop` everywhere this is computed.

- **A premise that becomes unsatisfiable when the fix works is not a weak premise — but it cannot also
  be the evidence.** The Matrix's readout reservation — the responsive `min-h` on that paragraph,
  held by `e2e/matrix-readout.spec.ts` — pads a three-line sentence into a four-line
  box, so the test written to prove "two sentences of different height do not move the panel" found
  equal heights once the fix landed and failed on its own precondition. The right shape is to assert
  the _consequence_ here and put the _evidence that the inputs differ_ in a sibling test.

  **The sibling test is owed rather than written, and saying otherwise was this entry making the
  mistake it is about.** The sweep that exists samples every seventh cell and asserts the rendered
  height is positive and no greater than the reservation — but the rendered height already _includes_
  the `min-height`, so a readout whose sentences all fit one line satisfies it, and deleting the
  reservation escapes it. That is evidence of the reservation working, not evidence that the inputs
  differ, which are the two halves this entry exists to separate. What it needs is a natural height
  measured with the reservation lifted, and an assertion that the line counts genuinely differ. It
  belongs with [#102](https://github.com/MrZoller/bench/issues/102), which reworks that reservation
  for a 32px root and has to measure the unreserved wrapping anyway.

- **Reflow tests measure the panel; the page is a different question.** The Matrix readout was moved to
  be the panel's last child so its height could not push the legend, and the commit claimed "nothing
  follows it". At the time `Bench.tsx` rendered the Usage section immediately after `<Matrix>`, so a
  height change still moved that panel: the claim was verified inside the component and written about
  the document.

  **The obvious repair — assert against the following section — is the one the current spec
  deliberately rejects**, and the reason is that #66 moved Usage above the figures. What follows
  `<Matrix>` now is the MoE aside, which is conditional, so there is no stable next section to measure
  against and a spec naming one would go quiet on the scenarios where the aside does not render. The
  canary in `e2e/matrix-readout.spec.ts` is therefore the two quantities that cannot stop being
  downstream of the readout: the **Matrix section's own height**, which is where the growth would be,
  and the **document's height**, which is everything below it whatever that turns out to be. The
  general lesson survives the move — measure past the boundary the claim is about — but "the following
  section" was a fact about one layout, not the rule.

- **A locator re-resolves, so an interaction that renames its target invalidates it.** A disclosure
  toggle's accessible name _is_ its label, and clicking it turns "Show the full hardware note" into
  "Hide…". Reading `aria-controls` off the `/show…/` locator after the click waited the full timeout
  for a button that no longer existed, and reported the failure against the region lookup two lines
  later — pointing at the wrong code entirely. Read what you need from an element before the gesture
  that changes it.

- **Pick the viewport the defect needs, then assert the defect is reachable there.** Two specs in the
  sweep passed against unfixed markup for want of this. A sticky-placement test written at 1280x900
  could not fail, because the grid is 745px tall and the readout is on screen anyway at that height;
  it needed 600px. A stress-font precondition measured `'Not measured'`, a short string of wide
  lowercase letters where a proportional sans and a monospace agree most — 1.191 on macOS and **1.019
  on the CI runner**, under its own 1.05 floor, so the guard failed the build for typography rather
  than layout. It probes `'Inline code completion'` now, the string `reflow.spec.ts` uses. That pair of
  ratios also measures something worth keeping: Courier New is metric-aliased on both platforms, so
  1.191/1.019 puts **the Linux runner's default sans about 17% wider than SF**.

- **A touch-target spec that names its controls will always be out of date.** The old one measured
  the three Matrix toggles it knew about, and three 16px buttons on other surfaces went unnoticed
  until someone looked (#29). It sweeps now: every pointer target on the page, with the `sr-only`
  radios resolved to the label that actually receives the tap, and both disclosures opened first —
  half the page's controls do not exist until something opens them, including the Envelope's table.
  Exemptions are data with a written reason, and each is asserted to still match an element, since
  an exception list is the one part of a sweep that fails open.

- **A row that declares its own grid is a table that measures its columns once per row**
  ([#70](https://github.com/MrZoller/bench/issues/70)). The workload strip put
  its three column tracks on each `<li>`, so every row was its own grid container and the middle
  `auto` track was sized from that row's own label — the reason column, the third one, started at
  444, 446, 457, 475, 495, 499 and 503px at 1440. Columns 1 and 2 lined up because the first track is
  fixed, and _that_ is what makes the third read as a column rather than as prose. It carries the
  panel's argument: seven archetypes, seven answers, and the written reasons are what explain the
  differences, so they have to be scannable against each other.

  A subgrid on the row, with the tracks moved to the `<ul>`. Not `display: contents`, which is the
  same idea and the tempting one-liner: a row that generates no box is a `<li>` shipping browsers drop
  from the accessibility tree, and `order` applies among siblings of _one_ container, so dissolving
  the rows would sort all twenty-one cells into a block of labels, a block of status words and a block
  of reasons at the stacked width. The row therefore keeps its own two-column grid below `sm` and only
  hands its columns back above it.

  **The row's own template has to be `max-sm:`-scoped, and that is the silent part.** Subgrid is
  Baseline widely available — Firefox 71, Safari 16, Chrome and Edge 117 — but it is _not_ inside
  Vite's default build target, which floors at Chrome 111 (`baseline-widely-available` resolves to
  chrome111/edge111/firefox114/safari16.4), so Chrome and Edge 111–116 are browsers this build targets
  and the feature is missing from. There `grid-template-columns: subgrid` is invalid and dropped —
  and anything the row declares _unconditionally_ survives, so an unscoped two-column template leaves
  the row a two-column grid at every width while `sm:order-none` cancels the stacking: `● Yes` renders
  before the label, which is neither of the two layouts the component supports. Verified by serving
  the built CSS with the subgrid value invalidated: tracks of 269.016px and 780.984px, status word at
  x=189, label at x=470, reason wrapped to a second line at x=189. Scoped to `max-sm:`, the same
  browsers get a row with no template — one implicit 1062px column, three cells stacked in DOM order,
  which is the order they read in left to right. `App.test.tsx` pins the scope, because no browser in
  CI can show the fallback.

  **The trap is that fixing it makes the obvious precondition vacuous.** With the tracks shared,
  every label _cell_ is exactly one width — the column's — so a spec that reads
  `getBoundingClientRect().width` off the label to prove the labels differ from each other reports
  0px of spread and fails on the fixed markup. It has to measure the glyphs: a `Range` over the
  cell's contents. Written the wrong way first and caught by running it, which is the argument for
  running a geometry spec against both states rather than one.

  **Swept rather than assumed.** A throwaway probe walked every container on the page whose children
  are same-tagged multi-child flex or grid boxes, at 640/768/1024/1440 across three scenarios, and
  reported the spread of each nth child's offset. One true instance — this one. The two candidates
  worth naming both measure clean: the Telemetry tiles are three independent flex columns whose
  internal rows could rag on the row axis and do not (0px at every width), and the BudgetBar and
  Envelope legends are wrap-flow rows where a shared column is not the reading. The Matrix and the
  two disclosure tables are real `<table>`s, which is this fix by other means.

- **A bar that scales to its own overflow stops measuring anything, and the answer is words rather
  than a broken axis** ([#73](https://github.com/MrZoller/bench/issues/73)). `scale = max(used,
ceiling)` keeps an over-budget stack on screen, which is right and stays. What it means is that the
  ceiling's _position_ is a function of the overshoot: DeepSeek V3 at Q4_K_M on one 5090 at 128K and
  8 users draws 448 GiB against a 31 GiB ceiling, so the rule lands 6.9% from the left edge and the
  panel reads "nearly full" instead of "fourteen times over". The bar's own docblock promised the
  opposite — an overflow region beyond the line, "because it turned red does not tell you by how
  much" — and past about 3x the budget is the sliver and the overflow is the entire picture.

  Three things about the fix are worth keeping. **The multiple is stated past 3x and deliberately not
  below it** (`OVERSHOOT_STATED`): at 1.1x "over by 2.2 GiB" and "1.1x the ceiling" are one fact said
  twice, and a clause on every overflow is a clause people stop reading — including at 14x, where it
  is the only thing carrying the scale. **The overshoot is `used / ceiling` computed in the panel**,
  not `placement.utilization`, because it is exactly the reciprocal of where the rule is drawn
  (`ceiling / scale`) — the sentence and the line are two readings of one expression and cannot come
  apart. **A broken axis was rejected** rather than overlooked: it is a real cost for a case a clause
  already covers, and the header states the true figures throughout.

  The other half was legibility, and it is **broader than the issue's 14x framing**: the rule is
  drawn on top of a fill at _every_ overshoot, not merely a large one. Once the bar overflows, `scale`
  is `used`, so the segments occupy the whole width and no empty track is left anywhere for the
  reference line to sit on — at 1.1x it falls at 91%, inside the amber cache that took the
  configuration over. The overshoot only decides _which_ fill.

  The class check is the useful part of the rest, and it split into two classes that look like one.
  On **separation**, the rule was the only overlay mark in the app drawn straight onto a fill — the
  Envelope's "you are here" ring strokes a 1px background counter-line under itself and the Matrix's
  selected cell carries `ring-offset` in the surface colour, so both were already right, and the house
  idiom was already "separate an overlapping mark with surface, never with more ink". The rule now sits
  in a track-coloured halo one `marks.gap` wide on each side, the same mechanism the segments already
  use between themselves.

  On **identity**, all three were wrong, and scoping the first pass to "the only mark in this bar"
  is what hid it: every fill in the app had a legend key and not one overlay did. The Envelope's ring
  was named only inside the canvas `aria-label` — so a screen-reader user was told what it was and a
  sighted reader met a double ring with nothing on the page saying so — and the Matrix's selection
  ring had only `aria-current` plus the accent hue, which is a channel a legend does not get to rely
  on. All three are keyed now, each only while its mark is drawn, which for the Matrix means "while a
  cell is marked at all": `isCurrent` is false for every cell on a linked rig, since the grid is
  scored at one device. A legend is the identity channel that does not depend on the mark being
  legible, which is exactly the property an overlay cannot promise.

  **Verification splits, and the e2e half records what it cannot claim.** The clause and the key are
  DOM and are asserted in Vitest against a reinjected defect in each direction, including the
  negative: dropping the threshold to 1 fails the 1.1x test. Position and halo are geometry, which
  jsdom reports as 0, so `e2e/budget-overshoot.spec.ts` measures them in a browser — at two scenarios
  (14x on a 5090, 2x on an M3 Ultra), because one position cannot tell a measurement from a constant,
  and with the expected offset derived from the header's own figures rather than written as 6.9%. What
  it does **not** do is read pixels: "distinguishable against the fill beneath it" would mean decoding
  a screenshot, so it asserts the mechanism — halo width, halo colour taken off the bar, the rule
  still at the true position — which fails if the bare 2px line comes back. Same shape of compromise
  as `hybrid-targets.spec.ts` makes for the pointer queries.

  **Three things review caught, all of them the same lesson: a rule stated in one channel is not
  applied until every channel states it.** (1) The table still printed `percent(bytes / ceiling)`, so
  the issue's own URL read "Weights 379 GiB / 1222%" — the exact form `multiple` exists to replace, in
  the channel that exists "for anyone who cannot use the bar" and therefore has no shape to fall back
  on. It switches at the same `OVERSHOOT_STATED`, so the two channels cannot disagree about whether a
  ratio is large, and **per row rather than per column**: the stack overshoots, the rows do not, and
  0.7 GiB of overhead against a 31 GiB ceiling is 0.02x, which `multiple` prints as "0x" — a real
  quantity rendered as nothing, the failure `percent`'s own `<1%` floor exists to prevent. Mixed forms
  in one column are the cost of not printing a wrong number. (2) The rule's ink was `border-l-2` while
  its halo was sized from `marks.lineWidth`; two literals for one weight means a token bumped to 3
  centres a 2px line inside 7px and moves the rule half a pixel off the ceiling, which the e2e
  position check's 3px tolerance would pass. Both now come from the token. (3) The e2e halo test
  asserted containment unconditionally, which the code does not promise: a fixed-width mark centred on
  a position within a gap of either edge is clipped by the bar, so a stack 0.5% over loses the right
  1–4px of separation and one hundreds of times over loses the left. The position is what is kept —
  clamping it inward would draw the ceiling where the ceiling is not — so the assertion now names the
  overhang and its own precondition instead of implying a guarantee.

**Catalog figures the UI quotes**

- **Three quantities are called "active params" and two of them are wrong for any given sentence.**
  `activeParams` is the _published_ convention, which is not one rule: `publishedActiveParams` returns
  `totalParams` outright on a dense model — embedding and all — and only on an MoE builds an
  embedding-subtracted dense residual back up with the routed share. So on a multimodal model it
  includes the non-language towers a token never touches, and on an untied dense one it includes an
  input embedding decode never reads. On a **tied, text-only dense** row it is exactly right, because
  a tied table _is_ the output projection and a full vocab matmul runs every step: Llama 3.2 3B,
  Qwen3 4B and granite-4.1-8b all have total, published and dense identical to the digit. The rule is
  not "dense rows are wrong" — it is that the published convention answers a different question and
  coincides with the physical one only where nothing is being excluded. `activeDenseParams`
  is the always-active dense part and excludes the routed experts. `effectiveActiveParams(model, b)`
  is `activeDenseParams + expertParams * expertFraction(model, b)`, and it is the physical count — the
  one the Bench's aside has to print, since that sentence says a token "routes through" a figure and
  then attributes the decode rate to it. It printed the first, was "corrected" to the second, and both
  were caught in review.

  **`speed.ts` divides by neither, and that is worth stating precisely because the aside sounds like
  it does.** `estimateDecode` reads `activeWeightBytes(model, quant, batch)`, which prices the dense
  and expert halves at their own widths rather than blending them — the correction recorded under
  **Engine** above, and about a factor of two on an expert-only scheme like MXFP4, where the dense
  tensors stay BF16. `effectiveActiveParams` is the parameter count behind that byte figure, not the
  divisor itself.

  **The two errors are per-model, and neither generalises.** Written as invariants they would be
  false in both directions:

  - Published against physical, at batch 1: they are **identical** on every text-only MoE in the
    catalog, so the gap opens only where non-language towers sit inside `activeParams` — and there it
    runs in _both_ directions. Mistral Small 4 publishes 6.524B against a 6.096B basis (7.0% high);
    Command A+ publishes 24.403B against 24.981B (2.3% _low_).
  - Dense against physical: `activeDenseParams` is low on every MoE, by a ratio that spans the
    catalog — 1.91x on GLM-4.7-Flash, 2.99x on Kimi K2 (10.6B where a token traverses 31.75B), and
    8.65x on Mixtral, whose dense stack is tiny beside eight experts. "A factor of three" is Kimi's
    number and nobody else's.

  **A test for this has to pick a model where the two bases differ, and that is the whole trick.** On
  an _untied_ text-only MoE they coincide exactly — gpt-oss-20b is 3.614B either way, as are
  Qwen3-30B-A3B, Mixtral and GLM-4.7-Flash — so a test written against one passes whichever figure the
  component prints. `App.test.tsx` selects on the numeric gap rather than on "is an MoE", which is
  what makes it right for the reason rather than by luck.

  **The rows that satisfy it today are the two multimodal MoEs, and that is a snapshot rather than the
  rule.** A tied _text-only_ MoE would discriminate just as well, with no tower involved, because the
  MoE branch subtracts a tied embedding the physical basis keeps — so a maintainer who reads "it has
  to be multimodal" would reject a perfectly good fixture. The rule is the gap; the list is today's
  catalog.

- **A row identified by capacity or price must not carry another configuration's compute.** Filed
  against one row and live on four. Apple sells most SoCs in two GPU bins and offers the same memory
  with both, so capacity does not identify the part: the $999 MacBook Air ships 8 GPU cores and carried
  the 10-core rate; the 64 GiB M1 Max, the 192 GiB M2 Ultra and the 96 GiB M3 Ultra each carried their
  larger bin's. Each overstated prefill 25–33% on a machine somebody owns. Every Apple row now states
  its core count — including the rows where capacity _does_ pin the bin, because "capacity implies it"
  is precisely the reasoning that was wrong four times — and `catalog.test.ts` pins the convention
  rather than the values: `fp16` must equal the stated core count times the generation's per-core rate,
  which is constant within a generation (the M4 family is 0.85 across Air, Pro and Max).

  **Where the bin is written is not cosmetic.** `headerColumns()` keeps a trailing parenthesised
  qualifier whenever a name-stem repeats, so putting the bin in the _name_ of the three M3 Ultra rows
  or the two Mac mini M4 Pro rows takes the longest rotated header label from 25 characters to 38 and
  regresses #64. Those rows carry it in the note; unique-stem rows carry it in the name, where the
  header strips the qualifier anyway.

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

    What the session bought was, at first, on the decode axis and only there. `estimatePrefill`
    derived its linear and attention work from `promptTokens` alone, so the context reached it
    through the placement or not at all: on the 4090 above, turn and session TTFT differed by 1.5%
    — the streaming term — and on a resident rig they were identical to the digit. The agent latency
    bars were a turn's prompt pass priced on the session's placement, which is a 16K prompt
    attending over _itself_. Fixed in #23; see **A cached prefix** under **Engine**.

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
- **There is a third attention family in the wild, and the generator refuses it rather than
  flattening it into GQA** ([#76](https://github.com/MrZoller/bench/issues/76)). `deriveAttention`
  knew two, and any model whose layer stack mixes attention with linear or state-space layers fell
  through to the GQA branch and was catalogued as if _every_ layer cached keys and values.
  Qwen3-Next-80B is 12 attention layers of 48, so it derived at 96.0 KiB/token against a true 24.0 —
  12.0 GiB against 3.0 at 128K, the README's own failure mode pointed the other way.
  granite-4.0-h-small is 4 of 40, which is 10x.

  **Two guards looked like they would catch it and did not**, which is the part worth keeping.
  Qwen3-Next carries `num_attention_heads`, `num_key_value_heads` and `head_dim` exactly where GQA
  expects them, so the branch reads as a clean hit with no signal that 36 layers were just charged
  for a cache they never allocate. And `deriveLayerWindows` _did_ refuse a `layer_types` array it
  could not trust — but its filter was `t.includes('sliding')`, so Granite's all-`mamba` array
  matched nothing, `sliding.length === 0` returned `undefined`, and every layer read as full
  attention. **An unrecognised layer type is the same defect as a missing one, one axis over**; the
  vocabulary is closed now (`full_attention`, `attention`, `sliding_attention`) and anything else
  throws.

  **The family presents under at least eight spellings, the issue named two, and the first draft of
  the guard enumerated four and believed that was all of them.** Do not trust a count here. What the
  guard matches, and why it is shaped that way:

  | Spelling                                                         | Model                      | Matched by             |
  | ---------------------------------------------------------------- | -------------------------- | ---------------------- |
  | `full_attention_interval` + `linear_*`, no per-layer array       | Qwen3-Next-80B             | exact key + `^linear_` |
  | `layer_types: ["mamba", ...]` + `mamba_d_*`                      | Granite 4.0-h-small        | vocabulary + `^mamba_` |
  | `hybrid_override_pattern` + `mamba_state_dim` / `mamba_head_dim` | Nemotron-H, Nemotron-Nano  | exact key + `^mamba_`  |
  | `attn_type_list` (per-layer `1`/`0`)                             | MiniMax-M1                 | per-entry test         |
  | nested `linear_attn_config.full_attn_layers`                     | Kimi-Linear-48B            | `^linear_`             |
  | `full_attn_idxs` + `conv_L_cache`, no per-layer array            | LFM2-1.2B, LFM2-350M       | exact keys             |
  | `layer_types: ["conv", ...]`                                     | LFM2-2.6B, LFM2-8B-A1B     | vocabulary             |
  | `mb_per_layer`                                                   | Phi-4-mini-flash-reasoning | exact key              |

  **The lesson is that an enumerated list of exact key names is a list of the configs its author
  happened to open.** The first draft listed thirteen and was already incomplete against configs
  fetched the same afternoon: Granite declares `mamba_chunk_size` / `mamba_conv_bias` /
  `mamba_proj_bias` beside the six that were on it, Nemotron-Nano spells the same block
  `mamba_state_dim` / `mamba_head_dim` / `mamba_num_heads` and shares **no** exact name with
  Granite's spelling, and Kimi-Linear puts its whole Kimi-Delta block inside one nested
  `linear_attn_config` object where a flat lookup sees nothing at all — so Kimi derived as clean
  27-layer MLA, 30.375 KiB/token against a true 7.875, 3.86x, on a model whose headline claim is a
  75%-smaller KV cache. So the guard matches **key prefixes** (`^linear_`, `^mamba_`) plus the
  handful of names that carry no generalisable prefix, and the prefixes are verified against all 17
  seeds: none matches, so this rejects nothing already in the product.

  `attn_type_list` is the one entry that has to _admit_ something — M2's list is all `1`, so M2
  really is full attention throughout and a guard keyed on the key's presence would have rejected
  the model that turned out not to be a hybrid. `layer_types` length is `!==` rather than `<` for the
  same reason the entries are: a longer array and `num_hidden_layers` disagree about the stack, and
  slicing chose one silently. And the split-count clause fires only when the config states a
  count _and_ the count is a genuine split: `full_attention_interval: 1` is legal and means every
  layer attends, which otherwise produced "48 of 48 layers attend and cache; the other 0 hold a
  recurrent state" — one sentence contradicting itself, the failure this file's own rule about
  predicates and their prose exists to prevent.

  **Chunked attention is a fourth window convention and needed its own guard, not a vocabulary
  entry.** Leaving `chunked_attention` out of `LAYER_TYPES` does not refuse Llama 4: Scout and
  Maverick ship no `layer_types` at all, so the vocabulary never runs and all 48 layers read as full
  attention — 192.0 KiB/token, 24.0 GiB at 128K, against 7.125 for the real 12-global /
  36-chunked-at-8192 split. 3.4x. **A closed vocabulary only fires for configs that use the key it is
  a vocabulary for.** `attention_chunk_size` is now its own refusal. Note what is deliberately _not_
  the signal: `cache_implementation: "hybrid"` is on `unsloth/gemma-3-12b-it` and `-27b-it`, two
  shipped seeds whose windows derive correctly from `sliding_window_pattern`, so guarding on it would
  have refused two rows that are already right — a fixture in the test file carries the key for
  exactly that reason. And unlike the linear stacks, Llama 4's split _is_ derivable (`no_rope_layers`
  is 48 entries of 1/0, one global layer every fourth); what is not derivable is how many tokens a
  chunked layer's cache holds, because the mask is block-diagonal rather than trailing and residency
  comes from the runtime's chunked-cache implementation. That is what [#77](https://github.com/MrZoller/bench/issues/77)
  needs to settle before Scout can be seeded — the refusal is what makes that visible instead of
  shipping a 3.4x row.

  **Refused rather than derived, deliberately, and this is the decision to reopen with new
  information.** Pricing a hybrid properly means a third `AttentionCore` kind carrying the per-layer
  split _and_ the block's constant state term, which `kv.ts` would dispatch on the way it already
  dispatches MLA. Only the first half is in `config.json`: the state's shape is specific to the block
  (DeltaNet's `num_v_heads * head_k_dim * head_v_dim` plus its conv window, Mamba-2's
  `n_heads * d_head * d_state` plus its own) and its width is set by the runtime rather than by
  `torch_dtype` — llama.cpp keeps recurrent state in fp32. Adding the field and filling it with a
  plausible figure would put an invented number inside the fix for an invented number, and a field is
  an invitation: that is exactly how `measuredBandwidthGBs` came to exist. So the error carries the
  evidence instead — which layers cache, which do not, and the key that said so — and adding one of
  these models is a real piece of work rather than a seed-list edit.

  **DeepSeek V3.2-Exp is refused on the same doctrine and a different quantity.** Its capacity
  derives correctly through the existing MLA path; what is wrong is that the lightning indexer keeps
  an `index_n_heads * index_head_dim` cache nothing here counts, and its main attention reads at most
  `index_topk` selected positions rather than everything before it. Right about the latent and
  silently short by the indexer is not a smaller version of deriving both.

  **The refusal also has to be _reached_, and it was not.** `deriveStackShape` ran first, and
  Qwen3-Next ships an MTP module under an `mtp.` prefix — so seeded, it was refused for 1,553
  unclassified tensors instead: a true statement about a different problem, pointing whoever read it
  at `LANGUAGE_PREFIXES` rather than at the layer split. Both derivations read `config.json` alone
  and now run before anything that touches the network again, which also saves a dozen range
  requests on a model that was never going to be admitted. Verified by seeding all four models named
  here: Qwen3-Next, Granite 4 and V3.2-Exp each refuse with their own reason, and MiniMax-M2 is
  admitted at 228.7B.

  **What is still not covered, so the next session does not have to re-derive it.** The refusals are
  the floor, not the fix: no model in the table above can be _added_ until the third `AttentionCore`
  kind exists. Llama 4 needs a chunked-attention window term. `attn_type_list`'s non-`1` values are
  refused without being read, so a future list using `2` for something benign would cost a false
  refusal. Nothing here reads `ssm_cfg`, the raw `state-spaces/mamba` spelling, because no live
  config checked carried it — if one appears it will refuse only if it also carries a `mamba_*` key.
  The list of spellings is open by construction; treat any claim that it is complete, including this
  one, as unverified until re-probed against live `config.json` files.

  **And the reason all of this was untested is mechanical**: `build-catalog.ts` called `main()` at
  module scope, so importing it started seventeen rounds of network fetches and no test could reach a
  single derivation. It carries `catalog-diff.ts`'s guard now, and `scripts/build-catalog.test.ts`
  pins both the refusals and the five shapes the shipped catalog is actually built from — the second
  half mattering as much as the first, since a tightened vocabulary is exactly the kind of change
  that quietly rejects the models already in the product.

  One thing the tests get wrong easily: a refusal test whose pattern is loose enough to match
  `require()`'s "could not determine \<field\> from config.json" passes whether or not the guard
  exists. The headline Qwen3-Next test shipped with `/could not|declares|refus/i` and would have
  stayed green with `refuseLinearStack` deleted. Match the guard's own wording, and read the
  before-figures out of what `deriveAttention` returns for the same fields with the hybrid keys
  removed — arithmetic on literals beside a refusal is documentation, not a test.

- **Three list orders were deliberate, load-bearing and stated nowhere**
  ([#79](https://github.com/MrZoller/bench/issues/79)). Nothing sorts `devices.json`, `QUANTS` or
  `RUNTIMES` — every surface renders the file as written — so the order was a convention living only
  in whoever last edited it. That is a silent-breakage shape rather than a cosmetic one: a plausible
  "tidy the list" edit regroups it against itself and no test, type or review notices. The order is
  now stated in `CLAUDE.md` and in each file's docblock, and split by whether a machine can check it.
  `catalog.test.ts` enforces what is structural — rows grouped by `class`, a vendor's rows contiguous
  within a class — and `$comment-order` carries what is not.

  **The unenforceable half is the half worth writing down.** Within a vendor the rule is
  newest-generation-leading and largest-bin-first _inside a tiered ladder_, but newest-released-first
  where a product line is not a ladder — which is why the datacenter GPUs read B200, H200, L40S,
  H100, A100 rather than grouping Hopper together. No assertion distinguishes that from a mistake, so
  a test asserting it would be asserting a snapshot. Stating it is the only available guard.

  **Grouping is rendered by adjacency, so the component cannot own the order.** `<optgroup>` is a run
  of contiguous options, so the picker labels each option with its group and splits on _change_ of
  label. A list whose groups are interleaved therefore renders as two `<optgroup>`s carrying the same
  heading instead of being tidied into one — deliberately, because the alternative is filtering the
  list three times, which would silently make the component the authority on an order `devices.json`
  is supposed to state. The honest rendering makes a broken order visible; the tidy one hides it.

- **`activeParams` is the _published_ convention, not the physical one — and on a dense row it is
  simply `totalParams`.** `publishedActiveParams` returns the total outright when there is no MoE
  derivation, so the embedding is subtracted only on the MoE branch, where the dense residual is
  rebuilt and the routed share added back. It is what reconciles every derived figure with its
  vendor's, and it is the wrong basis for decode wherever the two conventions **exclude different
  things**. Three ways they can, and the third is the one two drafts of this paragraph missed:

  - a **non-language tower**, which the published figure includes and a token never touches;
  - an **untied input embedding** on a dense row, which the published figure includes — there it is
    simply `totalParams` — and decode never reads;
  - a **tied input embedding on an MoE**, which the published figure _subtracts_ (the MoE branch does
    so unconditionally) and `activeDenseParams` correctly keeps, because a tied table is the output
    projection and is read in full every step.

  That last one is live: Command A+ is a tied multimodal MoE where the omitted 1.074B table outweighs
  the included 0.495B tower, which is the whole of why its published figure is 0.578B _low_ where
  Mistral Small 4's is 7% high. And it means a future tied text-only MoE would disagree with no tower
  involved at all — so the derivation is `nonLanguageParams` and `tiedEmbeddings` **crossed with
  whether the row is an MoE**, not either field alone.

  Everywhere else they agree to the digit, and that list is longer than it looks: every _untied_
  text-only MoE — gpt-oss, the Qwen3 MoEs, Mixtral, GLM, the DeepSeek rows — and every tied text-only
  dense row. Stated this way round because the invariant a maintainer needs is _when to distrust the
  published figure_, and "every MoE" would have them replacing correct values.

  As the catalog stands, a snapshot of the rule rather than a list to trust: **Command A+** (tied MoE
  plus tower), **Mistral Small 4** (untied MoE plus tower), **every Gemma 3 row and Ministral 3 3B**
  (tied dense plus tower — Ministral publishes 3.849B against a 3.429B basis on a 0.420B encoder, and
  it is not a Gemma, which is how it went missing from an earlier draft), and **the untied dense
  rows** on the embedding.
  The engine reads `activeDenseParams`:
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

Correctness follow-ups live in
[the repository's open issues](https://github.com/MrZoller/bench/issues). This section is for the
questions those issues cannot settle, and the table below is the record of the six filed out of the
July sweep — five of them now closed, kept because what each one turned out to need is not what the
issue said it would.

The pointer names the open issues rather than a range on purpose. It read "#12–#20" while six newer
correctness issues sat in the table underneath it, so a maintainer following the sentence walked past
exactly the deferred work this section exists to preserve — and the same sentence called #13, #14,
#15 and #19 open when the Status section four hundred lines up records all four as fixed. A range
goes stale in both directions and silently in both.

### The six filed out of the July 2026 sweep, and what settled five of them

Six findings were triaged, replied to and filed rather than patched, under the merge rule recorded
above. Each was real, each touched something shared enough that fixing it inside another PR would
have been the third patch at one root cause, and each issue carried its numbers so they were not
re-derived. **Five are fixed; the sixth is in flight.** What each one turned out to need is the part
worth keeping, because in four of the five the filed framing was not quite the fix:

| filed                                                                                               | what it needed                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [#90](https://github.com/MrZoller/bench/issues/90) cpu-ram compute basis, catalogued/peak 0.41–1.12 | **Fixed.** The theoretical vector peak, since `computeEfficiency` already owns the discount — the same #51 rule on the compute axis, and the five rows were being discounted twice. The open question resolved against the wider figure: llama.cpp accumulates in fp32 lanes, so AVX512-FP16 is not the rate. Decode is bandwidth-only, so the EPYC anchor is untouched at 5.94 tok/s; prefill moves up to 2.5×. |
| [#96](https://github.com/MrZoller/bench/issues/96) serving graded at the slider's concurrency       | **Fixed**, and the ungraded state went with it: once each tier is graded at its own user count the question is always asked, so `Fitness` has three arms. The narrow capacity symptom was the least of it — the sentence quoting `runnableContextTokens` at the reader's concurrency disproved its own verdict on the two rigs the issue named.                                                                  |
| [#97](https://github.com/MrZoller/bench/issues/97) TTFT ramp collapses                              | **Fixed.** 29 of 46 → 4,5,6,7,12,9,3 on the Envelope; 1,025 of 1,269 → 29,57,86,263,323,304,207 on the Matrix. The domain's floor was half of it: a zero anchor is right only where zero is a reading, and no cell answers in zero seconds. Mirroring the _placement_ rather than the ramp index was off by one bucket at every boundary.                                                                        |
| [#101](https://github.com/MrZoller/bench/issues/101) the unit suite                                 | **Fixed** by rendering fewer cells: 155s → 24s locally, and CI's build job 7m9s → 1m38s. The property the issue asked for is asserted rather than hoped for — `App.test.tsx` renders a fixed twelve cells whatever the catalog does next. The timeout stayed at 30s deliberately, with the runner ratio written down.                                                                                            |
| [#102](https://github.com/MrZoller/bench/issues/102) readout on touch, and at 200%                  | **In flight.** The issue's cheapest option does not exist — this panel has no table behind a disclosure, it _is_ the table. Two taps instead, keyed on whether the reader has been shown the figures rather than on a pointer type, which a contact-only stylus breaks. 280px against 160px reserved at 320px/200%, fixed by dropping the model from the narrow form and keeping the machine.                    |
| [#103](https://github.com/MrZoller/bench/issues/103) `NOT_SEEDED` never revalidated                 | **Fixed** with the issue's own first step — an expiry, plus the structured reason it said to think about first. Three rounds went on one mistake: claiming a mechanical check made a cause exempt from the calendar. It never does. Every cause has a window now, and the checks are what make the windows long.                                                                                                 |

**The shape of that is the thing to keep, more than any of the six.** In four, the filed framing was
a correct description of the symptom and the wrong description of the fix — the cheapest option
rested on a panel that does not exist, the "narrow patch" was the smallest of three defects at one
root, the ramp's direction was half the problem and its floor the other half, and a written policy
needed the structure the issue named as an afterthought. Filing them with their measurements is what
made that visible; the measurements survived and the framings mostly did not.

- **MLX has no native quantization entries** ([#18](https://github.com/MrZoller/bench/issues/18)).
  Other catalogued formats stand in _by width_ — Q4_K_M's 4.85 bpw against MLX's ~4.5, and the
  non-GGUF INT8 at a flat 8.0 — so every figure for an
  Apple-silicon configuration derives from a format MLX does not load. The alternative, BF16 only,
  makes Apple silicon unusable in a tool where it is a headline case, so the
  substitution stays; it is entangled with the `weightFormats` check, whose MLX list includes those
  formats precisely so it keeps working.

  **What changed is that it is no longer invisible.** `substituted.nativeFormats` on the runtime
  names the formats it genuinely loads — everything else in `weightFormats` is a stand-in and is
  marked by default. The polarity is the point, and the reason to state it here: a format added to
  a runtime later is marked until someone says otherwise, where a list of the stand-ins themselves
  would leave that format silently unmarked. The Bench marks every figure derived from one; the
  Matrix marks a grid containing any. That is the rule `devices.json` already followed for
  pre-release specs, applied to the other kind of uncertain input: a documented approximation is a
  modelling choice, and an invisible one is invented data. Note the marker is deliberately narrow —
  BF16 is a real MLX format and carries none, because a warning on the majority case trains people
  to ignore it where it matters. `int8` was on that list and should not have been: MLX's 8-bit is
  affine like its 4-bit, while the catalogued `int8` row is LLM.int8() at a flat 8.0 bpw, aimed at
  vLLM. Leaving it native inverted the two 8-bit stand-ins — the marked `q8_0` at 8.5 reported
  13.7 GiB _heavier_ than the unmarked `int8` on a 235B, which is the asymmetry this exists to
  abolish. Resolving it properly needs a measured MLX width; see #18.

  **What would still resolve it** is measured bits-per-weight for MLX's affine 4- and 8-bit schemes,
  at which point the substitution is deleted rather than explained. That needs real checkpoints on
  Apple hardware; the marker is what makes the interim honest rather than what makes it right.

  **The cache was a second, independent substitution, and it hid behind the first** (#33, closed by
  extending the marker). `kvElementBytes` falls back to the nominal width when a runtime declares no
  `kvBytesPerElement` — exact for a float format, not for an affine one. MLX's `--kv-bits 8` is a
  real flag, so the entry in `kvPrecisions` is not a fiction; what is missing is its _width_, since
  affine quantization carries a scale and a bias per group. The cache is therefore charged exactly
  one byte per element, which understates it — in the direction that reports a long-context
  configuration fitting when it does not, which is the direction this repo cares most about.

  **The axes are independent in both directions**, which is why `substituted` names its cache list
  alongside `nativeFormats` instead of folding them together: MLX at Q4_K_M with an FP16 cache
  substitutes only weights, and MLX at BF16 with an 8-bit cache substitutes only the cache — and
  that second combination **carried no marker at all**, which is worse than the half-described
  state the issue was filed about.

  **Then the width turned out to be derivable, and the marker was the wrong answer** (#38). The
  first pass concluded it needed "a real checkpoint on Apple hardware, which nobody here has" —
  wrong for the same reason the `--default-font-size` claim below was wrong: a negative result
  about a tool, written down without checking the second place. `mlx-lm`'s source states it.
  `QuantizedKVCache(group_size=64, bits=8)`, and `mx.quantize` returns packed data, a scale, **and
  a bias** — the last two at `keys.dtype`:

  ```
  8 + 16/64 + 16/64 = 8.5 bits = 17/16 bytes per element
  ```

  The same kind of derivation as llama.cpp's 34/32: published source, not hardware. It lands on
  exactly llama.cpp's figure by coincidence — one fp16 scale per 32 elements versus a scale _and_ a
  bias per 64 both come to half a bit — and that coincidence has no test, because `17/16 === 34/32`
  at runtime. Do not merge them into a shared constant.

  **The threshold is all-or-nothing, and the first draft of this paragraph got it backwards.**
  `--quantized-kv-start` defaults to 5000 on the CLI, which reads like "the first 5,000 tokens stay
  fp16 and the tail is quantized" — it is not. `to_quantized` quantizes the _whole_ array at once,
  so crossing the threshold converts everything. The figure is therefore **exact above 5,000
  tokens** rather than approximate; below it the cache is entirely fp16 and costs 2 bytes per
  element, so a short-context Apple configuration is under-charged ~1.9x. Not modelled, because the
  error is proportionally largest exactly where the cache is smallest, and every context this tool
  is interesting at is past 5,000. Caught in review — the wrong version was a plausible reading of
  a flag name, asserted without checking the function it names.

  **And resolving it forced the contract question this always carried** (#45). `nativeKvPrecisions`
  asked whether a precision was stored at its _nominal_ width, which was the same question as "is
  it known" only by accident — every non-nominal width in the catalog also happened to be
  unmeasured. MLX's 8.5 bits is not nominal, so under the old predicate it could never be listed,
  and the app would have gone on warning that a derived figure rested on a guess. It is
  `measuredKvPrecisions` now, and the invariant that came with the rename is the part worth
  keeping: **a precision listed as established whose real width is not nominal must also carry that
  width**, or the marker goes quiet while the arithmetic stays wrong — worse than either alone.

  The consequence is that no shipped precision is marked any more, so the mechanism has no live
  trigger. It is not dead — it fires for the next precision added without a width. The polarity
  test drives that with a synthetic runtime, and the two surfaces that render it are held by a
  mocked `kvSubstitutionFor`, because an unreachable branch is one nobody notices breaking.

- ~~Codex connector coverage is unconfirmed.~~ **Confirmed working**, and characterised in
  **The post-release sweep** above, which is the current runbook and supersedes what this entry used
  to say. Two corrections, because the superseded version was actively misleading in both halves:

  - "Reviews arrive roughly 40 minutes after a push, so don't conclude the connector is missing from
    a quiet first half-hour" was patience described as evidence. Every clean verdict in the sweep
    arrived within about two minutes of a nudge, and one push sat 27 hours in silence on a healthy
    connector. **A quiet half-hour is a skipped push, not a queue position** — nudge once.
  - "It signals no findings with a 👍 reaction, so merge-readiness needs the reaction's `created_at`
    to postdate the head commit" is the right mechanism attached to the wrong reaction. 👍 is indeed
    the no-findings form, but `+1` also arrives on pushes that were never reviewed at all, and 👀 is
    the only reaction meaning a run is in flight. **Ask whether _this head_ was reviewed**, against
    both the reviews API and the issue comments, comparing the SHA as a prefix — the no-findings
    form abbreviates it to ten characters. A reaction alone does not answer that question.

- ~~`main` is unprotected.~~ **Enforced since 28 July 2026**, when the repo went public. The
  ruleset requires a pull request, squash merges only, both CI checks green, and every review
  thread resolved; deletion and force-push are blocked, with no bypass actors. What had been
  convention for the whole build is now the rule — including for whoever writes the next commit,
  who can no longer push to `main` even by accident.
- ~~Device specs need a verification pass before publishing.~~ **Done, 28 July 2026.** All 25 rows
  as they then stood — the catalog is 43 rows since #78 — checked against vendor documentation.
  Bandwidth — the number that governs everything — is
  confirmed on every one: the four CPU rows are exact by arithmetic (12 channels × DDR5-4800 × 8
  bytes is 460.8 GB/s to the digit, and the other three likewise), and the rest match their
  datasheets. **One real error, in MI355X:** its whole compute row was the air-cooled MI350X's
  (2300/4600/9200 dense), because the source pointed at the MI350 _family_ page. MI355X is the
  1400W liquid-cooled bin at 2400 MHz — 2500/5000/10000 — and the row now cites the part's own
  page. Same silicon and same memory, so nothing else moved.

  Three conventions worth writing down, since all three look like bugs and are not:

  - **A raiseable allocation ceiling states how far it raises, and it is never physical capacity**
    ([#53](https://github.com/MrZoller/bench/issues/53)). `allocatableTunable` and
    `maxAllocatableGiB` only mean anything together, and the pairing went unenforced: every Apple
    row declared the first and omitted the second, so `maxAllocatablePerDevice` fell back to
    capacity and all six claimed 100% of RAM could be wired to the GPU. The app offered the owner
    of a 96 GiB Mac Studio a 95.5 GiB configuration. The trap is that `iogpu.wired_limit_mb` really
    will _accept_ that value — what loads is bounded by what macOS needs to keep running, not by
    what the sysctl parses, and the distance between those two is the whole subject of the field.
    The Apple rows now reserve `max(8 GiB, 1/16 of RAM)` with the reason in each `note`; the
    reserve is a judgement rather than a datasheet figure, which is exactly why it is written down.
    `catalog.ts` refuses a tunable row that states no maximum or states one at capacity, and
    `maxAllocatablePerDevice` reads an absent value as "not raiseable" — under-promising rather
    than over-promising, which is the direction this class of error keeps failing in.
  - **Marketed HBM capacities run ~0.4% above true binary capacity.** H200's "141 GB" is 143,771
    MiB — 140.4 GiB — against a stored `capacityGiB: 141`; H100's "80 GB" is 79.65 GiB against 80.
    It does not reach the engine, because what the engine budgets against is `allocatableGiB`,
    which is below the true figure in every case (139 and 79). The headline stays as vendors quote
    it; the number that decides a fit is conservative.
  - **Bandwidth is theoretical peak, never measured.** Strix Halo's 256 GB/s is AMD's rating and
    real workloads see ~213. That gap belongs to `bandwidthEfficiency` and
    `CLASS_BANDWIDTH_UTILIZATION`, which exist to model it — folding it into the catalog would
    double-count it and quietly break the calibration anchors.

    **This rule was written down here and broken in the same week** ([#51](https://github.com/MrZoller/bench/issues/51)).
    The Ryzen row carried `measuredBandwidthGBs: 213` against its 256 rating, and
    `effectiveBandwidth()` preferred it — so the constants discounted an already-discounted
    figure and every Strix Halo throughput number read 16.8% under the treatment the other 24
    devices get, on the one surface whose purpose is ranking hardware against hardware. The field
    and `effectiveBandwidth()` are both gone now rather than deprecated: `types.ts` had a docblock
    arguing _for_ preferring measured, naming Strix Halo as the case for it, which is how a stated
    convention and the catalog came apart without either looking wrong on its own. A field is an
    invitation. `catalog.test.ts` now pins the convention itself rather than the ordering — the old
    check only asserted measured ≤ theoretical, which passes just as happily with the override
    present.

  The `rumored` row (M5 Ultra) is still press-rumour grade and must stay visibly labelled in the UI.

- ~~The catalog was accurate about 25 machines and silent about the ones the audience owns.~~
  **43 rows since [#78](https://github.com/MrZoller/bench/issues/78).** Coverage is a different
  property from accuracy and fails where accuracy checks cannot look: a machine that is absent is not
  wrong about anything. Three vendors' consumer lines and the whole sub-$1000 tier were missing — the
  cheapest catalogued GPU was the 5080 at $999, AMD appeared only as datacenter Instinct parts, Intel
  not at all, and every Apple row was a maxed configuration — so most "will it run" questions had no
  row to ask about rather than an incomplete answer. `catalog.test.ts` now asserts the _shape_ of the
  coverage (a sub-$350 GPU, consumer silicon from all three vendors, Intel in both classes it
  competes in, an Apple row at 16 GiB and one back at M1) rather than a list of ids, and one of those
  checks runs the engine: at least one machine under $400 has to genuinely hold a 12B at Q4_K_M and
  decode above 15 tok/s, since coverage by a row that cannot run the model is coverage in name only.
  **`some`, not `every`, and the distinction is the test's whole correctness:** that tier is where the
  next rows will be an 8 GiB 5060 or a 10 GiB Arc B570, which are honest rows that cannot hold a 12B
  at all, so a swept assertion would fail on exactly the coverage work it was written to protect.

  Four things worth keeping, since each was a decision rather than a transcription:

  - **Every vendor states compute in a different unit, so each row's `tflops` is a derivation** and
    the derivation is now written at the top of `devices.json` per vendor. Intel publishes a peak
    INT8 figure and no FP16 one, and XMX runs INT8 at exactly twice FP16 (4096 against 2048 ops per
    Xe-core per clock), so the Arc rows are half the stated TOPS. AMD publishes both an FP16 _matrix_
    and an FP16 _vector_ rate and the two differ by 2x on RDNA 4 — the matrix one is what a tensor
    kernel reaches, and quoting the other would halve prefill on the generation that doubled it.
    NVIDIA's single "AI TOPS" headline is sparse _and_ quoted at the lowest precision the generation
    reaches, so dense fp16 is the headline over 8 on Blackwell (sparse FP4: 3352 → 419 on a 5090) and
    over 4 on Ada (sparse FP8: 1321 → 330 on a 4090) — and the datacenter parts are transcribed, not
    derived, because A100's dense 312 fp16 is 16x its FP32 rather than the 4x that holds on GA10x.
    **The first draft of that paragraph stated all three NVIDIA clauses wrongly** while the rows were
    right, which is the #51 failure again: a curator applying the sentence to the next Blackwell row
    would have entered double the correct fp16, and no test could see it. The check that now exists
    pins the dense _ladder_ inside each row (int8 and fp8 at 2x fp16, fp4 at 4x); the divisor itself
    is only guarded by the worked examples beside it naming the headline they start from.
  - **The Apple allocation rules cross over below 32 GiB.** The default is Metal's recommended
    working set (two thirds of RAM at 32 GiB and below, three quarters above) and the ceiling is
    capacity minus `max(8 GiB, 1/16 of RAM)`, per #53. On a 24 GiB machine both land on 16 GiB, and on
    a 16 GiB machine the reserve would put the ceiling _below_ the default. Six maxed configurations
    all sat far above the crossover, so the rule looked universal. Those rows state no raiseable
    ceiling at all rather than one promising less than the machine already offers — #53's
    under-promising direction, applied at curation time instead of in `maxAllocatablePerDevice`. Both
    rules are swept over every Apple row rather than spot-checked, because the next small Mac added
    will hit the same crossover.
  - **A device id is in other people's links, so renaming a row is a compatibility change.**
    `rtx-a6000-ada` fused two products that both exist (the Ampere RTX A6000 and the Ada RTX 6000
    Ada Generation) while every spec on it was the Ada card's. `url.ts` writes `deviceId` as `d`, and
    the failure without an alias is not a broken link: `coerce` cannot resolve the id, falls back to
    the default device, and shows a stranger a DGX Spark's numbers under the sender's URL. Hence
    `DEVICE_ID_ALIASES` and `canonicalDeviceId`, which the **store** applies before it keeps the
    value — resolving only inside `getDevice` would load the right device while leaving the stale id
    to re-encode into the URL and match no `<option>` in the picker. Two things about it are worth
    the next session's attention. The lookup uses `Object.hasOwn` rather than `?? id`, because the id
    comes from a querystring and `DEVICE_ID_ALIASES['toString']` otherwise resolves up the prototype
    chain to a function. And **`d` is the only one of the four ids in a shared link that has a
    canonicaliser at all**: `url.ts` writes `m`, `q` and `r` too, so renaming a model, quant or
    runtime id degrades in exactly the silent way this entry is about. Latent rather than live —
    nothing in those three catalogs has been renamed — and recorded here so the next rename is not
    where it gets discovered.
  - **The unit suite is 2.5x slower** (45s to ~110s) and nothing is wrong. `App.test.tsx` renders the
    whole page per test and the Matrix is 17 models × 42 shipping devices, so ~700 buttons are built
    per render against the old ~400. It is the cost of the coverage, not a regression to chase.

  Deliberately not done: **GB200 and B300**. The issue lists them as lower priority, and the
  per-GPU dense FP4 figures do not reconcile across NVIDIA's own rack-level claims — a `rumored` row
  is for specs that are uncertain, not for arithmetic the curator cannot close. MI325X went in
  because AMD publishes it as a part.

- **Final subdomain** on zoller.ai. The only thing genuinely left, and it is a naming decision
  rather than work: the site is live at the Pages project URL, and moving it is two repository
  variables — `PAGES_CUSTOM_DOMAIN` to the chosen host and `PAGES_BASE_PATH` back to `/` — plus a
  CNAME record. Both variables have to change together, which is why they are documented as a pair
  above.

## Verification

```
npm test && npm run lint && npm run format:check && npm run build
npm run test:e2e                # Playwright; builds and serves on 127.0.0.1:4173 itself
npm run catalog -- --dry-run    # re-derive the model catalog without writing
```

The engine's reference tests are the spec. If one fails, the model is wrong — do not widen the
band to make it pass.
