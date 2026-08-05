import type { ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from '@/engine/types';
import type { Placement } from '@/engine/placement';
import { effectiveDeviceCount, effectivePromptTokens } from '@/engine/placement';
import { isSlidingLayer, layersCacheAlike } from '@/engine/kv';
import { substitutionFor } from '@/data/runtimes';

/**
 * The runnable command for a placement (#136).
 *
 * Everything here is a formatter over answers the engine has already committed to — the layer
 * assignment, the KV precision, the context, the shard count. What it is *not* is a second
 * derivation of any of them: `-ngl` reads `Placement.assignment.residentLayers`, `--max-model-len`
 * reads the same `contextTokens` the budget bar drew, and where a figure is not on the placement
 * it is not printed. A limit stated twice is a limit that will disagree with itself, and here the
 * disagreement would be copy-pasteable.
 *
 * ## The three refusals, and why a refusal is the feature
 *
 * A command is a claim that something can be run, so this module refuses more often than it emits:
 *
 *   1. **A configuration the engine turned away.** `unsupported` and `impossible` both describe a
 *      placement that does not exist, and a command for one is a command that OOMs on load.
 *   2. **A checkpoint the catalog cannot name.** `ModelSpec.id` is the *source* repo, and
 *      `QuantSpec` carries sizing and compute properties only — there is no GGUF, AWQ or MLX
 *      artifact anywhere in the data. So a Q4_K_M selection under vLLM has no repo to print, and
 *      naming the source checkpoint instead would be a command for a different model than the one
 *      the placement priced. See {@link artifactFor}.
 *   3. **A format the runtime does not actually load.** The #18 substitution: MLX's figures at
 *      Q4_K_M derive from a format MLX cannot read, so a command naming that checkpoint is a
 *      command for a file that does not exist. `substitutionFor` already knows; this refuses on it.
 *
 * ## Every flag was read from upstream, and each template says where
 *
 * "A command is a claim, and flags drift" is the trap #136 names, and every template below carries
 * a `source` and the date it was checked, in the posture `devices.json` takes for a device spec.
 * Four things in here are *not* what a from-memory implementation would have written, and each one
 * was found by reading the source rather than by recalling it:
 *
 *   - **llama.cpp's `-ngl` counts the output tensor.** `n_gpu_layers` is the repeating blocks plus
 *     one position, so a fully-resident 48-layer model wants `-ngl 49`, and `-ngl 48` sheds *layer
 *     0* rather than the output tensor — the window comes off the front of the stack. It is one
 *     rule and not a rule for resident placements: the slot exists for any positive `-ngl`, so a
 *     spilling placement of `N` layers is `-ngl N + 1` too. See {@link gpuLayers}.
 *   - **`-ts` is normalised over that same window, so its ratios have to sum to `-ngl`.** The
 *     cumulative shares are compared against `(il - i_gpu_start) / act_gpu_layers` and
 *     `act_gpu_layers` is `min(ngl, n_layer_all + 1)` — so ratios summing to the repeating layers
 *     alone are stretched across one slot more than they describe, and the first card silently
 *     gains a layer. The output slot rides on the last non-zero share, and is emitted as part of
 *     it. See {@link tensorSplit}.
 *   - **`-c` is the whole cache, divided among the `-np` slots.** `n_ctx_seq` is
 *     `n_ctx / n_seq_max` unless the KV buffer is unified, so eight users at 64K wants `-c 524288`.
 *     Passing the per-user figure would give each slot an eighth of it.
 *   - **Ollama's Modelfile has no `num_gpu` parameter.** The documented list is `num_ctx`,
 *     `num_predict` and the sampler knobs — so the layer split this feature exists to print is the
 *     one thing that surface cannot take, and the template says so rather than inventing a flag.
 *   - **`mlx_lm.server` has no KV quantization flag.** `--kv-bits` is on `mlx_lm.generate`; the
 *     server's argument list does not carry it, so an 8-bit-cache scenario cannot be served at the
 *     precision it was priced at.
 *   - **`-ot`/`--override-tensor` moves a layer's weights and not its cache**, which is why there
 *     is no per-layer placement flag here and why #166 closes with a sentence rather than a
 *     command. See {@link packingNotes} for the read.
 *
 * A fourth kind of refusal is deliberately not on the list of three, because it is a property of
 * the *surface* rather than of the placement: `mlx_lm.server` has no cache-precision flag, and
 * Ollama's Modelfile has no parallelism parameter. Each turns away its own serving form for a
 * scenario the panel still answers — a sibling launcher for Ollama, since `llama-server` takes both
 * as flags, and the launcher's own measurement form for MLX, since `mlx_lm.generate` does take
 * `--kv-bits`. See {@link mlx} and {@link ollama}.
 */

/** Where a launcher's flags were read from, and when. */
export interface Launcher {
  id: string;
  /** What the reader actually types, which is not always the runtime's own label. */
  label: string;
  runtimeId: string;
  source: string;
  /** The date the flags above were checked against `source`, ISO. */
  checkedOn: string;
}

/**
 * One command, or the reason there is none.
 *
 * A refusal is a first-class result rather than an absence, because the reason is the useful part:
 * "vLLM cannot be told which checkpoint to load, because the catalog has no AWQ repo for this
 * model" is what a reader needs, and an empty panel is not.
 */
export type Emission =
  | { readonly ok: true; readonly text: string; readonly notes: readonly string[] }
  | { readonly ok: false; readonly reason: string };

/**
 * A launcher's two forms.
 *
 * Both, because a server being up is not a measurement (#139): the serving command starts the
 * thing, and the measurement command is the one that reproduces the *priced workload* — the
 * scenario's prompt length, generation length and resident prefix in the flags of that runtime's
 * own client. `llama-bench` loads GGUF and cannot measure a vLLM or MLX placement, so the client
 * travels with its launcher rather than standing in for the others.
 */
export interface LauncherCommands {
  launcher: Launcher;
  serve: Emission;
  measure: Emission;
}

export interface LaunchInput {
  model: ModelSpec;
  quant: QuantSpec;
  runtime: RuntimeSpec;
  rig: Rig;
  usage: UsageSpec;
  placement: Placement;
}

/**
 * The checkpoint Headroom can actually name for a model at a format, or nothing.
 *
 * **The catalog knows exactly one artifact per model: its own repo, at its own checkpoint
 * format.** `nativeQuant` is the `quantization_config.quant_method` the generator read from
 * `config.json`, and its absence means the repo ships unquantized — which is the `bf16` row, whose
 * label is "BF16 / FP16" precisely because the catalog does not distinguish the two at 16 bits.
 *
 * Everything else is a *conversion* somebody else published: a Q4_K_M GGUF, an AWQ pack, an
 * `mlx-community` port. Those live in repos this catalog has never seen, and the failure mode of
 * guessing one is not a 404 — it is a plausible-looking command that loads a different model than
 * the placement priced. So the answer is `undefined` and the caller refuses.
 *
 * An unrecognised `nativeQuant` — a `quant_method` string with no matching `QuantSpec` id — makes
 * every format on that model unnameable rather than making the wrong one nameable, which is the
 * direction this whole module fails in.
 */
export function artifactFor(model: ModelSpec, quantId: string): string | undefined {
  return (model.nativeQuant ?? 'bf16') === quantId ? model.id : undefined;
}

/**
 * The commit every figure on the page was derived from.
 *
 * `build-catalog.ts` records `revision` on each row precisely so a suspicious number is
 * reproducible, and a command naming only the repo id resolves the *mutable default branch* — so
 * after an upstream push the copied command loads a checkpoint the displayed memory and speed
 * figures do not describe (raised by Codex on #164). vLLM takes `--revision`, which accepts a
 * commit id; MLX's server does not, and that is stated in its notes rather than papered over.
 */
export function revisionOf(model: ModelSpec): string | undefined {
  return model.revision;
}

/**
 * A scenario whose window has no room left to answer in.
 *
 * The measurement form refuses rather than manufacturing a token: the prompt slider goes up to the
 * whole context, and a `--output-len 1` beside `--max-model-len <context>` is a command that
 * exceeds its own stated limit.
 */
function noRoomToAnswer(contextTokens: number): string {
  return (
    `The prompt fills the whole ${contextTokens.toLocaleString('en-US')}-token window, so there is ` +
    `no room left to generate into and nothing to measure. Lower the prompt length above and the ` +
    `benchmark command comes back.`
  );
}

/** The sentence a refused artifact gets, naming what would have to exist. */
function noArtifact(model: ModelSpec, quant: QuantSpec, launcher: string): string {
  const native = model.nativeQuant ?? 'bf16';
  return (
    `Headroom has no ${quant.label} checkpoint to name for ${model.name}. The catalog carries the ` +
    `source repo — ${model.id}, which ships at ${native.toUpperCase()} — and no per-format ` +
    `artifact, so ${launcher} would have to be pointed at a conversion published elsewhere. ` +
    `Naming the source repo here would start a different model than the one priced above.`
  );
}

/** llama.cpp's `--cache-type-k`/`-v` names for the precisions the catalog offers. */
const LLAMA_KV_TYPE = { fp16: 'f16', q8: 'q8_0', q4: 'q4_0' } as const;
/** vLLM's `--kv-cache-dtype` values. `auto` is "the model's own dtype", which is what fp16 means. */
const VLLM_KV_DTYPE = { fp16: 'auto', q8: 'fp8', q4: undefined } as const;

/**
 * `-ngl`, one rule: the resident layer count plus the output tensor's slot.
 *
 * Two corrections sit between `assignment.residentLayers` and the flag, and the engine declines
 * both because it has no notion of a flag:
 *
 *   - **A CPU-only rig offloads nothing.** `residentLayers` reports every layer, truthfully —
 *     nothing spilled, because there was nowhere to spill *from*. The honest flag is 0.
 *   - **llama.cpp counts the output tensor one position past the repeating blocks.**
 *     `n_gpu_layers` defaults to `n_layer_all + 1`, and `i_gpu_start = max(n_layer_all + 1 - ngl, 0)`
 *     — so "all of them" is `layers + 1`, and `-ngl 48` on a 48-layer model is one short: it sheds
 *     **layer 0**, at the front of the stack, and keeps the output tensor. The output tensor is slot
 *     `n_layer_all`, and `il - i_gpu_start = ngl - 1 < act_gpu_layers = ngl` holds there for every
 *     `ngl >= 1`, so it is on a GPU whenever anything is; upstream's own fitter says "the last device
 *     has the output layer, which cannot be a partial layer". Read 5 August 2026 from
 *     `src/llama-model.cpp:1318-1343` and `common/fit.cpp:581` at ggml-org/llama.cpp commit
 *     `360e134` (#202).
 *
 * **The `+ 1` used to be applied only where nothing spilled**, and that was #204's first defect: a
 * spilling `-ngl N` loaded `N - 1` repeating layers plus the table rather than the `N` the panel
 * priced. Measured over 361,200 catalog configurations (35 models x llama.cpp's eight weight
 * formats x 43 devices x {1,2,4,5,8} cards x {8K,32K,128K} x {1,4} users), 218,334 of which emit a
 * command: **56,719 of them were off by one, and 1,537 were `-ngl 1` loading no repeating layer at
 * all.** `Math.min(residentLayers, model.layers) + 1` collapses the two branches into the one rule
 * upstream actually has. The clamp is defensive — under a layer split `residentLayers` is a sum
 * bounded by the model's own count — and it is what keeps the flag from claiming a slot that does
 * not exist if that ever stops holding.
 *
 * **Zero is deliberately still zero, and it is the one case #204 left open.** Where
 * `residentLayers` floors to zero on a GPU rig the rule above would emit `-ngl 1`, which puts the
 * whole output table on the card and no repeating layer. Whether that is honest varies across the
 * range it fires on: `residentLayers` floors, so it reaches zero while a real fraction of the
 * weights is still resident — 74.6% to 99.9% of the rig spilling across the 5,807 configurations
 * that reach it in the sweep above, and never 100%. Near the low end the resident remainder is
 * roughly the fixed block and the table is close to the right thing to keep; near the high end
 * nothing meaningful is resident and offloading the whole table over-offloads relative to what was
 * priced. `placementRefusal` already turns away 17,464 of the 23,271 zero-resident GPU
 * configurations, so the open question is where that boundary belongs rather than whether to build
 * one — and it is not answered by an off-by-one fix. Left as it was so this change moves only what
 * it verified.
 */
function gpuLayers(input: LaunchInput): number {
  if (input.rig.device.class === 'cpu-ram') return 0;
  const { residentLayers } = input.placement.assignment;
  if (residentLayers <= 0) return 0;
  return Math.min(residentLayers, input.model.layers) + 1;
}

/**
 * A `-ts` split in its two readings, which differ by exactly the output tensor's slot.
 *
 * They are carried together rather than derived twice because both are printed: the flag needs the
 * window's proportions and the sentence beside it needs the layer counts the panel actually sized,
 * and a note that recomputed either from the other is a second derivation of the thing this module
 * exists not to derive twice.
 */
interface TensorSplit {
  /** Repeating layers per card, exactly as `layerSplitBins` packed them. */
  readonly sized: readonly number[];
  /** What `-ts` carries: {@link sized} with the output tensor's slot on the last non-zero share. */
  readonly ratios: readonly number[];
}

/**
 * `-ts`, when the packing actually produced an uneven split.
 *
 * This is the one flag that exists *because* the assignment was surfaced. llama.cpp's default split
 * is proportional to each device's memory, which on a homogeneous rig is an equal number of layers —
 * and on a hybrid model that is the wrong split, because one full-attention layer caches up to
 * ~128x what a sliding one does. `layerSplitBins` balances the combined load instead, and on Gemma
 * 3 12B at 128K over five cards its layer counts land 19 apart.
 *
 * **The counts are the *resident* ones, not the assigned ones, and the first version of this
 * emitted the assigned ones** — which is the trap #136 names, reached from the far side. `-ts` does
 * not distribute the model; it distributes the `-ngl` window, because llama.cpp puts the last `ngl`
 * layers on GPUs and splits *those* by these proportions. So `-ngl` and `-ts` are read together,
 * and giving them counts from two different scopes lets llama.cpp re-derive a per-device split that
 * is neither.
 *
 * It is reachable and it OOMs. Ministral 3 3B at Q8_0, 131,072 tokens over 4 users on four RTX
 * 5080s packs 7,7,6,6 layers and keeps 2,2,6,6 of them resident — so `-ngl 16 -ts 7,7,6,6` asks
 * llama.cpp to spread sixteen layers slightly-in-favour-of the two cards Headroom sized for two, which
 * are the constrained cards precisely because their cache already fills them. `2,2,6,6` is the
 * split that was actually sized.
 *
 * Where nothing spills the two are identical, so this changes only the case it exists for.
 *
 * ## The window has one more slot than the split describes, and that is what {@link ratios} is for
 *
 * `-ts` proportions the `-ngl` window, and the window is `act_gpu_layers = min(ngl, L + 1)` slots
 * — the repeating layers **and the output tensor's own position**. llama.cpp normalises the ratios
 * by their own sum and then compares the cumulative shares against `(il - i_gpu_start) /
 * act_gpu_layers`, so ratios summing to `L` are stretched across `L + 1` positions and device 0's
 * boundary lands at `c0/L` against a key stepping in `1/(L + 1)`: **the first card gains a slot
 * whenever its cumulative share is at least one.** Worked, and confirmed against a port of
 * `llama-model.cpp:1285-1343` at commit `360e134`: 32 layers, `-ngl 33 -ts 7,7,6,6,6` delivers
 * `8,7,6,6,5` with the output on the last card. Every one of the 42,037 `-ts`-emitting
 * configurations in the sweep {@link gpuLayers} describes was wrong this way — 100% of them, and
 * the first card gained the slot on 39,736; the rest lead with a zero share, which cannot gain.
 *
 * **The correction is to emit the output's slot rather than to leave it implicit**: `+1` on the
 * last non-zero share, which is the one llama.cpp puts the output on, and `-ngl` reads
 * `residentLayers + 1` from {@link gpuLayers}. The two then agree by construction — the ratios sum
 * to `-ngl` — and the arithmetic collapses: with `act_gpu_layers` equal to the ratio sum, key
 * `k / sum` against boundary `c_i / sum` assigns slot `k` to the first device with `c_i > k`, so
 * device `i` gets exactly `ratio_i` slots and the last non-zero one spends its extra slot on the
 * output. Exact on all 42,037, output on the intended card every time (#204).
 *
 * Emitted only when the counts really differ, so the flag never appears saying "split this evenly",
 * which is what llama.cpp would have done unaided — and that gate reads the resident counts too,
 * since a packing that assigns equal counts of *unequal* layers is exactly where the flag is needed.
 *
 * **That gate is measurably too tight and #204 deliberately did not widen it.** An even *resident*
 * split is not an even *window*: `L + 1` slots over `n` identical cards cannot divide evenly when
 * `n` divides `L`, so llama.cpp's memory-proportional default hands card 0 an extra layer and takes
 * one off the card holding the output. The same sweep finds 89,615 configurations that suppress the
 * flag on this gate and **not one** where the default reproduces what was packed. Widening it would
 * put a `-ts` on 89,615 commands that carry none today, which is a larger rewrite than the one
 * verified here and interacts with #182's packing change; it is reported rather than taken.
 *
 * **And never where the layers cache different amounts, which is the case the flag looked most
 * useful for** (raised by Codex on #164, P1). `-ts` partitions llama.cpp's ordered `-ngl` suffix
 * into contiguous device ranges; `layerSplitBins` assigns *individual* layers by greedy combined
 * load, so a share can be a non-contiguous mixture of full-attention and sliding layers. Where
 * those cache ~128x differently, equal counts do not reproduce equal loads — llama.cpp would hand a
 * card a different set of expensive layers than `planPlacement` priced, and the fit the panel
 * reported would not be the fit the command produces. A count is a faithful description of the
 * packing only where the layers are interchangeable, so that is the only place it is emitted.
 *
 * **The gate is `layersCacheAlike`, not `hasSlidingLayers`, and the difference is a scenario the
 * model-level question gets wrong.** Being hybrid is a property of the model; caching *unequal
 * amounts* is a property of the context. Below its shortest window a hybrid model's layers all hold
 * the whole context, so the packing hands out equal loads and the flag is exact — Gemma at a
 * 1,024-token context was refused the flag it could have had, and told why in terms of an imbalance
 * that does not exist there.
 *
 * What remains is still worth having: a model whose layers cache alike, over a device count that
 * does not divide it (48 layers over 5 cards is 10,10,10,9,9), is both uneven and exactly
 * expressible, and llama.cpp's memory-proportional default gets it wrong.
 */
function tensorSplit(input: LaunchInput): TensorSplit | undefined {
  const { parallelism, shares } = input.placement.assignment;
  if (parallelism !== 'layer' || effectiveDeviceCount(input.rig) <= 1) return undefined;
  if (!layersCacheAlike(input.model, input.usage.contextTokens)) return undefined;

  const sized = shares.flatMap((s) =>
    Array.from({ length: s.deviceCount }, () => s.residentLayers)
  );
  if (sized.length <= 1) return undefined;
  // Nothing is on a GPU at all, so there is no window to proportion. `-ngl 0` already says it.
  if (sized.every((c) => c === 0)) return undefined;
  if (Math.max(...sized) === Math.min(...sized)) return undefined;

  // The card llama.cpp will put the output tensor on: the last one with a share of the window.
  // A trailing zero is legal and gets nothing, so it is the last *non-zero* share and not the
  // last entry — and it is the share that has to carry the extra slot, since asking any other
  // card for it would move the table as well as the layer.
  const output = sized.reduce((last, c, i) => (c > 0 ? i : last), -1);
  return { sized, ratios: sized.map((c, i) => (i === output ? c + 1 : c)) };
}

/**
 * What Headroom packed, said out loud on the rigs `-ts` has to refuse (#166).
 *
 * `tensorSplit` declines wherever the layers do not cache alike, because a count is a faithful
 * description of a greedy per-layer packing only where the layers are interchangeable. Declining
 * is right and it is not the whole answer: leaving the flag off does not make llama.cpp reproduce
 * the assignment, it makes llama.cpp pick a different one — so on precisely the scenarios where the
 * packing is worth the most, the panel computed a better split than it could express and described
 * it as "more than the panel above shows".
 *
 * It can now say which layers, because `DeviceShare.layerIndices` carries them. That is the whole
 * of what #166 asked the engine for, and it is what makes these two sentences quantitative: the
 * counts and the *composition* are two different lists, and the panel had only ever been able to
 * print the one that does not carry the load.
 *
 * ## Why there is no flag to emit instead, which is the part that was worth reading upstream for
 *
 * #166 hoped for `-ot`/`--override-tensor`, since it takes a pattern and a buffer type and can
 * name individual tensors — so a per-layer override list looks expressible in principle. It is
 * not, and the reason is one level below the flag:
 *
 *   - `llama_model::load_tensors` computes `dev_layer[il]` from `n_gpu_layers` and `tensor_split`
 *     alone — `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`, then an `upper_bound` over
 *     the normalised splits — **before** any override is consulted.
 *   - The overrides are applied later, in `llama_model_loader::create_tensor`, by
 *     `std::regex_search` against a *tensor* name. They change where a weight lives and nothing
 *     else.
 *   - `llama_kv_cache`'s constructor takes each layer's cache buffer from
 *     `ggml_backend_dev_buffer_type(model.dev_layer(il))`.
 *
 * So `-ot` would move a layer's weights to the card Headroom chose and leave its KV cache on the card
 * `-ngl`/`-ts` chose. On a hybrid model the cache is the entire reason the packing is uneven — the
 * per-layer weights are uniform and a full-attention layer caches up to ~128x a sliding one at
 * 128K — so the flag moves the half that does not vary and leaves the half that does. A command
 * built from it would start a placement other than the one the panel priced, which is the first of
 * this module's three refusals rather than a caveat to print beside it.
 *
 * Read on 2 August 2026 from `src/llama-model.cpp`, `src/llama-model-loader.cpp` and
 * `src/llama-kv-cache.cpp` at ggml-org/llama.cpp master.
 */
function packingNotes(input: LaunchInput): readonly string[] {
  const { model, rig, usage, placement } = input;
  const { parallelism, shares } = placement.assignment;
  if (parallelism !== 'layer' || effectiveDeviceCount(rig) <= 1) return [];
  // The same predicate `tensorSplit` gates on, and it has to be the same one: these sentences
  // explain a refusal, so a scenario where the flag is exact is a scenario where they have nothing
  // to explain. Under `hasSlidingLayers` they fired on Gemma at a 1,024-token context — beside a
  // `-ts` that had just become emittable — and attributed the absent flag to an imbalance that
  // does not exist below the model's shortest window.
  if (layersCacheAlike(model, usage.contextTokens)) return [];
  /**
   * **Nothing is on a GPU, so there is no split to describe** — the guard `tensorSplit` has in the
   * function directly above and this one was written without, which is this repo's own N+1 pattern
   * arriving inside the sweep for it. `gpuLayers` rather than a second reading of the shares, so the note
   * and the flag beside it cannot disagree about whether anything was offloaded at all: it is 0 on
   * a `cpu-ram` rig and 0 when the whole rig spilled. gpt-oss 120B at BF16 on two 4090s at 128K
   * over 8 users is the reachable case — 96% of the weights spill, `18,18` layers packed and none
   * of them resident, so `-ngl 0` was emitted under two sentences about how llama.cpp would divide
   * the cards' layers between them.
   */
  if (gpuLayers(input) === 0) return [];

  const perDevice = shares.flatMap((s) => Array.from({ length: s.deviceCount }, () => s));
  if (perDevice.length <= 1) return [];

  // The assignment, not the resident subset of it. `residentLayers` is what a *flag* would carry
  // and these sentences are describing what was packed, which is the same list only where nothing
  // spills — that is to say, on most rigs, which is what makes the two easy to confuse.
  const counts = perDevice.map((s) => s.layers);
  /**
   * The layers with no window, which are the ones whose cache keeps growing with the context.
   *
   * A count rather than a per-layer cache figure, which is faithful only because a model has at
   * most one bounded window size: every sliding layer then holds the same amount, so the count of
   * unbounded ones fixes the rest of the split. Two sizes at a context between them would make
   * this understate a card holding the wider ones. That invariant is pinned at both ends of the
   * catalog pipeline — `assertOneBoundedWindow` in `scripts/build-catalog.ts` refuses to derive a
   * second size, and `catalog.test.ts` asserts the shipped rows carry one — so a future
   * architecture that breaks it fails there rather than quietly widening the error here.
   */
  const unbounded = perDevice.map(
    (s) => s.layerIndices.filter((layer) => !isSlidingLayer(model, layer)).length
  );

  /**
   * Two statements of fact and no prediction, which took two rewrites.
   *
   * The obvious first sentence — "the second list is why the first is uneven" — is false on a
   * hybrid model at a context inside its windows; that is now unreachable, since the gate above
   * returns before it, but the wording does not depend on the gate either.
   *
   * The obvious second one — "plan for that card to hold more than the panel above shows" —
   * predicts a comparison Headroom has not made. llama.cpp's contiguous split sometimes lands the same
   * composition the packing did (Gemma 3 12B on two 5090s at 128K packs `24,24` against `4,4`, and
   * so does an even contiguous halving), and modelling `upper_bound` over the normalised splits to
   * find out would be this module deriving llama.cpp's placement rather than formatting Headroom's.
   * What is left is what Headroom actually knows: it packed for a light busiest card and llama.cpp is
   * not packing for that at all, so the panel's figure is a floor to plan against rather than an
   * estimate of what the command will produce.
   */
  return [
    `Headroom packed ${counts.join(',')} layers onto the ${perDevice.length} cards, ` +
      `${unbounded.join(',')} of them attending over the whole context rather than a fixed ` +
      `window. It packed by cache weight rather than by layer count, so those two lists together — ` +
      `not the first one alone — are what the memory panel above priced.`,
    `llama.cpp cannot be given that assignment. -ts proportions a contiguous run of the -ngl ` +
      `window and this packing is a non-contiguous mixture; -ot names individual tensors, but it ` +
      `overrides where a weight lives, and a layer's KV cache follows the device -ngl and -ts put ` +
      `the layer on. So llama.cpp will divide by device memory instead — an equal number of layers ` +
      `on identical cards — and which layers land together is then its choice rather than Headroom's. ` +
      `Treat the busiest card above as a floor: Headroom packed to keep it as light as it could, and ` +
      `llama.cpp is not packing for that at all.`,
  ];
}

/**
 * The generation the scenario leaves room for — or nothing, when it leaves none.
 *
 * The window minus the prompt **and the resident prefix**, both of which occupy it. The first
 * version subtracted only the prompt and floored at 1 (raised by Codex on #164): a prompt slider at
 * the full window then produced `--input-len <context> --output-len 1 --max-model-len <context>`, a
 * command that exceeds its own stated limit, and a 47,616-token prefix under a 16,384-token prompt
 * in a 65,536-token window was handed another 49,152 tokens of output it has nowhere to put.
 *
 * `undefined` rather than a manufactured token: a scenario with no room to answer is one the
 * measurement form has to refuse, not one it can round into existence.
 */
function generationTokens(usage: UsageSpec): number | undefined {
  const room = usage.contextTokens - effectivePromptTokens(usage) - (usage.cachedPrefixTokens ?? 0);
  return room > 0 ? room : undefined;
}

/**
 * llama-bench's own default `-n`, and the ceiling that keeps it inside calibrate's tolerance.
 *
 * See {@link decodeBenchSpan}. A sixteenth rather than the tenth the tolerance itself allows,
 * because a power of two is exact in binary and a tenth is not — this repo already records a
 * threshold missed by float epsilon at the calibration band's edge — and because the depth is not
 * quite the quantity the run averages over.
 */
const BENCH_GEN_TOKENS = 128;
const BENCH_GEN_WINDOW_DIVISOR = 16;

/**
 * Where the decode benchmark measures from, and for how long: a **short** generation at the top of
 * the window.
 *
 * `estimateDecode` charges every step's cache read at `usage.contextTokens` — the whole window, not
 * the prompt — and that is right on this app's own terms. Context here is "prompt plus everything
 * generated so far" (`SETTING_NOTES.contextTokens`): a session fills its window across turns and
 * spends almost every token it emits near the top of it. So the state to reproduce is a nearly-full
 * cache, and `-d` is the flag that produces it.
 *
 * **The length is not the interesting half, and asking for the window's whole remainder was the
 * defect** (#180). Decode is a steady-state per-token rate — the same fact `describeMismatch` relies
 * on when it declines to check a decode run's length at all — so how many tokens you ask for does
 * not move the number, while the cache each of them reads does. The first version handed `-n` the
 * entire remainder: 24,576 tokens on the default scenario, which at the predicted 35.6 tok/s is 11.5
 * minutes of generation per repetition and llama-bench repeats five times by default. It also put
 * the cache at `prompt + prefix` — 8,192 against a figure charged at 32,768 — so the row the panel
 * told a reader to produce was one `compare` rejected.
 *
 * **What that cost the calibration record was decode points rather than correct ones**, and the two
 * are worth telling apart: `submissionUrl` writes only the pairs `compare` did *not* reject, so a
 * biased decode row could never have reached the corpus that feeds the next retune of
 * `bandwidthEfficiency`. It is the rows that got through that were the problem. `prompt + prefix`
 * lands inside the tolerance exactly when the prompt already nearly fills the window, so the only
 * scenarios the panel's own path could contribute a decode measurement from were the ones with
 * almost nothing left to generate — a narrow slice, selected on the axis being calibrated, and
 * shallow within it.
 *
 * **`gen` is llama-bench's own default of 128, held to a sixteenth of the window**, and the ceiling
 * is what makes the emitted command acceptable rather than merely quick. `describeMismatch` marks a
 * decode row whose depth is more than 10% off the context the prediction charges, and the depth here
 * is `contextTokens - gen`, so `gen` has to stay inside a tenth of the window at *every* context a
 * reader can reach — not just the default. The fixed stops start at 2,048 and 128/2,048 is exactly a
 * sixteenth, so the clamp never binds at a stop and 128 is what an ordinary scenario measures; below
 * the smallest stop — which no control offers, and which a hand-edited link reaches down to the 512
 * floor `coerce` clamps at — it keeps the ratio instead, since a flat 128 there is a quarter of the
 * window and would be rejected.
 *
 * The relation is a ratio and not a shared constant, which is why this returns a pair rather than
 * exporting a number for both sides to use: the cache grows as the run generates, so the depth the
 * measurement actually averages over is `contextTokens - gen / 2` — inside the ceiling by another
 * factor of two, and still not the `contextTokens` the engine charges. `calibrate.test.ts` pins that
 * both of them clear the tolerance at every stop.
 */
export function decodeBenchSpan(usage: UsageSpec): { depth: number; gen: number } {
  const context = Math.max(1, usage.contextTokens);
  const gen = Math.max(
    1,
    Math.min(BENCH_GEN_TOKENS, Math.floor(context / BENCH_GEN_WINDOW_DIVISOR))
  );
  return { depth: Math.max(0, context - gen), gen };
}

/** A shell command written one flag per line, which is how anyone would paste it back. */
function shell(head: string, args: readonly (string | undefined)[]): string {
  const kept = args.filter((a): a is string => a !== undefined);
  if (kept.length === 0) return head;
  return [head, ...kept].join(' \\\n  ');
}

/**
 * The local file the reader downloaded, which is theirs to name.
 *
 * A placeholder is honest for exactly this and for nothing else. `-m` takes a path on the reader's
 * own disk, so no catalog could supply it and inventing one would be noise; the *artifact* is the
 * opposite case, where a made-up value looks like a working answer. Angle brackets rather than a
 * plausible-looking path, so pasting it unedited fails in the shell instead of half-working.
 */
function ggufPlaceholder(model: ModelSpec, quant: QuantSpec): string {
  return `<path to your ${model.name} ${quant.label} .gguf>`;
}

const LLAMA_SERVER: Launcher = {
  id: 'llama-server',
  label: 'llama.cpp (llama-server)',
  runtimeId: 'llama.cpp',
  source: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md',
  checkedOn: '2026-08-01',
};

const LLAMA_BENCH: Launcher = {
  id: 'llama-bench',
  label: 'llama.cpp (llama-bench)',
  runtimeId: 'llama.cpp',
  source: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md',
  checkedOn: '2026-08-01',
};

const OLLAMA: Launcher = {
  id: 'ollama',
  label: 'Ollama',
  runtimeId: 'llama.cpp',
  source: 'https://docs.ollama.com/modelfile',
  checkedOn: '2026-08-01',
};

const VLLM: Launcher = {
  id: 'vllm',
  label: 'vLLM',
  runtimeId: 'vllm',
  source: 'https://docs.vllm.ai/en/stable/configuration/engine_args/',
  checkedOn: '2026-08-01',
};

const MLX: Launcher = {
  id: 'mlx',
  label: 'MLX',
  runtimeId: 'mlx',
  source: 'https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md',
  checkedOn: '2026-08-01',
};

/**
 * One catalog row, three launchers.
 *
 * `RUNTIMES` labels its first row "llama.cpp / Ollama" because the two share an engine and
 * therefore share every figure this app computes — but they are different *command surfaces*, and
 * a placement carries no signal for which one the reader runs. So the row produces both, labelled,
 * rather than a guess between them; `llama-bench` is a third because the measurement form is a
 * separate binary from the server.
 */
const LAUNCHERS: Record<string, readonly Launcher[]> = {
  'llama.cpp': [LLAMA_SERVER, OLLAMA, LLAMA_BENCH],
  vllm: [VLLM],
  mlx: [MLX],
};

/**
 * The reason no launcher for this scenario can emit anything, or nothing.
 *
 * Checked once and applied to every launcher, because these are properties of the *placement* and
 * not of any command surface: a configuration the engine refused is refused whichever binary you
 * would have typed.
 */
function placementRefusal(input: LaunchInput): string | undefined {
  const { placement } = input;
  if (placement.unsupported !== undefined) return placement.unsupported;
  if (placement.impossible) {
    /**
     * **Two different failures wear one flag, and the first draft named only the first** (raised by
     * Codex on #164). `impossible` is set either when the cache and activations alone are over the
     * ceiling — where lowering context or concurrency really does help — *or*, on a rig with
     * nowhere to spill, whenever anything is over at all, which on unified memory and CPU RAM
     * includes an oversized checkpoint whose cache is nowhere near the limit. Telling that reader
     * to lower the context is advice that will not work at any context.
     *
     * `floorBytesPerDevice` is the quantity that splits them: it is the cache plus activations,
     * carried on `Placement` precisely so a sentence and its predicate read one value.
     */
    const machine = input.rig.device.name;
    return placement.floorBytesPerDevice > placement.allocatableBytesPerDevice
      ? `This configuration does not run on ${machine}: the cache and activations alone are over ` +
          `the ceiling, so no flag rescues it. Lower the context or the concurrency and the ` +
          `commands come back.`
      : `This configuration does not run on ${machine}: the weights are over the ceiling and ` +
          `there is nowhere slower to spill them to. Lowering the context will not help — a ` +
          `narrower format or a smaller model is what changes the answer.`;
  }
  return undefined;
}

export function launchCommands(input: LaunchInput): readonly LauncherCommands[] {
  const launchers = LAUNCHERS[input.runtime.id] ?? [];
  const refusal = placementRefusal(input);

  return launchers.map((launcher) => {
    if (refusal !== undefined) {
      return {
        launcher,
        serve: { ok: false, reason: refusal },
        measure: { ok: false, reason: refusal },
      };
    }
    return { launcher, ...EMITTERS[launcher.id](input) };
  });
}

type Pair = { serve: Emission; measure: Emission };

const EMITTERS: Record<string, (input: LaunchInput) => Pair> = {
  'llama-server': llamaServer,
  'llama-bench': llamaBench,
  ollama: ollama,
  vllm: vllm,
  mlx: mlx,
};

function llamaServer(input: LaunchInput): Pair {
  const { model, quant, usage } = input;
  const ngl = gpuLayers(input);
  const kv = LLAMA_KV_TYPE[usage.kvPrecision];
  const split = tensorSplit(input);
  /**
   * The product, and it is the flag most likely to be got wrong by hand. `n_ctx_seq` is
   * `n_ctx / n_seq_max` unless the KV buffer is unified, and passing `-np` explicitly is what
   * turns unification off — so `-c` has to carry the whole rig's cache for each slot to get the
   * window the panel above priced. It is also exactly the quantity `totalKvBytes` was sized from.
   */
  const totalContext = usage.contextTokens * usage.concurrency;

  const notes = [
    `-c is the whole cache, not one user's window: llama.cpp hands each of the -np slots ` +
      `n_ctx / n_parallel, so ${fmt(usage.contextTokens)} tokens for each of ${usage.concurrency} ` +
      `is ${fmt(totalContext)} here.`,
    nglNote(input, ngl),
    `-m takes a path on your own disk, which no catalog can supply — the placeholder is the one ` +
      `thing here you are meant to replace.`,
    ...(split === undefined ? packingNotes(input) : []),
    ...(split === undefined ? [] : [tsNote(split, ngl)]),
  ];

  return {
    serve: {
      ok: true,
      text: shell('llama-server', [
        `-m ${ggufPlaceholder(model, quant)}`,
        `-c ${totalContext}`,
        usage.concurrency > 1 ? `-np ${usage.concurrency}` : undefined,
        `-ngl ${ngl}`,
        `-ctk ${kv} -ctv ${kv}`,
        split === undefined ? undefined : `-ts ${split.ratios.join(',')}`,
      ]),
      notes,
    },
    // The serving form's own client is a different binary, and it is `llama-bench` — which this
    // module offers as a launcher of its own rather than duplicating here.
    measure: {
      ok: false,
      reason: `llama-server serves; it does not measure. The llama-bench command below is the measurement form of this same placement.`,
    },
  };
}

function llamaBench(input: LaunchInput): Pair {
  const { model, quant, usage } = input;
  const ngl = gpuLayers(input);
  const kv = LLAMA_KV_TYPE[usage.kvPrecision];
  const prompt = effectivePromptTokens(usage);
  /**
   * `room` gates and no longer supplies a flag. A scenario whose prompt and prefix fill the window
   * has nothing to answer with and therefore nothing to measure, which is a property of the
   * scenario rather than of the benchmark's shape — so the refusal still reads it, while what the
   * decode run asks for comes from {@link decodeBenchSpan}.
   */
  const room = generationTokens(usage);
  const decode = decodeBenchSpan(usage);
  const prefix = usage.cachedPrefixTokens ?? 0;
  /**
   * `-ts` belongs here too, and leaving it off was the same defect one file over. `llama-bench`
   * takes the flag, and a sharded measurement run against llama.cpp's default even split is not a
   * measurement of the placement the server command reproduces — which makes the number it prints
   * unusable for the one thing the measurement form exists for.
   */
  const split = tensorSplit(input);

  const notes = [
    `Two runs, because -p and -n are separate tests: the generation one does not inherit the ` +
      `prompt as cache depth, so a single command would measure decoding from an empty cache. The ` +
      `second run fills the cache to ${fmt(decode.depth)} tokens and then generates ` +
      `${fmt(decode.gen)} from there, which puts it at the ${fmt(usage.contextTokens)}-token ` +
      `window the decode figure above is charged against.`,
    `The prompt length is this scenario's own, which is what makes the first result comparable ` +
      `with the figure above rather than with llama-bench's default of 512. The second run ` +
      `generates only ${fmt(decode.gen)} tokens, and deliberately: decode is a steady-state ` +
      `per-token rate, so asking for more does not sharpen the measurement — the depth is what ` +
      `decides the number, and a window's worth of generation is minutes of wall clock per ` +
      `repetition, five times over.`,
    nglNote(input, ngl),
    // The same sweep the `-ts` flag itself needed on this launcher: a measurement run at
    // llama.cpp's default split times a different placement from the one priced, and that is as
    // true when Headroom cannot express its split as when it declines to repeat an even one. Saying
    // it only on the serving command left the number this panel exists to collect unqualified.
    ...(split === undefined
      ? packingNotes(input)
      : [
          `-ts is the same split the serving command uses. Measuring against llama.cpp's default ` +
            `even split would time a different placement than the one priced above.`,
        ]),
    ...(prefix > 0
      ? [
          `The first run has ${fmt(prefix)} tokens already in the cache, which is what this ` +
            `archetype's resident prefix means. Without it the measurement is a standalone prompt ` +
            `and the prediction above is not: the turn's attention is charged against the prefix. ` +
            `The second run's -d is the window rather than the prefix, because decode is charged ` +
            `at the whole context and the prefix is already part of it.`,
        ]
      : []),
    ...(usage.concurrency > 1
      ? [
          `llama-bench measures one sequence and has no concurrency flag, so it cannot reproduce ` +
            `${usage.concurrency} users. The decode figure it prints is the single-user rate; the ` +
            `capacity above is what ${usage.concurrency} users cost in memory.`,
        ]
      : []),
  ];

  return {
    serve: {
      ok: false,
      reason: `llama-bench measures; it does not serve. The llama-server command above is the serving form of this same placement.`,
    },
    measure:
      room === undefined
        ? { ok: false, reason: noRoomToAnswer(usage.contextTokens) }
        : {
            ok: true,
            /**
             * **Two invocations, because `-p` and `-n` are two separate tests** (raised by Codex on
             * #173). One command with both runs a prompt-processing test *and* a generation test —
             * and the generation test does not inherit the prompt as cache depth, so it measures
             * decoding from an empty cache. That is the weight-bound job rather than the one the
             * panel priced, and at an 8K or 128K context the two are far apart.
             *
             * So prefill is measured at its own depth — the archetype's resident prefix, usually
             * none — and decode is measured at the **top of the window**: `-d` a short way under
             * `contextTokens`, and a generation short enough to leave room for itself.
             *
             * Not `prompt + prefix`, which is where this started and what #180 was filed against.
             * That is the cache at the moment generation *begins*, and it is neither what the
             * engine charges nor what the run would average over: `estimateDecode` prices every
             * step at the whole window, because a session fills that window and answers from near
             * the top of it. See {@link decodeBenchSpan} for the arithmetic and for why the
             * generation is short.
             */
            text: [
              shell('llama-bench', [
                `-m ${ggufPlaceholder(model, quant)}`,
                `-p ${prompt}`,
                `-n 0`,
                prefix > 0 ? `-d ${prefix}` : undefined,
                `-ngl ${ngl}`,
                `-ctk ${kv} -ctv ${kv}`,
                split === undefined ? undefined : `-ts ${split.ratios.join(',')}`,
                `-o md`,
              ]),
              '',
              shell('llama-bench', [
                `-m ${ggufPlaceholder(model, quant)}`,
                `-p 0`,
                `-n ${decode.gen}`,
                `-d ${decode.depth}`,
                `-ngl ${ngl}`,
                `-ctk ${kv} -ctv ${kv}`,
                split === undefined ? undefined : `-ts ${split.ratios.join(',')}`,
                `-o md`,
              ]),
            ].join('\n'),
            notes,
          },
  };
}

function ollama(input: LaunchInput): Pair {
  const { model, quant, usage } = input;

  // Ollama's measurement form, which is the same refusal in every scenario: there is no client to
  // emit, and the one that measures this engine is already on this panel.
  const measure: Emission = {
    ok: false,
    reason:
      `Ollama ships no benchmark client. It runs llama.cpp, so the llama-bench command in this ` +
      `panel measures the same engine on the same GGUF — that is the form to submit.`,
  };

  /**
   * **Ollama takes parallelism as a daemon setting, so a multi-user scenario has no Modelfile**
   * (#171, from Codex on #164).
   *
   * `planPlacement` charges KV and activations for `usage.concurrency` simultaneous users, and the
   * Modelfile carries only `num_ctx`. Ollama's parallelism is `OLLAMA_NUM_PARALLEL`, read by the
   * server at startup — so every Modelfile this surface can write sizes memory for one user
   * against a panel that priced several, and the block would be a command for a placement other
   * than the one above it.
   *
   * The same polarity as the MLX serve refusal below: where the surface cannot be told the thing
   * the figures rest on, there is no command rather than a command with a warning beside it. It
   * points at `llama-server` because that is the launcher which takes the count as a flag on the
   * one invocation. The alternative was to emit the daemon configuration, which is the direction
   * #171's other half is answered in — with a sentence rather than a command. See the cache note
   * below.
   */
  if (usage.concurrency > 1) {
    return {
      serve: {
        ok: false,
        reason:
          `The figures above price ${usage.concurrency} simultaneous users, and Ollama takes ` +
          `parallelism as a daemon setting — OLLAMA_NUM_PARALLEL, read at server startup — rather ` +
          `than as a Modelfile parameter. A Modelfile carrying only num_ctx would size the cache ` +
          `for one user against figures that priced ${usage.concurrency}, which is a different ` +
          `placement from the one above. Use the llama-server command above, which takes the ` +
          `count as a flag on the one command (-np ${usage.concurrency}), or set the concurrency ` +
          `to one to get an Ollama block.`,
      },
      measure,
    };
  }

  const tag = `headroom-${slug(model.name)}-${slug(quant.label)}`;
  /**
   * **Ollama's cache precision is a daemon setting, not a Modelfile parameter** (raised by Codex on
   * #164). `OLLAMA_KV_CACHE_TYPE` is read when the server starts and defaults to `f16`, so a
   * Modelfile cannot carry it — and a long-context configuration that fits only because an 8-bit
   * cache halves it would consume two to four times the modelled cache and OOM.
   *
   * **It is stated as a requirement rather than commanded** (#171). The block used to emit
   * `OLLAMA_KV_CACHE_TYPE=<type> ollama serve &` ahead of the Modelfile, which neither waits for
   * the daemon nor notices an existing one: `ollama serve` fails to bind when a server is already
   * listening, and everything after it then ran against *that* server, still on the default `f16`.
   * Fixing that honestly means a readiness poll, a check for a running daemon, possibly a `pkill` —
   * daemon lifecycle management, in a block whose whole value is that it is one readable
   * invocation, and which people run without reading. So the note states which daemon the figures
   * describe and leaves the restart to the reader, who is the only one who knows what else is
   * talking to it.
   */
  const kv = LLAMA_KV_TYPE[usage.kvPrecision];
  const quantizedCache = usage.kvPrecision !== 'fp16';

  /**
   * **Ollama's Modelfile documents no parameter for the GPU layer count**, and that is the one
   * number this whole feature exists to print. `num_ctx`, `num_predict` and the sampler knobs are
   * the list; `num_gpu` is not on it. So the template says what it cannot say rather than emitting
   * a plausible-looking line — which is the same rule as the artifact refusal, applied to a flag.
   */
  /* The layers, not the flag. `gpuLayers` carries the output tensor's slot on top of the resident
     count, so printing it here would name a split one layer larger than the panel above shows —
     and this sentence is about the panel. Read back off the flag rather than re-derived from the
     placement, so the two cannot come to disagree about how much stayed on the card. */
  const residentLayers = Math.max(gpuLayers(input) - 1, 0);

  const notes = [
    `Ollama's Modelfile has no parameter for the GPU layer count — num_ctx and num_predict are ` +
      `documented, num_gpu is not — so the ${residentLayers}-layer split above is the one thing ` +
      `this surface cannot be told. Ollama decides it. Use llama-server if you need to pin it.`,
    `num_ctx is per request here, unlike llama-server's -c, which is the whole cache across slots.`,
    `FROM takes a path on your own disk, absolute or relative to the Modelfile.`,
    `Written to ${tag}.Modelfile rather than to Modelfile, under set -C, so this cannot overwrite ` +
      `one you already have — and chained with && so a refusal there does not go on to run the ` +
      `old file's settings.`,
    ...(quantizedCache
      ? [
          `Start your daemon with OLLAMA_KV_CACHE_TYPE=${kv} before running this. It is a daemon ` +
            `setting rather than a Modelfile parameter — read once at server startup, defaulting ` +
            `to f16 — so an already-running daemon will not pick it up and has to be restarted. ` +
            `The figures above are sized for a ${kv} cache.`,
        ]
      : []),
  ];

  return {
    serve: {
      ok: true,
      text: [
        /**
         * `set -e` first, and it is the fix for a defect in this block's own previous fix (raised
         * by Codex on #164). Chaining `ollama create && ollama run` stops the *run* after a failed
         * create — but the heredoc above them is a separate command, so a noclobber refusal there
         * still fell through to a create against the old file. A heredoc cannot be `&&`-chained to
         * what follows it; aborting the whole block on any failure is the shell's own answer.
         */
        // A subshell, so `set -e` guards this block and does not survive into the reader's own
        // interactive shell — where it stays enabled and can close their terminal on any later
        // failure. A footgun the previous fix introduced while fixing another. (Codex, #173.)
        `(`,
        `set -e`,
        // A Headroom-specific filename, never the bare `Modelfile` this first wrote to. `cat >`
        // truncates unconditionally, and the directory an Ollama user runs this from is exactly the
        // one likely to already hold a Modelfile of their own — a copy-pasteable block that
        // silently destroys their file (raised by Codex on #164). `set -C` refuses to clobber even
        // this name, so the worst case is an error rather than a lost file.
        `(set -C; cat > ${tag}.Modelfile) <<'EOF'`,
        `FROM ${ggufPlaceholder(model, quant)}`,
        `PARAMETER num_ctx ${usage.contextTokens}`,
        `EOF`,
        ``,
        // Chained, so a noclobber refusal stops here rather than creating and running whatever the
        // old file said. Changing only the context keeps the same filename, so the second run is
        // exactly when the stale-settings case fires — and a successful final command would have
        // hidden the redirection error above it. Raised by Codex on #164.
        `ollama create ${tag} -f ${tag}.Modelfile && ollama run ${tag}`,
        `)`,
      ].join('\n'),
      notes,
    },
    measure,
  };
}

function vllm(input: LaunchInput): Pair {
  const { model, quant, usage, rig } = input;
  const repo = artifactFor(model, quant.id);
  if (repo === undefined) {
    const reason = noArtifact(model, quant, 'vLLM');
    return { serve: { ok: false, reason }, measure: { ok: false, reason } };
  }

  const tp = effectiveDeviceCount(rig);
  const kv = VLLM_KV_DTYPE[usage.kvPrecision];
  const prompt = effectivePromptTokens(usage);
  const gen = generationTokens(usage);
  const revision = revisionOf(model);

  /**
   * `--cpu-offload-gb`, without which a spilled placement is a command that OOMs (raised by Codex
   * on #164).
   *
   * `planPlacement` reports a *runnable* placement whenever the cache and activations fit and only
   * the weights are over — Qwen3 32B at BF16 on one 5090 is the reachable case — so both commands
   * were emitted for a configuration the reader cannot actually start. vLLM defaults the flag to
   * zero, meaning it will try to keep every weight on the GPUs.
   *
   * The field is documented as "the space in GiB to offload to CPU, **per GPU**", and vLLM is
   * tensor-parallel: every rank holds an equal shard, so the rig's spill divides evenly. Rounded
   * **up**, because the two directions are not symmetric — too little offload is an out-of-memory
   * error on load and too much is merely slower.
   */
  const spilledPerGpu =
    input.placement.offloadFraction > 0
      ? Math.ceil(
          (input.placement.offloadFraction * input.placement.totalWeightBytes) / tp / 1024 ** 3
        )
      : 0;

  /**
   * `--gpu-memory-utilization` is emitted rather than left default, and the reason is that the two
   * do not agree: `RuntimeSpec.preallocFraction` is 0.9, which is what every vLLM figure on this
   * page was budgeted against, while vLLM's own default has moved to 0.92. Stating it makes the
   * command reproduce the placement the panel priced instead of a slightly roomier one.
   */
  const notes = [
    `--gpu-memory-utilization is stated rather than left to vLLM's default, which is 0.92 — the ` +
      `figures above are budgeted at 0.9, and this is what makes the command match them.`,
    ...(revision === undefined
      ? []
      : [
          `--revision pins ${revision.slice(0, 10)}, the commit this model's parameter counts and ` +
            `attention shape were read from. Without it vLLM resolves the repo's default branch, ` +
            `which can move.`,
        ]),
    ...(spilledPerGpu > 0
      ? [
          `--cpu-offload-gb is per GPU, and it is what makes this placement start at all: ` +
            `${percentish(input.placement.offloadFraction)} of the weights do not fit, and vLLM ` +
            `defaults the flag to zero — so without it the command tries to keep everything on the ` +
            `cards. Rounded up, because too little offload is an OOM and too much is only slower.`,
        ]
      : []),
    `--max-model-len is one sequence's window; --max-num-seqs is how many of them vLLM will hold ` +
      `at once. Together they are the cache the panel above sized.`,
  ];
  /*
   * No "this rig cannot shard, so the command drives one device" note, and its absence is
   * deliberate rather than an omission. `planPlacement` marks any `count > 1` rig without an
   * interconnect `unsupported`, and `placementRefusal` turns every launcher away before an emitter
   * runs — so inside this function `effectiveDeviceCount(rig)` is always `rig.count`, and a note
   * keyed on the difference could never fire. The refusal upstream owns that case, with its own
   * sentence.
   */

  return {
    serve: {
      ok: true,
      text: shell(`vllm serve ${repo}`, [
        revision === undefined ? undefined : `--revision ${revision}`,
        `--max-model-len ${usage.contextTokens}`,
        tp > 1 ? `--tensor-parallel-size ${tp}` : undefined,
        kv === undefined ? undefined : `--kv-cache-dtype ${kv}`,
        `--gpu-memory-utilization 0.9`,
        `--max-num-seqs ${usage.concurrency}`,
        spilledPerGpu > 0 ? `--cpu-offload-gb ${spilledPerGpu}` : undefined,
      ]),
      notes,
    },
    measure:
      gen === undefined
        ? {
            ok: false,
            reason: noRoomToAnswer(usage.contextTokens),
          }
        : {
            ok: true,
            text: shell(`vllm bench latency`, [
              `--model ${repo}`,
              revision === undefined ? undefined : `--revision ${revision}`,
              `--input-len ${prompt}`,
              `--output-len ${gen}`,
              // The configured user count, not the client's own default of 8. Without it the
              // benchmark times a different batch from the one the panel priced, which makes the
              // number unusable for the thing a measurement is for (raised by Codex on #164).
              `--batch-size ${usage.concurrency}`,
              `--max-model-len ${usage.contextTokens}`,
              tp > 1 ? `--tensor-parallel-size ${tp}` : undefined,
              kv === undefined ? undefined : `--kv-cache-dtype ${kv}`,
              `--gpu-memory-utilization 0.9`,
              spilledPerGpu > 0 ? `--cpu-offload-gb ${spilledPerGpu}` : undefined,
            ]),
            notes: [
              `vllm bench latency times one batch offline, at this scenario's own prompt, ` +
                `generation length and user count — which is what makes the result comparable ` +
                `with the figures above rather than with the client's defaults of 32, 128 and 8.`,
              ...(revision === undefined
                ? []
                : [
                    `--revision pins the commit the catalog derived this model's shape from, so ` +
                      `the command cannot quietly load a newer checkpoint than the figures ` +
                      `describe.`,
                  ]),
              `Needs the bench extra: pip install vllm[bench].`,
            ],
          },
  };
}

function mlx(input: LaunchInput): Pair {
  const { model, quant, usage, runtime } = input;

  /**
   * The #18 refusal, and the sharp end of it. Every GGUF row in MLX's `weightFormats` is a
   * stand-in *by width* for MLX's own affine scheme — the figures are modelled, the checkpoint is
   * not one MLX loads. A command naming it would be a command for a file that does not exist, in
   * copy-pasteable form, which is exactly the failure the substitution marker was built to prevent.
   */
  const substitution = substitutionFor(runtime, quant.id);
  if (substitution !== undefined) {
    const reason =
      `MLX does not load ${quant.label}. ${substitution} That makes the figures above a modelled ` +
      `stand-in and this checkpoint a file that does not exist — so there is no command. Convert ` +
      `the weights yourself with mlx_lm.convert, or select BF16, which MLX reads natively.`;
    return { serve: { ok: false, reason }, measure: { ok: false, reason } };
  }

  const repo = artifactFor(model, quant.id);
  if (repo === undefined) {
    const reason = noArtifact(model, quant, 'MLX');
    return { serve: { ok: false, reason }, measure: { ok: false, reason } };
  }

  const prompt = effectivePromptTokens(usage);
  const gen = generationTokens(usage);
  /**
   * **`mlx_lm.server` has no KV quantization flag.** `--kv-bits`, `--kv-group-size` and
   * `--quantized-kv-start` are `mlx_lm.generate`'s; the server's argument list does not carry
   * them. So an 8-bit-cache scenario cannot be *served* at the precision it was priced at, and the
   * command says so rather than printing a flag that does not parse.
   */
  const quantizedCache = usage.kvPrecision !== 'fp16';

  return {
    /**
     * **A refusal rather than a warning, when the cache precision cannot be reproduced** (raised by
     * Codex on #164, P1). The first version emitted the server command with a loud note saying the
     * cache would be fp16 — but a long-context configuration that fits *because* an 8-bit cache
     * halves it will OOM when run at fp16, and a note beside a copy button does not stop that. The
     * command is not a command for the placement the panel priced, so there is no command.
     *
     * The measurement form survives, because `mlx_lm.generate` does take `--kv-bits`.
     */
    serve: quantizedCache
      ? {
          ok: false,
          reason:
            `The figures above are priced with an 8-bit cache, and mlx_lm.server has no flag for ` +
            `it — --kv-bits is on mlx_lm.generate, not the server. A served command would run an ` +
            `fp16 cache needing roughly twice the memory shown, which is a different placement ` +
            `from the one above. Select an FP16 cache to get a serving command, or use the ` +
            `measurement command below, which does take the precision.`,
        }
      : {
          ok: true,
          text: shell(`mlx_lm.server`, [
            `--model ${repo}`,
            ...(gen === undefined ? [] : [`--max-tokens ${gen}`]),
          ]),
          notes: [
            `MLX reads this repo directly; no conversion step, because BF16 is the one format the ` +
              `catalog and MLX agree on.`,
            // No `--revision` on this server, so the pin the vLLM command gets is unavailable here.
            ...(revisionOf(model) === undefined
              ? []
              : [
                  `mlx_lm.server takes no revision flag, so this resolves the repo's default ` +
                    `branch. The figures above were derived from ${revisionOf(model)!.slice(0, 10)}.`,
                ]),
          ],
        },
    measure:
      gen === undefined
        ? { ok: false, reason: noRoomToAnswer(usage.contextTokens) }
        : {
            ok: true,
            text: shell(`python -c "print('word ' * ${prompt})" | mlx_lm.generate`, [
              `--model ${repo}`,
              `--prompt -`,
              `--max-tokens ${gen}`,
              // `--quantized-kv-start` defaults to 5,000 on the CLI and the engine prices every token at
              // the selected precision, so a run finishing under that threshold would benchmark an
              // entirely fp16 cache against a Q8 prediction (raised by Codex on #164). Zero is what makes
              // the measured cache the priced one.
              ...(quantizedCache
                ? [`--kv-bits 8`, `--kv-group-size 64`, `--quantized-kv-start 0`]
                : []),
            ]),
            notes: [
              `mlx_lm.generate prints prompt and generation tokens/sec, which are the two figures above.`,
              ...(revisionOf(model) === undefined
                ? []
                : [
                    `mlx_lm.generate takes no revision flag either, so this resolves the repo's default ` +
                      `branch. The figures above were derived from ${revisionOf(model)!.slice(0, 10)}; if ` +
                      `it has moved, the measurement is of a different checkpoint.`,
                  ]),
              ...(usage.concurrency > 1
                ? [
                    `mlx_lm.generate processes one prompt and has no concurrency option, so it ` +
                      `cannot reproduce ${usage.concurrency} users. Read what it prints as the ` +
                      `single-user rate; the figures above are sized for the batch, and on an MoE ` +
                      `model the batch changes the speed as well as the memory.`,
                  ]
                : []),
              `The prompt is piped rather than quoted because its length is what is being measured — ` +
                `${fmt(prompt)} tokens. "word " is roughly one token each, so treat the count as ` +
                `approximate and read the tokens-per-second the tool reports back.`,
              ...(quantizedCache
                ? [
                    `--kv-bits 8 is the precision the figures above assume, and unlike the server, ` +
                      `mlx_lm.generate takes it. --quantized-kv-start 0 is what makes it apply from the ` +
                      `first token: the CLI default is 5,000, and the engine prices every token at 8 bits.`,
                  ]
                : []),
            ],
          },
  };
}

/**
 * The `-ts` sentence, which has to state both readings because the flag shows only one.
 *
 * The ratios a reader sees are the window's, and the window carries the output tensor — so the
 * largest number in the flag is one greater than the layer count that card holds, and a sentence
 * that offered the ratios as the layer split would be off by one on exactly the card whose share
 * the panel is least able to check. Saying which card the table lands on is the other half: it is
 * the reason that share is larger, and it is a placement fact `-ts` is otherwise silent about.
 */
function tsNote(split: TensorSplit, ngl: number): string {
  const output = split.ratios.findIndex((r, i) => r !== split.sized[i]);
  return (
    `-ts proportions the -ngl window, not the model: llama.cpp puts the last ${ngl} slots of the ` +
    `stack on GPUs and splits those by these ratios. The window counts the output tensor as a slot ` +
    `of its own, so the ratios sum to -ngl rather than to the layer count — ${split.sized.join(',')} ` +
    `layers is the split Headroom sized, and card ${output + 1} carries the output tensor on top of ` +
    `its ${split.sized[output]}. The default divides by device memory instead, and therefore evenly ` +
    `on identical cards, which is the wrong answer for a model whose layers cache different amounts.`
  );
}

/** The `-ngl` sentence, which differs in the three cases the flag has. */
function nglNote(input: LaunchInput, ngl: number): string {
  const { model, rig, placement } = input;
  if (rig.device.class === 'cpu-ram') {
    return `-ngl 0 because ${rig.device.name} has no GPU to offload to — every layer runs on the host, which is what the figures above price.`;
  }
  if (ngl > model.layers) {
    return `-ngl ${ngl} is all ${model.layers} layers plus one: llama.cpp counts the output tensor a position past the repeating blocks, so ${model.layers} would keep the output tensor and leave layer 0 on the host.`;
  }
  /* Zero is its own sentence, not the general one with a zero in it.
     The branch below counts the output slot, which is only there for a *positive* `-ngl` — at
     zero llama.cpp offloads nothing at all, so the general sentence would offer "-1 of L layers"
     as the resident count. It was reachable on 2,202 catalog configurations when the branch was
     added, Llama 3.2 3B BF16 on two 5080s at 32K among them.

     **Still live under #204, and deliberately.** `gpuLayers` now adds the output's slot on every
     positive count, and the one place it does not is here: a zero-resident GPU rig would emit
     `-ngl 1`, which is the whole output table on a card that had no room for a layer. #204 left
     that open rather than settling it — see {@link gpuLayers} — so this branch keeps its subject.
     Distinct from the `cpu-ram` case above, which is about a rig with no GPU rather than a GPU
     with no room. */
  if (ngl === 0) {
    /* And it does *not* claim to match the figures, which was this sentence's first mistake.
       `residentLayers` floors, so it reaches zero while a fraction of the weights is still
       resident — across the configurations that reach this note the spill runs from around 75%
       to 99.9% and is never 100%. `-ngl 0` puts nothing on the GPU at all, so the command is a
       slower placement than the panel priced, every time. */
    return (
      `-ngl 0 puts nothing on the GPU: ${rig.device.name} has no room for a whole layer beside ` +
      `the cache it has to hold. The figures above price ${percentish(placement.offloadFraction)} ` +
      `of the weights spilling rather than all of them, so expect this command to run slower ` +
      `than the panel estimates.`
    );
  }
  /* One sentence for what Headroom sized and what llama.cpp loads, because since #204 they are
     the same thing: the flag carries the resident count *plus* the output tensor's slot, so the
     layer count a reader wants is `ngl - 1` and the flag is what produces it. This used to emit
     the resident count bare and warn that llama.cpp would read it as one layer fewer. */
  return (
    `-ngl ${ngl} keeps ${ngl - 1} of ${model.layers} layers on the GPU, plus the output tensor, ` +
    `which llama.cpp counts as a slot of its own a position past the blocks — a count Headroom ` +
    `sized, not a fraction of the model. ${percentish(placement.offloadFraction)} of the weights ` +
    `spill to host RAM, and which layers stay is what decides it.`
  );
}

/** Thousands separators, since these are long token counts a reader has to check against a panel. */
function fmt(n: number): string {
  return n.toLocaleString('en-US');
}

function percentish(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** A shell- and Ollama-safe tag: lowercase, no spaces, no slashes. */
function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}
