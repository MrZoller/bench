import type { ModelSpec, QuantSpec } from './types';
import { denseParams } from './types';

/**
 * Weight sizing.
 *
 * `params * bits / 8` is right only when a scheme quantizes the whole network evenly, and the
 * models people most want to run are exactly the ones where it doesn't. gpt-oss ships MXFP4
 * routed experts alongside BF16 attention and embeddings; charging the entire 116.8B at 4.25
 * bpw understates it by about 4 GB.
 *
 * The same split has to be applied to the *active* parameters, and that is easier to get
 * wrong: active params are roughly half dense for gpt-oss, and charging that dense half the
 * blended whole-model rate understates bytes-read-per-token by ~1.9x — which lands directly
 * on decode throughput.
 *
 * Which dense parameters count is its own trap. `totalParams - expertParams` is the residency
 * figure, not the per-token one: it includes an untied embedding table that decode reads a
 * single row of, and any vision tower a text request never runs. Both belong to
 * `activeDenseParams`, and using the residency figure instead cost gpt-oss-20b 31% of its
 * decode throughput.
 */

export interface WeightBreakdown {
  /** Routed expert FFN weights — the bulk of any MoE model. */
  expertBytes: number;
  /** Everything else: attention, embeddings, norms, router, shared experts, dense FFN layers. */
  denseBytes: number;
  /**
   * Weights that sit outside the repeating stack — see {@link fixedParams}.
   *
   * A subset of `denseBytes`, orthogonal to the expert/dense split: that one asks what *rate* a
   * tensor is charged at, this one asks how many devices it divides across.
   *
   * **The answer is one device each, and not the *same* one** (#182). This field used to be
   * documented as "the answer is one, which is the whole reason the field exists", and that premise
   * is what upstream disproves: llama.cpp routes the three tensors to three different places, and
   * lumping them into a single scalar cannot express any of them. `token_embd.weight` is pinned to
   * the host by `llama-model.cpp:1333-1335` — *"there is very little benefit to offloading the
   * input layer, so always keep it on the CPU"* — with no `-ngl`, `-sm` or `-ts` input to that
   * decision; the output projection goes to the **last** `-ts` device, whole; a vision tower is
   * loaded by `tools/mtmd/clip.cpp` onto the **first** GPU, whole. Read at ggml-org/llama.cpp
   * commit `360e134`.
   *
   * So the sum survives, as the part of the file that is in no repeating layer and therefore
   * divides `layerBytes` off the file — and the three components beside it are what a placement
   * reads. `hostResidentBytes + outputBytes + towerBytes === fixedBytes` exactly, including when
   * the clamp below fires.
   */
  fixedBytes: number;
  /**
   * Of `fixedBytes`, the part that is resident in host RAM and on **no device of a discrete-GPU
   * rig** — the input embedding table.
   *
   * **Zero for a tied model, and the reason is not that the host holds nothing.** llama.cpp
   * materialises a tied table *twice*: `TENSOR_DUPLICATED` re-routes the output's creation to
   * `buft_list_output` (`llama-model-loader.cpp:1101-1106`), and the de-dup at `1285-1300` only
   * fires when the same buffer type already holds the tensor — input buft is CPU and output buft is
   * the last GPU, so a second tensor is created and llama.cpp adds its bytes to `size_data` itself.
   * Measured on `gemma-3-270m-GGUF` at `-ngl 19`: a 271.81 MiB file, resident as 170.00 MiB on the
   * CPU and 271.81 MiB on MTL0. The file carries one table and the GPU holds one table, so nothing
   * comes off the device total; what the host holds is a *second* materialisation that was never in
   * the file. An **untied** model has two tables in the file and the GPU holds one of them, which is
   * the whole of the over-charge #182 was filed for — 7.6% of the file on Qwen3 8B, 6.5% on
   * Llama 3.1 8B, and 26 of the 35 catalog rows are untied.
   *
   * **Stated here, applied by `planPlacement`**, which is the only layer that knows the device
   * class and the runtime. It is a discrete-GPU correction for a runtime that declares
   * `RuntimeSpec.hostResidentInputEmbedding`: on unified memory the host is the rig, and vLLM's
   * tensor-parallel ranks genuinely slice the embedding table and keep every slice on a card.
   */
  hostResidentBytes: number;
  /**
   * Of `fixedBytes`, the output projection — or a tied table's GPU duplicate, which is the same
   * bytes in the same place. Resident **whole on the last `-ts` device**, never divided.
   */
  outputBytes: number;
  /**
   * Of `fixedBytes`, any non-language tower. Resident **whole on the first GPU**, and only when
   * `--mmproj` is passed.
   *
   * Charged unconditionally all the same, because Headroom has no `--mmproj` axis and the failure
   * direction of assuming it absent is the one that OOMs a reader. What #182 changes is only where
   * it sits: `clip.cpp:184-205` builds its own backend list and takes the *first* GPU device,
   * never `tensor_split` — the opposite end of the rig from the output projection, which is why
   * seeding one bin with a lumped `fixedBytes` would have been wrong in a new way.
   */
  towerBytes: number;
  /**
   * The repeating transformer blocks — `totalBytes - fixedBytes`, and the only part of the file
   * that is `layers` of anything.
   *
   * **This, not `totalBytes`, is what a per-layer figure divides.** Charging `totalBytes / layers`
   * calls a layer "a layer plus a share of the embeddings", which is right as a byte total and
   * wrong the moment anything reads it back as a count (#165).
   */
  layerBytes: number;
  totalBytes: number;
  /** Blended bits per weight across the whole model. For display; never use it to size a subset. */
  effectiveBpw: number;
}

export function weightBreakdown(model: ModelSpec, quant: QuantSpec): WeightBreakdown {
  const dense = denseParams(model);
  // A uniform scheme spares nothing; `denseBpw` is set only when the scheme deliberately does.
  const denseBpw = quant.denseBpw ?? quant.bpw;

  const expertBytes = (model.expertParams * quant.bpw) / 8;
  const denseBytes = (dense * denseBpw) / 8;
  const totalBytes = expertBytes + denseBytes;

  // Fixed tensors are never routed experts, so `denseBpw` is their rate and `denseBytes` their
  // ceiling. The clamp is defensive rather than reachable: it keeps `layerBytes` non-negative on a
  // row whose vocabulary and tower somehow outweigh its own dense half.
  const unclamped = (fixedParams(model) * denseBpw) / 8;
  const fixedBytes = Math.min(denseBytes, unclamped);
  // The three components are scaled by whatever the clamp took, so they sum to `fixedBytes` on the
  // clamped path as well as the ordinary one. A defensive clamp that quietly broke the identity a
  // placement seeds bins from would be worse than the negative `layerBytes` it exists to prevent.
  const held = unclamped > 0 ? fixedBytes / unclamped : 0;

  const tableBytes = (outputProjectionParams(model) * denseBpw) / 8;
  const towerBytes = ((model.nonLanguageParams ?? 0) * denseBpw) / 8;

  return {
    expertBytes,
    denseBytes,
    fixedBytes,
    // One table for an untied model, none for a tied one — `fixedParams` counted one table there
    // and the GPU's duplicate is what it pays for. See the field's docblock.
    hostResidentBytes: model.tiedEmbeddings ? 0 : tableBytes * held,
    outputBytes: tableBytes * held,
    towerBytes: towerBytes * held,
    layerBytes: totalBytes - fixedBytes,
    totalBytes,
    effectiveBpw: (totalBytes * 8) / model.totalParams,
  };
}

/**
 * Parameters that live outside every repeating layer, and therefore outside any per-layer figure.
 *
 * Three tensors, none of which a layer split divides:
 *   - the **input embedding** table, `vocab x hidden`;
 *   - the **output projection**, a second table of the same size — unless it is tied, in which
 *     case there is one table doing both jobs and it must not be counted twice;
 *   - **non-text towers**, which are resident and are not part of the language stack.
 *
 * The same three corrections `activeDenseParams` is built from, read for a different purpose:
 * there the question is what a token *reads*, here it is what a device *holds whole*. They pull
 * apart on the tie — a tied table is read every step, and it occupies one device *and* the host,
 * because llama.cpp materialises it twice. See {@link WeightBreakdown.hostResidentBytes}; the sum
 * here is a **file** figure, and which device holds which part of it is
 * {@link WeightBreakdown.outputBytes} and {@link WeightBreakdown.towerBytes}.
 *
 * These are not a rounding error on the models people run smallest. Llama 3.2 3B carries a
 * 128,256-token vocabulary at 3,072 hidden against 3.2B total, so the shared table is over 12% of
 * the weights, and Gemma 3 4B's vocabulary and vision tower together are 25% — which no layer holds
 * any of.
 *
 * **`tiedEmbeddings` is stated, never inferred from its own absence.** It used to be optional and
 * read as untied when missing, which over-stated the fixed block and was conservative in both
 * directions it moved — a smaller per-layer weight, fewer resident layers. #182 removed that
 * safety: an untied model's `hostResidentBytes` is now deducted from what the cards are charged, so
 * a tied model that failed to say so would give away a whole `vocab x hidden` table of card budget,
 * the direction that reports a fit and then OOMs. The field is required on {@link ModelSpec} and
 * `toModel` rejects a catalog without it, so the two readings here and in
 * {@link WeightBreakdown.hostResidentBytes} cannot come apart on a missing value.
 *
 * What makes the stated value trustworthy is its *derivation*: `build-catalog.ts` reads the tie from
 * the safetensors tensor list — whether an output head exists at all — rather than from
 * `config.json`'s `tie_word_embeddings`, which is absent on both Gemma 3 repos even though they are
 * tied. Every catalog row states the field, and none of them guesses it.
 */
export function fixedParams(model: ModelSpec): number {
  const table = outputProjectionParams(model);
  return (model.tiedEmbeddings ? table : 2 * table) + (model.nonLanguageParams ?? 0);
}

export function weightBytes(model: ModelSpec, quant: QuantSpec): number {
  return weightBreakdown(model, quant).totalBytes;
}

/**
 * Fraction of routed experts touched by a batch of `batch` tokens.
 *
 * One token selects `perToken` of them; a batch collectively selects more, approaching all of
 * them as the batch grows. This is why MoE throughput improves with concurrency far less than
 * dense throughput does — an effect that surprises people putting these models behind a server.
 */
export function expertFraction(model: ModelSpec, batch: number): number {
  if (model.expertParams === 0) return 0;
  const n = Math.max(1, batch);

  if (model.experts) {
    const { total, perToken } = model.experts;
    return 1 - (1 - perToken / total) ** n;
  }

  // Without expert counts the union cannot be modelled, so fall back to the difference between
  // the published active figure and the physical dense basis, held flat across batch. Better a
  // known-conservative estimate than a fabricated curve.
  //
  // The subtrahend is `activeDenseParams` rather than an arithmetic reconstruction of it. Those
  // agree for every untied text-only model — the whole catalog today — but diverge for a tied
  // or multimodal one, and there the reconstruction double-counts: it removes a vocabulary
  // table that `effectiveActiveParams` then adds back through this fraction, charging it twice.
  // Deriving from the field that already states the basis cannot drift from it.
  const implied = (model.activeParams - model.activeDenseParams) / model.expertParams;
  return Math.min(1, Math.max(0, implied));
}

/**
 * Parameters read for one token: the dense stack it actually runs, plus the experts a batch of
 * `batch` tokens collectively routes through.
 *
 * At batch 1 this lands on the vendor's published active-parameter figure for an untied,
 * text-only model, and deliberately above it for a tied or multimodal one — see
 * `ModelSpec.activeDenseParams` for why those two cases pull apart.
 */
export function effectiveActiveParams(model: ModelSpec, batch: number): number {
  return model.activeDenseParams + model.expertParams * expertFraction(model, batch);
}

/**
 * Parameters a single prompt token is computed through — the FLOPs basis for prefill.
 *
 * Differs from the decode basis in two ways:
 *   - the expert term uses the experts *one* token routes through, since FLOPs scale per token
 *     while bytes scale with the union a whole batch touches;
 *   - the output projection is excluded. Logits are produced only for the positions that need
 *     them — one, for generation — so a `vocab x hidden` matmul on every prompt token is work
 *     no runtime performs. llama.cpp gathers the output rows before the LM head and vLLM slices
 *     the hidden states before it. Charging it anyway overstated prefill by 16% on gpt-oss-20b
 *     and 9% on Gemma 3 12B, the models with the largest vocabularies relative to their depth.
 *
 * That one output position still has to be paid for — see {@link outputProjectionParams}, which
 * the caller adds once per request rather than per token.
 */
export function prefillComputeParams(model: ModelSpec): number {
  const perToken = model.activeDenseParams + model.expertParams * expertFraction(model, 1);
  return Math.max(0, perToken - outputProjectionParams(model));
}

/**
 * The `vocab x hidden` output projection, charged **once per prefill request**.
 *
 * Excluding it per-token and never adding it back is right in the limit and wrong for short
 * prompts, where it is not a rounding error: on a one-token gpt-oss-20b prompt the projection is
 * 0.58B of a 3.60B pass, so dropping it understates that pass by 16%. Per-token minus,
 * once-per-request plus, is exact at every prompt length rather than only at long ones.
 */
export function outputProjectionParams(model: ModelSpec): number {
  return model.vocabSize * model.hiddenSize;
}

/**
 * Bytes of weight read for a decode step.
 *
 * Charges the dense and expert halves at their own rates rather than a blended average. For a
 * uniform scheme the two are identical; for expert-only schemes like MXFP4 the difference is
 * about a factor of two on the number that sets decode speed.
 */
export function activeWeightBytes(model: ModelSpec, quant: QuantSpec, batch = 1): number {
  const denseBpw = quant.denseBpw ?? quant.bpw;
  const activeExpertParams = model.expertParams * expertFraction(model, batch);
  return (model.activeDenseParams * denseBpw + activeExpertParams * quant.bpw) / 8;
}
