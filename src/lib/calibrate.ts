/**
 * Predicted versus measured (#139).
 *
 * The roofline is calibrated on two anchors — a DGX Spark on gpt-oss-20b and an EPYC 9654 on
 * DeepSeek-671B Q8 — and the ±30% band is asserted against them. This is the affordance that lets
 * anyone else check it: paste `llama-bench` output beside the prediction for the same scenario, see
 * both numbers, and optionally submit the pair through a pre-filled GitHub issue.
 *
 * Every competitor's numbers are rules of thumb; nobody can check them and nobody is asked to. A
 * public predicted-versus-measured record is the one argument that cannot be copied without the
 * community that feeds it.
 *
 * ## Parse, don't ask
 *
 * `llama-bench` emits markdown by default and JSON on `-o json`, and **JSON is strongly preferred
 * here** — it carries `build_commit`, which #139 names as the version-skew guard and which nothing
 * here checks (it is captured for the issue body, see {@link Measurement.buildCommit}), plus
 * `n_prompt`, `n_gen` and `n_depth` as numbers rather than as a string to re-parse. The markdown reader exists
 * because the default output is markdown and a reader who has already run the tool should not have
 * to run it again.
 *
 * The pasted text never leaves the page. The submission is a `github.com/…/issues/new` URL the
 * reader chooses to open, which is the same no-backend shape the weekly catalog refresh already
 * proved out.
 *
 * ## What a measurement has to carry before it means anything
 *
 * A measurement that cannot name its scenario is worth little to calibration, so {@link compare}
 * marks the ways that happens instead of reporting a delta against them. **{@link describeMismatch}
 * is that list, and this section deliberately does not restate it** — four rounds of #175 went on a
 * prose summary of that function which diverged from it in a different way each time, which is the
 * argument for a pointer over a paraphrase.
 *
 * The one property of it worth stating here, because it is easy to assume otherwise: **a check fires
 * only when the paste states the field it compares.** llama-bench's output is sparse and most of
 * those fields are optional in it, so a JSON row carrying neither a model name nor a cache type
 * compares clean — the suite's own `JSON_OUTPUT` fixture is exactly that row. Unstated is not
 * rejected, except where a default makes silence itself a claim.
 *
 * What this section is about is the four that are **invisible in the numbers**, because those are
 * the ones a reader can get wrong without noticing. **Two of them this module rejects; two it only
 * records** — a distinction worth keeping straight, since describing a recorded field as a check is
 * advertising a guard that does not exist (Codex spent two rounds on it in #175, and a third on
 * this paragraph reading as though the four were all of them).
 *
 * Rejected — marked, with no delta reported against them:
 *
 *   - **A different prompt length.** Prefill is quadratic in the prompt, so `pp512` against a
 *     prediction made at 16,384 tokens is not a disagreement about the model — it is two different
 *     jobs.
 *   - **A different depth.** `estimatePrefill` charges an agent turn's attention against a resident
 *     prefix, so a standalone `pp` run measures a different workload than the prediction. `-d` is
 *     what reproduces it, and `n_depth` is what says whether it was used.
 *
 * Recorded — nothing here compares them, because there is nothing to compare them against:
 *
 *   - **A different build.** A llama.cpp from six months ago is a different runtime for calibration
 *     purposes, and the catalog pins a runtime rather than a commit of one. `build_commit` is
 *     captured when the paste carries it and its absence is stated, for a human weighing the
 *     submission. See {@link Measurement.buildCommit}.
 *   - **A different machine.** `llama-bench` names the model file and the backend but not the host
 *     reliably, so the scenario URL is what ties a measurement to a device row — which is why the
 *     issue template makes that field non-optional.
 */

/** One row of `llama-bench` output, in the units bench compares against. */
export interface Measurement {
  /** `pp` rows measure prompt processing; `tg` rows measure generation. */
  kind: 'prefill' | 'decode';
  /** The `n_prompt` or `n_gen` the row was run at. */
  tokens: number;
  /** Tokens already in the cache — `-d`. Absent where the run did not state one. */
  depthTokens?: number;
  tokensPerSec: number;
  /** The `±` figure, where the format carried one. */
  stddev?: number;
  /**
   * llama.cpp's own commit, from JSON output only.
   *
   * **Captured, never checked.** `describeMismatch` does not read it and there is nothing to read
   * it against — the catalog pins a runtime, not a commit of one. It rides into the generated issue
   * body so a human weighing the submission can see it, and its absence is stated there rather than
   * assumed benign. #139 calls it the version-skew guard; that is the role it plays for a reviewer,
   * not a rejection this module makes.
   */
  buildCommit?: string;
  /** `-ngl`, where the format carried it — the layer split the run actually used. */
  gpuLayers?: number;
  /** `-ctk`/`-ctv`, from JSON. The cache precision the run used, which changes both bytes and rate. */
  kvTypes?: { k: string; v: string };
  /** However the format names the checkpoint — `model_type`, or the markdown table's first column. */
  modelLabel?: string;
  /**
   * Parameters, as the tool counts them — `model_n_params` in JSON, the `params` column in
   * markdown.
   *
   * **The load-bearing model check, and much better than the name.** Comparing labels catches a
   * DeepSeek paste against a Llama prediction and misses Qwen3 8B against Qwen3 32B, because
   * llama.cpp writes an architecture where the catalog writes a product — the two never agree past
   * the first word. A parameter count is the same quantity in both, derived rather than named.
   */
  params?: number;
  /** `CUDA`, `Metal`, `CPU`… — which silicon actually ran it. */
  backend?: string;
}

/**
 * Read whatever the reader pasted.
 *
 * Tries JSON first and falls back to the markdown table, because both arrive and neither is the
 * outlier: markdown is what `llama-bench` prints by default *and* what the panel's own command asks
 * for, and JSON is what a reader who wanted the richer fields ran instead. Returns an empty
 * list rather than throwing: a paste that is not llama-bench output is a mistake to report on the
 * surface, not an exception.
 */
export function parseLlamaBench(text: string): readonly Measurement[] {
  const trimmed = text.trim();
  if (trimmed === '') return [];
  return parseJson(trimmed) ?? parseMarkdown(trimmed);
}

/**
 * The richer format, and **not** the one the emitted command asks for — `launch.ts` emits `-o md`,
 * because the block beside it is read by a person before it is run by one. So the reader who
 * follows the panel arrives here with markdown and no `build_commit`, and {@link submissionUrl} is
 * what asks for the JSON re-run — in the issue body, at the point the missing commit would have
 * been printed. (`describeMismatch` asks for one too, and for a reason that is worse than it reads:
 * {@link parseMarkdown} never assigns `kvTypes` at all, so **every** markdown paste is
 * cache-unverifiable — including the one the panel's own `-ctk q8_0 -ctv q8_0 -o md` produces, where
 * llama-bench really did print the columns. See #181.) Worth revisiting as a pair rather than in
 * either file: whichever way it goes, the emitter and this comment have to agree.
 *
 * `-o json` produces an array of objects carrying every field the CSV header lists, so the depth,
 * the layer count and the build commit are all present as data rather than reconstructed from a
 * label. `undefined` — not an empty array — when the text is not JSON at all, so the caller can
 * distinguish "not JSON" from "JSON with no benchmark rows in it" and fall through to markdown.
 */
function parseJson(text: string): readonly Measurement[] | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (!Array.isArray(parsed)) return undefined;

  const rows: Measurement[] = [];
  for (const entry of parsed) {
    if (typeof entry !== 'object' || entry === null) continue;
    const row = entry as Record<string, unknown>;
    const rate = numberOf(row.avg_ts);
    if (rate === undefined) continue;

    const prompt = numberOf(row.n_prompt) ?? 0;
    const gen = numberOf(row.n_gen) ?? 0;
    /**
     * **A `-pg` row is dropped, not read as prefill**, and the first version read it as prefill on
     * a comment that was simply wrong. `llama-bench` computes a row's rate as
     * `(n_prompt + n_gen) / time`, so a combined row's `avg_ts` is a *blend* — and one dominated by
     * the slow half: 7,000 t/s of prefill and 100 t/s of decode come out around 473. Put beside
     * `prefillTokensPerSec` that is a 93% miss with nothing to mark it, and submittable as
     * calibration evidence. There is no way to recover the two rates from one number, so the row is
     * not a measurement of either.
     *
     * The markdown path was already safe by accident: `pp512+tg128` fails the anchored label
     * pattern and is skipped. This makes the two agree deliberately.
     */
    const kind =
      prompt > 0 && gen > 0 ? undefined : prompt > 0 ? 'prefill' : gen > 0 ? 'decode' : undefined;
    if (kind === undefined) continue;

    const depth = numberOf(row.n_depth);
    rows.push({
      kind,
      tokens: kind === 'prefill' ? prompt : gen,
      ...(depth !== undefined && depth > 0 ? { depthTokens: depth } : {}),
      tokensPerSec: rate,
      ...(numberOf(row.stddev_ts) === undefined ? {} : { stddev: numberOf(row.stddev_ts)! }),
      ...(typeof row.build_commit === 'string' ? { buildCommit: row.build_commit } : {}),
      ...(numberOf(row.n_gpu_layers) === undefined
        ? {}
        : { gpuLayers: numberOf(row.n_gpu_layers)! }),
      ...(typeof row.type_k === 'string' && typeof row.type_v === 'string'
        ? { kvTypes: { k: row.type_k, v: row.type_v } }
        : {}),
      ...(typeof row.model_type === 'string' ? { modelLabel: row.model_type } : {}),
      ...(numberOf(row.model_n_params) === undefined
        ? {}
        : { params: numberOf(row.model_n_params)! }),
      ...(typeof row.backend === 'string'
        ? { backend: row.backend }
        : typeof row.backends === 'string'
          ? { backend: row.backends }
          : {}),
    });
  }
  return rows;
}

/**
 * The default format.
 *
 * The table is `| model | size | params | backend | ngl | test | t/s |`, and only two of those
 * columns are read — the rest vary by build and by backend, and depending on their positions is how
 * a parser breaks on the next release. The `test` column is matched rather than indexed, so a
 * column added or reordered upstream costs nothing.
 *
 * The `test` label has been spelled `pp 512`, `pp512` and `pp512 @ d512` across versions, so the
 * pattern tolerates the whitespace rather than pinning one spelling.
 */
function parseMarkdown(text: string): readonly Measurement[] {
  const rows: Measurement[] = [];

  for (const line of text.split('\n')) {
    if (!line.includes('|')) continue;
    const cells = line
      .split('|')
      .map((c) => c.trim())
      .filter((c) => c !== '');

    const testCell = cells.find((c) => /^(pp|tg)\s*\d/i.test(c));
    if (testCell === undefined) continue;
    const test = /^(pp|tg)\s*(\d+)(?:\s*@\s*d\s*(\d+))?$/i.exec(testCell);
    if (test === null) continue;

    /**
     * The rate cell, **preferring the one with a `±` in it and falling back to the last number.**
     *
     * The first version took the *first* numeric cell and read `ngl` — 33 — as a throughput of 33
     * tokens per second on a row measuring 7,285. Every column between `model` and `t/s` is a
     * number on some backend, so "shaped like a number" is not a shape that identifies this column.
     * The spread is, when it is there; and `t/s` is last when it is not.
     */
    const numeric = cells.filter((c) => /^[\d.]+(\s*±\s*[\d.]+)?$/.test(c));
    const rateCell = numeric.find((c) => c.includes('±')) ?? numeric[numeric.length - 1];
    if (rateCell === undefined) continue;
    const [rate, spread] = rateCell.split('±').map((part) => Number.parseFloat(part.trim()));
    if (!Number.isFinite(rate)) continue;

    const depth = test[3] === undefined ? undefined : Number.parseInt(test[3], 10);
    /**
     * The first cell, which is the checkpoint as llama.cpp names it — `llama 8B Q4_K - Medium`.
     * That is the only place the markdown format says what was actually loaded, and a paste from a
     * different quantization is otherwise indistinguishable from a disagreement about the model.
     */
    const modelLabel = cells[0] === testCell || cells[0] === rateCell ? undefined : cells[0];
    /**
     * The `ngl` column, which the default output carries and this first discarded — so a run with
     * half the model on the host was accepted as comparable with a fully-resident prediction, and
     * only JSON pastes got the layer check. Found by position rather than by shape because a bare
     * integer is not a shape that identifies it: it is the cell immediately before the `test` one,
     * which is the table's own layout.
     */
    const nglCell = cells[cells.indexOf(testCell) - 1];
    const ngl = nglCell !== undefined && /^\d+$/.test(nglCell) ? Number(nglCell) : undefined;
    /**
     * The `params` and `backend` columns, found by shape rather than by index — `8.03 B` and a bare
     * alphabetic word are each distinctive enough, where a position would break on the next column
     * added. `params` is what actually identifies the model; see `Measurement.params`.
     */
    const paramCell = cells.find((c) => /^[\d.]+\s*B$/i.test(c));
    const params = paramCell === undefined ? undefined : Number.parseFloat(paramCell) * 1e9;
    const backend = cells.find((c) => /^[A-Za-z]+(\/[A-Za-z]+)*$/.test(c) && c !== testCell);
    rows.push({
      kind: test[1].toLowerCase() === 'pp' ? 'prefill' : 'decode',
      ...(modelLabel === undefined ? {} : { modelLabel }),
      ...(ngl === undefined ? {} : { gpuLayers: ngl }),
      ...(params === undefined || !Number.isFinite(params) ? {} : { params }),
      ...(backend === undefined ? {} : { backend }),
      tokens: Number.parseInt(test[2], 10),
      ...(depth !== undefined && depth > 0 ? { depthTokens: depth } : {}),
      tokensPerSec: rate,
      ...(Number.isFinite(spread) ? { stddev: spread } : {}),
    });
  }

  return rows;
}

function numberOf(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

/** What bench said, for the scenario the reader is looking at. */
export interface Prediction {
  /** Machine-wide prompt tokens per second, as `estimatePrefill` reports it. */
  prefillTokensPerSec: number;
  /** Per-user generation rate. */
  decodeTokensPerSec: number;
  promptTokens: number;
  generationTokens: number;
  /** Tokens the archetype assumes are already resident when the *prompt* arrives, if any. */
  cachedPrefixTokens?: number;
  /**
   * The cache `estimateDecode` charges every step against — the scenario's whole window.
   *
   * Not the prompt, which was the first answer: decode is priced at `usage.contextTokens`
   * throughout, so the run that reproduces it holds that much. Defaults to the prompt only so a
   * caller that predates the field is no worse off than before.
   */
  residentContextTokens?: number;
  concurrency: number;
  /**
   * The runtime the figures were priced under.
   *
   * `llama-bench` loads GGUF and speaks for llama.cpp placements alone, which is #139's own stated
   * limit. A vLLM prediction is computed at vLLM's efficiency constants, so a llama-bench number
   * beside it is a cross-runtime pair — and nothing in either figure says so.
   */
  runtimeId: string;
  /** The format the figures were priced at, so a paste of a different one can be caught. */
  quantLabel: string;
  /**
   * The model the figures were priced for.
   *
   * Checking only the *format* let a Llama Q4_K_M measurement pass against a DeepSeek Q4_K_M
   * prediction — and the generated issue then labelled it as the DeepSeek run, which is a wrong
   * data point entering the record under a name that will never be questioned.
   */
  modelName: string;
  /**
   * And its parameter count, which is what actually distinguishes two models.
   *
   * The name check catches a cross-family paste and misses Qwen3 8B against Qwen3 32B, because
   * llama.cpp writes an architecture where the catalog writes a product — they never agree past
   * the first word. Both formats print a parameter count, so this is the same quantity on both
   * sides rather than two spellings of a name.
   */
  totalParams: number;
  /** Which silicon the figures assume, so a Metal paste against an NVIDIA row can be caught. */
  deviceClass: 'discrete-gpu' | 'unified-soc' | 'cpu-ram';
  deviceVendor: string;
  /** True when the engine says this configuration does not run at all. */
  impossible?: boolean;
  /** Cache precision the figures were priced at — llama.cpp's `-ctk`/`-ctv` names. */
  kvType: string;
  /** The model's own layer count, so "all of them" can be recognised however it is spelled. */
  modelLayers: number;
  /**
   * Layers the placement expects on the GPU, when that is unambiguous.
   *
   * Absent means "no claim", and the check is skipped — which is the honest state wherever the
   * placement spills, since converting a spill fraction back into a layer count is the #14 shape
   * and this module is not the place to invent it. The caller states it only for a fully-resident
   * placement, where the answer is every layer.
   */
  gpuLayers?: number;
}

export interface Comparison {
  measurement: Measurement;
  predicted: number;
  /** `measured / predicted - 1`. Positive means bench under-predicted. */
  error: number;
  /** Whether the pair sits inside the band the engine's reference tests assert. */
  withinBand: boolean;
  /**
   * Why this pair is not evidence about the model, when it is not.
   *
   * A measurement of a *different* scenario is noise wearing a data point's chassis, and the three
   * ways that happens are invisible in the numbers. Present means the delta beside it is a
   * difference between two jobs rather than between a prediction and reality.
   */
  mismatch?: string;
}

/** The band the engine's reference tests assert, and therefore the one a submission is judged at. */
export const CALIBRATION_BAND = 0.3;

/**
 * Line up each measured row against what bench predicted for it.
 *
 * **A mismatch is reported, never corrected for.** It would be easy to rescale a `pp512` result to
 * the scenario's own prompt length and present a delta, and it would be wrong twice over: prefill is
 * quadratic so the rescaling is a model rather than an observation, and the whole point of this
 * surface is that the reader can check bench's arithmetic rather than take more of it on trust.
 */
export function compare(
  measurements: readonly Measurement[],
  prediction: Prediction
): readonly Comparison[] {
  return measurements.map((measurement) => {
    const predicted =
      measurement.kind === 'prefill'
        ? prediction.prefillTokensPerSec
        : prediction.decodeTokensPerSec;
    const expectedTokens =
      measurement.kind === 'prefill' ? prediction.promptTokens : prediction.generationTokens;
    /**
     * **The two kinds want different depths, and the first version gave them the same one.**
     *
     * For *prefill* the depth is the archetype's cached prefix: the turn's attention is charged
     * against tokens already resident, and a standalone run measures a different job.
     *
     * For *decode* it is the resident context, which is a different quantity entirely.
     * `estimateDecode` charges every step's cache read at the scenario's whole context — so the run
     * that reproduces it has that much in the cache, and `tg128` from an empty cache is measuring a
     * weight-bound job against a KV-bound prediction. The first version had `expectedDepth = 0` for
     * both, so it *flagged* the correctly-reproduced `tg … -d 8192` and *passed* the empty-cache run
     * that is not comparable at all. Backwards, in the direction that manufactures evidence.
     */
    const expectedDepth =
      measurement.kind === 'prefill'
        ? (prediction.cachedPrefixTokens ?? 0)
        : (prediction.residentContextTokens ?? prediction.promptTokens);

    const mismatch = describeMismatch(measurement, expectedTokens, expectedDepth, prediction);
    const error = predicted > 0 ? measurement.tokensPerSec / predicted - 1 : 0;

    return {
      measurement,
      predicted,
      error,
      /**
       * Judged on the figure as it will *print*, which is this repo's stated rule for every
       * threshold and is load-bearing here rather than tidy. A measurement exactly at the band's
       * edge comes out as 0.30000000000000004 in binary floating point and fails a `<= 0.3` — so
       * the one pair a reader is most likely to look hard at would be reported outside a band it
       * is exactly on. The surface prints whole percents, so that is what the comparison reads.
       */
      withinBand: Math.round(Math.abs(error) * 100) <= CALIBRATION_BAND * 100,
      ...(mismatch === undefined ? {} : { mismatch }),
    };
  });
}

function describeMismatch(
  measurement: Measurement,
  expectedTokens: number,
  expectedDepth: number,
  prediction: Prediction
): string | undefined {
  const reasons: string[] = [];

  /**
   * **A configuration the engine refuses has no prediction to check against.**
   *
   * `impossible` means the cache and activations alone are over the ceiling, so the rates beside it
   * describe a machine that cannot load the model — and any measurement pasted against them was
   * necessarily taken on something else. Comparing at all was the defect: the panel produced a
   * percentage and offered to submit it.
   */
  if (prediction.impossible === true) {
    reasons.push(
      `compared against a configuration this machine cannot run at all, so the predicted rates ` +
        `beside it describe nothing that could have produced this measurement`
    );
  }

  /**
   * The parameter count, which is what distinguishes two models of one family.
   *
   * Ten percent, because the two sides count slightly differently — llama.cpp reports what is in
   * the GGUF and the catalog reports the safetensors index — and because a wrong model is wrong by
   * far more than that: 8B against 32B is a factor of four.
   */
  if (
    measurement.params !== undefined &&
    prediction.totalParams > 0 &&
    Math.abs(measurement.params / prediction.totalParams - 1) > 0.1
  ) {
    reasons.push(
      `run on a ${(measurement.params / 1e9).toFixed(1)}B model where the figures above are for ` +
        `${(prediction.totalParams / 1e9).toFixed(1)}B`
    );
  }

  /**
   * The backend, checked only where it *contradicts* the device rather than against a full mapping.
   *
   * A vendor-to-backend table would be inventing data — llama.cpp's backend names vary by build,
   * and ROCm, SYCL, Vulkan and BLAS all appear. Two contradictions need no table: Metal is Apple's
   * alone, and a CPU-only run cannot have produced a discrete GPU's figures.
   */
  const backend = measurement.backend?.toUpperCase();
  if (backend !== undefined) {
    if (backend.includes('METAL') && prediction.deviceVendor !== 'Apple') {
      reasons.push(
        `run on Metal, which is Apple silicon, where the figures above are for a ${prediction.deviceVendor} device`
      );
    } else if (backend === 'CPU' && prediction.deviceClass === 'discrete-gpu') {
      reasons.push(`run on the CPU where the figures above are for a graphics card`);
    }
  }

  /**
   * **The scope #139 states outright, checked rather than assumed.** `llama-bench` loads GGUF, so
   * it speaks for llama.cpp placements alone — and a vLLM or MLX prediction is computed at that
   * runtime's own efficiency constants. Nothing in either number says the pair crosses runtimes,
   * which is exactly the sort of difference this function exists to name.
   */
  if (prediction.runtimeId !== 'llama.cpp') {
    reasons.push(
      `measured with llama-bench while the figures above are priced under a different runtime — ` +
        `llama-bench loads GGUF and speaks for llama.cpp placements only`
    );
  }

  /**
   * The checkpoint, which decides the weight bytes decode is bound by.
   *
   * Q8_0 against a Q4_K_M prediction is roughly twice the bytes per token on a memory-bound decode,
   * and both formats name the quantization somewhere — `model_type` in JSON, the first table cell
   * in markdown. Matched loosely (llama.cpp writes `Q4_K - Medium` where the catalog writes
   * `Q4_K_M`), because a strict compare would fire on every paste and teach people to ignore it.
   */
  /**
   * **The catalog's label is not the format's name, and matching the whole string marks every
   * paste.** `QuantSpec.label` carries a qualifier for the reader — `MXFP4 (expert-only)`,
   * `FP8 (E4M3)`, `BF16 / FP16` — and llama.cpp prints none of that. So the parenthetical is
   * dropped and a slash offers two spellings, either of which counts as a match.
   *
   * The remainder is then compared with punctuation stripped, because the two write it differently
   * at the same width: the catalog says `Q4_K_M` where llama.cpp says `Q4_K - Medium`, and
   * `Q4KMEDIUM` contains `Q4KM`.
   */
  const keys = prediction.quantLabel
    .split('(')[0]
    .split('/')
    .map((part) => part.toUpperCase().replace(/[^A-Z0-9]/g, ''))
    .filter((part) => part !== '');
  const labelKey = measurement.modelLabel?.toUpperCase().replace(/[^A-Z0-9]/g, '');
  if (labelKey !== undefined && keys.length > 0 && !keys.some((key) => labelKey.includes(key))) {
    reasons.push(
      `run on "${measurement.modelLabel}" where the figures above are for ${prediction.quantLabel}`
    );
  }

  /**
   * The cache precision — **both halves of it**, and equal rather than merely present.
   *
   * The first version asked whether the expected type was *in* the pair, so a mixed run of
   * `K=f16 V=q4_0` passed an f16 prediction on the strength of the K alone. They are charged
   * separately and they are separate flags; a run that matches on one is not a run at this
   * precision.
   */
  if (measurement.kvTypes !== undefined) {
    const { k, v } = measurement.kvTypes;
    if (k !== prediction.kvType || v !== prediction.kvType) {
      reasons.push(
        `run with a ${k === v ? k : `${k}/${v}`} cache where the figures above assume ` +
          `${prediction.kvType}`
      );
    }
  } else if (prediction.kvType !== 'f16') {
    /**
     * **Unverifiable is not the same as matching**, and the first version treated it as such: a
     * paste carrying no cache precision sailed past a Q8 or Q4 prediction.
     *
     * **This branch is reached by every markdown paste, not only a default one** (raised by Codex
     * on #175; the comment here previously claimed the narrower cause and was wrong).
     * `parseMarkdown` has no branch for the cache columns at all, so it never assigns `kvTypes`
     * whether or not llama-bench printed them — and the panel's own measure command passes
     * `-ctk`/`-ctv` explicitly, which makes it print them. So the reader who follows the panel
     * exactly gets told their correctly reproduced run looks like f16. Reading those columns is
     * #181.
     *
     * **The sentence names neither format**, which is the correction after the first one named the
     * wrong one (Codex again, on #175). A JSON row that simply omits `type_k`/`type_v` lands here
     * too — the fixture in the test file does exactly that — so "pasted as markdown, re-run with
     * `-o json`" told a JSON reader to re-run the command they had already run. `Measurement` does
     * not record which parser produced it, so the honest sentence describes the *absence* and names
     * the fields.
     */
    reasons.push(
      `pasted without a stated cache precision — no type_k/type_v — so it cannot be told apart ` +
        `from an f16 run, where the figures above assume ${prediction.kvType}. A JSON run stating ` +
        `those fields is what settles it`
    );
  }

  /**
   * The model itself, which the quant check does not cover.
   *
   * Matched on the model's leading word — llama.cpp writes its own name for an architecture
   * (`llama 8B Q4_K - Medium`, `deepseek2 671B`) and the catalog writes a product name, so nothing
   * stricter survives contact. It is enough for the case that matters: a different *model* at the
   * same format, which otherwise reads as a clean percentage and enters the record under the wrong
   * name.
   */
  const family = prediction.modelName
    .split(/[\s-]/)[0]
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, '');
  if (labelKey !== undefined && family.length > 2 && !labelKey.includes(family)) {
    reasons.push(
      `run on "${measurement.modelLabel}" where the figures above are for ${prediction.modelName}`
    );
  }

  /**
   * The layer split, which is the whole subject of the offload term.
   *
   * A run at `-ngl 20` against a fully-resident prediction is streaming most of the model across
   * the bus, and the prediction is not. Only flagged when the paste states it, since markdown's
   * `ngl` column is not read — this is the one field the parser captured and nothing consulted.
   */
  /**
   * **Two-sided, and it was one-sided.** Only rejecting *fewer* layers than the prediction let
   * every positive `n_gpu_layers` pass a `cpu-ram` scenario, whose predicted count is zero — so the
   * EPYC-shaped measurements this feature exists to collect could be satisfied by a GPU run. And
   * more layers than predicted is its own difference: the prediction charges host streaming the
   * measurement never paid.
   */
  /**
   * **And "all of them" has more than one spelling**, which is where the two halves of this project
   * disagreed with each other. llama.cpp counts the output tensor a position past the repeating
   * blocks, so #136's emitter passes `layers + 1` for a fully-resident placement — and readers type
   * `-ngl 99` for the same thing. Comparing against the layer count alone would have marked a run
   * that followed bench's own command.
   *
   * So a fully-resident prediction accepts anything at or above the layer count, and a partial one
   * is compared exactly.
   */
  if (measurement.gpuLayers !== undefined && prediction.gpuLayers !== undefined) {
    const allResident = prediction.gpuLayers >= prediction.modelLayers;
    const agrees = allResident
      ? measurement.gpuLayers >= prediction.modelLayers
      : measurement.gpuLayers === prediction.gpuLayers;
    if (!agrees) {
      reasons.push(
        `run with ${measurement.gpuLayers} layers on the GPU where the placement above puts ` +
          `${prediction.gpuLayers} of ${prediction.modelLayers} there`
      );
    }
  }

  /**
   * A tolerance rather than equality, and the tolerance is not laziness: `llama-bench` runs at the
   * lengths it is given, and the emitted command gives it the scenario's own — but a reader who
   * typed their own will land near rather than on. Ten percent is close enough that the quadratic
   * term has not moved much and far enough that `pp512` against 16,384 is caught.
   */
  /**
   * **The length is checked for prefill and not for decode**, which the third round got wrong in
   * both directions before settling here.
   *
   * Prefill is quadratic in the prompt, so `pp512` against a 16,384-token prediction is two
   * different jobs. Decode is a steady-state per-token rate — `perUserTokensPerSec` does not depend
   * on how many tokens you ask for — so requiring `n_gen` to match the window's remainder rejected
   * every ordinary `tg128` against a scenario that merely happened to leave 2,192 tokens spare.
   * What *does* matter for decode is the cache it reads, and that is the depth check below.
   */
  if (
    measurement.kind === 'prefill' &&
    expectedTokens > 0 &&
    Math.abs(measurement.tokens / expectedTokens - 1) > 0.1
  ) {
    reasons.push(
      `run at ${measurement.tokens.toLocaleString('en-US')} tokens where the prediction is for ` +
        `${expectedTokens.toLocaleString('en-US')}`
    );
  }

  /**
   * The depth, against whatever *this kind* of measurement's depth ought to be — see the two arms
   * of `expectedDepth` at the call site. The same 10% tolerance as the length, for the same reason:
   * the emitted command supplies the right figure and a reader who typed their own lands near it.
   */
  const depth = measurement.depthTokens ?? 0;
  const depthOff =
    expectedDepth === 0 ? depth > 0 : Math.abs(depth / Math.max(expectedDepth, 1) - 1) > 0.1;
  if (depthOff) {
    reasons.push(
      depth === 0
        ? `run against an empty cache where the prediction charges ` +
            `${expectedDepth.toLocaleString('en-US')} tokens of it — pass -d to reproduce that`
        : expectedDepth === 0
          ? `run at a depth of ${depth.toLocaleString('en-US')} where the prediction has none`
          : `run at a depth of ${depth.toLocaleString('en-US')} where the prediction charges ` +
            `${expectedDepth.toLocaleString('en-US')}`
    );
  }

  /**
   * The asymmetry the engine documents at length, surfaced where it can actually mislead.
   * `prefillTokensPerSec` is machine-wide and scales with concurrency; `llama-bench` measures one
   * sequence and has no concurrency flag at all. So a multi-user prediction and a `pp` row are not
   * the same quantity, and no tolerance makes them one.
   */
  if (prediction.concurrency > 1) {
    /**
     * **Both kinds, and the first version exempted decode on a half-true rationale.**
     *
     * "Decode amortises across the batch" is true of the *weights* and false of the *cache*:
     * `estimateDecode` charges every concurrent sequence's KV read on every step, so
     * `perUserTokensPerSec` at eight users sits well below a solo `tg` wherever the cache is what
     * decode is bound by — which is every long-context scenario. Prefill is the machine-wide rate
     * and scales the other way. Different arithmetic, same conclusion: llama-bench measures one
     * sequence and has no concurrency flag, so neither pair is comparable.
     */
    reasons.push(
      measurement.kind === 'prefill'
        ? `measured on one sequence where the prediction is the machine-wide rate across ` +
            `${prediction.concurrency} users — llama-bench has no concurrency flag`
        : `measured on one sequence where the prediction charges every step for ` +
            `${prediction.concurrency} sequences' cache reads — llama-bench has no concurrency flag`
    );
  }

  if (reasons.length === 0) return undefined;
  return `This pair is ${reasons.join('; and ')}.`;
}

/** Whether there is anything worth submitting — no comparable pair, no submission. */
export function hasSubmittablePair(comparisons: readonly Comparison[]): boolean {
  return comparisons.some((c) => c.mismatch === undefined);
}

/**
 * A pre-filled issue, carrying the scenario rather than a description of one.
 *
 * The querystring already round-trips a scenario, and that is the reproducible half of a data
 * point — so it goes in the body verbatim alongside the measured figures and the build. A
 * measurement that cannot name its scenario is unusable, so the link is only offered once there is
 * one to name.
 *
 * `issues/new` with `title` and `body` is the whole mechanism: no backend, no telemetry, and the
 * reader sees exactly what they are about to post before they post it.
 */
export function submissionUrl(options: {
  repoUrl: string;
  scenarioUrl: string;
  deviceName: string;
  /** How many of them, since "8x RTX 5090" and "RTX 5090" are different machines. */
  deviceCount: number;
  modelName: string;
  comparisons: readonly Comparison[];
}): string {
  const { repoUrl, scenarioUrl, deviceCount, modelName, comparisons } = options;
  // The count belongs in the machine's *name*, not only in the scenario link: the issue title and
  // the Machine field are what a maintainer groups by, and an 8-card run filed as "RTX 5090" is
  // grouped with the single-card ones.
  const deviceName = deviceCount > 1 ? `${deviceCount}x ${options.deviceName}` : options.deviceName;

  /**
   * **Only comparable pairs, and the reason is the whole feature.** A row the panel has just
   * called "not comparable" carries a percentage that is a difference between two *jobs*, and
   * writing it into the issue table strips the explanation and leaves a number that reads as
   * evidence. Filtered here rather than in the caller so no caller can forget.
   */
  const usable = comparisons.filter((c) => c.mismatch === undefined);
  const rows = usable.map((c) => {
    const kind = c.measurement.kind === 'prefill' ? 'prefill' : 'decode';
    return (
      `| ${kind} | ${c.measurement.tokens} | ${c.measurement.depthTokens ?? 0} | ` +
      `${c.predicted.toFixed(1)} | ${c.measurement.tokensPerSec.toFixed(1)} | ` +
      `${(c.error * 100).toFixed(0)}% |`
    );
  });

  const build = usable.find((c) => c.measurement.buildCommit !== undefined)?.measurement
    .buildCommit;

  const body = [
    `**Scenario:** ${scenarioUrl}`,
    '',
    `**Machine:** ${deviceName}`,
    `**Model:** ${modelName}`,
    // Named as missing rather than omitted, so a maintainer reading the record can tell a build
    // nobody recorded from one nobody asked for. A llama.cpp from six months ago is a different
    // runtime for calibration purposes.
    `**llama.cpp build:** ${build ?? '(not in the pasted output — re-run with `-o json` to capture it)'}`,
    '',
    '| measure | tokens | depth | predicted t/s | measured t/s | error |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
    ...rows,
    '',
    '<!-- The scenario link above is what ties this to a device row; please keep it. -->',
  ].join('\n');

  const params = new URLSearchParams({
    title: `calibration: ${modelName} on ${deviceName}`,
    body,
    labels: 'calibration',
  });
  return `${repoUrl}/issues/new?${params.toString()}`;
}
