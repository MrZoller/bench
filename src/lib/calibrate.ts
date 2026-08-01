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
 * here** — it carries `build_commit`, which is the version-skew guard #139 names, plus `n_prompt`,
 * `n_gen` and `n_depth` as numbers rather than as a string to re-parse. The markdown reader exists
 * because the default output is markdown and a reader who has already run the tool should not have
 * to run it again.
 *
 * The pasted text never leaves the page. The submission is a `github.com/…/issues/new` URL the
 * reader chooses to open, which is the same no-backend shape the weekly catalog refresh already
 * proved out.
 *
 * ## What a measurement has to carry before it means anything
 *
 * A measurement that cannot name its scenario is unusable for calibration, and three of the four
 * ways that happens are invisible in the numbers themselves:
 *
 *   - **A different prompt length.** Prefill is quadratic in the prompt, so `pp512` against a
 *     prediction made at 16,384 tokens is not a disagreement about the model — it is two different
 *     jobs. {@link compare} marks that rather than reporting a delta.
 *   - **A different depth.** `estimatePrefill` charges an agent turn's attention against a resident
 *     prefix, so a standalone `pp` run measures a different workload than the prediction. `-d` is
 *     what reproduces it, and `n_depth` is what says whether it was used.
 *   - **A different build.** A llama.cpp from six months ago is a different runtime for calibration
 *     purposes. `build_commit` is captured when the paste carries it and its absence is stated.
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
  /** llama.cpp's own commit, from JSON output only. The version-skew guard. */
  buildCommit?: string;
  /** `-ngl`, where the format carried it — the layer split the run actually used. */
  gpuLayers?: number;
  /** `-ctk`/`-ctv`, from JSON. The cache precision the run used, which changes both bytes and rate. */
  kvTypes?: { k: string; v: string };
  /** However the format names the checkpoint — `model_type`, or the markdown table's first column. */
  modelLabel?: string;
}

/**
 * Read whatever the reader pasted.
 *
 * Tries JSON first and falls back to the markdown table, because a reader who ran the tool with its
 * defaults has markdown and a reader who followed the emitted command has JSON. Returns an empty
 * list rather than throwing: a paste that is not llama-bench output is a mistake to report on the
 * surface, not an exception.
 */
export function parseLlamaBench(text: string): readonly Measurement[] {
  const trimmed = text.trim();
  if (trimmed === '') return [];
  return parseJson(trimmed) ?? parseMarkdown(trimmed);
}

/**
 * The richer format, and the one the emitted command asks for.
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
    rows.push({
      kind: test[1].toLowerCase() === 'pp' ? 'prefill' : 'decode',
      ...(modelLabel === undefined ? {} : { modelLabel }),
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
  /** Cache precision the figures were priced at — llama.cpp's `-ctk`/`-ctv` names. */
  kvType: string;
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
        : prediction.promptTokens;

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

  // The cache precision, which changes the bytes *and* the rate. JSON carries it; markdown does not.
  if (measurement.kvTypes !== undefined) {
    const used = new Set([measurement.kvTypes.k, measurement.kvTypes.v]);
    if (!used.has(prediction.kvType)) {
      reasons.push(
        `run with a ${[...used].join('/')} cache where the figures above assume ` +
          `${prediction.kvType}`
      );
    }
  }

  /**
   * The layer split, which is the whole subject of the offload term.
   *
   * A run at `-ngl 20` against a fully-resident prediction is streaming most of the model across
   * the bus, and the prediction is not. Only flagged when the paste states it, since markdown's
   * `ngl` column is not read — this is the one field the parser captured and nothing consulted.
   */
  if (
    measurement.gpuLayers !== undefined &&
    prediction.gpuLayers !== undefined &&
    measurement.gpuLayers < prediction.gpuLayers
  ) {
    reasons.push(
      `run with ${measurement.gpuLayers} layers on the GPU where the placement above puts ` +
        `${prediction.gpuLayers} there`
    );
  }

  /**
   * A tolerance rather than equality, and the tolerance is not laziness: `llama-bench` runs at the
   * lengths it is given, and the emitted command gives it the scenario's own — but a reader who
   * typed their own will land near rather than on. Ten percent is close enough that the quadratic
   * term has not moved much and far enough that `pp512` against 16,384 is caught.
   */
  if (Math.abs(measurement.tokens / Math.max(expectedTokens, 1) - 1) > 0.1) {
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
  modelName: string;
  comparisons: readonly Comparison[];
}): string {
  const { repoUrl, scenarioUrl, deviceName, modelName, comparisons } = options;

  const rows = comparisons.map((c) => {
    const kind = c.measurement.kind === 'prefill' ? 'prefill' : 'decode';
    return (
      `| ${kind} | ${c.measurement.tokens} | ${c.measurement.depthTokens ?? 0} | ` +
      `${c.predicted.toFixed(1)} | ${c.measurement.tokensPerSec.toFixed(1)} | ` +
      `${(c.error * 100).toFixed(0)}% |`
    );
  });

  const build = comparisons.find((c) => c.measurement.buildCommit !== undefined)?.measurement
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
