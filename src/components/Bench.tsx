import { useMemo } from 'react';
import { DEVICES, MODELS, RUNTIMES, evaluateConfig, useConfig } from '@/store/config';
import { useUrlSync } from '@/store/useUrlSync';
import { getRuntime, kvSubstitutionFor, runtimeDrives, substitutionFor } from '@/data/runtimes';
import { QUANTS, getQuant } from '@/data/quants';
import { getDevice, getModel } from '@/data/catalog';
import { BudgetBar } from './BudgetBar';
import { Telemetry } from './Telemetry';
import { Workloads } from './Workloads';
import { Envelope } from './Envelope';
import { DETAIL_ANCHOR_ID, Matrix } from './Matrix';
import { Segmented, Select, StopSlider } from './Controls';
import { compact, gibLabel, params, percent, tokens } from '@/lib/format';
import {
  canShard,
  maxAllocatablePerDevice,
  raisingCeilingWouldHelp,
  wasEvaluated,
} from '@/engine/placement';
import { classifyDecode } from '@/lib/verdicts';
import { quantApplies } from '@/lib/quantChoice';
import {
  CONCURRENCY_STOPS,
  DEVICE_COUNT_STOPS,
  KV_PRECISIONS,
  PROMPT_STOPS,
  SETTING_LABELS,
  SETTING_NOTES,
  contextStopsFor,
  deviceCountNote,
  kvLabel,
  withStored,
} from '@/lib/stops';

/**
 * The Bench — the hero surface.
 *
 * Direct manipulation: pick a model and hardware, drag usage, watch the budget fill. The engine
 * is pure arithmetic over a handful of numbers, so every control recomputes the whole scenario
 * on change; there is no submit step, because the point is to feel where the cliff is rather
 * than to query for it.
 */

export function Bench() {
  const config = useConfig();
  useUrlSync();
  const set = useConfig((s) => s.set);

  const evaluation = useMemo(() => evaluateConfig(config), [config]);
  const model = getModel(config.modelId);
  const device = getDevice(config.deviceId);
  const runtime = getRuntime(config.runtimeId);

  /**
   * Context stops, capped at what this model supports.
   *
   * Built per model rather than fixed, because `coerce` clamps the stored context to
   * `model.maxContext` and a fixed list would then show a value the engine is not using — drag
   * a 40,960-token Qwen to 64K and the store holds 40,960 while the slider reads 32K, with the
   * budget bar and throughput computed for neither.
   */
  const contextStops = useMemo(
    () => contextStopsFor(model.maxContext, config.contextTokens),
    [model.maxContext, config.contextTokens]
  );

  /**
   * Every discrete control includes whatever is stored, for the same reason the context slider
   * does: `coerce` accepts any integer in range, so a value from a URL — `?u=3` — would be
   * evaluated as three users while the slider displayed two.
   */
  const concurrencyStops = useMemo(
    () => withStored(CONCURRENCY_STOPS, config.concurrency),
    [config.concurrency]
  );
  const deviceCountStops = useMemo(
    () => withStored(DEVICE_COUNT_STOPS, config.deviceCount),
    [config.deviceCount]
  );

  /** The prompt is part of the context, so it cannot be offered beyond it. */
  const promptStops = useMemo(() => {
    const within = PROMPT_STOPS.filter((t) => t < config.contextTokens);
    // Same rule: whatever is stored has to be selectable, or the label lies about the estimate.
    const stops = new Set([...within, config.contextTokens, config.promptTokens]);
    return [...stops].filter((t) => t <= config.contextTokens).sort((a, b) => a - b);
  }, [config.contextTokens, config.promptTokens]);

  /** Formats that cannot run here, or would do nothing here. See `quantApplies`. */
  const quantOptions = useMemo(
    () =>
      QUANTS.filter((q) => quantApplies(q, model, device, runtime)).map((q) => ({
        value: q.id,
        label: q.label,
        // A short claim, not the whole derivation — the panel below carries that, and printing the
        // same forty words twice on one screen taught people to skip both. `Select` renders only
        // the *selected* option's note, so this was never what informs a choice between formats
        // anyway; what it does is tag the control that caused the panel.
        note:
          [
            substitutionFor(runtime, q.id) && `Stand-in for a format ${runtime.label} cannot load.`,
            q.qualityNote,
          ]
            .filter(Boolean)
            .join(' ') || undefined,
      })),
    [model, device, runtime]
  );

  /**
   * Set when the memory and speed figures on this page derive from a format the runtime cannot
   * actually load.
   *
   * The engine cannot tell — a roofline consumes bits per weight, and a stand-in of the right width
   * produces plausible arithmetic either way — which is exactly why it has to be said out loud. The
   * same rule `devices.json` already follows for pre-release specs: an approximation that is
   * documented is a modelling choice, and an approximation that is invisible is invented data.
   *
   * Gated on `wasEvaluated`, because the banner's first clause promises "the figures below" and
   * there are none when the runtime cannot drive the device: pick Q4_K_M on a 5090 under llama.cpp,
   * switch to MLX, and BudgetBar, Telemetry, Workloads and the Envelope all render a refusal while
   * this asserted their arithmetic was sound for a width nothing used.
   *
   * Not `runnable`, which is the trap on the other side. A configuration that was measured and came
   * up short — DeepSeek V3 on a 256 GB Mac at Q4_K_M, drawn at 382 GiB over a 192 GiB bar — got
   * every one of those figures from the stand-in's width and has to stay marked. That is the same
   * distinction the Matrix legend draws, and gating on "does it run" is the polarity error that was
   * fixed there earlier in this PR. Raised by Codex on PR #32.
   */
  const substitution = wasEvaluated(evaluation.placement)
    ? substitutionFor(runtime, config.quantId)
    : undefined;

  /**
   * The same question about the cache, kept as its own value because it is its own claim.
   *
   * The two are independent in both directions, which is exactly why folding them together hid
   * this one for a release: MLX at Q4_K_M with an FP16 cache substitutes only the weights, and MLX
   * at BF16 with an 8-bit cache substitutes only the cache — and that second combination showed no
   * marker at all, on a page whose every memory figure included a byte nobody measured (#33).
   */
  const kvSubstitution = wasEvaluated(evaluation.placement)
    ? kvSubstitutionFor(runtime, config.kvPrecision)
    : undefined;

  /** Whether the configuration runs at all. */
  const runnable = !evaluation.placement.unsupported && !evaluation.placement.impossible;
  /**
   * Whether a *speed* claim is defensible — a stricter question than whether it runs, and one I
   * have now got wrong in three different ways.
   *
   * Offload is not the only route to slow: DeepSeek V3 at Q4 fits an EPYC 9654 with nothing
   * spilled, and decodes at ~10 tok/s, which the tile beside this correctly calls "Slow".
   *
   * The threshold is imported rather than repeated. Holding a local copy is how this ended up
   * claiming "fast" across the 15-30 band that the tile calls merely "Usable" — the fifth way
   * this one sentence has managed to contradict the number printed beside it.
   */
  const fast = runnable && classifyDecode(evaluation.decode.perUserTokensPerSec).isFast;
  /**
   * Sharding needs a transport between devices, which is what `interconnect` records — not the
   * device class. Keying off the class disabled it for the DGX Spark, whose catalog row
   * declares ConnectX-7 200GbE and which `tpEfficiency` already models as a network link; the
   * two-Spark cluster is the case that hardware exists to serve.
   *
   * Deliberately separate from `canOffload` below, which really is a discrete-GPU property:
   * spilling needs a slower *tier*, sharding needs a *link*, and only one device has one
   * without the other.
   */
  const shardable = canShard(device);

  const modelOptions = useMemo(
    () =>
      [...MODELS]
        .sort((a, b) => (b.popularity?.downloads ?? 0) - (a.popularity?.downloads ?? 0))
        .map((m) => ({
          value: m.id,
          label: `${m.name} — ${params(m.totalParams)}${
            m.expertParams > 0 ? ` (${params(m.activeParams)} active)` : ''
          }`,
          // The override note takes precedence: six models carry a hand-entered totalParams,
          // and every figure on screen derives from it. That provenance outranks a download count.
          note:
            m.overrideNote ??
            (m.popularity && m.popularity.downloads > 0
              ? `${compact(m.popularity.downloads)} downloads/mo${
                  m.popularity.measuredOn ? ` on ${m.popularity.measuredOn}` : ''
                }`
              : undefined),
        })),
    []
  );

  const deviceOptions = useMemo(
    () =>
      DEVICES.map((d) => ({
        value: d.id,
        label: `${d.name} — ${gibLabel(d.capacityBytes)}`,
        // Pre-release specs must stay visibly labelled, not silently mixed in with shipping ones.
        // The tunable note matters for the same reason in reverse: the ceiling is a default, and
        // treating it as a hardware limit turns a raiseable setting into a flat "will not run".
        // Status warning first, then the tunable ceiling, then whatever the curator wrote. The
        // last of those was being dropped entirely — including the 3090's note that estimates
        // assume PCIe and do not model its optional NVLink bridge, which is precisely the
        // caveat an owner of a bridged pair needs.
        // Combined rather than ranked: the Ryzen AI Max+ is both tunable *and* carries a note —
        // that its 256 GB/s is AMD's rating, real workloads land near 213, and the engine charges
        // that gap through its calibration constants rather than the catalog. Ranking these
        // dropped the provenance.
        note:
          [
            d.status !== 'shipping'
              ? `${d.status === 'rumored' ? 'Rumoured' : 'Announced'} — specs may change`
              : undefined,
            d.allocatableTunable && maxAllocatablePerDevice(d) > d.allocatableBytes
              ? `${gibLabel(d.allocatableBytes)} allocatable by default, raiseable to ${gibLabel(
                  maxAllocatablePerDevice(d)
                )}`
              : undefined,
            d.note,
          ]
            .filter(Boolean)
            .join(' ') || undefined,
      })),
    []
  );

  /**
   * The runtimes, each with something to say about itself at every scenario.
   *
   * The note is the control's accessible description, so a "does not run here" warning has to live
   * in it — a screen-reader user tabbing the picker hears nothing otherwise. That is also why the
   * first clause below is unconditional. `Select` renders only the *selected* option's note, and
   * the two conditions this used to hold — unsupported hardware, and a runtime that preallocates —
   * are both false for llama.cpp on any machine it drives. So at the default scenario the Runtime
   * picker emitted no `aria-describedby` at all, and a description that exists only for vLLM
   * appears and vanishes as the choice moves: the same defect #80 tabulated for the Usage sliders,
   * in the panel #80 cited as doing the opposite (found in review of that fix).
   *
   * `nativeLowPrecision` is the fact worth spending the sentence on. `runtimes.ts` calls it "the
   * single biggest lever on time-to-first-token, and no VRAM calculator models it" — llama.cpp
   * dequantizes every GGUF to fp16 before the matmul, so a Blackwell card's FP4 headline is
   * unreachable from it, and prefill was overstated 8x when this was inferred from bit width.
   * Every runtime has the field, so every option has a sentence, which is what stops the
   * description flickering.
   *
   * Not the multi-device layout, which is the other always-present fact: `parallelism` is what
   * `deviceCountNote` says under the Device count slider, and saying it twice on one screen is how
   * two copies of one claim come to disagree. MLX would also be the wrong place to say it — it
   * declares `layer` because the field is required, and no Apple machine in the catalog has an
   * interconnect, so it never divides anything.
   */
  const runtimeOptions = useMemo(
    () =>
      RUNTIMES.map((r) => ({
        value: r.id,
        label: r.label,
        note: !runtimeDrives(r, device)
          ? `Does not run on ${device.name}.`
          : [
              /* "every weight" was wrong in a configuration two clicks away. BF16 is a real format
                 here — MLX coerces to it — and there is nothing to dequantize when the checkpoint is
                 already FP16-or-wider, so the claim was false for the one selection where it is
                 easiest to check. `nativeLowPrecision` describes what the runtime does with a
                 *quantized* checkpoint, which is what the sentence now says. Left as a capability
                 rather than derived from `config.quant`: this is an option list, and each note
                 describes the runtime a reader has not selected yet. */
              r.nativeLowPrecision
                ? 'Sends low-precision weights straight to the tensor cores.'
                : 'Dequantizes a quantized checkpoint to FP16 before the matmul, so a card’s low-precision peak is out of reach.',
              r.preallocFraction
                ? `Reserves ${Math.round(r.preallocFraction * 100)}% of the device up front.`
                : undefined,
            ]
              .filter(Boolean)
              .join(' '),
      })),
    [device]
  );

  /**
   * KV precisions the selected runtime can actually store.
   *
   * vLLM's `--kv-cache-dtype` takes native or FP8 variants and has no 4-bit cache; offering one
   * charged 0.5 bytes per element and could turn a long-context OOM into a reported fit.
   */
  const kvOptions = useMemo(
    () =>
      KV_PRECISIONS.filter((k) => runtime.kvPrecisions.includes(k.value)).map((k) => ({
        ...k,
        // A runtime's own name for the format wins, so the control names something the user
        // could actually pass on a command line. Shared with the Matrix heading, which had its
        // own resolution and a different fallback.
        label: kvLabel(runtime, k.value),
      })),
    [runtime]
  );

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-5 p-[min(1rem,4vw)] sm:p-6">
      {/* Setup: what you are running, and on what.

          Labels from `SETTING_LABELS`, like the Usage panel below: these controls own the wording,
          they just no longer own the only copy of it. The Matrix already names one of these settings
          a second time — its row axis is an `sr-only` "Model" — and agrees with the control here by
          coincidence rather than by construction, which is the coincidence `kvLabel` was written
          after. */}
      <section
        aria-label="Configuration"
        className="panel grid gap-4 p-[min(1.25rem,5vw)] sm:grid-cols-2"
      >
        <Select
          label={SETTING_LABELS.modelId}
          value={config.modelId}
          onChange={(v) => set('modelId', v)}
          options={modelOptions}
        />
        <Select
          label={SETTING_LABELS.deviceId}
          value={config.deviceId}
          onChange={(v) => set('deviceId', v)}
          options={deviceOptions}
        />
        <Select
          label={SETTING_LABELS.quantId}
          value={config.quantId}
          onChange={(v) => set('quantId', v)}
          options={quantOptions}
        />
        <Select
          label={SETTING_LABELS.runtimeId}
          value={config.runtimeId}
          onChange={(v) => set('runtimeId', v)}
          options={runtimeOptions}
        />
      </section>

      {/* The hero, the three answers it does not collapse into one, and what they add up to.
          The bar and the tiles read `canOffload` from the same expression, so they cannot describe
          one placement two different ways — which they did, over exactly this distinction.

          The anchor is where a Matrix click scrolls back to: the detail it loads sits several
          sections above the grid, so without one the viewport stayed on an unchanged Matrix and
          the click looked like it had done nothing. */}
      {/* `h-0 -mb-5` rather than `contents`: `display: contents` generates no principal box, and
          scrollIntoView returns early for an element without one — so the anchor was silently a
          no-op in every real browser while jsdom, which has no scrollIntoView at all, could never
          show it. Zero height with the flex `gap-5` cancelled costs no layout. */}
      <div id={DETAIL_ANCHOR_ID} aria-hidden="true" className="h-0 -mb-5" />

      {/* Above every figure it applies to, rather than tucked under the picker that caused it.
          The picker's note tells someone choosing a format; this tells someone *reading a number*,
          which is a different person arriving at a different moment — usually from a shared link
          that chose the format for them. Warning tone rather than critical: the arithmetic is sound
          for the width it was given, and what is uncertain is whether the width is right. */}
      {(substitution || kvSubstitution) && (
        <div
          role="note"
          className="panel flex flex-col gap-2 border-[var(--color-warning)] p-[min(1rem,4vw)] text-sm leading-relaxed text-[var(--color-text-muted)]"
        >
          {substitution && (
            <p>
              <span aria-hidden="true" className="text-[var(--color-warning)]">
                ◐{' '}
              </span>
              The memory and speed figures below are derived from a format {runtime.label} cannot
              load. {substitution} They use {getQuant(config.quantId).label}’s{' '}
              {getQuant(config.quantId).bpw} bpw, and the arithmetic is sound for that width;
              whether it is the width {runtime.label} would really use is the approximation.
            </p>
          )}
          {/* One panel, two paragraphs, rather than two panels: they are the same kind of caveat
              about the same set of figures, and stacking two identical warning boxes reads as two
              problems. Each keeps its own ◐ so neither is skimmed as a continuation of the other,
              and either can appear without the other. */}
          {kvSubstitution && (
            <p>
              <span aria-hidden="true" className="text-[var(--color-warning)]">
                ◐{' '}
              </span>
              The cache is charged {kvLabel(runtime, config.kvPrecision)} at its nominal width.{' '}
              {kvSubstitution} The cache is what pushes a long-context configuration over, so this
              one errs towards reporting a fit.
            </p>
          )}
        </div>
      )}

      <BudgetBar evaluation={evaluation} canOffload={device.class === 'discrete-gpu'} />
      <Telemetry
        evaluation={evaluation}
        canOffload={device.class === 'discrete-gpu'}
        tunableCeiling={raisingCeilingWouldHelp(device, evaluation.placement.usedBytesPerDevice)}
      />
      <Workloads evaluation={evaluation} config={config} />
      <Envelope config={config} />
      <Matrix config={config} />

      {/* Usage: the half of the question that is about you, not the hardware.

          The labels come from `SETTING_LABELS` rather than being written here, because the Envelope
          draws two of these settings as its axes and titles them with the same words. The notes come
          from `SETTING_NOTES` for the same reason and one more: these five controls *are* the
          KV-cache argument — context times users times bits per token is most of what the budget bar
          draws — and until they carried a sentence each, the panel's whole text content was the
          labels and the values, with the argument made only in `Envelope.tsx`'s docstring (#80). */}
      <section aria-label="Usage" className="panel grid gap-5 p-[min(1.25rem,5vw)] sm:grid-cols-2">
        <StopSlider
          label={SETTING_LABELS.contextTokens}
          stops={contextStops}
          value={nearestStop(contextStops, config.contextTokens)}
          onChange={(v) => set('contextTokens', v)}
          format={tokens}
          note={SETTING_NOTES.contextTokens}
        />
        <StopSlider
          label={SETTING_LABELS.concurrency}
          stops={concurrencyStops}
          value={nearestStop(concurrencyStops, config.concurrency)}
          onChange={(v) => set('concurrency', v)}
          format={(v) => String(v)}
          note={SETTING_NOTES.concurrency}
        />
        {/* The note is the only place the coupling with the context is stated: `promptStops` above
            filters the prompt to what the context can hold, so dragging the context down drags the
            prompt with it — a jump nothing on screen explained, from a slider a reader could
            reasonably have read as *additional* to the context rather than part of it. */}
        <StopSlider
          label={SETTING_LABELS.promptTokens}
          stops={promptStops}
          value={nearestStop(promptStops, config.promptTokens)}
          onChange={(v) => set('promptTokens', v)}
          format={tokens}
          note={SETTING_NOTES.promptTokens}
        />
        <Segmented
          label={SETTING_LABELS.kvPrecision}
          value={config.kvPrecision}
          onChange={(v) => set('kvPrecision', v)}
          options={kvOptions}
          note={SETTING_NOTES.kvPrecision}
        />
        {shardable ? (
          /* The note goes on the *control*, which is the branch that had no prose at all. The
             explanation below is about the absence of the control and cannot double as its
             description — a sentence that renders only where there is nothing to configure is how
             this panel came to hold exactly one explanatory line and hide it from everyone who
             could act on it.

             It is the one note that reads the runtime, because what a second device buys is the one
             thing here that the runtime decides: `deviceCountNote` derives it from `parallelism`,
             the same field `achievedBandwidth` short-circuits on. */
          <StopSlider
            label={SETTING_LABELS.deviceCount}
            stops={deviceCountStops}
            value={nearestStop(deviceCountStops, config.deviceCount)}
            onChange={(v) => set('deviceCount', v)}
            format={(v) => `${v}x`}
            note={deviceCountNote(runtime, runtimeDrives(runtime, device))}
          />
        ) : (
          /* Any split needs a link, not only a tensor-parallel one — `canShard` is
             `interconnect !== undefined` and asks nothing about the runtime. Naming one layout here
             said the layer split was available on a Mac, which is the same conflation the note above
             was carrying in the other direction. */
          <p className="self-end text-xs text-[var(--color-text-muted)]">
            Single machine. Sharding a model across devices needs a transport between them, which
            unified-memory and CPU hosts do not have.
          </p>
        )}
      </section>

      {/*
       * The teaching moment. Total versus active parameters is the most misunderstood thing in
       * local inference, and an MoE model makes it visceral: a huge weights block next to a
       * small per-token read. Shown only when the distinction exists.
       */}
      {model.expertParams > 0 && (
        <aside className="panel p-[min(1.25rem,5vw)] text-sm leading-relaxed text-[var(--color-text-muted)]">
          <h2 className="mb-1 text-sm font-semibold text-[var(--color-text)]">
            {/*
              The heading follows the verdict, and `fits` alone is not the verdict: it is computed
              even when the runtime cannot drive the device at all, which would put "why this fits
              but still runs fast" directly under three tiles reading "Unsupported". When the
              configuration cannot run, the architecture lesson still stands — the speed claim
              does not, so the heading drops it.
            */}
            {fast
              ? evaluation.placement.fits
                ? 'Why this fits but still runs fast'
                : 'Why this is heavy but would still run fast'
              : `How ${model.name} is put together`}
          </h2>
          <p>
            {model.name} holds{' '}
            <strong className="text-[var(--color-text)]">{params(model.totalParams)}</strong> of
            weights, so all of them occupy memory — but routes each token through only{' '}
            <strong className="text-[var(--color-text)]">{params(model.activeParams)}</strong>
            {/*
              Every branch below reads a decode estimate, so all of them are gated on `runnable`
              — not just `fast`. When the runtime cannot drive the device or the model cannot be
              placed, `evaluate` still returns numbers, and they describe nothing: an unsupported
              MLX-on-5090 selection blamed host-bus spill, and a vLLM-on-Mac one pointed at a
              decode tile that reads "Unsupported". Gating the heading and the classification but
              not the sentences left the aside asserting what the tile beside it refuses to.
            */}
            {!runnable
              ? '. Whether that is fast is not a question this configuration reaches — it does not run as selected.'
              : fast
                ? ', so it decodes at roughly that model size rather than its full one.'
                : evaluation.placement.offloadFraction > 0
                  ? // Only claimed when the engine's own resident estimate agrees: a model can
                    // spill *and* still be slow with everything resident, and blaming the spill
                    // then sends someone to buy memory that will not fix it.
                    classifyDecode(
                      evaluation.decode.offloadPenalty?.withoutOffloadTokensPerSec ?? 0
                    ).isFast
                    ? `. That would make it fast — but not here, with ${percent(
                        evaluation.placement.offloadFraction
                      )} of the weights crossing the host bus every token.`
                    : '. Even resident it would be slow here, so fitting it is not the whole story.'
                  : '. Whether that is fast depends on the memory it is reading from, which the decode figure above measures.'}{' '}
            Total parameters set what fits; active parameters set how fast it feels.
          </p>
          {model.experts && (
            <p className="mt-2">
              Raising concurrency erodes that: one token picks {model.experts.perToken} of{' '}
              {model.experts.total} experts, but a batch collectively picks most of them, so an MoE
              gains far less from batching than a dense model of the same active size.
            </p>
          )}
        </aside>
      )}
    </div>
  );
}

/** Snap an arbitrary value (from a URL, say) to the nearest slider stop. */
function nearestStop<T extends number>(stops: readonly T[], value: number): T {
  return stops.reduce((best, stop) =>
    Math.abs(stop - value) < Math.abs(best - value) ? stop : best
  );
}
