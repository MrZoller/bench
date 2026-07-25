import { useMemo } from 'react';
import { DEVICES, MODELS, RUNTIMES, evaluateConfig, useConfig } from '@/store/config';
import { getRuntime, runtimeDrives } from '@/data/runtimes';
import { QUANTS } from '@/data/quants';
import { CATALOG_GENERATED_AT, getDevice, getModel } from '@/data/catalog';
import { BudgetBar } from './BudgetBar';
import { Telemetry } from './Telemetry';
import { Segmented, Select, StopSlider } from './Controls';
import { compact, gibLabel, params, percent, tokens } from '@/lib/format';
import type { KvPrecision } from '@/engine/types';
import { canShard, maxAllocatablePerDevice, raisingCeilingWouldHelp } from '@/engine/placement';
import { classifyDecode } from '@/lib/verdicts';
import { quantApplies } from '@/lib/quantChoice';

/**
 * The Bench — the hero surface.
 *
 * Direct manipulation: pick a model and hardware, drag usage, watch the budget fill. The engine
 * is pure arithmetic over a handful of numbers, so every control recomputes the whole scenario
 * on change; there is no submit step, because the point is to feel where the cliff is rather
 * than to query for it.
 */

/**
 * Log-spaced stops. The interesting jumps in context are 4K -> 32K -> 128K, so a linear range
 * would spend most of its travel in a region nobody is deciding between.
 */
const CONTEXT_STOPS = [
  2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
] as const;
const CONCURRENCY_STOPS = [1, 2, 4, 8, 16, 32, 64, 128] as const;
const PROMPT_STOPS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;
const DEVICE_COUNT_STOPS = [1, 2, 4, 8] as const;

const KV_PRECISIONS: readonly { value: KvPrecision; label: string }[] = [
  { value: 'fp16', label: 'FP16' },
  { value: 'q8', label: 'Q8' },
  { value: 'q4', label: 'Q4' },
];

export function Bench() {
  const config = useConfig();
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
  const contextStops = useMemo(() => {
    const withinModel = CONTEXT_STOPS.filter((t) => t < model.maxContext);
    // The stored value is always a stop, even when it is not one of the fixed ones. Switching
    // from a 40,960-token model to a larger one keeps that 40,960 — `coerce` only caps — so a
    // list of "fixed stops plus this model's maximum" would show 32K for a context the engine
    // is evaluating at 40,960. Including it means the control cannot display a value the
    // engine is not using.
    const stops = new Set([...withinModel, model.maxContext, config.contextTokens]);
    return [...stops].filter((t) => t <= model.maxContext).sort((a, b) => a - b);
  }, [model.maxContext, config.contextTokens]);

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
        note: q.qualityNote,
      })),
    [model, device, runtime]
  );

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
          // The override note takes precedence: three models carry a hand-entered totalParams,
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
        // Combined rather than ranked: the Ryzen AI Max+ is both tunable *and* carries a note
        // that the engine uses its measured 213 GB/s instead of the 256 GB/s sticker — a 17%
        // difference in every throughput figure. Ranking these dropped the provenance.
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

  const runtimeOptions = useMemo(
    () =>
      RUNTIMES.map((r) => ({
        value: r.id,
        label: r.label,
        // The note is the control's accessible description, so a "does not run here" warning
        // has to live in it — a screen-reader user tabbing the picker hears nothing otherwise.
        note: !runtimeDrives(r, device)
          ? `Does not run on ${device.name}`
          : r.preallocFraction
            ? `Reserves ${Math.round(r.preallocFraction * 100)}% of the device up front`
            : undefined,
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
        // could actually pass on a command line.
        label: runtime.kvLabels?.[k.value] ?? k.label,
      })),
    [runtime]
  );

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-5 p-4 sm:p-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">bench</h1>
          <p className="mt-0.5 text-sm text-[var(--color-text-muted)]">
            What runs on your hardware, and how comfortably.
          </p>
        </div>
        <p className="max-w-md text-xs leading-relaxed text-[var(--color-text-faint)]">
          Estimates from a roofline model calibrated against published measurements. Treat them as a
          band, not a promise. Model catalog generated{' '}
          <time dateTime={CATALOG_GENERATED_AT}>
            {new Date(CATALOG_GENERATED_AT).toISOString().slice(0, 10)}
          </time>
          .
        </p>
      </header>

      {/* Setup: what you are running, and on what. */}
      <section aria-label="Configuration" className="panel grid gap-4 p-5 sm:grid-cols-2">
        <Select
          label="Model"
          value={config.modelId}
          onChange={(v) => set('modelId', v)}
          options={modelOptions}
        />
        <Select
          label="Hardware"
          value={config.deviceId}
          onChange={(v) => set('deviceId', v)}
          options={deviceOptions}
        />
        <Select
          label="Quantization"
          value={config.quantId}
          onChange={(v) => set('quantId', v)}
          options={quantOptions}
        />
        <Select
          label="Runtime"
          value={config.runtimeId}
          onChange={(v) => set('runtimeId', v)}
          options={runtimeOptions}
        />
      </section>

      {/* The hero, and the three answers it does not collapse into one. */}
      {/* Both read `canOffload` from the same expression, so the bar and the tiles cannot describe
          one placement two different ways — which they did, over exactly this distinction. */}
      <BudgetBar evaluation={evaluation} canOffload={device.class === 'discrete-gpu'} />
      <Telemetry
        evaluation={evaluation}
        canOffload={device.class === 'discrete-gpu'}
        tunableCeiling={raisingCeilingWouldHelp(device, evaluation.placement.usedBytesPerDevice)}
      />

      {/* Usage: the half of the question that is about you, not the hardware. */}
      <section aria-label="Usage" className="panel grid gap-5 p-5 sm:grid-cols-2">
        <StopSlider
          label="Context per sequence"
          stops={contextStops}
          value={nearestStop(contextStops, config.contextTokens)}
          onChange={(v) => set('contextTokens', v)}
          format={tokens}
        />
        <StopSlider
          label="Concurrent users"
          stops={concurrencyStops}
          value={nearestStop(concurrencyStops, config.concurrency)}
          onChange={(v) => set('concurrency', v)}
          format={(v) => String(v)}
        />
        <StopSlider
          label="Prompt length"
          stops={promptStops}
          value={nearestStop(promptStops, config.promptTokens)}
          onChange={(v) => set('promptTokens', v)}
          format={tokens}
        />
        <Segmented
          label="KV precision"
          value={config.kvPrecision}
          onChange={(v) => set('kvPrecision', v)}
          options={kvOptions}
        />
        {shardable ? (
          <StopSlider
            label="Device count"
            stops={deviceCountStops}
            value={nearestStop(deviceCountStops, config.deviceCount)}
            onChange={(v) => set('deviceCount', v)}
            format={(v) => `${v}x`}
          />
        ) : (
          <p className="self-end text-xs text-[var(--color-text-muted)]">
            Single machine. Tensor-parallel sharding needs a transport between devices, which
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
        <aside className="panel p-5 text-sm leading-relaxed text-[var(--color-text-muted)]">
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

/** The fixed stops plus whatever is currently stored, so the control can always show it. */
function withStored(stops: readonly number[], stored: number): number[] {
  return [...new Set([...stops, stored])].sort((a, b) => a - b);
}

/** Snap an arbitrary value (from a URL, say) to the nearest slider stop. */
function nearestStop<T extends number>(stops: readonly T[], value: number): T {
  return stops.reduce((best, stop) =>
    Math.abs(stop - value) < Math.abs(best - value) ? stop : best
  );
}
