import { useMemo } from 'react';
import { DEVICES, MODELS, RUNTIMES, evaluateConfig, useConfig } from '@/store/config';
import { QUANTS } from '@/data/quants';
import { CATALOG_GENERATED_AT, getDevice, getModel } from '@/data/catalog';
import { BudgetBar } from './BudgetBar';
import { Telemetry } from './Telemetry';
import { Workloads } from './Workloads';
import { Segmented, Select, StopSlider } from './Controls';
import { compact, gibLabel, params, tokens } from '@/lib/format';
import type { KvPrecision } from '@/engine/types';

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

  /** Whether the configuration runs at all. */
  const runnable = !evaluation.placement.unsupported && !evaluation.placement.impossible;
  /**
   * Whether a *speed* claim is defensible, which is a stricter question than whether it runs.
   * A discrete GPU that spills most of its weights over PCIe is runnable and slow — DeepSeek V3
   * on a 5090 offloads 93% and decodes at 3.9 tok/s — so "would still run fast" contradicts the
   * verdict tiles directly above it.
   */
  const fast = runnable && evaluation.placement.offloadFraction === 0;
  /** Only discrete GPUs shard a model across devices. */
  const shardable = device.class === 'discrete-gpu';

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
        note:
          d.status !== 'shipping'
            ? `${d.status === 'rumored' ? 'Rumoured' : 'Announced'} — specs may change`
            : undefined,
      })),
    []
  );

  const runtimeOptions = useMemo(() => {
    const device = DEVICES.find((d) => d.id === config.deviceId);
    return RUNTIMES.map((r) => ({
      value: r.id,
      label: r.label,
      note:
        device && !r.supports.includes(device.class)
          ? `Does not run on ${device.name}`
          : r.preallocFraction
            ? `Reserves ${Math.round(r.preallocFraction * 100)}% of the device up front`
            : undefined,
    }));
  }, [config.deviceId]);

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
          options={QUANTS.map((q) => ({ value: q.id, label: q.label, note: q.qualityNote }))}
        />
        <Select
          label="Runtime"
          value={config.runtimeId}
          onChange={(v) => set('runtimeId', v)}
          options={runtimeOptions}
        />
      </section>

      {/* The hero, the three answers it does not collapse into one, and what they add up to. */}
      <BudgetBar evaluation={evaluation} />
      <Telemetry evaluation={evaluation} canOffload={shardable} />
      <Workloads evaluation={evaluation} config={config} />

      {/* Usage: the half of the question that is about you, not the hardware. */}
      <section aria-label="Usage" className="panel grid gap-5 p-5 sm:grid-cols-2">
        <StopSlider
          label="Context per sequence"
          stops={CONTEXT_STOPS}
          value={nearestStop(CONTEXT_STOPS, config.contextTokens)}
          onChange={(v) => set('contextTokens', v)}
          format={tokens}
        />
        <StopSlider
          label="Concurrent users"
          stops={CONCURRENCY_STOPS}
          value={nearestStop(CONCURRENCY_STOPS, config.concurrency)}
          onChange={(v) => set('concurrency', v)}
          format={(v) => String(v)}
        />
        <StopSlider
          label="Prompt length"
          stops={PROMPT_STOPS}
          value={nearestStop(PROMPT_STOPS, config.promptTokens)}
          onChange={(v) => set('promptTokens', v)}
          format={tokens}
        />
        <Segmented
          label="KV precision"
          value={config.kvPrecision}
          onChange={(v) => set('kvPrecision', v)}
          options={KV_PRECISIONS}
        />
        {shardable ? (
          <StopSlider
            label="Device count"
            stops={DEVICE_COUNT_STOPS}
            value={nearestStop(DEVICE_COUNT_STOPS, config.deviceCount)}
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
            {fast
              ? ', so it decodes at roughly that model size rather than its full one.'
              : evaluation.placement.offloadFraction > 0
                ? '. That would make it fast — but not here, with most of the weights crossing the host bus every token.'
                : '. That is why a model this size can be fast anywhere it does fit.'}{' '}
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
