import { useId, useMemo, useState } from 'react';
import {
  computeMatrix,
  measureMax,
  measureValue,
  type MatrixCell,
  type MatrixMeasure,
} from '@/engine/matrix';
import { DEVICES, MODELS, getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime } from '@/data/runtimes';
import { quantApplies } from '@/lib/quantChoice';
import { sequential } from '@/design/tokens';
import { params, percent, rate, seconds } from '@/lib/format';
import { useConfig, type Config } from '@/store/config';

/**
 * Every model against every device — the surface for "what are my options", which is the
 * question that comes before the one the Bench answers.
 *
 * Three measures, switchable over the same grid, because that is the clearest way to show that
 * "fits" and "usable" are different questions. Toggling between them visibly rearranges which
 * hardware looks good, which is the capacity/bandwidth/compute triangle made concrete rather
 * than asserted in prose.
 */

const MEASURES: readonly { value: MatrixMeasure; label: string; hint: string }[] = [
  { value: 'fit', label: 'Does it fit', hint: 'Headroom left after weights, cache and overhead.' },
  { value: 'decode', label: 'How fast', hint: 'Tokens per second for one user.' },
  { value: 'ttft', label: 'How responsive', hint: 'Time until the first token appears.' },
];

/**
 * The sequential ramp, reversed.
 *
 * `sequential` runs light to dark for a light surface. On this chassis the darkest step is the
 * one that recedes into the panel, so higher values have to be *brighter* — otherwise the best
 * cells are the ones you cannot see.
 */
const RAMP = [...sequential].reverse();

/**
 * What a row is evaluated at when the selected format does not apply to it.
 *
 * Q4_K_M rather than the store's BF16 fallback. That one is chosen for *safety* — it always
 * applies — but on a grid meant to compare hardware it makes every dense model look far worse
 * than anyone would actually run it, which is the opposite of informative. Q4_K_M is the
 * default local trade and the honest stand-in.
 */
const SUBSTITUTE_QUANT_ID = 'q4_k_m';

export function Matrix({ config }: { config: Config }) {
  const headingId = useId();
  const [measure, setMeasure] = useState<MatrixMeasure>('fit');
  const set = useConfig((s) => s.set);

  const quant = getQuant(config.quantId);
  const runtime = getRuntime(config.runtimeId);

  const models = useMemo(
    () =>
      [...MODELS].sort((a, b) => (b.popularity?.downloads ?? 0) - (a.popularity?.downloads ?? 0)),
    []
  );
  // Shipping hardware only: a rumoured row would put speculative specs into a comparison people
  // read as a shortlist, and `status` exists precisely so that never happens silently.
  const devices = useMemo(() => DEVICES.filter((d) => d.status === 'shipping'), []);

  /**
   * The selected format where it applies, and a universal one where it does not.
   *
   * Forcing one format across the grid blanked more than half the rows at the default config:
   * MXFP4 is expert-only, so every dense model reported "does not apply" — a quantization fact
   * standing in for a hardware comparison, on the one surface whose job is comparing hardware.
   * Substituting keeps every row informative, and the substitution is stated rather than hidden.
   */
  const quantFor = useMemo(
    () => (model: (typeof models)[number], device: (typeof devices)[number]) =>
      quantApplies(quant, model, device) ? quant : getQuant(SUBSTITUTE_QUANT_ID),
    [quant]
  );

  const substituted = useMemo(
    () => models.some((m) => devices.some((d) => !quantApplies(quant, m, d))),
    [models, devices, quant]
  );

  const cells = useMemo(
    () =>
      computeMatrix({
        models,
        devices,
        runtime,
        usage: {
          contextTokens: config.contextTokens,
          concurrency: config.concurrency,
          promptTokens: config.promptTokens,
          kvPrecision: config.kvPrecision,
        },
        deviceCount: 1,
        quantFor,
      }),
    [models, devices, quantFor, runtime, config]
  );

  const max = measureMax(cells, measure);
  const runnable = cells.flat().filter((c) => c.runs).length;

  return (
    <section aria-labelledby={headingId} className="panel p-5">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
        <h2 id={headingId} className="text-sm font-semibold tracking-wide">
          Every model on every machine
          <span className="ml-2 font-normal text-[var(--color-text-faint)]">
            at {quant.label}
            {substituted &&
              `, ${getQuant(SUBSTITUTE_QUANT_ID).label} where it does not apply`}, {runtime.label}
          </span>
        </h2>
        <p className="text-sm whitespace-nowrap text-[var(--color-text-muted)]">
          <span className="tabular text-[var(--color-text)]">{runnable}</span> of{' '}
          {cells.flat().length} combinations run
        </p>
      </header>

      {/* One filter row above the grid, as the dataviz guidance puts it. */}
      <fieldset className="mt-4">
        <legend className="sr-only">Colour the grid by</legend>
        <div className="flex flex-wrap gap-1 rounded-md border border-[var(--color-control-border)] bg-[var(--color-surface-raised)] p-1">
          {MEASURES.map((m) => (
            <button
              key={m.value}
              type="button"
              aria-pressed={m.value === measure}
              onClick={() => setMeasure(m.value)}
              className={`rounded px-3 py-1 text-sm transition-colors ${
                m.value === measure
                  ? 'bg-[var(--color-accent-dim)] text-[var(--color-text)]'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-1.5 text-xs text-[var(--color-text-muted)]">
          {MEASURES.find((m) => m.value === measure)?.hint} Switching between these rearranges which
          hardware looks good — that disagreement is the whole point.
        </p>
      </fieldset>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full border-separate border-spacing-0.5 text-left text-xs">
          <caption className="sr-only">
            Every catalogued model against every shipping device, coloured by{' '}
            {MEASURES.find((m) => m.value === measure)?.label}. {runnable} of {cells.flat().length}{' '}
            combinations run.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="sticky left-0 bg-[var(--color-surface)] pr-2 font-normal">
                <span className="sr-only">Model</span>
              </th>
              {devices.map((d) => (
                <th
                  key={d.id}
                  scope="col"
                  // Fixed width, and the label taken out of flow below, so a long name cannot
                  // stretch its own column — "RTX PRO 6000 Blackwell" was three times the width
                  // of its neighbours and skewed the whole grid.
                  className="relative h-24 w-7 min-w-7 p-0 align-bottom font-normal text-[var(--color-text-faint)]"
                >
                  {/*
                    Rotated rather than truncated. Horizontally these clipped to "GeForc…" four
                    times over — a header that cannot distinguish its own columns is worse than
                    none, and the names are what make the grid readable at all.
                  */}
                  <span
                    className="absolute bottom-1 left-1/2 origin-bottom-left -rotate-45 whitespace-nowrap"
                    title={d.name}
                  >
                    {shortName(d.name)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model, r) => (
              <tr key={model.id}>
                <th
                  scope="row"
                  className="sticky left-0 max-w-[9rem] truncate bg-[var(--color-surface)] pr-2 font-normal text-[var(--color-text-muted)]"
                  title={`${model.name} — ${params(model.totalParams)}`}
                >
                  {model.name}
                </th>
                {cells[r].map((cell, c) => (
                  <td key={devices[c].id} className="p-0">
                    <button
                      type="button"
                      // Clicking loads the pair into the Bench, which is where the detail lives.
                      onClick={() => {
                        set('modelId', cell.modelId);
                        set('deviceId', cell.deviceId);
                      }}
                      title={tooltip(cell, measure, quant.id)}
                      aria-label={tooltip(cell, measure, quant.id)}
                      className={`h-7 w-full rounded-sm focus:ring-2 focus:ring-[var(--color-accent)] focus:outline-none ${
                        cell.runs ? '' : 'border border-dashed border-[var(--color-border)]'
                      }`}
                      style={{ background: fill(cell, measure, max) }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* A ramp legend, since a continuous scale has no discrete keys to list. */}
      <div className="mt-4 flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
        <span>worse</span>
        <span aria-hidden="true" className="flex h-3 flex-1 overflow-hidden rounded-sm">
          {RAMP.map((step) => (
            <span key={step} className="flex-1" style={{ background: step }} />
          ))}
        </span>
        <span>better</span>
        <span className="ml-3 flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-3 w-3 rounded-sm border border-dashed border-[var(--color-border)]"
          />
          will not run
        </span>
      </div>
    </section>
  );
}

/**
 * A device name with the parts every row shares removed.
 *
 * "GeForce RTX 5090" and "GeForce RTX 5080" differ in one character at the end, so a truncating
 * header shows the same string for both. The vendor line is the redundant part.
 */
function shortName(name: string): string {
  return name.replace(/^(GeForce|Instinct|Radeon)\s+/, '');
}

/** Colour for a cell: a step of the ramp, or the recessive "did not run" fill. */
function fill(cell: MatrixCell, measure: MatrixMeasure, max: number): string {
  const value = measureValue(cell, measure);
  // Absence is not a low score. A pair that cannot run gets the empty fill, so the ramp is only
  // ever read across things that actually ran.
  // A hole, not a dark value: the panel surface, so an unrunnable pair is never mistaken for a
  // poor score at the bottom of a ramp whose darkest step is also nearly black.
  if (value === undefined || max <= 0) return 'transparent';

  /**
   * Log-scaled, because these ranges span orders of magnitude: decode runs from ~2 tok/s on a
   * CPU host to ~300 on a B200, and a linear ramp spends every step but the last on the top
   * device while the whole rest of the grid sits in one indistinguishable dark band. The
   * comparison people need is "is this twice as fast", not "what fraction of the best is it".
   */
  const index = Math.min(
    RAMP.length - 1,
    Math.floor((Math.log1p(value) / Math.log1p(max)) * RAMP.length)
  );
  return RAMP[index];
}

/** What a cell says on hover, and to a screen reader. Never colour alone. */
function tooltip(cell: MatrixCell, measure: MatrixMeasure, selectedQuantId: string): string {
  const model = getModel(cell.modelId).name;
  const device = getDevice(cell.deviceId).name;
  if (!cell.runs) return `${model} on ${device}: ${cell.blockedBy ?? 'does not run'}.`;

  const detail =
    measure === 'fit'
      ? cell.offloadFraction > 0
        ? `runs only by spilling ${percent(cell.offloadFraction)} of its weights to host RAM`
        : `${percent(Math.max(0, 1 - cell.utilization))} of the ceiling free`
      : measure === 'decode'
        ? `${rate(cell.tokensPerSec)} tok/s per user`
        : `${seconds(cell.ttftSeconds)} to first token`;

  const at = cell.quantId === selectedQuantId ? '' : ` at ${getQuant(cell.quantId).label}`;
  return `${model} on ${device}${at}: ${detail}.`;
}
