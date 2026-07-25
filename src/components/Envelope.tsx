import { useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  computeEnvelope,
  type CellState,
  type EnvelopeCell,
  type EnvelopeGrid,
} from '@/engine/envelope';
import { getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime } from '@/data/runtimes';
import { CONCURRENCY_STOPS, contextStopsFor, withStored } from '@/lib/stops';
import { colors, marks, withAlpha } from '@/design/tokens';
import {
  CAPACITY_TIGHT,
  DECODE_USABLE,
  TTFT_TOLERABLE,
  parseDisplayedSeconds,
} from '@/lib/verdicts';
import { rate, seconds, tokens, uniqueLabels } from '@/lib/format';
import type { Config } from '@/store/scenario';

/**
 * The Envelope: how much room is left, as a region rather than a number.
 *
 * The Bench answers "does this work at my settings". People actually want to know how far they
 * can push before it stops working, and that is a shape — context on X, concurrent users on Y,
 * the two things a deployment grows in and the two that multiply into the KV cache.
 *
 * Drawn on canvas rather than as DOM cells because the grid is dense and redrawn on every
 * slider frame; a table view carries the same figures for anyone who cannot use the picture.
 */

/**
 * Ordinal, not sequential: these are four named states, not four points on a magnitude scale,
 * so they take categorical hues rather than steps of one ramp.
 *
 * These are the reserved *status* colours, not the budget series — `tokens.ts` validates that
 * trio, and this set was measured separately: worst normal-vision pair is serious/critical at
 * ΔE 20.7, worst CVD is good/warning at ~13.7 under protanopia. Colour is not the only channel
 * regardless; every state carries a label and a written hint.
 */
const STATE_STYLE: Record<CellState, { fill: string; label: string; hint: string }> = {
  comfortable: {
    fill: colors.good,
    label: 'Comfortable',
    hint: 'Fits with room, types fast enough to read along, and starts answering promptly.',
  },
  tight: {
    fill: colors.warning,
    label: 'Tight',
    hint: 'Runs, but near the ceiling, slow to type, or slow to start — the table says which.',
  },
  offloaded: {
    fill: colors.serious,
    label: 'Spilling to RAM',
    hint: 'Loads only because weights cross the host bus every token.',
  },
  over: { fill: colors.critical, label: 'Will not run', hint: 'Past what this hardware can hold.' },
  unsupported: {
    fill: colors.critical,
    label: 'Runtime cannot drive it',
    // Same colour as `over` — both mean "no" — but a different sentence, because the two need
    // opposite advice and the legend previously gave the memory one to both.
    hint: 'This runtime does not support this hardware, at any size.',
  },
};

export function Envelope({ config }: { config: Config }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const headingId = useId();
  const [showTable, setShowTable] = useState(false);

  const model = getModel(config.modelId);

  const grid = useMemo<EnvelopeGrid>(
    () =>
      computeEnvelope({
        model,
        quant: getQuant(config.quantId),
        runtime: getRuntime(config.runtimeId),
        rig: { device: getDevice(config.deviceId), count: config.deviceCount },
        usage: {
          contextTokens: config.contextTokens,
          concurrency: config.concurrency,
          promptTokens: config.promptTokens,
          kvPrecision: config.kvPrecision,
        },
        // Only contexts the model can reach — a column past its limit would be shaded from an
        // evaluation the model cannot perform — *and* always the one currently selected, so the
        // ring below marks a cell that was really computed. Snapping it to the nearest axis
        // value instead put a green marker under three "Will not run" tiles at 128 users.
        contexts: contextStopsFor(model.maxContext, config.contextTokens),
        concurrencies: withStored(CONCURRENCY_STOPS, config.concurrency),
        usableTokensPerSec: DECODE_USABLE,
        tightUtilization: CAPACITY_TIGHT,
        usableTtftSeconds: TTFT_TOLERABLE,
        // Classified on the printed figure, so a cell never reads "Tight · 15 tok/s" against a
        // threshold of 15 while the Telemetry tile calls the same number "Usable".
        displayedRate: (n) => Number(rate(n)),
        // Same reason, for latency: a cell must not be painted green on 10.3s while the tile
        // beside it prints "10 s" and calls it the edge of tolerable.
        displayedTtft: (n) => parseDisplayedSeconds(seconds(n), n),
      }),
    [config, model]
  );

  // Bumped by a ResizeObserver so the effect below redraws at the new size. Without it the
  // bitmap is stretched until the next config change — sharpness, and a distorted ring.
  /**
   * Column headers, disambiguated against each other rather than formatted one at a time — a
   * hand-edited `?ctx=131073` sits beside 131,072 and both round to "128K" otherwise, and the
   * colliding column is the one the "you are here" marker is meant to identify.
   */
  const contextLabels = useMemo(() => uniqueLabels(grid.contexts), [grid.contexts]);

  const [resizeTick, setResizeTick] = useState(0);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => setResizeTick((n) => n + 1));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw at device resolution so the cell edges stay crisp on a retina display.
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const cols = grid.contexts.length;
    const rows = grid.concurrencies.length;
    if (cols === 0 || rows === 0) return;

    const cellW = width / cols;
    const cellH = height / rows;

    grid.cells.forEach((row, r) => {
      row.forEach((cell, c) => {
        // Rows are drawn bottom-up: concurrency increases upward, as an axis should.
        const x = c * cellW;
        const y = height - (r + 1) * cellH;
        ctx.fillStyle = STATE_STYLE[cell.state].fill;
        // The 2px spacer, so adjacent cells never bleed into one continuous wash.
        ctx.fillRect(x, y, Math.max(1, cellW - marks.gap), Math.max(1, cellH - marks.gap));
      });
    });

    // "You are here" — the Bench's current scenario. Exact indexes, not nearest: the axes now
    // contain the selected values, so a miss means something is genuinely off-grid and the ring
    // should be absent rather than approximate.
    const ci = grid.contexts.indexOf(Math.min(config.contextTokens, model.maxContext));
    const ri = grid.concurrencies.indexOf(config.concurrency);
    if (ci >= 0 && ri >= 0) {
      const x = ci * cellW + (cellW - marks.gap) / 2;
      const y = height - (ri + 1) * cellH + (cellH - marks.gap) / 2;

      // A ring, not a filled dot: the cell's own colour has to stay readable underneath it.
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = marks.lineWidth;
      ctx.beginPath();
      ctx.arc(x, y, Math.min(cellW, cellH) / 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = withAlpha(colors.bg, 0.9);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [grid, config.contextTokens, config.concurrency, model.maxContext, resizeTick]);

  const counts = grid.cells.flat().reduce<Record<string, number>>((acc, cell) => {
    acc[cell.state] = (acc[cell.state] ?? 0) + 1;
    return acc;
  }, {});
  const total = grid.cells.flat().length;
  const currentCell = grid.cells.flat().find((c) => isCurrent(c, config, model));

  return (
    <section aria-labelledby={headingId} className="panel p-5">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={headingId} className="text-sm font-semibold tracking-wide">
          How much room is left
          <span className="ml-2 font-normal text-[var(--color-text-faint)]">
            context against concurrent users
          </span>
        </h2>
        <p className="text-sm whitespace-nowrap text-[var(--color-text-muted)]">
          <span className="tabular text-[var(--color-text)]">{counts.comfortable ?? 0}</span> of{' '}
          {total} comfortable
        </p>
      </header>

      <div className="mt-4 flex gap-2">
        {/*
          One track per row, so each label sits at its cell's centre. `justify-between` puts the
          first and last at the extremes instead — which pushed "1" below the grid entirely and
          left every other label straddling a boundary.
        */}
        <ol
          aria-hidden="true"
          className="tabular grid h-48 text-[10px] text-[var(--color-text-faint)]"
          style={{ gridTemplateRows: `repeat(${grid.concurrencies.length}, 1fr)` }}
        >
          {[...grid.concurrencies].reverse().map((n) => (
            <li key={n} className="flex items-center justify-end pr-1">
              {n}
            </li>
          ))}
        </ol>

        <div className="flex-1">
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={describe(grid, counts, total, currentCell)}
            className="h-48 w-full rounded"
          />
          <ol
            aria-hidden="true"
            className="tabular mt-1 grid text-[10px] text-[var(--color-text-faint)]"
            style={{ gridTemplateColumns: `repeat(${grid.contexts.length}, 1fr)` }}
          >
            {grid.contexts.map((c, i) => (
              <li key={c} className="text-center">
                {contextLabels[i]}
              </li>
            ))}
          </ol>
        </div>
      </div>

      {/*
        The hint carries what the colour cannot: "Tight" means three unrelated things.

        Only the states actually on screen. A legend is a key to this picture, not a catalogue of
        everything the engine can return — and `unsupported` deliberately shares `over`'s red, so
        listing both unconditionally would put two identical swatches under different sentences.
        They never co-occur: an unsupported runtime closes the whole grid.
      */}
      <ul className="mt-4 grid gap-x-5 gap-y-2 sm:grid-cols-2">
        {(Object.keys(STATE_STYLE) as CellState[])
          .filter((state) => counts[state])
          .map((state) => (
            <li key={state} className="flex items-baseline gap-2 text-sm">
              <span
                aria-hidden="true"
                className="mt-1 inline-block h-3 w-3 shrink-0 rounded-sm"
                style={{ background: STATE_STYLE[state].fill }}
              />
              <span>
                <span className="text-[var(--color-text)]">{STATE_STYLE[state].label}</span>{' '}
                <span className="text-xs text-[var(--color-text-muted)]">
                  {STATE_STYLE[state].hint}
                </span>
              </span>
            </li>
          ))}
      </ul>

      <button
        type="button"
        onClick={() => setShowTable((v) => !v)}
        aria-expanded={showTable}
        className="mt-4 text-xs text-[var(--color-accent)] underline underline-offset-2"
      >
        {showTable ? 'Hide' : 'Show'} the region as a table
      </button>

      {showTable && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">
              Feasibility by context length and concurrent users
            </caption>
            <thead>
              <tr className="text-[var(--color-text-faint)]">
                <th scope="col" className="py-1 pr-3 font-normal">
                  Users
                </th>
                {grid.contexts.map((c, i) => (
                  <th key={c} scope="col" className="py-1 pr-3 text-right font-normal">
                    {contextLabels[i]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-muted)]">
              {[...grid.cells].reverse().map((row, i) => (
                <tr key={i} className="border-t border-[var(--color-border)]">
                  <th
                    scope="row"
                    className="tabular py-1 pr-3 font-normal text-[var(--color-text)]"
                  >
                    {grid.concurrencies[grid.concurrencies.length - 1 - i]}
                  </th>
                  {row.map((cell) => (
                    <td
                      key={cell.contextTokens}
                      className={`tabular py-1 pr-3 text-right ${
                        isCurrent(cell, config, model) ? 'text-[var(--color-text)]' : ''
                      }`}
                    >
                      {isCurrent(cell, config, model) && (
                        <span className="mr-1 text-[var(--color-accent)]">▸</span>
                      )}
                      {describeCell(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

/** A sentence for a screen reader: the shape carries the meaning, so it has to be said. */
function describe(
  grid: EnvelopeGrid,
  counts: Record<string, number>,
  total: number,
  current: EnvelopeCell | undefined
): string {
  // The ring is the only mark on this panel with no textual equivalent, so it goes first.
  const here = current
    ? `Currently at ${tokens(current.contextTokens)} context and ${current.concurrency} ${
        current.concurrency === 1 ? 'user' : 'users'
      }: ${STATE_STYLE[current.state].label.toLowerCase()}. `
    : '';

  const comfortable = counts.comfortable ?? 0;
  if (comfortable === 0) {
    // Two different sentences, because the two failures need opposite advice.
    if (counts.unsupported) {
      return `${here}This runtime cannot drive this hardware, so none of the ${total} combinations run.`;
    }
    return `${here}No comfortable configuration in this range. ${counts.over ?? 0} of ${total} combinations will not run at all.`;
  }
  const widest = grid.cells[0].filter((c) => c.state === 'comfortable').at(-1);
  return (
    `${here}${comfortable} of ${total} combinations of context and concurrency are comfortable. ` +
    `At ${grid.concurrencies[0]} user, up to ${tokens(widest?.contextTokens ?? 0)} of context stays comfortable.`
  );
}

/** What a cell says in the table: its state, why, and what it costs. */
function describeCell(cell: EnvelopeCell): string {
  if (cell.state === 'over' || cell.state === 'unsupported') return STATE_STYLE[cell.state].label;
  const why =
    cell.tightBecause === 'capacity'
      ? ' (near the ceiling)'
      : cell.tightBecause === 'speed'
        ? ' (slow)'
        : cell.tightBecause === 'latency'
          ? ' (slow to start)'
          : '';
  return `${STATE_STYLE[cell.state].label}${why} · ${rate(cell.tokensPerSec)} tok/s, ${seconds(
    cell.ttftSeconds
  )} to first token`;
}

/** Whether this cell is the scenario the Bench is currently showing. */
function isCurrent(cell: EnvelopeCell, config: Config, model: { maxContext: number }): boolean {
  return (
    cell.contextTokens === Math.min(config.contextTokens, model.maxContext) &&
    cell.concurrency === config.concurrency
  );
}
