import { useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  computeEnvelope,
  type CellState,
  type EnvelopeCell,
  type EnvelopeGrid,
} from '@/engine/envelope';
import { measureOf, type Measure } from '@/engine/measure';
import { getDevice, getModel } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime } from '@/data/runtimes';
import {
  CONCURRENCY_STOPS,
  MEASURES,
  SETTING_LABELS,
  contextStopsFor,
  withStored,
} from '@/lib/stops';
import { colors, magnitudeFill, magnitudeRamp, marks, withAlpha } from '@/design/tokens';
import {
  CAPACITY_TIGHT,
  DECODE_USABLE,
  HOST_RAM_UNCHECKED,
  TTFT_RESPONSIVE,
  parseDisplayedSeconds,
} from '@/lib/verdicts';
import { rate, seconds, tokens, uniqueLabels } from '@/lib/format';
import { PanelCount } from './PanelCount';
import { DisclosureToggle } from './DisclosureToggle';
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
 *
 * **The fill is a magnitude, and the verdict deliberately is not the fill** (#65). Painted from the
 * three-state classification, the default scenario came out 45 of 56 cells in one amber with one
 * cell green: a channel carrying almost nothing, in front of readings that span 20x of decode and
 * 561x of first-token latency. And *none* of the amber was amber for capacity — the tight-on-capacity
 * band is the last tenth of the ceiling, while both axes double per step, so the grid goes from 82%
 * of the ceiling straight to 112% and the one threshold that could have drawn a boundary out of the
 * two quantities these axes measure is skipped in every column. The field titled "how much room is
 * left" was coloured entirely by speed and latency.
 *
 * So the fill takes a step of `magnitudeRamp` under a measure the reader chooses, exactly as the
 * Matrix does: the same three questions over different axes, the same control, the same `MEASURES`.
 * Capacity gets a whole channel instead of a threshold nothing lands in. The verdict is not lost —
 * it is in this panel's count, in every row of the table, in the ring's sentence and in the canvas
 * description, which are the channels that can carry a word.
 */

/**
 * What each state is called — and, for the three that are not magnitudes, what colour it is.
 *
 * **A `fill` here means "this state is not a point on any scale".** `over` and `unsupported` are
 * refusals, and `offloaded` is a different placement rather than a worse one: its weights cross the
 * host bus every token, which is a structural fact a reader can act on and the most common
 * explanation for a machine being mysteriously slow. Grading those on a ramp would be answering
 * "how much room is left" for a configuration that has already run out of it. Everything else — the
 * cells that fit resident, whether the verdict called them comfortable or tight — is a magnitude,
 * and takes its colour from `fieldFill`.
 *
 * The three that keep a colour keep the reserved *status* hues, not the budget series: `tokens.ts`
 * validates that trio, and this set was measured separately — worst normal-vision pair is
 * serious/critical at ΔE 20.7, worst CVD is good/warning at ~13.7 under protanopia. Both hues still
 * in use sit in a different family from the blue ramp, so a state cannot read as a step of it.
 *
 * All five keep a label and a hint, including the two with no colour left. The words are still what
 * the header count, the table and the ring's sentence say, and the legend still defines them — as
 * prose rather than as a key, which is the shape a thing with no mark has to take.
 */
const STATE_STYLE: Record<CellState, { fill?: string; label: string; hint: string }> = {
  comfortable: {
    label: 'Comfortable',
    hint: 'Fits with room, types fast enough to read along, and starts answering promptly.',
  },
  tight: {
    label: 'Tight',
    hint: 'Runs, but near the ceiling, slow to type, or slow to start — the table says which.',
  },
  offloaded: {
    fill: colors.serious,
    label: 'Spilling to RAM',
    /**
     * Conditional, because the engine cannot check the condition — the qualifier is
     * `HOST_RAM_UNCHECKED`, which this legend and Telemetry now share rather than each keeping a
     * near-copy. Theirs had already drifted to "the spilled *part*". "Loads" stated flatly promised
     * something never verified.
     *
     * A swatch caption, so the tail this appends is one clause. Telemetry's is longer because its
     * tile is where someone goes to ask why the thing is slow.
     */
    hint: `${HOST_RAM_UNCHECKED} What does spill crosses the bus every token.`,
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

/** Whether this cell's colour is a magnitude at all, rather than one of the flat states. */
function graded(cell: EnvelopeCell): boolean {
  return STATE_STYLE[cell.state].fill === undefined;
}

/**
 * The colour a cell is painted: a step of the ramp, or the flat colour of a state that is not a
 * magnitude.
 *
 * The domain is the grid's own span rather than zero-to-best, and that is the half of this that
 * matters here — see `magnitudeFill`. Headroom at the default scenario runs 18% to 49% of the
 * ceiling because 60 GiB of weights is in every cell, and measured from zero that whole range lands
 * on three steps of seven. The ramp has to be spent on what differs along these two axes, which is
 * the cache.
 */
function fieldFill(
  cell: EnvelopeCell,
  measure: Measure,
  domain: { min: number; max: number }
): string {
  return STATE_STYLE[cell.state].fill ?? magnitudeFill(measureOf(cell, measure), domain);
}

/**
 * Narrowest a column may be, in `rem`.
 *
 * Set by the widest label the axis can produce: `uniqueLabels` falls back to an exact count like
 * "131,072" when two columns would otherwise share a header, and that needs about 50px at the
 * label's own size. Below this the container scrolls rather than the labels overlapping.
 *
 * **In `rem` rather than pixels, because it is a floor on text.** 3.25rem is 52px at the default
 * root, which is what this was written as — and when the labels were made to scale (#42) a pixel
 * floor would have left 200% text overlapping inside a column sized for 100%. A length derived
 * from a glyph width is only correct at the root size it was measured at, and this one is derived
 * from a glyph width.
 */
const MIN_COLUMN_REM = 3.25;

/**
 * The axis labels' size.
 *
 * 0.625rem is 10px at the default root — identical to the `text-[10px]` it replaces, and unlike
 * it, responsive to the browser's text-size setting. Absolute pixel type does not scale at all,
 * so at 200% every other figure on the page doubled while these stayed put: a WCAG 1.4.4 failure
 * outright, and worse than the raw numbers suggest, because the labels became *relatively* half
 * the size on the surface a low-vision reader had just asked to enlarge (#42).
 *
 * Still the smallest type in the app, deliberately. Density is real — the column floor above is
 * set by this size, and larger labels mean fewer columns before the axis scrolls — but that is a
 * trade against the viewport, not against the reader's own setting, and the two were conflated.
 */
const AXIS_LABEL = 'text-[0.625rem]';

/**
 * An axis *title's* size and ink — deliberately not the tick labels' above.
 *
 * Both axes are powers of two in overlapping ranges, so before these titles existed the entire
 * distinction between "128 users" and "128K tokens" was a `K` rendered in the smallest, faintest
 * type on the surface. Putting the titles at that size and colour too would leave the whole meaning
 * of the picture in the least legible ink available (#81); `text-xs` at `text-muted` is one step up
 * on both, and still recessive against the legend and the heading beside it.
 *
 * In `rem`, for the reason recorded on `AXIS_LABEL`: this is text, and text that ignores the
 * browser's size setting fails 1.4.4 the moment a reader changes it.
 */
const AXIS_TITLE = 'text-xs text-[var(--color-text-muted)]';

export function Envelope({ config }: { config: Config }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const headingId = useId();
  const [showTable, setShowTable] = useState(false);
  /**
   * What the fill means, defaulting to the question the panel's own heading asks.
   *
   * `fit` rather than the liveliest picture: the axes are the two quantities that multiply into the
   * cache, so capacity is the reading the field was drawn for, and it is the Matrix's default too.
   * The other two are one press away and both are genuinely better graded — decode spans 20x here
   * and latency 561x — which is the argument for a toggle rather than for picking one.
   */
  const [measure, setMeasure] = useState<Measure>('fit');

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
        // The *responsive* boundary, not the tolerable one. "Comfortable" promises the answer
        // starts promptly, and the Telemetry tile calls anything past two seconds "Noticeable"
        // in amber — so a 10-second threshold here painted green over a scenario the tile beside
        // it was already warning about. Adding latency to this classification and then judging
        // it more leniently than the surface it was meant to agree with left the disagreement
        // in place with an extra step.
        usableTtftSeconds: TTFT_RESPONSIVE,
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

  /**
   * The span the ramp is stretched over: the selected measure across the cells that take a step of
   * it, and nothing else.
   *
   * The flat states are excluded from the domain as well as from the ramp, which is the pairing that
   * matters. A grid where most cells are closed has a handful of resident ones left, and scaling
   * those against readings taken from cells painted a flat colour would spend the ramp on a range
   * the field never shows.
   */
  const domain = useMemo(() => {
    const values = grid.cells
      .flat()
      .filter(graded)
      .map((cell) => measureOf(cell, measure));
    return values.length > 0
      ? { min: Math.min(...values), max: Math.max(...values) }
      : { min: 0, max: 0 };
  }, [grid, measure]);

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
        ctx.fillStyle = fieldFill(cell, measure, domain);
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
      /**
       * The counter-line under the ring, which is what keeps it legible now that the fills are a
       * ramp rather than four flat hues.
       *
       * Two tones on one path: light ink at `marks.lineWidth`, then a 1px near-black inside it. The
       * invariant is not that either clears 3:1 on every step — it is that *one of them always
       * does*, the dark line on the ramp's light end and the light ring on its dark one. That is the
       * rule #67 wrote down for the Matrix's selected square on this same ramp, and this mark
       * inherited the obligation the moment the cells underneath it stopped being status colours.
       * `Envelope.test.tsx` measures both tones against every fill the field actually paints.
       */
      ctx.strokeStyle = withAlpha(colors.bg, 0.9);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [
    grid,
    measure,
    domain,
    config.contextTokens,
    config.concurrency,
    model.maxContext,
    resizeTick,
  ]);

  /**
   * Whether the closed cells are closed by a ceiling rather than by the hardware. A grid is one
   * device, so this is a property of the whole picture — and the Telemetry tile beside it already
   * tells the user they can raise it.
   */
  const closed = grid.cells.flat().filter((c) => c.state === 'over');
  const raiseable = closed.some((c) => c.overBecause === 'allocation');
  const beyondHardware = closed.some((c) => c.overBecause === 'capacity');

  const counts = grid.cells.flat().reduce<Record<string, number>>((acc, cell) => {
    acc[cell.state] = (acc[cell.state] ?? 0) + 1;
    return acc;
  }, {});
  const total = grid.cells.flat().length;
  const currentCell = grid.cells.flat().find((c) => isCurrent(c, config, model));

  return (
    <section aria-labelledby={headingId} className="panel p-[min(1.25rem,5vw)]">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={headingId} className="text-sm font-semibold tracking-wide">
          How much room is left
          {/* Prose, and deliberately *not* read from `SETTING_LABELS` — the rule recorded there.
              This continues the heading as a sentence and is the section's `aria-labelledby`
              target, so substituting the control names welds two capitalised labels into the
              middle of it: "How much room is left Context per sequence against Concurrent users",
              announced verbatim every time the landmark is. The axis titles, the caption and the
              column header are the surfaces that *name* the settings, and those read the
              constant. */}
          <span className="ml-2 font-normal text-[var(--color-text-faint)]">
            context against concurrent users
          </span>
        </h2>
        <PanelCount count={counts.comfortable ?? 0} total={total}>
          comfortable
        </PanelCount>
      </header>

      {/*
        One filter row above the field, as the dataviz guidance puts it — and the Matrix's own
        control, reading the Matrix's own `MEASURES`, because these are the same three questions
        asked over different axes. Toggling rearranges which corner of the region looks survivable,
        which is the capacity/bandwidth/compute triangle made concrete on a second surface rather
        than asserted in prose.

        The count in the header stays a comfort count, deliberately, and is the reason it can: the
        verdict is a word, so it lives in the channels that carry words. The fill is a magnitude.
      */}
      <fieldset className="mt-4">
        <legend className="sr-only">Colour the field by</legend>
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
        {/* The ramp's domain, stated. Its ends are this grid's own best and worst cell rather than
            absolute figures — that is what makes a field whose variation sits on top of a large
            constant legible at all, and it is also a thing a reader would otherwise assume the
            other way round. */}
        <p className="mt-1.5 text-xs text-[var(--color-text-muted)]">
          {MEASURES.find((m) => m.value === measure)?.hint} The ramp runs between the best and worst
          cell on this grid, so it says which way the region falls off rather than by how much — the
          table has the figures.
        </p>
      </fieldset>

      <div className="mt-4">
        {/*
          The y title, stacked above the gutter rather than rotated beside it.

          Rotating it is the conventional choice, and the cost is worth stating accurately because
          the wrong figure is what a later session would inherit: a `writing-mode: vertical-rl`
          title costs one line box of horizontal space — a fraction of one `MIN_COLUMN_REM` column
          — and none of the plot's content width, only its scroll viewport. (A `rotate-90`
          *transform* is the expensive one: transforms do not affect layout, so it reserves the
          label's full unrotated length.) So this is a legibility call rather than a width one:
          12px type turned on its side, on the surface whose whole complaint was that its meaning
          rode on the least legible ink, and an arrow that has to read as up. Stacked it costs one
          line of vertical space, which this panel has and the width it does not.

          The arrow is the point of it, not decoration. Rows are drawn bottom-up (see the paint
          effect below), which is correct for an axis and the opposite of every other list on this
          page: a reader who assumes top-to-bottom reads the default field as "128 users at 2K is
          the comfortable one" when it is 1 user at 2K — close to the opposite claim. That
          direction was stated only in a source comment until now (#81).

          `aria-hidden`, like the tick labels either side of it, because the canvas `aria-label` is
          this picture's textual equivalent and already names both quantities. Visible titles that
          joined the accessible tree would have a screen reader hear the axes named twice.
        */}
        <p aria-hidden="true" className={`${AXIS_TITLE} leading-tight`}>
          {SETTING_LABELS.concurrency} ↑
        </p>

        <div className="mt-1 flex gap-2">
          {/*
            One track per row, so each label sits at its cell's centre. `justify-between` puts the
            first and last at the extremes instead — which pushed "1" below the grid entirely and
            left every other label straddling a boundary.
          */}
          <ol
            aria-hidden="true"
            className={`tabular grid h-48 ${AXIS_LABEL} text-[var(--color-text-faint)]`}
            style={{ gridTemplateRows: `repeat(${grid.concurrencies.length}, 1fr)` }}
          >
            {[...grid.concurrencies].reverse().map((n) => (
              <li key={n} className="flex items-center justify-end pr-1">
                {n}
              </li>
            ))}
          </ol>

          <div className="min-w-0 flex-1">
            {/*
              The plot and its axis scroll together as one unit rather than being squeezed.

              Truncating the labels instead would undo the disambiguation they exist for —
              "131,072" and "131,073" clipped to the same width are indistinguishable again, which
              is the bug `uniqueLabels` was written to fix. An equal-width grid on a phone gives
              each column about 25px, and an exact count needs roughly 50, so the columns get a
              floor and the container scrolls when they do not fit. The canvas shares the floor so
              the cells stay aligned with their headers.
            */}
            <div className="overflow-x-auto">
              <div style={{ minWidth: `${grid.contexts.length * MIN_COLUMN_REM}rem` }}>
                <canvas
                  ref={canvasRef}
                  role="img"
                  aria-label={describe(grid, counts, total, currentCell, contextLabels, measure)}
                  className="h-48 w-full rounded"
                />
                <ol
                  aria-hidden="true"
                  className={`tabular mt-1 grid ${AXIS_LABEL} text-[var(--color-text-faint)]`}
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
              The x title, centred on the plot and *outside* its scroll container.

              Inside, it would be centred on the scrolled content instead — which at 320px is
              wider than the panel, so the title would sit half off-screen and its box would
              extend past the panel edge. Outside, it stays centred on what the reader can see and
              the scroller keeps clipping only the thing that is meant to scroll.
            */}
            <p aria-hidden="true" className={`mt-1 text-center ${AXIS_TITLE}`}>
              {SETTING_LABELS.contextTokens}
            </p>
          </div>
        </div>
      </div>

      {/*
        The hint carries what the colour cannot: a ramp step is a magnitude, while "will not run" and
        "spilling to RAM" are states, and "tight" means three unrelated things.

        Only what the picture and the panel actually contain. A legend is a key to this surface, not
        a catalogue of everything the engine can return — and `unsupported` deliberately shares
        `over`'s red, so listing both unconditionally would put two identical swatches under
        different sentences. They never co-occur: an unsupported runtime closes the whole grid. The
        ramp key follows the same rule and is absent from a grid where no cell is graded.
      */}
      <ul className="mt-4 grid gap-x-5 gap-y-2 sm:grid-cols-2">
        {/*
          The ring, which had no key at all — the same defect the budget bar's ceiling rule had one
          panel over (#73). Every entry below keys a *fill*; the ring is an overlay drawn on top of
          one, and it was named only inside the canvas `aria-label`, so a screen-reader user was told
          what it was and a sighted reader met a double ring on the grid with nothing on the page
          saying so. The legend is the dependable identity channel precisely because a mark that
          crosses a fill may be hard to make out.

          Keyed as a ring rather than a dot, and from the same `marks.lineWidth` the canvas strokes:
          a key only works if it is the mark. First in the list for the same reason `describe` puts
          it first — it is the reader's own position, and the states below are the field it sits in.

          Only while it is drawn, exactly as the state keys are listed only while the grid contains
          them. The ring needs the selected context and concurrency to both be on the axes, which is
          what `currentCell` reports: the axes are built to contain them, so in practice it is always
          there, and keying it unconditionally would be a claim this component does not check.
        */}
        {currentCell && (
          <li className="flex items-baseline gap-2 text-sm">
            <span
              aria-hidden="true"
              className="mt-1 inline-block h-3 w-3 shrink-0 rounded-full border-solid border-[var(--color-text)]"
              style={{ borderWidth: marks.lineWidth }}
            />
            <span>
              <span className="text-[var(--color-text)]">You are here</span>{' '}
              <span className="text-xs text-[var(--color-text-muted)]">
                The context and concurrency the Bench above is set to. The table marks the same
                cell.
              </span>
            </span>
          </li>
        )}
        {/*
          The ramp, which is the field's primary encoding and has no discrete keys to list — the
          same shape of key the Matrix's legend carries for the same ramp, and spanning both columns
          because it is one scale rather than one of a set of states.

          `flex-wrap` with the gradient's floor capped at `100%`, for the reason recorded on the
          Matrix's legend (#34) and on `min-w-[min(12rem,100%)]` in the ROADMAP: a bare rem floor is
          a floor the viewport cannot argue with, so under browser text scaling it takes the document
          into a sideways scroll at 320px. Capped, the gradient claims its own line instead.

          Only while some cell is actually graded. Under a runtime that cannot drive the device the
          whole field is one flat red, and a ramp key there describes a scale nothing on screen is
          painted from.
        */}
        {grid.cells.flat().some(graded) && (
          <li className="flex flex-wrap items-center gap-2 text-sm sm:col-span-2">
            <span className="text-[var(--color-text)]">worse</span>
            <span
              aria-hidden="true"
              className="flex h-3 min-w-[min(4rem,100%)] flex-1 overflow-hidden rounded-sm"
            >
              {magnitudeRamp.map((step) => (
                <span key={step} className="flex-1" style={{ background: step }} />
              ))}
            </span>
            <span className="text-[var(--color-text)]">better</span>
            {/* Naming the measure again here would print the caption above twice on one panel; the
                control is what says which of the three the ramp is currently spending itself on. */}
            <span className="text-xs text-[var(--color-text-muted)]">
              Every cell that fits, graded by the selected measure.
            </span>
          </li>
        )}
        {/*
          Every state the grid holds, in one of two shapes: a swatch and a sentence for the three
          that own a colour, and a sentence alone for the two verdicts that no longer do.

          The swatch-less shape is deliberate rather than a leftover. "A key to a mark that appears
          nowhere is worse than prose" is the rule the Matrix's own stand-in warning follows, and it
          applies exactly here — but the words themselves have to stay, because the count in this
          panel's header, the first thing every row of the table says and the ring's own sentence are
          all "comfortable" or "tight", and this list is the only place on the surface that says what
          either one means.
        */}
        {(Object.keys(STATE_STYLE) as CellState[])
          .filter((state) => counts[state])
          .map((state) => {
            /**
             * The red cells can be closed for two reasons at once, and one grid routinely holds
             * both: on a 512 GiB Mac the small-context corner is a raiseable ceiling away from
             * running while the far corner is past the machine however it is tuned. Picking the
             * legend from whether *any* cell is raiseable told the reader the far corner was
             * fixable too.
             *
             * Said as one entry naming both, exactly as `tight` handles its three causes — the
             * table is where a specific cell is identified.
             */
            const { label, hint } =
              state === 'over' && raiseable
                ? beyondHardware
                  ? {
                      label: 'Will not run',
                      hint: 'Some of these are past what this machine holds; the rest are only past the ceiling it hands out by default, which you can raise. The table says which.',
                    }
                  : {
                      label: 'Past the default allocation',
                      hint: 'Within the memory this machine has, but past the ceiling it hands out by default — which you can raise.',
                    }
                : STATE_STYLE[state];

            const fill = STATE_STYLE[state].fill;

            return (
              <li key={state} className="flex items-baseline gap-2 text-sm">
                {fill && (
                  <span
                    aria-hidden="true"
                    className="mt-1 inline-block h-3 w-3 shrink-0 rounded-sm"
                    style={{ background: fill }}
                  />
                )}
                <span>
                  <span className="text-[var(--color-text)]">{label}</span>{' '}
                  <span className="text-xs text-[var(--color-text-muted)]">{hint}</span>
                </span>
              </li>
            );
          })}
      </ul>

      <DisclosureToggle expanded={showTable} onToggle={() => setShowTable((v) => !v)}>
        {showTable ? 'Hide' : 'Show'} the region as a table
      </DisclosureToggle>

      {showTable && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-sm">
            {/* Both names read from the shared constant, like the axis titles on the picture. The
                caption said "context length" and the column said "Users" — the same two settings
                the sliders name, under two more spellings, on the surface that is the picture's
                textual equivalent. */}
            <caption className="sr-only">
              Feasibility by {SETTING_LABELS.contextTokens} and {SETTING_LABELS.concurrency}
            </caption>
            <thead>
              <tr className="text-[var(--color-text-faint)]">
                <th scope="col" className="py-1 pr-3 font-normal">
                  {SETTING_LABELS.concurrency}
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
  current: EnvelopeCell | undefined,
  contextLabels: readonly string[],
  measure: Measure
): string {
  /**
   * The closed cells, split the way the legend splits them.
   *
   * This summary is the *only* form the picture takes for a screen reader, so a distinction the
   * legend draws and this does not is a distinction that reader never gets. It went on saying
   * every closed combination "will not run at all" after the visible legend learned to say some
   * of them are one setting away.
   */
  const closed = grid.cells.flat().filter((c) => c.state === 'over');
  const raiseable = closed.filter((c) => c.overBecause === 'allocation').length;
  const beyond = closed.length - raiseable;

  const whyClosed =
    raiseable > 0 && beyond > 0
      ? `${beyond} of ${total} will not run at all, and ${raiseable} more exceed only the default allocation ceiling, which you can raise.`
      : raiseable > 0
        ? `${raiseable} of ${total} exceed the default allocation ceiling, which you can raise.`
        : `${closed.length} of ${total} combinations will not run at all.`;
  /**
   * The ring is the only mark on this panel with no textual equivalent, so it goes first.
   *
   * Two things it has to borrow rather than re-derive. The context reads from the same
   * disambiguated axis labels the visible headers use — formatting it independently rendered
   * 131,072 and 131,073 both as "128K", so a screen-reader user could not tell which of two
   * evaluated columns the ring was on, which is precisely what `uniqueLabels` exists to prevent.
   * And the state reads through `describeCell`, so a cell that merely exceeds a raiseable ceiling
   * is not announced as "will not run" while the table beside it says it is one setting away.
   */
  const index = current ? grid.contexts.indexOf(current.contextTokens) : -1;
  const where = index >= 0 ? contextLabels[index] : current ? tokens(current.contextTokens) : '';
  const here = current
    ? `Currently at ${where} context and ${current.concurrency} ${
        current.concurrency === 1 ? 'user' : 'users'
      }: ${describeCell(current).toLowerCase()}. `
    : '';

  /**
   * What the rest of the grid is doing, said the same way whether or not anything is comfortable.
   *
   * Both branches used to omit this, in opposite directions. With no comfortable cell the closed
   * count was appended unconditionally, so an entirely runnable region announced "0 of N
   * combinations will not run at all" — true, and it reads as though nothing works. With some
   * comfortable cells nothing else was mentioned at all, so a grid of 3 comfortable and 6 spilling
   * was described only by its 3.
   *
   * This is the canvas's only textual equivalent and the data table is hidden by default, so
   * whatever this sentence leaves out is simply not available to a screen-reader user. Built once
   * rather than per branch: fixing the branch a review named and leaving its neighbour is how the
   * two came to disagree in the first place.
   */
  const runnable = [
    (counts.tight ?? 0) > 0 && `${counts.tight} run but sit near a limit`,
    (counts.offloaded ?? 0) > 0 &&
      `${counts.offloaded} run only by spilling weights to host RAM, if the host has room for them`,
  ].filter((s): s is string => typeof s === 'string');
  /**
   * What the colour means, which a sighted reader gets from the ramp and the toggle's caption.
   *
   * Said only when some cell is actually graded, the same condition the visible ramp key is drawn
   * under: on a grid the runtime cannot drive there is no ramp on screen and nothing for this clause
   * to describe. Without it the three measure buttons are silent for a screen-reader user — they
   * repaint a picture whose only textual equivalent never mentioned colour at all.
   */
  const coloured = grid.cells.flat().some(graded)
    ? `Cells that fit are coloured by ${MEASURES.find((m) => m.value === measure)?.paints}, worst to best.`
    : '';
  // Suppressed only when it would print a zero — `closed.length > 0` is exactly the condition
  // under which `whyClosed` has something non-empty to report, in all three of its forms.
  const rest = [
    runnable.length > 0 ? `${runnable.join(', and ')}.` : '',
    closed.length > 0 ? whyClosed : '',
    coloured,
  ];

  const comfortable = counts.comfortable ?? 0;
  if (comfortable === 0) {
    // Two different sentences, because the two failures need opposite advice.
    if (counts.unsupported) {
      return `${here}This runtime cannot drive this hardware, so none of the ${total} combinations run.`;
    }
    return [`${here}No comfortable configuration in this range.`, ...rest]
      .filter(Boolean)
      .join(' ');
  }
  const widest = grid.cells[0].filter((c) => c.state === 'comfortable').at(-1);
  return [
    `${here}${comfortable} of ${total} combinations of context and concurrency are comfortable.`,
    `At ${grid.concurrencies[0]} user, up to ${tokens(widest?.contextTokens ?? 0)} of context stays comfortable.`,
    ...rest,
  ]
    .filter(Boolean)
    .join(' ');
}

/** What a cell says in the table: its state, why, and what it costs. */
function describeCell(cell: EnvelopeCell): string {
  if (cell.state === 'over') {
    return cell.overBecause === 'allocation'
      ? 'Past the default allocation'
      : STATE_STYLE.over.label;
  }
  if (cell.state === 'unsupported') return STATE_STYLE.unsupported.label;
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
