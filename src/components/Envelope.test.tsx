import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Envelope } from './Envelope';
import { DEFAULT_CONFIG } from '@/store/scenario';
import { colors, magnitudeRamp, withAlpha } from '@/design/tokens';

/**
 * What the feasibility field is actually painted in (#65).
 *
 * The panel's whole subject is a *shape*, and at the default scenario it had none: 45 of its 56
 * cells were one amber, one was green, and not one amber cell was amber for capacity — the
 * tight-on-capacity band is the last tenth of the ceiling while both axes double per step, so the
 * grid steps from 82% of the ceiling to 112% and skips it in every column. A field titled "how much
 * room is left" was coloured entirely by speed and latency, at three buckets over readings that
 * span 20x of decode and 561x of first-token latency.
 *
 * **These read the fills back out of the draw loop, which is the only place they exist.** The cells
 * are canvas, so there is no DOM node carrying a colour and nothing in `App.test.tsx` can see them —
 * the unit suite has always emitted "Not implemented: HTMLCanvasElement.getContext" here and
 * asserted nothing about what was drawn. A recording context closes that: it is the real component,
 * the real default scenario and the real paint effect, with the two-dimensional geometry (which
 * jsdom cannot answer, and `e2e/canvases.spec.ts` does) replaced by the order the loop paints in.
 *
 * Everything below is written to fail against the unfixed panel, which paints three colours in
 * total and one of them over 80% of the field.
 */

/** A cell fill or a ring stroke, in the order the effect painted it. */
interface Painted {
  fills: string[];
  strokes: string[];
}

/**
 * A 2D context that records rather than rasterises.
 *
 * Deliberately a plain object with exactly the members the paint effect uses, not a permissive
 * proxy: a call this fake does not have fails loudly, which is the right outcome — a new drawing
 * operation on this field is a new thing for these tests to describe rather than something to
 * absorb silently.
 */
function recordPainting(): Painted {
  const painted: Painted = { fills: [], strokes: [] };
  const context = {
    fillStyle: '' as string | CanvasGradient | CanvasPattern,
    strokeStyle: '' as string | CanvasGradient | CanvasPattern,
    lineWidth: 0,
    setTransform: () => {},
    clearRect: () => {},
    fillRect: () => painted.fills.push(String(context.fillStyle)),
    beginPath: () => {},
    arc: () => {},
    stroke: () => painted.strokes.push(String(context.strokeStyle)),
  };

  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
    context as unknown as CanvasRenderingContext2D
  );
  return painted;
}

const field = () => screen.getByRole('region', { name: /how much room is left/i });

/** The ramp step a fill is, or -1 for the flat state colours. */
const step = (fill: string) => (magnitudeRamp as readonly string[]).indexOf(fill);

/** The table, which is the field's textual equivalent and is used here to count its cells. */
async function openTable(user: ReturnType<typeof userEvent.setup>) {
  await user.click(within(field()).getByRole('button', { name: /region as a table/i }));
  return within(field()).getByRole('table');
}

describe('the feasibility field', () => {
  let painted: Painted;

  beforeEach(() => {
    painted = recordPainting();
    render(<Envelope config={DEFAULT_CONFIG} />);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('paints every cell the table lists', async () => {
    const user = userEvent.setup();
    const table = await openTable(user);

    // The picture and its textual equivalent describe the same grid, which is what makes the
    // counting below mean anything: `getAllByRole('cell')` is the data cells, since the concurrency
    // column is a `rowheader`.
    expect(painted.fills).toHaveLength(within(table).getAllByRole('cell').length);
    expect(painted.fills.length).toBeGreaterThan(40);
  });

  it('grades the region instead of painting one flat colour', () => {
    const distinct = new Set(painted.fills);
    const share = Math.max(
      ...[...distinct].map((fill) => painted.fills.filter((f) => f === fill).length)
    );

    /*
     * Three colours were reachable before this — good, warning, critical — and the default scenario
     * used all three: 45 amber, 10 red, 1 green. Both assertions fail on those figures, and the
     * second is the one that matters: a picture where four cells in five share a colour is not
     * carrying a shape however many colours exist in principle.
     */
    expect(distinct.size, `only ${distinct.size} colours on the field`).toBeGreaterThanOrEqual(4);
    expect(share / painted.fills.length, 'one colour covers most of the field').toBeLessThan(0.6);
  });

  it('spends the whole ramp, both ends of it, on the cells that fit', () => {
    /*
     * The domain is this grid's own span, and this is the assertion that says so. Headroom here runs
     * 18% to 49% of the ceiling — 60 GiB of gpt-oss weights sits in every cell — so measured from
     * zero the darkest three steps are never reached at all and the field loses the boundary it is
     * drawn for. Both ends present is what a `{min, max}` domain buys.
     */
    const steps = painted.fills.map(step).filter((s) => s >= 0);
    expect(steps.length, 'no cell was painted from the ramp').toBeGreaterThan(0);
    expect(Math.min(...steps)).toBe(0);
    expect(Math.max(...steps)).toBe(magnitudeRamp.length - 1);
  });

  it('keeps a refusal categorical rather than putting it on the ramp', async () => {
    const user = userEvent.setup();
    const table = await openTable(user);

    // The states that are not magnitudes: a red cell is a refusal and never a step of the ramp, and
    // there are exactly as many as the table says will not run.
    const closed = painted.fills.filter((fill) => fill === colors.critical);
    expect(closed.length).toBe(within(table).getAllByText(/Will not run/).length);
    expect(closed.length).toBeGreaterThan(0);
    for (const fill of closed) expect(step(fill)).toBe(-1);
  });

  /**
   * The capacity claim, and the one the unfixed panel could not make at all: under `fit` the colour
   * has to fall as the cache grows, along both axes.
   *
   * Asserted on the order the loop paints in rather than on figures, because the engine's own
   * reference test already pins the direction — KV scales with context times concurrency, so pushing
   * either axis can only make a cell worse. The loop walks `cells[concurrency][context]`, so index
   * order runs fewest users first and each row from the smallest context up: a graded cell may never
   * be brighter than its predecessor on either axis.
   */
  it('darkens as the cache grows, along both axes', async () => {
    const user = userEvent.setup();
    const table = await openTable(user);
    const columns = within(table).getAllByRole('columnheader').length - 1;
    const rows = painted.fills.length / columns;
    expect(Number.isInteger(rows) && rows > 1, 'the grid is not rectangular').toBe(true);

    const at = (row: number, column: number) => step(painted.fills[row * columns + column]);

    /*
     * The guard that stops this being one of the three tests this repo has shipped that could not
     * fail. Every comparison below is skipped for a cell that is not on the ramp, so against the
     * three-state fill — where *no* cell is on it — the whole sweep is vacuous and green. It needs
     * most of the field graded, and it needs the steps to actually differ, or "never brighter" is a
     * claim about one colour.
     */
    const steps = painted.fills.map(step).filter((s) => s >= 0);
    expect(steps.length, 'almost nothing is on the ramp').toBeGreaterThan(painted.fills.length / 2);
    expect(new Set(steps).size, 'every graded cell is the same step').toBeGreaterThan(2);

    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const here = at(row, column);
        if (here < 0) continue; // a flat state: not on the scale, so not part of the ordering
        if (column > 0 && at(row, column - 1) >= 0) {
          expect(
            at(row, column - 1),
            `row ${row} column ${column} is brighter than the narrower context beside it`
          ).toBeGreaterThanOrEqual(here);
        }
        if (row > 0 && at(row - 1, column) >= 0) {
          expect(
            at(row - 1, column),
            `row ${row} column ${column} is brighter than the same context at fewer users`
          ).toBeGreaterThanOrEqual(here);
        }
      }
    }
  });

  it('repaints when the measure changes, and says so where the colour is described', async () => {
    const user = userEvent.setup();

    const byFit = [...painted.fills];
    const description = () => within(field()).getByRole('img').getAttribute('aria-label') ?? '';
    expect(description()).toMatch(/coloured by headroom left/i);

    painted.fills.length = 0;
    await user.click(within(field()).getByRole('button', { name: 'How fast' }));

    expect(painted.fills.length).toBe(byFit.length);
    // A different picture, not merely a redrawn one: these are two different readings of the same
    // cells and only the closed corner is common to both.
    expect(painted.fills).not.toEqual(byFit);
    /*
     * And the canvas description follows, because it is the picture's only textual equivalent — the
     * table is hidden by default. Without this the three buttons repaint something a screen-reader
     * user is never told about, which is the same gap the ring had before #73.
     */
    expect(description()).toMatch(/coloured by tokens per second/i);
    expect(within(field()).getByRole('button', { name: 'How fast' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
  });

  /**
   * The "you are here" ring, measured against the fills it is now drawn on.
   *
   * #67 recorded this obligation for the Matrix's selected square and stated it as a general one:
   * anything drawn on a cell inherits it and none of the existing measurements, because
   * `tokens.ts` validates the marks against `--color-surface` and says nothing about the ramp. This
   * mark inherited it the moment the cells underneath it stopped being flat status colours.
   *
   * The invariant is not that either tone clears 3:1 everywhere — it is that *one of them always
   * does*: the near-black counter-line on the ramp's light steps, the light ring on its dark ones.
   * Same arithmetic `App.test.tsx` runs over the Matrix's fills; kept local rather than reaching
   * into that file's helpers, since the two are measuring different marks on different surfaces.
   */
  it('keeps one of the ring’s two tones 3:1 against every fill it can land on', () => {
    const MINIMUM_CONTRAST = 3;

    const channels = (value: string): [number, number, number, number] => {
      if (value.startsWith('#')) {
        const [r, g, b] = [1, 3, 5].map((i) => Number.parseInt(value.slice(i, i + 2), 16));
        return [r, g, b, 1];
      }
      const parts = value.match(/[\d.]+/g)?.map(Number) ?? [];
      return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0, parts[3] ?? 1];
    };

    /** A stroke composited over the fill beneath it, since the counter-line is `rgba(...)`. */
    const over = (mark: string, behind: string) => {
      const [r, g, b, alpha] = channels(mark);
      const [br, bg, bb] = channels(behind);
      return [
        r * alpha + br * (1 - alpha),
        g * alpha + bg * (1 - alpha),
        b * alpha + bb * (1 - alpha),
      ];
    };

    const luminance = (rgb: number[]) => {
      const [r, g, b] = rgb
        .map((c) => c / 255)
        .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    const contrast = (a: number[], b: number[]) => {
      const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
      return (high + 0.05) / (low + 0.05);
    };

    // Vacuity guards: the ring is two strokes on one path, and a field with one fill would make the
    // sweep below trivially true.
    expect(painted.strokes, 'the ring is not two tones').toEqual([
      colors.text,
      withAlpha(colors.bg, 0.9),
    ]);
    const fills = new Set(painted.fills);
    expect(
      fills.size,
      'the field paints one colour, so the ramp is not being exercised'
    ).toBeGreaterThan(3);

    const best = (fill: string) =>
      Math.max(...painted.strokes.map((tone) => contrast(over(tone, fill), channels(fill))));
    const unreadable = [...fills]
      .filter((fill) => best(fill) < MINIMUM_CONTRAST)
      .map((fill) => `${fill} at ${best(fill).toFixed(2)}:1`);
    expect(unreadable, `the ring below ${MINIMUM_CONTRAST}:1 on a fill the field paints`).toEqual(
      []
    );

    /*
     * And that the second tone is load-bearing rather than belt-and-braces. If one tone cleared the
     * bar on every fill, this test would pass on a single-tone ring — which is exactly what shipped
     * 304 unreadable squares on the Matrix before #67.
     */
    const defeatsOneTone = painted.strokes.filter(
      (tone) =>
        ![...fills].every((fill) => contrast(over(tone, fill), channels(fill)) >= MINIMUM_CONTRAST)
    );
    expect(
      defeatsOneTone.length,
      'every tone clears the bar alone, so this measures nothing the ramp can break'
    ).toBeGreaterThan(0);
  });
});
