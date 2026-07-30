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
     *
     * **A claim about this scenario, and deliberately not a universal bar.** How flat the field is
     * depends on the rig: on gpt-oss-20b against an EPYC 9755 the cache is negligible against a
     * 1,450 GiB ceiling until the far corner, so 50 of 56 cells genuinely do share a step — that is the
     * machine, not the scale, and flooring the domain at zero makes it 55 of 56. What has to hold at
     * every scenario is that the ramp is keyed as a ranking rather than as a verdict, which is the
     * sweep below at exactly that rig.
     */
    expect(distinct.size, `only ${distinct.size} colours on the field`).toBeGreaterThanOrEqual(4);
    expect(share / painted.fills.length, 'one colour covers most of the field').toBeLessThan(0.6);
  });

  /**
   * The comfort count, which the field deliberately stopped drawing.
   *
   * "1 of 56 comfortable" sits in this panel's header, and the one comfortable cell is painted the
   * ramp's brightest step — as are 27 tight ones, 28 of 56 sharing that hex. Putting the verdict back
   * on the fill is the defect #65 fixed, so the count is located in the channel that carries words:
   * the legend line that is the only place on this surface saying what "comfortable" means.
   */
  it('locates the comfort count in words, since the field no longer draws it', () => {
    const brightest = magnitudeRamp[magnitudeRamp.length - 1];
    // The premise, asserted rather than assumed: the count cannot be found by colour because the
    // comfortable cell's colour is not its own.
    expect(
      painted.fills.filter((f) => f === brightest).length,
      'the brightest step is unique, so the count is locatable by colour after all'
    ).toBeGreaterThan(1);

    const description = within(field()).getByRole('img').getAttribute('aria-label') ?? '';
    const edge = description.match(/At \d+ users?, up to [\d,.]+[KM]? of context/)?.[0];
    expect(edge, 'the description does not state the comfortable frontier').toBeTruthy();

    // The same phrase, from the same derivation, on the visible legend — which had the definition of
    // the word and nothing about where the cells were.
    const key = within(field()).getByText('Comfortable').closest('li');
    expect(key?.textContent).toContain(edge);
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

/**
 * The ramp's ends, on a rig where nothing is short of room.
 *
 * A domain of `{min, max}` over the grid's own cells is what makes a field whose variation sits on a
 * large constant legible — and it is a *ranking*, so the legend and the description are only entitled
 * to rank. Keyed "worse" and "better" (the Matrix's words for the same seven hexes, where the domain
 * is floored at zero and a verdict is a claim it can make) this panel told the reader the emptiest
 * machine in the catalogue was out of room.
 *
 * Measured at this scenario — gpt-oss-20b, llama.cpp, MXFP4, 32K, 1 user, 8K prompt, one EPYC 9755:
 * all 56 cells run, the `fit` domain is headroom 0.726 to 0.991 of the ceiling, and the darkest step
 * lands on the 128K x 128-user corner at 27.4% utilization — 1,052 of 1,450 allocatable GiB unused.
 *
 * The default scenario cannot show this: it has a red wall, so the darkest ramp step there really is
 * next to one. Which is why this is its own render rather than another assertion on the block above.
 */
describe('the ramp keys itself as a ranking, not a verdict', () => {
  /** A rig with room to spare in every cell, which the default deliberately does not have. */
  const roomy = { ...DEFAULT_CONFIG, modelId: 'openai/gpt-oss-20b', deviceId: 'epyc-9755' };

  let painted: Painted;

  beforeEach(() => {
    painted = recordPainting();
    render(<Envelope config={roomy} />);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  const description = () => within(field()).getByRole('img').getAttribute('aria-label') ?? '';

  it('reaches the bottom of the ramp on a grid whose worst cell has room to spare', () => {
    // The premise. Every cell on the ramp, no refusals — so nothing dark here is dark because it hit
    // a wall, and the two assertions below are about a cell the reader can actually run.
    expect(painted.fills.length).toBeGreaterThan(40);
    expect(
      painted.fills.filter((f) => step(f) < 0),
      'a flat state colour on a grid where everything fits'
    ).toEqual([]);
    expect(painted.fills.map(step)).toContain(0);
  });

  it('names the ends by the measure rather than by a verdict', async () => {
    const user = userEvent.setup();
    const legend = () => field();

    // The words that were there, and are a claim this domain cannot support.
    expect(within(legend()).queryByText('worse')).toBeNull();
    expect(within(legend()).queryByText('better')).toBeNull();

    expect(within(legend()).getByText('less room')).toBeInTheDocument();
    expect(within(legend()).getByText('more room')).toBeInTheDocument();

    /*
     * Per measure rather than one generic pair, and this is the assertion that pins it. `measureOf`
     * inverts latency so larger is better throughout, so a generic "least to most" would read
     * backwards against the caption beside it ("Time until the first token appears.") — a reader
     * would take the dark end for the quick one.
     */
    await user.click(within(legend()).getByRole('button', { name: 'How responsive' }));
    expect(within(legend()).getByText('slower to start')).toBeInTheDocument();
    expect(within(legend()).getByText('quicker to start')).toBeInTheDocument();
  });

  it('states the comparison class in the legend and in the canvas description', () => {
    // In the legend, not only in the 12px caption under the toggle — which was the sole disclosure,
    // and a reader meets it after they have already read the field.
    expect(
      within(field()).getByText(/graded against the others on this grid rather than against an/i)
    ).toBeInTheDocument();

    // And in the picture's only textual equivalent, which said "worst to best" — the absolute reading,
    // for the reader with none of the visible caveats.
    expect(description()).not.toMatch(/worst to best/i);
    expect(description()).toMatch(/less room to more room/i);
    expect(description()).toMatch(/ranked against the other cells on this grid/i);
  });
});

/**
 * The two things the measure control promised and could not deliver (found in review on #65).
 *
 * Both are the same shape: a claim in the caption that the rest of the panel does not honour.
 */
describe('the measure control only claims what the panel can show', () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  /**
   * **The ramp caption says "the table has the figures", and for the default measure it did not.**
   *
   * `describeCell` printed the verdict, the decode rate and the time to first token — so a reader
   * switching to "How fast" or "How responsive" could indeed find the number behind the shade. `fit`
   * is the measure the panel opens on, it paints headroom, and headroom appeared nowhere: not in the
   * table, not in the canvas description. So the one measure nobody has to choose was the one whose
   * "by how much" was unanswerable, on a field whose variation sits on top of a large constant and is
   * therefore least readable from the ramp alone.
   */
  it('puts the headroom figure in the table, which is where the caption sends the reader', async () => {
    const user = userEvent.setup();
    recordPainting();
    render(<Envelope config={DEFAULT_CONFIG} />);

    // The claim under test, quoted from the caption rather than paraphrased.
    expect(within(field()).getByText(/the table has the figures/i)).toBeInTheDocument();

    await user.click(within(field()).getByRole('button', { name: /table/i }));
    const table = within(field()).getByRole('table');

    // Rate and first-token were already there; room is the one that was missing. Every graded cell
    // carries it, so this is `getAllByText` — one match would mean it reached a single cell.
    expect(within(table).getAllByText(/room left/i).length).toBeGreaterThan(1);
    expect(within(table).getAllByText(/tok\/s/).length).toBeGreaterThan(0);
  });

  /**
   * **Three buttons and a ramp caption on a field with no ramp in it.**
   *
   * `graded` is false for every cell when the runtime cannot drive the device — a state that stays
   * reachable, because such a runtime remains selectable with a warning rather than being removed
   * from the control. The legend and the canvas description already withhold the ramp key under that
   * condition; this fieldset did not, so it offered three measures over an entirely categorical
   * picture and a caption promising "the ramp runs between this grid's own extremes". Pressing a
   * button moved `aria-pressed` and changed nothing else.
   */
  it('offers no measures on a grid where nothing is graded', () => {
    recordPainting();
    // vLLM cannot drive a Mac: every cell is `unsupported`, so no cell carries a magnitude.
    render(
      <Envelope
        config={{ ...DEFAULT_CONFIG, deviceId: 'mac-studio-m3-ultra-256', runtimeId: 'vllm' }}
      />
    );

    expect(within(field()).queryByRole('button', { name: 'Does it fit' })).toBeNull();
    expect(within(field()).queryByRole('button', { name: 'How fast' })).toBeNull();
    expect(within(field()).queryByText(/the ramp runs between/i)).toBeNull();

    // And the panel still explains itself — withholding the ramp control is not withholding the
    // reason, which is the distinction that makes this a fix rather than a deletion.
    expect(within(field()).getByText(/runtime cannot drive it/i)).toBeInTheDocument();
  });
});

/**
 * The measure group's description, which is #80's defect reappearing on the surface that copied the
 * control (found in review on #65).
 *
 * The caption was already written and already on screen — it simply was not the group's *accessible*
 * description, so entering the fieldset announced "Colour the field by, Does it fit, pressed" and
 * nothing about what a bright cell means. `Matrix.tsx` wires exactly this, for exactly this reason,
 * and the comment there names #80.
 */
describe('the measure group says what it colours by', () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('attaches the caption to the fieldset, and follows the selection', async () => {
    const user = userEvent.setup();
    recordPainting();
    render(<Envelope config={DEFAULT_CONFIG} />);

    const group = within(field()).getByRole('group', { name: /colour the field by/i });
    // The ids are split rather than handed to `getElementById` whole, because `aria-describedby` is
    // an IDREF *list*: a second id appended here would resolve to nothing and report the group as
    // undescribed, failing on the component instead of on this line. Same form as `App.test.tsx`'s
    // `description`.
    const describedBy = (el: HTMLElement) =>
      (el.getAttribute('aria-describedby') ?? '')
        .split(/\s+/)
        .filter(Boolean)
        .map((id) => document.getElementById(id)?.textContent?.trim() ?? '')
        .join(' ')
        .trim();

    expect(describedBy(group)).toMatch(/headroom left/i);

    // Tracking the selection is what makes it the group's description rather than a static caption:
    // each measure means something different by a bright cell.
    await user.click(within(group).getByRole('button', { name: 'How fast' }));
    expect(describedBy(group)).toMatch(/tokens per second/i);
  });
});

/**
 * The boundary the first version of the gate got wrong (found in review on #65).
 *
 * `some(graded)` is true with exactly one graded cell, and one cell is a degenerate domain:
 * `magnitudeFill` returns the brightest step whenever its span is not positive, so all three measures
 * paint that cell identically while every other cell keeps its categorical fill. The control was still
 * offered, and pressing it moved `aria-pressed` and nothing else — the same defect the gate was added
 * to fix, one cell further in.
 */
describe('the ramp control needs a ramp, not merely a graded cell', () => {
  /** Verified: 1 graded cell, 20 spilling, 35 over the ceiling. */
  const oneCell = {
    ...DEFAULT_CONFIG,
    modelId: 'Qwen/Qwen3-14B',
    deviceId: 'rtx-5070',
    quantId: 'q5_k_m',
    runtimeId: 'llama.cpp',
  };

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('offers no measures when exactly one cell is graded', async () => {
    const user = userEvent.setup();
    const painted = recordPainting();
    render(<Envelope config={oneCell} />);

    // The premise, asserted rather than assumed: exactly one cell takes a step of the ramp. Without
    // this the test passes on any grid that happens to have no graded cells at all, which is the
    // case one test above and a different branch.
    const table = await openTable(user);
    const rampCells = painted.fills.filter((f) => step(f) >= 0);
    expect(rampCells, 'this scenario no longer has exactly one graded cell').toHaveLength(1);
    expect(within(table).getAllByRole('cell').length).toBeGreaterThan(40);

    // No measure varies over a single cell, so the switch itself has nothing to offer.
    expect(within(field()).queryByRole('button', { name: 'Does it fit' })).toBeNull();
    expect(within(field()).queryByText(/the ramp runs between/i)).toBeNull();

    // And the ramp key goes with it: a two-ended scale drawn over one value is a key for a gradient
    // the picture does not contain.
    expect(within(field()).queryByText('less room')).toBeNull();
    expect(within(field()).queryByText('more room')).toBeNull();

    // The canvas sentence makes the same claim to a screen reader, so it is withheld too.
    const described = within(field()).getByRole('img').getAttribute('aria-label') ?? '';
    expect(described).not.toMatch(/coloured by/i);
    // It still says what the cells are, which is the part that does not depend on a ramp.
    expect(described).toMatch(/spilling|will not run|context/i);
  });
});

/**
 * The trap door the previous version of that gate opened (found in review on #65).
 *
 * One measure can be flat while another varies, and deriving the *switch* from the *selected* measure
 * meant choosing the flat one removed the only route to any other. A control that deletes itself in
 * response to being used is worse than the dead control the gate replaced.
 *
 * Gemma 3 27B at INT8 on the 36 GiB M4 Max under MLX with a 512-token prompt is the case: two graded
 * cells whose headroom (0.0071 vs 0.0129) and decode rate (11.81 vs 11.88 tok/s) differ, and whose
 * time to first token is identical to the last digit — so `fit` and `decode` have a ramp and `ttft`
 * has none.
 */
describe('a flat measure is a state, not a dead end', () => {
  const flatTtft = {
    ...DEFAULT_CONFIG,
    modelId: 'unsloth/gemma-3-27b-it',
    deviceId: 'mac-studio-m4-max-36',
    quantId: 'int8',
    runtimeId: 'mlx',
    promptTokens: 512,
  };

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('keeps the switch reachable after choosing a measure with no ramp', async () => {
    const user = userEvent.setup();
    recordPainting();
    render(<Envelope config={flatTtft} />);

    const group = () => within(field()).getByRole('group', { name: /colour the field by/i });
    expect(within(group()).getByRole('button', { name: 'How responsive' })).toBeInTheDocument();

    await user.click(within(group()).getByRole('button', { name: 'How responsive' }));

    // The switch survives its own use — this is the assertion the previous gate failed.
    expect(
      within(field()).queryByRole('group', { name: /colour the field by/i }),
      'choosing a flat measure removed the only route back'
    ).not.toBeNull();
    expect(within(group()).getByRole('button', { name: 'Does it fit' })).toBeInTheDocument();

    // But nothing claims a gradient: no ramp key, and the caption says why rather than trailing off.
    expect(within(field()).queryByText('slower to start')).toBeNull();
    expect(within(field()).queryByText(/the ramp runs between/i)).toBeNull();
    expect(within(field()).getByText(/reads the same on this measure/i)).toBeInTheDocument();

    // And switching back restores it, which is what makes the flat state a state.
    await user.click(within(group()).getByRole('button', { name: 'Does it fit' }));
    expect(within(field()).getByText('less room')).toBeInTheDocument();
    expect(within(field()).getByText(/the ramp runs between/i)).toBeInTheDocument();
  });
});
