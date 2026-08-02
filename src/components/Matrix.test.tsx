/**
 * The Matrix as a grid — geometry, bands, refusals, the roving tab index, the readout, and the
 * marks painted over the ramp (#115). Split out of `App.test.tsx`, which keeps the panels
 * agreeing with each other; the all-blocked pinned-catalog case lives in
 * `Matrix.allBlocked.test.tsx`, whose mock is incompatible with the bounded fixture here.
 *
 * These are the suites that hold most of the full-grid opt-ins: #52's roving index and #64's
 * rotated header were both caught by the real extent, and a shrunken grid would have passed
 * both. Every `atFullGrid()` below is that decision, written down.
 */
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '@/App';
import { useConfig, DEFAULT_CONFIG } from '@/store/config';
import { DEVICES, getModel } from '@/data/catalog';
import { colors, magnitudeRamp } from '@/design/tokens';
import { atFullGrid, boundGridByDefault } from '@/test/grid';
import { TABBABLE } from '@/test/tabbable';

/**
 * The Matrix's extent is bounded by default and the real grid opted into — the whole design,
 * and the fixture itself, live in `src/test/grid.ts` (#101, #115). The mock is declared here
 * because `vi.mock` is hoisted per test file and cannot ride an import; the fixture's own
 * preconditions are held in `src/test/grid.test.ts`.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

boundGridByDefault();

afterEach(() => {
  cleanup();
  useConfig.setState(DEFAULT_CONFIG);
});

describe('the Matrix stays informative', () => {
  const matrix = () => screen.getByRole('region', { name: /Every model on every machine/i });

  it('does not blank half the catalog when the format is expert-only', () => {
    // The claim is about the catalog, so it is measured against the catalog.
    atFullGrid();
    render(<App />);
    // The default quant is MXFP4, which applies to no dense model. Forcing it across the grid
    // reported "does not apply" for a majority of rows — a quantization fact standing in for a
    // hardware comparison, on the surface whose only job is comparing hardware.
    // The phrase appears in the header and again in the table caption; either will do.
    const [, ran, total] = within(matrix())
      .getAllByText(/combinations run/)[0]
      .textContent!.match(/(\d+)\D+(\d+)/)!;

    expect(Number(ran) / Number(total)).toBeGreaterThan(0.8);
    expect(within(matrix()).getByText(/where it does not apply/i)).toBeInTheDocument();
  });

  it('says what every cell means without relying on its colour', () => {
    atFullGrid();
    render(<App />);
    const cells = within(matrix()).getAllByRole('button', { name: / on / });
    expect(cells.length).toBeGreaterThan(100);
    // Each carries model, device and the measured figure.
    expect(cells[0]).toHaveAccessibleName(/ on .+:/);
  });

  it('loads a cell into the Bench when clicked', async () => {
    const user = userEvent.setup();
    render(<App />);

    const before = useConfig.getState().modelId;
    const cells = within(matrix()).getAllByRole('button', { name: / on / });
    await user.click(cells[cells.length - 1]);

    const after = useConfig.getState();
    expect(`${after.modelId}/${after.deviceId}`).not.toBe(`${before}/${DEFAULT_CONFIG.deviceId}`);
  });

  it('rearranges when the measure changes, which is the point of having three', async () => {
    const user = userEvent.setup();
    render(<App />);

    const fills = () =>
      within(matrix())
        .getAllByRole('button', { name: / on / })
        .map((b) => b.getAttribute('style'));

    const byFit = fills();
    await user.click(within(matrix()).getByRole('button', { name: 'How fast' }));
    expect(fills()).not.toEqual(byFit);
  });

  /**
   * The ramp has to be **spent** on this grid too, and it was the worse of the two surfaces
   * ([#97](https://github.com/MrZoller/bench/issues/97)).
   *
   * The Matrix has offered "How responsive" for longer than the Envelope has, and its domain was
   * floored at zero — which for a reciprocal reduces the placement to `t_fastest / t`, so a cell ten
   * times slower than the grid's fastest painted the bottom step on a grid spanning a desktop CPU to
   * a B200. Measured at the default scenario: **1,025 of 1,269 timed cells on one step of seven**,
   * 81%, against decode's healthy 262 at its busiest. After the fix, 323 at its busiest — 25%.
   *
   * Full grid deliberately. The claim is about the shape of the real field, and a dozen cells cannot
   * have one; a bounded grid would satisfy every threshold here for want of anything to distribute.
   */
  it('spends the ramp across the field, not on one step of it', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(matrix()).getByRole('button', { name: 'How responsive' }));

    /*
     * The painted cells, which are the ones on the scale — a pair that cannot run is `transparent`
     * by design and is not a low score. Reading the inline background is the same channel the ramp
     * arithmetic sweep below uses.
     */
    const painted = within(matrix())
      .getAllByRole('button', { name: / on / })
      .map((b) => (b as HTMLElement).style.background)
      .filter((fill) => fill && fill !== 'transparent');
    expect(painted.length, 'the grid painted nothing from the ramp').toBeGreaterThan(300);

    // jsdom serialises an inline colour as `rgb(r, g, b)`, so the ramp's hexes are put in the same
    // form rather than compared across notations — which silently matches nothing.
    const asRgb = (hex: string) =>
      `rgb(${[1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16)).join(', ')})`;
    const held = magnitudeRamp.map((step) => painted.filter((f) => f === asRgb(step)).length);
    expect(
      held.reduce((a, b) => a + b, 0),
      'no painted cell matched a step of the ramp, so the notations disagree'
    ).toBe(painted.length);
    /*
     * Two properties, and the second is the one the old code failed. "Both ends are reached" was
     * already true of the collapsed ramp — the fastest and slowest cells still landed at the ends —
     * so only the distribution tells the two apart.
     */
    expect(
      held.filter((n) => n === 0),
      'a step of the ramp went unused'
    ).toEqual([]);
    expect(
      Math.max(...held) / painted.length,
      `one step holds ${Math.max(...held)} of ${painted.length} painted cells`
    ).toBeLessThan(0.5);
  });
});

/**
 * Every figure on the grid used to be behind a native `title` (#71).
 *
 * A mouse, a second of dwell, one cell at a time, gone on the next move. That leaves the colour and
 * nothing else for three readers: a sighted keyboard user, who sees the ring move and never hears the
 * `aria-label`; a touch user, who has no hover at all and can only read a cell by committing to it,
 * which rewrites five store keys and scrolls several sections away; and anyone comparing two cells,
 * since a native tooltip shows one and dismisses it on the way to the second. `fill` log-scales onto
 * seven steps deliberately, so the colour is a rank and never a magnitude — which leaves no other
 * route to one.
 *
 * All of this is DOM: which string the line holds, when it fills, when it clears, and whether it is
 * announced. jsdom answers every one of those in a second. What it cannot answer is whether the
 * reservation actually holds a line of height, which is `e2e/matrix-readout.spec.ts` — the same split
 * the focus-indicator suite draws between the indicator a control *declares* and the one it paints.
 */
describe('the comparison grid puts a cell’s value where it can be read', () => {
  const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });

  /**
   * The readout, addressed structurally.
   *
   * It has no role of its own on purpose — see the `aria-hidden` test below — so there is no accessible
   * name to find it by. The section's only direct paragraph is unambiguous: the workload caveat sits
   * inside the `header` and the measure hint inside the `fieldset`. Deliberately *not* keyed on
   * `aria-hidden`, which is the subject of one of the tests below and would make that one circular —
   * and the assertions here all compare its text against a cell's own `aria-label`, so a locator that
   * found the wrong element could not pass for the wrong reason either way.
   */
  const line = () => matrix().querySelector<HTMLElement>(':scope > p')!;
  /**
   * The wide form, which is what every assertion below is about.
   *
   * The line holds two sentences since #102 — the full one and a preamble-less one for widths where
   * the reservation cannot hold the wrapping — and CSS shows one. jsdom resolves no media query, so
   * it renders both and `textContent` returns them concatenated; `data-readout` is the seam that
   * lets this ask for the one it means. The narrow form has its own tests below.
   */
  const readout = () => line().querySelector('[data-readout="full"]')!.textContent;
  const briefReadout = () => line().querySelector('[data-readout="brief"]')!.textContent;

  /**
   * A tap, with the focus the browser interposes — which is the whole of what makes this test real.
   *
   * Tapping a button focuses it, so `onFocus` fires *before* `click` and the readout is already this
   * cell's by the time the handler runs. Written as `pointerDown` then `click`, these tests passed
   * against a guard that compared the live readout target and therefore never fired — the first tap
   * committed, and so did a tap on a different cell. Raised in review on #102, and the shape is the
   * repo's own: an event sequence the real browser produces and the fixture did not.
   */
  const tap = (cell: HTMLElement) => {
    // `pointerenter` with a button already down, which is what a finger arriving *is* — and the
    // reason the readout's hover record does not pick it up. A mouse arrives with `buttons: 0`.
    fireEvent.pointerEnter(cell, { pointerType: 'touch', buttons: 1, pointerId: 7 });
    fireEvent.pointerDown(cell, { pointerType: 'touch', pointerId: 7 });
    // `act`, so React has *committed* the focus before the click — which is the half that makes this
    // bite. Without it the state update is still queued when `click` runs and the handler reads the
    // previous render's target, which is the value the fix stores deliberately: the test would pass
    // against the defect for the same reason the defect was invisible.
    act(() => {
      cell.focus();
    });
    // `detail: 1`, because a pointer-generated click carries a count and a keyboard one does not —
    // which is how the handler tells them apart. A bare `fireEvent.click` is `detail: 0` and would
    // be read as Enter.
    fireEvent.click(cell, { detail: 1 });
  };
  const cells = () => [...matrix().querySelectorAll<HTMLButtonElement>('td button')];
  const legend = () => [...matrix().querySelectorAll<HTMLElement>(':scope > div')].at(-1)!;

  /**
   * The narrow form, and the two halves of [#102](https://github.com/MrZoller/bench/issues/102) it
   * belongs to.
   *
   * The readout's reservation is in `rem`, so at a 32px root it doubles — and the glyphs double with
   * it, so the same sentence wraps into roughly twice as many lines each twice as tall. Reserved
   * height grows linearly and required height closer to quadratically, so the line escaped its box:
   * measured at 320px with the browser default at 32px, the longest sentence needs **280px against
   * 160px reserved**, and 240 against 160 at 390. Above `sm` it fits at both root sizes.
   *
   * What makes it long is the preamble, and at phone width the preamble is the part the reader
   * already has — the model is the row heading and the device the column heading. So the narrow form
   * states the figure alone. `e2e/reflow.spec.ts` measures that it now fits; these pin what it says.
   */
  it('states the figure alone at a width the preamble does not fit', async () => {
    const user = userEvent.setup();
    render(<App />);
    const cell = cells()[0];
    await user.hover(cell);

    const full = readout()!;
    const brief = briefReadout()!;

    /*
     * The model goes and the machine stays, which is "drop what is sticky, keep what scrolls": the
     * row heading is `sticky left-0` and on screen at every scroll position, while the column
     * headings sit at the top of a 35-row grid. Dropping both left a reader in a lower row with
     * `69% of the ceiling free` and nothing anywhere saying which machine (found in review).
     */
    expect(full).toMatch(/^Qwen3 8B on GeForce RTX 5090/);
    expect(brief).not.toContain('Qwen3 8B');
    expect(brief).toContain('RTX 5090');
    // The figure itself, which is the payload both forms carry.
    expect(brief).toContain('69% of the ceiling free');
    expect(full).toContain('69% of the ceiling free');
    /*
     * Materially shorter, which is the property the reservation depends on — the geometry itself is
     * `e2e/reflow.spec.ts`, since jsdom reports every height as 0. The margin is smaller than it was
     * when this dropped the device too, and deliberately: the sentence has to fit *and* say which
     * machine, and the browser check is what holds the first of those.
     */
    expect(brief.length).toBeLessThan(full.length * 0.8);
  });

  it('keeps the stand-in qualifier in the narrow form, since no axis carries it', async () => {
    const user = userEvent.setup();
    render(<App />);
    // MXFP4 is expert-only, so a dense row is scored at a substitute — the one part of the preamble
    // that is not on an axis, and the one a figure derived from it has to keep at every width.
    await user.hover(cells()[0]);

    expect(readout()).toMatch(/at Q4_K_M/);
    expect(briefReadout()).toMatch(/at Q4_K_M/);
  });

  it('gives a screen reader the full sentence at every width', () => {
    render(<App />);
    // The accessible name is the one channel with no axis headings beside it, so the preamble the
    // readout may drop is exactly what a screen-reader user has instead of them. Never abbreviated.
    const label = cells()[0].getAttribute('aria-label') ?? '';
    expect(label).toMatch(/^Qwen3 8B on GeForce RTX 5090/);
  });

  /**
   * Inspection separated from activation, for the reader who has only a tap.
   *
   * #71 left this open and named it: on a touch-only device the only gesture a cell offers is a tap,
   * and that tap *is* `onClick` — five store keys rewritten and a scroll several sections away. So
   * the readout either filled while navigation was already happening or never filled at all, and a
   * touch reader could not compare two cells. Unlike the Envelope and the budget bar, this panel has
   * no table behind a disclosure to fall back to: it *is* the table, and its cells show colour.
   *
   * **The rule is a state, not a gesture: you may commit to a cell whose figures you have already
   * been shown.** Keying on `pointerType === 'touch'` was the first draft and assumed `pen` hovers,
   * which a direct-contact stylus does not — so a pen tap fell straight through to activation exactly
   * as the unfixed touch path did. Reading the readout's target at `pointerdown` answers every input
   * from one comparison instead: a mouse has hovered, a hovering pen has hovered, a finger has not,
   * and the keyboard has no `pointerdown` at all.
   */
  it('fills the line on the first tap and loads the cell on the second', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    tap(cell);

    // Inspected, not committed.
    expect(readout()).toContain('RTX 3060');
    expect(useConfig.getState().deviceId).toBe(before);

    tap(cell);

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('loads a cell on a single click from a mouse, on the same markup', () => {
    render(<App />);
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    // A mouse arrives before it clicks, which is what makes one click enough — and the reason this
    // needs no pointer-type test: the hover *is* the inspection.
    fireEvent.pointerEnter(cell, { pointerType: 'mouse', buttons: 0, pointerId: 1 });
    fireEvent.mouseEnter(cell);
    fireEvent.pointerDown(cell, { pointerType: 'mouse', pointerId: 1 });
    fireEvent.click(cell, { detail: 1 });

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('asks a contact-only stylus for a second tap, since it never hovered', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    /*
     * A direct-contact stylus reports `pen` and cannot show the readout before it lands, so the
     * first draft's `pointerType === 'touch'` test let it through and it committed on contact — the
     * unfixed touch path, one pointer type over. Nothing here mentions `pen`: it inspects first
     * because it has not hovered, which is the same reason a finger does.
     */
    // The contact itself generates the enter, with the button already down — which is exactly why
    // it is not a hover, and why inferring provenance from the *reading* gesture's pointer type took
    // this for a mouse and committed on contact.
    fireEvent.pointerEnter(cell, { pointerType: 'pen', buttons: 1, pointerId: 3 });
    fireEvent.pointerDown(cell, { pointerType: 'pen', pointerId: 3 });
    act(() => {
      cell.focus();
    });
    fireEvent.click(cell, { detail: 1 });

    expect(readout()).toContain('RTX 3060');
    expect(useConfig.getState().deviceId).toBe(before);
  });

  it('keeps one click for a mouse that has not moved since the keyboard did', () => {
    render(<App />);
    const [under, elsewhere] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    /*
     * The mouse rests on one cell while the keyboard arrows to another — and a stationary mouse
     * emits no further `mouseenter`, so the record of where it is has to survive the focus move.
     * Expiring it there was the previous round's fix for a returning *tap* and it broke this: an
     * ordinary click needed two after any keyboard use. Provenance at `pointerdown` answers both,
     * which is why nothing expires here now (found in review).
     */
    fireEvent.pointerEnter(under, { pointerType: 'mouse', buttons: 0, pointerId: 1 });
    fireEvent.mouseEnter(under);
    // The premise: the hover really did register, or this measures a mouse that never arrived.
    expect(readout(), 'the mouse hover did not reach the readout').toContain('RTX 3060');
    act(() => {
      elsewhere.focus();
    });
    // And the keyboard really did take the line, which is the state the record has to survive.
    expect(readout(), 'the keyboard did not take the readout').toContain('DGX Spark');

    fireEvent.pointerDown(under, { pointerType: 'mouse', pointerId: 1 });
    fireEvent.click(under, { detail: 1 });

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('does not let a finger inherit a mouse’s hover on a hybrid', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const [hovered, focusedElsewhere] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    /*
     * A mouse hovers one cell, the keyboard moves the readout to another, and a finger then lands on
     * the hovered one. Without an identity on the record the finger inherits the mouse's hover and
     * commits on its first tap, while the line still describes the cell the keyboard is on — which is
     * the whole gesture broken on any laptop with a touchscreen (found in review).
     */
    fireEvent.pointerEnter(hovered, { pointerType: 'mouse', buttons: 0, pointerId: 1 });
    act(() => {
      focusedElsewhere.focus();
    });

    fireEvent.pointerEnter(hovered, { pointerType: 'touch', buttons: 1, pointerId: 9 });
    fireEvent.pointerDown(hovered, { pointerType: 'touch', pointerId: 9 });
    act(() => {
      hovered.focus();
    });
    fireEvent.click(hovered, { detail: 1 });

    expect(useConfig.getState().deviceId, 'the finger committed on its first tap').toBe(before);
    expect(readout()).toContain('RTX 3060');
  });

  it('names the runtime when a narrow readout reports a refusal', async () => {
    const user = userEvent.setup();
    render(<App />);
    // MLX is Apple-only, so most columns are struck and every cell in them is a refusal.
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const refused = cells().find((c) =>
      /does not (run|support)/i.test(c.getAttribute('aria-label') ?? '')
    )!;
    expect(refused, 'no refused cell under MLX, so this has no subject').toBeDefined();
    await user.hover(refused);

    /*
     * The machine comes from the column heading and the runtime does not — the Runtime picker is
     * above the grid and off screen for a lower row, so eliding it left "the runtime does not drive
     * this", which is the one fact the axes cannot supply (found in review).
     */
    expect(briefReadout()).toContain('MLX (Apple)');
    expect(briefReadout()).not.toMatch(/^the runtime/i);
  });

  it('loads a cell from the keyboard, with no pointer gesture in front of it', () => {
    render(<App />);
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    // Enter on a focused cell fires `click` with no `pointerdown`, and focus has already filled the
    // line — so there is nothing the gesture could have meant other than "load this".
    act(() => {
      cell.focus();
    });
    fireEvent.click(cell);

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('lets the keyboard activate after a tap, rather than inheriting it', () => {
    render(<App />);
    const [tapped, typed] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    /*
     * The snapshot is consumed on every click, and this is why. Held, it would still describe the
     * tap when a later keyboard `click` arrived with no `pointerdown` of its own — so Enter would be
     * read as that tap still in progress and refuse to activate, for ever, since nothing else would
     * clear it. Found in review.
     */
    tap(tapped);
    act(() => {
      typed.focus();
    });
    fireEvent.click(typed);

    expect(useConfig.getState().deviceId).toBe('dgx-spark');
  });

  it('moves the line to the next cell a finger taps rather than loading it', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const [first, second] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    tap(first);
    tap(second);

    // Comparing two cells is the thing #71 said a touch reader could not do, and the second tap
    // going to a *different* cell is what makes it possible without committing to either.
    expect(readout()).toContain('DGX Spark');
    expect(useConfig.getState().deviceId).toBe(before);
  });

  it('reserves the line before anything is pointed at', () => {
    render(<App />);

    // Present and empty, rather than absent until it has something to say: a line that appears
    // reflows whatever is under it every time a reader moves between two cells. Whether the
    // reservation is worth a line of height is geometry, and geometry is e2e's half.
    expect(line()).toBeInTheDocument();
    expect(readout()).toBe('');
    expect(line().className).toMatch(/min-h-/);
  });

  it('fills on focus, which is the half that answers the keyboard', () => {
    render(<App />);
    const cell = cells()[5];

    act(() => cell.focus());

    // The same sentence the cell announces, not a second wording of it: one `tooltip()` call feeding
    // both channels is what keeps the line and the accessible name from drifting apart.
    expect(readout()).toBe(cell.getAttribute('aria-label'));
    expect(readout()).toMatch(/ on .+:/);

    act(() => cell.blur());
    expect(readout()).toBe('');
  });

  it('fills on hover, without the dwell the tooltip charged for', () => {
    render(<App />);
    const cell = cells()[7];

    fireEvent.mouseEnter(cell);
    expect(readout()).toBe(cell.getAttribute('aria-label'));

    fireEvent.mouseLeave(cell);
    expect(readout()).toBe('');
  });

  it('falls back to the focused cell when the pointer leaves, rather than blanking', () => {
    render(<App />);
    const [held, pointed] = [cells()[0], cells()[9]];

    act(() => held.focus());
    fireEvent.mouseEnter(pointed);
    // The pointer wins while there is one — it is the more recent intent.
    expect(readout()).toBe(pointed.getAttribute('aria-label'));

    fireEvent.mouseLeave(pointed);
    // And the focus ring is still drawn on the first cell, so its sentence is still the true one. A
    // single `hovered` flag blanks here, which is a visible mark with nothing on the page explaining
    // it — the disagreement `isCurrent` exists to prevent, pointed the other way.
    expect(document.activeElement).toBe(held);
    expect(readout()).toBe(held.getAttribute('aria-label'));
  });

  /**
   * And the other order, which is the one that shipped broken.
   *
   * `hovered ?? focused` is the fallback above with no way back out of it: a pointer resting anywhere
   * on the grid fires `mouseenter` and, because the mouse does not move, never fires `mouseleave`. A
   * reader who then tabs in and arrows across a row moves the ring while the line goes on printing the
   * cell the pointer happens to be sitting on — indefinitely, and with no `hover:` style on the cells,
   * the ring is the only mark on screen. That is a *wrong* figure where #71's sighted keyboard reader
   * previously got none, which is worse than the defect being fixed.
   *
   * No contrivance in this test but the missing `mousemove`, which is the whole point: the pointer is
   * where it was and the keyboard has moved twice.
   */
  it('lets the keyboard take the line back from a pointer that has stopped moving', () => {
    atFullGrid();
    render(<App />);
    const resting = cells()[0];

    fireEvent.mouseEnter(resting);
    expect(readout()).toBe(resting.getAttribute('aria-label'));

    const stepped = cells()[300];
    act(() => stepped.focus());
    fireEvent.keyDown(stepped, { key: 'ArrowRight' });
    const arrowed = document.activeElement as HTMLButtonElement;

    // The ring moved, so the line has to have moved with it.
    expect(arrowed).not.toBe(resting);
    expect(readout()).toBe(arrowed.getAttribute('aria-label'));
    expect(readout()).not.toBe(resting.getAttribute('aria-label'));

    // And the pointer is not disabled by having been outranked once: the next `mouseenter` is a new
    // move, and the rule is that the newer input wins.
    const pointed = cells()[7];
    fireEvent.mouseEnter(pointed);
    expect(readout()).toBe(pointed.getAttribute('aria-label'));
  });

  /**
   * And the way back that `mouseenter` alone cannot provide (found in review on #71).
   *
   * The test above proves the pointer can reclaim the line by entering a *different* cell. It cannot
   * prove it can reclaim the cell it is already in, and it could not: `mouseenter` needs a boundary
   * crossing, so once `onFocus` expired the hover claim, a reader whose mouse was resting on a cell
   * had to leave it and come back. Last-input-wins held in one direction only, and the direction it
   * failed in is the one a mixed keyboard-and-mouse reader hits first — the mouse is where they left
   * it, and the hand that moves is the one already on it.
   *
   * `mousemove` without a preceding `mouseenter` is exactly that state, and it is why the contrivance
   * is the *absence* of an enter rather than the presence of a move.
   */
  it('lets a pointer that never left reclaim the line by moving in place', () => {
    atFullGrid();
    render(<App />);
    const resting = cells()[3];

    fireEvent.mouseEnter(resting);
    expect(readout()).toBe(resting.getAttribute('aria-label'));

    // Keyboard takes it, which also expires the pointer's claim.
    const stepped = cells()[200];
    act(() => stepped.focus());
    expect(readout()).toBe(stepped.getAttribute('aria-label'));

    // The pointer has not moved between cells and so fires no `mouseenter` — only a move inside the
    // one it is already in.
    fireEvent.mouseMove(resting);
    expect(
      readout(),
      'a pointer resting on a cell cannot get the line back without leaving it'
    ).toBe(resting.getAttribute('aria-label'));

    // The focus ring has not moved, so the fallback is still the stepped cell once the pointer goes.
    expect(document.activeElement).toBe(stepped);
    fireEvent.mouseLeave(resting);
    expect(readout()).toBe(stepped.getAttribute('aria-label'));
  });

  /**
   * The line is derived from where the reader is, not stored when they get there.
   *
   * Reachable without contriving anything: a pointer resting on a cell while the keyboard drives one
   * of the Usage sliders — which is what this does, since that is exactly what a slider does to the
   * store. Nothing moves the pointer, so no `mouseleave` fires, and every figure on the grid changes
   * underneath it. A stored string would keep quoting the old scenario over the new grid.
   */
  it('tracks the scenario instead of freezing the sentence it arrived with', () => {
    render(<App />);
    // A cell that runs, so it has a figure to report at all.
    const cell = cells().find((c) => /: \d/.test(c.getAttribute('aria-label') ?? ''))!;

    fireEvent.mouseEnter(cell);
    const before = readout();
    expect(before).toBe(cell.getAttribute('aria-label'));

    act(() => useConfig.getState().set('contextTokens', 131072));

    expect(readout()).not.toBe(before);
    expect(readout()).toBe(cell.getAttribute('aria-label'));
  });

  /**
   * The one place this deliberately departs from `BudgetBar`'s hint, which is `aria-live="polite"`.
   *
   * There the sentence exists nowhere else — a legend item is named "Weights 14 GiB" and the
   * explanation is not part of that name. Here the line is a verbatim copy of the focused cell's
   * accessible name, so a live region would announce every cell twice: once as the name, once as the
   * update, on every arrow key across a 42-column row. The channel #71 is missing is the visual one.
   */
  it('does not read the cell’s own sentence out a second time', () => {
    render(<App />);
    const cell = cells()[11];

    act(() => cell.focus());

    expect(line().getAttribute('aria-hidden')).toBe('true');
    expect(line().getAttribute('aria-live')).toBeNull();
    // And the spoken channel is untouched: the cell still carries the whole sentence.
    expect(cell.getAttribute('aria-label')).toBe(readout());
  });

  /**
   * The sweep. #71 names the cells, and the same defect was live on both headings.
   *
   * `headerColumns` shortens every device name — the vendor line goes, and the bracketed qualifier
   * with it wherever the stem is already unique — and the full name lived only in a `title`. The row
   * heading truncates at 9rem and put the model name *and* its parameter count in one. Three
   * hover-only strings, one line to put them in.
   */
  it('names the machine a shortened column heading stands for', () => {
    render(<App />);
    const heading = [...matrix().querySelectorAll('thead th')].at(-1)!;
    const label = heading.querySelector('span[title]')!;

    fireEvent.mouseEnter(heading);

    // The full catalog name, which is longer than what the column can print.
    expect(readout()).toBe(label.getAttribute('title'));
    expect(readout()!.length).toBeGreaterThan(label.textContent!.length);
  });

  it('carries the runtime refusal a struck heading only says in colour and ink', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const struck = [...matrix().querySelectorAll('thead th')].find((th) =>
      th.querySelector('span[title]')?.className.includes('line-through')
    )!;
    fireEvent.mouseEnter(struck);

    // Same string as the heading's own `aria-label`, from the same derivation — the strike is the
    // sighted channel and it says nothing about why on its own.
    expect(readout()).toBe(struck.getAttribute('aria-label'));
    expect(readout()).toMatch(/does not support this hardware, at any size/i);
  });

  it('punctuates a refusal once, now that the sentence is printed rather than hovered', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    // vLLM drives no Mac, no Strix Halo and no CPU host, so this fills the grid with the refusals
    // `planPlacement` writes — each of which already ends in a full stop.
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const blocked = cells().filter((c) =>
      /does not run on/i.test(c.getAttribute('aria-label') ?? '')
    );
    expect(blocked.length).toBeGreaterThan(50);

    fireEvent.mouseEnter(blocked[0]);
    expect(readout()).toMatch(/does not run on .+[^.]\.$/);
    for (const cell of blocked) expect(cell.getAttribute('aria-label')).not.toContain('..');
  });

  it('spells out a truncated model row, parameter count and all', () => {
    atFullGrid();
    render(<App />);
    const heading = matrix().querySelectorAll('tbody th')[3];

    fireEvent.mouseEnter(heading);

    expect(readout()).toBe(heading.getAttribute('title'));
    expect(readout()).toContain(heading.textContent);
    // The parameter count, which had no channel at all besides that `title`.
    expect(readout()).toMatch(/\d+(\.\d+)?B$/);
  });

  /**
   * And the parameter count needs a spoken channel of its own, which the line cannot be.
   *
   * The readout is `aria-hidden` for the reason above — it copies a cell's accessible name, so a live
   * region says every cell twice — and these headings take no focus, so hover was the *only* route to
   * the count: a cell's sentence names the model but never its size. An `aria-label` on the row header
   * is the same answer the column header already uses for its runtime refusal, from the same one
   * derivation, and a grid announces a row header once per row rather than once per cell.
   */
  it('tells a screen reader the row’s parameter count, which no cell states', () => {
    atFullGrid();
    render(<App />);
    const headings = [...matrix().querySelectorAll('tbody th')];

    expect(headings.length).toBeGreaterThan(10);
    for (const heading of headings) {
      // One string in all three channels — the visible name, the tooltip and the accessible name.
      expect(heading.getAttribute('aria-label')).toBe(heading.getAttribute('title'));
      expect(heading.getAttribute('aria-label')).toMatch(/\d+(\.\d+)?B$/);
      expect(heading.getAttribute('aria-label')).toContain(heading.textContent);
    }
  });

  /**
   * And the ramp is a scale now rather than an ordering.
   *
   * `worse [gradient] better` says which way to read the colour and nothing about what it spans, so a
   * mid-blue under "How fast" could be 20 tok/s or 200. The endpoints are asserted against the
   * *cells'* own sentences rather than recomputed here: whatever the engine says, the figure at each
   * end of the legend has to be the figure some cell reports, and the extreme one.
   */
  const spoken = (cells: HTMLButtonElement[], pattern: RegExp) =>
    cells.flatMap((cell) => {
      const found = pattern.exec(cell.getAttribute('aria-label') ?? '');
      return found ? [found[1]] : [];
    });

  /**
   * The extreme is chosen by number and asserted as the *string the cell printed*.
   *
   * `Math.min(...rates.map(Number))` and `toContain` looks equivalent and is not: `rate()` keeps a
   * decimal below 10 tok/s, so the day the slowest running cell lands on an exact tenth-free value the
   * legend correctly renders "worse 3.0 tok/s" while the round trip through `Number` asks for "worse
   * 3 tok/s" — a red test with nothing wrong in the code. It passes today only because the minimum is
   * 0.5. Same reason the TTFT test below compares by `asSeconds` and asserts the figure verbatim.
   */
  const extremeBy = (figures: string[], value: (figure: string) => number, want: 'min' | 'max') =>
    figures.reduce((a, b) =>
      want === 'min' ? (value(b) < value(a) ? b : a) : value(b) > value(a) ? b : a
    );

  it('anchors the ramp with the throughput its own cells report', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(matrix()).getByRole('button', { name: 'How fast' }));

    const rates = spoken(cells(), /: ([\d.]+) tok\/s per user\.$/);
    expect(rates.length).toBeGreaterThan(100);

    const text = legend().textContent!;
    expect(text).toContain(`worse ${extremeBy(rates, Number, 'min')} tok/s`);
    expect(text).toContain(`${extremeBy(rates, Number, 'max')} tok/s better`);
  });

  it('puts the longest wait at the worse end, which the stored value inverts', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(matrix()).getByRole('button', { name: 'How responsive' }));

    /** "188 ms", "2.7 s", "84 min" — one axis, three units, as `seconds()` prints them. */
    const UNIT: Record<string, number> = { ms: 0.001, s: 1, min: 60 };
    const asSeconds = (figure: string) => {
      const [value, unit] = figure.split(' ');
      return Number(value) * UNIT[unit];
    };

    const waits = spoken(cells(), /: (.+) to first token\.$/);
    expect(waits.length).toBeGreaterThan(100);
    const longest = extremeBy(waits, asSeconds, 'max');
    const shortest = extremeBy(waits, asSeconds, 'min');

    // `measureValue` inverts TTFT so that larger is better, so the ramp's worst end is the *slowest*
    // machine. An endpoint pair ordered by the number it prints puts the fastest one under "worse"
    // and leaves every colour on the grid correct, which is what makes it invisible.
    const text = legend().textContent!;
    expect(text).toContain(`worse ${longest}`);
    expect(text).toContain(`${shortest} better`);
    expect(asSeconds(longest)).toBeGreaterThan(asSeconds(shortest));
  });

  it('anchors the fit ramp with the most headroom any cell has', () => {
    atFullGrid();
    render(<App />);

    const free = spoken(cells(), /: (\d+)% of the ceiling free\.$/);
    expect(free.length).toBeGreaterThan(100);

    const text = legend().textContent!;
    expect(text).toContain(`${extremeBy(free, Number, 'max')}% free better`);
  });

  /**
   * And the low end of the *fit* ramp is the one endpoint that is not a cell's own figure.
   *
   * `measureValue('fit')` collapses every offloaded cell to zero headroom deliberately, so the dark
   * end is a population: a pair that just fits and a pair spilling most of its weights paint the same
   * square. "0% free" is the one statement true of all of them, which is why it is what the label
   * says — and the consequence is that the figure appears on no tooltip, since a spilled cell's own
   * sentence quotes the spill instead. Pinned rather than left to `/worse \d+% free/`, which any digit
   * satisfies: printing the tied cell's own 66% spill would describe every other square wrongly, and
   * `measureRange` hands back whichever tied cell came first in row-major order, so a label reading
   * any other field off it would be reading an arbitrary cell.
   */
  it('says the fit ramp’s dark end has no headroom, not what the worst cell spills', () => {
    render(<App />);

    const spilling = cells().filter((c) =>
      /spilling \d+% of its weights/.test(c.getAttribute('aria-label') ?? '')
    );
    expect(spilling.length, 'no cell spills, so the low end is a resident cell').toBeGreaterThan(0);

    const text = legend().textContent!;
    expect(text).toContain('worse 0% free');
    expect(text).not.toMatch(/worse \d+% (spilled|of its weights)/);
  });
});

/**
 * "Will not run" and "this runtime cannot drive it" were the same empty cell (#72).
 *
 * Select vLLM and every Mac, every Strix Halo and every CPU host empties out completely — 10 of the
 * 24 shipping columns as the catalog stands at this commit — every cell drawn `transparent` behind
 * the same dashed border as a pair that was measured and did not fit, under the same one-line
 * legend. A uniformly empty column is the pattern that reads as a confident finding, so the picture
 * said "this hardware cannot hold the model" — quantitatively backwards, since a 256 GB Mac Studio
 * holds Qwen3 8B many times over, and the fix a reader would derive from it (buy more memory) is not
 * the fix (change runtime). Every other surface already split them: the Envelope has an
 * `unsupported` state with its own sentence, Telemetry says `Unsupported` rather than `Will not
 * run`, and BudgetBar draws no stack at all.
 *
 * All of it is DOM, so all of it is here. The one thing jsdom cannot answer is whether
 * `line-through` and a dropped border actually *paint* — Tailwind classes are strings in this
 * environment — which is `e2e/matrix-undrivable.spec.ts`.
 */
describe('the Matrix tells a runtime refusal from a memory one', () => {
  const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });

  /**
   * Every device column, paired with its own cells.
   *
   * Read out of the DOM in column order rather than zipped against the catalog, for the reason the
   * header suite gives about its own pairing: the association between a heading and the cells under
   * it is part of what is being tested, and an assertion that assumes it cannot catch it going
   * wrong.
   */
  const columns = () => {
    const rows = [...matrix().querySelectorAll('tbody tr')];
    return [...matrix().querySelectorAll('thead th')].slice(1).map((th, i) => {
      const label = th.querySelector<HTMLElement>('span[title]')!;
      return {
        head: th,
        device: label.getAttribute('title') ?? '',
        struck: label.className.includes('line-through'),
        spoken: th.getAttribute('aria-label'),
        cells: rows.map((row) => row.querySelectorAll<HTMLButtonElement>('td button')[i]),
      };
    });
  };

  const legendKey = () =>
    within(matrix()).queryByText(/does not support this hardware, at any size/i);

  const caption = () => matrix().querySelector('caption')!.textContent ?? '';

  it('strikes the columns the runtime cannot drive, and only those', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);

    // llama.cpp drives every class of hardware in the catalog, so nothing is struck and nothing is
    // keyed — the precondition that keeps the vLLM half below from passing for a trivial reason.
    expect(columns().every((c) => !c.struck)).toBe(true);
    expect(legendKey()).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const struck = columns().filter((c) => c.struck);
    // Both sides populated: vLLM drives NVIDIA and AMD cards and drives no Mac, no Strix Halo and
    // no CPU host. A grid struck everywhere, or nowhere, would make every claim below vacuous.
    expect(struck.length).toBeGreaterThan(1);
    expect(struck.length).toBeLessThan(columns().length);
    expect(struck.some((c) => /Mac Studio/.test(c.device))).toBe(true);
    expect(columns().some((c) => !c.struck && /RTX 5090/.test(c.device))).toBe(true);

    /**
     * And the strike is the engine's own verdict rather than a second opinion about it.
     *
     * The component decides from `runtimeDrives` while every cell's refusal comes from
     * `planPlacement`'s own copy of that check, so this is the assertion that keeps the two from
     * drifting: a struck column's cells must *all* carry the runtime-level reason, and no cell
     * anywhere else may carry it.
     *
     * Marking a column that merely came up empty would be the same misattribution pointed the other
     * way. At #72's own URL the two sets happen to coincide — the DGX Spark still runs a good share
     * of its rows there, so the only empty columns are the undrivable ones — which is exactly why
     * deriving from emptiness looks safe. Take that grid to 32 concurrent users and the RTX 3090,
     * 4090 and 5080 columns empty out too, on counted bytes, under a runtime that drives all three.
     */
    for (const column of struck) {
      for (const cell of column.cells) {
        expect(cell).toHaveAccessibleName(/vLLM does not run on/i);
      }
      expect(column.spoken).toMatch(/vLLM does not support this hardware, at any size/i);
      // The device name stays in it: the visible label is deliberately shortened, so a name that
      // said only the runtime would trade one missing fact for another.
      expect(column.spoken).toContain(column.device);
      // And the sentence really is the column's accessible name rather than an attribute nothing
      // reads — this is the whole channel a reader who cannot see the strike has.
      expect(column.head).toHaveAccessibleName(column.spoken!);
    }
    for (const column of columns().filter((c) => !c.struck)) {
      for (const cell of column.cells) {
        expect(cell).not.toHaveAccessibleName(/does not run on/i);
      }
      // No name at all, so the heading keeps announcing the device it names.
      expect(column.spoken).toBeNull();
    }
  });

  it('keeps the dashed swatch for the cells that were actually measured', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const dashed = (cell: HTMLButtonElement) => cell.className.includes('border-dashed');

    // The swatch keys "measured, and over the ceiling". A column the runtime cannot open at all was
    // never measured, so it wears no ink — which is what stops the two states being byte-identical.
    for (const column of columns().filter((c) => c.struck)) {
      expect(column.cells.some(dashed)).toBe(false);
    }

    /**
     * And the capacity refusal still has its swatch, on a column the runtime does drive.
     *
     * Asserted rather than assumed: DeepSeek V3 does not fit a 3090 under any runtime that can load
     * it, so this is reachable — but if it ever stopped being, the assertion above would be the only
     * one left and "no cell has a dashed border" is a state this fix must not produce.
     */
    const measured = columns()
      .filter((c) => !c.struck)
      .flatMap((c) => c.cells)
      .filter(dashed);
    expect(measured.length).toBeGreaterThan(0);
    for (const cell of measured) {
      expect(cell).toHaveAccessibleName(/does not fit|past the default allocation/i);
    }
  });

  /**
   * Every hole on the grid, and nothing else, sits under a struck heading.
   *
   * The exhaustive version of the assertion above, and the one that keeps the two predicates from
   * coming apart in the direction nothing else watches. The heading is struck from `runtimeDrives`;
   * the cell's ink is dropped on `evaluated`, which `planPlacement` clears on **five** categorical
   * grounds, of which "this runtime does not drive this device" is one. The other four are filtered
   * out upstream today — the store coerces `kvPrecision` into `runtime.kvPrecisions`, `quantFor`
   * only ever returns a format the runtime lists, and this grid hardcodes one device per cell — so
   * the two sets coincide, and an assertion that only checked the `!drives` wording would keep
   * passing on the day one of the other four became reachable. That day the grid grows a column of
   * identical unexplained holes, which is #72 restated with a different ground.
   *
   * A hole is read off what the grid paints rather than out of the engine: `transparent` and no
   * dashed border is exactly "not judged on its numbers", since `fill` returns the panel surface for
   * anything that does not run and the border is what separates counted bytes from a categorical
   * refusal. Asserted as an equality in both directions, so it fails for a stray hole *and* for a
   * struck column whose cells kept their ink.
   */
  it('leaves no hole on the grid that a struck heading does not explain', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const hole = (cell: HTMLButtonElement) =>
      (cell.getAttribute('style') ?? '').includes('transparent') &&
      !cell.className.includes('border-dashed');

    const holes = columns()
      .filter((c) => !c.struck)
      .flatMap((c) => c.cells.filter(hole).map((cell) => `${c.device}: ${cell.ariaLabel}`));
    expect(holes, 'cells refused before the arithmetic with no struck heading saying why').toEqual(
      []
    );

    const closed = columns().filter((c) => c.struck);
    expect(closed.length).toBeGreaterThan(1);
    for (const column of closed) {
      expect(column.cells.every(hole)).toBe(true);
      // And the proxy really is reading refusals rather than figures, so "every cell is a hole"
      // cannot be satisfied by a grid that stopped measuring.
      for (const cell of column.cells) {
        expect(cell).not.toHaveAccessibleName(
          /of the ceiling free|tok\/s|to first token|spilling/i
        );
      }
    }
  });

  /**
   * A square with no ink is not a control.
   *
   * The other half of narrowing the border: these cells now have nothing drawn in them at all, and
   * they were still enabled buttons in the arrow-key sequence whose click set five config keys and
   * smooth-scrolled three sections up to a Bench that can only blank. `tokens.ts` puts the rule as
   * "a control's boundary is what identifies it as interactive, so it needs the 3:1 non-text minimum
   * *before* it is focused", and records `--color-border` at 1.18:1 — so no hairline was going to
   * make these look interactive either. They are inert instead.
   *
   * Still focusable and still named, which is why `aria-disabled` rather than `disabled`: a disabled
   * button takes no focus, so the arrows would stop dead at the first struck column and the per-cell
   * sentence — the only channel that says which machine and which runtime — would go with it.
   */
  it('makes a closed column inert without taking it out of the grid', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const closed = columns().filter((c) => c.struck);
    const open = columns().filter((c) => !c.struck);
    expect(closed.length).toBeGreaterThan(1);
    expect(open.length).toBeGreaterThan(1);

    for (const column of closed) {
      for (const cell of column.cells) {
        expect(cell).toHaveAttribute('aria-disabled', 'true');
        expect(cell.className).toContain('cursor-not-allowed');
        // Not `disabled`: it has to keep taking focus for the roving tab stop to cross the column.
        expect(cell.disabled).toBe(false);
      }
    }
    for (const column of open) {
      for (const cell of column.cells) {
        expect(cell).not.toHaveAttribute('aria-disabled');
        expect(cell.className).not.toContain('cursor-not-allowed');
      }
    }

    // And clicking one loads nothing. `aria-disabled` is advisory — the browser still fires the
    // click — so the handler has to refuse it, which is what this actually checks.
    const before = useConfig.getState();
    await user.click(closed[0].cells[0]);
    const after = useConfig.getState();
    expect(`${after.modelId}/${after.deviceId}/${after.quantId}`).toBe(
      `${before.modelId}/${before.deviceId}/${before.quantId}`
    );

    // While a column the runtime does drive still adopts its cell, so the refusal above is the
    // narrow one and not a click handler that stopped working.
    await user.click(open[0].cells[0]);
    expect(useConfig.getState().deviceId).not.toBe(before.deviceId);
  });

  it('keys the strike in the legend, and only while the grid holds one', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    // The Envelope's reviewed sentence, with the runtime named — one wording for one refusal.
    expect(legendKey()).toBeInTheDocument();
    expect(legendKey()).toHaveTextContent(/vLLM does not support this hardware, at any size/i);
    // The sample is the mark: struck text, not a swatch beside it.
    expect(legendKey()!.querySelector('.line-through')).not.toBeNull();

    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    // Still keyed, and now naming MLX — the sentence follows the runtime rather than being frozen
    // at whichever one first rendered it.
    expect(legendKey()).toHaveTextContent(/MLX \(Apple\) does not support this hardware/i);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    expect(legendKey()).not.toBeInTheDocument();
  });

  it('states the closed columns in the caption, which is the channel with no strike to see', async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(caption()).not.toMatch(/does not support/i);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    /**
     * The count read back out of the sentence and checked against the grid, rather than written
     * here as a literal.
     *
     * A hard-coded "10 of 24" would pass a caption that had stopped counting and would fail every
     * time the catalog gains a device. What matters is that the three channels agree: the number of
     * struck headings, the number of columns, and the sentence a screen-reader user hears instead of
     * seeing either.
     */
    const stated = caption().match(
      /(\d+) of the (\d+) device columns are hardware vLLM does not support at any size/i
    );
    expect(stated, `the caption does not state the closed columns: "${caption()}"`).not.toBeNull();
    expect(Number(stated![1])).toBe(columns().filter((c) => c.struck).length);
    expect(Number(stated![2])).toBe(columns().length);
    // And it says which way to read the empty column, since that is the whole misreading.
    expect(caption()).toMatch(/not for want of memory/i);
  });
});

/**
 * The device headers are rotated 45 degrees, which costs height *and* width — one length, spent
 * twice. #64 is what happened when the component derived it once: 246px of header band reserved at
 * every viewport, from a 40-character label, while the same label's sideways extent went unreserved
 * and leaned 142px out of a scroll container the grid otherwise fitted exactly. The app paid a phone
 * screen of vertical space for four names it then cut off.
 *
 * Split across the two suites the way the quantity itself splits. The *derivation* is a string —
 * label text, an inline height, an inline `min-width`, a rotation class — and jsdom reads all four in
 * milliseconds. What those lengths buy in a laid-out browser is `e2e/matrix-header.spec.ts`, because
 * jsdom reports every width on this surface as 0, which is exactly why the sideways half went
 * unnoticed.
 */
describe('the Matrix header reserves the rotation once', () => {
  // #64 is arithmetic over the labels the catalog actually renders — a 40-character name at 45
  // degrees — and the qualifier rule only has something to do where two rows share a name stem.
  // A four-column fixture reproduces neither, so this whole describe takes the real header.
  beforeEach(atFullGrid);
  const matrix = () => screen.getByRole('region', { name: /Every model on every machine/i });

  /**
   * The rendered labels, each paired with the device it names.
   *
   * Read off the `title`, which carries the full catalog name, rather than by zipping the header
   * against `DEVICES` in column order — the pairing is the thing under test, and an assertion that
   * assumes it cannot catch it going wrong.
   */
  const headerLabels = () =>
    [...matrix().querySelectorAll('thead th span[title]')].map((span) => ({
      name: span.getAttribute('title') ?? '',
      label: span.textContent ?? '',
    }));

  /** The part of a name that is not the vendor line and not the trailing parenthetical. */
  const stem = (name: string) =>
    name.replace(/^(GeForce|Instinct|Radeon)\s+/, '').replace(/\s*\([^)]*\)\s*$/, '');

  it('drops the qualifier from every column that can be identified without it', () => {
    render(<App />);
    const labels = headerLabels();
    // The header is the whole shipping catalog; a locator that found four of them would make
    // everything below it pass for the wrong reason.
    expect(labels.length).toBe(DEVICES.filter((d) => d.status === 'shipping').length);

    for (const { label } of labels) {
      expect(label, `"${label}" still spends characters on brackets`).not.toMatch(/[()]/);
    }

    /**
     * Minimal, stated as a rule rather than against a list of names: a label may only be longer
     * than its own stem where another column answers to that same stem.
     *
     * This is the half of the fix that is easy to get wrong in the other direction. Stripping the
     * parenthetical unconditionally is the obvious reading of the issue and it reintroduces the
     * defect the rotation exists to prevent — the three Mac Studio M3 Ultra rows differ *only* in
     * capacity, so they would collapse into one string three columns wide, and a header that
     * cannot distinguish its own columns is worse than none.
     */
    const stems = labels.map((l) => stem(l.name));
    for (const { name, label } of labels) {
      if (label === stem(name)) continue;
      expect(
        stems.filter((s) => s === stem(name)).length,
        `"${label}" is longer than "${stem(name)}", which no other column answers to`
      ).toBeGreaterThan(1);
      expect(label.startsWith(stem(name))).toBe(true);
    }

    // And the rule bites on the shipped catalog rather than being vacuously true of it.
    expect(labels.filter(({ name, label }) => label !== stem(name)).length).toBeGreaterThan(1);
    expect(new Set(labels.map((l) => l.label)).size).toBe(labels.length);
  });

  it('spends one derived length on the band and on the column it leans over, not two', () => {
    render(<App />);

    const band = matrix().querySelector<HTMLElement>('thead th:nth-child(2)');
    const reservation = matrix().querySelector<HTMLElement>('thead th:first-child');
    expect(band, 'the header row reserves no height at all').not.toBeNull();

    const bandRem = parseFloat(band!.style.height);
    const leanRem = parseFloat(reservation!.style.minWidth);
    expect(
      leanRem,
      'the model column reserves no room for the labels leaning over it'
    ).toBeGreaterThan(0);
    // sin(45) and cos(45) are the same number: the band is the lean plus the row's own padding, so
    // the two axes cannot drift apart without this failing.
    expect(bandRem - leanRem).toBeCloseTo(1.25, 6);
    // Both in `rem`, because both are lengths measured from text — a px reservation stops covering
    // its own labels the moment the root font size moves, which is #44 twice over.
    expect(band!.style.height.endsWith('rem')).toBe(true);
    expect(reservation!.style.minWidth.endsWith('rem')).toBe(true);

    /**
     * And the labels lean the way the reservation faces.
     *
     * The reservation is on the model column, to the *left* of every label, which is only the right
     * place if the labels lean left: anchored bottom-right and turned clockwise. The first fix for
     * #64 kept them leaning right and reserved a trailing lane out of whatever free space the
     * viewport had going spare, which is not a quantity — it ran out between 948px and 1009px, and
     * the grid scrolled 50px for header text there while both browser assertions sat above the
     * window. jsdom cannot see a pixel of that, but it can see that the two halves still agree.
     */
    const label = matrix().querySelector('thead th span[title]')!;
    expect(label.className).toContain('origin-bottom-right');
    expect(label.className).toMatch(/(?:^|\s)rotate-45(?:\s|$)/);
  });

  it('reserves a band for the labels it renders, not for the ones it used to', () => {
    render(<App />);

    const bandRem = parseFloat(
      matrix().querySelector<HTMLElement>('thead th:nth-child(2)')!.style.height
    );
    const longest = Math.max(...headerLabels().map((l) => l.label.length));

    // Still long enough for the longest label at the same 0.5rem-per-character estimate the
    // rotation has always been sized by — the band may only shrink because the labels did, never
    // because someone capped it. Clipping the names is the failure the rotation exists to prevent.
    expect(bandRem).toBeGreaterThanOrEqual(longest * 0.5 * Math.SQRT1_2);

    /**
     * And the 246px in the issue has actually moved.
     *
     * 15.39rem was the band when the reservation carried `(12-ch DDR5-4800)` and its neighbours:
     * 40 characters, 246px at the default root, 16% of the Matrix panel on a phone, and unchanged
     * between a 320px screen and a 1440px one. 11rem is 176px — comfortably above the 10.09rem the
     * shipped catalog now asks for, and far enough below 15.39 that restoring the parentheticals
     * fails here rather than in review.
     */
    expect(bandRem).toBeLessThan(11);
  });
});

describe('clicking a Matrix cell loads what that cell was scored under', () => {
  it('carries the quantization the cell was evaluated at, not the one selected', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);

    // Default is MXFP4, which the Matrix substitutes for dense rows — so the grid and the Bench
    // would otherwise disagree about the square that was just clicked.
    expect(useConfig.getState().quantId).toBe('mxfp4');

    const matrix = screen.getByRole('region', { name: /Every model on every machine/i });
    const dense = within(matrix)
      .getAllByRole('button', { name: /Qwen3 32B on / })
      .at(0)!;
    await user.click(dense);

    const after = useConfig.getState();
    expect(after.modelId).toBe('Qwen/Qwen3-32B');
    expect(after.quantId).not.toBe('mxfp4');
    expect(after.deviceCount).toBe(1);
  });
});

/**
 * The ring has no visual equivalent for a screen reader, so its sentence has to carry everything
 * the table carries about that one cell — the same wording and the same disambiguated label.
 */
describe('the spoken marker describes the same cell the table does', () => {
  it('borrows the table wording rather than re-deriving it', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-512');
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const region = screen.getByRole('region', { name: /how much room/i });
    const spoken = within(region).getByRole('img').getAttribute('aria-label') ?? '';

    // Open the table and read the marked cell, which is the same scenario the ring sits on.
    await user.click(within(region).getByRole('button', { name: /region as a table/i }));
    const table = within(region).getByRole('table');
    const marked = within(table).getByText(/▸/).closest('td')?.textContent ?? '';

    // Whatever the table says about that cell, the ring's sentence must say too.
    const said = marked.replace('▸', '').trim().toLowerCase();
    expect(spoken.toLowerCase()).toContain(said);
  });
});

/**
 * The Matrix substitutes a format wherever the selected one does not apply, and that substitution
 * used to bypass the runtime check entirely — it asked `quantApplies` without the runtime and then
 * returned a hardcoded Q4_K_M. Under vLLM, which reads no GGUF K-quant, every dense row was
 * therefore sized, coloured and ranked at a checkpoint that cannot be loaded, and clicking one
 * landed in a Bench that coerced it to something else and showed different figures.
 */
describe('the Matrix only ever scores a format the runtime can load', () => {
  it('substitutes something vLLM can read, not a GGUF K-quant', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    // MXFP4 is expert-only, so every dense row needs a stand-in.
    await user.selectOptions(screen.getByLabelText('Quantization'), 'mxfp4');

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });

    // The heading states what is standing in, and it cannot be a format vLLM does not load.
    expect(within(matrix).queryByText(/Q4_K_M where it does not apply/i)).not.toBeInTheDocument();

    // No cell may be blocked by the tool's own substitution being unloadable. That string comes
    // from `planPlacement`, which now refuses these pairs — so before the substitute learned about
    // the runtime, this is exactly what the grid filled up with.
    const unloadable = within(matrix)
      .getAllByRole('button')
      .filter((b) => /cannot load/i.test(b.getAttribute('aria-label') ?? ''));
    expect(unloadable).toHaveLength(0);
  });

  it('still prefers the 4-bit stand-in where the runtime does read it', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'mxfp4');

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    // llama.cpp loads GGUF, so the honest local trade is still the substitute — the fix must not
    // have demoted every grid to BF16 on its way to being correct for vLLM.
    expect(within(matrix).getByText(/Q4_K_M where it does not apply/i)).toBeInTheDocument();
  });
});

/**
 * A cell that already matches the selection changes nothing the Matrix renders, so before the
 * selected square was marked, clicking one was indistinguishable from the click not registering.
 * The scroll that accompanies it cannot be tested here — jsdom has no `scrollIntoView` at all,
 * which is how an anchor that generates no box passed for a working one.
 */
describe('the Matrix acknowledges the cell you clicked', () => {
  it('marks the selected square, and moves the mark when the selection moves', async () => {
    const user = userEvent.setup();
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const marked = () =>
      within(matrix)
        .getAllByRole('button')
        .filter((b) => b.getAttribute('aria-current') === 'true');

    // Exactly one square is current at any time — the one the Bench above is showing.
    expect(marked()).toHaveLength(1);
    const before = marked()[0].getAttribute('aria-label');

    // Change the selection from outside the grid; the mark has to follow the store, not the click.
    // Deliberately not the Spark, which is the default rig — selecting it changes nothing, and
    // the assertion below would then pass against a mark that never moved.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    expect(marked()).toHaveLength(1);
    expect(marked()[0].getAttribute('aria-label')).not.toBe(before);
  });

  /**
   * Every cell is scored with `deviceCount: 1`. On a linked rig the mark therefore pointed at a
   * square whose capacity and speed describe a different machine from the one the Bench is
   * showing — and clicking it, the one square that ought to be a no-op, silently reset the
   * configuration to a single device.
   */
  it('marks nothing when the Bench is on a rig this grid does not score', async () => {
    const user = userEvent.setup();
    render(<App />);

    const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });
    const marked = () =>
      within(matrix())
        .getAllByRole('button')
        .filter((b) => b.getAttribute('aria-current') === 'true');

    // A device with an interconnect, so the count is offered rather than clamped back to 1.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    expect(marked()).toHaveLength(1);

    act(() => useConfig.getState().set('deviceCount', 4));
    expect(marked()).toHaveLength(0);

    // And the grid says why, rather than leaving the reader to notice the mark has gone.
    expect(within(matrix()).getByRole('heading', { level: 2 })).toHaveTextContent(
      /one device per cell/i
    );

    // Back to a rig the grid does score, and the mark returns.
    act(() => useConfig.getState().set('deviceCount', 1));
    expect(marked()).toHaveLength(1);
  });
});

/**
 * `kvPrecision` is an internal width, not a name anyone types. vLLM has no integer-Q8 cache — the
 * catalog maps that value to FP8 for exactly that reason — so upper-casing it in the heading
 * described a setting that does not exist, in the panel most likely to be screenshotted.
 */
describe('the Matrix names the cache the runtime actually has', () => {
  const heading = () =>
    within(screen.getByRole('region', { name: /every model on every machine/i })).getByRole(
      'heading',
      { level: 2 }
    );

  it('calls the one-byte cache FP8 under vLLM, as the Bench control does', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    expect(heading()).toHaveTextContent(/FP8 KV/);
    expect(heading()).not.toHaveTextContent(/Q8 KV/);
  });

  // llama.cpp keeps the table's own name, which is the fallback path — worth its own case so the
  // fix cannot be mistaken for "always print FP8". (Its real flag is `q8_0`; naming the width
  // rather than the flag is a milder version of the same gap, and is filed separately rather than
  // grown into this change.)
  it('leaves llama.cpp’s Q8 alone, so the fallback path is exercised too', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    expect(heading()).toHaveTextContent(/Q8 KV/);
  });
});

/**
 * The Matrix is 408 cells at the catalog #52 was measured against — 714 today — each a `<button>`
 * carrying a full-sentence `aria-label`, and it sat above the Usage controls in DOM order. Every one
 * of those cells in the tab sequence put 422 Tab presses between the top of the page and the context
 * slider that drives every figure on the page, and a screen-reader user heard 408 sentences on the
 * way. #66 has since moved those controls above the grid, which does not retire the pattern: the grid
 * is the page's last tab stop, so the 714 presses it used to cost are now the price of leaving the
 * document rather than of reaching the next panel. One press either way, and only with the roving
 * index.
 *
 * The counting lives here rather than in `e2e/` because the tab *sequence* is a DOM property —
 * `tabindex="-1"` is reachable by script and never by Tab — and jsdom can answer it in a second.
 * What jsdom cannot answer is whether pressing Tab actually lands where the sequence says, since
 * it implements no sequential focus navigation at all; that assertion is in `e2e/matrix-grid.spec.ts`.
 */
describe('the comparison grid is one tab stop, not four hundred', () => {
  // #52's defect is a property of the real grid: 408 cells was the measurement, and a roving
  // index over a dozen would satisfy every assertion below while the page went back to one tab
  // stop per cell on the shipped catalog. Whole-describe rather than per-test for the same
  // reason — Home/End and the five-row page step mean nothing on a grid three rows tall.
  beforeEach(atFullGrid);
  const grid = (container: HTMLElement) =>
    container.querySelector<HTMLTableElement>('table[role="grid"]')!;
  const cellsOf = (container: HTMLElement) => [
    ...grid(container).querySelectorAll<HTMLButtonElement>('td button'),
  ];

  it('offers exactly one of its cells to Tab', () => {
    const { container } = render(<App />);
    const cells = cellsOf(container);

    // The grid really is the size the issue describes, so a fix that emptied it would not pass.
    expect(cells.length).toBeGreaterThan(300);
    expect(cells.filter((c) => c.tabIndex === 0)).toHaveLength(1);
    expect(cells.filter((c) => c.tabIndex === -1)).toHaveLength(cells.length - 1);
  });

  /**
   * The whole page, not the walk to one panel — which is what this counted until #66.
   *
   * It measured the index of the first Usage control, because the Usage panel was the one *after* the
   * grid: 422 before the roving index, 19 after it. #66 moved those controls to the top of the page,
   * where their index is 6 whatever the grid does with its 714 cells, so the original assertion would
   * have passed against a grid that had never been fixed. The total is the property #52 actually
   * bought, and it is indifferent to where any panel sits.
   */
  it('keeps the whole page inside forty tab stops', () => {
    const { container } = render(<App />);
    const stops = [...container.querySelectorAll<HTMLElement>(TABBABLE)];

    // The grid has to be in this page, or the count is of a page without the problem on it.
    expect(cellsOf(container).length).toBeGreaterThan(300);
    /* 41 as it stands, one of which is the grid. 1,510 if every cell were in the sequence again —
       41 − 1 + 1,470 — which is what replacing the roving `tabIndex` with `tabIndex={0}` reports.
       The subtrahend is the *shipping* device count times the model count, which is what this grid
       renders; it read 714 until #77 doubled the model list, and a counterfactual quoting the wrong
       grid is a wrong expected value for whoever reinjects the defect.

       **The bound moves from 40 to 55, and that is the second raise, so it is argued rather than
       nudged.** It read 26 before the v2 features and 41 after them: #138's recommendation panel is
       four stops (a workload picker and three shortlist rows), #136's launch panel is nine for its
       three launchers — a copy button, a provenance link and the command block, which is a scroll
       container Chrome makes focusable so a keyboard reader can scroll it — and #139's calibration
       panel is two.

       The old comment predicted this exact failure and argued for the loose bound because of it:
       "the next disclosure would have failed a test named after the grid while nothing about the
       grid had changed". That is what happened, so the prediction is the reason to trust the raise
       rather than to distrust it.

       **What the bound is for, stated so the next person does not have to infer it:** the defect is
       1,506, not 41. Forty was an order-of-magnitude line drawn against that, never a budget. What
       it catches is a whole *collection* re-entering the sequence — a grid, a table, a list of
       cells — and 55 catches that exactly as well as 40 did, because the failure mode is three
       orders of magnitude away and not one panel away.

       **The pressure underneath it is real and is not a test problem.** Four panels landed on one
       page in one pass, and a page that keeps adding panels is the thing to look at before this
       number is raised a third time. Shedding stops was considered and declined: the copy buttons
       could be one shared button at the cost of the reader knowing which command they copied, and
       the provenance links are the "flags drift" trap #136 names. The command block's stop is
       Chrome's own and not this app's to remove. */
    expect(stops.length).toBeLessThan(55);
  });

  it('moves between cells with the arrow keys', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);
    const columns = grid(container).querySelectorAll('tbody tr')[0].querySelectorAll('td').length;

    cells[0].focus();
    expect(document.activeElement).toBe(cells[0]);

    await user.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(cells[1]);

    await user.keyboard('{ArrowDown}');
    expect(document.activeElement).toBe(cells[columns + 1]);

    await user.keyboard('{ArrowLeft}');
    expect(document.activeElement).toBe(cells[columns]);

    await user.keyboard('{ArrowUp}');
    expect(document.activeElement).toBe(cells[0]);
  });

  it('carries the tab stop with the reader rather than resetting it', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{ArrowRight}{ArrowDown}');

    // Returning to the grid returns to where they were, which is the point of a roving index.
    const moved = document.activeElement as HTMLButtonElement;
    expect(moved.tabIndex).toBe(0);
    expect(cells.filter((c) => c.tabIndex === 0)).toEqual([moved]);
  });

  it('stops at the edges rather than wrapping into another row', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{ArrowLeft}{ArrowUp}');
    // A wrap here would move the reader to the far end of the grid for a keypress that should do
    // nothing at all — and the event must stay unhandled so the page can still scroll.
    expect(document.activeElement).toBe(cells[0]);
  });

  it('jumps to the ends of a row, and of the grid', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const rows = grid(container).querySelectorAll('tbody tr');
    const columns = rows[0].querySelectorAll('td').length;
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{End}');
    expect(document.activeElement).toBe(cells[columns - 1]);

    await user.keyboard('{Home}');
    expect(document.activeElement).toBe(cells[0]);

    await user.keyboard('{Control>}{End}{/Control}');
    expect(document.activeElement).toBe(cells[cells.length - 1]);

    await user.keyboard('{Control>}{Home}{/Control}');
    expect(document.activeElement).toBe(cells[0]);
  });

  it('still loads a cell into the Bench from the keyboard', async () => {
    // The navigation must not have cost the grid its actual purpose.
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{ArrowDown}{ArrowRight}');
    const target = document.activeElement as HTMLButtonElement;
    const label = target.getAttribute('aria-label')!;

    await user.keyboard('{Enter}');
    await waitFor(() => {
      const config = useConfig.getState();
      expect(label).toContain(getModel(config.modelId).name);
    });
  });

  it('tells a screen reader how to drive it', () => {
    const { container } = render(<App />);
    const caption = grid(container).querySelector('caption')!;

    expect(caption.textContent).toMatch(/single tab stop/i);
    expect(caption.textContent).toMatch(/arrow keys/i);
  });
});

/**
 * A mark drawn *on* the heatmap, measured against the heatmap.
 *
 * The rule above — focus and a resting state never share a channel — moved the Matrix's
 * selected-square mark from an offset ring outside the cell to a frame inside it, and that moved it
 * off `--color-surface`, where `tokens.ts` validated the accent at 7.14:1, and onto the ramp, where
 * it was never validated at all. A single-tone accent frame measures **1.06:1 to 4.52:1** across
 * the seven steps of `sequential` — below the 3:1 non-text minimum on 304 of the grid's 408 squares,
 * the default selection among them. So the mark is two tones, and this is the arithmetic that says
 * so: for every fill the grid actually paints, at least one of the mark's tones has to clear 3:1
 * against it.
 *
 * **jsdom can answer this one, which is why it is here rather than in the browser suite.** The fill
 * is an inline style and the mark's tones are token names in the class list; the rest is the WCAG
 * contrast formula. What jsdom cannot answer — whether the two tones land in the geometry the class
 * list implies, 2px of accent with the separator inside it — is `e2e/focus-indicators.spec.ts`.
 */
describe('a mark drawn on the heatmap stays visible on every step of the ramp', () => {
  /** SC 1.4.11's floor for a non-text mark. */
  const MINIMUM_CONTRAST = 3;

  const parseColour = (value: string): [number, number, number] => {
    if (value.startsWith('#')) {
      const hex = value.slice(1);
      return [0, 2, 4].map((i) => Number.parseInt(hex.slice(i, i + 2), 16)) as [
        number,
        number,
        number,
      ];
    }
    const parts = value.match(/[\d.]+/g)?.map(Number) ?? [];
    return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
  };

  const luminance = (value: string) => {
    const [r, g, b] = parseColour(value)
      .map((c) => c / 255)
      .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const contrast = (a: string, b: string) => {
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high + 0.05) / (low + 0.05);
  };

  /** `--color-surface-raised` -> `surfaceRaised`, so a token name resolves against `colors`. */
  const token = (cssName: string) =>
    cssName.replace('--color-', '').replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());

  /**
   * The tones a resting mark is drawn in, read off the element that wears it.
   *
   * Only the utilities that paint inside the cell count: the inset ring and the inset shadow
   * separator beneath it. The `focus:`-gated ring is drawn *outside* the box, over the panel
   * surface, so it is measured in the browser suite against a different colour entirely.
   */
  const restingTones = (el: Element) =>
    (el.getAttribute('class') ?? '')
      .split(/\s+/)
      .filter((u) => !u.includes('focus') && /^(inset-ring|inset-shadow|shadow)-\[/.test(u))
      .flatMap((u) => [...u.matchAll(/--color-[a-z-]+/g)].map((m) => m[0]))
      .map(token)
      .filter((name): name is keyof typeof colors => name in colors)
      .map((name) => colors[name]);

  it('keeps one of the selected square’s tones 3:1 against every fill it can land on', () => {
    atFullGrid();
    render(<App />);
    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const cells = within(matrix)
      .getAllByRole('button')
      .filter((button) => button.closest('td'));

    const marked = cells.filter((cell) => cell.getAttribute('aria-current') === 'true');
    expect(marked, 'no square is marked, so there is no mark to measure').toHaveLength(1);
    const tones = restingTones(marked[0]);

    // Vacuity guards. A mark with no tones, or a grid the selector stopped matching, would
    // otherwise report a clean sweep over nothing.
    expect(
      tones.length,
      'the selected square declares no mark inside the cell — if selection moved back outside it, ' +
        'the channel-collision rule above is what has to hold instead of this one'
    ).toBeGreaterThan(0);
    expect(cells.length, 'the grid is not rendering').toBeGreaterThan(300);

    /** Every fill the grid paints, with how many squares wear it. `transparent` shows the panel. */
    const fills = new Map<string, number>();
    for (const cell of cells) {
      const background = (cell as HTMLElement).style.background;
      const behind = background === 'transparent' || !background ? colors.surface : background;
      fills.set(behind, (fills.get(behind) ?? 0) + 1);
    }
    expect(
      fills.size,
      'the grid paints one colour, so the ramp is not being exercised'
    ).toBeGreaterThan(4);

    const best = (fill: string) => Math.max(...tones.map((tone) => contrast(tone, fill)));
    const unreadable = [...fills.entries()]
      .filter(([fill]) => best(fill) < MINIMUM_CONTRAST)
      .map(([fill, count]) => `${fill} (${count} squares) at ${best(fill).toFixed(2)}:1`);
    expect(unreadable, `the mark below ${MINIMUM_CONTRAST}:1 on a fill the grid paints`).toEqual(
      []
    );

    /**
     * And that the second tone is load-bearing rather than belt-and-braces. If every step of the
     * ramp were readable under one tone, this test would pass on the single-tone frame that shipped
     * 304 unreadable squares — the ramp is what makes it fail, so the ramp has to still be there.
     */
    const defeatsOneTone = tones.filter(
      (tone) => ![...fills.keys()].every((fill) => contrast(tone, fill) >= MINIMUM_CONTRAST)
    );
    expect(
      defeatsOneTone.length,
      'every tone clears the bar alone, so this measures nothing the ramp can break'
    ).toBeGreaterThan(0);
  });
});
