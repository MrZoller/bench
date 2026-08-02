import { expect, test, type Page } from '@playwright/test';
import { colors } from '@/design/tokens';

/**
 * Which control has focus, measured on a painted focus ring — issue #67.
 *
 * The four primary selects removed the outline and replaced it with a 1px border colour change
 * measuring **1.95:1 against the unfocused edge**, where WCAG 2.2 SC 2.4.13 asks for 3:1 at a 2px
 * minimum thickness. In a screenshot of the focused Model select you could not tell which control
 * had focus, and a deuteranope or a protanope loses most of what separates violet from slate-blue at
 * that size. They were the only control class in the app with a degraded indicator, and they are the
 * two inputs — model and hardware — that everything else on the page derives from.
 *
 * **jsdom cannot answer this.** It has no Tailwind cascade, no layout and no painted outline, so
 * `getComputedStyle` there reports nothing about either the thickness or the colour. What it *can*
 * answer is which indicator each control declares, which is a class-list property, and that half
 * lives in `App.test.tsx` — where it sweeps all 400-odd focusable elements in a second. The split is
 * the same one #52 made for the tab sequence: the declaration in Vitest, the behaviour here.
 *
 * **Focus is moved with Tab rather than with `focus()`.** The UA's own ring is `:focus-visible`-gated,
 * and a scripted focus does not reliably satisfy that heuristic — a sweep driven by `focus()` would
 * report every slider and every button on the page as painting nothing, and would be switched off
 * within a week for crying wolf. Tabbing is also what the criterion is about.
 */

/** SC 2.4.13's minimum thickness, in px. */
const MINIMUM_THICKNESS = 2;

/**
 * SC 1.4.11's floor for a non-text mark, which is also the change contrast 2.4.13 asks for. The
 * unfixed border swap measured 1.95:1 against the edge it replaced.
 */
const MINIMUM_CONTRAST = 3;

/** Long enough to wrap all the way round the page; asserted to be more than it takes. */
const MAX_STOPS = 80;

/** The attribute stamped on every control so the resting pass and the tab walk can be matched up. */
const PROBE = 'data-focus-probe';

/** The computed properties a focus indicator can live in, plus what it is drawn against. */
interface IndicatorStyle {
  outlineStyle: string;
  outlineWidth: string;
  outlineColor: string;
  boxShadow: string;
  borderColor: string;
  backgroundColor: string;
  /** The first opaque background *behind* the element — see `readIndicator`. */
  behind: string;
}

declare global {
  interface Window {
    /** Installed by the init script below. */
    readIndicator(el: Element): IndicatorStyle;
  }
}

test.beforeEach(async ({ page }) => {
  /**
   * One reader, installed before load, used by both passes.
   *
   * Two copies of this is how one of them ends up measuring a different box from the other — and the
   * box is the part that is easy to get wrong, because the element that *has* focus is not always the
   * element that *shows* it.
   */
  await page.addInitScript(() => {
    window.readIndicator = (el: Element): IndicatorStyle => {
      /**
       * A segmented control's radio is `sr-only` — 1x1, clipped, and focusable on purpose — and the
       * mark belongs to the label around it. `touch-targets.spec.ts` resolves the same way for the
       * same reason: measuring the input measures the wrong box.
       */
      const box = el.getBoundingClientRect();
      const host = box.width < 2 || box.height < 2 ? (el.closest('label') ?? el) : el;
      const style = getComputedStyle(host);

      /**
       * What a mark *outside* the box is drawn against: the nearest opaque ancestor, not the
       * control's own fill. An outline at a 2px offset sits outside the control entirely, and the
       * grid's focus ring sits in the 2px gap between two cells, so the element's own background is
       * a colour those marks never touch.
       *
       * An `inset` layer is the other case and is measured against `backgroundColor` below — the
       * Matrix's selected square is marked inside its own box, over a ramp fill that runs from
       * `#cde2fb` to `#0d366b`, and scoring that against the panel surface behind it reports 7.14:1
       * for a mark that can paint at 1.06:1. This reader returns both; `painted` picks per layer.
       */
      let behind = getComputedStyle(document.body).backgroundColor;
      for (let parent = host.parentElement; parent; parent = parent.parentElement) {
        const bg = getComputedStyle(parent).backgroundColor;
        if (bg && bg !== 'transparent' && !/,\s*0\)$/.test(bg)) {
          behind = bg;
          break;
        }
      }

      return {
        outlineStyle: style.outlineStyle,
        outlineWidth: style.outlineWidth,
        outlineColor: style.outlineColor,
        boxShadow: style.boxShadow,
        borderColor: style.borderColor,
        backgroundColor: style.backgroundColor,
        behind,
      };
    };
  });
  await page.goto('/');
});

/** Every element with a focus state, stamped, named, and read in its resting state. */
async function stampControls(page: Page) {
  return page.evaluate((probe) => {
    const controls = Array.from(
      document.querySelectorAll<HTMLElement>('a[href], button, input, select, textarea, [tabindex]')
    );
    return controls.map((el, index) => {
      el.setAttribute(probe, String(index));
      const label =
        (el as HTMLSelectElement).labels?.[0]?.textContent?.trim() ||
        el.getAttribute('aria-label') ||
        (el.textContent ?? '').trim() ||
        '(no name)';
      return {
        index,
        name: `<${el.tagName.toLowerCase()}> "${label.slice(0, 40)}"`,
        resting: window.readIndicator(el),
      };
    });
  }, PROBE);
}

/**
 * Walk the tab sequence, reading each stop as the keyboard reaches it.
 *
 * Stops at the wrap rather than at a fixed count, and reports an unstamped stop instead of stopping
 * there — a control the selector above does not match is a hole in the sweep, not the end of it.
 */
async function tabWalk(page: Page) {
  const stops: { index: number; focused: IndicatorStyle }[] = [];
  const unstamped: string[] = [];
  const seen = new Set<number>();

  for (let i = 0; i < MAX_STOPS; i += 1) {
    await page.keyboard.press('Tab');
    const stop = await page.evaluate(
      (probe): { escaped: boolean; index?: number; tag?: string; focused?: IndicatorStyle } => {
        const el = document.activeElement;
        if (!el || el === document.body || el === document.documentElement)
          return { escaped: true };
        const raw = el.getAttribute(probe);
        if (raw === null) return { escaped: false, tag: el.tagName.toLowerCase() };
        return { escaped: false, index: Number(raw), focused: window.readIndicator(el) };
      },
      PROBE
    );

    if (stop.escaped) break;
    if (stop.index === undefined) {
      unstamped.push(stop.tag ?? '(unknown)');
      continue;
    }
    if (seen.has(stop.index)) break;
    seen.add(stop.index);
    stops.push({ index: stop.index, focused: stop.focused! });
  }

  return { stops, unstamped };
}

/** One `box-shadow` layer as Chromium serialises it: `rgb(192, 132, 252) 0px 0px 0px 2px`. */
interface Shadow {
  raw: string;
  color: string;
  spread: number;
  /** An `inset` layer is painted over the element's own fill, not over what is behind it. */
  inset: boolean;
}

function parseShadows(value: string): Shadow[] {
  if (!value || value === 'none') return [];

  // Split on the commas *between* layers. A naive split also cuts `rgba(192, 132, 252, 0.5)` into
  // pieces, and every layer then parses as a colourless zero-spread shadow.
  const layers: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < value.length; i += 1) {
    if (value[i] === '(') depth += 1;
    else if (value[i] === ')') depth -= 1;
    else if (value[i] === ',' && depth === 0) {
      layers.push(value.slice(start, i).trim());
      start = i + 1;
    }
  }
  layers.push(value.slice(start).trim());

  return layers.map((raw) => {
    const color = /rgba?\([^)]*\)/.exec(raw)?.[0] ?? 'rgb(0, 0, 0)';
    const lengths = raw.replace(/rgba?\([^)]*\)/, '').match(/-?[\d.]+px/g) ?? [];
    return {
      raw,
      color,
      spread: lengths.length >= 4 ? Number.parseFloat(lengths[3]) : 0,
      inset: /\binset\b/.test(raw),
    };
  });
}

/** `rgb(...)`/`rgba(...)` to a tuple, defaulting alpha to 1. */
function parseColor(value: string): [number, number, number, number] {
  const parts = value.match(/-?[\d.]+/g)?.map(Number) ?? [0, 0, 0, 0];
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0, parts[3] ?? 1];
}

/** WCAG relative luminance, with a translucent mark composited over what is behind it. */
function luminance(value: string, behind: string): number {
  const [r, g, b, a] = parseColor(value);
  const [br, bg, bb] = parseColor(behind);
  const channel = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  const mix = (fg: number, bgc: number) => channel(fg * a + bgc * (1 - a));
  return 0.2126 * mix(r, br) + 0.7152 * mix(g, bg) + 0.0722 * mix(b, bb);
}

function contrast(mark: string, behind: string): number {
  const a = luminance(mark, behind);
  const b = luminance(behind, behind);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

/**
 * What focusing actually painted.
 *
 * `ua` is the browser's own ring and is deliberately not measured: Chromium computes
 * `outline-style: auto` with a reported width of 1px for a dual-tone ring it paints at 2px, so the
 * thickness check cannot see it and would fail every control that correctly leaves the ring alone.
 * The specs' teeth are on controls that declare their own indicator, which is where #67's defect and
 * both of its siblings lived.
 *
 * The thickness credited to a ring is the shadow's spread, which over-credits a ring drawn at an
 * offset — Tailwind folds the offset into the spread. The declared width is pinned exactly, in
 * `App.test.tsx`; what this number has to establish is that a mark of roughly the right size appeared
 * where there was none.
 *
 * Each mark carries the colour it is drawn `against`, because that differs per layer rather than per
 * element: an outline and an outset ring cover what is behind the box, an `inset` layer covers the
 * element's own fill. Measuring every mark against the ancestor is how an inset indicator on a
 * heatmap cell scores 7.14:1 against the panel while painting at 1.06:1 over the ramp.
 */
type Painted =
  | { kind: 'ua' }
  | { kind: 'none' }
  | { kind: 'colour-only'; detail: string }
  | { kind: 'declared'; thickness: number; color: string; against: string };

/** Whether a computed colour is `transparent` or fully transparent `rgba(…, 0)`. */
const invisible = (color: string) => !color || color === 'transparent' || /,\s*0\)$/.test(color);

/**
 * The colour a mark covers: its own fill for an `inset` layer, what is behind the box otherwise.
 *
 * A cell with no fill of its own shows the surface behind it, so an inset mark there is measured
 * against the same colour an outside mark would be.
 */
const over = (style: IndicatorStyle, inset: boolean) =>
  inset && !invisible(style.backgroundColor) ? style.backgroundColor : style.behind;

function painted(resting: IndicatorStyle, focused: IndicatorStyle): Painted {
  if (focused.outlineStyle === 'auto') return { kind: 'ua' };

  const marks: { thickness: number; color: string; against: string }[] = [];
  const width = Number.parseFloat(focused.outlineWidth);
  if (focused.outlineStyle !== 'none' && focused.outlineStyle !== 'hidden' && width > 0) {
    // An outline is always drawn outside the border box, whatever the offset.
    marks.push({ thickness: width, color: focused.outlineColor, against: focused.behind });
  }

  // Only what appeared. A ring the element already wore cannot be what says it has focus — that is
  // the Matrix's selected square, which lit an identical ring on top of its own.
  const before = new Set(parseShadows(resting.boxShadow).map((s) => s.raw));
  for (const shadow of parseShadows(focused.boxShadow)) {
    if (before.has(shadow.raw) || shadow.spread <= 0) continue;
    marks.push({
      thickness: shadow.spread,
      color: shadow.color,
      against: over(focused, shadow.inset),
    });
  }

  if (marks.length === 0) {
    const swapped: string[] = [];
    if (resting.borderColor !== focused.borderColor) swapped.push('border-color');
    if (resting.backgroundColor !== focused.backgroundColor) swapped.push('background-color');
    return swapped.length > 0
      ? { kind: 'colour-only', detail: swapped.join(' and ') }
      : { kind: 'none' };
  }

  marks.sort((a, b) => b.thickness - a.thickness);
  return { kind: 'declared', ...marks[0] };
}

/** `#c084fc` as Chromium serialises it, so the spec can name the token rather than a literal. */
function rgbOf(hex: string): string {
  const h = hex.replace('#', '');
  const [r, g, b] = [0, 2, 4].map((i) => Number.parseInt(h.slice(i, i + 2), 16));
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * The named instance, control by control.
 *
 * Asserted on the four selects directly as well as in the sweep below, because these are the ones the
 * issue measured and the numbers in it are the specification: an outline where there was none, 2px
 * where there was 1, and 3:1 against what it sits on where the border swap managed 1.95:1 against the
 * edge it replaced.
 */
test('each of the four primary selects paints a focus ring you can see', async ({ page }) => {
  const controls = await stampControls(page);

  for (const label of ['Model', 'Hardware', 'Quantization', 'Runtime']) {
    // `exact`, because the default is a case-insensitive substring: "Model" also matches the
    // Matrix's own accessible name, "Every model on every machine", and the locator then resolves
    // to two elements and fails on the wrong thing.
    const select = page.getByLabel(label, { exact: true });
    await expect(select, `no select labelled ${label}`).toHaveCount(1);

    const resting = controls.find((c) => c.name === `<select> "${label}"`)?.resting;
    expect(resting, `${label} was not stamped`).toBeDefined();
    // Nothing before focus: the outline is the indicator, so it must not be part of the resting look.
    expect(resting!.outlineStyle, `${label} draws an outline unfocused`).toBe('none');

    await select.focus();
    const focused = await select.evaluate((el) => window.readIndicator(el));

    expect(focused.outlineStyle, `${label} paints no outline on focus`).not.toBe('none');
    expect(
      Number.parseFloat(focused.outlineWidth),
      `${label}'s outline thickness`
    ).toBeGreaterThanOrEqual(MINIMUM_THICKNESS);
    expect(focused.outlineColor, `${label}'s outline colour`).toBe(rgbOf(colors.accent));
    expect(
      contrast(focused.outlineColor, focused.behind),
      `${label}'s outline against ${focused.behind}`
    ).toBeGreaterThanOrEqual(MINIMUM_CONTRAST);
  }
});

/**
 * The sweep, over every stop in the tab sequence.
 *
 * It sweeps rather than names for the reason `touch-targets.spec.ts` does: the issue named the four
 * selects and two more instances of the same defect were live elsewhere on the page. A control added
 * later is covered by default instead of by remembering to add it here.
 */
test('every tab stop paints an indicator that clears 2px and 3:1', async ({ page }) => {
  const controls = await stampControls(page);
  const { stops, unstamped } = await tabWalk(page);

  /**
   * Vacuity guards first. A walk that ended after two presses, or one that never reached the
   * controls, makes every assertion below a statement about nothing — the failure mode this suite has
   * shipped three of.
   */
  expect(controls.length, 'nothing was stamped').toBeGreaterThan(300);
  expect(stops.length, 'the tab walk found almost nothing').toBeGreaterThan(15);
  expect(stops.length, 'the walk never wrapped, so it is partial').toBeLessThan(MAX_STOPS);
  expect(unstamped, 'a focusable control the sweep cannot see').toEqual([]);
  /**
   * The four primary selects, **named rather than counted**.
   *
   * This asserted `=== 4` and failed the moment #138 added a fifth select — the recommendation
   * panel's workload picker — reporting "the four primary selects are not in the tab sequence"
   * about a page where all four were. A count is a proxy for the claim in its own message, and it
   * goes wrong in the direction that reads as a real defect: the guard exists to prove the walk
   * *reached* Setup, and another select elsewhere on the page is neither evidence for nor against
   * that. Naming them is strictly stronger — it would also catch a walk that found five selects and
   * none of these — and it is what this file's own sibling test already does one block up.
   */
  const selects = stops
    .filter((s) => controls[s.index].name.startsWith('<select>'))
    .map((s) => controls[s.index].name);
  for (const label of ['Model', 'Hardware', 'Quantization', 'Runtime']) {
    expect(selects, `${label} is not in the tab sequence`).toContain(`<select> "${label}"`);
  }

  const named = (index: number) => controls[index].name;
  /** What was computed, so a failure here can be read without re-running the browser. */
  const evidence = (style: IndicatorStyle) =>
    `[outline ${style.outlineStyle} ${style.outlineWidth} ${style.outlineColor}; ` +
    `shadow ${style.boxShadow.slice(0, 120)}]`;
  const failures: string[] = [];
  let declared = 0;
  let ua = 0;

  for (const stop of stops) {
    const mark = painted(controls[stop.index].resting, stop.focused);
    if (mark.kind === 'ua') {
      ua += 1;
      continue;
    }
    if (mark.kind === 'none') {
      failures.push(
        `${named(stop.index)} paints nothing at all on focus ${evidence(stop.focused)}`
      );
      continue;
    }
    if (mark.kind === 'colour-only') {
      failures.push(
        `${named(stop.index)} changes only its ${mark.detail} on focus ${evidence(stop.focused)}`
      );
      continue;
    }

    declared += 1;
    if (mark.thickness < MINIMUM_THICKNESS) {
      failures.push(`${named(stop.index)} paints ${mark.thickness}px ${evidence(stop.focused)}`);
    }
    const ratio = contrast(mark.color, mark.against);
    if (ratio < MINIMUM_CONTRAST) {
      failures.push(
        `${named(stop.index)} paints ${mark.color} at ${ratio.toFixed(2)}:1 against ` +
          `${mark.against}`
      );
    }
  }

  // Both categories have to be populated, or the loop above is only exercising one branch.
  expect(declared, 'no control paints an indicator of its own').toBeGreaterThan(5);
  expect(
    ua,
    'nothing relies on the browser’s own ring, so that exemption is stale'
  ).toBeGreaterThan(0);
  expect(failures, 'focus indicators below the 2px / 3:1 bar').toEqual([]);
});

/**
 * The second instance the sweep above cannot reach: the Matrix's *marked* square.
 *
 * Selection was an accent ring at a 1px offset and focus is an accent ring — the same channel, width
 * and colour — so focusing the marked square changed nothing whatsoever. That is a 1:1 change
 * contrast, #67's 1.95:1 border in its most extreme form, and it is not a corner case: clicking a
 * cell makes it both the selection and the roving tab stop, so the marked square is exactly where Tab
 * lands when a reader comes back to the grid. The sweep walks the pristine page, where the tab stop
 * is the top-left cell and the mark is somewhere else entirely.
 */
test('the marked square still says when the keyboard is on it', async ({ page }) => {
  /**
   * Pinned by position rather than by `:not([aria-current])`, which is the tempting way to find a
   * square that is not already the selection — and it stops matching the moment the click lands, so
   * every assertion after it would silently be about a different cell. The grid's row and column
   * order does not depend on the selection, so an index is stable across the click.
   */
  const cell = page.locator('table[role="grid"] td button').nth(30);
  await expect(cell).toHaveCount(1);
  await expect(cell, 'cell 30 is already the selection').not.toHaveAttribute(
    'aria-current',
    'true'
  );
  await cell.click();

  // The click is what makes this square both the selection and the grid's single tab stop.
  await expect(cell).toHaveAttribute('aria-current', 'true');
  await expect(cell).toHaveAttribute('tabindex', '0');

  // Focus off the square to read the mark it wears at rest, then Tab straight back onto it.
  await page.keyboard.press('Shift+Tab');
  await expect(cell).not.toBeFocused();
  const resting = await cell.evaluate((el) => window.readIndicator(el));

  await page.keyboard.press('Tab');
  await expect(cell).toBeFocused();
  const focused = await cell.evaluate((el) => window.readIndicator(el));

  // The mark it already wore is the precondition — without it the assertion below would pass on a
  // square that is simply unmarked.
  const selection = parseShadows(resting.boxShadow).filter((s) => s.spread > 0);
  expect(selection.length, 'the square is not marked').toBeGreaterThan(0);

  /**
   * And the selection mark has to be readable in its own right, measured against the cell's own
   * ramp fill rather than against the panel behind it.
   *
   * This is the assertion that was missing while the mark was a single accent frame inside the
   * cell: `--color-accent` is validated at 7.14:1 on `--color-surface` and measures 1.06:1 to
   * 4.52:1 on the seven steps of the ramp, so the marked square was unreadable on 304 of 408 cells
   * with every other assertion in this file still green. Two tones is the fix, and the property
   * that makes it one is exactly this — *one of them* clears 3:1 against whatever the cell is
   * painted. `App.test.tsx` does the same arithmetic over every fill the grid paints; this checks
   * that the tones the class list promises are the tones the browser actually painted.
   */
  const fill = over(resting, true);
  const readable = selection.map((layer) => contrast(layer.color, fill));
  expect(
    Math.max(...readable),
    `the selection mark on ${fill}: ${selection
      .map((l, i) => `${l.color} at ${readable[i].toFixed(2)}:1`)
      .join(', ')}`
  ).toBeGreaterThanOrEqual(MINIMUM_CONTRAST);

  const mark = painted(resting, focused);
  expect(mark.kind, `focusing the marked square painted ${mark.kind}`).toBe('declared');
  if (mark.kind !== 'declared') return;
  expect(mark.thickness, 'the focus ring on the marked square').toBeGreaterThanOrEqual(
    MINIMUM_THICKNESS
  );
  expect(
    contrast(mark.color, mark.against),
    `the focus ring against ${mark.against}`
  ).toBeGreaterThanOrEqual(MINIMUM_CONTRAST);
});

/**
 * And that the sweep really covered the page.
 *
 * Half the page's content is behind a disclosure, and `touch-targets.spec.ts` opens all three before
 * measuring for exactly that reason. The walk above deliberately does not: clicking a toggle moves the
 * sequential-focus starting point, so the walk would begin in the middle of the page and report a
 * clean bill for everything above it. This is the assertion that makes that safe — if a focusable
 * control is ever added inside one of those regions, this fails and the walk has to change.
 */
test('the disclosures hide no focusable control from the walk', async ({ page }) => {
  const focusable = () =>
    page.evaluate(
      () => document.querySelectorAll('a[href], button, input, select, textarea, [tabindex]').length
    );

  const before = await focusable();
  for (const name of [
    /show figures as a table/i,
    /show the region as a table/i,
    /show what each workload means/i,
  ]) {
    const toggle = page.getByRole('button', { name });
    await expect(toggle, `nothing matched ${String(name)}`).toHaveCount(1);
    await toggle.click();
  }

  expect(await focusable(), 'a revealed region contains a control the walk never reaches').toBe(
    before
  );
});
