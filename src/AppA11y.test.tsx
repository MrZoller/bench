/**
 * Page-level accessibility — the heading outline, focus indicators, control order, and ARIA
 * reference integrity (#115). These four sweep the whole page by design, which is why they are
 * a file: they share no fixture with the panel-agreement suites in `App.test.tsx`, and a sweep
 * that renders everything is exactly what should not be a neighbour of two hundred bounded-grid
 * tests. The grid-fixture-dependent counting — the roving tab index and the page's tab-stop
 * budget — lives with the grid in `Matrix.test.tsx`, per the same criterion.
 */
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG } from '@/store/config';
import { DETAIL_ANCHOR_ID } from '@/components/Matrix';
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

/**
 * ARIA reference integrity, as a sweep rather than per instance (#131).
 *
 * Two disclosures mounted their region only while expanded, so collapsed they pointed
 * `aria-controls` at an id that was not in the DOM — a reference a screen reader's "jump to
 * controlled region" cannot resolve, and an axe `aria-valid-attr-value` failure. The contract on
 * `DisclosureToggle.controls` now requires the region in both states; this sweeps every
 * `aria-controls` on the default page, which renders both offending disclosures collapsed, so a
 * new call site that unmounts its region fails here rather than in an audit.
 */
describe('every aria-controls points at a node that exists', () => {
  it('resolves each reference on the default page, collapsed states included', () => {
    render(<App />);

    const referencing = Array.from(document.querySelectorAll('[aria-controls]'));
    // The sweep must be sweeping something: both #131 instances ship collapsed by default.
    expect(referencing.length).toBeGreaterThanOrEqual(2);
    for (const el of referencing) {
      const id = el.getAttribute('aria-controls')!;
      expect(
        document.getElementById(id),
        `aria-controls="${id}" resolves to nothing`
      ).not.toBeNull();
    }
  });
});

/**
 * The page leads with the controls that drive it, which #52 explicitly left open.
 *
 * The five Usage sliders set the scenario every figure here is computed at — the memory bar, the three
 * verdict tiles, the workload strip, the Envelope, and a Matrix heading that prints the very numbers
 * they hold ("32K context, 8K prompt, 1 user, FP16 KV"). They used to render after all five: 2,260px
 * below the memory bar at 1440x900 when the issue measured it, 2,402px on `main` by the time this
 * landed — two and a half viewport heights either way — and past four screens of grid on an iPhone 14,
 * where the slider and the bar it fills were never on screen together at any scroll position. #52 took the keyboard cost from 422 Tab presses to 15 and closed by naming this as the
 * open question — "whether the Usage controls should sit above the two large grids in DOM order. They
 * are the primary input of the tool and are currently last." #66 is that question answered.
 *
 * **DOM order is the half jsdom can answer, and it is not the lesser half.** Whether the slider and
 * the bar land in one viewport is geometry, so it is `e2e/usage-placement.spec.ts` — every rect here
 * reads 0. But DOM order *is* reading order: it is the sequence a screen-reader user is handed, one
 * panel at a time, and no amount of CSS `order` changes it. Six panels of output before the first
 * input was the same defect in the channel that has no viewport at all. So the sequence is pinned
 * here, where it costs a second, and where a panel slipped in between later fails a test.
 */
describe('the controls come before the figures they drive', () => {
  /** True when `a` is announced before `b` — document order, which is reading order. */
  const precedes = (a: Element, b: Element) =>
    Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

  /**
   * Every landmark the Bench renders, in the order a reader must meet them.
   *
   * Named rather than indexed, so the assertion survives a panel being added and fails on one being
   * moved — which is the direction that matters. The two inputs first, then the four figures they
   * drive, then the two grids as the terminal panels.
   */
  const ORDER: readonly (string | RegExp)[] = [
    // "Setup" since #74: the two control panels take their names from `sr-only` headings rather than
    // from `aria-label` strings, so they are in the document outline as well as in this list.
    'Setup',
    'Usage',
    /memory budget/i,
    'Verdicts',
    /what you could do with it/i,
    /how much room is left/i,
    /every model on every machine/i,
  ];

  it('announces the two control panels before any of the six they drive', () => {
    render(<App />);
    const landmarks = ORDER.map((name) => screen.getByRole('region', { name }));

    for (let i = 1; i < landmarks.length; i++) {
      expect(
        precedes(landmarks[i - 1], landmarks[i]),
        `${String(ORDER[i])} is announced before ${String(ORDER[i - 1])}`
      ).toBe(true);
    }
  });

  /**
   * The same claim in the channel a keyboard reader travels, and a stronger form of it: the controls
   * are not merely early, they are a *prefix* of the page's tab sequence. Nothing that reports a
   * figure is reachable before every control that sets one.
   *
   * Scoped to `<main>` so the masthead's share button — a real stop, and legitimately first — is not
   * counted as a figure sitting ahead of the sliders.
   */
  it('offers every control before any figure in the tab sequence', () => {
    const { container } = render(<App />);
    const main = container.querySelector<HTMLElement>('main')!;
    // By accessible name rather than by attribute: both panels are named by an `sr-only` heading
    // since #74, so there is no `aria-label` left to select on.
    const setup = screen.getByRole('region', { name: 'Setup' });
    const usage = screen.getByRole('region', { name: 'Usage' });

    const stops = [...main.querySelectorAll<HTMLElement>(TABBABLE)];
    const controls = stops.filter((el) => setup.contains(el) || usage.contains(el));

    // Both panels are actually in the sweep — twelve elements today: four selects, the hardware
    // note's disclosure, four sliders and three KV options. A zero here would satisfy the prefix
    // trivially.
    //
    // Twelve *elements*, ten stops: a radio group offers Tab only its checked member, so the three
    // KV options are one press in a browser and three in this enumeration. That does not weaken the
    // prefix — a stop the browser skips cannot reorder the ones it does not — but it is why this
    // figure and `e2e/matrix-grid.spec.ts`'s differ by two on identical markup.
    expect(controls.length).toBeGreaterThan(8);
    expect(stops.slice(0, controls.length), 'a figure is reachable before a control').toEqual(
      controls
    );
  });

  /**
   * And the anchor a Matrix click scrolls back to still aims at the figures.
   *
   * It is the one thing in this section that had to *stay* where it was. A Matrix click rewrites the
   * model and device, so the detail it loads is the budget bar and the tiles; moving the anchor up
   * with the controls would scroll two panels of input into view and push the figures the click
   * actually changed back under the fold. #66 named this as the thing to check when moving the panel.
   */
  it('leaves the detail anchor between the controls and the figures', () => {
    const { container } = render(<App />);
    const anchor = container.querySelector(`#${DETAIL_ANCHOR_ID}`)!;
    const usage = screen.getByRole('region', { name: 'Usage' });
    const budget = screen.getByRole('region', { name: /memory budget/i });

    expect(anchor, 'the anchor is missing, so a Matrix click scrolls nowhere').not.toBeNull();
    expect(precedes(usage, anchor), 'the anchor scrolls the controls into view').toBe(true);
    expect(precedes(anchor, budget), 'the anchor sits past the detail it is meant to show').toBe(
      true
    );
  });

  /**
   * The same rule one level down, where it was already being followed and had nothing holding it.
   *
   * Both picture panels carry a measure switch that recolours the whole picture, and both put it
   * *above* the picture — which is the #66 property inside a panel rather than across the page. A
   * review of this change audited the class and found these two clean and two disclosures (the
   * workload glossary, the budget table) sitting under their output, which is the "show more under a
   * list" convention and stays: they reveal detail already in the same viewport rather than setting a
   * scenario something else is computed from. The two that recolour a whole figure are the ones worth
   * pinning, because moving one is a plausible tidy-up and nothing would have failed.
   */
  it('puts each measure switch above the picture it recolours', () => {
    const { container } = render(<App />);

    const gridSwitch = screen.getByRole('group', { name: /colour the grid by/i });
    const table = container.querySelector('table[role="grid"]')!;
    expect(precedes(gridSwitch, table), 'the Matrix recolours a grid drawn above its switch').toBe(
      true
    );

    const fieldSwitch = screen.getByRole('group', { name: /colour the field by/i });
    const canvas = screen
      .getByRole('region', { name: /how much room is left/i })
      .querySelector('canvas')!;
    expect(canvas, 'the Envelope drew no canvas, so this proves nothing').not.toBeNull();
    expect(
      precedes(fieldSwitch, canvas),
      'the Envelope recolours a field drawn above its switch'
    ).toBe(true);
  });
});

/**
 * The other way through a long page, and the one the controls were missing from entirely (#74).
 *
 * Reading order is what the suite above pins. **Heading navigation is the mechanism a screen-reader
 * user actually uses to skip around a page this tall** — 3,043px at 1440 and 4,887px on a phone — and
 * the full outline was `h1 bench`, `h2 Memory budget`, three `h3` tiles, and four more `h2`s. Both
 * control panels carried an `aria-label` and no heading at all, so all nine inputs the tool takes were
 * unreachable that way: a reader could jump to five panels of output and to none of the controls that
 * produce them. Landmark navigation did reach them, which is a second, less-used mechanism and puts
 * nothing in the outline.
 *
 * The mis-parenting is the half that is an active wrong claim rather than an absence. With the verdict
 * strip unheaded, the nearest `h2` above `h3 Capacity` was the memory budget's, so the outline said
 * capacity is a subsection of the budget — of the one panel whose whole design is that capacity, decode
 * and time-to-first-token are three independent axes that must not be collapsed. See `Telemetry.tsx`'s
 * docstring, which calls collapsing them "precisely the move that makes existing calculators give bad
 * advice".
 *
 * **Driven off every panel and every control rather than off the three sections the issue tabulates**,
 * which is what stops a panel added later reintroducing the gap — and is how the sweep found two more
 * instances: the memory panel's runtime-refusal branch, whose `<section>` had a heading and no
 * `aria-labelledby` and so was not a landmark at all, and the MoE aside, an unnamed `complementary`
 * with a perfectly good `h2` sitting inside it.
 *
 * Which of these are DOM and which are pixels splits the usual way. Three of the eight headings are
 * `sr-only`, and whether they are really invisible and really take no grid track is layout — every rect
 * here reads 0 — so it is `e2e/heading-outline.spec.ts`. Everything below is attributes and document
 * order, which jsdom answers exactly.
 */
describe('the heading outline reaches every control and mis-parents nothing', () => {
  /** Every heading the page renders, in document order, with its level and its text. */
  const outline = (container: HTMLElement) =>
    [...container.querySelectorAll<HTMLElement>('h1, h2, h3, h4, h5, h6')].map((el) => ({
      el,
      level: Number(el.tagName[1]),
      text: (el.textContent ?? '').replace(/\s+/g, ' ').trim(),
    }));

  /**
   * Every panel on the page: the `<section>`s and the MoE `<aside>`.
   *
   * Both element types are landmarks here — `region` for a named section, `complementary` for the
   * aside — and both are what a heading is supposed to be the outline entry for. Deliberately not a
   * list of the seven known panels: the point of the sweep is the eighth.
   */
  const panels = (container: HTMLElement) => [
    ...container.querySelectorAll<HTMLElement>('section, aside'),
  ];

  /** Enough of a panel to find it from a failure message. */
  const label = (panel: HTMLElement) =>
    panel.getAttribute('aria-label') ??
    panel.querySelector('h1, h2, h3, h4, h5, h6')?.textContent?.trim() ??
    `${(panel.textContent ?? '').replace(/\s+/g, ' ').trim().slice(0, 40)}…`;

  /**
   * The heading a panel's accessible name is computed from, where it has one.
   *
   * `aria-labelledby` is an IDREF *list*, so the value is split rather than handed to
   * `getElementById` whole — the same reason the `aria-describedby` resolver further down splits.
   * A panel written as `aria-labelledby="headingId subheadId"`, which is the obvious way to append a
   * subhead to a landmark name, would otherwise resolve to `null` and be reported by the sweep below
   * as having no heading at all: a red pointing at correctly-named markup instead of at the resolver.
   */
  const namingHeading = (panel: HTMLElement) =>
    (panel.getAttribute('aria-labelledby') ?? '')
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => panel.ownerDocument.getElementById(id))
      .find(
        (target): target is HTMLElement => target !== null && /^H[1-6]$/.test(target.tagName)
      ) ?? null;

  /**
   * The three ways a panel can be outside the outline, listed rather than counted.
   *
   * A per-element matcher can only report the first failure, and naming the instances nobody thought
   * of is the whole job of a sweep — the same reason the `aria-describedby` sweep below writes its
   * own resolver instead of reaching for `toHaveAccessibleDescription`.
   */
  const outsideTheOutline = (container: HTMLElement) => ({
    /** Named by a string, so the name exists for landmark navigation and nowhere else. */
    byString: panels(container)
      .filter((panel) => panel.hasAttribute('aria-label'))
      .map(label),
    /**
     * Named by something that is not a level-2 heading — which covers both a panel with no heading
     * of its own and one nested a level down. Every panel here is a sibling of every other, and that
     * is the general form of the mis-parenting: the verdict tiles read as part of the memory budget
     * precisely because their panel had no `h2` between them and the budget's.
     */
    outOfLevel: panels(container)
      .filter((panel) => namingHeading(panel)?.tagName !== 'H2')
      .map((panel) => `${label(panel)} (${namingHeading(panel)?.tagName ?? 'no heading'})`),
    /**
     * Named by a heading inside an `aria-hidden` subtree. `aria-labelledby` resolves into one, so a
     * heading marked hidden goes on naming the landmark perfectly while disappearing from the
     * outline — the half of this fix with no other symptom.
     */
    hidden: panels(container)
      .filter((panel) => namingHeading(panel)?.closest('[aria-hidden="true"]') != null)
      .map(label),
  });

  it('names every panel with a heading rather than with an aria-label string', () => {
    const { container } = render(<App />);

    // Eight today: Setup, Usage, the memory budget, the verdicts, the workloads, the Envelope, the
    // Matrix and the MoE aside. A lower bound, because the sweep must not go green by matching
    // nothing — and an exact count would fail on the next panel added rather than on the property.
    expect(panels(container).length, 'the panel sweep matched nothing').toBeGreaterThanOrEqual(7);

    const { byString, outOfLevel, hidden } = outsideTheOutline(container);
    expect(byString, 'panels named by a string, so their name is in no outline').toEqual([]);
    expect(outOfLevel, 'panels not named by a level-2 heading').toEqual([]);
    expect(hidden, 'panels named by a heading a screen reader cannot navigate to').toEqual([]);
  });

  /**
   * And in the branch that draws no bar — the memory panel's own refusal, which is a different
   * `<section>` in the same component and had no `aria-labelledby` on it at all.
   *
   * It is the branch where the loss is worst: nothing is computed, so the refusal is the only thing
   * in the panel to read, and the panel was not a landmark to arrive at.
   */
  it('names the memory panel in the branch that refuses to draw a budget', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    // vLLM cannot drive a Mac, and the ceiling and overhead band are vLLM's own numbers.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    // The other branch really is on screen, or this sweeps the drawn one a second time.
    expect(screen.queryByRole('img', { name: /allocatable used/i })).not.toBeInTheDocument();
    expect(screen.getByText(/No budget to show/i)).toBeInTheDocument();

    const { byString, outOfLevel, hidden } = outsideTheOutline(container);
    expect(byString).toEqual([]);
    expect(outOfLevel).toEqual([]);
    expect(hidden).toEqual([]);

    // Under the same name as the branch that does draw one, since it is the same panel.
    expect(screen.getByRole('region', { name: /memory budget/i })).toHaveTextContent(
      /No budget to show/i
    );
  });

  /**
   * The mis-parenting, asserted the way a reader meets it: the `h2` they last passed on the way to
   * an `h3` is the section that `h3` belongs to.
   */
  it('parents each verdict tile under Verdicts rather than under the memory budget', () => {
    const { container } = render(<App />);
    const headings = outline(container);

    for (const tile of ['Capacity', 'Decode', 'Time to first token']) {
      const index = headings.findIndex((heading) => heading.text === tile);
      expect(index, `no heading reads “${tile}”`).toBeGreaterThan(-1);
      expect(headings[index].level, `“${tile}” is not a subsection of anything`).toBe(3);

      const parent = headings
        .slice(0, index)
        .reverse()
        .find((heading) => heading.level === 2);
      expect(parent?.text, `“${tile}” is announced as part of “${parent?.text}”`).toBe('Verdicts');
    }
  });

  /**
   * Every control, not the nine the issue counts — the two measure switches are `fieldset`s in the
   * picture panels and belong to the same class, and a tenth control added to either control panel
   * has to be covered without anyone remembering to add it here.
   *
   * The predicate is what heading navigation actually gives a reader: they land on a heading, and the
   * controls they then meet before the next one are the controls that heading covers. So a control is
   * reachable when the heading immediately before it in document order is inside its own panel. Before
   * the fix that heading was the masthead's `h1 bench` for all nine inputs, which is the failure
   * stated exactly — the outline's only entry above the memory budget was the page's title.
   */
  it('puts a heading in front of every control the tool takes', () => {
    const { container } = render(<App />);
    const headings = outline(container);
    const controls = [
      ...container.querySelectorAll<HTMLElement>(
        'main select, main input[type="range"], main fieldset'
      ),
    ];

    // Eleven today: the nine the issue counts — four selects, four sliders and the KV group — plus the
    // Envelope's and the Matrix's measure switches. A lower bound, so the sweep cannot go green by
    // matching nothing and does not go red on the next control added.
    expect(controls.length, 'the control sweep matched nothing').toBeGreaterThanOrEqual(9);

    /** `Prompt length`, enough to name the offender. */
    const named = (control: HTMLElement) =>
      (control as HTMLInputElement).labels?.[0]?.textContent?.trim() ??
      control.querySelector('legend')?.textContent?.trim() ??
      `<${control.tagName.toLowerCase()}>`;

    /** The heading a reader arrives from: the last one before this control in document order. */
    const arrivedFrom = (control: HTMLElement) =>
      [...headings]
        .reverse()
        .find((heading) =>
          Boolean(heading.el.compareDocumentPosition(control) & Node.DOCUMENT_POSITION_FOLLOWING)
        );

    const unreachable = controls
      .filter((control) => {
        const heading = arrivedFrom(control);
        return heading === undefined || !heading.el.closest('section, aside')?.contains(control);
      })
      .map(named);
    expect(unreachable, 'controls that no heading in the outline leads to').toEqual([]);
  });

  /**
   * No level skipped, which is the assertion in this suite that was **green before the fix** and is
   * here anyway — said plainly, because a test that cannot fail against the defect it is filed under
   * is worth less than the line it takes and this repo has shipped three of them.
   *
   * What it guards is the fix's own failure mode rather than the bug's. Three headings that nobody
   * can see are three headings whose level nothing on screen betrays: written as `h3`, "Setup" lands
   * under the masthead's `h1` with no `h2` between them, reads as a subsection of the page title, and
   * every other assertion in this file still passes. The outline is a tree or it is decoration.
   */
  it('skips no level, so no heading claims a parent that is not there', () => {
    const { container } = render(<App />);
    const headings = outline(container);

    expect(headings[0]?.level, 'the page does not open at h1').toBe(1);
    const skips = headings
      .filter((heading, i) => i > 0 && heading.level - headings[i - 1].level > 1)
      .map((heading) => `h${heading.level} “${heading.text}”`);
    expect(skips, 'headings that jump more than one level past their predecessor').toEqual([]);
  });
});

/**
 * Which focus indicator each control declares.
 *
 * The four primary selects suppressed the outline and replaced it with a 1px border colour change
 * measuring **1.95:1 against the unfocused edge** — WCAG 2.2 SC 2.4.13 asks for 3:1 at a 2px
 * minimum thickness, and a colour-only change at that size is most of what a deuteranope loses
 * (#67). Two further instances came out of the sweep for the same shape, both listed in the issue as
 * already correct: the budget legend drew `focus:ring-1`, which is half the minimum thickness and is
 * the whole indicator once the outline is suppressed; and the Matrix marked its selected square with
 * the same channel, width and colour as its focus ring, so focusing the marked square changed
 * nothing at all — a 1:1 change contrast, and the square Tab lands on after a click.
 *
 * **The split across the two suites is deliberate, and it is the one #52 made.** Whether an
 * indicator paints 2px and clears 3:1 against what it sits on is a question about a real stylesheet
 * and a real focus ring: jsdom has no Tailwind cascade, no layout and no painted outline, so
 * `getComputedStyle` here says nothing about any of it. That half is `e2e/focus-indicators.spec.ts`.
 * What jsdom can answer is which indicator each control *declares*, which is a DOM property — and it
 * answers it for every focusable element on the page in a second, so a control added later is
 * covered by default rather than by someone remembering this file exists.
 */
describe('every control declares an indicator that says it has focus', () => {
  /** SC 2.4.13's minimum thickness, in px. */
  const MINIMUM_THICKNESS = 2;

  /** The channels an indicator can be drawn in. A border colour is deliberately not one — see below. */
  type Channel = 'outline' | 'ring' | 'inset-ring';

  /**
   * Split a utility into its variants and its base, bracket-aware.
   *
   * A naive split on `:` loses `[@media(pointer:coarse)]:h-11` — the variant contains a colon of its
   * own — and the Matrix cells carry exactly that, so the sweep would have quietly stopped reading
   * the class list of all 408 of them.
   */
  const parse = (utility: string) => {
    const parts: string[] = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < utility.length; i += 1) {
      const char = utility[i];
      if (char === '[' || char === '(') depth += 1;
      else if (char === ']' || char === ')') depth -= 1;
      else if (char === ':' && depth === 0) {
        parts.push(utility.slice(start, i));
        start = i + 1;
      }
    }
    return { variants: parts, base: utility.slice(start) };
  };

  /**
   * What one element declares.
   *
   * `restingChannels` is what is drawn while the element is *not* focused, which is what a focus
   * indicator has to be distinguishable from. A `border-*` swap is not counted in either direction:
   * treating it as an indicator is precisely the mistake #67 documents, and it stays on the select as
   * a redundant second cue.
   */
  const declared = (el: Element) => {
    const focusChannels = new Set<Channel>();
    const restingChannels = new Set<Channel>();
    const focusColours = new Set<string>();
    let thickness: number | null = null;
    let suppressesOutline = false;

    for (const utility of (el.getAttribute('class') ?? '').split(/\s+/).filter(Boolean)) {
      const { variants, base } = parse(utility);
      const match = /^(inset-ring|ring|outline)(?:-(.*))?$/.exec(base);
      if (!match) continue;

      const channel = match[1] as Channel;
      const rest = match[2] ?? '';
      const onFocus = variants.some((v) => v.startsWith('focus'));

      // `outline-none`/`outline-hidden` removes the indicator the browser supplies for free, which
      // is the state that puts an element under any obligation here at all.
      if (channel === 'outline' && (rest === 'none' || rest === 'hidden')) {
        suppressesOutline = true;
        continue;
      }
      // `ring-offset-*` and `outline-offset-*` place a mark; they never are one.
      if (rest.startsWith('offset-')) continue;

      // A bare `ring`/`outline` is 1px in Tailwind v4; anything else numeric is that many px.
      const width = rest === '' ? 1 : Number(rest);
      if (Number.isFinite(width)) {
        if (onFocus) thickness = Math.max(thickness ?? 0, width);
      } else if (onFocus) {
        focusColours.add(rest);
      }
      (onFocus ? focusChannels : restingChannels).add(channel);
    }

    return { focusChannels, restingChannels, focusColours, thickness, suppressesOutline };
  };

  /**
   * Everything with a focus state worth looking at: what Tab or a script can focus, plus anything
   * hosting a `focus-within:` indicator on behalf of a control inside it. The segmented controls are
   * the second kind — their radios are `sr-only`, so the mark belongs to the label around them, and a
   * sweep of focusable elements alone would have looked straight past it.
   */
  const controls = (container: HTMLElement) => [
    ...container.querySelectorAll<HTMLElement>(
      'a[href], button, input, select, textarea, [tabindex], [class*="focus-within:"]'
    ),
  ];

  /** `<button> "Qwen3 32B on ..."`, enough to find the offender from the failure message. */
  const name = (el: Element) =>
    `<${el.tagName.toLowerCase()}> "${
      (el.textContent ?? '').trim().slice(0, 40) ||
      el.getAttribute('aria-label')?.slice(0, 40) ||
      el.getAttribute('id') ||
      '(no text)'
    }"`;

  it('never declares an indicator thinner than the 2px minimum', () => {
    atFullGrid();
    const { container } = render(<App />);
    const summaries = controls(container).map((el) => ({ el, ...declared(el) }));
    const withIndicator = summaries.filter((s) => s.thickness !== null);

    // Vacuity guards. The grid alone declares one on every cell, and the page has to have several
    // outside it — a selector that stopped matching would otherwise report a clean sweep.
    expect(withIndicator.length, 'nothing on the page declares a focus indicator').toBeGreaterThan(
      300
    );
    expect(
      withIndicator.filter((s) => !s.el.closest('table')).length,
      'the sweep found no indicators outside the grid'
    ).toBeGreaterThan(5);

    const tooThin = withIndicator
      .filter((s) => s.thickness! < MINIMUM_THICKNESS)
      .map((s) => `${name(s.el)} declares ${s.thickness}px`);
    expect(tooThin, `thinner than ${MINIMUM_THICKNESS}px`).toEqual([]);
  });

  it('never takes the browser’s indicator away without replacing it', () => {
    const { container } = render(<App />);
    const summaries = controls(container).map((el) => ({ el, ...declared(el) }));
    const suppressing = summaries.filter((s) => s.suppressesOutline);

    expect(
      suppressing.length,
      'nothing suppresses the outline, so this asserts nothing'
    ).toBeGreaterThan(0);
    expect(
      suppressing.filter((s) => !s.el.closest('table')).length,
      'only grid cells suppress it, so the rule is narrower than it reads'
    ).toBeGreaterThan(0);

    const unreplaced = suppressing
      .filter((s) => (s.thickness ?? 0) < MINIMUM_THICKNESS)
      .map((s) => `${name(s.el)} suppresses the outline and declares ${s.thickness ?? 'nothing'}`);
    expect(unreplaced, 'outline removed with no compliant replacement').toEqual([]);
  });

  /**
   * The Matrix instance, stated as the general rule it is. An indicator that shares its channel with
   * a mark the element already wears cannot be a *change*, however thick it is — the selected square
   * wore an accent ring at rest and lit an identical accent ring on focus.
   */
  it('never draws focus in a channel a resting state already uses', () => {
    const { container } = render(<App />);
    const summaries = controls(container).map((el) => ({ el, ...declared(el) }));

    // At least one element has to wear a resting mark, or the rule is trivially satisfied. The
    // marked square is that element, and it is the one this rule was written for.
    expect(
      summaries.filter((s) => s.restingChannels.size > 0).length,
      'no element wears a resting mark, so this asserts nothing'
    ).toBeGreaterThan(0);

    const collisions = summaries
      .filter((s) => [...s.focusChannels].some((c) => s.restingChannels.has(c)))
      .map((s) => `${name(s.el)} draws focus and state both as ${[...s.focusChannels].join('/')}`);
    expect(collisions, 'focus and a resting state share a channel').toEqual([]);
  });

  /**
   * And the named instance, control by control.
   *
   * The channel is pinned as an outline rather than left free, because the reason for it is invisible
   * from Chromium: a ring is a `box-shadow`, and a native `menulist` select is painted by the
   * platform in WebKit, which does not reliably paint one. The browser spec would go on passing while
   * Safari showed nothing, so this is the assertion that holds that decision.
   */
  it('gives each of the four primary selects a 2px accent outline', () => {
    render(<App />);

    for (const label of ['Model', 'Hardware', 'Quantization', 'Runtime']) {
      const select = screen.getByLabelText(label);
      const summary = declared(select);

      expect(summary.suppressesOutline, `${label} suppresses its outline`).toBe(false);
      expect([...summary.focusChannels], `${label}'s indicator channel`).toEqual(['outline']);
      expect(summary.thickness, `${label}'s indicator thickness`).toBeGreaterThanOrEqual(
        MINIMUM_THICKNESS
      );
      expect([...summary.focusColours].join(), `${label}'s indicator colour`).toContain(
        '--color-accent'
      );
    }
  });
});
