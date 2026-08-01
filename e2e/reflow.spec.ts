import { expect, test, type Page } from '@playwright/test';

/**
 * Reflow at 200% text-only zoom — WCAG 1.4.4 (Resize Text). Issue #35.
 *
 * **This is a different test from 1.4.10 (Reflow)**, and conflating them is why the defect
 * survived a responsive pass. 1.4.10 asks whether a 320px viewport works at the *default* text
 * size, which this app already satisfied. 1.4.4 asks whether text can be scaled to 200% without
 * loss of content or function — and browser text-only zoom does that by growing the root font
 * size while leaving the viewport where it is. So every `rem`-derived width grows, every
 * `whitespace-nowrap` line grows, and nothing gives them more room. A layout can pass one and fail
 * the other badly; this one did, by 89px.
 *
 * Why it belongs here rather than in Vitest: jsdom has no layout engine, so `scrollWidth` and
 * `clientWidth` are both 0 and every assertion below is a tautology there.
 *
 * **The two floors a narrow viewport cannot negotiate with**, both of which this found:
 *
 *   - `whitespace-nowrap` on a whole sentence. The four panel headers each carried one, and the
 *     `flex-wrap` on the header above only decides which line the sentence lands on, not how wide
 *     it is. The Matrix's took the document to 409/320 on its own.
 *   - A non-wrapping flex row of options. Its min-content is the *sum* of its children, and
 *     because that row sets the width of its grid column, every `w-full` slider sharing the column
 *     inherited it — four KV options widening three sliders in a panel they are not even part of.
 *
 * Neither is visible at the default root size, which is why this scales the root rather than
 * shrinking the viewport.
 */

/** The narrowest width anything still ships at, matching `matrix-legend.spec.ts`. */
const NARROW = { width: 320, height: 900 };

/**
 * The widths where the layout is a different layout.
 *
 * 320 was the only one this file measured at first, which is below every breakpoint — so **every
 * `sm:` rule was inactive in the entire suite** (#41). The stacked single-column layout was proven
 * to reflow and the multi-column grids were never tested at 200% at all, which is a different
 * question: `sm:grid-cols-2` and `sm:grid-cols-3` divide the viewport among two or three tracks,
 * so each has less room than the single column had, not more.
 *
 * Under real 200% text these widths straddle the boundaries rather than sitting near them. `sm` is
 * 40rem, which is **1280px** once the browser default is 32px, and `lg` is 2048px — so 640 is a
 * stacked column here, 1280 is `sm`'s three columns at their tightest reachable size, and 1920 is
 * still `sm` with room. All four are ordinary devices, and the run does not have to reason about
 * which layout each produces, because the project performs the zoom rather than imitating it —
 * see `playwright.config.ts`.
 *
 * **This list assumes `sm:` is the only breakpoint the app uses**, which is true today and is not
 * self-maintaining: `1280` is chosen as `sm`'s tightest reachable width, and an `lg:` rule added
 * later would activate at 2048px under real zoom, which nothing here measures. Add a width when a
 * breakpoint is added, or the sweep goes back to reporting on layouts it never rendered.
 */
const VIEWPORTS = [
  { name: '320px, a phone', size: NARROW },
  { name: '640px, a small tablet', size: { width: 640, height: 900 } },
  { name: '1280px, a laptop — sm at 200%', size: { width: 1280, height: 900 } },
  { name: '1920px, a desktop', size: { width: 1920, height: 1080 } },
] as const;

/**
 * 32px, because the browser default is 16px and this is a 200% test. Set by the `reflow`
 * project's launch switch, not by this file — see `playwright.config.ts`.
 *
 * Named as a doubling of the default rather than as an absolute, since that is what the success
 * criterion asks for. Past this the page is *not* clean — around 250% long single words like
 * "Unsupported" and the slider labels start escaping. That is recorded rather than fixed: 1.4.4
 * stops at 200%, and "the bar is met" is a different claim from "the layout is unbreakable".
 */
const ROOT_200 = 32;

/**
 * A font deliberately wider than any the app will actually resolve.
 *
 * **Without this the spec measures the host's fonts, not the app**, and the two disagree. The app
 * asks for `ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, …`; a Mac resolves that
 * to SF, and a CI runner has none of them and falls through to whatever fontconfig calls sans —
 * which is wider. The first version of this file passed on macOS with 18px to spare and failed on
 * CI by 4px, on markup neither run had changed. The overflow was real: the page genuinely scrolled
 * sideways for anyone whose system sans is wider than SF, which is most Linux users.
 *
 * So the layout is measured against an upper bound instead of against one machine's typography.
 * Courier New is chosen because it is *much* wider than any realistic UI sans and is present, or
 * metric-aliased, on all three platforms — Liberation Mono stands in on Linux. It is not a claim
 * about what anyone runs; it is a claim that the layout holds however wide the text turns out to
 * be.
 *
 * Verdana was tried first and is not enough: on macOS it does not reproduce the CI failure even
 * with the fix reverted, so a spec written at Verdana would have shipped the same green-here,
 * red-there result again.
 *
 * Applied through `--font-sans` rather than an inline `font-family`. `body` sets
 * `font-family: var(--font-sans)`, so a `style.fontFamily` on `<html>` loses to that rule and
 * changes nothing — which it silently did, in the draft before this one.
 */
const WIDE_FONT = "'Courier New', monospace";

/**
 * Scenarios that put materially different content on the page, because the offending line is
 * built from the numbers in it. The default page and MLX/Q5_K_M are the two the issue was
 * measured at; the rest vary count width, runtime label and the conditional panels.
 */
const SCENARIOS = [
  { name: 'the default page', url: '/', heading: /32K context, 8K prompt, 1 user, FP16 KV/ },
  {
    name: 'MLX at Q5_K_M, where the Matrix legend is fullest',
    url: '/?r=mlx&q=q5_k_m',
    heading: /at Q5_K_M.*MLX \(Apple\)/,
  },
  {
    name: 'vLLM at 128K over eight users',
    url: '/?r=vllm&q=fp8&ctx=131072&u=8',
    heading: /vLLM — 128K context, .*8 users/,
  },
  {
    name: 'llama.cpp at 128 users on eight devices',
    url: '/?r=llama_cpp&q=q4_k_m&ctx=131072&u=128&n=8',
    heading: /128K context, .*128 users.*one device per cell/,
  },
] as const;

/**
 * That the querystring actually reached the store.
 *
 * `url.ts` reads only the keys in its own map and ignores everything else in silence, so a wrong
 * key is not an error — it is a page that loads at the defaults while the test name goes on
 * claiming 128K. The first draft said `c=131072`, where the key is `ctx`, and both long-context
 * scenarios were quietly measuring the default page.
 *
 * **The obvious guard for that is itself vacuous**, which is worth recording because it was
 * written and it passed: reading `searchParams.get('ctx')` and skipping when absent means the
 * wrong key produces no expectation at all, so restoring the bug leaves every test green. That is
 * why the expected heading is spelled out per scenario above instead of derived from the URL — an
 * expectation that comes from the same string it is checking cannot contradict it.
 *
 * Asserted through the Matrix heading because it names every input that moves the grid, and it is
 * rendered from the same config the layout is.
 */
async function assertScenarioLoaded(page: Page, expected: RegExp) {
  await expect(
    page.getByRole('heading', { name: /every model on every machine/i }),
    'the querystring did not reach the page'
  ).toContainText(expected);
}

/**
 * Applies the stress font, if one is asked for. The root font size is not touched — the browser
 * is already at 32px by the time the page loads, which is the whole point of the project.
 *
 * Through `--font-sans` rather than an inline `font-family`: `body` sets
 * `font-family: var(--font-sans)`, so a `style.fontFamily` on `<html>` loses to that rule and
 * changes nothing — which it silently did, in an earlier draft.
 */
async function useStressFont(page: Page, font?: string) {
  if (!font) return;
  await page.evaluate((f) => {
    document.documentElement.style.setProperty('--font-sans', f);
  }, font);
}

/**
 * How wide a fixed string renders, used to prove the stress font is doing something.
 *
 * `WIDE_FONT` is a request, not a guarantee — a host with neither Courier New nor a metric alias
 * falls back to its default sans and the stress silently becomes an ordinary run. That is the
 * shape of failure this whole file keeps hitting, so it is measured rather than trusted.
 */
async function probeTextWidth(page: Page) {
  return page.evaluate(() => {
    const probe = document.createElement('span');
    probe.textContent = 'Inline code completion';
    probe.style.cssText = 'position:absolute;white-space:nowrap;visibility:hidden;font-size:32px';
    document.body.append(probe);
    const width = probe.getBoundingClientRect().width;
    probe.remove();
    return width;
  });
}

/**
 * Every element whose right edge escapes the document, ignoring anything inside a deliberate
 * scroll container.
 *
 * The exclusion is the point: the Matrix grid and the Envelope canvas are *supposed* to scroll
 * horizontally, inside a box of their own. What 1.4.4 forbids is the overflow reaching the
 * document, so the walk stops descending at the first `overflow-x` that is not `visible` — which
 * is also what keeps a failure naming the element that actually needs fixing rather than the
 * thousand table cells inside a scroller that is behaving correctly.
 */
async function escapees(page: Page) {
  return page.evaluate(() => {
    const docWidth = document.documentElement.clientWidth;
    const out: string[] = [];
    const walk = (el: HTMLElement) => {
      if (getComputedStyle(el).overflowX !== 'visible') return;
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.right > docWidth + 1) {
        const cls = (el.className?.toString() ?? '').slice(0, 70);
        out.push(
          `${Math.round(rect.right)}px <${el.tagName.toLowerCase()}> ` +
            `"${(el.textContent ?? '').trim().slice(0, 45)}" .${cls}`
        );
      }
      for (const child of Array.from(el.children)) walk(child as HTMLElement);
    };
    walk(document.body);
    return out;
  });
}

test.describe('at 200% text size', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize(NARROW);
  });

  /**
   * The precondition, asserted before any width is.
   *
   * Same reasoning as the touch project asserting `(pointer: coarse)` matches before measuring a
   * hit target: if the launch switch stopped working, every assertion below would run at the
   * default text size, where the defects do not exist and all of them pass. `--blink-settings` is
   * a Blink-internal switch with no stability promise, so this is the test that turns a silent
   * downgrade into a loud one.
   */
  test('the browser really is at 200% text', async ({ page }) => {
    await page.goto('/');

    const sizes = await page.evaluate(() => ({
      root: parseFloat(getComputedStyle(document.documentElement).fontSize),
      heading: parseFloat(getComputedStyle(document.querySelector('h1')!).fontSize),
    }));

    expect(sizes.root, 'the launch switch did not take — this is a 100% run').toBe(ROOT_200);
    /*
     * The masthead wordmark is `clamp(2rem, 10vw, 5rem)`. At this viewport the `10vw` preferred
     * term is 32px, well under the 2rem floor, so the floor is what applies — 64px at this root
     * and 32px at the default.
     *
     * Asserted so a change that pinned the root while leaving rem-derived text alone would still
     * fail here. The floor being in `rem` is the load-bearing part: swapping it for a `px` or a
     * `vw`-only size would leave the largest text on the page ignoring a low-vision reader's
     * request outright, and this is the assertion that says so.
     */
    expect(sizes.heading, 'rem-derived text did not follow the root').toBeCloseTo(ROOT_200 * 2, 0);
  });

  /**
   * And that the **breakpoints moved with the text**, which is the reason this project exists.
   *
   * Tailwind's breakpoints are `rem`, and `rem` inside a media query resolves against the
   * browser's default font size — not against an author-set `documentElement.style.fontSize`. So
   * simulating zoom by setting the root leaves every breakpoint where it was, and the page ends up
   * in layout states no reader can reach: an earlier draft of this file reported three columns
   * crushed into 213px each at 640px, a state that exists only because the breakpoint did not
   * move.
   *
   * Performing the zoom fixes that, and this is the assertion that says so. `sm` is 40rem, which
   * is 1280px at this root rather than 640px — so 640 must be *below* it and 1280 at it.
   */
  test('the breakpoints moved with the text, not just the type', async ({ page }) => {
    const sm = () => page.evaluate(() => matchMedia('(min-width: 40rem)').matches);

    await page.setViewportSize({ width: 640, height: 900 });
    await page.goto('/');
    expect(await sm(), '640px is still past sm — the media queries did not scale').toBe(false);

    await page.setViewportSize({ width: 1280, height: 900 });
    expect(await sm(), '1280px is not past sm — the breakpoint moved too far').toBe(true);
  });

  /**
   * And that the *smallest* text scales too, which the heading above does not establish.
   *
   * The precondition originally checked only the `<h1>`, proving that rem-derived text responds —
   * which it always did. Text sized in absolute pixels does not respond at all, and both Envelope
   * axes were `text-[10px]`: at 200% every other figure doubled while the axis labels stayed at
   * 10px, failing WCAG 1.4.4 outright and leaving them *relatively* half the size on the surface a
   * low-vision reader had just asked to enlarge (#42).
   *
   * So the assertion is on the axis labels specifically, and it is the check that would have found
   * them. A page-wide "nothing is absolute" sweep was the alternative and is worse: it would have
   * to exempt borders, hairlines and icon boxes, and the exemption list is where the next 10px
   * label would go to hide.
   */
  test('the smallest text on the page scales too, not just the headings', async ({ page }) => {
    await page.goto('/');

    const axisLabel = page
      .getByRole('region', { name: /how much room is left/i })
      .locator('ol li')
      .first();
    await expect(axisLabel).toBeVisible();

    const measured = await axisLabel.evaluate((el) => ({
      label: parseFloat(getComputedStyle(el).fontSize),
      root: parseFloat(getComputedStyle(document.documentElement).fontSize),
    }));

    expect(measured.label, 'no axis label was measured').toBeGreaterThan(0);
    // Derived from the root rather than fixed: 0.625rem, which is the 10px it used to be, at the
    // default root only.
    expect(measured.label / measured.root, 'the axis label is not rem-derived').toBeCloseTo(
      0.625,
      2
    );
    // And therefore actually bigger here. This is the assertion a regression to `text-[10px]`
    // fails, since that renders 10px at every root there is.
    expect(measured.label, 'the axis labels ignore the text-size setting').toBeGreaterThan(10);
  });

  /**
   * The second precondition: that the stress font is actually wider than the host's own.
   *
   * Without it, a runner missing Courier New and every metric alias falls back to its default
   * sans, and the four stress runs below quietly become duplicates of the four ordinary ones —
   * green, and testing nothing they do not already test.
   */
  test('the stress font is wider than whatever this host resolves', async ({ page }) => {
    await page.goto('/');
    const native = await probeTextWidth(page);

    await useStressFont(page, WIDE_FONT);
    const stressed = await probeTextWidth(page);

    expect(native, 'nothing was measured').toBeGreaterThan(0);
    expect(stressed / native, 'the stress font did not widen the text').toBeGreaterThan(1.05);
  });

  /**
   * Every scenario, at every breakpoint, at both typographies.
   *
   * The two axes cover different failures and neither implies the other. **Font** is what made the
   * verdict portable — the first version passed on macOS with 18px to spare and failed on CI by
   * 4px, on markup neither run had changed. **Width** is what stops the suite testing one layout
   * and reporting on three: below `sm` the page is a single stacked column, and above it the same
   * content is divided among two or three grid tracks, each with less room than the single column
   * had. Passing at 320 says nothing about either grid (#41).
   */
  for (const { name: at, size } of VIEWPORTS) {
    for (const { name, url, heading } of SCENARIOS) {
      for (const [typography, font] of [
        ['this host’s own fonts', undefined],
        ['a font wider than any it will resolve', WIDE_FONT],
      ] as const) {
        test(`no sideways scroll at ${at} on ${name}, at ${typography}`, async ({ page }) => {
          await page.setViewportSize(size);
          await page.goto(url);
          await assertScenarioLoaded(page, heading);
          await useStressFont(page, font);

          // Named before the numeric assertion, so a failure says which element to fix rather than
          // only by how much the document is too wide.
          expect(await escapees(page), 'elements escaping the document').toEqual([]);

          const doc = await page.evaluate(() => ({
            scrollWidth: document.documentElement.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
          }));
          expect(doc.scrollWidth, 'the page scrolls sideways at 200% text').toBeLessThanOrEqual(
            doc.clientWidth + 1
          );
        });
      }
    }
  }

  /**
   * The segmented control's own fix, asserted directly.
   *
   * The document-width test above passes if this row is fixed *or* if something else absorbs it,
   * so it does not pin the mechanism. This does: at 200% the four KV options cannot share a line,
   * so a wrapping row puts them on more than one. Remove `flex-wrap` and the options return to a
   * single line whose min-content is their sum — which is the state that widened the grid column
   * and dragged three sliders out of the viewport with it.
   */
  test('the segmented options wrap rather than summing their widths', async ({ page }) => {
    await page.goto('/');

    const kv = page.getByRole('group', { name: /kv precision/i });
    await expect(kv).toBeVisible();

    const lines = await kv.locator('label').evaluateAll((labels) => {
      const tops = labels.map((l) => Math.round(l.getBoundingClientRect().top));
      return { count: labels.length, rows: new Set(tops).size };
    });

    expect(lines.count, 'the KV control lost its options').toBeGreaterThan(2);
    expect(lines.rows, 'the options are all on one line, so the row still sums').toBeGreaterThan(1);
  });

  /**
   * The Matrix legend's two endpoint figures, which the sweep above cannot reach on two counts.
   *
   * **The stress font never touches them.** `useStressFont` sets `--font-sans`, and every figure in
   * this app carries `.tabular`, which resolves `--font-mono` (`index.css`). So the four stress runs
   * above measure the app's prose in Courier New and its numbers in whatever monospace the host has —
   * a class-wide gap that predates these labels and now has two more members. Stressing the sweep's
   * own runs through `--font-mono` as well is the fix for the class and is deliberately not done here:
   * the file's own precondition is that the stress font is *wider* than what the host resolves, and
   * one monospace substituted for another is not, so it would add red risk to unrelated panels without
   * adding a stress. Set on both variables for these two labels instead, where the question is
   * specifically whether a protected figure fits.
   *
   * **And no run reaches the widest labels.** `measure` is component state rather than a URL key, so
   * every scenario above lays out `fit`'s "worse 0% free" while the longest pair the app can print is
   * a decode one — 17 characters at "1011 tok/s better", measured over three runtimes, every
   * catalogued format, 4K/32K/128K of context and 1/8/128 users. Clicked, because that is the only way
   * in.
   *
   * What is asserted is containment of each label rather than the document's own width: the point is
   * the two `whitespace-nowrap` figures, and a document-level assertion here would report on every
   * other `.tabular` element the mono stress touches — a different (and pre-existing) question.
   */
  test('the legend endpoints hold their figures inside the panel', async ({ page }) => {
    await page.setViewportSize(NARROW);
    await page.goto('/');

    const matrix = page.getByRole('region', { name: /every model on every machine/i });
    await useStressFont(page, WIDE_FONT);
    await page.evaluate((f) => {
      document.documentElement.style.setProperty('--font-mono', f);
    }, WIDE_FONT);

    // The measure with the widest labels, and a precondition that it took: at `fit` these read
    // "0% free" and the assertion below would be measuring the short case.
    await matrix.getByRole('button', { name: 'How fast' }).click();
    const ends = matrix.locator(':scope > div').last().locator(':scope > span').first();
    await expect(ends).toContainText(/tok\/s/);

    const boxes = await ends.locator(':scope > span').evaluateAll((els) =>
      els.map((el) => {
        const rect = el.getBoundingClientRect();
        const panel = el.closest('section')!.getBoundingClientRect();
        return { left: rect.left, right: rect.right, width: rect.width, panel };
      })
    );

    // Label, gradient, label. The gradient is the zero-basis item #34 is about and is measured in
    // `matrix-legend.spec.ts`; these two are the ones carrying text that cannot break.
    expect(boxes).toHaveLength(3);
    for (const box of [boxes[0], boxes[2]]) {
      expect(box.width, 'an endpoint label is not laid out').toBeGreaterThan(0);
      expect(box.left, 'an endpoint label escapes the panel at 200%').toBeGreaterThanOrEqual(
        box.panel.left - 1
      );
      expect(box.right, 'an endpoint label escapes the panel at 200%').toBeLessThanOrEqual(
        box.panel.right + 1
      );
    }
  });

  /**
   * And that wrapping did not fix the overflow by breaking the thing worth protecting.
   *
   * "12 of 425" split across a line reads as two unrelated numbers, so `PanelCount` keeps a
   * `whitespace-nowrap` on the numeral pair alone while letting the noun after it wrap.
   *
   * Asserted structurally rather than by geometry, and deliberately so: the pair is short enough
   * that it never breaks on its own at any root size from 32px to 64px — measured, not assumed —
   * so `getClientRects().length === 1` is true with the class and true without it. That assertion
   * would pass against markup with the protection deleted, which is the failure mode this suite
   * keeps producing. The computed style is the honest thing to check here.
   */
  test('the numeral pair keeps its own nowrap', async ({ page }) => {
    await page.goto('/');

    const pair = page
      .getByRole('region', { name: /every model on every machine/i })
      .locator('p', { hasText: /combinations run/ })
      .locator('span.whitespace-nowrap')
      .first();

    await expect(pair).toHaveText(/^\d+ of \d+$/);
    await expect(pair).toHaveCSS('white-space', 'nowrap');
    // The sentence around it must *not* be protected, or the fix is undone one level up.
    const parent = page
      .getByRole('region', { name: /every model on every machine/i })
      .locator('p', { hasText: /combinations run/ });
    await expect(parent).not.toHaveCSS('white-space', 'nowrap');
  });
  /**
   * The Matrix readout, filled, at 200% — the case neither suite covered
   * ([#102](https://github.com/MrZoller/bench/issues/102)).
   *
   * `min-h` is `5rem` below `sm`, `2.5rem` at `sm` and `1.25rem` at `lg`, measured against the
   * widest sentence at a 16px root. At a 32px root the reservation doubles — and so do the glyphs, so
   * the same sentence wraps into roughly twice as many lines each twice as tall. **Reserved height
   * scales linearly and required height closer to quadratically**, which is the whole defect.
   *
   * It survived because each suite covered one axis: `reflow.spec.ts` runs a 32px root at 320px and
   * never filled the readout, and `matrix-readout.spec.ts` fills it at the default text size. This is
   * both at once, in the project whose browser default is 32px rather than an author-set root — per
   * this file's own lesson, since `rem` in a media query resolves against the browser default and the
   * breakpoints have to move with the text.
   *
   * Measured before the fix, on the longest sentence the grid can produce: **280px needed against
   * 160px reserved at 320px**, and 240 against 160 at 390. Both fixed by the narrow form of the
   * sentence, which drops the preamble the axes already carry.
   *
   * The assertion is the *natural* height against the reservation rather than the rendered height
   * against it: the rendered height already includes the `min-height`, so comparing the two is true
   * of every sentence that fits on one line and is the shape that let this ship. Lifting the floor
   * and re-measuring is what asks the question.
   */
  for (const { width, query, at } of [
    { width: 320, query: '', at: '' },
    { width: 390, query: '', at: '' },
    /*
     * And on a linked rig, because the narrow form carries ", one device" there — the qualifier the
     * panel heading states and a touch reader in a lower row cannot see (found in review on #102).
     * It is the longest the sentence gets, so the reservation has to hold it and not merely the
     * single-device case the two rows above measure.
     */
    { width: 320, query: '?n=4', at: ' on four devices' },
    /*
     * And under a runtime that refuses most of the grid, because the refusal sentence is a different
     * shape from the figures — it names the runtime as well as the machine, and llama.cpp drives
     * everything, so the default scenario contains no such cell at all and the three rows above
     * measure a case this branch never reaches.
     */
    { width: 320, query: '?r=mlx', at: ' under a runtime that refuses most of it' },
  ]) {
    test(`the filled readout stays inside its reservation at ${width}px${at}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`/${query}`);

      const matrix = page.getByRole('region', { name: /every model on every machine/i });
      const cells = matrix.locator('table[role="grid"] td button');
      await expect(cells.first()).toBeVisible();
      const readout = matrix.locator(':scope > p');

      const reserved = await readout.evaluate((el) => el.getBoundingClientRect().height);
      // The reservation itself, so a run against a panel that had stopped reserving anything cannot
      // pass by having nothing to overflow.
      expect(reserved, 'the readout reserves no height at all').toBeGreaterThan(0);

      /*
       * The cells whose sentence is longest, ranked before any of them is focused.
       *
       * Which cell overflows moves with the catalog, the runtime and the measure — the pre-fix worst
       * case was a spilling row, neither the first cell nor the last — so a fixed sample would be a
       * claim about the cells that happened to be picked. Focusing all 1,470 and waiting a frame each
       * is the honest sweep and costs 25 seconds a viewport, which is most of this suite.
       *
       * So the ranking is done first, off each cell's `title` — the same sentence, already in the DOM
       * for every cell without focusing anything. The key is the part *after* the colon, because that
       * is the detail, and the narrow form the reservation has to hold is the detail plus at most a
       * stand-in qualifier. Ranking on the whole title would sort by model and device name, which is
       * exactly the half the narrow form drops.
       */
      const worst = await page.evaluate(async (sample: number) => {
        const p = document.querySelector('section p.sticky') as HTMLElement;
        const detail = (button: HTMLElement) => {
          const said = button.getAttribute('title') ?? '';
          return said.slice(said.indexOf(': ') + 1);
        };
        const buttons = [...document.querySelectorAll<HTMLElement>('table[role="grid"] td button')]
          .sort((a, b) => detail(b).length - detail(a).length)
          .slice(0, sample);

        let tallest = { natural: 0, text: '', checked: 0 };
        for (const button of buttons) {
          button.focus();
          // A frame, so React has rendered the sentence this focus asked for.
          await new Promise((resolve) => requestAnimationFrame(resolve));
          const floor = p.style.minHeight;
          p.style.minHeight = '0px';
          const natural = p.getBoundingClientRect().height;
          p.style.minHeight = floor;
          tallest.checked++;
          if (natural > tallest.natural) {
            tallest = {
              ...tallest,
              natural,
              text: (p.textContent ?? '').trim().slice(0, 80),
            };
          }
        }
        return tallest;
      }, 25);

      // The sample really was taken, so a selector that stopped matching cannot report a clean run
      // over nothing — the failure this repo has shipped three variants of.
      expect(worst.checked, 'no cell was measured').toBe(25);

      // The premise: something was actually rendered into the line. Without it a readout that had
      // stopped filling would report a natural height of zero and pass every bound below.
      expect(worst.natural, 'nothing filled the readout, so nothing was measured').toBeGreaterThan(
        0
      );
      expect(worst.text.length, 'the readout filled with an empty sentence').toBeGreaterThan(10);

      expect(
        worst.natural,
        `the tallest readout needs ${Math.round(worst.natural)}px against ${Math.round(reserved)}px reserved: "${worst.text}"`
      ).toBeLessThanOrEqual(reserved);
    });
  }
});
