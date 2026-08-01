import { expect, test, type Locator, type Page } from '@playwright/test';
import { SETTING_LABELS } from '@/lib/stops';

/**
 * Whether the controls and the figures they drive are ever on screen together.
 *
 * The five Usage controls set the scenario every figure on this page is computed at, and they used to
 * render after all of it. The issue's table, measured at 1440x900 on the default scenario before #66,
 * with `main`'s figures as re-measured in headless Chromium when this landed — the masthead and two
 * catalog rows have moved every row, and the shape is what matters:
 *
 * ```
 * section                              top      height     top on main today
 * Memory budget                         342        220     545  <- driven by the sliders below
 * Verdicts                              582        155     785
 * What you could do with it             757        304     ...
 * How much room is left                1081        409     ...
 * Every model on every machine         1510       1072    1920
 * Usage                                2602        232    2959  <- the sliders
 * gap, memory bar -> context slider    2260px            2402px  (2.5 viewport heights)
 * ```
 *
 * On an iPhone 14 the gap was 3,505px in a 5,302px document — the two were never on screen together at
 * any scroll position. That is a direct-manipulation tool — "drag usage, watch the budget fill" — in
 * which you cannot watch the budget fill while you drag.
 *
 * **jsdom cannot answer this at all.** It has no layout engine, so every `getBoundingClientRect` in it
 * reads 0 and any version of the assertion below is a tautology there. The reading-order half — that
 * the controls precede the figures in the DOM, which is what a screen reader is handed — is a DOM
 * property and lives in `App.test.tsx`, where it costs a second. This file owns the half that needs
 * pixels.
 *
 * Written as a relation between two elements rather than as absolute offsets. The figures above were
 * taken before the masthead landed and every one of them has moved since; what has to hold is that the
 * distance between a slider and the bar it fills stays inside a viewport, at any page height.
 */

/** The issue's own viewport, and its acceptance bar. */
const LAPTOP = { width: 1440, height: 900 };

/** iPhone 14, the second measurement in the issue — where the gap was worse than 2.5 screens. */
const PHONE = { width: 390, height: 844 };

const usage = (page: Page) => page.getByRole('region', { name: 'Usage' });
// "Setup", not "Configuration": the panel's name is its own `sr-only` heading now, so it is in the
// document outline rather than only in landmark navigation (#74) — and the name matches what this
// file, the component's comment and `App.test.tsx` were all already calling it.
const setup = (page: Page) => page.getByRole('region', { name: 'Setup' });
const budget = (page: Page) => page.getByRole('region', { name: /memory budget/i });
const matrix = (page: Page) => page.getByRole('region', { name: /every model on every machine/i });

/** The first of the five, and the one the issue names: the slider the memory bar is drawn from. */
const contextSlider = (page: Page) =>
  page.getByRole('slider', { name: SETTING_LABELS.contextTokens });

/** The bar itself, not its panel — the shape the reader is meant to watch fill. */
const memoryBar = (page: Page) => budget(page).getByRole('img', { name: /allocatable used/i });

/**
 * A box in *document* coordinates, so nothing here depends on where the page happens to be scrolled.
 * `boundingBox()` is viewport-relative and every spec that forgot it reported a shift per run.
 */
const boxOf = async (locator: Locator) => {
  const box = await locator.evaluate((el) => {
    const rect = el.getBoundingClientRect();
    return {
      top: rect.top + window.scrollY,
      bottom: rect.bottom + window.scrollY,
      height: rect.height,
    };
  });
  expect(box.height, 'measured an element with no box').toBeGreaterThan(0);
  return box;
};

test('the context slider and the bar it fills are within one viewport of each other', async ({
  page,
}) => {
  await page.setViewportSize(LAPTOP);
  await page.goto('/');
  // The whole page has to have laid out, or the bar is measured against a grid that has not arrived.
  await expect(matrix(page).locator('table td button').first()).toBeVisible();

  const slider = await boxOf(contextSlider(page));
  const bar = await boxOf(memoryBar(page));

  // Above it, which is the ordering half of the fix.
  expect(bar.top, 'the memory bar is still above the slider that sizes its cache').toBeGreaterThan(
    slider.top
  );
  // And close enough to it to be watched. 2,402px on `main` when this landed, 388px after — two and a
  // half of these viewports down to under half of one.
  expect(
    bar.top - slider.top,
    'the slider and the bar it fills are more than a viewport apart'
  ).toBeLessThan(LAPTOP.height);
});

/**
 * The panels, not just the one slider — the same claim against every figure the controls drive, so a
 * fix that hoisted the context slider alone out of its panel would not pass.
 */
test('every figure on the page is laid out after the controls that set it', async ({ page }) => {
  await page.setViewportSize(LAPTOP);
  await page.goto('/');
  await expect(matrix(page).locator('table td button').first()).toBeVisible();

  const tops = {
    setup: (await boxOf(setup(page))).top,
    usage: (await boxOf(usage(page))).top,
    budget: (await boxOf(budget(page))).top,
    matrix: (await boxOf(matrix(page))).top,
  };

  expect(tops.usage, 'Usage is not under Setup').toBeGreaterThan(tops.setup);
  expect(tops.budget, 'the memory budget is above the Usage panel').toBeGreaterThan(tops.usage);
  // The two grids stay terminal, which is the other half of the placement decision.
  expect(tops.matrix, 'the comparison grid is not the last panel').toBeGreaterThan(tops.budget);
});

/**
 * The phone case, which was the worse one: the Usage panel started 4,316px into a 5,302px document on
 * `main`, past four screens of grid, and now starts at 774px in a document the same height.
 *
 * Asserted as "inside the first two screens" rather than as a gap in viewport heights. Both panels
 * stack to one column here, so nine controls sit between the top of the page and the bar, and that is
 * the real cost of this placement — worth stating honestly rather than dressing up. What the fix has
 * to buy on a phone is that the controls are somewhere a reader arrives at, instead of somewhere they
 * have to scroll past the entire catalog to find. The slider and the bar do come within one screen
 * here — 683px of the 844 — but the bar is still the second screen, not the landing one.
 */
test('the controls are in the first two screens on a phone, not the last', async ({ page }) => {
  await page.setViewportSize(PHONE);
  await page.goto('/');
  await expect(matrix(page).locator('table td button').first()).toBeVisible();

  const panel = await boxOf(usage(page));
  const bar = await boxOf(memoryBar(page));
  const documentHeight = await page.evaluate(() => document.documentElement.scrollHeight);

  // The page is still long — this is not measuring a viewport that happens to hold everything.
  expect(documentHeight, 'the page fits a phone, so this proves nothing').toBeGreaterThan(
    PHONE.height * 3
  );
  /**
   * The panel's *bottom*, and the difference from its top is the whole assertion.
   *
   * `panel.top < 2 x height` says only that the panel begins inside two screens, and stays green with
   * the last three sliders below the fold — which is both the property the title claims and the exact
   * shape of the regression this test exists to catch, since anything added above Usage pushes it down
   * a control at a time. Measured: the panel ends at 1,401px against the 1,688px bar, so there is 287px
   * of real margin rather than slack.
   */
  expect(panel.top, 'the Usage panel starts past the second screen').toBeLessThan(PHONE.height * 2);
  expect(panel.bottom, 'the last Usage control is past the second screen').toBeLessThan(
    PHONE.height * 2
  );
  expect(bar.top, 'the memory bar is above the controls that size it').toBeGreaterThan(panel.top);
});

/**
 * And the anchor a Matrix click scrolls back to still lands on the figures.
 *
 * `matrix-scroll.spec.ts` asserts the scroll happens and lands at the top of the viewport; this
 * asserts what is *at* the top when it does. The anchor sits between the controls and the bar, so a
 * click brings the detail it changed into view rather than two panels of input — which is the thing
 * #66 named as needing checking when the panel moved, since both are positional.
 */
test('a Matrix click scrolls to the figures it changed, not to the controls', async ({ page }) => {
  await page.setViewportSize(LAPTOP);
  await page.goto('/');
  await expect(matrix(page).locator('table td button').first()).toBeVisible();
  // Instant, so the measurement does not race the animation. The component reads this itself: a
  // reduced-motion block neutralises CSS durations and cannot reach a scroll asked for in JS.
  await page.emulateMedia({ reducedMotion: 'reduce' });

  const anchor = page.locator('#bench-detail');
  const panel = await boxOf(usage(page));
  const bar = await boxOf(memoryBar(page));
  const anchorTop = await anchor.evaluate((el) => el.getBoundingClientRect().top + window.scrollY);

  expect(anchorTop, 'the anchor scrolls the controls into view').toBeGreaterThanOrEqual(
    panel.top + panel.height - 1
  );
  // A pixel of tolerance: the anchor is a zero-height box between a `gap-5` and a `-mb-5` that
  // cancels it, and `matrix-scroll.spec.ts` records that fractional layout rounding half a pixel past
  // on the Linux runner. The failure this distinguishes is a whole panel of controls, not a pixel.
  expect(anchorTop, 'the anchor sits past the detail it is meant to show').toBeLessThanOrEqual(
    bar.top + 1
  );

  // And the consequence: after a click, the bar is on screen. The already-selected cell, so the click
  // cannot change the configuration — an unsupported pair renders a refusal in place of the bar, and
  // this test would then fail for a reason that is not about placement.
  await matrix(page).locator('table td button[aria-current="true"]').first().click();
  await expect(memoryBar(page)).toBeInViewport();
});
