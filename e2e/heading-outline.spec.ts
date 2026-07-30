import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The half of the heading outline that has a box (#74).
 *
 * The outline itself is DOM — which panel names which heading, what parents what — and it is pinned in
 * `App.test.tsx`, where it costs a second. Three of the eight headings are `sr-only`, and that is a
 * layout claim in two parts that jsdom cannot evaluate at all: it has no layout engine, so every rect
 * below reads 0 there and both assertions are tautologies.
 *
 * **Why it is worth a spec rather than trusting the utility.** `sr-only` is doing two jobs here, and a
 * typo in the class name breaks both silently in the direction nothing else notices — the accessible
 * name still resolves, `App.test.tsx` stays green, and the page grows a heading nobody asked for:
 *
 *   1. The headings must not be *on* the page. #66 measured what the landing view costs — 620px of
 *      controls and no computed figure at 1440x900 — and traded it deliberately against the slider and
 *      the bar being in one viewport. Two visible headings spend more of that budget, so "hidden" is a
 *      measured decision and not a preference.
 *   2. They must take no track from the grids they sit in. Both control panels are `sm:grid-cols-2`
 *      and the verdict strip is `sm:grid-cols-3`, and an in-flow heading is a grid *item*: it would
 *      claim the first cell and push every control one place along — Model beside a heading, Hardware
 *      starting the second row, and the third verdict tile wrapping onto a row of its own. An
 *      absolutely-positioned box is not a grid item, which is the property being relied on.
 */

/** The issue's own viewport, and wide enough that both panels are actually in their two columns. */
const LAPTOP = { width: 1440, height: 900 };

/** The three that carry no ink, and the panels they name. */
const HIDDEN = ['Setup', 'Usage', 'Verdicts'] as const;

/**
 * Four headings that are meant to be read, as the control.
 *
 * Without them "every heading measures 1px" is satisfied by a page whose headings are all broken, and
 * the assertion below would go green on exactly the regression it exists to catch.
 */
const VISIBLE = [
  /^Memory budget/,
  /^What you could do with it/,
  /^How much room is left/,
  /^Every model on every machine/,
] as const;

const heading = (page: Page, name: string | RegExp) =>
  page.getByRole('heading', { name, level: 2 });

/**
 * A box in *document* coordinates, plus the computed values the assertions below need.
 *
 * Document rather than viewport, like `usage-placement.spec.ts`: `boundingBox()` is viewport-relative
 * and every spec in this directory that forgot it reported a shift per run. `evaluate` throws on a
 * locator that matched nothing or matched several, so a mis-addressed element fails loudly here rather
 * than measuring zero and passing.
 *
 * `borderTopWidth` rides along with `paddingTop` because `top` is the *border-box* edge and
 * `paddingTop` excludes the border, so the two are only a content edge together. `.panel` is
 * `border: 1px` in `src/index.css`: computing the content edge from padding alone leaves the
 * "nothing above the first control" assertion below reading exactly 1 against its 2px tolerance, so a
 * design tweak to that one declaration would turn this spec red pointing at a heading that had not
 * moved.
 */
const boxOf = async (locator: Locator) => {
  const box = await locator.evaluate((el) => {
    const rect = el.getBoundingClientRect();
    const styles = getComputedStyle(el);
    return {
      top: rect.top + window.scrollY,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      position: styles.position,
      paddingTop: parseFloat(styles.paddingTop),
      borderTopWidth: parseFloat(styles.borderTopWidth),
    };
  });
  return box;
};

test.beforeEach(async ({ page }) => {
  await page.setViewportSize(LAPTOP);
  await page.goto('/');
  // The whole page has to have laid out, or a panel is measured before the catalog arrives.
  await expect(page.getByRole('region', { name: /every model on every machine/i })).toBeVisible();
});

test('the three hidden headings are in the outline without being on the page', async ({ page }) => {
  for (const name of HIDDEN) {
    // In the accessibility tree, which is the whole point of them: each is its panel's accessible
    // name, so the landmark not resolving would fail here before any geometry is read.
    await expect(page.getByRole('region', { name })).toBeVisible();

    const box = await boxOf(heading(page, name));
    expect(box.position, `the ${name} heading is in flow, so it is a grid item`).toBe('absolute');
    // 1x1 and clipped, which is what `sr-only` is: a real box in the tree, no ink on the page.
    expect(box.width, `the ${name} heading is drawn ${box.width}px wide`).toBeLessThanOrEqual(1);
    expect(box.height, `the ${name} heading is drawn ${box.height}px tall`).toBeLessThanOrEqual(1);
  }

  // And the four that are meant to be read still are, so the rule above is about these three rather
  // than about every heading on the page.
  for (const name of VISIBLE) {
    const box = await boxOf(heading(page, name));
    expect(box.height, `a panel heading is only ${box.height}px tall`).toBeGreaterThan(8);
    expect(box.width).toBeGreaterThan(50);
  }
});

test('a hidden heading takes no cell from the grid it names', async ({ page }) => {
  const setup = page.getByRole('region', { name: 'Setup' });
  const panel = await boxOf(setup);
  // The four `Select`s are the section's only element children besides the heading.
  const cells = await setup.locator(':scope > div').all();
  expect(cells, 'the Setup panel does not hold four controls').toHaveLength(4);
  const [model, hardware, quantization] = await Promise.all(cells.slice(0, 3).map(boxOf));

  // Two columns, filled across: Model and Hardware share a row, Quantization starts the next. With
  // the heading in flow it would hold cell one and shift all four along by one, putting Model beside
  // a heading and Hardware at the start of row two.
  expect(model.top, 'Model and Hardware are not on one row').toBeCloseTo(hardware.top, 0);
  expect(model.left, 'Model is not the first column').toBeLessThan(hardware.left);
  expect(quantization.top, 'Quantization is not on the second row').toBeGreaterThan(model.top);
  expect(quantization.left, 'Quantization is not under Model').toBeCloseTo(model.left, 0);
  // Nothing above the first row: the top control starts at the panel's content edge — border and
  // padding, since `top` is the border-box edge. This is the assertion a visible heading fails
  // outright, whatever the columns end up doing.
  expect(
    model.top - (panel.top + panel.borderTopWidth + panel.paddingTop),
    'something is laid out above the first control'
  ).toBeLessThan(2);

  // The same claim on the three-column strip, where the wrap is what would give it away: a heading in
  // the first cell pushes the time-to-first-token tile onto a second row.
  const verdicts = page.getByRole('region', { name: 'Verdicts' });
  const tiles = await verdicts.locator(':scope > article').all();
  expect(tiles, 'the verdict strip does not hold three tiles').toHaveLength(3);
  const [capacity, decode, prefill] = await Promise.all(tiles.map(boxOf));

  expect(capacity.top, 'the verdict tiles are not on one row').toBeCloseTo(decode.top, 0);
  expect(prefill.top, 'the third verdict tile has wrapped onto its own row').toBeCloseTo(
    capacity.top,
    0
  );
  expect(capacity.left, 'the first tile does not start the row').toBeLessThan(decode.left);
});
