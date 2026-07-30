import { expect, test } from '@playwright/test';

/**
 * The Matrix legend, at a width where its keys do not fit on one line.
 *
 * Issue #34. The legend is a flex row, and `flex` defaults to `nowrap`, so the two prose keys —
 * neither of which can shrink below its own min-content width — pushed the row past the panel. It
 * matters more than a normal overflow because the legend sits *outside* the table's
 * `overflow-x-auto` wrapper: the grid has a scroll container of its own, the legend has none, so
 * the overflow escapes to the document and the whole page scrolls sideways.
 *
 * Filed rather than fixed when it was found, because nothing in the Vitest suite could prove it.
 * jsdom has no layout engine, so `scrollWidth` and `clientWidth` are both 0 there and every
 * assertion below is a tautology — the same blind spot that shipped the `display: contents` scroll
 * bug this directory exists for.
 *
 * The trap the issue records: **most of these keys are conditional**, and in the default scenario
 * the prose ones do not render at all. A spec written against a fresh page passes without the fix. So
 * the scenario is pinned in the querystring and the coexisting keys are asserted as a precondition,
 * before any geometry is read.
 */

/**
 * 320px, not merely "a phone".
 *
 * Measured rather than picked: at 390px the row does not overflow at all — the prose keys wrap
 * their own text and the panel absorbs the rest — so a spec written there passes without the fix
 * for the same reason one written in the default scenario does. The overflow appears at 360 (299px
 * of content in a 286px box, contained by the panel's padding) and escapes to the document at 320
 * (336/320). 320 is also the narrowest width anything still ships at, so it is the floor worth
 * holding rather than an arbitrary tighter number.
 */
const NARROW = { width: 320, height: 640 };

/** A laptop, where the ramp collapse below is a surprise rather than an inevitability. */
const LAPTOP = { width: 1024, height: 768 };

/**
 * MLX at Q5_K_M, which is the state that renders every prose key at once:
 *
 *   - "will not run" — unconditional
 *   - "a struck column heading — MLX (Apple) does not support this hardware, at any size" — MLX runs
 *     on Apple silicon and nothing else, so all but the Apple columns are struck (32 of the 42
 *     shipping devices as the catalog stands at this commit) (#72)
 *   - "some rows scored at a stand-in format MLX (Apple) cannot load" — MLX's only native format is
 *     BF16, so every Apple row is scored at a stand-in
 *   - "past the default allocation, which this machine lets you raise" — DeepSeek V3 at Q5_K_M is
 *     past the 512 GB Mac Studio's 384 GiB default and inside the ceiling it can be tuned to
 *
 * The grid is the whole catalogue regardless of what the Bench holds, so runtime and format are the
 * only two fields that need setting; the rest come from the defaults.
 */
const THREE_KEYS = '/?r=mlx&q=q5_k_m';

const matrix = (page: import('@playwright/test').Page) =>
  page.getByRole('region', { name: /every model on every machine/i });

/**
 * The legend is the section's last direct `div` — the other one is the table's scroll wrapper.
 *
 * Addressed structurally because it has no role of its own, so the ends of the ramp are asserted
 * below to confirm this found the legend rather than something that merely sits where it does. A
 * spec that measures the wrong element passes for the wrong reason, which this suite has already
 * produced three of.
 */
const legend = (page: import('@playwright/test').Page) =>
  matrix(page).locator(':scope > div').last();

/**
 * The ramp group: the first item in the legend row, holding the two endpoint labels and the gradient
 * between them.
 *
 * Its children are asserted in order below — label, ramp, label — which is also what confirms this
 * found the group rather than a key that happens to sit first.
 */
const rampGroup = (page: import('@playwright/test').Page) =>
  legend(page).locator(':scope > span').first();

test.beforeEach(async ({ page }) => {
  await page.setViewportSize(NARROW);
  await page.goto(THREE_KEYS);
  await expect(matrix(page).locator('table td button').first()).toBeVisible();
});

test('renders every prose key at once, which is the state the overflow needs', async ({ page }) => {
  const keys = legend(page);

  // Confirms the locator found the legend and not a neighbour.
  await expect(keys).toContainText('worse');
  await expect(keys).toContainText('better');

  await expect(keys).toContainText(/will not run/i);
  await expect(keys).toContainText(/does not support this hardware, at any size/i);
  await expect(keys).toContainText(/stand-in format/i);
  await expect(keys).toContainText(/past the default allocation/i);
});

test('the legend stays inside the panel on a narrow viewport', async ({ page }) => {
  const box = await legend(page).evaluate((el) => ({
    scrollWidth: el.scrollWidth,
    clientWidth: el.clientWidth,
    right: el.getBoundingClientRect().right,
    panelRight: el.closest('section')!.getBoundingClientRect().right,
  }));

  // A pixel of tolerance, because both widths are integers rounded off a fractional layout — the
  // failure this guards is the min-content width of a whole sentence, ~180px past the panel.
  expect(box.scrollWidth, 'the legend overflows its own box').toBeLessThanOrEqual(
    box.clientWidth + 1
  );
  expect(box.right, 'the legend escapes the panel').toBeLessThanOrEqual(box.panelRight + 1);
});

test('the overflow does not reach the document', async ({ page }) => {
  // The point of the issue rather than a restatement of the test above: the grid has an
  // `overflow-x-auto` of its own and the legend has none, so a legend that is too wide scrolls the
  // page instead of scrolling itself.
  const doc = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));

  expect(doc.scrollWidth, 'the page scrolls sideways').toBeLessThanOrEqual(doc.clientWidth + 1);
});

/**
 * The endpoints, at the width the legend has already overflowed at once.
 *
 * `worse [gradient] better` gave the ramp a direction and no scale, so a mid-blue was unanchored —
 * under "How fast" it could be 20 tok/s or 200 (#71). Naming both ends is one line of text and about
 * 190px of it, added to the row whose min-content width took the document to 336/320 in #34. Adding
 * text to this row is exactly the change that reopens that, which is why it is measured here rather
 * than assumed: the containment tests above now cover the figures too, and this one asserts they are
 * really there, so those cannot pass by describing a legend that never grew.
 */
async function assertEndpointsInsidePanel(page: import('@playwright/test').Page) {
  const parts = rampGroup(page).locator(':scope > span');

  // Label, gradient, label — in order, which also proves the group locator found the ramp.
  await expect(parts).toHaveCount(3);
  await expect(parts.nth(0)).toHaveText(/^worse .*\d/);
  await expect(parts.nth(2)).toHaveText(/\d.* better$/);

  for (const index of [0, 2]) {
    const box = await parts.nth(index).evaluate((el) => ({
      left: el.getBoundingClientRect().left,
      right: el.getBoundingClientRect().right,
      width: el.getBoundingClientRect().width,
      panelLeft: el.closest('section')!.getBoundingClientRect().left,
      panelRight: el.closest('section')!.getBoundingClientRect().right,
    }));

    expect(box.width, 'an endpoint label is not laid out').toBeGreaterThan(0);
    // Both edges: the figure inside each label is `whitespace-nowrap`, so an overrun leaves the panel
    // rather than breaking mid-number, and the left edge is the one no reader can pan to.
    expect(box.left, 'an endpoint label escapes the panel').toBeGreaterThanOrEqual(
      box.panelLeft - 1
    );
    expect(box.right, 'an endpoint label escapes the panel').toBeLessThanOrEqual(
      box.panelRight + 1
    );
  }
}

test('the ramp is labelled with what it spans, inside the panel', async ({ page }) => {
  await assertEndpointsInsidePanel(page);
});

/**
 * And again at the measure whose labels are the longest, which no other run reaches.
 *
 * `measure` is component state rather than a URL key, so every spec in this suite lays out the `fit`
 * labels — "worse 0% free", "100% free better" — and the widest the app can print is a decode pair:
 * 17 characters at "1011 tok/s better", measured across three runtimes, every catalogued format,
 * 4K/32K/128K of context and 1/8/128 users. Asserting containment at `fit` and calling the row safe
 * is measuring the short case. Reached by clicking, because that is the only way in.
 */
test('the widest labels the grid can print stay inside it too', async ({ page }) => {
  await matrix(page).getByRole('button', { name: 'How fast' }).click();
  await expect(rampGroup(page).locator(':scope > span').first()).toHaveText(/tok\/s/);

  await assertEndpointsInsidePanel(page);
});

/**
 * And that wrapping did not fix the overflow by destroying the thing being explained.
 *
 * The ramp is `flex-1`, so its flex basis is 0 — it is the only item here that yields, and under
 * `flex-wrap` an item with a zero basis still takes only the free space left on its line. Every
 * assertion above passes with the gradient at zero width, which is not hypothetical: it is what
 * the legend measured before this fix, at **every viewport below about 1280px**. The three keys
 * together reach the panel's full width on their own, so the ramp — the thing the legend is a
 * legend for — was absent on any laptop while the prose about the exceptions sat at full size.
 */
for (const [name, size] of [
  ['a phone', NARROW],
  ['a laptop', LAPTOP],
] as const) {
  test(`the colour ramp keeps a width worth reading on ${name}`, async ({ page }) => {
    await page.setViewportSize(size);
    const ramp = legend(page).locator('span[aria-hidden="true"]').first();
    const box = await ramp.boundingBox();

    expect(box, 'the ramp is not laid out').not.toBeNull();
    expect(box!.width, 'the ramp collapsed to a hairline').toBeGreaterThan(80);
    expect(box!.height).toBeGreaterThan(0);
  });
}
