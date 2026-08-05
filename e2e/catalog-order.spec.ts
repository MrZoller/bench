import { expect, test } from '@playwright/test';

/**
 * The catalog's list order, made visible — the half of #79 that is geometry.
 *
 * `devices.json`'s row order is the display order, and both surfaces now show the grouping it
 * encodes: the Hardware picker with an `<optgroup>` per class band, the Matrix with a gap between the
 * column bands. Which columns carry the gap, which headings the groups get, and what the caption says
 * are all markup, and `src/App.test.tsx` asserts them in a second. What cannot be asserted there is
 * whether the gap is a *visible* separator and whether paying for it reopens the two overflow bugs
 * this header already has a history of: jsdom reports every width on this surface as 0, which is
 * exactly how #64's 142px of leaning labels survived a full unit suite.
 *
 * The Matrix header's cost is two spacing steps per boundary — 8px at the default root — twice, on a
 * grid whose min-content is ~1405px. That is deliberately the whole cost: the gap is a border on a
 * column widened by the same length in the same unit, so it is in flow and it scales *with* the text
 * rather than against it. But "deliberately" is what #44 and #64 were both said to be, and the first
 * version of this gap was 8px flat against rem columns — which is why the first test below runs at two
 * root sizes.
 */

const matrix = (page: import('@playwright/test').Page) =>
  page.getByRole('region', { name: /every model on every machine/i });

/** The Setup panel, by the name its `sr-only` heading gives the landmark. */
const setup = (page: import('@playwright/test').Page) =>
  page.getByRole('region', { name: 'Setup', exact: true });

/** The grid's squares in the first row, left to right, with the gutter before each. */
async function firstRowGeometry(page: import('@playwright/test').Page) {
  return matrix(page)
    .locator('tbody tr')
    .first()
    .evaluate((row) => {
      const cells = [...row.querySelectorAll('td')];
      const square = (td: Element) => td.querySelector('button')!.getBoundingClientRect();
      return cells.map((td, i) => {
        // Square to square rather than cell to cell, because the gap *is* the difference between the
        // two: the band border sits inside the cell and outside the square it separates.
        const previous = i === 0 ? null : square(cells[i - 1]);
        const button = square(td);
        return {
          index: i,
          // `data-band-start`, not the border utility: the width is expressed in the spacing unit
          // (`calc(var(--spacing) * 2)`) so that it stays equal to `w-9 − w-7` at every root, and a
          // locator naming the class would have to be rewritten every time the length is.
          bandStart: td.hasAttribute('data-band-start'),
          // The gap a reader sees: from the end of the previous square to the start of this one.
          gutter: previous === null ? 0 : button.left - previous.right,
          width: button.width,
        };
      });
    });
}

/**
 * At the default root, and again at 200% text — and the second case is the one that was broken.
 *
 * Both lengths the gap is compensated against are multiples of Tailwind's `--spacing`, which is a
 * `rem`: `w-9` against `w-7` is a two-step difference, and `border-spacing-0.5` is half a step. The
 * first implementation wrote the gap as a flat 8 *pixels* of left border, so "the column is widened
 * by exactly the gap" was true at a 16px root and nowhere else. At 32px the band columns grew by 16px
 * while the border stayed at 8, which paints two of the 42 squares 8px (14%) wider than every other
 * square and drops the boundary gutter from 5x an ordinary one to 3x — both of the assertions below,
 * failing on a grid that looked right in every screenshot anybody took. That is #44's lesson exactly,
 * which `matrix-header.spec.ts` parameterises over these same two roots for the same reason.
 */
for (const root of [16, 32]) {
  test(`the Matrix separates its class bands with a gap a reader can see at a ${root}px root`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto('/');
    if (root !== 16) {
      await page.evaluate((px) => {
        document.documentElement.style.fontSize = `${px}px`;
      }, root);
      // Confirms the scaling landed before anything is measured against it — otherwise the 32px case
      // silently re-runs the 16px one and passes for the wrong reason.
      await page.waitForFunction(
        (px) => getComputedStyle(document.documentElement).fontSize === `${px}px`,
        root
      );
    }
    await expect(matrix(page).locator('tbody tr').first()).toBeVisible();

    const cells = await firstRowGeometry(page);
    const boundaries = cells.filter((c) => c.bandStart);
    const interior = cells.filter((c) => !c.bandStart && c.index > 0);

    // The premise. Two boundaries — discrete GPUs to unified memory, unified memory to CPU — and a
    // long run of ordinary columns to compare them against. A grid where nothing was marked would make
    // every assertion below vacuous rather than false.
    expect(
      boundaries.map((c) => c.index),
      'no column opens a band'
    ).toHaveLength(2);
    expect(interior.length).toBeGreaterThan(30);

    /**
     * The ordinary gutter is the table's `border-spacing`, and the band gap has to be unmistakably
     * more than that — a separator the same width as the thing it separates from is not a separator.
     * Three times is well inside the 5x the implementation spends (two spacing steps of border on top
     * of a half-step gutter) and well outside the sub-pixel noise a fractional layout produces. Both
     * lengths scale with the root, so the ratio is the claim and it holds at either size.
     */
    const ordinary = Math.max(...interior.map((c) => c.gutter));
    expect(ordinary).toBeGreaterThan(0);
    for (const boundary of boundaries) {
      expect(
        boundary.gutter,
        `the band gap at column ${boundary.index} is ${boundary.gutter}px against an ordinary ${ordinary}px gutter`
      ).toBeGreaterThan(ordinary * 3);
    }

    /**
     * And the gap is added to the column rather than taken out of the cell, which is the reason it is a
     * border on a widened column instead of padding. A square a gap narrower than its neighbours would
     * be a heatmap whose marks are not comparable — and on a coarse pointer it would be a hit target
     * under the 44px this repo declares, which `touch-targets.spec.ts` measures on these same columns.
     */
    const widths = cells.map((c) => c.width);
    expect(
      Math.max(...widths) - Math.min(...widths),
      `square widths: ${widths.join(', ')}`
    ).toBeLessThan(1);
  });
}

/**
 * #64's own detector, re-run with the gaps in place: a container may scroll only as far as its
 * in-flow content reaches.
 *
 * `matrix-header.spec.ts` owns this claim at 1280 for the rotated labels. It is repeated here at the
 * narrowest supported viewport and for a different subject — the two 8px borders are new in-flow
 * width on the one surface whose overflow has been filed twice (#64, #34), and 320px is where #34 was
 * measured. A separator that asked for width the layout does not have would show up as scrollable
 * space with no child that wide, exactly as the leaning labels did.
 */
for (const width of [320, 1280]) {
  test(`the banded grid scrolls no further than its own columns reach at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');

    const scroller = matrix(page).locator('div.overflow-x-auto');
    await expect(scroller).toBeVisible();

    const measured = await scroller.evaluate((el) => ({
      scroll: el.scrollWidth,
      table: el.querySelector('table')!.getBoundingClientRect().width,
      bands: el.querySelectorAll('thead th[data-band-start]').length,
    }));

    expect(measured.bands, 'no band separators rendered, so this measures the wrong page').toBe(2);
    expect(
      measured.scroll,
      'the grid scrolls past everything in flow inside it, so a separator is asking for a scrollbar'
    ).toBeLessThanOrEqual(measured.table + 1);
  });
}

/**
 * And the page itself does not scroll sideways with the picker grouped, at 320px.
 *
 * An `<optgroup>` heading is drawn by the platform inside the popup, so it should cost the closed
 * control nothing — but "should cost nothing" is the assumption behind every reflow bug this repo has
 * filed (#35 was a non-wrapping row of four KV options setting the width of a grid column). The
 * headings are the longest strings in the Hardware control by some margin, and the check is cheap.
 */
test('grouping the Hardware picker does not widen the page at 320px', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 900 });
  await page.goto('/');

  const picker = page.getByLabel('Hardware', { exact: true });
  await expect(picker).toBeVisible();

  /**
   * The headings, read from the rendered control rather than from the source: this is the one claim
   * here that a real engine answers differently from jsdom, since a browser that ignored `<optgroup>`
   * entirely would still parse it into the DOM. Order and text, because the bands are a progression —
   * discrete GPUs through unified memory to CPU hosts — and a reader who cannot infer the sequence is
   * back where #79 found them.
   */
  expect(
    await picker
      .locator('optgroup')
      .evaluateAll((groups) => groups.map((g) => (g as HTMLOptGroupElement).label))
  ).toEqual(['Discrete GPUs', 'Unified memory', 'CPU + system RAM']);

  const overflow = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(overflow.document, 'the page scrolls sideways at 320px').toBeLessThanOrEqual(
    overflow.viewport
  );
});

/**
 * The model order stated where a sighted reader meets it, on both surfaces (#179).
 *
 * **`toBeVisible()` is necessary and nowhere near sufficient, which is the whole reason this test
 * has geometry in it.** Playwright calls an element visible when it has a non-empty bounding box
 * and is not `visibility: hidden` — and Tailwind's `sr-only` is a 1x1px absolutely-positioned box
 * with a `clip` rect, so it passes both. A guard written as `toBeVisible()` alone would stay green
 * through exactly the regression this issue was filed about twice: the explanation moving back into
 * screen-reader-only text. So the claim is measured instead — a box wide enough to hold the
 * sentence and tall enough for a line of it — and `sr-only` fails it by two orders of magnitude.
 *
 * jsdom cannot answer any of that: it computes no layout, reports every box as zero, and would
 * report an `sr-only` paragraph and a painted one identically. `src/AppCatalog.test.tsx` owns the
 * DOM half — that the sentence exists, sits in the Setup panel, and matches the rendered option
 * order — and this owns whether anybody can read it.
 *
 * Nothing is clicked, hovered or focused before the assertions, which is the other half of what the
 * issue asked for.
 */
async function assertReadable(locator: import('@playwright/test').Locator, subject: string) {
  await expect(locator, `${subject} is not visible`).toBeVisible();

  /**
   * Three independent detectors, because `sr-only` defeats each of them differently and a utility
   * beside it can rescue any one. Measured with the class applied, which is how these bounds were
   * picked rather than guessed: the caption reports 523x41.5 painted at 1280 and **9px tall**
   * `sr-only` — not 1px, because the `pt-2` on the same element outlives `sr-only`'s `padding: 0`.
   * A threshold set at "taller than nothing" would have been one Tailwind ordering away from
   * green.
   *
   * Width is the unambiguous one (`width: 1px`, against 260px at the narrowest supported viewport),
   * height allows the sentence to shrink to a single line of the smallest type this app ships
   * (10px at `leading-relaxed`), and the overflow check catches the case where a wider box still
   * hides its text behind `white-space: nowrap` and `overflow: hidden`.
   */
  const box = (await locator.boundingBox())!;
  expect(box.width, `${subject} is ${box.width}px wide, which is not a sentence`).toBeGreaterThan(
    100
  );
  expect(
    box.height,
    `${subject} is ${box.height}px tall, which is not a line of text`
  ).toBeGreaterThanOrEqual(16);
  const clipped = await locator.evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(clipped, `${subject} overflows its own box, so its text is clipped`).toBeLessThanOrEqual(
    1
  );
}

test('both model surfaces state their order where a sighted reader can read it', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/');

  /**
   * The Bench's Model picker: the surface that carried no explanation at all until #179.
   *
   * By the sentence's own words rather than by `MODEL_ORDER_RULE` itself, which is the one
   * compromise here. Specs in this directory do import from `@/` — `SETTING_LABELS`, `marks`,
   * `colors` all arrive that way — but `catalog.ts` imports `devices.json`, and Playwright's ESM
   * loader refuses a JSON import without a `with { type: 'json' }` attribute the app's own build
   * does not need. Adding one to the source to satisfy a test runner is the wrong direction, so the
   * phrase is pinned in `catalog.test.ts` against the constant instead: reword the caption past
   * these words and a unit test fails a second before this one does.
   */
  await assertReadable(
    setup(page).getByText(/most-downloaded first, by Hugging Face downloads/i),
    'the Model picker’s order caption'
  );

  // And the Matrix's, which has had a visible sentence since #135 and no browser-level guard on it.
  // Its `sr-only` `<caption>` carries the same facts deliberately, so this names a phrase only the
  // visible paragraph has.
  await assertReadable(
    matrix(page).getByText(/curated set, not a top-N chart/i),
    'the Matrix’s row-order sentence'
  );
});
