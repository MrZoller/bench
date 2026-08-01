import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The readout under the comparison grid, and the only question about it jsdom cannot answer.
 *
 * Every figure the grid computes used to live in a native `title` (#71), so a sighted keyboard user
 * arrowing a row got the colour and nothing else. The fix is a reserved line under the grid that
 * fills on hover *and* on focus — and the word doing the work is **reserved**. A line that appears
 * when it has something to say reflows whatever sits under it on every move between two cells, and if
 * the line were above the grid it would reflow the cells themselves: the square under the pointer
 * moves as the readout describes it, which is a hover trap rather than a readout.
 *
 * Which string the line holds, when it fills and when it clears is DOM, and `App.test.tsx` pins all
 * of it in a second — including the pointer half, via synthetic mouse events. What no unit test can
 * see is whether `min-h` is worth a line of height and whether a filled line leaves the geometry
 * where it was: jsdom has no layout engine, so every rect below reads 0 there and every assertion
 * would be a tautology. That is the split this directory exists for.
 */

/** Wide enough that the sentence fits one line, which is what makes "nothing moved" a real claim. */
const DESKTOP = { width: 1280, height: 900 };

/**
 * 320px, the narrowest width anything ships at — and the width the legend's own overflow history
 * lives at (#34). A whole sentence is a much longer string than the legend's keys, and it lands in
 * the same panel with no scroll container of its own.
 */
const NARROW = { width: 320, height: 640 };

const matrix = (page: Page) => page.getByRole('region', { name: /every model on every machine/i });

/**
 * The readout is the section's only direct paragraph — the workload caveat is inside the `header` and
 * the measure hint inside the `fieldset`. It carries no role of its own on purpose: the sentence is a
 * verbatim copy of the focused cell's accessible name, so a live region would announce every cell
 * twice. Asserted below to hold a cell's sentence before anything is measured, so a locator that
 * found some other paragraph cannot pass for the wrong reason.
 */
const readout = (page: Page) => matrix(page).locator(':scope > p');
const grid = (page: Page) => matrix(page).locator('table[role="grid"]');
/** The legend: the section's last direct `div`, the other one being the table's scroll wrapper. */
const legend = (page: Page) => matrix(page).locator(':scope > div').last();

/**
 * A box in *document* coordinates.
 *
 * Focusing a cell scrolls it into view, so viewport rects move for a reason that has nothing to do
 * with the readout — and a spec that measured those would report a shift on every run. Adding the
 * scroll offset back means only a real reflow can change these numbers.
 */
const boxOf = (locator: Locator) =>
  locator.evaluate((el) => {
    const rect = el.getBoundingClientRect();
    return {
      top: rect.top + window.scrollY,
      left: rect.left + window.scrollX,
      width: rect.width,
      height: rect.height,
    };
  });

test('the reserved line holds a line of height while it is empty', async ({ page }) => {
  await page.setViewportSize(DESKTOP);
  await page.goto('/');
  await expect(grid(page).locator('td button').first()).toBeVisible();

  // Empty, and still occupying the space the sentence will need. `min-h-[1.25rem]` is 20px at the
  // default root, which is exactly one line of `text-sm` — so the fill below can cost nothing.
  await expect(readout(page)).toHaveText('');
  const box = await boxOf(readout(page));
  expect(
    box.height,
    'the line collapses while empty, so filling it must push something'
  ).toBeGreaterThanOrEqual(19);
});

test('filling it moves neither the grid above nor the legend below', async ({ page }) => {
  await page.setViewportSize(DESKTOP);
  await page.goto('/');
  const cell = grid(page).locator('td button[tabindex="0"]');
  // Brought into view first, so the only scrolling in this test happens before anything is measured.
  await cell.scrollIntoViewIfNeeded();

  await expect(readout(page)).toHaveText('');
  const before = { grid: await boxOf(grid(page)), legend: await boxOf(legend(page)) };

  await cell.focus();
  // The precondition. Without this the two comparisons below are satisfied by a readout that never
  // filled at all, which is the shape of spec this suite has already produced three of.
  await expect(readout(page)).toContainText(/ on .+:/);

  const after = { grid: await boxOf(grid(page)), legend: await boxOf(legend(page)) };

  // The grid cannot move, because the line is under it — which is the reason it is under it.
  expect(after.grid.top, 'the grid moved under the pointer').toBeCloseTo(before.grid.top, 0);
  expect(after.grid.height).toBeCloseTo(before.grid.height, 0);
  // And the legend does not move either, because the space was already reserved.
  expect(after.legend.top, 'the reservation was not worth the sentence').toBeCloseTo(
    before.legend.top,
    0
  );
});

test('a whole sentence wraps at 320px instead of scrolling the page', async ({ page }) => {
  await page.setViewportSize(NARROW);
  await page.goto('/');
  // The last cell: the longest device name in the catalog, and a row far enough down the popularity
  // sort to have a long model name — the longest sentence the grid can produce.
  const cell = grid(page).locator('td button').last();
  await cell.focus();
  await expect(readout(page)).toContainText(/ on .+:/);

  const box = await readout(page).evaluate((el) => ({
    right: el.getBoundingClientRect().right,
    panelRight: el.closest('section')!.getBoundingClientRect().right,
    scrollWidth: el.scrollWidth,
    clientWidth: el.clientWidth,
  }));
  expect(box.scrollWidth, 'the sentence overflows its own box').toBeLessThanOrEqual(
    box.clientWidth + 1
  );
  expect(box.right, 'the readout escapes the panel').toBeLessThanOrEqual(box.panelRight + 1);

  // The point of measuring at all: the grid has an `overflow-x-auto` of its own and this paragraph
  // has none, so a line that cannot wrap scrolls the document, exactly as the legend did in #34.
  const doc = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(doc.scrollWidth, 'the page scrolls sideways').toBeLessThanOrEqual(doc.clientWidth + 1);
});

/**
 * The readout has to be *visible* from wherever the focused cell is (found in review on #71).
 *
 * The paragraph sits after the table, and the table is 17 rows plus a rotated header band. So a
 * sighted keyboard reader arrowing across the first rows had the figures rendered below the fold —
 * #71's defect surviving its own fix, for the reader who had the least before it. The cell's native
 * `title` does not cover that case: it needs a pointer and a dwell, and this reader has neither.
 *
 * `sticky bottom-0` on an element already in flow, so it reflows nothing — which the sibling test
 * above pins from the other side.
 */
test('the readout stays on screen while a cell near the top of the grid is focused', async ({
  page,
}) => {
  /**
   * A laptop rather than `DESKTOP`, and the height is the whole reason.
   *
   * The grid measures 745px tall at this width, so at `DESKTOP`'s 900 the readout after it is on
   * screen anyway and the sticky placement is doing nothing to observe — the test would pass against
   * the unfixed markup. 600px is where the defect lives, which is also the range the finding named:
   * laptops and smaller.
   */
  const LAPTOP = { width: 1280, height: 600 };
  await page.setViewportSize(LAPTOP);
  await page.goto('/');
  await expect(grid(page)).toBeVisible();

  /**
   * Viewport coordinates, deliberately not `boxOf` — which adds `scrollY` back to give *document*
   * coordinates so the sibling reflow tests are not fooled by focus scrolling. This test asks the
   * opposite question: where the thing is relative to the screen right now.
   */
  const onScreen = (locator: Locator) =>
    locator.evaluate((el) => {
      const r = el.getBoundingClientRect();
      return { top: r.top, bottom: r.bottom, height: r.height };
    });

  // The state a keyboard reader is in immediately after tabbing into the grid: focused on the first
  // cell, with the top of the panel in view.
  const first = grid(page).locator('td button').first();
  await first.focus();
  await first.scrollIntoViewIfNeeded();

  // The premise, asserted rather than assumed: the grid is taller than the viewport, so there is
  // something for the sticky placement to do. Without it this passes on any short grid.
  const table = await onScreen(grid(page));
  expect(table.height, 'the grid fits the viewport, so this proves nothing').toBeGreaterThan(
    LAPTOP.height
  );

  const cell = await onScreen(first);
  const line = await onScreen(readout(page));

  // The focused cell is on screen…
  expect(cell.top).toBeGreaterThanOrEqual(-1);
  expect(cell.bottom).toBeLessThanOrEqual(LAPTOP.height + 1);
  // …and so is the sentence describing it, which is the whole claim.
  expect(
    line.bottom,
    'the readout is below the fold while the cell it describes is on screen'
  ).toBeLessThanOrEqual(LAPTOP.height + 1);
  expect(await readout(page).textContent()).toBeTruthy();
});

/**
 * The property that actually matters, and the one my first version of this got wrong.
 *
 * A readout whose height changes changes the Matrix section's height, and everything downstream of
 * the section moves with it — so "the readout is last in the panel, nothing follows it" was true
 * inside the panel and false on the page (found in review on #71). Being last saves the legend and
 * not what comes after the section.
 *
 * **What is downstream is not a fixed thing, so it is not what this measures.** The original canary
 * here was the Usage panel, which `Bench.tsx` then rendered immediately after `<Matrix>`. #66 moved
 * those five controls to the top of the page, *above* the grid — which would have left this test
 * asserting that a panel upstream of the readout does not move when the readout grows. That is true
 * of any markup at all, including markup with the reservation deleted: a passing geometry assertion
 * measuring nothing, which this repo has now had to unpick three times.
 *
 * So the canary is the two things that cannot stop being downstream of the readout: the **section's
 * own height**, which is where the growth would be, and the **document's height**, which is every
 * panel after it at once, whichever ones those turn out to be.
 */
for (const width of [320, 640]) {
  test(`a wrapped readout changes neither the section nor the page height, at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto('/');
    await expect(grid(page)).toBeVisible();

    const cells = grid(page).locator('td button');
    const pageHeight = () => page.evaluate(() => document.documentElement.scrollHeight);
    const before = { section: (await boxOf(matrix(page))).height, document: await pageHeight() };

    const at = async (index: number) => {
      await cells.nth(index).focus();
      await expect(readout(page)).not.toBeEmpty();
      return {
        line: (await boxOf(readout(page))).height,
        section: (await boxOf(matrix(page))).height,
        document: await pageHeight(),
      };
    };

    const a = await at(0);
    const b = await at((await cells.count()) - 1);

    /**
     * Equal heights here are the fix, not a weak premise — the reservation pads a three-line sentence
     * to the same box as a four-line one, which is the whole mechanism. So this cannot also be the
     * evidence that the sentences differ; that is the sweep below, which measures them against the
     * reservation and would fail if they all fitted one line. What this test owns is the consequence.
     */
    expect(a.line, 'the readout is not at its reserved height').toBeCloseTo(b.line, 0);
    expect(a.section).toBeCloseTo(before.section, 0);
    expect(b.section, 'the Matrix section grew when the readout did').toBeCloseTo(
      before.section,
      0
    );
    expect(a.document).toBeCloseTo(before.document, 0);
    expect(b.document, 'the page grew when the readout did').toBeCloseTo(before.document, 0);
  });
}

/**
 * And the reservation is a measured constant, so it is enforced rather than trusted.
 *
 * 80px at 320 and 40px at 640 come from the widest sentence today. #78 lengthened device names in
 * this same sweep ("MacBook Pro M1 Max (64 GB, 32-core GPU)"), and a length derived from text that
 * grows is the header band's #44 defect — so a longer name added later has to break a test rather
 * than the layout.
 *
 * **The measurement is the *natural* height, with the floor lifted, and the earlier version of this
 * could not fail** (raised in review on #108). `getBoundingClientRect().height` on a box with a
 * `min-height` already includes the floor, so every sentence that fits one line reported exactly the
 * reservation and satisfied `<= reserved` — including a run with the reservation deleted, where the
 * heights would simply have been smaller. `ROADMAP.md` credited this sweep with being the evidence
 * that the sentences differ in height, which is the one thing it was not measuring. Lifting
 * `min-height` and re-measuring is what asks the question, and the spread assertion below is the
 * evidence half stated outright rather than implied.
 */
for (const { width, reserved } of [
  { width: 320, reserved: 80 },
  { width: 640, reserved: 40 },
]) {
  test(`no sentence outgrows the space reserved for it at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto('/');
    await expect(grid(page)).toBeVisible();

    const cells = grid(page).locator('td button');
    const count = await cells.count();
    expect(count, 'no cells, so this sweeps nothing').toBeGreaterThan(100);

    const natural = async () =>
      readout(page).evaluate((el) => {
        const floor = el.style.minHeight;
        el.style.minHeight = '0px';
        const height = el.getBoundingClientRect().height;
        el.style.minHeight = floor;
        return height;
      });

    let tallest = { height: 0, text: '' };
    const heights = new Set<number>();
    for (let i = 0; i < count; i += 7) {
      await cells.nth(i).focus();
      const height = await natural();
      heights.add(height);
      if (height > tallest.height)
        tallest = { height, text: (await readout(page).textContent()) ?? '' };
    }

    expect(tallest.height, 'nothing was measured').toBeGreaterThan(0);
    /*
     * The evidence half: the sentences really do differ in height, so the bound above is a bound on
     * something. With the floor still applied they were all identical by construction, which is what
     * made the old version of this test unfalsifiable.
     */
    expect(
      heights.size,
      'every sentence renders the same height, so the reservation is what is being measured'
    ).toBeGreaterThan(1);
    expect(
      tallest.height,
      `"${tallest.text}" renders ${tallest.height}px against ${reserved}px reserved — raise the reservation for this breakpoint`
    ).toBeLessThanOrEqual(reserved);
  });
}
