import { expect, test } from '@playwright/test';

/**
 * How long the keyboard walk across this page actually is.
 *
 * The Matrix is 408 cells at the catalog #52 was measured against and 714 today, every one a
 * `<button>` with a full-sentence `aria-label`. With each cell in the tab sequence, crossing the page
 * took 422 presses of Tab, and a screen-reader user heard 408 sentences on the way.
 *
 * **jsdom cannot answer this.** It implements no sequential focus navigation at all: dispatching a
 * Tab keydown moves nothing, and `document.activeElement` stays where it was. So a unit test can
 * check the tab *sequence* as a DOM property — which `App.test.tsx` does, and which is the cheaper
 * check — but only a real browser can be asked where Tab goes. That is the same gap that shipped
 * the `display: contents` scroll-anchor bug this suite exists for: a property every unit test
 * agreed on, and no unit test could falsify.
 *
 * **The walk is measured across `<main>` rather than to a named panel**, which is a change #66 forced
 * and an improvement anyway. This used to count the presses to reach the Usage controls, because they
 * were the panel *after* the grid; #66 moved them to the top of the page, two stops in, so the
 * original bound would have been satisfied without the roving index doing anything at all. Counting
 * every stop inside `<main>` asks the same question of whatever the page's last panel happens to be:
 * 714 cells in the sequence blows any bound, one does not.
 *
 * Written as a bound rather than an exact count on purpose. The figure that matters is the order
 * of magnitude — nobody presses Tab 422 times — and pinning 23 exactly would fail the next time an
 * unrelated control is added, which is how a spec stops being read.
 */

/**
 * Tab from the top of the document and count the stops inside `<main>`.
 *
 * Counted from entry to exit rather than from the first press, so the masthead's own controls are not
 * in the figure and adding one there cannot move it. `Infinity` when the walk never gets out of
 * `<main>` inside `limit` presses, which is the pre-#52 result and the failure this reports.
 */
async function tabStopsInsideMain(
  page: import('@playwright/test').Page,
  limit: number
): Promise<number> {
  await page.evaluate(() => {
    (document.activeElement as HTMLElement | null)?.blur();
    document.body.focus();
  });

  let stops = 0;
  for (let presses = 1; presses <= limit; presses++) {
    await page.keyboard.press('Tab');
    const inside = await page.evaluate(() => {
      const main = document.querySelector('main');
      return main !== null && document.activeElement !== null
        ? main.contains(document.activeElement)
        : false;
    });
    if (inside) stops++;
    // Focus has been in `<main>` and is not any more, so the walk is over. Whether the browser hands
    // focus to its own chrome or wraps to the masthead, either lands outside `<main>`. Observed in
    // headless Chromium: the press after the grid's single cell puts focus on `<body>`, so the walk
    // terminates here rather than at the `limit`.
    else if (stops > 0) return stops;
  }
  return Number.POSITIVE_INFINITY;
}

test('the whole page is a short keyboard walk, grid included', async ({ page }) => {
  await page.goto('/');
  // The grid has to actually be there, or this measures a page without the problem on it.
  const cells = page.locator('table[role="grid"] td button');
  expect(await cells.count()).toBeGreaterThan(300);

  const stops = await tabStopsInsideMain(page, 80);

  // 23 as measured in this browser: nine controls (four selects, four sliders, and the KV group as
  // one stop), four disclosures, three legend keys, six measure buttons, and exactly one cell. 422
  // before the roving tabindex landed, which the 80-press ceiling never reaches — so that regression
  // reports Infinity.
  //
  // Where this figure and `App.test.tsx`'s 26 diverge, and the divergence is the point of running
  // both. Three apart, from two unrelated causes: a radio group offers Tab only its checked member,
  // so the three KV options are one stop here and three in a `querySelectorAll`, and the masthead's
  // copy-link button is a real stop this walk excludes because it sits outside `<main>`. The comment
  // here read "23 against 25 in jsdom" and attributed the whole gap to the radios, which was one
  // short and pointed at the wrong mechanism for the missing one. Compare a new measurement against
  // the number from the same channel, and against a stated reason for the difference.
  expect(stops).toBeLessThan(40);
  expect(stops, 'the walk never entered or never left <main>').toBeGreaterThan(0);
});

test('the whole comparison grid costs one press of Tab to cross', async ({ page }) => {
  await page.goto('/');
  const grid = page.locator('table[role="grid"]');

  // Land on the grid's single stop, then confirm one more Tab is already past every cell.
  await grid.locator('td button[tabindex="0"]').focus();
  await expect(grid.locator('td button:focus')).toHaveCount(1);

  await page.keyboard.press('Tab');

  const stillInGrid = await page.evaluate(() => {
    const table = document.querySelector('table[role="grid"]');
    return table !== null && document.activeElement !== null
      ? table.contains(document.activeElement)
      : false;
  });
  expect(stillInGrid).toBe(false);
});

/**
 * The other half of the pattern: what Tab no longer reaches, the arrow keys must.
 *
 * Asserted in a browser as well as in jsdom because the two answer different questions. jsdom
 * confirms the handler moves `document.activeElement`; only a real browser confirms the cell it
 * moves to is one the user can then act on, and that the grid's own `overflow-x-auto` scroller
 * follows focus instead of leaving it off-screen.
 */
test('the arrow keys move between cells, and the scroller follows', async ({ page }) => {
  await page.goto('/');
  const grid = page.locator('table[role="grid"]');
  const first = grid.locator('td button[tabindex="0"]').first();

  await first.focus();
  const firstLabel = await first.getAttribute('aria-label');

  await page.keyboard.press('ArrowRight');
  const afterRight = page.locator('table[role="grid"] td button:focus');
  await expect(afterRight).toHaveCount(1);
  expect(await afterRight.getAttribute('aria-label')).not.toBe(firstLabel);

  // Walk to the far end of the row and confirm focus is somewhere a user could actually click,
  // rather than clipped outside the horizontal scroller the grid lives in.
  await page.keyboard.press('Control+End');
  const last = page.locator('table[role="grid"] td button:focus');
  await expect(last).toBeInViewport();
});

test('the grid says how it is driven', async ({ page }) => {
  await page.goto('/');
  // The instruction has to reach a screen reader, which is the user the arrow keys are for.
  const caption = page.locator('table[role="grid"] caption');
  await expect(caption).toContainText(/single tab stop/i);
  await expect(caption).toContainText(/arrow keys/i);
});

/**
 * The left edge's half of focus-follows-scroll (#123).
 *
 * The model column is `sticky left-0` and opaque, and the browser's minimal focus-reveal aligns an
 * off-screen-left cell with the scrollport's *content* edge — directly underneath it. The cell, its
 * colour, and the focus ring — the keyboard reader's only positional mark — were all occluded, and
 * every further ArrowLeft kept them there. `Home` escaped by accident (column 1 forces scrollLeft
 * to 0), which is why a manual pass never caught it; and the existing Ctrl+End spec only ever
 * navigates *right*, where the reveal is unobstructed. The container's `scroll-padding-left` is
 * the fix, and this walks the direction the defect needs.
 */
test('a leftward walk never parks focus under the sticky model column', async ({ page }) => {
  // The defect's own viewport (#123 measured at 500px), and not decoration: at the desktop
  // default the scrollable range is ~175px, so 25 presses from Ctrl+End never reach a cell that
  // needs revealing and the loop verifies clearance nobody threatened — this spec passed against
  // the unfixed container at 1280px. At 500px the reveals start about seven presses in.
  await page.setViewportSize({ width: 500, height: 800 });
  await page.goto('/');
  const grid = page.locator('table[role="grid"]');
  await grid.locator('td button[tabindex="0"]').first().focus();
  await page.keyboard.press('Control+End');

  let revealedWhileScrolled = 0;
  for (let press = 1; press <= 25; press++) {
    await page.keyboard.press('ArrowLeft');
    const state = await page.evaluate(() => {
      const table = document.querySelector('table[role="grid"]')!;
      const scroller = table.closest('div')!;
      const focused = document.activeElement!;
      const sticky = focused.closest('tr')!.querySelector('th')!;
      return {
        scrollLeft: scroller.scrollLeft,
        cellLeft: focused.getBoundingClientRect().left,
        stickyRight: sticky.getBoundingClientRect().right,
      };
    });
    // Only a scrolled grid can occlude, so the clearance claim is made there — and the walk has
    // to actually reach that state, or the loop verifies nothing (vacuity guard below).
    if (state.scrollLeft > 0) {
      revealedWhileScrolled++;
      expect(state.cellLeft, `press ${press}`).toBeGreaterThanOrEqual(state.stickyRight - 1);
    }
  }
  expect(revealedWhileScrolled).toBeGreaterThan(0);
});
