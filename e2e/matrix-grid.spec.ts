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
 * of magnitude — nobody presses Tab 422 times — and pinning 25 exactly would fail the next time an
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
    // focus to its own chrome or wraps to the masthead, either lands outside `<main>`.
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

  // 25 as it stands: eleven controls (four selects, four sliders, three KV options), four
  // disclosures, three legend keys, six measure buttons, and exactly one cell. 422 before the roving
  // tabindex landed, which the 80-press ceiling never reaches — so that regression reports Infinity.
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
