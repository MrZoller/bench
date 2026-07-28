import { expect, test } from '@playwright/test';

/**
 * How far the keyboard actually is from the Usage controls.
 *
 * The Matrix is 408 cells, every one a `<button>` with a full-sentence `aria-label`, and it sits
 * above the Usage panel in DOM order. With each cell in the tab sequence, reaching the context
 * slider — which drives every figure on the page — took 422 presses of Tab, and a screen-reader
 * user heard 408 sentences on the way.
 *
 * **jsdom cannot answer this.** It implements no sequential focus navigation at all: dispatching a
 * Tab keydown moves nothing, and `document.activeElement` stays where it was. So a unit test can
 * check the tab *sequence* as a DOM property — which `App.test.tsx` does, and which is the cheaper
 * check — but only a real browser can be asked where Tab goes. That is the same gap that shipped
 * the `display: contents` scroll-anchor bug this suite exists for: a property every unit test
 * agreed on, and no unit test could falsify.
 *
 * Written as a bound rather than an exact count on purpose. The figure that matters is the order
 * of magnitude — nobody presses Tab 422 times — and pinning 15 exactly would fail the next time an
 * unrelated control is added, which is how a spec stops being read.
 */

/** Presses Tab until the focused element is inside `selector`, or gives up. */
async function tabsUntilInside(
  page: import('@playwright/test').Page,
  selector: string,
  limit: number
): Promise<number> {
  for (let presses = 1; presses <= limit; presses++) {
    await page.keyboard.press('Tab');
    const arrived = await page.evaluate((sel) => {
      const target = document.querySelector(sel);
      return target !== null && document.activeElement !== null
        ? target.contains(document.activeElement)
        : false;
    }, selector);
    if (arrived) return presses;
  }
  return Number.POSITIVE_INFINITY;
}

test('the Usage controls are a short walk from the top of the page', async ({ page }) => {
  await page.goto('/');
  // The grid has to actually be there, or this measures a page without the problem on it.
  const cells = page.locator('table[role="grid"] td button');
  expect(await cells.count()).toBeGreaterThan(300);

  await page.evaluate(() => {
    (document.activeElement as HTMLElement | null)?.blur();
    document.body.focus();
  });

  const presses = await tabsUntilInside(page, 'section[aria-label="Usage"]', 60);

  // 422 before the roving tabindex landed, which the 60-press ceiling would not have reached.
  expect(presses).toBeLessThan(40);
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
