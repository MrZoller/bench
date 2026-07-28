import { expect, test } from '@playwright/test';

/**
 * The bug this file exists for.
 *
 * The Matrix sits several sections below the detail it loads, so clicking a cell has to bring that
 * detail back into view. The anchor was first a `display: contents` element, which generates no
 * principal box — and `scrollIntoView` returns early for an element without one, so the scroll was
 * a silent no-op in every real browser. jsdom has no `scrollIntoView` at all, so the optional call
 * passed every test in the suite. The replacement is a zero-height box, and until this spec existed
 * it had been reasoned about rather than observed.
 */

const matrix = (page: import('@playwright/test').Page) =>
  page.getByRole('region', { name: /every model on every machine/i });

/**
 * The grid's own squares, not every button in the section.
 *
 * A region-wide `getByRole('button')` also picks up the three measure toggles above the grid, so
 * `nth(n)` addresses a cell `n - 3` positions from where the test says it does, and a fourth toggle
 * would shift every index silently. Scoped the same way `touch-targets.spec.ts` is — that file was
 * narrowed after a failure and this one was left, which is the exact "fix the instance, leave the
 * neighbour" pattern the roadmap keeps recording.
 */
const cells = (page: import('@playwright/test').Page) => matrix(page).locator('table td button');

/**
 * How far above the viewport top the anchor may sit and still count as scrolled to it.
 *
 * `block: 'start'` aims the anchor at y = 0, and on macOS it lands there exactly. On the Linux CI
 * runner it lands at **-0.5** — the zero-height anchor sits between a flex `gap-5` and a `-mb-5`
 * that cancels it, and the fractional layout that produces rounds the scroll offset a half pixel
 * past. The detail below it is fully visible; the scroll is correct.
 *
 * This is not the assertion being widened to make a red run green. What the spec distinguishes is a
 * scroll that happened from one that did not, and the failing case is not half a pixel — it is
 * **-1323px**, the anchor still down at the Matrix where the click was. Two pixels of tolerance
 * leaves that gap entirely intact, and restoring `display: contents` still fails this test.
 */
const SUBPIXEL = 2;

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  // The grid is the last section to lay out; waiting on a cell avoids racing the first paint.
  await expect(cells(page).first()).toBeVisible();
});

test('clicking a cell scrolls the detail it loads back into view', async ({ page }) => {
  const anchor = page.locator('#bench-detail');

  // Start at the grid, which is far enough down that the detail is off screen.
  await matrix(page).scrollIntoViewIfNeeded();
  await expect.poll(async () => (await anchor.boundingBox())?.y ?? 0).toBeLessThan(0);

  // Reduced motion, so the scroll is instant and the assertion does not race an animation. The
  // component reads this preference itself rather than leaving it to the stylesheet, because a
  // reduced-motion block neutralises CSS durations and has no effect on a scroll asked for in JS.
  await page.emulateMedia({ reducedMotion: 'reduce' });

  // Not the first cell: the top-left of the grid is the default selection on some viewports, and
  // the already-selected case has its own test below.
  await cells(page).nth(12).click();

  const box = await anchor.boundingBox();
  expect(box).not.toBeNull();
  // At the top of the viewport — `block: 'start'` — rather than merely somewhere in it.
  expect(box!.y).toBeGreaterThan(-SUBPIXEL);
  expect(box!.y).toBeLessThan(page.viewportSize()!.height / 2);
});

/**
 * The case the scroll was added for. A cell that already matches the selection changes nothing the
 * Matrix renders, so without the scroll the click was indistinguishable from not registering.
 */
test('scrolls even when the cell clicked is the one already selected', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });

  // `aria-current`, not `getByRole('button', { selected })` — `selected` is only defined for
  // gridcell, option, row, tab and friends, and Playwright throws rather than returning nothing.
  const selected = matrix(page).locator('table td button[aria-current="true"]');
  await expect(selected.first()).toBeAttached();

  await matrix(page).scrollIntoViewIfNeeded();
  const anchor = page.locator('#bench-detail');
  await expect.poll(async () => (await anchor.boundingBox())?.y ?? 0).toBeLessThan(0);

  await selected.first().click();

  const box = await anchor.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.y).toBeGreaterThan(-SUBPIXEL);
  expect(box!.y).toBeLessThan(page.viewportSize()!.height / 2);
});

/**
 * The anchor has to generate a principal box, which is the property that was wrong. Asserted
 * directly as well as through its effect, because the effect could pass for an unrelated reason —
 * a browser that happens to scroll the focused button into view, say.
 */
test('the scroll anchor generates a box for scrollIntoView to find', async ({ page }) => {
  const anchor = page.locator('#bench-detail');

  await expect(anchor).toBeAttached();
  expect(await anchor.evaluate((el) => getComputedStyle(el).display)).not.toBe('contents');
  // Zero height is deliberate — it costs no layout — but it is still a box.
  expect(await anchor.boundingBox()).not.toBeNull();
});
