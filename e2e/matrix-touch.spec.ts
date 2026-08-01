import { expect, test, type Page } from '@playwright/test';

/**
 * Inspecting a cell with a finger, in a browser that actually has one
 * ([#102](https://github.com/MrZoller/bench/issues/102)).
 *
 * The Matrix's readout reaches a mouse and a keyboard, and until this it could not reach touch: the
 * only gesture a cell offers is a tap, and that tap *is* `onClick` — five store keys rewritten and a
 * scroll several sections away — so the line either filled while navigation was already happening or
 * never filled at all. Unlike the Envelope and the budget bar, this panel has no table behind a
 * disclosure to fall back to; it *is* the table, and its cells show colour. So the first tap inspects
 * and the second commits.
 *
 * **Why this is here and not only in `App.test.tsx`.** The guard turns on an event *order* the
 * browser produces and a fixture does not: tapping a button focuses it, so `onFocus` runs before
 * `click` and the readout is already this cell's by the time the handler is reached. The first
 * version of the fix compared against the live target and therefore never fired — the first tap
 * committed — and the unit test passed, because `fireEvent.pointerDown` plus `fireEvent.click`
 * interposes no focus and React had not committed one anyway. The unit test now forces both and does
 * bite; this is the check that does not depend on my having modelled the sequence correctly.
 *
 * Same argument as the `scrollIntoView` on a `display: contents` anchor: a guarded call that jsdom
 * could not execute passed every test while returning early in every real browser.
 */

const matrix = (page: Page) => page.getByRole('region', { name: /every model on every machine/i });
const cells = (page: Page) => matrix(page).locator('table[role="grid"] td button');
/** The reserved line: the section's only direct paragraph — see `matrix-readout.spec.ts`. */
const readout = (page: Page) => matrix(page).locator(':scope > p');

/** The scenario the Bench is holding, read off the querystring the store round-trips into. */
const scenario = (page: Page) => new URL(page.url()).search;

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(cells(page).first()).toBeVisible();
});

test('the run really is a touch one', async ({ page }) => {
  /*
   * Asserted before anything else measures a tap, for the reason `touch-targets.spec.ts` asserts its
   * own pointer query first: this project is an emulation, and a change in how Playwright emulates a
   * device would silently move every assertion below onto the mouse branch, where they all pass.
   */
  expect(await page.evaluate(() => matchMedia('(any-pointer: coarse)').matches)).toBe(true);
  expect(await page.evaluate(() => navigator.maxTouchPoints)).toBeGreaterThan(0);
});

test('the first tap fills the readout without loading the cell', async ({ page }) => {
  const before = scenario(page);
  // Any cell but the one already marked, so "the scenario did not change" is a claim about the tap
  // rather than about it having been the current cell all along.
  const cell = cells(page).nth(5);
  const said = await cell.getAttribute('aria-label');

  await cell.tap();

  // The figures, where a touch reader can read them — which is the whole of what #71 left open.
  await expect(readout(page)).toContainText((said ?? '').split(':')[1].trim().slice(0, 20));
  expect(scenario(page), 'the first tap loaded the cell').toBe(before);
});

test('the second tap on the same cell loads it', async ({ page }) => {
  const cell = cells(page).nth(5);

  await cell.tap();
  await cell.tap();

  // The commit, which is the gesture's other half — inspection is only separated from activation if
  // activation is still reachable.
  expect(scenario(page), 'the second tap did not load the cell').not.toBe('');
  await expect(page).toHaveURL(/[?&]d=/);
});

test('the keyboard still activates after a tap', async ({ page }) => {
  /*
   * The snapshot is consumed on every click, and this is why (found in review). Held, it would still
   * describe the tap when a later keyboard `click` arrived with no `pointerdown` of its own — so
   * Enter would be read as that tap still in progress and refuse to activate, for ever, since
   * nothing else would clear it. A touch device with a keyboard is an ordinary tablet.
   */
  const before = scenario(page);
  await cells(page).nth(5).tap();

  await cells(page).nth(9).focus();
  await page.keyboard.press('Enter');

  expect(scenario(page), 'Enter after a tap did not load the cell').not.toBe(before);
});

test('a tap on a different cell moves the line rather than committing', async ({ page }) => {
  const before = scenario(page);
  const [first, second] = [cells(page).nth(5), cells(page).nth(9)];
  const secondSaid = await second.getAttribute('aria-label');

  await first.tap();
  await second.tap();

  /*
   * Comparing two cells is the thing #71 named a touch reader as unable to do, and it is also the
   * case the first version of this fix broke: with the guard reading the live readout target, the
   * browser's own focus had already made the second cell the target before `click`, so the second
   * tap committed rather than inspecting.
   */
  await expect(readout(page)).toContainText((secondSaid ?? '').split(':')[1].trim().slice(0, 20));
  expect(scenario(page), 'tapping a second cell loaded it').toBe(before);
});
