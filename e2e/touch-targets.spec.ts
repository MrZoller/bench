import { expect, test } from '@playwright/test';
import { marks } from '@/design/tokens';

/**
 * `@media (pointer: coarse)` cannot be forced with a viewport size, so the branch it gates is
 * invisible to jsdom and to any desktop browser run. This project emulates a real touch device.
 *
 * What it guards is a claim the repo makes about itself: `marks.hitTarget` declares 44px, the
 * Matrix's 28px squares sit two pixels apart, and with hundreds of neighbours a touch user hitting
 * the wrong scenario is the likely outcome rather than the unlucky one. The coarse-pointer rules
 * that fix it are three utility classes, and nothing else can tell whether they still apply.
 */

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('the run really is a coarse-pointer one', async ({ page }) => {
  // Asserted first and on its own, so a change in how Playwright emulates touch fails here rather
  // than silently letting every size assertion below measure the mouse branch and pass.
  const coarse = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
  expect(coarse).toBe(true);
});

test('Matrix cells meet the hit target this repo declares', async ({ page }) => {
  // Scoped to the table, not the whole region: the measure toggles above the grid are buttons in
  // the same section and are deliberately not on this rule, so a region-wide locator measures a
  // control the coarse-pointer branch was never written for and fails on the wrong thing.
  const cells = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('table td button');
  await expect(cells.first()).toBeVisible();

  // A sample rather than all of them: the rule is one CSS class on every cell, so a handful spread
  // across the grid proves it applies without paying for hundreds of round trips.
  const total = await cells.count();
  for (const index of [0, Math.floor(total / 2), total - 1]) {
    const box = await cells.nth(index).boundingBox();
    expect(box, `cell ${index} is not laid out`).not.toBeNull();
    expect(box!.height, `cell ${index} height`).toBeGreaterThanOrEqual(marks.hitTarget);
    expect(box!.width, `cell ${index} width`).toBeGreaterThanOrEqual(marks.hitTarget);
  }
});

/**
 * The Matrix's own controls are *not* on the 44px rule, and that is deliberate rather than an
 * omission this spec should assert away: the grid earns it because its squares sit two pixels apart
 * with hundreds of neighbours, where a mis-hit loads the wrong scenario silently. A row of three
 * labelled toggles is not that situation.
 *
 * What they do have to clear is the 24px WCAG 2.5.8 floor, with no crowding exception available —
 * so that is what is asserted. Written down because the first version of this file measured them
 * against 44 and failed, which is a claim the app never made.
 */
const WCAG_MINIMUM_TARGET = 24;

test('the Matrix controls clear the accessibility floor, if not the grid’s bar', async ({
  page,
}) => {
  for (const name of [/does it fit/i, /how fast/i, /how responsive/i]) {
    const control = page.getByRole('button', { name }).first();
    await expect(control).toBeVisible();
    const box = await control.boundingBox();
    expect(box, `${String(name)} is not laid out`).not.toBeNull();
    expect(box!.height, `${String(name)} height`).toBeGreaterThanOrEqual(WCAG_MINIMUM_TARGET);
    expect(box!.width, `${String(name)} width`).toBeGreaterThanOrEqual(WCAG_MINIMUM_TARGET);
  }
});
