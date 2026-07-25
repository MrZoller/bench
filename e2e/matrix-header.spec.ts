import { expect, test } from '@playwright/test';

/**
 * The Matrix's device headers are rotated 45 degrees and taken out of flow, so the row's height is
 * a *computed guess* — `ceil(longest * 6.5 * sin(45)) + 20`, where 6.5px per character is an
 * estimate at this type size rather than a measurement. The comment in the component says it errs
 * long on purpose, because the cost of erring short is a header that clips.
 *
 * jsdom cannot check that: it has no layout, so every rotated label is zero by zero there and the
 * estimate is unfalsifiable. Whether 6.5 is still right after a font, a type scale or a catalogue
 * change is exactly the kind of thing that breaks silently.
 */

test('the header row is tall enough for the longest rotated device label', async ({ page }) => {
  await page.goto('/');

  const headers = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('thead th');
  await expect(headers.first()).toBeVisible();

  const count = await headers.count();
  expect(count).toBeGreaterThan(1);

  const headerBox = await headers.nth(1).boundingBox();
  expect(headerBox).not.toBeNull();

  // Every rotated label has to sit inside the row it belongs to. The labels are absolutely
  // positioned and rotated, so their painted box is what matters, not their text length.
  for (let i = 1; i < count; i++) {
    const label = headers.nth(i).locator('span');
    const box = await label.boundingBox();
    expect(box, `header ${i} has no laid-out label`).not.toBeNull();

    // The rotation origin is the label's bottom-left, so it extends *upward* from the bottom of
    // the cell. Clipping shows up as the label's top escaping above the header row's top.
    expect(box!.y, `header ${i} label overflows the row`).toBeGreaterThanOrEqual(headerBox!.y - 1);
    expect(box!.height).toBeGreaterThan(0);
  }
});

/**
 * And that erring long has not turned into erring *absurdly* long — a header taking half the
 * viewport would be its own defect, and nothing else would catch it.
 */
test('does not reserve more header height than the labels need', async ({ page }) => {
  await page.goto('/');

  const headers = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('thead th');
  await expect(headers.first()).toBeVisible();

  const headerBox = await headers.nth(1).boundingBox();
  const tallest = await headers.evaluateAll((cells) =>
    Math.max(
      ...cells.map((cell) => {
        const span = cell.querySelector('span');
        return span ? span.getBoundingClientRect().height : 0;
      })
    )
  );

  expect(headerBox!.height).toBeGreaterThanOrEqual(tallest);

  /**
   * A *ratio*, not a fixed slack, and that distinction cost a round to see.
   *
   * The reservation is font-independent — a per-character estimate times a character count — while
   * `tallest` is font-dependent. `tallest + 60` therefore tightens whenever the platform's
   * `system-ui` renders narrower than macOS's SF: at 247px reserved against 198.9px measured here,
   * a font 10% narrower would have taken the bound to 239 and failed on a header that is fine.
   *
   * As a ratio it moves with the measurement instead. 1.24 here, ~1.38 against a font 10% narrower,
   * and 1.6 still catches the failure this guards: someone raising the per-character constant "to
   * be safe" and reserving half the viewport.
   */
  expect(headerBox!.height).toBeLessThan(tallest * 1.6);
});

/**
 * The reason the rotation exists at all: horizontally these clipped to "GeForc…" four times over,
 * and two Mac Studio variants differing only in a trailing capacity suffix became indistinguishable.
 * A header that cannot tell its own columns apart is worse than none.
 */
test('every device column is distinguishable by its label', async ({ page }) => {
  await page.goto('/');

  const labels = await page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('thead th span')
    .allInnerTexts();

  const visible = labels.filter((l) => l.trim().length > 0);
  expect(visible.length).toBeGreaterThan(1);
  expect(new Set(visible).size).toBe(visible.length);
  // Nothing truncated to an ellipsis by CSS.
  for (const label of visible) expect(label).not.toContain('…');
});
