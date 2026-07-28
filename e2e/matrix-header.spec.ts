import { expect, test } from '@playwright/test';

/**
 * The Matrix's device headers are rotated 45 degrees and taken out of flow, so the row's height is
 * a *computed guess* — `ceil(longest * 8 * sin(45)) + 20`, where 8px per character is an
 * estimate at this type size rather than a measurement. The comment in the component says it errs
 * long on purpose, because the cost of erring short is a header that clips.
 *
 * jsdom cannot check that: it has no layout, so every rotated label is zero by zero there and the
 * estimate is unfalsifiable. Whether 6.5 is still right after a font, a type scale or a catalogue
 * change is exactly the kind of thing that breaks silently.
 */

/**
 * At the default root, and again at 200% text.
 *
 * The second case is the one that was broken (#44). The labels are `text-xs`, so their width — and
 * therefore the height their rotation needs — scales with the root font size, while the reserved
 * height was a character count times a constant in *CSS pixels*. At a 32px root the text doubled
 * and the row it has to fit in did not, so the `overflow-x-auto` wrapper clipped the device names:
 * precisely the failure the rotation was introduced to prevent, reappearing at the one setting a
 * low-vision reader would be using. `headerHeight` is in `rem` now.
 *
 * Parameterised rather than copied, because the first version of this check existed only at the
 * default root and passed throughout — the estimate was never wrong at 16px.
 */
for (const root of [16, 32]) {
  test(`the header row is tall enough for the longest rotated label at a ${root}px root`, async ({
    page,
  }) => {
    await page.goto('/');
    if (root !== 16) {
      await page.evaluate((px) => {
        document.documentElement.style.fontSize = `${px}px`;
      }, root);
      // Confirms the scaling landed before anything is measured against it — otherwise the 32px
      // case silently re-runs the 16px one and passes for the wrong reason.
      await page.waitForFunction(
        (px) => getComputedStyle(document.documentElement).fontSize === `${px}px`,
        root
      );
    }

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
      expect(box!.y, `header ${i} label overflows the row`).toBeGreaterThanOrEqual(
        headerBox!.y - 1
      );
      expect(box!.height).toBeGreaterThan(0);
    }
  });
}

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

  const headers = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('thead th');
  await expect(headers.first()).toBeVisible();

  // The first cell is the row-header stub, and its only span is `sr-only` — deliberately clipped to
  // a 1px box, so it would read as truncated on the geometry check below.
  const labels = await headers.evaluateAll((cells) =>
    cells.slice(1).map((cell) => {
      const span = cell.querySelector('span');
      if (!span) return { text: '', display: 'none', clipped: false };
      return {
        text: span.textContent ?? '',
        /**
         * Both properties return 0 on an element with no layout box, and Blink gives an inline box
         * no `clientWidth` at all — so `0 > 0 + 1` is false and the clipping check below silently
         * becomes a tautology. That is not a hypothetical shape: the regression this test guards
         * against is a revert to truncated in-flow headers, which is exactly what would put the
         * label back to `display: inline`. Asserted so the measurement fails loudly rather than
         * degrading into one that cannot fail — the defect this whole test was rewritten to escape.
         */
        display: getComputedStyle(span).display,
        /**
         * Clipping has to be read off the layout, not the text. `text-overflow: ellipsis` paints a
         * glyph the browser never writes back into the DOM, so `textContent` stays whole while the
         * column becomes unreadable — the assertion this replaces could not fail.
         *
         * A 1px tolerance because both properties are integers: a shrink-to-fit box whose content
         * measures 198.6px can round its scroll and client widths apart by one. Real truncation
         * overflows by a whole character, 6–8px at this type size.
         */
        clipped: span.scrollWidth > span.clientWidth + 1,
      };
    })
  );

  const visible = labels.filter((l) => l.text.trim().length > 0);
  expect(visible.length).toBeGreaterThan(1);
  expect(new Set(visible.map((l) => l.text)).size).toBe(visible.length);
  for (const label of visible) {
    expect(
      label.display,
      `the header "${label.text}" is inline, so the clipping check cannot measure it`
    ).not.toBe('inline');
    expect(label.clipped, `the header "${label.text}" is clipped by its own box`).toBe(false);
  }
});
