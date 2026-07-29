import { expect, test, type Page } from '@playwright/test';

/**
 * The Configuration panel's geometry, once a picker note stops being a page of prose. Issue #68.
 *
 * The Hardware note was `[statusWarning, ceilingClause, row.note].join(' ')` — up to 180 words of
 * catalog provenance under a `<select>` at `text-xs`. The issue's own "side effect worth noting" is
 * what that does to a two-column grid: the Hardware cell sets the height of the row it shares with
 * Model, so a note that wraps to ten or eleven lines pushes the Quantization/Runtime row down and
 * leaves the same amount of empty space under Model. The prose now lives behind a disclosure.
 *
 * **Why this is here and not in Vitest.** Every number below is a line box or a laid-out rectangle:
 * jsdom has no layout engine, `getClientRects()` returns nothing, and every assertion would be a
 * tautology. Whether the derivation is *reachable* and whether it is in `aria-describedby` are DOM
 * questions and are asserted in `src/App.test.tsx`; how many lines it occupies is not.
 *
 * The assertions are deliberately relative — a line count, and a before/after on the same page — so
 * they state the property rather than a font metric. An absolute pixel budget would be measuring
 * Chrome's text shaping.
 */

/** Wide enough that the two-column grid is the layout under test, matching the issue's screenshot. */
const DESKTOP = { width: 1440, height: 900 };

/**
 * A picker note is picker copy: a claim you choose by, on one line, or two if the machine has both
 * a status warning and a raiseable ceiling. Anything above this is reference prose that has found
 * its way back into the control.
 */
const MAX_NOTE_LINES = 2;

/** Every option the Hardware picker offers, so the sweep covers rows nobody thought to name. */
async function deviceIds(page: Page): Promise<string[]> {
  return page
    .getByLabel('Hardware', { exact: true })
    .locator('option')
    .evaluateAll((options) => options.map((o) => (o as HTMLOptionElement).value));
}

/**
 * A control's cell and note, measured in the page.
 *
 * Resolved by walking from the visible label rather than with a CSS selector, because the ids are
 * `useId`'s and contain characters a selector would have to escape. The note is found through
 * `aria-describedby`, which is also the assertion `App.test.tsx` makes about it: the note the
 * reader sees and the description a screen reader resolves are the same element.
 */
async function panelGeometry(page: Page) {
  return page.evaluate(() => {
    const cellFor = (labelText: string) => {
      const label = Array.from(document.querySelectorAll('label')).find(
        (l) => l.textContent?.trim() === labelText
      );
      if (!label) throw new Error(`no control labelled ${labelText}`);
      const select = document.getElementById(label.htmlFor) as HTMLSelectElement | null;
      if (!select) throw new Error(`${labelText} has no select`);

      const noteId = select.getAttribute('aria-describedby');
      const note = noteId === null ? null : document.getElementById(noteId);
      const lines = (() => {
        if (note === null) return 0;
        const range = document.createRange();
        range.selectNodeContents(note);
        // One rect per line box. The same technique the Envelope's title assertions use, and the
        // only way to count wrapped lines without hard-coding a line height.
        return range.getClientRects().length;
      })();

      // The grid item, which `align-items: stretch` sizes to the whole track — so its height is the
      // row's height and its bottom is where the next row begins.
      const cell = select.parentElement!;
      return {
        cell: cell.getBoundingClientRect().toJSON(),
        /**
         * Where this cell's own content stops, which is not where its box stops.
         *
         * Read off the last element in the stack rather than off the note, because not every
         * control has one — a model with no download count carries no note at all, and reaching
         * for `note.bottom` would make this file fail for a reason that has nothing to do with it.
         */
        contentBottom: cell.lastElementChild!.getBoundingClientRect().bottom,
        lines,
        // From the label, which every cell has and which is `text-xs` like the notes. A line is the
        // unit the budgets below are expressed in, so it must not depend on an optional element.
        lineHeight: parseFloat(getComputedStyle(cell.firstElementChild!).lineHeight),
      };
    };

    const panel = document
      .querySelector('section[aria-label="Configuration"]')!
      .getBoundingClientRect();

    return {
      panel: panel.toJSON(),
      model: cellFor('Model'),
      hardware: cellFor('Hardware'),
      quantization: cellFor('Quantization'),
    };
  });
}

const showFullNote = (page: Page) =>
  page.getByRole('button', { name: /show the full hardware note/i });

test.beforeEach(async ({ page }) => {
  await page.setViewportSize(DESKTOP);
  await page.goto('/');
  await expect(page.getByLabel('Hardware', { exact: true })).toBeVisible();

  // The precondition the whole file rests on: `sm:grid-cols-2` is what makes one cell's height
  // another cell's void, and below `sm` the panel stacks and there is nothing here to measure.
  expect(
    await page.evaluate(() => matchMedia('(min-width: 40rem)').matches),
    'this viewport is below sm, so the Configuration panel is not the two-column layout'
  ).toBe(true);
});

/**
 * The sweep, not the row the issue named. Nine rows composed more than one fragment and the issue
 * listed seven of them, one of which was not affected at all — so the shape of the bug is "the row
 * nobody checked".
 */
test('no Hardware note runs past two lines, on any row in the catalog', async ({ page }) => {
  const ids = await deviceIds(page);
  expect(
    ids.length,
    'the picker offered no options, so this sweep measured nothing'
  ).toBeGreaterThan(20);

  const overflowing: string[] = [];
  for (const id of ids) {
    await page.getByLabel('Hardware', { exact: true }).selectOption(id);
    const { hardware } = await panelGeometry(page);
    if (hardware.lines > MAX_NOTE_LINES) overflowing.push(`${id} (${hardware.lines} lines)`);
  }

  // Before the split this was every row with a curated note: the M5 Ultra's 146 words wrapped to
  // eleven lines in a 540px column, and the shortest note in the catalog is still 25 words.
  expect(overflowing, 'Hardware notes wrapping past a claim into reference prose').toEqual([]);
});

/**
 * The void, measured as the issue describes it: the space under Model that the Hardware cell's own
 * height creates, since they share a grid row.
 */
test('the Hardware cell no longer sets the height of the row Model is in', async ({ page }) => {
  await page.getByLabel('Hardware', { exact: true }).selectOption('mac-studio-m5-ultra-512');

  const closed = await panelGeometry(page);
  const voidUnderModel = closed.model.cell.bottom - closed.model.contentBottom;
  const line = closed.model.lineHeight;
  expect(line, 'no computed line height, so the budget below is meaningless').toBeGreaterThan(0);

  // Five lines of slack covers what the Hardware cell legitimately carries beyond what Model does:
  // a claim of one or two lines, plus the disclosure button and its margin. The unfixed layout put
  // ten or eleven lines of prose in that cell, so the void was more than twice this.
  expect(
    voidUnderModel,
    `${Math.round(voidUnderModel)}px of empty space under Model, at a ${line}px line`
  ).toBeLessThan(line * 5);

  // And the row above the Quantization/Runtime row is the thing that was pushing it down, so the
  // panel is measured as a whole too: opening the disclosure is the only way to reach that height.
  await showFullNote(page).click();
  const open = await panelGeometry(page);

  expect(
    open.panel.height - closed.panel.height,
    'opening the disclosure adds no height, so the prose is not really in there'
  ).toBeGreaterThan(80);
  expect(
    open.quantization.cell.top,
    'the Quantization row did not move, so the two rows are not stacked as this test assumes'
  ).toBeGreaterThan(closed.quantization.cell.top);
});

/**
 * The disclosure is a control, so it is a target — and it sits inside a panel of them.
 *
 * `touch-targets.spec.ts` sweeps every pointer target on the page and would catch a 16px button,
 * but only on the touch project and only while the disclosure is *rendered*: it is the Hardware
 * picker's, and the Hardware picker's default row has a note. This asserts it is reachable and
 * labelled for the control it belongs to, which is what stops a page of identical "Show more"
 * buttons the moment a second picker grows one.
 */
test('the disclosure names the control it belongs to', async ({ page }) => {
  await page.getByLabel('Hardware', { exact: true }).selectOption('rtx-3090');

  const toggle = showFullNote(page);
  await expect(toggle).toBeVisible();

  await toggle.click();
  // The 3090's caveat, which was dropped from the picker entirely once before — the estimates
  // assume PCIe and do not model its optional NVLink bridge.
  await expect(page.getByText(/NVLink bridge/i)).toBeVisible();
  await expect(page.getByRole('button', { name: /hide the full hardware note/i })).toBeVisible();
});
