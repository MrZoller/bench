import { expect, test, type Page } from '@playwright/test';
import { marks } from '@/design/tokens';

/**
 * `@media (pointer: coarse)` cannot be forced with a viewport size, so the branch it gates is
 * invisible to jsdom and to any desktop browser run. This project emulates a real touch device.
 *
 * What it guards is a claim the repo makes about itself: `marks.hitTarget` declares 44px, the
 * Matrix's 28px squares sit two pixels apart, and with hundreds of neighbours a touch user hitting
 * the wrong scenario is the likely outcome rather than the unlucky one. The coarse-pointer rules
 * that fix it are three utility classes, and nothing else can tell whether they still apply.
 *
 * **It sweeps rather than names.** It used to measure the three Matrix controls it knew about,
 * which is how three 16px buttons on other surfaces went unnoticed until someone looked (#29).
 * The sweep below measures every pointer target on the page and fails on any that is too small, so
 * a control added later is covered by default instead of by remembering to add it here.
 */

/**
 * WCAG 2.5.8 (AA). The floor every pointer target has to clear.
 *
 * Distinct from `marks.hitTarget`, and the difference is deliberate rather than an inconsistency
 * to tidy away: 24px is the standard's minimum, 44px is the stricter bar this repo declares for
 * targets that are *crowded* or that are the only route to an accessibility affordance. Holding
 * everything to 44 would fail controls the app never claimed it for.
 */
const WCAG_MINIMUM_TARGET = 24;

/**
 * Things that are focusable but are not pointer targets, with the reason each is exempt.
 *
 * Kept as data rather than as a filter buried in the query, because an exception nobody can see is
 * how a real failure gets classified as expected. Each entry is asserted to still match something
 * below — a stale exemption is itself a bug, and this is the file where it would rot unnoticed.
 */
const NOT_POINTER_TARGETS = [
  {
    selector: 'li[tabindex="0"]',
    why:
      'The budget legend cross-highlights its series on hover, and `tabIndex` is there to give ' +
      'that hover affordance a keyboard equivalent. Nothing is activated by pointing at one — ' +
      'there is no click handler — so 2.5.8 does not reach them.',
  },
] as const;

/**
 * Every candidate target, with the sr-only inputs resolved to the label that actually receives
 * the tap.
 *
 * The radios inside the segmented controls measure 1x1 because they are `sr-only` — they have to
 * stay focusable for arrow-key navigation, so they are hidden by clipping rather than removed
 * from the layout. Measuring the input is measuring the wrong box: the user taps the label. A
 * sweep that did not resolve this would report three impossible-to-hit 1px targets and be
 * disabled within a week for crying wolf.
 */
async function sweep(page: Page) {
  return page.evaluate(
    ({ exempt }) => {
      const candidates = Array.from(
        document.querySelectorAll<HTMLElement>(
          'button, a[href], select, input, [role="button"], [tabindex]:not([tabindex="-1"])'
        )
      );

      const targets: { label: string; width: number; height: number; inGrid: boolean }[] = [];
      const seen = new Set<HTMLElement>();

      for (const el of candidates) {
        if (exempt.some((s) => el.matches(s))) continue;

        const style = getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;

        // A visually-hidden control is tapped through its label.
        const box = el.getBoundingClientRect();
        const target = box.width < 2 || box.height < 2 ? (el.closest('label') ?? el) : el;
        if (seen.has(target)) continue;
        seen.add(target);

        // A zero-size box is *reported*, not skipped. Skipping it is the tempting reading — it
        // looks like something that is not really on screen — but `display: none` and
        // `visibility: hidden` are already handled above, so what is left is a laid-out control
        // collapsed to nothing, which is the worst version of the defect this file exists for.
        const rect = target.getBoundingClientRect();

        targets.push({
          label:
            `<${target.tagName.toLowerCase()}> ` +
            `"${(target.textContent ?? '').trim().slice(0, 40) || target.getAttribute('aria-label')?.slice(0, 40) || '(no text)'}"`,
          width: rect.width,
          height: rect.height,
          inGrid: !!target.closest('table'),
        });
      }
      return targets;
    },
    { exempt: NOT_POINTER_TARGETS.map((e) => e.selector) }
  );
}

/**
 * Both disclosures are behind their own toggle, so half the page's controls do not exist until
 * something opens them — including the Envelope's table, which is the surface the issue cared
 * most about. A sweep run on the initial page measures what it can see and reports a clean bill
 * for the rest.
 */
async function revealEverything(page: Page) {
  for (const name of [
    /show figures as a table/i,
    /show the region as a table/i,
    /show what each workload means/i,
  ]) {
    const toggle = page.getByRole('button', { name });
    await expect(toggle, `nothing matched ${String(name)}`).toHaveCount(1);
    await toggle.click();
  }
}

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
 * The columns the sample above cannot reach, and which are the ones at risk (#79).
 *
 * A column that opens a class band carries an 8px border in the panel's colour, which is how the
 * catalog's grouping is shown on the grid. The column is widened by exactly that border so the square
 * inside keeps its size — and if it were not, the coarse-pointer branch would hand a touch user a
 * 36px target, since `w-11` sets the *cell* and the border is inside it. The three sampled indices are
 * 0, 357 and 713, and the two band starts are 25 and 37: the failure would sit in the gaps of the
 * existing sweep, which is the shape #29 was filed about in the first place.
 *
 * Located by the class rather than by index, so this measures whatever columns the catalog actually
 * bands rather than the two it bands today.
 */
test('a column that opens a class band keeps the full hit target, gap and all', async ({
  page,
}) => {
  const banded = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('table td.border-l-8 button');

  const count = await banded.count();
  expect(count, 'no column opens a class band, so this measures nothing').toBeGreaterThan(0);

  for (let i = 0; i < count; i++) {
    const box = await banded.nth(i).boundingBox();
    expect(box, `band-start cell ${i} is not laid out`).not.toBeNull();
    expect(box!.height, `band-start cell ${i} height`).toBeGreaterThanOrEqual(marks.hitTarget);
    expect(box!.width, `band-start cell ${i} width`).toBeGreaterThanOrEqual(marks.hitTarget);
  }
});

/**
 * The three "show this as a table" toggles, which #29 found at 16px.
 *
 * Held to 44 rather than to the 24px floor because two of them are the only route to the textual
 * equivalent of a canvas — the Envelope's table is how a screen-reader or low-vision user reads
 * the field at all. That is the same bar the grid holds, so it reads `marks.hitTarget` rather
 * than a literal 44 — the token is what the app declares, and a spec asserting its own copy of the
 * number is a spec that will one day disagree with the CSS it is checking.
 */
test('the table disclosures meet the hit target, being the accessibility affordance', async ({
  page,
}) => {
  for (const name of [
    /show figures as a table/i,
    /show the region as a table/i,
    /show what each workload means/i,
  ]) {
    const toggle = page.getByRole('button', { name });
    await expect(toggle, `nothing matched ${String(name)}`).toHaveCount(1);
    const box = await toggle.boundingBox();
    expect(box, `${String(name)} is not laid out`).not.toBeNull();
    expect(box!.height, `${String(name)} height`).toBeGreaterThanOrEqual(marks.hitTarget);
    expect(box!.width, `${String(name)} width`).toBeGreaterThanOrEqual(marks.hitTarget);
  }
});

/**
 * The sweep. Every pointer target on the page, in both disclosure states.
 *
 * The Matrix's own measure toggles are the reason this asserts 24 and not 44: a row of three
 * labelled controls with room around them is not the crowded situation the grid's squares are in,
 * and holding them to 44 would fail a claim the app never made. Written down because the first
 * version of this file measured them against 44 and failed on exactly that.
 */
test('every pointer target clears the WCAG 2.5.8 floor', async ({ page }) => {
  await revealEverything(page);

  const targets = await sweep(page);

  /**
   * Guards the sweep itself: an empty or partial result set makes every assertion below vacuous,
   * which is the failure mode this suite has produced three of.
   *
   * Derived from the page rather than written as a literal. A hardcoded floor near today's 408
   * cells would fail the week the catalog refresh drops a model — a false alarm that teaches
   * people to raise the number rather than read it — and one set far below it would stop catching
   * anything. Counting the grid's own buttons holds regardless of how big the catalog gets.
   */
  const gridCells = await page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('table td button')
    .count();

  expect(gridCells, 'the grid rendered no cells').toBeGreaterThan(0);
  expect(targets.filter((t) => t.inGrid).length, 'the sweep missed grid cells').toBe(gridCells);
  expect(
    targets.filter((t) => !t.inGrid).length,
    'the sweep found no controls outside the grid'
  ).toBeGreaterThan(10);

  const tooSmall = targets
    .filter((t) => t.height < WCAG_MINIMUM_TARGET || t.width < WCAG_MINIMUM_TARGET)
    .map((t) => `${t.label} is ${Math.round(t.width)}x${Math.round(t.height)}`);

  expect(tooSmall, `below the ${WCAG_MINIMUM_TARGET}px floor`).toEqual([]);
});

/**
 * And that every exemption above still describes something real.
 *
 * An exception list is the one part of a sweep that fails open: delete the element it was written
 * for and the entry stops excluding anything, so it sits there looking like a considered decision
 * while quietly covering whatever matches next.
 */
test('each documented exemption still matches an element', async ({ page }) => {
  for (const { selector, why } of NOT_POINTER_TARGETS) {
    await expect(page.locator(selector), `stale exemption — ${why}`).not.toHaveCount(0);
  }
});
