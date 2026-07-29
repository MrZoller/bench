import { expect, test, type Page } from '@playwright/test';

/**
 * The workload strip's three columns, measured across all seven rows. Issue #70.
 *
 * The strip reads as a table, and it used to compute its column widths seven separate times: the
 * grid was declared on each `<li>`, so every row was its own grid container and the middle `auto`
 * track was sized from that row's own label. Columns 1 and 2 lined up anyway — the first track is
 * fixed at `9rem` — while the third, the written reason, started anywhere from 444px to 503px at
 * 1440px. That third column is the panel's argument: seven archetypes get seven different answers
 * from the same hardware and the reasons are what explain the differences, so they have to be
 * scannable against each other.
 *
 * Why this is here and not in Vitest: jsdom has no layout engine, so every offset below reads back
 * as 0 and the equality assertions are tautologies there — the blind spot this directory exists
 * for. The mutation check is the same shape: restoring the three tracks to the `<li>` fails
 * `the reason column starts at the same x on every row` by 59px and nothing else.
 */

/** The issue's own measuring width. */
const DESKTOP = { width: 1440, height: 900 };

/**
 * The tightest viewport the three-column layout is reachable at — `sm` is 40rem, so 640px matches
 * it exactly. Included because the columns are shared by an intrinsic track, and an intrinsic track
 * is where a layout that only holds when there is slack gives itself away.
 */
const AT_SM = { width: 640, height: 900 };

/** Below `sm`, where the row deliberately stacks and keeps a grid of its own. */
const PHONE = { width: 390, height: 844 };

const strip = (page: Page) => page.getByRole('region', { name: /what you could do with it/i });

/**
 * The three cells of every row, as rectangles, in DOM order — status word, label, reason.
 *
 * Read in one `evaluateAll` rather than through seven locators, because a viewport change between
 * two measurements would compare two layouts and report the difference as rag.
 */
async function rows(page: Page) {
  return strip(page)
    .locator('li')
    .evaluateAll((items) =>
      items.map((li) => ({
        text: (li.textContent ?? '').trim(),
        left: li.getBoundingClientRect().left,
        width: li.getBoundingClientRect().width,
        cells: Array.from(li.children).map((cell) => {
          const rect = cell.getBoundingClientRect();
          /*
           * The glyphs, not the box. A grid item is blockified and stretched to its track, so a
           * cell's own width is the width of the *column* — which is the thing under test, and
           * therefore useless as the precondition that the labels differ from each other. A range
           * over the cell's contents measures the text itself either way.
           */
          const range = document.createRange();
          range.selectNodeContents(cell);
          return {
            text: (cell.textContent ?? '').trim(),
            left: rect.left,
            right: rect.right,
            top: rect.top,
            width: rect.width,
            height: rect.height,
            textWidth: range.getBoundingClientRect().width,
          };
        }),
      }))
    );
}

const spread = (values: number[]) => Math.max(...values) - Math.min(...values);

test.beforeEach(async ({ page }) => {
  await page.setViewportSize(DESKTOP);
  await page.goto('/');
  await expect(strip(page).locator('li').first()).toBeVisible();
});

/**
 * The preconditions, asserted before any offset is compared — because two of them can make the
 * whole file pass against the unfixed markup.
 *
 * The labels have to be of *different* widths, or a per-row `auto` track lands in the same place on
 * every row and aligned columns are true by accident. And every row has to carry a reason: when
 * nothing can run at all the strip drops the reasons into one sentence above the list and renders
 * seven empty third cells, which align perfectly at zero width.
 */
test('seven rows, each with a reason, and labels of differing widths', async ({ page }) => {
  const measured = await rows(page);

  expect(measured, 'the strip did not render its seven archetypes').toHaveLength(7);
  for (const row of measured) {
    expect(row.cells, `row "${row.text}" is not three cells`).toHaveLength(3);
    expect(row.cells[2].text, `row "${row.text}" has no written reason`).not.toBe('');
  }

  // ~59px between "Coding agent" and "Inline code completion", which is precisely the rag the issue
  // measured. A 20px floor states the fact rather than the measurement; without it every alignment
  // assertion below would hold on the unfixed markup too.
  expect(
    spread(measured.map((row) => row.cells[1].textWidth)),
    'every label renders the same width, so nothing here could detect a per-row track'
  ).toBeGreaterThan(20);
});

for (const [name, size] of [
  ['1440px, the width the issue measured', DESKTOP],
  ['640px, exactly at sm', AT_SM],
] as const) {
  test.describe(`at ${name}`, () => {
    test.beforeEach(async ({ page }) => {
      await page.setViewportSize(size);
      // The three-column layout is a `sm:` rule, so a viewport that does not match it would put
      // this whole block on the stacked branch — where the columns coincide for the wrong reason.
      expect(
        await page.evaluate(() => matchMedia('(min-width: 40rem)').matches),
        'this viewport is below sm, so the three-column layout is not the one being measured'
      ).toBe(true);
    });

    test('the reason column starts at the same x on every row', async ({ page }) => {
      const measured = await rows(page);
      const lefts = measured.map((row) => row.cells[2].left);

      // A pixel of tolerance for fractional layout; the defect was 59px of it.
      expect(
        spread(lefts),
        `the reason column wanders: ${lefts.map((l) => Math.round(l)).join(', ')}`
      ).toBeLessThanOrEqual(1);
    });

    test('and the two columns that already aligned still do', async ({ page }) => {
      const measured = await rows(page);

      expect(
        spread(measured.map((row) => row.cells[0].left)),
        'the status column wanders'
      ).toBeLessThanOrEqual(1);
      expect(
        spread(measured.map((row) => row.cells[1].left)),
        'the label column wanders'
      ).toBeLessThanOrEqual(1);
    });

    /**
     * And that the columns align because they are columns, not because the row collapsed into one.
     *
     * A single stacked column satisfies every equality above — all three cells would start at the
     * row's left edge, on seven rows, perfectly aligned. So the third column is asserted to begin
     * past the widest label, which is the arrangement the alignment is supposed to describe.
     */
    test('the reason is a third column, clear of every label', async ({ page }) => {
      const measured = await rows(page);
      const widestLabel = Math.max(...measured.map((row) => row.cells[1].right));

      for (const row of measured) {
        expect(row.cells[1].left, `row "${row.text}" is not laid out in columns`).toBeGreaterThan(
          row.cells[0].left
        );
        expect(
          row.cells[2].left,
          `row "${row.text}" puts its reason under a label rather than beside it`
        ).toBeGreaterThanOrEqual(widestLabel - 1);
        // Each row is one row: three cells sharing a top edge rather than stacked.
        expect(
          spread(row.cells.map((cell) => cell.top)),
          `row "${row.text}" is stacked`
        ).toBeLessThan(6);
      }
    });

    /**
     * And with the descriptions shown, which is the state the third column is longest in.
     *
     * Worth its own case because the rows then have visibly different heights, and a shared set of
     * tracks is where a row-axis mistake — a subgrid on the row axis as well, say — would show up as
     * every reason dragged to the tallest row's baseline. The columns are shared; the rows are
     * emphatically not.
     */
    test('the column holds when the reasons carry their descriptions too', async ({ page }) => {
      const before = await rows(page);
      await page.getByRole('button', { name: /show what each workload means/i }).click();
      await expect(
        page.getByRole('button', { name: /hide what each workload means/i })
      ).toBeVisible();

      const measured = await rows(page);
      const lefts = measured.map((row) => row.cells[2].left);

      // The preconditions: the toggle really landed on every row, and the rows really do differ in
      // height now. Both are what make this case something other than a repeat of the one above.
      for (const [i, row] of measured.entries()) {
        expect(
          row.cells[2].text.length,
          `row "${row.text}" did not take its description`
        ).toBeGreaterThan(before[i].cells[2].text.length);
      }
      expect(
        spread(measured.map((row) => row.cells[2].height)),
        'every row is the same height, so nothing here could detect a shared row track'
      ).toBeGreaterThan(4);

      expect(
        spread(lefts),
        `the reason column wanders once expanded: ${lefts.map((l) => Math.round(l)).join(', ')}`
      ).toBeLessThanOrEqual(1);
      // Each row still starts its own text at its own top, rather than at the tallest row's.
      for (const row of measured) {
        expect(
          spread(row.cells.map((cell) => cell.top)),
          `row "${row.text}" no longer shares a baseline across its own cells`
        ).toBeLessThan(6);
      }
    });
  });
}

/**
 * The half of the fix that is about not breaking anything: below `sm` the row keeps a grid of its
 * own, because the stacked layout is built from `order` and a spanning third cell — relationships
 * among one row's three children. Sharing the list's tracks at every width would sort all
 * twenty-one cells into a block of labels, a block of status words and a block of reasons.
 */
test.describe('below sm', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize(PHONE);
    expect(
      await page.evaluate(() => matchMedia('(min-width: 40rem)').matches),
      'this viewport is past sm, so the stacked layout is not the one being measured'
    ).toBe(false);
  });

  test('each row stacks its reason under a label-and-status line', async ({ page }) => {
    const measured = await rows(page);
    expect(measured).toHaveLength(7);

    for (const row of measured) {
      const [status, label, reason] = row.cells;

      // `order-1`/`order-2` put the label first and the status word beside it, on one line.
      expect(
        label.left,
        `row "${row.text}" did not reorder its label ahead of the status`
      ).toBeLessThan(status.left);
      expect(
        Math.abs(label.top - status.top),
        `row "${row.text}" split its first line`
      ).toBeLessThan(6);

      // And the reason below them, spanning both tracks from the row's own left edge.
      expect(reason.top, `row "${row.text}" did not stack its reason`).toBeGreaterThan(label.top);
      expect(reason.left - row.left, `row "${row.text}" indented its reason`).toBeLessThan(2);
      expect(
        reason.width / row.width,
        `row "${row.text}" lost the col-span on its reason`
      ).toBeGreaterThan(0.9);
    }
  });

  /** And the rows are still rows: seven separate stacks, in catalog order, none interleaved. */
  test('the rows do not interleave', async ({ page }) => {
    const measured = await rows(page);

    for (const [i, row] of measured.entries()) {
      if (i === 0) continue;
      expect(row.cells[1].top, `row "${row.text}" overlaps the row above it`).toBeGreaterThan(
        measured[i - 1].cells[2].top
      );
    }
  });
});
