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

/**
 * The narrowest width anything still ships at, matching `reflow.spec.ts` and `matrix-legend.spec.ts`
 * — and the width the status cell's min-content is measured against, because that is where a floor
 * set by one unbreakable word has the least room to hide.
 */
const NARROW = { width: 320, height: 900 };

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

/**
 * The two spans *inside* each status cell — the aria-hidden glyph and the word.
 *
 * `rows()` above measures cell boxes, and every alignment assertion built on those kept passing
 * while the fourth status word sat 4.1px left of the other six (#75): a grid item is stretched to
 * its track, so the cell box is identical on all seven rows by construction and says nothing about
 * where the *word* starts. That depends on the width of the glyph before it, and the dash is not the
 * width of the circles.
 */
async function statusWords(page: Page) {
  return strip(page)
    .locator('li')
    .evaluateAll((items) =>
      items.map((li) => {
        const cell = li.children[0] as HTMLElement;
        const [icon, word] = Array.from(cell.children) as HTMLElement[];
        return {
          word: (word.textContent ?? '').trim(),
          wordLeft: word.getBoundingClientRect().left,
          iconWidth: icon.getBoundingClientRect().width,
          whiteSpace: getComputedStyle(cell).whiteSpace,
        };
      })
    );
}

/**
 * A font deliberately wider than any the app will resolve, and the same one `reflow.spec.ts` uses —
 * for the same reason recorded at length there: an earlier version of that file measured the host's
 * own typography, passed on macOS with 18px to spare and failed on CI by 4px, on markup neither run
 * had changed. Courier New is present or metric-aliased on all three platforms.
 *
 * Applied through `--font-sans`, because `body` sets `font-family: var(--font-sans)` and an inline
 * `font-family` on `<html>` loses to that rule.
 */
const WIDE_FONT = "'Courier New', monospace";

/** Whether the stress font actually widened anything, rather than falling back to the host's sans. */
async function stressFontRatio(page: Page) {
  const probe = () =>
    page.evaluate(() => {
      const span = document.createElement('span');
      span.textContent = 'Not measured';
      span.style.cssText =
        'position:absolute;white-space:nowrap;visibility:hidden;font-size:32px;font-family:var(--font-sans)';
      document.body.append(span);
      const width = span.getBoundingClientRect().width;
      span.remove();
      return width;
    });
  const native = await probe();
  await page.evaluate((f) => {
    document.documentElement.style.setProperty('--font-sans', f);
  }, WIDE_FONT);
  return (await probe()) / native;
}

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
     * The status track is a *fixed* 9rem, and every status word has to fit inside it (#75).
     *
     * The three original words were "Yes", "Tight" and "No"; the ungraded state added "Not measured",
     * which is roughly three times the widest of them. A fixed track does not grow for its content,
     * and the cell is `whitespace-nowrap` here — so a word too long for 9rem does not wrap and does
     * not push the label column across, it simply paints over the label beside it. Every alignment
     * assertion in this file keeps passing while the two columns overlap, because the *boxes* are
     * still where they should be.
     *
     * **This is the easy half of the question, and on its own it was the wrong half.** A fixed track
     * cannot widen the layout, so this block — both its viewports are `sm`+ at root 16px, where the
     * track is 144px and the word 98px — has 46px of slack and cannot reach the failure the long word
     * actually caused, which is below `sm`, where the same cell sits in a `1fr` track and its
     * min-content becomes a floor on the row. That is `the status column does not set a floor wider
     * than the viewport` in the block below; this one guards the overlap, that one guards the reflow.
     *
     * jsdom cannot answer either — no layout engine, so the glyph width is 0 there and the comparison
     * is a tautology — which is what puts a text-fits check in this directory rather than in Vitest.
     * The precondition is asserted first: the default scenario is one concurrent user, so the seventh
     * row really is the long word, and without that this measures three short ones against 144px.
     */
    test('every status word fits inside the fixed status column', async ({ page }) => {
      const measured = await rows(page);

      expect(
        measured.some((row) => row.cells[0].text.includes('Not measured')),
        'no row is ungraded at the default scenario, so the longest status word is not on screen'
      ).toBe(true);

      // The mechanism the paragraph above depends on: at `sm` the cell must not wrap, or a word too
      // long for the track becomes a two-line cell rather than an overlap, and this measures nothing.
      for (const status of await statusWords(page)) {
        expect(status.whiteSpace, `the status cell for "${status.word}" wraps at sm`).toBe(
          'nowrap'
        );
      }

      for (const row of measured) {
        const status = row.cells[0];
        expect(
          status.textWidth,
          `the status word "${status.text}" is ${Math.round(status.textWidth)}px in a ${Math.round(status.width)}px column, so it paints over the label`
        ).toBeLessThanOrEqual(status.width);
      }
    });

    /**
     * And every status word starts at the same x, whichever glyph precedes it.
     *
     * The word's left edge is the glyph's width plus the flex gap, so before the glyph was boxed the
     * column had a jog in it exactly where the fourth state was added: measured at 1440px, the three
     * circles are 11.1px wide and the dash 10.5px, so "Not measured" started 4.1px left of "Yes",
     * "Tight" and "No" when the dash was the narrower en dash it first shipped as. Small, and in the
     * one column #70 was about aligning.
     *
     * Not caught by anything above, and could not be: every other assertion in this file reads cell
     * boxes, and a grid item is stretched to its track, so all seven are identical no matter what is
     * inside them. This reads the spans.
     *
     * The icon widths are asserted equal as well, because that is the *mechanism* — a fixed `w-3` box
     * with the glyph centred in it — and it is what makes the alignment hold on a runner whose fonts
     * draw these four characters at four different widths. Without it this test is a claim about
     * whatever font the runner happened to resolve.
     */
    test('every status word starts at the same x, whatever glyph precedes it', async ({ page }) => {
      const measured = await statusWords(page);

      expect(
        measured.some((row) => row.word === 'Not measured'),
        'no row is ungraded at the default scenario, so the odd glyph is not on screen'
      ).toBe(true);

      // The claim first, so a regression reports the jog rather than its cause.
      expect(
        spread(measured.map((row) => row.wordLeft)),
        `the status words do not share a left edge: ${measured.map((r) => `${r.word} at ${Math.round(r.wordLeft * 10) / 10}`).join(', ')}`
      ).toBeLessThanOrEqual(1);
      // Then the mechanism, which is what makes the claim above hold on a runner whose fonts draw
      // these four characters at four different widths rather than only on this one.
      expect(
        spread(measured.map((row) => row.iconWidth)),
        `the status glyphs are not in equal boxes: ${measured.map((r) => Math.round(r.iconWidth * 10) / 10).join(', ')}`
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

  /**
   * And the status cell does not set a floor wider than the viewport (#75, #35).
   *
   * **This is the assertion the fourth status word needed and the `sm`+ block above cannot make.**
   * Below `sm` the row is `grid-cols-[auto_1fr]` with the status cell in the `1fr`, whose automatic
   * minimum is that cell's min-content — so an unbreakable string here is a hard floor on the row,
   * and the row is inside a panel inside the document. Three short words never came near it. "Not
   * measured" with a `whitespace-nowrap` on it took the seventh row's status cell to a 199px
   * min-content against 55px for "○ No", the document to `scrollWidth` 371 in a 320px viewport, and
   * 29 elements outside it — the 200%-text reflow failure of WCAG 1.4.4 that #35 fixed and
   * `reflow.spec.ts` exists to keep fixed.
   *
   * Two stresses, and the defect needs both:
   *
   *   - **The root font size**, because at 16px the cell is 93px in a 390px row and has all the room
   *     it wants. Set here with `style.fontSize` rather than by the browser's own default-size
   *     switch, which is what the `reflow` project uses and what this project cannot. The usual
   *     objection to doing it this way is recorded in `reflow.spec.ts` and is real — `rem` inside a
   *     media query resolves against the browser default, so the breakpoints do *not* move, and a
   *     sweep written this way reports on layout states no reader can reach. It does not apply at
   *     320px: `sm` is unmatched at 320px whether it evaluates to 640px or to 1280px, so the stacked
   *     row measured here is the same box a reader at 200% text gets. The `beforeEach` asserts that.
   *     Anything about the layout *at or past* `sm` under real zoom stays in `reflow.spec.ts`.
   *   - **A font wider than the host's**, because macOS resolves the app's stack to SF and passes the
   *     unfixed markup at this viewport with 8px to spare, while a CI runner falls through to a wider
   *     sans and fails. That exact green-here-red-there result has already shipped from this repo
   *     once, which is why the stress font is a precondition rather than a hope.
   *
   * Scoped to the strip's own rows rather than to `document.scrollWidth`: the whole-document question
   * belongs to `reflow.spec.ts`, which asks it at the real zoom, on four scenarios, at four widths.
   * What this pins is that *this panel* is not the thing that breaks it.
   */
  test('the status column does not set a floor wider than the viewport', async ({ page }) => {
    await page.setViewportSize(NARROW);
    expect(
      await page.evaluate(() => matchMedia('(min-width: 40rem)').matches),
      'sm matches at 320px, so this is no longer the stacked layout'
    ).toBe(false);

    // Text at 200%, and the breakpoint deliberately left where it is — see above.
    await page.evaluate(() => {
      document.documentElement.style.fontSize = '32px';
    });
    // Applies the stress font and returns what it bought, so a runner without Courier New fails here
    // rather than quietly measuring its own typography.
    const ratio = await stressFontRatio(page);
    expect(
      ratio,
      'the stress font did not widen the text, so this run proves nothing'
    ).toBeGreaterThan(1.05);

    const measured = await rows(page);
    const statuses = await statusWords(page);

    // The precondition: the long word has to be on screen, and at the default scenario's one
    // concurrent user it is. Without this the case measures three short words that never had a
    // problem.
    expect(
      statuses.some((row) => row.word === 'Not measured'),
      'no row is ungraded at the default scenario, so the longest status word is not on screen'
    ).toBe(true);

    // The claim, first, so a regression reports the harm — the unfixed markup fails here on the
    // seventh row at 371px against 320.
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    for (const row of measured) {
      for (const cell of row.cells) {
        expect(
          cell.right,
          `"${row.text}" pushes its ${Math.round(cell.right)}px right edge past the ${clientWidth}px viewport`
        ).toBeLessThanOrEqual(clientWidth + 1);
      }
    }

    // Then the mechanism, because the assertion above passes if the cell is allowed to break *or* if
    // something else happens to absorb it — a shorter word, a narrower label — and neither of those
    // keeps holding when the next status word is added. The floor has to be the longest word rather
    // than the whole string.
    for (const status of statuses) {
      expect(
        status.whiteSpace,
        `the status cell for "${status.word}" is nowrap below sm, so its min-content is the whole string`
      ).toBe('normal');
    }
  });
});
