import { expect, test } from '@playwright/test';

/**
 * The two refusals the Matrix used to draw identically (#72), measured as paint rather than as
 * class names.
 *
 * A column the runtime cannot drive is now a struck heading over cells with no ink, and a pair that
 * was measured and did not fit keeps the dashed border the legend keys as "will not run". Both
 * halves of that are computed style: `line-through` and a dropped border are strings in jsdom, where
 * `App.test.tsx` asserts the markup, and nothing there can say whether a browser painted either. The
 * failure this guards is the one that shape of gap keeps producing here — a class list that reads
 * correctly and a rule that never applies, because the utility was renamed, purged as unused, or
 * overridden by the neighbour it shares a `border` shorthand with.
 *
 * What it does **not** do is read pixels. "Distinguishable from a dashed cell" would mean decoding a
 * screenshot, so it asserts the mechanism: the decoration line on the heading, the border style on
 * the cells, the cursor those border-less cells resolve to, and both groups non-empty so no claim is
 * vacuous. Same compromise `budget-overshoot.spec.ts` makes for its halo.
 */

/**
 * vLLM, which drives NVIDIA and AMD cards and drives no Mac, no Strix Halo and no CPU host — so one
 * grid carries both states at once. That is the precondition every assertion below rests on, and it
 * is asserted rather than assumed: with a runtime that drives everything (llama.cpp, the default)
 * nothing is struck and this whole file would pass without testing anything.
 */
const BOTH_STATES = '/?r=vllm';

interface Column {
  device: string;
  decoration: string;
  spoken: string | null;
  /** The computed border of every cell in this column, top edge — they are drawn as one. */
  borders: { style: string; width: number }[];
  /**
   * What the pointer is told before it clicks, per cell.
   *
   * The only channel a mouse user has on a square with no boundary, and a computed value: the
   * utility is a class name in jsdom, so `App.test.tsx` can say it is present and nothing there can
   * say the browser resolved it to a cursor.
   */
  cursors: string[];
}

const readColumns = (page: import('@playwright/test').Page) =>
  page.locator('table[role="grid"]').evaluate((table): Column[] => {
    const rows = [...table.querySelectorAll('tbody tr')];
    return [...table.querySelectorAll('thead th')].slice(1).map((th, i) => {
      const label = th.querySelector('span[title]')!;
      return {
        device: label.getAttribute('title') ?? '',
        decoration: getComputedStyle(label).textDecorationLine,
        spoken: th.getAttribute('aria-label'),
        borders: rows.map((row) => {
          const cell = row.querySelectorAll('td button')[i];
          const style = getComputedStyle(cell);
          return {
            style: style.borderTopStyle,
            width: parseFloat(style.borderTopWidth) || 0,
          };
        }),
        cursors: rows.map((row) => getComputedStyle(row.querySelectorAll('td button')[i]).cursor),
      };
    });
  });

test.beforeEach(async ({ page }) => {
  await page.goto(BOTH_STATES);
  await expect(page.locator('table[role="grid"] td button').first()).toBeVisible();
});

test('a column the runtime cannot drive is painted struck, and its neighbours are not', async ({
  page,
}) => {
  const columns = await readColumns(page);
  expect(columns.length).toBeGreaterThan(1);

  const struck = columns.filter((c) => c.decoration.includes('line-through'));
  const plain = columns.filter((c) => !c.decoration.includes('line-through'));

  // Both populated, or one of the two assertions below is about nothing.
  expect(struck.length, 'no heading is struck, so vLLM drives everything here now').toBeGreaterThan(
    1
  );
  expect(plain.length, 'every heading is struck, so the mark says nothing').toBeGreaterThan(1);

  // The mark and the accessible name are one claim; a strike with no name is a mark only a sighted
  // reader can meet, which is the inversion this repo has shipped before.
  for (const column of struck) {
    expect(column.spoken, `"${column.device}" is struck with no name saying why`).toMatch(
      /does not support this hardware, at any size/i
    );
  }
  for (const column of plain) {
    expect(column.spoken, `"${column.device}" is named as unsupported but not struck`).toBeNull();
  }
});

test('the cells under a struck heading carry no border, and measured ones still do', async ({
  page,
}) => {
  const columns = await readColumns(page);
  const drawn = (b: { style: string; width: number }) => b.style !== 'none' && b.width > 0;

  const struck = columns.filter((c) => c.decoration.includes('line-through'));
  expect(struck.length).toBeGreaterThan(1);

  for (const column of struck) {
    const painted = column.borders.filter(drawn);
    expect(
      painted.length,
      `${painted.length} cells under the struck "${column.device}" still wear the "will not run" border`
    ).toBe(0);
  }

  /**
   * And the swatch the legend keys is still painted somewhere the runtime *is* supported.
   *
   * DeepSeek V3 does not fit a 3090 under any runtime that can load it, so this is reachable — and
   * without it, deleting the border outright would pass every assertion above. Dashed specifically:
   * the legend's key is a dashed square, and a solid border here would be a different mark under the
   * same caption.
   */
  const measured = columns
    .filter((c) => !c.decoration.includes('line-through'))
    .flatMap((c) => c.borders)
    .filter(drawn);
  expect(measured.length, 'nothing on the grid keys "will not run" any more').toBeGreaterThan(0);
  for (const border of measured) expect(border.style).toBe('dashed');
});

/**
 * The cells under a struck heading say "inert" to the pointer as well as to the accessibility tree.
 *
 * They have no boundary left to look at — `tokens.ts` records `--color-border` at 1.18:1 on the
 * raised fill, so there was never one worth looking at — and the answer to that is state rather than
 * ink. `App.test.tsx` asserts the `aria-disabled` attribute and the utility class; what it cannot
 * assert is that the class resolved to a cursor, which is the one thing a mouse user actually meets.
 */
test('a closed column tells the pointer it is closed', async ({ page }) => {
  const columns = await readColumns(page);
  const struck = columns.filter((c) => c.decoration.includes('line-through'));
  const plain = columns.filter((c) => !c.decoration.includes('line-through'));
  expect(struck.length).toBeGreaterThan(1);
  expect(plain.length).toBeGreaterThan(1);

  for (const column of struck) {
    const wrong = column.cursors.filter((c) => c !== 'not-allowed');
    expect(
      wrong.length,
      `cells under the struck "${column.device}" invite a click: ${[...new Set(wrong)].join(', ')}`
    ).toBe(0);
  }
  // And a column the runtime drives still invites one, so the rule above is the narrow one.
  for (const column of plain) {
    expect(column.cursors.every((c) => c !== 'not-allowed')).toBe(true);
  }
});

test('the legend key is struck the same way the heading is', async ({ page }) => {
  const key = page
    .getByRole('region', { name: /every model on every machine/i })
    .getByText(/does not support this hardware, at any size/i);
  await expect(key).toBeVisible();

  // The sample *is* the mark, so it has to be painted like the mark. A key that describes a strike
  // without wearing one sends the reader looking for something else entirely.
  const sample = key.locator('span').first();
  await expect(sample).toHaveText(/struck column heading/i);
  expect(await sample.evaluate((el) => getComputedStyle(el).textDecorationLine)).toContain(
    'line-through'
  );
});
