import { expect, test, type Page } from '@playwright/test';
import { USAGE_LABELS } from '@/lib/stops';

/**
 * The Envelope's axis titles, at the width where a label can only be bought with plot columns.
 *
 * Issue #81. Neither axis said what it measured, and the fix is two titles — so the risk the fix
 * itself carries is geometric rather than textual: this is the one surface on the page with no
 * width to spare. `MIN_COLUMN_REM` puts a floor under every column and the plot sits in an
 * `overflow-x-auto` of its own precisely because the axis cannot be squeezed, so a **rotated** y
 * title — the conventional choice — takes fixed width out of the gutter on the surface that has
 * none to give. The title here is stacked above the gutter instead, which costs a line of vertical
 * space, and these are the assertions that say the cost landed there.
 *
 * jsdom cannot answer any of it: no layout engine, so every width and every rect it reports is 0
 * and every assertion below is a tautology there. The wording, and that both titles read from the
 * same constant the Usage sliders do, is checked in `App.test.tsx` where it runs in a second.
 */

/**
 * 320px, not a comfortable phone.
 *
 * This is where the plot already scrolls sideways inside its own box — asserted as a precondition
 * below rather than assumed, because a spec written at 1440px passes against a gutter that breaks
 * the phone layout: there, the columns fit, nothing scrolls, and 100px of extra gutter is absorbed
 * by a panel with room to spare. It is also the narrowest width anything still ships at.
 */
const NARROW = { width: 320, height: 720 };

/** The gap between the y gutter and the plot — Tailwind's `gap-2`, which is 0.5rem. */
const ROW_GAP = 8;

const envelope = (page: Page) => page.getByRole('region', { name: /how much room is left/i });

/**
 * The two titles, found structurally.
 *
 * Neither is in the accessible tree — both are `aria-hidden`, because the canvas `aria-label` is
 * this picture's textual equivalent and already names both quantities — so there is no role to ask
 * for. The count is asserted before anything is measured, and the text of each is asserted against
 * `USAGE_LABELS` below, so a spec that found some other paragraph fails rather than quietly
 * measuring it. Measuring the wrong element is how this suite has already produced three tests that
 * could not fail.
 */
const titles = (page: Page) => envelope(page).locator('p[aria-hidden="true"]');

/**
 * Every box this file compares, read in one pass so the numbers describe one layout.
 *
 * The scroll container is found by *behaviour* — the nearest ancestor of the canvas whose computed
 * `overflow-x` is not `visible` — rather than by class name, which is the same thing
 * `reflow.spec.ts` does when it decides where to stop descending. A class-based selector would go
 * quietly stale the day the utility moves one div.
 */
async function geometry(page: Page) {
  return envelope(page).evaluate((section: HTMLElement) => {
    const rect = (el: Element) => {
      const r = el.getBoundingClientRect();
      return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width };
    };

    const canvas = section.querySelector('canvas');
    const gutter = section.querySelector('ol');
    const paragraphs = Array.from(section.querySelectorAll('p[aria-hidden="true"]'));
    if (!canvas || !gutter || paragraphs.length < 2) return null;

    let plot: HTMLElement | null = canvas.parentElement;
    while (plot && getComputedStyle(plot).overflowX === 'visible') plot = plot.parentElement;
    if (!plot) return null;

    // The panel's content box, which is what the titles have to stay inside: the padding is
    // `p-[min(1.25rem,5vw)]`, so at this viewport it is 5vw rather than the 20px it is on a laptop.
    const style = getComputedStyle(section);
    const panel = rect(section);
    const content = {
      left: panel.left + parseFloat(style.borderLeftWidth) + parseFloat(style.paddingLeft),
      right: panel.right - parseFloat(style.borderRightWidth) - parseFloat(style.paddingRight),
    };

    return {
      content: { ...content, width: content.right - content.left },
      canvas: rect(canvas),
      gutter: rect(gutter),
      plot: { ...rect(plot), scrollWidth: plot.scrollWidth, clientWidth: plot.clientWidth },
      // In DOM order, which is y then x: the y title sits above the row, the x title under the
      // column labels.
      y: { ...rect(paragraphs[0]), text: (paragraphs[0].textContent ?? '').trim() },
      x: { ...rect(paragraphs[1]), text: (paragraphs[1].textContent ?? '').trim() },
      document: {
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      },
    };
  });
}

test.beforeEach(async ({ page }) => {
  await page.setViewportSize(NARROW);
  await page.goto('/');
  await expect(envelope(page).locator('canvas')).toBeVisible();
});

test('both axes carry a title at all', async ({ page }) => {
  // The assertion a deleted title fails. Everything below measures boxes, and a box that does not
  // exist reports nothing rather than reporting a failure, so the existence claim comes first.
  await expect(titles(page)).toHaveCount(2);
  await expect(titles(page).first()).toBeVisible();
  await expect(titles(page).last()).toBeVisible();
});

test('the titles say what the Usage controls say, and the y one says which way it runs', async ({
  page,
}) => {
  const box = await geometry(page);
  expect(box, 'the Envelope did not lay out both axis titles').not.toBeNull();

  // Read from the shared constant, so a title rewritten in place of the control's own wording fails
  // here as well as in the unit suite — this is also what confirms the locator found the titles
  // rather than some neighbouring paragraph.
  expect(box!.y.text).toContain(USAGE_LABELS.concurrency);
  expect(box!.x.text).toBe(USAGE_LABELS.contextTokens);
  // The direction cue. Rows are drawn bottom-up, which is right for an axis and the opposite of
  // every other list on this page.
  expect(box!.y.text, 'the y title does not say which way concurrency runs').toContain('↑');
});

test('the plot still scrolls here, which is what makes the width claim below worth making', async ({
  page,
}) => {
  const box = await geometry(page);
  expect(box).not.toBeNull();

  // The precondition, asserted rather than assumed. If the columns ever fit at 320px, the gutter
  // has slack again and the next test would pass with a rotated title in it — green, and testing
  // nothing it is written to test.
  expect(
    box!.plot.scrollWidth,
    'the plot no longer scrolls at 320px — this spec is measuring a viewport with slack'
  ).toBeGreaterThan(box!.plot.clientWidth);
});

test('the y title costs vertical space, not plot columns', async ({ page }) => {
  const box = await geometry(page);
  expect(box).not.toBeNull();

  // Above the plot, not beside it: the whole of the title sits in the band over the canvas.
  expect(box!.y.bottom, 'the y title overlaps the plot').toBeLessThanOrEqual(box!.canvas.top + 1);

  // And nothing but the tick gutter and the row gap stands between the panel edge and the plot, so
  // the title bought no columns. A rotated title in the gutter subtracts its own line height here.
  expect(
    box!.plot.width,
    'something other than the tick labels is taking width from the plot'
  ).toBeGreaterThanOrEqual(box!.content.width - box!.gutter.width - ROW_GAP - 1);
});

test('the titles stay inside the panel, and the page does not scroll sideways', async ({
  page,
}) => {
  const box = await geometry(page);
  expect(box).not.toBeNull();

  for (const [name, title] of [
    ['y', box!.y],
    ['x', box!.x],
  ] as const) {
    // A pixel of tolerance on each side: these are fractional layout numbers, and the failure being
    // guarded is a whole label's width past the edge rather than a rounding difference.
    expect(title.left, `the ${name} title escapes the panel on the left`).toBeGreaterThanOrEqual(
      box!.content.left - 1
    );
    expect(title.right, `the ${name} title escapes the panel on the right`).toBeLessThanOrEqual(
      box!.content.right + 1
    );
  }

  // The point of the pair above rather than a restatement: the plot has a scroll container and the
  // titles have none, so a title too wide for the panel scrolls the document instead of itself.
  expect(box!.document.scrollWidth, 'the page scrolls sideways').toBeLessThanOrEqual(
    box!.document.clientWidth + 1
  );
});

test('the x title is centred on the plot the reader can see', async ({ page }) => {
  const box = await geometry(page);
  expect(box).not.toBeNull();

  /*
   * Centred on the scroll *viewport*, which is why the title sits outside it. Inside, it would be
   * centred on the scrolled content — wider than the panel at this width — so the title would be
   * half off-screen and its box would extend past the panel edge, failing the test above for a
   * reason that has nothing to do with the gutter.
   */
  const titleCentre = box!.x.left + box!.x.width / 2;
  const plotCentre = box!.plot.left + box!.plot.width / 2;
  expect(Math.abs(titleCentre - plotCentre), 'the x title is not centred on the plot').toBeLessThan(
    2
  );
});
