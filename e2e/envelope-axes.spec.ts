import { expect, test, type Page } from '@playwright/test';
import { SETTING_LABELS } from '@/lib/stops';

/**
 * The Envelope's axis titles, at the width where the plot has nothing to spare.
 *
 * Issue #81. Neither axis said what it measured, and the fix is two titles — so the risk the fix
 * itself carries is geometric rather than textual: `MIN_COLUMN_REM` puts a floor under every column
 * and the plot sits in an `overflow-x-auto` of its own precisely because the axis cannot be
 * squeezed. Anything placed *beside* the plot therefore comes out of the width the reader can see.
 * The titles here are stacked above and below instead, and these are the assertions that say the
 * cost landed in vertical space.
 *
 * Worth being exact about the alternative, because a wrong figure in a spec is the figure a later
 * session inherits: a `writing-mode: vertical-rl` title in the row costs **one line box** of
 * horizontal space, not the label's full length, and none of the plot's content width — only its
 * scroll viewport. That is still a real cost at this width and the width assertion below still
 * catches it; it is a fraction of a column rather than two columns. (A `rotate-90` transform is the
 * expensive one, since transforms do not affect layout at all and it reserves its unrotated box.)
 * Stacking is a legibility call — 12px type on its side, and an arrow that has to read as up.
 *
 * jsdom cannot answer any of it: no layout engine, so every width and every rect it reports is 0
 * and every assertion below is a tautology there. The wording, and that both titles read from the
 * same constant the controls do, is checked in `App.test.tsx` where it runs in a second.
 */

/**
 * 320px, not a comfortable phone.
 *
 * This is where the plot already scrolls sideways inside its own box — asserted as a precondition
 * below rather than assumed, because a spec written at 1440px passes against a gutter that breaks
 * the phone layout: there, the columns fit, nothing scrolls, and a line box taken out of the row is
 * absorbed by a panel with room to spare. It is also the narrowest width anything still ships at.
 */
const NARROW = { width: 320, height: 720 };

const envelope = (page: Page) => page.getByRole('region', { name: /how much room is left/i });

/**
 * The two titles, found structurally.
 *
 * Neither is in the accessible tree — both are `aria-hidden`, because the canvas `aria-label` is
 * this picture's textual equivalent and already names both quantities — so there is no role to ask
 * for. The count is asserted before anything is measured, and the text of each is asserted against
 * `SETTING_LABELS` below, so a spec that found some other paragraph fails rather than quietly
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
 *
 * Each title is measured twice, and the second measurement is the one that carries the assertions.
 * Both titles are `display: block` children of full-width containers, so their *element* rects are
 * their parents' rects by construction — identical for any text alignment and unmoved by text that
 * overflows. `ink` is a `Range` over the title's contents, which is where the glyphs actually are.
 * The element rect is still worth having for the vertical claim, where the block box is the thing
 * that reserves space.
 *
 * The row gap is read off the row rather than transcribed: `gap-2` is 0.5rem, so writing 8 here is
 * only right at a 16px root, and this spec would be wrong-by-7px the day it is added to the
 * `reflow` project — which runs with a 32px browser default.
 */
async function geometry(page: Page) {
  return envelope(page).evaluate((section: HTMLElement) => {
    const rect = (el: Element) => {
      const r = el.getBoundingClientRect();
      return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width };
    };

    /** Where the glyphs are, as opposed to where their block is. */
    const ink = (el: Element) => {
      const range = document.createRange();
      range.selectNodeContents(el);
      const r = range.getBoundingClientRect();
      return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width };
    };

    const title = (el: Element) => ({
      ...rect(el),
      ink: ink(el),
      text: (el.textContent ?? '').trim(),
      textAlign: getComputedStyle(el).textAlign,
    });

    const canvas = section.querySelector('canvas');
    const gutter = section.querySelector('ol');
    const paragraphs = Array.from(section.querySelectorAll('p[aria-hidden="true"]'));
    if (!canvas || !gutter?.parentElement || paragraphs.length < 2) return null;

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
      rowGap: parseFloat(getComputedStyle(gutter.parentElement).columnGap) || 0,
      plot: { ...rect(plot), scrollWidth: plot.scrollWidth, clientWidth: plot.clientWidth },
      // In DOM order, which is y then x: the y title sits above the row, the x title under the
      // column labels.
      y: title(paragraphs[0]),
      x: title(paragraphs[1]),
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
  expect(box!.y.text).toContain(SETTING_LABELS.concurrency);
  expect(box!.x.text).toBe(SETTING_LABELS.contextTokens);
  // The direction cue. Rows are drawn bottom-up, which is right for an axis and the opposite of
  // every other list on this page.
  expect(box!.y.text, 'the y title does not say which way concurrency runs').toContain('↑');
});

test('the plot still scrolls here, which is what makes the width claim below worth making', async ({
  page,
}) => {
  const box = await geometry(page);
  expect(box).not.toBeNull();

  // The precondition, asserted rather than assumed. If the columns ever fit at 320px, the row has
  // slack again and the next test would pass with a vertical title in it — green, and testing
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
  // the title bought no width at all. A title in the row — `writing-mode: vertical-rl`, the cheap
  // way to rotate one — subtracts its line box and its own share of the gap here.
  expect(
    box!.plot.width,
    'something other than the tick labels is taking width from the plot'
  ).toBeGreaterThanOrEqual(box!.content.width - box!.gutter.width - box!.rowGap - 1);
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
    /*
     * The `ink` rect, not the element's. Both titles are block children of full-width containers,
     * so `title.left`/`title.right` are the container's edges whatever the text does — give either
     * one `whitespace-nowrap` and the words visibly leave the panel while those two numbers do not
     * move. The Range around the glyphs is the thing that moves.
     *
     * A pixel of tolerance on each side: these are fractional layout numbers, and the failure being
     * guarded is a whole label's width past the edge rather than a rounding difference.
     */
    expect(
      title.ink.left,
      `the ${name} title escapes the panel on the left`
    ).toBeGreaterThanOrEqual(box!.content.left - 1);
    expect(title.ink.right, `the ${name} title escapes the panel on the right`).toBeLessThanOrEqual(
      box!.content.right + 1
    );
  }

  // And the document itself, which is what a title too wide for the panel does to a page whose
  // scroll container belongs to the plot: the titles have none of their own, so the overflow
  // arrives here.
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
   * half off-screen.
   *
   * Measured over the glyphs and not over the paragraph, for the reason the test above gives: the
   * `<p>` has no margins of its own, so it fills exactly the box the scroller above it fills —
   * comparing element centres compares one box with itself, and is 0.0px apart with `text-center`
   * deleted (measured, in a browser, at this viewport). The slack is
   * asserted first, because a line of text that happened to fill its box would make the centring
   * claim vacuous again — silently, and only at some viewports.
   */
  expect(
    box!.x.ink.width,
    'the x title fills its box, so its centre cannot report an alignment'
  ).toBeLessThan(box!.x.width - 8);
  expect(box!.x.textAlign, 'the x title is no longer centred by CSS').toBe('center');

  const inkCentre = box!.x.ink.left + box!.x.ink.width / 2;
  const plotCentre = box!.plot.left + box!.plot.width / 2;
  expect(Math.abs(inkCentre - plotCentre), 'the x title is not centred on the plot').toBeLessThan(
    2
  );
});
