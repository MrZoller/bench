import { expect, test } from '@playwright/test';

/**
 * The Envelope draws its feasibility field on a 2D canvas — the only canvas on the page — and jsdom
 * stubs `getContext`. The unit suite emits "Not implemented: HTMLCanvasElement.getContext" on every
 * run and asserts nothing about the plot, which could be entirely blank in a real browser with
 * every test still passing.
 *
 * These read the pixels back. Not a visual comparison — a screenshot baseline would fail on every
 * font or catalogue change and teach people to re-bless it — just the claim that something was
 * painted, in more than one colour, at the size the layout gave it.
 */

/**
 * What a canvas is currently showing, reduced to something comparable.
 *
 * A colour *count* is not enough on its own to tell two fields apart: the Envelope fills every cell
 * from a five-colour palette, so the count is dominated by cell-edge anti-aliasing and moves by one
 * or two between wholly different scenarios. The first version of the repaint test below turned on
 * exactly that difference of one, which would have become a five-second poll timeout reporting
 * `expected true, received false` the moment a catalogue change made two counts collide.
 *
 * So this also returns a `digest` — an order-sensitive hash over the sampled bytes — which changes
 * if any sampled pixel changes. The counts stay because they are what a *human* reads out of a
 * failure; the digest is what the assertion compares.
 */
async function painted(page: import('@playwright/test').Page, selector: string) {
  return page.evaluate((sel) => {
    const canvas = document.querySelector<HTMLCanvasElement>(sel);
    if (!canvas) return { error: 'no canvas' as const };
    const ctx = canvas.getContext('2d');
    if (!ctx) return { error: 'no 2d context' as const };
    if (canvas.width === 0 || canvas.height === 0) return { error: 'zero-sized bitmap' as const };

    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const colours = new Set<string>();
    let opaque = 0;
    let digest = 0;
    // Every 4th pixel: enough to characterise a grid of flat-filled cells, cheap enough to run on
    // a retina-scaled bitmap without serialising a megabyte back across the bridge.
    for (let i = 0; i < data.length; i += 16) {
      digest = (digest * 31 + data[i] + data[i + 1] * 7 + data[i + 2] * 13 + data[i + 3] * 17) | 0;
      if (data[i + 3] === 0) continue;
      opaque += 1;
      colours.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return { colours: colours.size, opaque, digest, area: canvas.width * canvas.height };
  }, selector);
}

test('the Envelope actually paints its feasibility field', async ({ page }) => {
  await page.goto('/');

  const canvas = page.getByRole('region', { name: /how much room/i }).locator('canvas');
  await expect(canvas).toBeVisible();

  const box = await canvas.boundingBox();
  expect(box!.width).toBeGreaterThan(0);
  expect(box!.height).toBeGreaterThan(0);

  const field = await painted(page, 'canvas');
  expect(field, 'the canvas could not be read back').not.toHaveProperty('error');
  if ('colours' in field) {
    expect(field.opaque).toBeGreaterThan(0);
    // More than one, or the grid is a single flat rectangle and the classification is not being
    // drawn — which is what a silently-failing draw loop looks like.
    expect(field.colours).toBeGreaterThan(1);
  }
});

/**
 * And that it redraws when the scenario moves. A canvas painted once at mount and never again
 * would pass every assertion above while showing a stale field — the failure mode a retained-mode
 * surface has and a DOM one does not.
 */
test('the Envelope repaints when the scenario changes', async ({ page }) => {
  await page.goto('/');
  const canvas = page.getByRole('region', { name: /how much room/i }).locator('canvas');
  await expect(canvas).toBeVisible();

  const before = await painted(page, 'canvas');
  // Checked here as well as in the test above: without it a zero-sized bitmap surfaces below as an
  // uninformative poll timeout rather than as the thing that actually went wrong.
  expect(before, 'the canvas could not be read back').not.toHaveProperty('error');

  // A model that changes the field wholesale rather than by one cell.
  // `exact`, because the Matrix section's accessible name is "Every model on every machine…" and a
  // loose label match resolves to two elements.
  await page.getByLabel('Model', { exact: true }).selectOption('deepseek-ai/DeepSeek-V3');

  await expect
    .poll(
      async () => {
        const after = await painted(page, 'canvas');
        return 'digest' in after && 'digest' in before ? after.digest !== before.digest : false;
      },
      { message: 'the canvas never repainted after the model changed' }
    )
    .toBe(true);
});

/**
 * The bitmap is sized from `getBoundingClientRect` times the device pixel ratio, in an effect that
 * runs after layout. Getting that wrong gives a stretched or blank plot rather than an error.
 */
test('the Envelope bitmap matches its laid-out size', async ({ page }) => {
  await page.goto('/');
  const canvas = page.getByRole('region', { name: /how much room/i }).locator('canvas');
  await expect(canvas).toBeVisible();

  const { cssWidth, bitmapWidth, dpr } = await page.evaluate(() => {
    const el = document.querySelector<HTMLCanvasElement>('canvas')!;
    return {
      cssWidth: el.getBoundingClientRect().width,
      bitmapWidth: el.width,
      dpr: window.devicePixelRatio || 1,
    };
  });

  expect(bitmapWidth).toBeGreaterThan(0);
  expect(Math.abs(bitmapWidth - cssWidth * dpr)).toBeLessThanOrEqual(2);
});
