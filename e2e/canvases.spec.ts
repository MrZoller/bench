import { expect, test } from '@playwright/test';

/**
 * The Envelope draws its feasibility field on a 2D canvas, the masthead draws its backdrop on a
 * second one, and jsdom stubs `getContext`. The unit suite emits "Not implemented:
 * HTMLCanvasElement.getContext" on every run and asserts nothing about either, both of which could
 * be entirely blank in a real browser with every test still passing.
 *
 * These read the pixels back. Not a visual comparison — a screenshot baseline would fail on every
 * font or catalogue change and teach people to re-bless it — just the claim that something was
 * painted, in more than one colour, at the size the layout gave it.
 *
 * Every canvas here is reached through a *scoped* locator, never `document.querySelector('canvas')`.
 * That was how this file read the Envelope when it was the only canvas on the page, and the
 * masthead lands earlier in the DOM: a bare selector would now hand every Envelope assertion below
 * a picture of the backdrop instead, and all of them would still pass.
 */

/**
 * What a canvas is currently showing, reduced to something comparable.
 *
 * A colour *count* is not enough on its own to tell two fields apart: the Envelope fills every cell
 * from a small fixed set — the seven steps of `magnitudeRamp` plus the flat `serious` and `critical`
 * status hues, of which the default scenario paints five — so the count is dominated by cell-edge
 * anti-aliasing and moves by one or two between wholly different scenarios. (It was a three-state
 * palette when this was written and "five-colour" was the figure; #65 put the field on the ramp, and
 * the reasoning survived the change while the number did not. A tolerance sized from a stale figure
 * is exactly the mis-measurement this helper exists to prevent.) The first version of the repaint
 * test below turned on exactly that difference of one, which would have become a five-second poll
 * timeout reporting `expected true, received false` the moment a catalogue change made two counts
 * collide.
 *
 * So this also returns a `digest` — an order-sensitive hash over the sampled bytes — which changes
 * if any sampled pixel changes. The counts stay because they are what a *human* reads out of a
 * failure; the digest is what the assertion compares.
 *
 * `opaqueTop` counts only the pixels in the upper half, and exists because `opaque` and `colours`
 * cannot tell the masthead's intro from a dead one. Its bottom fade is painted unconditionally, at
 * every value of the animation clock including zero — and being an alpha ramp it reads back through
 * `getImageData` as many distinct RGB triplets, because the bitmap stores premultiplied alpha and
 * un-premultiplying rounds. So a backdrop frozen on its first frame still reports thousands of
 * opaque pixels in dozens of colours. Everything that actually depends on the clock — the bloom and
 * the lattice — is the only thing that puts ink above the fade.
 */
async function painted(canvas: import('@playwright/test').Locator) {
  return canvas.evaluate((el: HTMLCanvasElement) => {
    const ctx = el.getContext('2d');
    if (!ctx) return { error: 'no 2d context' as const };
    if (el.width === 0 || el.height === 0) return { error: 'zero-sized bitmap' as const };

    const { data } = ctx.getImageData(0, 0, el.width, el.height);
    const colours = new Set<string>();
    let opaque = 0;
    let opaqueTop = 0;
    let digest = 0;
    // Comfortably above the fade, which starts at 55% of the height.
    const topRows = Math.floor(el.height * 0.5);
    // Every 4th pixel: enough to characterise a grid of flat-filled cells, cheap enough to run on
    // a retina-scaled bitmap without serialising a megabyte back across the bridge.
    for (let i = 0; i < data.length; i += 16) {
      digest = (digest * 31 + data[i] + data[i + 1] * 7 + data[i + 2] * 13 + data[i + 3] * 17) | 0;
      if (data[i + 3] === 0) continue;
      opaque += 1;
      if (Math.floor(i / 4 / el.width) < topRows) opaqueTop += 1;
      colours.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return { colours: colours.size, opaque, opaqueTop, digest, area: el.width * el.height };
  });
}

test('the Envelope actually paints its feasibility field', async ({ page }) => {
  await page.goto('/');

  const canvas = page.getByRole('region', { name: /how much room/i }).locator('canvas');
  await expect(canvas).toBeVisible();

  const box = await canvas.boundingBox();
  expect(box!.width).toBeGreaterThan(0);
  expect(box!.height).toBeGreaterThan(0);

  const field = await painted(canvas);
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

  const before = await painted(canvas);
  // Checked here as well as in the test above: without it a zero-sized bitmap surfaces below as an
  // uninformative poll timeout rather than as the thing that actually went wrong.
  expect(before, 'the canvas could not be read back').not.toHaveProperty('error');
  if ('error' in before) return; // unreachable past that assertion; narrows the union for TS

  /**
   * And that the baseline was painted in the first place — the symmetric case of the blank-`after`
   * hole closed below. `painted` only reports an error for a *zero*-sized bitmap, so a canvas read
   * before its first draw (300×150 and fully transparent, the HTML default) clears the check above,
   * after which any subsequent paint reads as a successful repaint and the `selectOption` has
   * proven nothing. Unreachable today, since `goto` waits for load and `toBeVisible` adds another
   * round trip, but it costs a line to stop it becoming reachable.
   */
  expect(before.opaque, 'the canvas was already blank before the scenario changed').toBeGreaterThan(
    0
  );

  // A model that changes the field wholesale rather than by one cell.
  // `exact`, because the Matrix section's accessible name is "Every model on every machine…" and a
  // loose label match resolves to two elements.
  await page.getByLabel('Model', { exact: true }).selectOption('deepseek-ai/DeepSeek-V3');

  /**
   * A changed digest is necessary but not sufficient. Clearing the canvas and then failing to
   * redraw changes it too — a transparent bitmap hashes differently from a painted one — so a
   * digest comparison on its own accepts a blank Envelope as a successful repaint. The paint test
   * above only ever sees a fresh page, which leaves a draw that fails for one particular model
   * with nothing watching it.
   *
   * Reported as a state name rather than a boolean because these fail in different ways, and a
   * poll timeout reading `expected true, received false` distinguishes none of them.
   */
  await expect
    .poll(
      async () => {
        const after = await painted(canvas);
        if ('error' in after || 'error' in before) return 'unreadable';
        if (after.digest === before.digest) return 'unchanged';
        if (after.opaque === 0) return 'cleared';
        if (after.colours <= 1) return 'flat';
        return 'repainted';
      },
      { message: 'the canvas never repainted into a painted state after the model changed' }
    )
    .toBe('repainted');
});

/**
 * And that the measure toggle repaints it (#65).
 *
 * The grading itself is asserted in `src/components/Envelope.test.tsx`, which reads the draw loop's
 * own `fillStyle` calls back through a stubbed 2D context. What that cannot see is the bitmap: the
 * effect sizes the canvas from `getBoundingClientRect` before it draws anything, and jsdom reports
 * that as 0×0 — so a repaint that clears the canvas and puts nothing legible back is
 * indistinguishable there from a correct one. This is the same claim, and the same three failure
 * names, that the model-change test above makes for the other trigger.
 *
 * Scoped to the Envelope's own region, since the Matrix carries the same three buttons.
 */
test('the Envelope repaints when the measure changes', async ({ page }) => {
  await page.goto('/');
  const field = page.getByRole('region', { name: /how much room/i });
  const canvas = field.locator('canvas');
  await expect(canvas).toBeVisible();

  const before = await painted(canvas);
  expect(before, 'the canvas could not be read back').not.toHaveProperty('error');
  if ('error' in before) return; // unreachable past that assertion; narrows the union for TS
  expect(before.opaque, 'the canvas was blank before the measure changed').toBeGreaterThan(0);

  // Decode rather than latency: it varies along both axes at the default scenario, so the two
  // pictures differ over the whole field rather than in one corner.
  await field.getByRole('button', { name: 'How fast' }).click();

  await expect
    .poll(
      async () => {
        const after = await painted(canvas);
        if ('error' in after) return 'unreadable';
        if (after.digest === before.digest) return 'unchanged';
        if (after.opaque === 0) return 'cleared';
        if (after.colours <= 1) return 'flat';
        return 'repainted';
      },
      { message: 'the field never repainted after the measure changed' }
    )
    .toBe('repainted');
});

/**
 * The bitmap is sized from `getBoundingClientRect` times the device pixel ratio, in an effect that
 * runs after layout. Getting that wrong gives a stretched or blank plot rather than an error.
 */
test('the Envelope bitmap matches its laid-out size', async ({ page }) => {
  await page.goto('/');
  const canvas = page.getByRole('region', { name: /how much room/i }).locator('canvas');
  await expect(canvas).toBeVisible();

  // Measured through the locator asserted visible above, not a fresh `document.querySelector`.
  // The masthead's backdrop is now exactly the second canvas this anticipated, and it lands
  // earlier in the DOM — a bare selector here would assert about an element it never checked.
  const { cssWidth, cssHeight, bitmapWidth, bitmapHeight, dpr } = await canvas.evaluate(
    (el: HTMLCanvasElement) => {
      const rect = el.getBoundingClientRect();
      return {
        cssWidth: rect.width,
        cssHeight: rect.height,
        bitmapWidth: el.width,
        bitmapHeight: el.height,
        dpr: window.devicePixelRatio || 1,
      };
    }
  );

  expect(bitmapWidth).toBeGreaterThan(0);
  expect(bitmapHeight).toBeGreaterThan(0);
  expect(Math.abs(bitmapWidth - cssWidth * dpr)).toBeLessThanOrEqual(2);
  /**
   * Height as well as width, which is the half that actually catches the stretch described above.
   * The CSS box is a fixed `h-48`, so a bitmap height left at the 150px default — or stale from a
   * previous layout — while the width keeps tracking the rect scales the plot vertically. Every
   * assertion in this file still finds nonzero, multicoloured, repainting pixels in that state.
   */
  expect(Math.abs(bitmapHeight - cssHeight * dpr)).toBeLessThanOrEqual(2);
});

/**
 * And the masthead's backdrop, which has the same failure mode and one the Envelope does not: it
 * animates in over ~700ms from a frame that is legitimately almost blank, so "it painted" and "the
 * intro ran to completion" are different claims and only the second one is worth anything.
 */
test('the masthead actually paints its backdrop', async ({ page }) => {
  await page.goto('/');

  // Through the banner landmark rather than a positional selector: it is the semantic that puts
  // this canvas outside <main>, and reaching it any other way would keep passing if that broke.
  const canvas = page.getByRole('banner').locator('canvas');
  await expect(canvas).toBeVisible();

  /**
   * Polled, not read once: a single read taken the moment the page loads can legitimately catch
   * the first frame, where the bloom is still at zero alpha and the lattice unlit.
   *
   * And discriminated on `opaqueTop`, not on `opaque` or `colours`. Those two are the obvious
   * choice and they are worthless here — the bottom fade satisfies both at every value of the
   * animation clock, so an intro that never advanced past frame zero would poll straight to
   * "painted" and this test would go green on exactly the regression it exists to catch. Ink above
   * the fade line is the one thing only the bloom and the lattice can put there.
   */
  await expect
    .poll(
      async () => {
        const back = await painted(canvas);
        if ('error' in back) return 'unreadable';
        if (back.opaque === 0) return 'blank';
        if (back.opaqueTop === 0) return 'fade-only';
        return 'painted';
      },
      { message: 'the masthead backdrop never settled into a painted state' }
    )
    .toBe('painted');
});

/**
 * The masthead is a `banner` landmark holding the page's `h1`, and its backdrop is decoration.
 *
 * Both halves are structural claims that nothing else checks. The landmark exists only because the
 * <header> sits outside <main> — nest it back inside and the role silently disappears, taking the
 * "skip to header" affordance with it and failing no other test in the suite.
 */
test('the masthead is a banner landmark and its backdrop is not announced', async ({ page }) => {
  await page.goto('/');

  const banner = page.getByRole('banner');
  await expect(banner.getByRole('heading', { level: 1 })).toHaveText('headroom');

  // The Envelope's canvas is `role="img"` carrying a described plot. This one encodes nothing, so
  // it must not reach the accessibility tree at all — announcing "image" here describes a gradient.
  await expect(banner.getByRole('img')).toHaveCount(0);
});

/**
 * The backdrop still paints with reduced motion asked for.
 *
 * This is the branch the blanket CSS rule in index.css cannot reach — it collapses declared
 * animations and transitions, and a `requestAnimationFrame` loop is neither. The component asks
 * `matchMedia` itself and jumps to the settled frame instead of animating to it, so the failure
 * this guards is not a stutter: get the branch wrong and a reduced-motion reader gets a masthead
 * that never paints at all, which is the one outcome worse than the animation they opted out of.
 */
test('the masthead backdrop paints without animating under reduced motion', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');

  const canvas = page.getByRole('banner').locator('canvas');
  await expect(canvas).toBeVisible();

  /*
   * Wait for it to be painted at all — on `opaqueTop`, so the unconditional bottom fade cannot
   * answer for the bloom and the lattice. The draw runs in a passive effect rather than a layout
   * one, so there is a real window between the canvas being visible and the first paint; asserting
   * a single read here would race it and flake as blank on a loaded CI machine.
   */
  await expect
    .poll(
      async () => {
        const back = await painted(canvas);
        return 'error' in back ? -1 : back.opaqueTop;
      },
      { message: 'the backdrop never painted at all under reduced motion' }
    )
    .toBeGreaterThan(0);

  /*
   * Then that it is *static*, which is the claim this test is actually for and the one polling
   * alone can never make. The intro runs 700ms, so two reads a fraction of that apart differ while
   * it is animating and match once it is not. Reading the settled frame straight out of the
   * `matchMedia` branch is what makes them match here.
   */
  const first = await painted(canvas);
  await page.waitForTimeout(150);
  const second = await painted(canvas);

  expect(first, 'the backdrop could not be read back').not.toHaveProperty('error');
  if ('error' in first || 'error' in second) return; // narrows the union past the assertion above

  expect(second.digest, 'the backdrop is still animating with motion reduced').toBe(first.digest);
  expect(second.opaqueTop, 'the backdrop went blank between reads').toBeGreaterThan(0);
});
