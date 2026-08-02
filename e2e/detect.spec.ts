import { expect, test } from '@playwright/test';

/**
 * Machine detection, and the one half of it a browser suite can assert honestly (#137).
 *
 * **The live path is host-dependent and is deliberately not asserted.** Playwright's WebGPU story
 * varies by platform: headless Chromium may expose no adapter at all, may expose a software one,
 * and reports different vendor and architecture strings on a Mac, a Linux runner and a Windows box.
 * A spec asserting "detection found an NVIDIA card" would be a description of the machine that ran
 * it — the mistake `reflow.spec.ts` records at length about measuring one machine's typography.
 *
 * What *is* deterministic is the **fallback**, and it is also the branch that matters most: it is
 * what every Safari reader and every hardened browser gets, and the issue's first trap is that
 * whatever ships must degrade to the picker "without a console error". Both halves of that are
 * checked here, and the console half is the one no unit test can see.
 *
 * The mapping itself lives in `src/lib/detect.test.ts`, against recorded signals, where it runs in
 * a millisecond and does not depend on what silicon the runner has.
 */

test('with no adapter, detection names the picker and logs nothing', async ({ page }) => {
  const noise: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') noise.push(message.text());
  });
  page.on('pageerror', (error) => noise.push(String(error)));

  // Removed before any script runs, which is the state Safari without the flag is in. Deleting it
  // after load would leave whatever the page had already read.
  await page.addInitScript(() => {
    Reflect.deleteProperty(navigator, 'gpu');
    Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
  });
  await page.goto('/');

  const detect = page.getByRole('button', { name: /what can my machine run/i });
  await expect(detect).toBeVisible();
  await detect.click();

  await expect(page.getByText(/exposes no graphics adapter/i)).toBeVisible();

  /**
   * The picker still works, which is the actual promise. A detection affordance that degrades to a
   * dead control has not degraded to anything.
   */
  const hardware = page.getByLabel('Hardware', { exact: true });
  await expect(hardware).toBeEnabled();
  await expect(hardware.locator('option')).not.toHaveCount(0);

  expect(noise, 'the fallback path wrote to the console').toEqual([]);
});

/**
 * And the affordance is reachable by keyboard and hits the coarse-pointer floor.
 *
 * Its own test rather than a clause in the sweep, because this control is the one a reader who does
 * *not* know their hardware depends on — the same argument `DisclosureToggle` records for holding
 * the table toggles to 44 rather than to 24. `touch-targets.spec.ts` sweeps it too; this says why
 * the number is what it is.
 */
test('the detect button is a real target and a real tab stop', async ({ page }) => {
  await page.goto('/');
  const detect = page.getByRole('button', { name: /what can my machine run/i });

  const box = await detect.boundingBox();
  expect(box, 'the detect button is not laid out').not.toBeNull();
  expect(box!.height, 'the detect button height').toBeGreaterThanOrEqual(44);

  await detect.focus();
  await expect(detect).toBeFocused();
});
