import { expect, test } from '@playwright/test';

/**
 * `color-scheme: dark` is what the UA paints its *own* surfaces from — slider range tracks,
 * Chromium's select popups on Windows and Linux, classic scrollbars, form-control base colours.
 * None of that is reachable from jsdom, which applies no stylesheets: a unit test could only
 * grep the CSS text, and a rule that fails to parse or is overridden would still pass it. The
 * computed value on the root element is the claim that matters, and only a browser can answer
 * it (#133).
 */
test('the root declares a dark colour scheme for UA-painted surfaces', async ({ page }) => {
  await page.goto('/');
  const scheme = await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme);
  expect(scheme).toBe('dark');
});
