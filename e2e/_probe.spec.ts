import { test, chromium } from '@playwright/test';
test('blink-settings defaultFontSize', async () => {
  for (const size of [16, 32]) {
    const browser = await chromium.launch({ args: [`--blink-settings=defaultFontSize=${size}`] });
    for (const width of [320, 640, 1280]) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      await page.goto('http://127.0.0.1:4173/');
      const r = await page.evaluate(() => ({
        root: getComputedStyle(document.documentElement).fontSize,
        sm: matchMedia('(min-width: 40rem)').matches,
        lg: matchMedia('(min-width: 64rem)').matches,
        h1: getComputedStyle(document.querySelector('h1')!).fontSize,
        doc: `${document.documentElement.scrollWidth}/${document.documentElement.clientWidth}`,
      }));
      console.log(`### font=${size} w=${width} → ${JSON.stringify(r)}`);
      await page.close();
    }
    await browser.close();
  }
});
