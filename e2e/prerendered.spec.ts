import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

/**
 * That a browser with JavaScript switched off paints the figures — the half of #178 nothing else
 * can answer.
 *
 * The unit suite has two guards and neither reaches this. `src/prerender/page.test.ts` asserts the
 * composed HTML carries the numbers, which passes on markup a browser would never paint.
 * `src/App.hydration.test.tsx` asserts React keeps that markup, which passes on a page that was an
 * empty shell to begin with. What is left is the question the issue was opened about — what a
 * client that does not execute the bundle actually gets — and jsdom cannot answer it, because
 * jsdom has no rendering at all.
 *
 * **Two assertions per route, because either one alone is satisfiable by the wrong thing.** The
 * served bytes are fetched and matched, which is what a crawler sees and cannot be produced by the
 * app running; and the painted element is measured, which is what a reader sees and cannot be
 * produced by markup that is present but clipped to nothing. `toBeVisible()` is deliberately not
 * the second one: Playwright calls a 1px `sr-only` box visible, and this repo has already shipped
 * a test that passed on hidden text for exactly that reason (see ROADMAP, Tests).
 *
 * **This must not be used to prove the 404 fallback.** `playwright.config.ts` serves from
 * `vite preview`, which SPA-falls-back on an unknown path where GitHub Pages hard-404s — so every
 * route here is one the build genuinely wrote, and the fallback is checked against `dist/` on disk
 * at the bottom of this file instead.
 */

const DIST = fileURLToPath(new URL('../dist', import.meta.url));

/** A page directory the build really wrote, at a given depth, so no route is named from memory. */
function builtRoute(depth: 1 | 2): string {
  const dirs = (at: string) =>
    readdirSync(at, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && entry.name !== 'assets')
      .map((entry) => entry.name)
      .sort();

  for (const first of dirs(DIST)) {
    if (depth === 1) {
      if (existsSync(join(DIST, first, 'index.html'))) return `/${first}/`;
      continue;
    }
    for (const second of dirs(join(DIST, first))) {
      if (existsSync(join(DIST, first, second, 'index.html'))) return `/${first}/${second}/`;
    }
  }
  throw new Error(`dist/ has no prerendered route ${depth} level(s) deep`);
}

/** The verdict tile with a given heading — the panel whose whole content is a computed number. */
function tile(page: Page, heading: string) {
  return page
    .locator('article')
    .filter({ has: page.getByRole('heading', { name: heading, exact: true }) });
}

/**
 * That an element is not merely in the DOM but occupies a readable box.
 *
 * Width and height are both measured because the two hiding mechanisms differ: `sr-only` clips to
 * roughly a pixel wide, and a collapsed container keeps its width while losing its height. The
 * thresholds are far below anything a real figure renders at — the tiles here measure ~100 x 30 —
 * and far above what either failure mode leaves behind.
 */
async function isReallyPainted(selector: ReturnType<typeof tile>) {
  await expect(selector).toHaveCount(1);
  const box = await selector.boundingBox();
  expect(box, 'the element has no box at all').not.toBeNull();
  expect(box!.width, 'painted width').toBeGreaterThan(40);
  expect(box!.height, 'painted height').toBeGreaterThan(16);
  expect(box!.width * box!.height, 'painted area').toBeGreaterThan(1000);
  return box!;
}

test.describe('with JavaScript disabled', () => {
  test.use({ javaScriptEnabled: false });

  test('a prerendered device page serves and paints its figures', async ({ page }) => {
    const route = builtRoute(1);
    const served = await (await page.request.get(route)).text();

    // What a crawler gets. The units are required and the numbers are not, so a catalog refresh
    // does not break this while an empty shell cannot pass it.
    expect(served, `${route} was served without a memory budget`).toMatch(
      /[0-9.]+ [GTM]iB of [0-9.]+ [GTM]iB allocatable used/
    );
    expect(served).toMatch(/Weights [0-9.]+ [GTM]iB, KV cache [0-9.]+ [GTM]iB/);
    expect(served).toContain('data-prerendered');

    await page.goto(route);

    const capacity = tile(page, 'Capacity');
    await expect(capacity).toContainText(/[0-9.]+ [GTM]iB/);
    await isReallyPainted(capacity);
  });

  test('a two-level page resolves its assets and paints the same way', async ({ page }) => {
    // The nesting is the point: the asset URLs are absolute and base-prefixed, so a page one level
    // deeper has to resolve them identically. A relative base would break exactly here.
    const route = builtRoute(2);
    const served = await (await page.request.get(route)).text();
    expect(served).toMatch(/[0-9.]+ [GTM]iB of [0-9.]+ [GTM]iB allocatable used/);

    await page.goto(route);
    await isReallyPainted(tile(page, 'Capacity'));

    const stylesheet = /<link rel="stylesheet"[^>]*href="([^"]+)"/.exec(served)?.[1];
    expect(stylesheet, 'the page names no stylesheet').toBeTruthy();
    expect(stylesheet!.startsWith('/'), 'a relative asset URL breaks at depth').toBe(true);
    expect((await page.request.get(stylesheet!)).status()).toBe(200);
  });

  test('the model listing and the numbers are text, not canvas', async ({ page }) => {
    // Both canvases are drawn in effects, so with scripting off they are blank — which is correct,
    // and only acceptable because the figures they illustrate are also written as text. This is
    // the assertion that keeps it true.
    await page.goto(builtRoute(1));
    const decode = tile(page, 'Decode');
    await expect(decode).toContainText(/tok\/s per user|no speed to report/);
    await isReallyPainted(decode);
  });
});

/**
 * The fallback, checked on disk rather than over the preview server.
 *
 * `vite preview` answers an unknown path with `index.html` at 200, where GitHub Pages returns
 * `404.html` at 404 — so a request-based test here would prove nothing about the deployed site and
 * would keep passing if `404.html` stopped being written at all. What the file has to be is the
 * *un-prerendered shell*: a document that boots the app and claims no scenario, so that an unknown
 * URL does not arrive as somebody else's device page.
 */
test.describe('the built directory', () => {
  const read = (file: string) => readFileSync(join(DIST, file), 'utf8');

  test('404.html is the shell, and carries no page’s figures', () => {
    expect(existsSync(join(DIST, '404.html')), 'dist/404.html was not written').toBe(true);
    const fallback = read('404.html');

    // The marker `main.tsx` branches on is absent, so the client renders from scratch rather than
    // hydrating an empty container.
    expect(fallback).toContain('<div id="root"></div>');
    expect(fallback).not.toContain('data-prerendered');

    for (const figure of [
      /[0-9.]+ [GTM]iB of [0-9.]+ [GTM]iB allocatable used/,
      /Weights [0-9.]+ [GTM]iB, KV cache [0-9.]+ [GTM]iB/,
      /tok\/s per user/,
    ]) {
      expect(
        fallback,
        'the fallback is a prerendered page, so every unknown URL claims to be one'
      ).not.toMatch(figure);
    }

    // And it is the same shell the pages were injected into: same assets, same head, so it boots.
    const root = read('index.html');
    const assets = (html: string) =>
      [...html.matchAll(/(?:src|href)="(\/[^"]*assets\/[^"]*)"/g)].map((match) => match[1]).sort();
    expect(assets(fallback)).toEqual(assets(root));
    expect(assets(fallback).length).toBeGreaterThan(0);
  });

  /**
   * `SITE_ORIGIN` is unset for this build — `playwright.config.ts` runs a plain `npm run build` —
   * and a sitemap's `<loc>` must be a complete URL. So there is nothing valid to write, and the
   * decision is to write nothing rather than a file a crawler reports as a parse error.
   */
  test('no sitemap is written when the build has no publishing origin', () => {
    expect(process.env.SITE_ORIGIN ?? '').toBe('');
    expect(existsSync(join(DIST, 'sitemap.xml'))).toBe(false);
  });
});
