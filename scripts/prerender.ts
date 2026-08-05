import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/**
 * Writes the site as real HTML files, one per route, plus the sitemap that advertises them.
 *
 * Runs after both Vite builds and reads their output: `dist/index.html` for the shell — the
 * hashed asset URLs, the meta tags, the whole document — and `dist-ssr/entry-server.js` for the
 * app, which it renders once per route and injects into that shell. Nothing here knows how to
 * render React, where the assets live, or what a page should say; it is I/O, two caps, and one
 * content check, over functions the SSR bundle exports and `src/prerender/` tests.
 *
 * **Why a script rather than a plugin.** Requirement 3 of
 * [#178](https://github.com/MrZoller/headroom/issues/178) is that the route list derive from the
 * catalog and be cap-checked *before anything is written*. That wants a plain import and a
 * `process.exit(1)`, not a value returned from inside a render — which is what every prerender
 * plugin gives you, and why the only maintained one (`vite-prerender-plugin`) was still the wrong
 * trade for about sixty lines.
 *
 * **Nothing here hardcodes `/headroom/`.** Every emitted path is built from `BASE_PATH`, the same
 * environment variable `deploy.yml` feeds Vite's `base` from, and the SSR bundle was built with
 * the same value — so `import.meta.env.BASE_URL` agrees on both sides of the hydrate. The client's
 * asset URLs are absolute and base-prefixed, which is what makes a page at one level of nesting
 * resolve them identically to one at the root.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(HERE, '..', 'dist');
const SSR_ENTRY = resolve(HERE, '..', 'dist-ssr', 'entry-server.js');

const BASE_PATH = process.env.BASE_PATH || '/';

/**
 * The origin the site is published at, from the `PAGES_SITE_ORIGIN` repository variable.
 *
 * Empty is supported and means "not published anywhere yet": the canonical link goes
 * root-relative, `og:url` is omitted, and no `sitemap.xml` is written at all. The reasoning for
 * each is at its own seam — `src/prerender/page.ts` and `src/prerender/sitemap.ts`.
 */
const SITE_ORIGIN = process.env.SITE_ORIGIN ?? '';

/**
 * The most pages this build is allowed to write.
 *
 * Not a round number picked for comfort. GitHub Pages hard-limits a published site to 1 GB, and a
 * page of this app measures about 820 KiB of HTML — 94% of it the comparison grid's cells — which
 * puts the real ceiling near 1,250 pages. 400 leaves a 3x margin, and it is the number that stops
 * a catalog which doubles from silently quadrupling the model x device tier, since that tier is a
 * product of two axes that both grow.
 *
 * The full tiered list is 199 of it: 1 root, 43 devices, 35 models, 120 pairs. A catalog that
 * doubled on both axes would be 1 + 86 + 70 + 120 = 277 — still inside, because the pair tier is a
 * fixed 10 x 12 by construction and does not grow with the catalog. What would cross this is
 * somebody widening that shortlist, which is exactly the change worth stopping to think about.
 */
const MAX_ROUTES = 400;

/**
 * The most bytes `dist/` is allowed to reach, checked because page weight moves independently of
 * page count: 400 routes is ~320 MiB today and would be far more if the grid grew. Both caps bind
 * at roughly the same place, which is the point of having two.
 */
const MAX_DIST_BYTES = 512 * 1024 * 1024;

interface RenderedPage {
  /** Path below `dist/`, e.g. `rtx-5090/index.html`. */
  readonly file: string;
  readonly url: string;
  readonly tier: number;
  readonly html: string;
}

interface Route {
  readonly segments: readonly string[];
  readonly tier: number;
  readonly config: Record<string, unknown>;
  readonly title: string;
  readonly description: string;
  readonly indexable: boolean;
}

interface ServerBundle {
  prerenderRoutes: () => readonly Route[];
  routePath: (route: Route, base: string) => string;
  renderRoute: (config: Record<string, unknown>) => string;
  pageHtml: (shell: string, route: Route, url: string, body: string, origin: string) => string;
  missingFigures: (html: string) => readonly string[];
  sitemapXml: (routes: readonly Route[], base: string, origin: string) => string | null;
}

function fail(message: string): never {
  console.error(`prerender: ${message}`);
  process.exit(1);
}

/** Total bytes of a directory tree, so the cap is checked against what is really there. */
async function treeBytes(dir: string): Promise<number> {
  let total = 0;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) total += await treeBytes(full);
    else total += (await stat(full)).size;
  }
  return total;
}

/** The lowest tier at which a running total crosses a cap — which is the tier to shrink. */
function overflowingTier(pages: readonly RenderedPage[], cap: number, cost: 'count' | 'bytes') {
  const tiers = [...new Set(pages.map((page) => page.tier))].sort((a, b) => a - b);
  let running = 0;
  for (const tier of tiers) {
    const inTier = pages.filter((page) => page.tier === tier);
    running +=
      cost === 'count'
        ? inTier.length
        : // Bytes rather than characters, so this agrees with the projection that tripped the cap.
          // The pages are full of non-ASCII — every em dash and every `×` is three bytes — so the
          // two differ by enough to name the wrong tier.
          inTier.reduce((sum, page) => sum + Buffer.byteLength(page.html), 0);
    if (running > cap) return tier;
  }
  return tiers[tiers.length - 1] ?? 0;
}

/** What each tier contributed, so the log says what was built rather than only how much. */
function byTier(pages: readonly RenderedPage[]): string {
  const tiers = [...new Set(pages.map((page) => page.tier))].sort((a, b) => a - b);
  return tiers
    .map((tier) => `tier ${tier}: ${pages.filter((page) => page.tier === tier).length}`)
    .join(', ');
}

async function main(): Promise<void> {
  const bundle = (await import(pathToFileURL(SSR_ENTRY).href)) as ServerBundle;
  const routes = bundle.prerenderRoutes();

  // Checked before a single render, because a route list that is already too long is a mistake
  // about the catalog and not about the output.
  if (routes.length > MAX_ROUTES) {
    const tier = overflowingTier(
      routes.map((route) => ({ file: '', url: '', tier: route.tier, html: '' })),
      MAX_ROUTES,
      'count'
    );
    fail(
      `${routes.length} routes exceeds MAX_ROUTES=${MAX_ROUTES}; tier ${tier} is where the ` +
        'total crosses it. Narrow that tier in src/data/routes.ts — do not raise the cap without ' +
        'redoing the arithmetic in this file, which is anchored on the 1 GB GitHub Pages limit.'
    );
  }

  const shell = await readFile(join(DIST, 'index.html'), 'utf8');
  /**
   * This is not idempotent, and says so rather than failing obscurely.
   *
   * The root route overwrites `dist/index.html` with its own rendered page, so a second run has no
   * shell left to inject into and every pattern below stops matching. Inside `npm run build` that
   * never happens — `vite build` rewrites the shell first — but running this alone twice is an
   * easy thing to do while working on it, and "expected exactly one title" is a poor way to learn
   * why.
   */
  if (shell.includes('data-prerendered')) {
    fail(
      'dist/index.html is already a prerendered page, so there is no shell to inject into. ' +
        'Run `vite build` first — this script overwrites the root and cannot be run twice.'
    );
  }

  const pages: RenderedPage[] = routes.map((route) => {
    const url = bundle.routePath(route, BASE_PATH);
    let html: string;
    try {
      html = bundle.pageHtml(shell, route, url, bundle.renderRoute(route.config), SITE_ORIGIN);
    } catch (error) {
      fail(`${url}: ${error instanceof Error ? error.message : String(error)}`);
    }
    /**
     * The check that stops this shipping the thing it was built to replace.
     *
     * A page missing its figures is a page of layout chrome, which is what the live site already
     * was — and it arrives looking like a success: right filename, right size, right shape. Every
     * page is asked, rather than the handful a unit test renders being trusted for the rest.
     */
    const missing = bundle.missingFigures(html);
    if (missing.length > 0) {
      fail(
        `${url} carries no ${missing.join(', no ')}. A page without figures is the empty shell ` +
          '#178 exists to replace, so this is a build failure rather than a page.'
      );
    }
    return { file: join(...route.segments, 'index.html'), url, tier: route.tier, html };
  });

  const newBytes = pages.reduce((sum, page) => sum + Buffer.byteLength(page.html), 0);
  /**
   * What `dist/` will hold once this finishes.
   *
   * The shell counts once either way: `index.html` is overwritten by the root page, which is
   * already in `newBytes`, and the same bytes reappear as `404.html`. So the projection is simply
   * what Vite left plus every page written here — and it is checked *before* the first write, so
   * a build that would overshoot leaves no half-published directory behind.
   */
  const projected = (await treeBytes(DIST)) + newBytes;
  if (projected > MAX_DIST_BYTES) {
    const tier = overflowingTier(pages, MAX_DIST_BYTES, 'bytes');
    fail(
      `${(projected / 1024 / 1024).toFixed(1)} MiB of output exceeds MAX_DIST_BYTES=` +
        `${(MAX_DIST_BYTES / 1024 / 1024).toFixed(0)} MiB; tier ${tier} is where the total ` +
        'crosses it. GitHub Pages refuses a site over 1 GB, so this is a real ceiling and not a ' +
        'style rule.'
    );
  }

  for (const page of pages) {
    const target = join(DIST, page.file);
    // The segments are catalog ids and model slugs, both hand-curated or derived from a curated
    // id: one containing `..` would write outside the build directory. Cheap to check, and
    // impossible to notice if not.
    const inside = relative(DIST, target);
    if (inside.startsWith('..') || isAbsolute(inside)) {
      fail(`route ${page.url} would write outside dist/ (${target})`);
    }
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, page.html);
  }

  /**
   * The fallback, and it is the *shell* rather than a prerendered page on purpose.
   *
   * Pages serves a real file first, so a prerendered route answers with 200 and only genuinely
   * unknown paths reach this. Those still have to boot the app — the store reads `location` and
   * renders whatever the address asks for — but they must not claim to be a device page: a
   * prerendered copy here would make every unknown URL on the site an RTX 5090 page. It is served
   * with a 404 status, so nothing in it is discoverable; it exists for a human who followed a
   * stale link, and for nothing else.
   */
  await writeFile(join(DIST, '404.html'), shell);

  const sitemap = bundle.sitemapXml(routes, BASE_PATH, SITE_ORIGIN);
  if (sitemap) await writeFile(join(DIST, 'sitemap.xml'), sitemap);

  const listed = routes.filter((route) => route.indexable).length;
  console.log(
    `prerender: ${pages.length} routes (${byTier(pages)}) + 404.html, ` +
      `${(newBytes / 1024 / 1024).toFixed(1)} MiB\n` +
      (sitemap
        ? `prerender: sitemap.xml lists ${listed} of ${routes.length}`
        : 'prerender: no sitemap.xml — SITE_ORIGIN is unset, and a sitemap needs absolute URLs')
  );
}

await main();
