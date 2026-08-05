import { routePath, type PrerenderRoute } from '@/data/routes';

/**
 * `sitemap.xml`, from the same route list the pages are written from.
 *
 * Generated rather than committed, and generated from {@link prerenderRoutes} rather than by
 * walking `dist/` afterwards, so the two cannot drift: a sitemap listing a page that was never
 * written is a soft 404 in a crawler's report, and a page that was written and never listed is the
 * work of #178 done and not advertised. One definition, and the tests that constrain the route
 * list constrain this too.
 *
 * **`<loc>` only.** The protocol's other three elements are all optional and all worse than
 * absent here. `lastmod` would have to be a lie or a guess — the catalog's `generatedAt` moves on
 * a refresh but not on an engine correction or a copy change, so quoting it tells a crawler
 * nothing moved when the figures did. `changefreq` and `priority` are hints Google has said for
 * years it ignores.
 */

/** XML's five, since a `<loc>` is character data and the protocol names all five explicitly. */
function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * The sitemap for a build, or `null` when there is no sitemap to write.
 *
 * **`null` when the origin is unset, and that is a decision rather than a gap.** The protocol
 * requires `<loc>` to be a complete URL beginning with a scheme; a root-relative path is not one,
 * and a sitemap full of them is not a lesser sitemap but an invalid file that a crawler reports as
 * a parse error. The other two ways out are worse: inventing `<owner>.github.io` is the guess
 * `PAGES_SITE_ORIGIN` exists to stop, and failing the build would break every fork, every
 * pull-request build and `npm run build` on a laptop over a setting that only matters once
 * something is published. A build with no publishing address has nothing to submit to a crawler,
 * so it submits nothing — the same call `pageHtml` makes when it omits `og:url`.
 */
export function sitemapXml(
  routes: readonly PrerenderRoute[],
  base: string,
  origin: string
): string | null {
  if (!origin) return null;
  const locs = routes
    .filter((route) => route.indexable)
    .map((route) => `  <url><loc>${escapeXml(`${origin}${routePath(route, base)}`)}</loc></url>`);
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...locs,
    '</urlset>',
    '',
  ].join('\n');
}
