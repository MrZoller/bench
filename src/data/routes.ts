import { DEVICES, canonicalDeviceId, getDevice, type CatalogDevice } from '@/data/catalog';
import { GB, GIB } from '@/engine/types';
import type { Config } from '@/store/scenario';

/**
 * The paths the site is built as files for, and how to read one back.
 *
 * Two directions of the same fact, deliberately in one module ([#178](https://github.com/MrZoller/headroom/issues/178)):
 *
 *   - {@link prerenderRoutes} is what `scripts/prerender.ts` writes to disk, and what Phase 3's
 *     `sitemap.xml` and its content regression test will read. Three consumers, one definition,
 *     so they cannot disagree about which pages exist. This mirrors `comparisonGrid()` in
 *     `catalog.ts`, which exists for the same reason and whose docblock argues the case.
 *   - {@link configFromPath} is the inverse, and it runs in the browser: the store reads it at
 *     import time so a visitor landing on `/rtx-5090/` gets the RTX 5090 rather than the default
 *     device. A separate parser would be a second definition of the same mapping and would drift
 *     from the first the day a route shape changed — and the failure would be silent, because an
 *     unparsed path falls back to the default scenario and simply shows the wrong machine.
 *
 * **A path is a lossy entry point; the querystring is the lossless encoding.** `Config` is nine
 * fields (`src/store/scenario.ts`) and `/<device>/` carries one; the other eight come from
 * `DEFAULT_CONFIG`. That is the framing rather than a shortcoming, and it is what settles the
 * precedence rule stated on {@link configFromPath}: a query wins over a path, because the query
 * names an exact scenario and the path names *a scenario worth having a page for*.
 *
 * The `Config` import is type-only and therefore erased: nothing in `src/data/` depends on the
 * store at runtime, and the shape is the only thing the two genuinely share — the same reason
 * `scenario.ts` sits apart from both the store and the URL codec.
 */

/**
 * Which layer of the route inventory a page belongs to.
 *
 * Phase 2 builds tiers 0 and 1 only. Phase 3 adds `/m/<model-slug>/` and
 * `/<device>/<model-slug>/`; the tier travels on the route so `scripts/prerender.ts` can name the
 * overflowing layer when a cap trips, rather than reporting a bare total that says nothing about
 * which addition caused it.
 */
export type RouteTier = 0 | 1;

export interface PrerenderRoute {
  /**
   * Path segments below the site's base, without slashes. Empty for the root.
   *
   * Segments rather than a joined path because the base is not known here — it is
   * `import.meta.env.BASE_URL` in the browser and `process.env.BASE_PATH` in the build script,
   * and nothing in this repo may hardcode either. {@link routePath} joins the two.
   */
  readonly segments: readonly string[];
  readonly tier: RouteTier;
  /** The scenario fields this path names. Everything absent comes from `DEFAULT_CONFIG`. */
  readonly config: Partial<Config>;
  readonly title: string;
  readonly description: string;
}

/**
 * The devices Phase 2 prerenders: one per device class, named rather than sliced.
 *
 * **Three, and one would not do.** One route proves `renderToString` runs; three prove the two
 * things that actually break. That per-route injection *varies* the output — with a single route
 * "injected the config" and "rendered the default" are the same bytes, which is precisely the
 * failure mode of a store whose initial state is read from `window.location` at import time. And
 * that routes in one process do not leak into each other — the store is a module-level singleton,
 * so route 2 renders whatever route 1 left behind unless the scenario is replaced every time.
 * Three is the smallest slice that *can* fail.
 *
 * **One per class, because class is what makes the pages differ.** `class` drives the runtime
 * filter, the quant fallback in the store's `coerce`, and whether a rig can shard at all; three
 * discrete GPUs would render three near-identical pages and prove strictly less than these do.
 *
 * **Ids rather than indices**, which is the same call `src/test/grid.ts` makes about the bounded
 * comparison grid and for the same reason: a slice of the catalog — "the first row of each class"
 * — silently becomes a different slice the next time `devices.json` is reordered, and row order
 * there is display order that is expected to move. Naming them means the pages the maintainer
 * asked for are the pages that get built. Each id is resolved through `getDevice`, which throws
 * on an unknown one, so a renamed device fails the build loudly instead of quietly prerendering
 * a page of default figures under the old name. `routes.test.ts` holds the rest of the
 * preconditions — that the three resolve, and that they really are three different classes.
 *
 * Phase 3 replaces this with every device, at which point there is nothing to name.
 */
const PHASE_2_DEVICE_IDS: readonly string[] = ['rtx-5090', 'dgx-spark', 'epyc-9654'];

const DEVICE_IDS = new Set(DEVICES.map((device) => device.id));

/** The root, which is the URL people actually have, and the page whose emptiness opened #178. */
function rootRoute(): PrerenderRoute {
  return {
    segments: [],
    tier: 0,
    // Empty rather than `DEFAULT_CONFIG`: the root asserts nothing about the scenario, which is
    // the same reason a bare querystring stays bare. `replace` fills it from the defaults.
    config: {},
    title: 'Headroom — what LLM runs on your hardware?',
    description:
      'Work out which open-weight LLMs run on your hardware, and how comfortably — across ' +
      'discrete GPUs, unified-memory machines, and CPU+RAM.',
  };
}

function deviceRoute(device: CatalogDevice): PrerenderRoute {
  const capacity = Math.round(device.capacityBytes / GIB);
  const bandwidth = Math.round(device.bandwidthBytesPerSec / GB);
  return {
    segments: [device.id],
    tier: 1,
    config: { deviceId: device.id },
    title: `${device.name} — what LLM runs on it? · Headroom`,
    description:
      `Which open-weight LLMs run on the ${device.name} — ${capacity} GiB at ` +
      `${bandwidth} GB/s — and how comfortably. Fit verdict, memory footprint, prefill and ` +
      'decode, computed per model rather than approximated.',
  };
}

/**
 * Every page the build writes as a file.
 *
 * Derived rather than listed: the device rows, their names and every figure in the metadata come
 * from the catalog, so a spec correction reaches the prerendered pages on the next build without
 * anybody editing this file.
 */
export function prerenderRoutes(): readonly PrerenderRoute[] {
  return [rootRoute(), ...PHASE_2_DEVICE_IDS.map((id) => deviceRoute(getDevice(id)))];
}

/**
 * The URL path a route is served at, under a given base.
 *
 * Trailing slash on everything, because that is what a directory of `index.html` files answers to
 * on GitHub Pages and what the canonical link therefore has to say.
 */
export function routePath(route: PrerenderRoute, base: string): string {
  const prefix = base.endsWith('/') ? base : `${base}/`;
  return route.segments.length === 0 ? prefix : `${prefix}${route.segments.join('/')}/`;
}

/**
 * The scenario a pathname names, if it names one.
 *
 * Total, like the querystring reader it sits beside: a path a stranger typed, a stale link, or
 * the 404 fallback answering some arbitrary depth all return `{}` rather than throwing, and the
 * caller fills the rest from `DEFAULT_CONFIG`. Returning a *partial* is what lets the querystring
 * override it field by field — a full `Config` here would overwrite query fields with defaults
 * and quietly invert the precedence rule.
 *
 * Aliases resolve, so an old device id in a path lands on the row it was renamed to, exactly as
 * it does in a querystring.
 */
export function configFromPath(pathname: string, base: string): Partial<Config> {
  const segments = pathSegments(pathname, base);
  if (segments.length !== 1) return {};
  const id = canonicalDeviceId(segments[0]);
  return DEVICE_IDS.has(id) ? { deviceId: id } : {};
}

/**
 * The segments of a pathname below the site's base.
 *
 * A pathname outside the base has no segments at all rather than being read from its own root:
 * under a `/headroom/` base, `/rtx-5090/` is a different site's page and reading a device out of
 * it would be this app claiming an address it is not served at.
 *
 * `index.html` is dropped because a directory URL and the file it serves are the same page, and
 * somebody arrives at the second form often enough — a saved link, a file:// open, a crawler that
 * expanded it — that treating it as an unknown extra segment would silently drop the device.
 */
function pathSegments(pathname: string, base: string): string[] {
  const prefix = base.endsWith('/') ? base : `${base}/`;
  // The base without its trailing slash is the bare site root, which Pages answers with a
  // redirect rather than a 404 — so it is the same page and not an outside address.
  if (pathname === prefix.slice(0, -1)) return [];
  if (!pathname.startsWith(prefix)) return [];
  const segments = pathname
    .slice(prefix.length)
    .split('/')
    .filter((segment) => segment !== '');
  if (segments[segments.length - 1] === 'index.html') segments.pop();
  return segments.map(decodeSegment);
}

/**
 * `decodeURIComponent` throws on a lone `%`, and this reads a URL a stranger may have edited by
 * hand — so a malformed escape reads as an unknown segment rather than as an exception thrown
 * out of the store's module initializer, which is a blank page.
 */
function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}
