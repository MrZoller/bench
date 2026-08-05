import {
  DEVICES,
  canonicalDeviceId,
  modelIdFromSlug,
  modelSlug,
  modelsByPopularity,
  type CatalogDevice,
  type CatalogModel,
} from '@/data/catalog';
import { params, tokens } from '@/lib/format';
import { GB, GIB } from '@/engine/types';
import type { Config } from '@/store/scenario';

/**
 * The paths the site is built as files for, and how to read one back.
 *
 * Two directions of the same fact, deliberately in one module ([#178](https://github.com/MrZoller/headroom/issues/178)):
 *
 *   - {@link prerenderRoutes} is what `scripts/prerender.ts` writes to disk, what `sitemap.xml` is
 *     generated from, and what the content regression test reads. Three consumers, one definition,
 *     so they cannot disagree about which pages exist. This mirrors `comparisonGrid()` in
 *     `catalog.ts`, which exists for the same reason and whose docblock argues the case.
 *   - {@link configFromPath} is the inverse, and it runs in the browser: the store reads it at
 *     import time so a visitor landing on `/rtx-5090/` gets the RTX 5090 rather than the default
 *     device. A separate parser would be a second definition of the same mapping and would drift
 *     from the first the day a route shape changed — and the failure would be silent, because an
 *     unparsed path falls back to the default scenario and simply shows the wrong machine.
 *
 * **A path is a lossy entry point; the querystring is the lossless encoding.** `Config` is nine
 * fields (`src/store/scenario.ts`) and `/<device>/<model>/` carries two; the rest come from
 * `DEFAULT_CONFIG`. That is the framing rather than a shortcoming, and it is what settles the
 * precedence rule stated on {@link configFromPath}: a query wins over a path, because the query
 * names an exact scenario and the path names *a scenario worth having a page for*.
 *
 * **No model id and no device id is written down here.** Every route derives from `DEVICES`,
 * `modelsByPopularity()` and the class counts below, so the pages the site publishes follow the
 * catalog on the next build rather than following whoever last remembered to edit this file. Phase
 * 2 kept three named device ids because no catalog-order rule produced the set it wanted; the tier
 * rules below supersede that shortlist, so the literals are gone.
 *
 * The `Config` import is type-only and therefore erased: nothing in `src/data/` depends on the
 * store at runtime, and the shape is the only thing the two genuinely share — the same reason
 * `scenario.ts` sits apart from both the store and the URL codec.
 */

/**
 * Which layer of the route inventory a page belongs to.
 *
 * The tier travels on the route so `scripts/prerender.ts` can name the overflowing layer when a
 * cap trips, rather than reporting a bare total that says nothing about which addition caused it.
 */
export type RouteTier = 0 | 1 | 2 | 3;

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
  /**
   * Whether `sitemap.xml` advertises this page.
   *
   * A page is written either way. This is the narrower question of what the site *submits to
   * crawlers*, and it is false for exactly one thing: hardware that is not shipping. A rumoured
   * row is a specification somebody wrote down before the product existed, and the catalog's rule
   * is that such a figure stays visibly labelled — which it does on the page, in the status line
   * the device panel renders like any other. What it must not do is arrive in a search result
   * stripped of that label, as an answer to "does X run on Y" for a machine nobody can buy. So the
   * page works if someone lands on it, and nothing invites them to.
   *
   * Decided here rather than in the sitemap writer, because `status` is a catalog field and the
   * rule about it is a catalog rule — leaving the filter in the writer would put it somewhere the
   * route tests cannot reach, which is the same argument `comparisonGrid()` makes for holding its
   * own shipping filter.
   */
  readonly indexable: boolean;
}

/**
 * The segment that separates the model namespace from the device one.
 *
 * Depth alone would disambiguate `/m/<model>/` from `/<device>/<model>/` if models sat at the top
 * level too, but then a device id and a model slug would share one namespace with nothing keeping
 * them apart — and the collision would resolve to whichever branch was tried first, silently.
 * Devices keep the top level because they are what people search for; models get a prefix.
 * `catalog.test.ts` asserts no device id is this string and no device id is any model slug.
 */
const MODEL_PREFIX = 'm';

/**
 * How many devices of each class the model x device tier pairs every shortlisted model against.
 *
 * **Per class, because a flat slice of `DEVICES` would be twelve NVIDIA cards.** `devices.json` is
 * grouped by class in display order (CLAUDE.md), so its first twelve rows are all discrete GPUs
 * from one vendor — twelve pages that answer the same question at different price points, and
 * nothing at all for the unified-memory and CPU machines whose whole appeal is that a large model
 * fits. Class is also what makes the pages differ: it drives the runtime filter, the quant
 * fallback, and whether a rig can shard at all.
 *
 * **The leading rows of each class**, which is `$comment-order`'s newest-and-largest-first, so the
 * shortlist tracks the file's own statement of what leads a line rather than a second opinion
 * about it. 6/4/2 splits 12 roughly in proportion to the catalog's 25/13/5 while keeping at least
 * two of the smallest class.
 */
const PAIR_DEVICES_PER_CLASS: readonly (readonly [CatalogDevice['class'], number])[] = [
  ['discrete-gpu', 6],
  ['unified-soc', 4],
  ['cpu-ram', 2],
];

/**
 * How many models the model x device tier pairs.
 *
 * Ten from {@link modelsByPopularity}, so the shortlist is Hugging Face's answer to "what are
 * people actually running" rather than a hand-kept list that rots between catalog refreshes — 13
 * of 35 rows changed rank across a single week's refresh, so any list written down here would be
 * wrong within the month. Ten against twelve devices is 120 pages, which is where the tier is
 * sized: the count cap is 400 and the other three tiers are 79 of it.
 */
const PAIR_MODELS = 10;

const DEVICE_IDS = new Set(DEVICES.map((device) => device.id));

/** What every page's description ends with: the claim that separates this from a rule of thumb. */
const COMPUTED =
  'Fit verdict, memory footprint, prefill and decode, computed from the architecture rather ' +
  'than approximated.';

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
    indexable: true,
  };
}

/** A device's headline pair, in the units the catalog states them in. */
function deviceFigures(device: CatalogDevice): string {
  return `${Math.round(device.capacityBytes / GIB)} GiB at ${Math.round(device.bandwidthBytesPerSec / GB)} GB/s`;
}

function deviceRoute(device: CatalogDevice): PrerenderRoute {
  return {
    segments: [device.id],
    tier: 1,
    config: { deviceId: device.id },
    title: `${device.name} — what LLM runs on it? · Headroom`,
    description:
      `Which open-weight LLMs run on the ${device.name} — ${deviceFigures(device)} — and how ` +
      `comfortably. ${COMPUTED}`,
    indexable: device.status === 'shipping',
  };
}

function modelRoute(model: CatalogModel): PrerenderRoute {
  return {
    segments: [MODEL_PREFIX, modelSlug(model)],
    tier: 2,
    config: { modelId: model.id },
    title: `${model.name} — what hardware runs it? · Headroom`,
    description:
      `Which machines run ${model.org} ${model.name} — ${params(model.totalParams)} parameters, ` +
      `${tokens(model.maxContext)} context — and how comfortably. ${COMPUTED}`,
    indexable: true,
  };
}

function pairRoute(device: CatalogDevice, model: CatalogModel): PrerenderRoute {
  return {
    segments: [device.id, modelSlug(model)],
    tier: 3,
    config: { deviceId: device.id, modelId: model.id },
    title: `${model.name} on the ${device.name} — does it fit? · Headroom`,
    description:
      `Does ${model.org} ${model.name}, ${params(model.totalParams)} parameters, run on the ` +
      `${device.name} — ${deviceFigures(device)}? ${COMPUTED}`,
    indexable: device.status === 'shipping',
  };
}

/** The leading rows of each class, in the order `devices.json` lists both. */
function pairDevices(): readonly CatalogDevice[] {
  return PAIR_DEVICES_PER_CLASS.flatMap(([deviceClass, count]) =>
    DEVICES.filter((device) => device.class === deviceClass).slice(0, count)
  );
}

/**
 * Every page the build writes as a file.
 *
 * Four tiers, and the shape of the inventory is the whole answer to a combinatorial space that has
 * 468 million points in it: every device (43) and every model (35) get a page, and the product of
 * the two is taken only across a shortlist (120). Ordered by tier so a cap that trips can name the
 * layer to narrow, and within a tier by the catalog's own order.
 *
 * Derived rather than listed: the rows, their names and every figure in the metadata come from the
 * catalog, so a spec correction reaches the prerendered pages on the next build without anybody
 * editing this file.
 */
export function prerenderRoutes(): readonly PrerenderRoute[] {
  const byPopularity = modelsByPopularity();
  return [
    rootRoute(),
    ...DEVICES.map(deviceRoute),
    ...byPopularity.map(modelRoute),
    ...pairDevices().flatMap((device) =>
      byPopularity.slice(0, PAIR_MODELS).map((model) => pairRoute(device, model))
    ),
  ];
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
 *
 * **A two-segment path claims nothing unless both halves resolve.** `/rtx-5090/nonsense/` reads as
 * no route at all rather than as an RTX 5090 page: the address names a page that does not exist,
 * and answering it with a *different* page's figures under the reader's own URL is the failure the
 * device aliases exist to prevent.
 */
export function configFromPath(pathname: string, base: string): Partial<Config> {
  const segments = pathSegments(pathname, base);

  if (segments.length === 1) {
    const deviceId = canonicalDeviceId(segments[0]);
    return DEVICE_IDS.has(deviceId) ? { deviceId } : {};
  }

  if (segments.length === 2) {
    const modelId = modelIdFromSlug(segments[1]);
    if (modelId === undefined) return {};
    if (segments[0] === MODEL_PREFIX) return { modelId };
    const deviceId = canonicalDeviceId(segments[0]);
    return DEVICE_IDS.has(deviceId) ? { deviceId, modelId } : {};
  }

  return {};
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
