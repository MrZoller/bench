import { configFromPath } from '@/data/routes';
import { DEFAULT_CONFIG, type Config } from './scenario';

/**
 * The scenario, in the URL.
 *
 * A link has to reproduce exactly what the sender was looking at — that is the whole
 * distribution mechanism for a tool like this, and it is also how someone asks "is this right?"
 * without screenshots.
 *
 * Two properties the encoding has to hold:
 *
 *   - **A querystring is complete or absent.** Either the scenario is exactly what the rest of
 *     the URL already says and the query is bare, or every one of the nine fields is written out.
 *     Nothing in between.
 *
 *     The tempting version writes only what differs, which is shorter and reads nicely. It is
 *     also wrong the first time a default changes: a link shared today as `?d=rtx-5090` would
 *     silently pick up tomorrow's default model, and the scenario the sender was looking at is
 *     gone. These links are meant to survive in forum threads for months, so the querystring
 *     cannot depend on a deployment constant. Nine short fields is roughly 90 characters —
 *     cheap insurance against a link that quietly means something else later.
 *
 *     A bare query is exempt because it makes no claim beyond the address it sits on: it means
 *     "however the app opens *here*", which stays true whatever the defaults become.
 *
 *     **That last sentence used to say "however the app opens", full stop, and #178 is why it now
 *     says where.** The rule was stated against `DEFAULT_CONFIG` because until path routes existed
 *     the address carried nothing else. It does now: a prerendered `/rtx-5090/` means the default
 *     scenario *on an RTX 5090*, and the original argument transfers to it whole — a path is not
 *     a deployment constant the way a default is, because it is written into the address the
 *     reader is already looking at and cannot come to mean something else while their link sits
 *     in a thread. Leaving the comparison at `DEFAULT_CONFIG` was the version that broke: on every
 *     prerendered page the device differs from the default, so {@link configToSearch} returned all
 *     nine fields and `useUrlSync` rewrote `/rtx-5090/` to `/rtx-5090/?m=…&d=rtx-5090&…` within
 *     400ms of hydrating — a pretty URL erasing itself while the reader watched. So the baseline
 *     became an argument rather than a constant, and that is the only thing that moved. A query
 *     that is present is still complete; nothing is ever written as a diff.
 *   - **Reading is total.** Every value arrives from a URL a stranger may have edited by hand,
 *     so nothing here throws or trusts; the store's `coerce` is the single validation point and
 *     this layer's job is only to hand it strings.
 *
 * **Path and query, when both say something: the query wins.** `/rtx-5090/?d=dgx-spark` renders
 * the Spark. The path carries one of the nine fields and is an entry point; the query carries all
 * nine and names an exact scenario, and every link already handed out is a query. Precedence in
 * the other direction would quietly change what those links mean, which is the one thing this
 * whole module exists to prevent. {@link locationToConfig} is where it is applied, once.
 */

/** Short keys, because these end up in a link someone pastes into a message. */
const KEYS = {
  modelId: 'm',
  quantId: 'q',
  runtimeId: 'r',
  deviceId: 'd',
  deviceCount: 'n',
  contextTokens: 'ctx',
  concurrency: 'u',
  promptTokens: 'p',
  kvPrecision: 'kv',
} as const satisfies Record<keyof Config, string>;

const NUMERIC: readonly (keyof Config)[] = [
  'deviceCount',
  'contextTokens',
  'concurrency',
  'promptTokens',
];

/**
 * The whole scenario, every field written out. The one encoder; the two exports below differ
 * only in whether they are allowed to return nothing instead.
 */
function encodeAll(config: Config): string {
  const params = new URLSearchParams();
  for (const key of Object.keys(KEYS) as (keyof Config)[]) {
    params.set(KEYS[key], String(config[key]));
  }
  return `?${params.toString()}`;
}

/**
 * The address bar: all of the scenario, or none of it.
 *
 * Nothing is omitted for matching the default, because the reader has no way to tell an omitted
 * field from one the sender never touched — and the default it would fall back to is whatever
 * ships on the day the link is opened, not the day it was written.
 *
 * `baseline` is what the address already says without a query — `DEFAULT_CONFIG` on the root, and
 * the route's own scenario on a prerendered path (#178). It decides only whether the query may be
 * empty; it never selects which fields get written, because a query that is present is complete.
 * Defaulted rather than required so every caller that has no path to speak of keeps the behaviour
 * it had.
 */
export function configToSearch(config: Config, baseline: Config = DEFAULT_CONFIG): string {
  const saysNothingNew = (Object.keys(KEYS) as (keyof Config)[]).every(
    (key) => config[key] === baseline[key]
  );
  return saysNothingNew ? '' : encodeAll(config);
}

/**
 * A link someone asked for, which is always a claim about a specific scenario — so it is always
 * written in full, including the untouched default one.
 *
 * The address bar can stay bare on a fresh page because it asserts nothing: it means "however
 * the app opens", which stays true whatever the defaults become. The moment a share button hands
 * that URL to someone, it stops being true — it now names the configuration the sender was
 * looking at, and that has to survive the next time a default moves.
 */
export function configToShareSearch(config: Config): string {
  return encodeAll(config);
}

/**
 * Read a scenario out of a querystring.
 *
 * Returns a full `Config` by filling every absent or unparseable field from the default, so the
 * caller always receives something usable. Validation of the *values* — unknown model ids,
 * out-of-range contexts, a prompt longer than its context — belongs to the store's `coerce`,
 * which runs over the result. Doing it in one place is what stops the two disagreeing.
 */
export function searchToConfig(search: string): Config {
  return { ...DEFAULT_CONFIG, ...searchToPartialConfig(search) };
}

/**
 * The fields the querystring actually named, and no others.
 *
 * The same read as {@link searchToConfig} without the defaults poured in, which is what a caller
 * needs when something else in the URL has already spoken. Spreading a full `Config` over a path's
 * scenario would overwrite the path's device with the default device and silently invert the
 * precedence rule the module docblock states; a partial spreads as an override, which is the
 * behaviour that rule describes.
 */
export function searchToPartialConfig(search: string): Partial<Config> {
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const config: Partial<Config> = {};

  for (const key of Object.keys(KEYS) as (keyof Config)[]) {
    const raw = params.get(KEYS[key]);
    if (raw === null) continue;

    if (NUMERIC.includes(key)) {
      // `Number('')` is 0, which is finite — so an empty value would read as a real zero and
      // then be clamped to the minimum, rather than meaning "not specified".
      if (raw.trim() === '') continue;
      const parsed = Number(raw);
      // NaN is left to the default rather than passed on: `coerce` would clamp it, but a
      // hand-edited `ctx=lots` should read as "unset", not as the smallest legal context.
      if (Number.isFinite(parsed)) {
        (config[key] as number) = parsed;
      }
    } else {
      (config[key] as string) = raw;
    }
  }

  return config;
}

/**
 * The scenario a whole address names — path and query together, query winning.
 *
 * One function rather than a rule each caller applies, because there are two callers that must
 * never disagree: the store reads it once at import time to decide what the page opens as, and
 * `useUrlSync` reads it again on `popstate`, when someone edits the URL or follows a second link
 * into the same page. Two spreads in two files is how a back button comes to show a different
 * scenario than a reload of the same address.
 *
 * `base` is the site's base path — `import.meta.env.BASE_URL` in the browser — because a route is
 * only a route below it, and nothing here may assume what it is.
 */
export function locationToConfig(pathname: string, search: string, base: string): Config {
  return {
    ...DEFAULT_CONFIG,
    ...configFromPath(pathname, base),
    ...searchToPartialConfig(search),
  };
}

/**
 * The scenario an address names *without* its query: the baseline a bare query is measured
 * against, and what {@link configToSearch} is given so a prerendered page's pretty URL survives
 * its own first effect tick.
 */
export function pathBaseline(pathname: string, base: string): Config {
  return { ...DEFAULT_CONFIG, ...configFromPath(pathname, base) };
}

/**
 * Whether two configs describe the same scenario.
 *
 * Used to avoid pushing history entries for a URL that already says what it needs to — dragging
 * a slider back to where it started should not leave two entries in the back button.
 */
export function sameScenario(a: Config, b: Config): boolean {
  return (Object.keys(KEYS) as (keyof Config)[]).every((key) => a[key] === b[key]);
}
