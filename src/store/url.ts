import { DEFAULT_CONFIG, type Config } from './scenario';

/**
 * The scenario, in the querystring.
 *
 * A link has to reproduce exactly what the sender was looking at — that is the whole
 * distribution mechanism for a tool like this, and it is also how someone asks "is this right?"
 * without screenshots.
 *
 * Two properties the encoding has to hold:
 *
 *   - **A querystring is complete or absent.** Either the scenario is exactly the default and
 *     the URL is bare, or every one of the nine fields is written out. Nothing in between.
 *
 *     The tempting version writes only what differs, which is shorter and reads nicely. It is
 *     also wrong the first time a default changes: a link shared today as `?d=rtx-5090` would
 *     silently pick up tomorrow's default model, and the scenario the sender was looking at is
 *     gone. These links are meant to survive in forum threads for months, so the querystring
 *     cannot depend on a deployment constant. Nine short fields is roughly 90 characters —
 *     cheap insurance against a link that quietly means something else later.
 *
 *     A bare URL is exempt because it makes no claim: it means "however the app opens", which
 *     stays true whatever the defaults become.
 *   - **Reading is total.** Every value arrives from a URL a stranger may have edited by hand,
 *     so nothing here throws or trusts; the store's `coerce` is the single validation point and
 *     this layer's job is only to hand it strings.
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
 */
export function configToSearch(config: Config): string {
  const isDefault = (Object.keys(KEYS) as (keyof Config)[]).every(
    (key) => config[key] === DEFAULT_CONFIG[key]
  );
  return isDefault ? '' : encodeAll(config);
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
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const config: Config = { ...DEFAULT_CONFIG };

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
 * Whether two configs describe the same scenario.
 *
 * Used to avoid pushing history entries for a URL that already says what it needs to — dragging
 * a slider back to where it started should not leave two entries in the back button.
 */
export function sameScenario(a: Config, b: Config): boolean {
  return (Object.keys(KEYS) as (keyof Config)[]).every((key) => a[key] === b[key]);
}
