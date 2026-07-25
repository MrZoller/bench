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
 *   - **Only what differs from the default is written.** A fresh page has a clean URL, and a
 *     shared one shows at a glance what was changed. It also keeps links short enough to paste
 *     into a chat message without wrapping.
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

/** Serialise the parts that differ from the default. */
export function configToSearch(config: Config): string {
  const params = new URLSearchParams();

  for (const key of Object.keys(KEYS) as (keyof Config)[]) {
    const value = config[key];
    if (value === DEFAULT_CONFIG[key]) continue;
    params.set(KEYS[key], String(value));
  }

  const search = params.toString();
  return search ? `?${search}` : '';
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
