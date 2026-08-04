import { describe, expect, it } from 'vitest';
import {
  configToSearch,
  configToShareSearch,
  locationToConfig,
  pathBaseline,
  sameScenario,
  searchToConfig,
  shouldHydrate,
} from './url';
import { DEFAULT_CONFIG, type Config } from './config';

/**
 * A link is the distribution mechanism for a tool like this, so the encoding has to survive a
 * stranger editing it by hand. Every test here is either "a round trip is lossless" or "nonsense
 * degrades to something usable".
 */
describe('scenario URLs', () => {
  it('writes nothing for a default scenario', () => {
    expect(configToSearch(DEFAULT_CONFIG)).toBe('');
  });

  it('still writes the default scenario in full when someone asks for a link to it', () => {
    // The bare address bar is honest — it claims nothing. A copied link claims something, so the
    // default scenario is the one case where the two encoders differ, and the share side has to
    // spell it out or the highest-volume share of all drifts with the next defaults change.
    const shared = configToShareSearch(DEFAULT_CONFIG);
    const params = new URLSearchParams(shared.slice(1));

    expect([...params.keys()].sort()).toEqual(['ctx', 'd', 'kv', 'm', 'n', 'p', 'q', 'r', 'u']);
    expect(searchToConfig(shared)).toEqual(DEFAULT_CONFIG);
  });

  it('writes every field once anything differs, not just the field that differs', () => {
    // The sparse version wrote `?d=rtx-5090` and let the other eight fall back to whatever
    // DEFAULT_CONFIG says on the day the link is *opened*. A link pasted into a forum thread
    // would then change meaning the next time a default moved. Completeness is what makes the
    // querystring independent of the deployment that reads it.
    const search = configToSearch({ ...DEFAULT_CONFIG, deviceId: 'rtx-5090' });
    const params = new URLSearchParams(search.slice(1));

    expect([...params.keys()].sort()).toEqual(['ctx', 'd', 'kv', 'm', 'n', 'p', 'q', 'r', 'u']);
    expect(params.get('d')).toBe('rtx-5090');
    expect(params.get('m')).toBe(DEFAULT_CONFIG.modelId);
  });

  it('reproduces a shared scenario even when every default has since changed', () => {
    const shared: Config = {
      modelId: 'Qwen/Qwen3-32B',
      quantId: 'q5_k_m',
      runtimeId: 'vllm',
      deviceId: 'rtx-5090',
      deviceCount: 2,
      contextTokens: 32768,
      concurrency: 4,
      promptTokens: 4096,
      kvPrecision: 'q8',
    };
    const link = configToSearch(shared);

    // Stand in for a future deployment: read the link back against defaults that share not one
    // value with it. Nothing may leak through from the reader's side.
    const laterDefaults: Config = {
      modelId: 'meta-llama/Llama-3.1-8B-Instruct',
      quantId: 'q4_k_m',
      runtimeId: 'llama.cpp',
      deviceId: 'm3-ultra-512',
      deviceCount: 1,
      contextTokens: 8192,
      concurrency: 1,
      promptTokens: 1024,
      kvPrecision: 'fp16',
    };
    const params = new URLSearchParams(link.slice(1));
    const reread: Config = { ...laterDefaults };
    const fields = {
      m: 'modelId',
      q: 'quantId',
      r: 'runtimeId',
      d: 'deviceId',
      n: 'deviceCount',
      ctx: 'contextTokens',
      u: 'concurrency',
      p: 'promptTokens',
      kv: 'kvPrecision',
    } as const;

    for (const [short, full] of Object.entries(fields)) {
      const raw = params.get(short);
      if (raw === null) continue;
      if (typeof laterDefaults[full] === 'number') {
        (reread[full] as number) = Number(raw);
      } else {
        (reread[full] as string) = raw;
      }
    }

    expect(reread).toEqual(shared);
  });

  it('round-trips a fully specified scenario', () => {
    const scenario: Config = {
      modelId: 'deepseek-ai/DeepSeek-V3',
      quantId: 'q4_k_m',
      runtimeId: 'vllm',
      deviceId: 'h100-sxm',
      deviceCount: 4,
      contextTokens: 65536,
      concurrency: 8,
      promptTokens: 16384,
      kvPrecision: 'q8',
    };
    expect(searchToConfig(configToSearch(scenario))).toEqual(scenario);
  });

  it('fills absent keys from the default rather than leaving holes', () => {
    const config = searchToConfig('?d=rtx-5090');
    expect(config.deviceId).toBe('rtx-5090');
    expect(config.modelId).toBe(DEFAULT_CONFIG.modelId);
    expect(config.contextTokens).toBe(DEFAULT_CONFIG.contextTokens);
  });

  /**
   * The values themselves are validated by the store's `coerce`, which runs over whatever this
   * returns. What matters here is that an unparseable *number* reads as "unset" rather than
   * being handed on as NaN — `coerce` would clamp that, and a hand-typed `ctx=lots` should not
   * silently become the smallest legal context.
   */
  it('treats an unparseable number as unset', () => {
    expect(searchToConfig('?ctx=lots').contextTokens).toBe(DEFAULT_CONFIG.contextTokens);
    expect(searchToConfig('?n=').deviceCount).toBe(DEFAULT_CONFIG.deviceCount);
  });

  it('never throws on hostile input', () => {
    for (const search of ['', '?', '?????', '?m=', '?m=%%%', '?unknown=1', '?ctx=-1e999']) {
      expect(() => searchToConfig(search)).not.toThrow();
    }
  });

  it('compares scenarios by value, for deciding whether the URL needs rewriting', () => {
    expect(sameScenario(DEFAULT_CONFIG, { ...DEFAULT_CONFIG })).toBe(true);
    expect(sameScenario(DEFAULT_CONFIG, { ...DEFAULT_CONFIG, concurrency: 4 })).toBe(false);
  });
});

/**
 * A path route carries one of the nine fields and a querystring carries all nine, so the two can
 * both speak about the same page. These are the rules for when they do.
 */
describe('a path and a query together', () => {
  it('reads the scenario a path names', () => {
    expect(locationToConfig('/rtx-5090/', '', '/')).toEqual({
      ...DEFAULT_CONFIG,
      deviceId: 'rtx-5090',
    });
  });

  it('lets the query win, because the query is the complete encoding', () => {
    // The path is an entry point that names one field; the query names an exact scenario, and it
    // is the form every link already handed out takes. Precedence the other way would change
    // what those links mean.
    expect(locationToConfig('/rtx-5090/', '?d=dgx-spark', '/').deviceId).toBe('dgx-spark');
  });

  it('lets a query override one field without discarding the path for the rest', () => {
    const config = locationToConfig('/rtx-5090/', '?u=4', '/');
    expect(config.deviceId).toBe('rtx-5090');
    expect(config.concurrency).toBe(4);
  });

  it('falls back to the defaults for a path it does not recognise', () => {
    // 404.html answers at any unmatched path and has to boot as something usable.
    expect(locationToConfig('/nope/whatever/', '', '/')).toEqual(DEFAULT_CONFIG);
  });

  it('reads a path under a base path', () => {
    expect(locationToConfig('/headroom/rtx-5090/', '', '/headroom/').deviceId).toBe('rtx-5090');
  });
});

/**
 * The property #178 turned on: a prerendered page must not rewrite its own pretty URL into a
 * nine-field query on the first effect tick.
 */
describe('the baseline a bare query is measured against', () => {
  it('is the route the address names', () => {
    expect(pathBaseline('/rtx-5090/', '/')).toEqual({ ...DEFAULT_CONFIG, deviceId: 'rtx-5090' });
    expect(pathBaseline('/', '/')).toEqual(DEFAULT_CONFIG);
  });

  it('leaves a prerendered page bare while it still shows that page scenario', () => {
    const onRoute = { ...DEFAULT_CONFIG, deviceId: 'rtx-5090' };
    expect(configToSearch(onRoute, pathBaseline('/rtx-5090/', '/'))).toBe('');
  });

  it('writes the whole scenario the moment it diverges from the path', () => {
    // And the caller keeps the path, so the address becomes `/rtx-5090/?…&d=dgx-spark&…`: one
    // source of truth, no history churn, and the query decides.
    const diverged = { ...DEFAULT_CONFIG, deviceId: 'dgx-spark' };
    const search = configToSearch(diverged, pathBaseline('/rtx-5090/', '/'));

    expect(new URLSearchParams(search.slice(1)).get('d')).toBe('dgx-spark');
    expect([...new URLSearchParams(search.slice(1)).keys()]).toHaveLength(9);
  });

  it('still measures against the defaults when nothing passes a baseline', () => {
    expect(configToSearch(DEFAULT_CONFIG)).toBe('');
    expect(configToSearch({ ...DEFAULT_CONFIG, deviceId: 'rtx-5090' })).not.toBe('');
  });
});

/**
 * Whether the markup on the page is the markup this address asked for.
 *
 * `main.tsx` shipped branching on the `data-prerendered` attribute alone, which says only that
 * markup exists. Every link the share button hands out is the root path plus a complete
 * nine-field query, so the common arrival is a query naming one device on a file rendered for
 * another — marker present, scenario wrong, whole tree discarded. These are the cases that come
 * apart, tested here rather than through a DOM entry point because the decision is arithmetic
 * over a URL and nothing about it needs a document.
 */
describe('deciding whether to hydrate', () => {
  const shared = configToShareSearch;
  const onRtx: Config = { ...DEFAULT_CONFIG, deviceId: 'rtx-5090' };

  it('hydrates the root arrived at bare', () => {
    expect(shouldHydrate('/', '', '/', true)).toBe(true);
  });

  it('hydrates the root when the query spells out the scenario it already renders', () => {
    // The highest-volume share of all: someone copies a link without touching a control, so the
    // query is nine fields of `DEFAULT_CONFIG` on the page that was rendered from it.
    expect(shouldHydrate('/', shared(DEFAULT_CONFIG), '/', true)).toBe(true);
  });

  it('does not hydrate the root when the query names another device', () => {
    // The finding. `/` is a DGX Spark page; this address is an RTX 5090, and hydrating would
    // paint the Spark's figures and then swap them.
    expect(shouldHydrate('/', shared(onRtx), '/', true)).toBe(false);
  });

  it('hydrates a device route arrived at bare', () => {
    expect(shouldHydrate('/rtx-5090/', '', '/', true)).toBe(true);
  });

  it('does not hydrate a device route whose query overrides the path', () => {
    expect(shouldHydrate('/rtx-5090/', '?d=dgx-spark', '/', true)).toBe(false);
  });

  it("hydrates a device route whose query is that route's own scenario", () => {
    expect(shouldHydrate('/rtx-5090/', shared(onRtx), '/', true)).toBe(true);
  });

  it('does not hydrate a device route over a query differing in one untouched field', () => {
    // Not only the device: any of the nine puts the markup out of date, and `ctx` changes every
    // figure on the page without changing the machine it names.
    expect(shouldHydrate('/rtx-5090/', shared({ ...onRtx, contextTokens: 8192 }), '/', true)).toBe(
      false
    );
  });

  /**
   * The outer guard, and it is not redundant with the comparison.
   *
   * `404.html` is the bare shell served at every unmatched path, where the address names no route
   * — so both sides of the comparison are `DEFAULT_CONFIG` and it says "same scenario" about a
   * container with nothing in it. Only the marker can tell those apart, which is why it stays.
   */
  it('never hydrates without the marker, whatever the address says', () => {
    for (const [pathname, search] of [
      ['/', ''],
      ['/', shared(DEFAULT_CONFIG)],
      ['/', shared(onRtx)],
      ['/rtx-5090/', ''],
      ['/rtx-5090/', '?d=dgx-spark'],
      ['/rtx-5090/', shared(onRtx)],
      ['/nope/whatever/', ''],
    ]) {
      expect(shouldHydrate(pathname, search, '/', false)).toBe(false);
    }
  });

  it('reads the address below a base path, like everything else here', () => {
    expect(shouldHydrate('/headroom/rtx-5090/', '', '/headroom/', true)).toBe(true);
    expect(shouldHydrate('/headroom/rtx-5090/', '?d=dgx-spark', '/headroom/', true)).toBe(false);
  });
});
