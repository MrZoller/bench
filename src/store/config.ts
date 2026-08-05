import { create } from 'zustand';
import {
  estimateScenario,
  evaluate,
  type Evaluation,
  type Scenario,
  type ScenarioEstimate,
} from '@/engine';
import { canShard } from '@/engine/placement';
import { FALLBACK_QUANT_ID, quantApplies } from '@/lib/quantChoice';
import { canonicalDeviceId, getDevice, getModel, MODELS, DEVICES } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime, RUNTIMES } from '@/data/runtimes';
import { locationToConfig, pathBaseline } from './url';
import { DEFAULT_CONFIG, type Config } from './scenario';

export { DEFAULT_CONFIG, type Config };

interface ConfigStore extends Config {
  set: <K extends keyof Config>(key: K, value: Config[K]) => void;
  /** Adopt a whole scenario at once — used when the user navigates back or forward. */
  replace: (config: Partial<Config>) => void;
  reset: () => void;
}

/**
 * Ids are validated against the catalogs before they are accepted, because they can arrive from
 * a URL. An unknown id falls back to the default rather than throwing — a stale shared link
 * should degrade to something usable, not a blank page.
 */
function coerce(config: Config): Config {
  const known = <T>(id: string, lookup: (id: string) => T, fallback: string): string => {
    try {
      lookup(id);
      return id;
    } catch {
      return fallback;
    }
  };

  /**
   * Canonicalised before it is validated, not after.
   *
   * `getDevice` follows an alias, so a stale id from a shared link resolves to the right row either
   * way — but what the store *keeps* is what re-encodes into the URL and what the hardware picker
   * compares its options against. Keeping `rtx-a6000-ada` would load the Ada card, show its
   * figures, and leave the `<select>` matching no option at all.
   */
  const deviceId = known(canonicalDeviceId(config.deviceId), getDevice, DEFAULT_CONFIG.deviceId);
  const device = getDevice(deviceId);

  const modelId = known(config.modelId, getModel, DEFAULT_CONFIG.modelId);
  /**
   * A model cannot be asked for more context than it was trained for. Catalogued models top out
   * anywhere from 16K (Phi-4) to 1M (Mistral Small 4 119B) while the slider offers the same
   * stops for all of them, so without this a 40K Qwen would report fit, memory and speed for a 1M
   * window it cannot process. The spread is the point rather than the endpoints: the clamp is what
   * keeps one slider honest across a range that now covers a factor of 64.
   */
  const runtime = getRuntime(known(config.runtimeId, getRuntime, DEFAULT_CONFIG.runtimeId));
  const resolvedQuant = known(config.quantId, getQuant, DEFAULT_CONFIG.quantId);
  const quantId = quantApplies(getQuant(resolvedQuant), getModel(modelId), device, runtime)
    ? resolvedQuant
    : FALLBACK_QUANT_ID;

  const contextTokens = Math.min(
    getModel(modelId).maxContext,
    clamp(config.contextTokens, 512, 1_048_576, DEFAULT_CONFIG.contextTokens)
  );

  return {
    ...config,
    modelId,
    /**
     * An expert-only scheme on a dense model is a no-op the label denies: MXFP4 spares
     * everything that is not a routed expert, so with none present it computes exactly BF16
     * while still calling itself 4-bit. The picker hides it there, so the *selection* has to
     * move too — otherwise switching from gpt-oss to Qwen leaves the store holding an option
     * the control no longer offers.
     *
     * Applicability is checked *after* resolving the id, not before. Checked first, an unknown
     * id fails its lookup, reports "not expert-only", and is then replaced by the default —
     * which is `mxfp4`, so a dense model with a junk quant in its URL landed on exactly the
     * option this rule exists to prevent.
     */
    quantId,
    runtimeId: known(config.runtimeId, getRuntime, DEFAULT_CONFIG.runtimeId),
    deviceId,
    // The same predicate the picker uses. Holding a second copy here meant the Spark's slider
    // was offered and then silently reset to 1x on every change, so the linked-Spark
    // configuration could never actually be evaluated.
    deviceCount: canShard(device) ? clamp(config.deviceCount, 1, 8, DEFAULT_CONFIG.deviceCount) : 1,
    contextTokens,
    concurrency: clamp(config.concurrency, 1, 128, DEFAULT_CONFIG.concurrency),
    /**
     * The prompt is part of the context, not a separate budget — `UsageSpec.contextTokens` is
     * prompt plus generation. Left independent, a 2K context with a 128K prompt sized KV for
     * 2048 tokens while reporting a seven-minute time-to-first-token: a comfortable-looking
     * budget bar for a request needing 64x the cache it drew.
     */
    promptTokens: Math.min(
      contextTokens,
      clamp(config.promptTokens, 1, 1_048_576, DEFAULT_CONFIG.promptTokens)
    ),
    // Held inside what the runtime can actually store, not merely inside the type. A 4-bit
    // cache on vLLM would charge 0.5 bytes per element for something it cannot allocate.
    kvPrecision: runtime.kvPrecisions.includes(config.kvPrecision)
      ? config.kvPrecision
      : (runtime.kvPrecisions[0] ?? DEFAULT_CONFIG.kvPrecision),
  };
}

/**
 * Clamp to a range, falling back for values that are not numbers at all.
 *
 * NaN takes the fallback rather than the minimum: these arrive from sliders and from a
 * hand-editable querystring, where `Number(params.get('ctx'))` on nonsense yields NaN, and
 * silently pinning that to the smallest legal context is its own wrong answer.
 */
function clamp(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}

export const useConfig = create<ConfigStore>((set) => ({
  // Coerced on the way in: the initial state can come from a hand-edited link.
  ...coerce(readInitialConfig()),
  set: (key, value) => set((state) => coerce({ ...state, [key]: value })),
  replace: (config) => set(coerce({ ...DEFAULT_CONFIG, ...config })),
  reset: () => set(DEFAULT_CONFIG),
}));

/**
 * The scenario the page was opened with — the whole address, not just its query.
 *
 * Guarded for a non-browser environment because the store is imported by tests and by the
 * prerender step, neither of which has a `location`. **The guard is also the reason prerendering
 * needed a seam** (#178): this store is a module-level singleton whose initial state is evaluated
 * at import time, so under `renderToString` it is always the default and a naive prerender writes
 * every device page with the default device's figures. `scripts/prerender.ts` calls `replace`
 * before each route instead, which runs the same `coerce` this does.
 *
 * The client half is the half that has to match. A visitor landing on `/rtx-5090/` has no query
 * at all, so reading only `location.search` here would hydrate the default scenario over markup
 * rendered for an RTX 5090 — a whole-tree mismatch, and React would discard the prerendered page.
 * Reading the path is what closes that. It cannot move to an effect: the singleton is built at
 * import time, and a first paint of the wrong scenario is the same mismatch one tick later.
 */
function readInitialConfig(): Config {
  if (typeof window === 'undefined') return DEFAULT_CONFIG;
  return locationToConfig(
    window.location.pathname,
    window.location.search,
    import.meta.env.BASE_URL
  );
}

/**
 * The scenario a prerendered path was actually rendered with — coerced, as the server coerced it.
 *
 * `pathBaseline` answers "what does this path declare", which is a raw partial: `/m/qwen--qwen3-8b/`
 * declares a model and nothing else, so the rest is `DEFAULT_CONFIG`. That is the right answer to
 * its own question and the wrong baseline to compare a *store* against, because the store has been
 * through `coerce`. `DEFAULT_CONFIG.quantId` is `mxfp4`, which `quantApplies` refuses on a model
 * with no experts, so every dense model coerces to `bf16` and the store no longer equals the
 * baseline. `useUrlSync` reads that as a scenario that has diverged from its path and writes the
 * whole nine-field query in — erasing the pretty URL the prerenderer just advertised as canonical.
 *
 * It was 101 of the 199 routes: 17 of 35 model pages (every dense row) and 84 of 120 pair pages
 * (7 dense models across 12 devices). Tiers 0 and 1 were clean, which is why #178's Phase 2 fix
 * looked complete — `coerce(DEFAULT_CONFIG ∪ {deviceId})` is a fixed point for all 43 devices, so
 * the failure could not appear until model-bearing routes existed. `microsoft/phi-4` drifts on two
 * fields, its 16K `maxContext` clamping the default 32K as well.
 *
 * Coercing here rather than teaching `src/data/routes.ts` the coerced fields is what keeps the two
 * halves in agreement *by construction*: the prerenderer renders `coerce(DEFAULT ∪ route.config)`
 * and `routes.test.ts` already proves `configFromPath(routePath(route)) === route.config`, so this
 * is the same expression the server evaluated rather than a second implementation of it. The
 * alternative would put `quantApplies`, `getQuant` and the context clamp inside a module the issue
 * requires to be a pure catalog derivation, and it would drift the day `coerce` changes.
 */
export function coercedPathBaseline(pathname: string, base: string): Config {
  return coerce(pathBaseline(pathname, base));
}

/**
 * The config as the engine wants it: catalogs resolved, usage gathered.
 *
 * `cachedPrefixTokens` is not part of `Config` and is passed separately on purpose: it is a
 * property of the *archetype being graded*, not of the scenario the user configured. The Bench's
 * own three tiles describe a standalone prompt — that is what the sliders mean — and only the
 * verdict layer has an archetype to ask.
 */
function scenarioFor(config: Config, cachedPrefixTokens?: number): Scenario {
  return {
    model: getModel(config.modelId),
    quant: getQuant(config.quantId),
    runtime: getRuntime(config.runtimeId),
    rig: { device: getDevice(config.deviceId), count: config.deviceCount },
    usage: {
      contextTokens: config.contextTokens,
      concurrency: config.concurrency,
      promptTokens: config.promptTokens,
      cachedPrefixTokens,
      kvPrecision: config.kvPrecision,
    },
  };
}

/**
 * Resolve the config into engine inputs and evaluate.
 *
 * Cheap enough to call freely: the engine is pure arithmetic over a handful of numbers, which is
 * what lets a slider feel like direct manipulation instead of a form. The Bench memoises it on the
 * config all the same, since every panel below reads the one result.
 */
export function evaluateConfig(config: Config): Evaluation {
  return evaluate(scenarioFor(config));
}

/**
 * Placement, decode and prefill for a config, without the two context-limit searches.
 *
 * "Cheap enough to call freely" holds for one evaluation per render and stops holding when the
 * verdict layer asks for eight or nine of them, because each `evaluate` runs `maxContextThatFits`
 * twice and each of those is a binary search over the model's whole context range. The archetype
 * grades need none of that; they need the three answers here.
 */
export function estimateConfig(config: Config, cachedPrefixTokens?: number): ScenarioEstimate {
  return estimateScenario(scenarioFor(config, cachedPrefixTokens));
}

/** Re-exported so components import catalogs from one place. */
export { MODELS, DEVICES, RUNTIMES };
