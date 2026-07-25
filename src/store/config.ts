import { create } from 'zustand';
import type { KvPrecision } from '@/engine/types';
import { evaluate, type Evaluation } from '@/engine';
import { canShard } from '@/engine/placement';
import { getDevice, getModel, MODELS, DEVICES } from '@/data/catalog';
import { getQuant } from '@/data/quants';
import { getRuntime, RUNTIMES } from '@/data/runtimes';

/**
 * The whole scenario, in one object.
 *
 * Deliberately flat and made of ids rather than objects: this is what serialises into the
 * querystring, so a link reproduces an exact scenario. Anything that cannot be reconstructed
 * from these seven fields does not belong in the store.
 */
export interface Config {
  modelId: string;
  quantId: string;
  runtimeId: string;
  deviceId: string;
  deviceCount: number;
  contextTokens: number;
  concurrency: number;
  promptTokens: number;
  kvPrecision: KvPrecision;
}

/**
 * Openers chosen to land on the comparison the tool exists to make: a 120B MoE that fits
 * comfortably in a Spark's unified memory and would need offload on a consumer card. Starting
 * on a model that trivially fits would hide the point.
 */
export const DEFAULT_CONFIG: Config = {
  modelId: 'openai/gpt-oss-120b',
  quantId: 'mxfp4',
  runtimeId: 'llama.cpp',
  deviceId: 'dgx-spark',
  deviceCount: 1,
  contextTokens: 32768,
  concurrency: 1,
  promptTokens: 8192,
  kvPrecision: 'fp16',
};

interface ConfigStore extends Config {
  set: <K extends keyof Config>(key: K, value: Config[K]) => void;
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

  const deviceId = known(config.deviceId, getDevice, DEFAULT_CONFIG.deviceId);
  const device = getDevice(deviceId);

  const modelId = known(config.modelId, getModel, DEFAULT_CONFIG.modelId);
  /**
   * A model cannot be asked for more context than it was trained for. Every catalogued model
   * tops out between 32K and 164K while the slider offers stops up to 1M, so without this a
   * 40K Qwen would report fit, memory and speed for a 1M window it cannot process.
   */
  const resolvedQuant = known(config.quantId, getQuant, DEFAULT_CONFIG.quantId);
  const quantId = expertOnlyOnDense(resolvedQuant, modelId) ? 'bf16' : resolvedQuant;

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
    kvPrecision: (['fp16', 'q8', 'q4'] as const).includes(config.kvPrecision)
      ? config.kvPrecision
      : DEFAULT_CONFIG.kvPrecision,
  };
}

/**
 * Clamp to a range, falling back for values that are not numbers at all.
 *
 * NaN takes the fallback rather than the minimum: these arrive from sliders and from a
 * hand-editable querystring, where `Number(params.get('ctx'))` on nonsense yields NaN, and
 * silently pinning that to the smallest legal context is its own wrong answer.
 */
/** True when the quant spares non-expert tensors and the model has no experts to spare. */
function expertOnlyOnDense(quantId: string, modelId: string): boolean {
  try {
    return getQuant(quantId).denseBpw !== undefined && getModel(modelId).expertParams === 0;
  } catch {
    return false;
  }
}

function clamp(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}

export const useConfig = create<ConfigStore>((set) => ({
  ...DEFAULT_CONFIG,
  set: (key, value) => set((state) => coerce({ ...state, [key]: value })),
  reset: () => set(DEFAULT_CONFIG),
}));

/**
 * Resolve the config into engine inputs and evaluate.
 *
 * Called on every render rather than memoised: the engine is pure arithmetic over a handful of
 * numbers, so recomputing costs less than the bookkeeping to avoid it, and this is what lets a
 * slider feel like direct manipulation instead of a form.
 */
export function evaluateConfig(config: Config): Evaluation {
  return evaluate({
    model: getModel(config.modelId),
    quant: getQuant(config.quantId),
    runtime: getRuntime(config.runtimeId),
    rig: { device: getDevice(config.deviceId), count: config.deviceCount },
    usage: {
      contextTokens: config.contextTokens,
      concurrency: config.concurrency,
      promptTokens: config.promptTokens,
      kvPrecision: config.kvPrecision,
    },
  });
}

/** Re-exported so components import catalogs from one place. */
export { MODELS, DEVICES, RUNTIMES };
