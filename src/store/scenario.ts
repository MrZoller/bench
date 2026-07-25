import type { KvPrecision } from '@/engine/types';

/**
 * The scenario itself: what is being run, on what, under what usage.
 *
 * Lives apart from the store and the URL codec because both need it and neither should depend
 * on the other — importing the store from the codec and the codec from the store is a cycle,
 * and the shape is the only thing they genuinely share.
 *
 * Deliberately flat and made of ids rather than objects: this is what serialises into the
 * querystring, so a link reproduces an exact scenario. Anything that cannot be reconstructed
 * from these nine fields does not belong here.
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
