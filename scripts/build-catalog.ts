/**
 * Builds src/data/models.generated.json from Hugging Face.
 *
 * The model landscape moves faster than any training cutoff, so this catalog is *derived*,
 * never typed from memory. Two sources per model, both authoritative:
 *
 *   - `/api/models/{id}?expand[]=safetensors` — exact parameter counts by dtype, summed from
 *     the repo's own safetensors index. Not a rounded marketing figure.
 *   - `/{id}/raw/<sha>/config.json` — the architecture itself: layers, KV heads, head dim,
 *     expert counts, attention window pattern, native quantization. Every fetch after the first
 *     is pinned to the commit that one resolved, so a row cannot straddle a publisher push.
 *
 * Everything the engine needs is computed from those. Where a field can't be determined the
 * script throws rather than guessing: a wrong KV formula silently costs someone a GPU, and a
 * loud failure during a weekly refresh is much cheaper than a plausible wrong number shipped
 * to a page people trust.
 *
 * Usage:
 *   npm run catalog                    # write the catalog; any seed failure blocks the write
 *   npm run catalog -- --dry-run       # fetch and report, write nothing
 *   npm run catalog -- --allow-partial # write even though some seeds failed
 *
 * Set HF_TOKEN to include gated repos (meta-llama in particular returns 401 without one).
 */

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, '../src/data/models.generated.json');

// ---------------------------------------------------------------------------
// The seed list
// ---------------------------------------------------------------------------

interface Seed {
  /** Hugging Face repo id. */
  id: string;
  /** Display name, since repo ids are inconsistent about capitalisation and suffixes. */
  name: string;
  org: string;
  /**
   * Documented corrections applied after derivation. Each needs a reason — this is the one
   * place the script accepts a hand-entered number, so it must never become a dumping ground
   * for "the derived value looked wrong".
   */
  overrides?: {
    totalParams?: number;
    reason: string;
  };
  /**
   * Repo to read downloads and likes from, when the seed is a mirror.
   *
   * Weights come from the mirror because the original is gated, but its traffic does not:
   * NousResearch's Llama 3.1 70B has 4.8K downloads against Meta's 1.24M, which sorted the
   * best-known model in the catalog to last place. Gating applies to `/raw/` and `/resolve/`,
   * not to API metadata, so the canonical figures are readable without a token.
   */
  popularityId?: string;
}

/**
 * Curated rather than "top N by downloads": the download charts are dominated by tiny models,
 * embedding models and one-off GGUF re-uploads. This list is the set of models people
 * actually weigh hardware against, hand-reviewed once and refreshed as the field moves.
 */
const SEEDS: Seed[] = [
  // --- Dense, small enough to run anywhere ---
  { id: 'Qwen/Qwen3-4B', name: 'Qwen3 4B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-8B', name: 'Qwen3 8B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-14B', name: 'Qwen3 14B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-32B', name: 'Qwen3 32B', org: 'Alibaba' },
  // Gemma is gated on google/*, so these point at open mirrors of the same weights.
  {
    id: 'unsloth/gemma-3-12b-it',
    name: 'Gemma 3 12B',
    org: 'Google',
    popularityId: 'google/gemma-3-12b-it',
  },
  {
    id: 'unsloth/gemma-3-27b-it',
    name: 'Gemma 3 27B',
    org: 'Google',
    popularityId: 'google/gemma-3-27b-it',
  },
  { id: 'mistralai/Mistral-Small-24B-Instruct-2501', name: 'Mistral Small 24B', org: 'Mistral' },

  // --- Llama: gated on meta-llama, so mirrors keep the catalog buildable without a token ---
  {
    id: 'NousResearch/Meta-Llama-3.1-8B-Instruct',
    name: 'Llama 3.1 8B Instruct',
    org: 'Meta',
    popularityId: 'meta-llama/Llama-3.1-8B-Instruct',
  },
  {
    id: 'NousResearch/Meta-Llama-3.1-70B-Instruct',
    name: 'Llama 3.1 70B Instruct',
    org: 'Meta',
    popularityId: 'meta-llama/Llama-3.1-70B-Instruct',
  },

  // --- MoE: the interesting cases for unified-memory hardware ---
  { id: 'openai/gpt-oss-20b', name: 'gpt-oss 20B', org: 'OpenAI' },
  { id: 'openai/gpt-oss-120b', name: 'gpt-oss 120B', org: 'OpenAI' },
  { id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3 30B-A3B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-235B-A22B', name: 'Qwen3 235B-A22B', org: 'Alibaba' },
  { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', name: 'Mixtral 8x7B', org: 'Mistral' },

  // --- MLA: the family the naive KV formula gets most wrong ---
  {
    id: 'deepseek-ai/DeepSeek-V3',
    name: 'DeepSeek V3',
    org: 'DeepSeek',
    overrides: {
      totalParams: 671e9,
      reason:
        "HF's safetensors index reports 684.5B, which includes the Multi-Token Prediction " +
        'module. MTP ships in the repo but is not loaded for ordinary inference, so counting ' +
        'it would overstate weights by ~13B. 671B is the published figure.',
    },
  },
  {
    id: 'deepseek-ai/DeepSeek-R1',
    name: 'DeepSeek R1',
    org: 'DeepSeek',
    overrides: {
      totalParams: 671e9,
      reason: 'Same MTP module as DeepSeek V3; 671B is the published figure.',
    },
  },
  {
    id: 'zai-org/GLM-4.5-Air',
    name: 'GLM 4.5 Air',
    org: 'Z.ai',
    overrides: {
      totalParams: 106e9,
      reason:
        "HF's safetensors index reports 110.5B including the MTP module. The published " +
        'figure is 106B, which also reproduces the stated 12B active exactly.',
    },
  },
];

// ---------------------------------------------------------------------------
// Shapes we read from Hugging Face
// ---------------------------------------------------------------------------

interface HfApiModel {
  id: string;
  /** Commit the API resolved. Returned only when explicitly requested via `expand[]=sha`. */
  sha?: string;
  downloads?: number;
  likes?: number;
  createdAt?: string;
  safetensors?: { total?: number; parameters?: Record<string, number> };
}

/** config.json is untyped by nature — every architecture adds its own fields. */
type HfConfig = Record<string, unknown>;

/**
 * Multimodal repos nest the language model under `text_config` and keep the vision tower
 * alongside it. Everything this script derives is about the text stack, so unwrap when present.
 *
 * Note the vision tower still counts toward the safetensors total — for Gemma 3 27B that is
 * roughly 0.4B of the reported parameters. Left in deliberately: those weights do occupy
 * memory when the model is loaded, unlike an MTP module that inference never touches.
 */
function textConfig(config: HfConfig): HfConfig {
  const nested = config.text_config;
  return nested && typeof nested === 'object' ? { ...config, ...(nested as HfConfig) } : config;
}

function num(config: HfConfig, key: string): number | undefined {
  const value = config[key];
  return typeof value === 'number' ? value : undefined;
}

function firstNum(config: HfConfig, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = num(config, key);
    if (value !== undefined) return value;
  }
  return undefined;
}

class DerivationError extends Error {}

function require(value: number | undefined, id: string, what: string): number {
  if (value === undefined || !Number.isFinite(value)) {
    throw new DerivationError(`${id}: could not determine ${what} from config.json`);
  }
  return value;
}

// ---------------------------------------------------------------------------
// Architecture derivation
// ---------------------------------------------------------------------------

type AttentionCore =
  | { kind: 'gqa'; kvHeads: number; headDim: number }
  | { kind: 'mla'; kvLoraRank: number; qkRopeHeadDim: number };

/**
 * Multi-head latent attention caches one compressed latent per token per layer; grouped-query
 * caches keys and values per KV head. Detected by the presence of `kv_lora_rank`, which is
 * what DeepSeek's config uses and no GQA model defines.
 *
 * Also returns the **attention projection width**, which is what QK^T and AV actually scale by
 * and is emphatically *not* `hidden_size`. A model is free to project to a wider or narrower
 * query space than its residual stream, and most current ones do: GLM-4.5-Air is 3x its hidden
 * size, Qwen3's MoEs 2x, while Gemma 3 27B and Mistral Small are *narrower*. Using hidden size
 * mis-scaled long-prompt TTFT in both directions.
 */
function deriveAttention(
  id: string,
  config: HfConfig
): { core: AttentionCore; projectionWidth: number } {
  const heads = require(num(config, 'num_attention_heads'), id, 'num_attention_heads');
  const kvLoraRank = num(config, 'kv_lora_rank');

  if (kvLoraRank !== undefined) {
    const qkRopeHeadDim = require(num(config, 'qk_rope_head_dim'), id, 'qk_rope_head_dim');
    const qkNopeHeadDim = require(num(config, 'qk_nope_head_dim'), id, 'qk_nope_head_dim');
    const vHeadDim = require(num(config, 'v_head_dim'), id, 'v_head_dim');

    return {
      core: { kind: 'mla', kvLoraRank, qkRopeHeadDim },
      // MLA is the case that forces a single averaged width rather than one head dimension:
      // its query space (qk_nope + qk_rope) and value space differ — 24576 against 16384 for
      // DeepSeek V3 — and the engine charges QK and AV at one rate.
      projectionWidth: (heads * (qkNopeHeadDim + qkRopeHeadDim) + heads * vHeadDim) / 2,
    };
  }

  const hidden = require(num(config, 'hidden_size'), id, 'hidden_size');
  // Most configs state head_dim; older ones imply it from hidden_size / num_attention_heads.
  const headDim = require(num(config, 'head_dim') ?? hidden / heads, id, 'head_dim');

  return {
    core: {
      kind: 'gqa',
      // Absent num_key_value_heads means full multi-head attention: one KV head per query head.
      kvHeads: num(config, 'num_key_value_heads') ?? heads,
      headDim,
    },
    projectionWidth: heads * headDim,
  };
}

/**
 * Per-layer attention window, or undefined when every layer attends over the full context.
 *
 * Three conventions in the wild, in order of precedence:
 *   - `layer_types` — an explicit per-layer array (gpt-oss, recent transformers exports)
 *   - `sliding_window_pattern` — Gemma 3's "every Nth layer is full attention"
 *   - a bare `sliding_window` — applies to every layer (Mistral-style), unless switched off
 */
function deriveLayerWindows(
  id: string,
  config: HfConfig,
  layers: number
): (number | null)[] | undefined {
  const window = num(config, 'sliding_window');
  const layerTypes = config.layer_types;

  if (Array.isArray(layerTypes)) {
    const sliding = layerTypes.filter((t) => typeof t === 'string' && t.includes('sliding'));

    // Both of these silently read as full attention downstream — an absent array entry and an
    // absent array are indistinguishable to `layerWindows?.[i]` — which overstates KV and
    // prefill attention for exactly the models the sliding-window handling exists to get right.
    if (layerTypes.length < layers) {
      throw new DerivationError(
        `${id}: layer_types lists ${layerTypes.length} entries for ${layers} layers. ` +
          'The missing ones would silently read as full attention.'
      );
    }
    if (sliding.length > 0 && window === undefined) {
      throw new DerivationError(
        `${id}: layer_types names ${sliding.length} sliding layers but no sliding_window size. ` +
          'Refusing to treat them as full attention.'
      );
    }
    if (sliding.length === 0) return undefined;

    return layerTypes
      .slice(0, layers)
      .map((t) => (typeof t === 'string' && t.includes('sliding') ? window! : null));
  }

  if (window === undefined || config.use_sliding_window === false) return undefined;

  const pattern = num(config, 'sliding_window_pattern');
  if (pattern !== undefined && pattern > 0) {
    // Gemma 3: layers are sliding except every `pattern`-th, which is full attention.
    return Array.from({ length: layers }, (_, i) => ((i + 1) % pattern === 0 ? null : window));
  }

  return Array.from({ length: layers }, () => window);
}

/**
 * Dtypes whose element count is a packed byte count rather than a logical parameter count.
 *
 * `I8`/`INT8` are deliberately absent. An int8-quantized tensor stores exactly one logical
 * parameter per element, so counting it as packed would send every dense INT8 repository into
 * the reconstruction path, fail the MXFP4-specific 33/32 ratio guard, and — since any seed
 * failure now blocks the write — make such a model unaddable rather than merely unusual.
 *
 * `U8` stays, because that is how MXFP4 stores its blocks. A repository that genuinely holds
 * one parameter per unsigned byte will trip the ratio guard and land in the override path,
 * which is the loud failure this file prefers to a silent rebuild.
 */
const PACKED_DTYPES = new Set(['U8', 'UINT8', 'U4', 'I4']);

/**
 * Logical parameter count.
 *
 * `safetensors.total` is a sum of tensor *elements*, which equals the parameter count only for
 * formats that store one element per parameter. FP8 does; MXFP4 does not — gpt-oss-120b reports
 * 118.24B U8 elements against 114.66B logical expert parameters, the extra 1/32 being one shared
 * scale byte per 32-value block. Taking the total at face value overstates the model by 3.6B and
 * puts the headline size at 120B where the vendor says 117B.
 *
 * For packed formats the count is rebuilt as "everything stored unpacked, plus the analytic
 * expert count", and the packed figure is used to check that assumption rather than to trust it.
 */
function deriveTotalParams(id: string, api: HfApiModel, expertParams: number): number {
  const byDtype = api.safetensors?.parameters ?? {};
  const total = api.safetensors?.total;

  const packed = Object.entries(byDtype)
    .filter(([dtype]) => PACKED_DTYPES.has(dtype.toUpperCase()))
    .reduce((sum, [, count]) => sum + count, 0);

  if (packed === 0) {
    if (total === undefined) {
      throw new DerivationError(`${id}: no safetensors parameter count published`);
    }
    return total;
  }

  if (expertParams === 0) {
    throw new DerivationError(
      `${id}: stores packed tensors but derived no routed experts, so the logical parameter ` +
        'count cannot be reconstructed. Add an override with a reason.'
    );
  }

  /**
   * MXFP4 carries one scale byte per 32 values, so packed elements are 33/32 of logical —
   * exactly, not approximately. The band is tight on purpose: a loose one would also admit a
   * model quantized *uniformly* to int8 whenever experts happen to be ~91%+ of it, and the
   * rebuild below would then throw away every non-expert parameter. That is a several-GB
   * understatement presented as a precise figure, which is the exact failure this file exists
   * to prevent. Anything not MXFP4-shaped should land in the override path instead.
   */
  const ratio = packed / expertParams;
  const expected = 33 / 32;
  if (Math.abs(ratio - expected) / expected > 0.005) {
    throw new DerivationError(
      `${id}: packed element count is ${ratio.toFixed(5)}x the analytic expert count, ` +
        `expected ${expected.toFixed(5)} for MXFP4. Either the expert shape is wrong or this ` +
        'is a different packing — add an override with a reason rather than trusting this.'
    );
  }

  const unpacked = Object.entries(byDtype)
    .filter(([dtype]) => !PACKED_DTYPES.has(dtype.toUpperCase()))
    .reduce((sum, [, count]) => sum + count, 0);

  return unpacked + expertParams;
}

interface MoeDerivation {
  expertParams: number;
  experts: { total: number; perToken: number };
}

/**
 * Routed-expert parameter count, or null for dense models.
 *
 * Assumes gated FFNs (gate, up, down — three matrices per expert), which every current MoE
 * language model uses. Partial MoE fields throw rather than defaulting, because a model that
 * declares experts but not how many are routed is one this script does not actually understand.
 */
function deriveMoe(id: string, config: HfConfig, layers: number): MoeDerivation | null {
  const total = firstNum(config, 'num_local_experts', 'n_routed_experts', 'num_experts');
  const perToken = firstNum(config, 'num_experts_per_tok', 'experts_per_token');

  if (total === undefined && perToken === undefined) return null;
  if (total === undefined || perToken === undefined) {
    throw new DerivationError(
      `${id}: partial MoE config — expert total ${total}, per-token ${perToken}. ` +
        'Refusing to guess the other.'
    );
  }

  const hidden = require(num(config, 'hidden_size'), id, 'hidden_size');
  const moeIntermediate = require(firstNum(
    config,
    'moe_intermediate_size',
    'intermediate_size'
  ), id, 'moe_intermediate_size');

  /**
   * Which layers actually carry experts. Two families use different rules, and transformers
   * implements each with a specific phase — getting it wrong overcounts by a whole layer
   * whenever the layer count isn't a multiple of the step, which for a large MoE is billions
   * of parameters in silence.
   *
   *   DeepSeek: `i >= first_k_dense_replace && i % moe_layer_freq == 0`
   *   Qwen:     `(i + 1) % decoder_sparse_step == 0`
   *
   * Both then exclude anything listed in `mlp_only_layers`.
   */
  const mlpOnly = new Set(
    Array.isArray(config.mlp_only_layers) ? (config.mlp_only_layers as number[]) : []
  );
  const firstDense = num(config, 'first_k_dense_replace');
  const moeLayerFreq = num(config, 'moe_layer_freq') ?? 1;
  const sparseStep = num(config, 'decoder_sparse_step') ?? 1;

  let moeLayers = 0;
  for (let layer = 0; layer < layers; layer++) {
    if (mlpOnly.has(layer)) continue;

    const isMoe =
      firstDense === undefined
        ? (layer + 1) % sparseStep === 0
        : layer >= firstDense && layer % moeLayerFreq === 0;

    if (isMoe) moeLayers++;
  }

  return {
    expertParams: moeLayers * total * 3 * hidden * moeIntermediate,
    experts: { total, perToken },
  };
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

const TOKEN = process.env.HF_TOKEN;
const headers: Record<string, string> = TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {};

async function fetchJson<T>(url: string, what: string): Promise<T> {
  const response = await fetch(url, { headers });
  if (!response.ok) {
    const hint =
      response.status === 401 || response.status === 403
        ? ' (gated repo — set HF_TOKEN, or seed an open mirror instead)'
        : '';
    throw new DerivationError(`${what}: HTTP ${response.status} from ${url}${hint}`);
  }
  return (await response.json()) as T;
}

/**
 * Tensors belonging to the language stack. Everything a text token's forward pass touches
 * lives under one of these; `lm_head` is listed because untied models keep it at the root.
 */
const LANGUAGE_PREFIXES = ['model.', 'language_model.', 'transformer.', 'lm_head.'];

/**
 * Tensors belonging to a non-text tower. These occupy memory when the model loads — so they
 * stay in `totalParams` — but a text-only request never runs them, so they must not be charged
 * per token. For Gemma 3 that is ~0.42B, which is a few percent of prefill.
 */
const NON_LANGUAGE_PREFIXES = [
  'vision_tower.',
  'vision_model.',
  'multi_modal_projector.',
  'audio_tower.',
  'audio_projector.',
];

/**
 * Non-language prefixes are tested first, and against the name with any leading `model.`
 * removed.
 *
 * Order matters here in a way that fails silently if reversed. `model.` is a language prefix,
 * and newer transformers multimodal exports nest the whole model under it —
 * `model.vision_tower.*` alongside `model.language_model.*`. Matching language first would
 * classify every vision tensor as language, report zero non-language parameters, and fold the
 * tower straight into the per-token count with no error anywhere. The seeded Gemma 3 mirrors
 * use the flat layout today, so this guards the next multimodal repo rather than a current one.
 */
function classifyTensor(name: string): 'language' | 'other' | 'unknown' {
  const unwrapped = name.startsWith('model.') ? name.slice('model.'.length) : name;
  if (NON_LANGUAGE_PREFIXES.some((p) => unwrapped.startsWith(p))) return 'other';
  if (LANGUAGE_PREFIXES.some((p) => name.startsWith(p) || unwrapped.startsWith(p))) {
    return 'language';
  }
  return 'unknown';
}

/**
 * A safetensors file opens with a little-endian u64 header length followed by that many bytes
 * of JSON describing every tensor's dtype and shape. Two range requests get it without pulling
 * the weights themselves, which for these repos would be hundreds of gigabytes.
 */
async function fetchSafetensorsHeader(
  id: string,
  revision: string,
  shard: string
): Promise<Record<string, { dtype?: string; shape?: number[] }>> {
  const url = `https://huggingface.co/${id}/resolve/${revision}/${shard}`;

  const lengthResponse = await fetch(url, { headers: { ...headers, Range: 'bytes=0-7' } });
  // 206 specifically, not merely ok: a mirror that ignores Range answers 200 with the whole
  // shard, which on the unsharded path means buffering an entire model into memory.
  if (lengthResponse.status !== 206) {
    throw new DerivationError(
      `${id}: ${shard} answered ${lengthResponse.status} to a range request, expected 206. ` +
        'Refusing to download a full shard to read its header.'
    );
  }
  const lengthBytes = Buffer.from(await lengthResponse.arrayBuffer());
  if (lengthBytes.length < 8) {
    throw new DerivationError(`${id}: ${shard} returned a short range, so shapes are unreadable`);
  }

  // The length is whatever the first eight bytes happen to say, so cap it: a file that is not
  // safetensors at all would otherwise become a multi-gigabyte allocation.
  const headerLength = Number(lengthBytes.readBigUInt64LE(0));
  const MAX_HEADER_BYTES = 100 * 1024 * 1024;
  if (!Number.isFinite(headerLength) || headerLength <= 0 || headerLength > MAX_HEADER_BYTES) {
    throw new DerivationError(
      `${id}: ${shard} declares a ${headerLength}-byte header, which is not a safetensors file`
    );
  }

  const headerResponse = await fetch(url, {
    headers: { ...headers, Range: `bytes=8-${8 + headerLength - 1}` },
  });
  // The same 206 check as above, and it has to be repeated rather than inferred: the first
  // range being honoured does not promise the second one will be. A 200 here would buffer the
  // entire shard, which is the exact download this function exists to avoid.
  if (headerResponse.status !== 206) {
    throw new DerivationError(
      `${id}: ${shard} answered ${headerResponse.status} to the header range request, ` +
        'expected 206. Refusing to buffer a full shard.'
    );
  }
  const headerBytes = Buffer.from(await headerResponse.arrayBuffer());
  if (headerBytes.length !== headerLength) {
    throw new DerivationError(
      `${id}: ${shard} returned ${headerBytes.length} header bytes, expected ${headerLength}`
    );
  }
  const header = JSON.parse(headerBytes.toString('utf8'));
  delete header.__metadata__;
  return header as Record<string, { dtype?: string; shape?: number[] }>;
}

/** Tensor names in a repo, and which shard each lives in. */
async function fetchTensorMap(id: string, revision: string): Promise<Record<string, string>> {
  const url = `https://huggingface.co/${id}/raw/${revision}/model.safetensors.index.json`;
  const response = await fetch(url, { headers });

  if (response.ok) {
    const index = (await response.json()) as { weight_map?: Record<string, string> };
    if (!index.weight_map) {
      throw new DerivationError(`${id}: safetensors index has no weight_map`);
    }
    return index.weight_map;
  }
  if (response.status !== 404) {
    throw new DerivationError(`${id}: HTTP ${response.status} fetching the safetensors index`);
  }

  // Unsharded repo — the single file's own header is the index.
  const header = await fetchSafetensorsHeader(id, revision, 'model.safetensors');
  return Object.fromEntries(Object.keys(header).map((name) => [name, 'model.safetensors']));
}

interface StackShape {
  /**
   * True when the output projection reuses the input embedding table. Read from the tensor
   * list rather than `config.tie_word_embeddings`, which is absent on both Gemma 3 repos even
   * though they are tied — trusting it would wrongly subtract a 1B-parameter table that decode
   * reads in full on every step.
   */
  tiedEmbeddings: boolean;
  /** Parameters in non-text towers, excluded from the per-token count but kept in the total. */
  nonLanguageParams: number;
}

/** Names an untied output projection is stored under, across architectures. */
const OUTPUT_HEAD_SUFFIXES = [
  'lm_head.weight',
  'output_layer.weight',
  'output.weight',
  'embed_out.weight',
];

async function deriveStackShape(
  id: string,
  revision: string,
  declaredTied: boolean | undefined
): Promise<StackShape> {
  const weightMap = await fetchTensorMap(id, revision);
  const names = Object.keys(weightMap);

  const unknown = names.filter((name) => classifyTensor(name) === 'unknown');
  if (unknown.length > 0) {
    throw new DerivationError(
      `${id}: ${unknown.length} tensors match no known prefix (e.g. ${unknown[0]}). ` +
        'Classify them before shipping, rather than silently charging them per token.'
    );
  }

  /**
   * Tied means there is no separate output projection — so the test has to be able to find one
   * under whatever name the architecture uses. `lm_head.weight` is the transformers convention;
   * GLM-family exports use `output_layer.weight`, and older ones `output.weight` or
   * `embed_out.weight`. Matching only the first would declare such a model tied and keep its
   * input embedding in the per-token count, overstating decode traffic by a whole vocabulary
   * table — the same magnitude of error, in the opposite direction, as the bug this field
   * exists to fix.
   */
  const outputHead = names.find((name) => OUTPUT_HEAD_SUFFIXES.some((s) => name.endsWith(s)));
  const tiedEmbeddings = outputHead === undefined;

  /**
   * `tie_word_embeddings` is not trustworthy enough to derive from — it is absent on both Gemma
   * 3 repos despite them being tied — but when a repo *does* state it, a disagreement means
   * this list of names is incomplete for that architecture. Better to stop than to guess which
   * side is right.
   */
  if (declaredTied === false && tiedEmbeddings) {
    throw new DerivationError(
      `${id}: config says the embeddings are untied, but no output projection matched ` +
        `${OUTPUT_HEAD_SUFFIXES.join(', ')}. Add this architecture's output tensor name.`
    );
  }

  const otherShards = [
    ...new Set(names.filter((n) => classifyTensor(n) === 'other').map((n) => weightMap[n])),
  ];
  if (otherShards.length === 0) return { tiedEmbeddings, nonLanguageParams: 0 };

  let nonLanguageParams = 0;
  for (const shard of otherShards) {
    const header = await fetchSafetensorsHeader(id, revision, shard);
    for (const [name, tensor] of Object.entries(header)) {
      if (classifyTensor(name) !== 'other') continue;
      if (tensor.dtype && PACKED_DTYPES.has(tensor.dtype.toUpperCase())) {
        throw new DerivationError(
          `${id}: non-language tensor ${name} is packed (${tensor.dtype}), so its element ` +
            'count is not a parameter count. Add an override rather than subtracting it.'
        );
      }
      nonLanguageParams += (tensor.shape ?? []).reduce((a, b) => a * b, 1);
    }
  }

  return { tiedEmbeddings, nonLanguageParams };
}

async function buildModel(seed: Seed) {
  const api = await fetchJson<HfApiModel>(
    `https://huggingface.co/api/models/${seed.id}?expand[]=safetensors&expand[]=downloads&expand[]=likes&expand[]=createdAt&expand[]=sha`,
    seed.id
  );

  /**
   * Every subsequent fetch is pinned to this commit rather than to `main`.
   *
   * A model row is assembled from four requests — this one, config.json, the safetensors index,
   * and the shard headers. Resolving `main` separately each time means a publisher pushing
   * mid-run can straddle two revisions and produce a row whose expert counts, embedding size
   * and totals describe different models. Nothing would fail; the numbers would just be wrong
   * together, which is the failure mode this file exists to prevent.
   *
   * One caveat the pin does not cover: the API's `safetensors` block is an asynchronously
   * computed cache and is not guaranteed to correspond to the `sha` reported beside it.
   */
  const revision = api.sha;
  if (!revision) {
    throw new DerivationError(
      `${seed.id}: API returned no commit sha, so the fetches cannot be pinned to one revision`
    );
  }

  const config = textConfig(
    await fetchJson<HfConfig>(
      `https://huggingface.co/${seed.id}/raw/${revision}/config.json`,
      seed.id
    )
  );

  const layers = require(num(config, 'num_hidden_layers'), seed.id, 'num_hidden_layers');
  const hiddenSize = require(num(config, 'hidden_size'), seed.id, 'hidden_size');
  const vocabSize = require(num(config, 'vocab_size'), seed.id, 'vocab_size');

  const moe = deriveMoe(seed.id, config, layers);
  const expertParams = moe?.expertParams ?? 0;

  /**
   * Models carrying a Multi-Token Prediction module report it in their safetensors index even
   * though ordinary inference never loads it — DeepSeek V3 by ~13B, GLM-4.5-Air by ~4B.
   * Subtracting it analytically would mean reconstructing an architecture-specific block, so
   * instead this refuses to guess and asks for the published figure. A new MTP model appearing
   * in a weekly refresh should stop the build, not quietly ship an inflated weight estimate.
   */
  if (
    (num(config, 'num_nextn_predict_layers') ?? 0) > 0 &&
    seed.overrides?.totalParams === undefined
  ) {
    throw new DerivationError(
      `${seed.id}: declares num_nextn_predict_layers, so its safetensors total includes an ` +
        'MTP module that inference does not load. Add a totalParams override with the ' +
        "vendor's published parameter count and a reason."
    );
  }

  const totalParams = seed.overrides?.totalParams ?? deriveTotalParams(seed.id, api, expertParams);

  if (expertParams >= totalParams) {
    throw new DerivationError(
      `${seed.id}: derived expert params (${expertParams}) exceed total (${totalParams}) — ` +
        'the expert-shape assumption is wrong for this architecture'
    );
  }

  /**
   * The input embedding is a row lookup, not a matmul: decoding reads one row of it per token,
   * not the whole table. Excluding it from the active count is both physically right and what
   * reconciles this derivation with vendors' published figures — it is the difference between
   * 5.75B and the stated 5.1B for gpt-oss-120b, and between 12.6B and 12B for GLM-4.5-Air.
   */
  const denseParams = totalParams - expertParams;
  const embeddingParams = vocabSize * hiddenSize;
  const activeDense = Math.max(0, denseParams - embeddingParams);
  const activeParams = moe
    ? activeDense + (moe.experts.perToken / moe.experts.total) * expertParams
    : activeDense;

  /**
   * `activeParams` above is the *published* convention, and it is not what a decode step reads.
   * Two corrections separate them, and both were wrong in the direction of a slower machine:
   *
   *   - **Tied embeddings.** When a model reuses the embedding table as its output projection,
   *     that table is a full vocab matmul on every step. Subtracting it is right for untied
   *     models like gpt-oss and wrong for tied ones like Gemma 3 and Qwen3-4B.
   *   - **Non-text towers.** Gemma 3's vision encoder occupies memory but does not run for a
   *     text token, so it belongs in `totalParams` and not in the per-token count.
   *
   * Kept as its own field rather than folded into `activeParams`, because the published figure
   * is what the catalog tests check against vendors and what users recognise on a model card.
   */
  const stack = await deriveStackShape(
    seed.id,
    revision,
    typeof config.tie_word_embeddings === 'boolean' ? config.tie_word_embeddings : undefined
  );
  const activeDenseParams = Math.max(
    0,
    denseParams - stack.nonLanguageParams - (stack.tiedEmbeddings ? 0 : embeddingParams)
  );

  const layerWindows = deriveLayerWindows(seed.id, config, layers);
  const attention = deriveAttention(seed.id, config);
  const quantMethod = (config.quantization_config as Record<string, unknown> | undefined)
    ?.quant_method;

  // A mirror's own traffic is not the model's. Weights still come from the mirror, so only
  // the popularity figures are re-fetched, and only when a seed names a canonical repo.
  const popularitySource = seed.popularityId
    ? await fetchJson<HfApiModel>(
        `https://huggingface.co/api/models/${seed.popularityId}?expand[]=downloads&expand[]=likes`,
        `${seed.id} popularity`
      )
    : api;

  if (seed.popularityId) {
    // HF silently redirects renamed repos, so a stale or mistyped canonical id would return a
    // different model's traffic and look entirely plausible.
    if (popularitySource.id !== seed.popularityId) {
      throw new DerivationError(
        `${seed.id}: popularity id ${seed.popularityId} resolved to ${popularitySource.id}`
      );
    }
    // These were asked for explicitly, so absent is a signal rather than a default. Falling back
    // to 0 would recreate the exact bug this indirection exists to fix.
    if (popularitySource.downloads === undefined) {
      throw new DerivationError(
        `${seed.id}: ${seed.popularityId} returned no download count to substitute`
      );
    }
  }

  return {
    id: seed.id,
    name: seed.name,
    org: seed.org,
    totalParams,
    activeParams,
    activeDenseParams,
    expertParams,
    ...(moe ? { experts: moe.experts } : {}),
    tiedEmbeddings: stack.tiedEmbeddings,
    ...(stack.nonLanguageParams > 0 ? { nonLanguageParams: stack.nonLanguageParams } : {}),
    layers,
    hiddenSize,
    vocabSize,
    attention: {
      core: attention.core,
      projectionWidth: attention.projectionWidth,
      ...(layerWindows ? { layerWindows } : {}),
    },
    ...(typeof quantMethod === 'string' ? { nativeQuant: quantMethod } : {}),
    maxContext: require(num(config, 'max_position_embeddings'), seed.id, 'max_position_embeddings'),
    popularity: {
      downloads: popularitySource.downloads ?? 0,
      likes: popularitySource.likes ?? 0,
      // Recorded so the figures are attributable: these describe the canonical repo, while
      // every other field on this row describes the mirror the weights were read from.
      ...(seed.popularityId ? { measuredOn: seed.popularityId } : {}),
    },
    ...(api.createdAt ? { releasedAt: api.createdAt } : {}),
    // The exact commit every figure on this row was derived from, so a suspicious number can be
    // reproduced rather than merely re-fetched against whatever `main` says today.
    revision,
    source: `https://huggingface.co/${seed.id}/tree/${revision}`,
    ...(seed.overrides ? { overrideNote: seed.overrides.reason } : {}),
  };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  const allowPartial = process.argv.includes('--allow-partial');
  const models = [];
  const failures: string[] = [];

  for (const seed of SEEDS) {
    try {
      const model = await buildModel(seed);
      models.push(model);
      const moe =
        model.expertParams > 0 ? ` MoE ${(model.activeParams / 1e9).toFixed(1)}B act` : '';
      const sliding = model.attention.layerWindows ? ' sliding' : '';
      console.log(
        `  ok  ${seed.id.padEnd(48)} ${(model.totalParams / 1e9).toFixed(1).padStart(6)}B` +
          ` ${model.attention.core.kind}${moe}${sliding}`
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push(message);
      console.error(`  FAIL ${message}`);
    }
  }

  console.log(`\n${models.length} ok, ${failures.length} failed, ${SEEDS.length} seeded`);

  /**
   * Any failure blocks the write.
   *
   * The artifact is committed, so a partial run does not merely produce a smaller catalog — it
   * deletes models from the product, and the loader reads only `models` and never surfaces
   * `failures`. A tolerance threshold made that outcome reachable from a transient Hugging Face
   * error: five of seventeen seeds could 503 and the run would still exit 0 having dropped 29%
   * of the catalog.
   *
   * `--allow-partial` keeps the original escape hatch, because a single permanently-gated repo
   * should not block every future refresh. It just has to be asked for.
   */
  if (failures.length > 0 && !allowPartial) {
    console.error(
      `\n${failures.length} seed(s) failed — refusing to overwrite the catalog with a partial ` +
        'list. Fix the failures, or pass --allow-partial to write anyway.'
    );
    process.exit(1);
  }
  if (failures.length > 0) {
    console.warn(`\n--allow-partial: writing without ${failures.length} seed(s).`);
  }

  if (dryRun) {
    console.log('\n--dry-run: nothing written.');
    return;
  }

  writeFileSync(
    OUT,
    JSON.stringify(
      {
        $comment:
          'GENERATED by scripts/build-catalog.ts from Hugging Face. Do not edit by hand — ' +
          'run `npm run catalog`. Corrections belong in the seed list, with a reason.',
        generatedAt: new Date().toISOString(),
        // Recorded so a refresh that quietly lost models is visible in the artifact rather
        // than only in a CI log nobody reads.
        failures,
        models,
      },
      null,
      2
    ) + '\n'
  );
  console.log(`\nWrote ${OUT}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
