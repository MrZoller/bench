import { describe, expect, it } from 'vitest';
import {
  NOT_SEEDED,
  SEEDS,
  deriveAttention,
  deriveLayerWindows,
  deriveMoe,
  seededIds,
  unseededCandidates,
} from './build-catalog';

/**
 * What the generator decides about a model's attention stack, and what it refuses to decide.
 *
 * Untested until now for a mechanical reason worth recording: `build-catalog.ts` called `main()` at
 * module scope, so importing it started seventeen rounds of network fetches and no test could reach
 * the pure derivations at all. Both defects below were three lines from a unit test the whole time.
 *
 * Every config fragment here is the real thing, trimmed to the keys these two functions read and
 * cited to the repo it came from. Nothing in this file is recalled — that is the same rule the
 * script itself follows, and the reason a fixture with a made-up `head_dim` would be worse than no
 * fixture.
 */

const KIB = 1024;
const GIB = 1024 ** 3;

/**
 * The naive formula — keys and values, per KV head, per layer — written out here rather than
 * imported so it cannot drift into agreeing with whatever the engine happens to do. This is what
 * the GQA branch's output means once `kv.ts` consumes it, and the whole subject of the issue is
 * which *layers* it gets applied to.
 */
const gqaKvBytesPerToken = (layers: number, kvHeads: number, headDim: number, elemBytes = 2) =>
  layers * 2 * kvHeads * headDim * elemBytes;

/** The MLA form of the same, matching `src/engine/kv.ts`: one compressed latent plus the RoPE part. */
const mlaKvBytesPerToken = (
  layers: number,
  kvLoraRank: number,
  qkRopeHeadDim: number,
  elemBytes = 2
) => layers * (kvLoraRank + qkRopeHeadDim) * elemBytes;

/**
 * The KV figures in this file are read out of what `deriveAttention` actually returns rather than
 * written down beside it, so a test that names a ratio cannot keep passing once the derivation it
 * describes has been deleted. These two pull the numbers back out of the returned core.
 */
const gqaOf = (core: { kind: string }) => {
  if (core.kind !== 'gqa') throw new Error(`expected a gqa core, got ${core.kind}`);
  return core as { kind: 'gqa'; kvHeads: number; headDim: number };
};

const mlaOf = (core: { kind: string }) => {
  if (core.kind !== 'mla') throw new Error(`expected an mla core, got ${core.kind}`);
  return core as { kind: 'mla'; kvLoraRank: number; qkRopeHeadDim: number };
};

// ---------------------------------------------------------------------------
// The shapes the shipped catalog is built from
// ---------------------------------------------------------------------------

/** https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json — 24 entries, sliding first. */
const GPT_OSS_20B = {
  num_hidden_layers: 24,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  head_dim: 64,
  hidden_size: 2880,
  sliding_window: 128,
  layer_types: Array.from({ length: 24 }, (_, i) =>
    i % 2 === 0 ? 'sliding_attention' : 'full_attention'
  ),
};

/**
 * https://huggingface.co/unsloth/gemma-3-12b-it/raw/main/config.json — pattern, not an array.
 *
 * `cache_implementation: "hybrid"` is in the fixture because it is in the config, and because it is
 * the trap on the chunked-attention axis: it looks like a general "this stack is not uniform" signal
 * and is not one. Gemma 3 12B and 27B both carry it while deriving their windows correctly from
 * `sliding_window_pattern`, so a guard keyed on it would refuse two shipped seeds. `text_config` is
 * merged into the top level by `textConfig` before either derivation sees it, which is the shape
 * these fixtures are written in.
 */
const GEMMA_3_12B = {
  num_hidden_layers: 48,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  head_dim: 256,
  hidden_size: 3840,
  sliding_window: 1024,
  sliding_window_pattern: 6,
  cache_implementation: 'hybrid',
};

/** https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json — states a window and switches it off. */
const QWEN3_32B = {
  num_hidden_layers: 64,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  head_dim: 128,
  hidden_size: 5120,
  sliding_window: null,
  use_sliding_window: false,
};

/**
 * https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/raw/main/config.json
 *
 * The plainest shape in the catalog and the majority of it: no window keys of any kind, and no
 * `head_dim` either, so the dimension is implied from `hidden_size / num_attention_heads`. Nine of
 * the seventeen seeds look like this.
 */
const LLAMA_31_8B = {
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  hidden_size: 4096,
};

/** https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json — the MLA branch. */
const DEEPSEEK_V3 = {
  num_hidden_layers: 61,
  num_attention_heads: 128,
  hidden_size: 7168,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  qk_nope_head_dim: 128,
  v_head_dim: 128,
};

// ---------------------------------------------------------------------------
// The third family — a stack that mixes attention with something else
// ---------------------------------------------------------------------------

/**
 * https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
 *
 * The hard case, because it carries **no per-layer array at all**: `full_attention_interval: 4`
 * with the gated DeltaNet block's own dimensions beside it. `num_attention_heads`,
 * `num_key_value_heads` and `head_dim` all sit exactly where the GQA branch expects them.
 */
const QWEN3_NEXT_80B = {
  num_hidden_layers: 48,
  num_attention_heads: 16,
  num_key_value_heads: 2,
  head_dim: 256,
  hidden_size: 2048,
  full_attention_interval: 4,
  linear_conv_kernel_dim: 4,
  linear_key_head_dim: 128,
  linear_num_key_heads: 16,
  linear_num_value_heads: 32,
  linear_value_head_dim: 128,
  use_sliding_window: false,
};

/**
 * https://huggingface.co/ibm-granite/granite-4.0-h-small/raw/main/config.json
 *
 * The other half of the class: the split *is* in a per-layer array, and every entry in it is one
 * the old substring filter did not match. `attention` at layers 5, 15, 25 and 35; `mamba` on the
 * other 36.
 */
const GRANITE_4_H_SMALL = {
  num_hidden_layers: 40,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  hidden_size: 4096,
  mamba_d_state: 128,
  mamba_d_conv: 4,
  mamba_d_head: 64,
  mamba_n_heads: 128,
  mamba_expand: 2,
  mamba_n_groups: 1,
  layer_types: Array.from({ length: 40 }, (_, i) => ((i - 5) % 10 === 0 ? 'attention' : 'mamba')),
};

/**
 * https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json
 *
 * The case a flat list of exact key names structurally cannot see: the entire Kimi-Delta linear
 * block lives inside one nested `linear_attn_config` object, while `kv_lora_rank` sits at the top
 * level exactly where the MLA branch looks for it. So the model derived as clean 27-layer MLA —
 * right about the latent's shape, 3.86x wrong about how many layers hold one, on a model whose
 * headline claim is a 75%-smaller KV cache.
 */
const KIMI_LINEAR_48B = {
  num_hidden_layers: 27,
  num_attention_heads: 32,
  num_key_value_heads: 32,
  head_dim: 72,
  hidden_size: 2304,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  qk_nope_head_dim: 128,
  v_head_dim: 128,
  linear_attn_config: {
    full_attn_layers: [4, 8, 12, 16, 20, 24, 27],
    head_dim: 128,
    kda_layers: [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26],
    num_heads: 32,
    short_conv_kernel_size: 4,
  },
};

/**
 * https://huggingface.co/LiquidAI/LFM2-1.2B/raw/main/config.json
 *
 * The same architecture spelled two ways in two exports, which is why both axes are guarded. The
 * 1.2B and 350M state their split as `full_attn_idxs` with no `layer_types` at all; the 2.6B and
 * 8B-A1B state it as `layer_types: ["conv", ...]`. Guarding only `layer_types` refuses one and
 * silently mis-prices the other, and mis-pricing is the direction that ships.
 *
 * No `head_dim`, so the dimension is implied from `hidden_size / num_attention_heads` — the same
 * path Llama 3.1 takes, which is what made this read as an ordinary GQA row.
 */
const LFM2_1_2B = {
  num_hidden_layers: 16,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  hidden_size: 2048,
  conv_L_cache: 3,
  full_attn_idxs: [2, 5, 8, 10, 12, 14],
};

/**
 * https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning/raw/main/config.json
 *
 * A hybrid that says so in one key and nothing else: `mb_per_layer: 2` on a `Phi4FlashForCausalLM`
 * whose other 34 fields describe a perfectly ordinary 32-layer GQA stack with `sliding_window: 512`.
 * How the Mamba blocks are distributed is in the modelling code rather than the config, so unlike
 * the four above there is no split to state — which makes it the case that proves the refusal does
 * not depend on being able to count.
 */
const PHI_4_MINI_FLASH = {
  num_hidden_layers: 32,
  num_attention_heads: 40,
  num_key_value_heads: 20,
  hidden_size: 2560,
  sliding_window: 512,
  mb_per_layer: 2,
};

/**
 * https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/raw/main/config.json
 *
 * The regression case for matching key *prefixes* rather than exact names. This export spells its
 * Mamba-2 block `mamba_state_dim`, `mamba_head_dim`, `mamba_num_heads`, `mamba_num_groups`,
 * `mamba_hidden_act`, `mamba_proj_bias` — not one of which appeared on the first draft's list of
 * thirteen exact keys, which had been read off Granite's `mamba_d_state` / `mamba_d_head` spelling.
 * It refused only because `hybrid_override_pattern` happened to be in the same config.
 */
const NEMOTRON_NANO_9B = {
  num_hidden_layers: 56,
  num_attention_heads: 40,
  num_key_value_heads: 8,
  head_dim: 128,
  hidden_size: 4480,
  hybrid_override_pattern: 'M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-',
  mamba_head_dim: 80,
  mamba_hidden_act: 'silu',
  mamba_num_groups: 8,
  mamba_num_heads: 128,
  mamba_proj_bias: false,
  mamba_state_dim: 128,
  sliding_window: null,
};

/**
 * https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json (`text_config`)
 *
 * Chunked attention, which is a window convention rather than a layer-stack one — Scout's 48 layers
 * are all attention, but 36 of them attend inside an 8192-token block instead of over everything
 * before them. There is no `layer_types` here at all, so a closed `layer_types` vocabulary never
 * runs: the only thing in the config that says the stack is not uniform is `attention_chunk_size`.
 */
const LLAMA_4_SCOUT_TEXT = {
  num_hidden_layers: 48,
  num_attention_heads: 40,
  num_key_value_heads: 8,
  head_dim: 128,
  hidden_size: 5120,
  attention_chunk_size: 8192,
  cache_implementation: 'hybrid',
  // 1 where the layer uses RoPE and attends chunked, 0 for the NoPE global layers — every fourth.
  no_rope_layers: Array.from({ length: 48 }, (_, i) => ((i + 1) % 4 === 0 ? 0 : 1)),
};

/**
 * https://huggingface.co/unsloth/gemma-3-4b-it/raw/main/config.json (`text_config`)
 *
 * The small end of the catalog, and the row that has to keep deriving once the Gemma *4* guards
 * exist: same family, same `sliding_window_pattern`, and none of the three cache keys that refuse
 * Gemma 4. Its 4 KV heads at 256 with a 1024-token window on 29 of 34 layers is also why the
 * all-blocked Matrix scenario is no longer reachable — see `src/components/Matrix.test.tsx`.
 */
const GEMMA_3_4B = {
  num_hidden_layers: 34,
  num_attention_heads: 8,
  num_key_value_heads: 4,
  head_dim: 256,
  hidden_size: 2560,
  sliding_window: 1024,
  sliding_window_pattern: 6,
};

/**
 * https://huggingface.co/ibm-granite/granite-4.1-8b/raw/main/config.json
 *
 * IBM's current generation, and the plainest possible shape: `GraniteForCausalLM` with no window
 * keys, no `head_dim` and — unlike every Granite 4.0-h export — no `mamba_*` block. Seeded, and the
 * reason IBM is represented at all.
 */
const GRANITE_4_1_8B = {
  num_hidden_layers: 40,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  hidden_size: 4096,
};

/**
 * https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16/raw/main/config.json (`text_config`)
 *
 * `Cohere2MoeForCausalLM`, and a third `layer_types` phase: full attention every fourth layer where
 * gpt-oss alternates and Gemma 3 states a pattern of 6. Seeded, so this is a regression test for the
 * vocabulary as much as a shape.
 */
const COMMAND_A_PLUS = {
  num_hidden_layers: 32,
  num_attention_heads: 128,
  num_key_value_heads: 8,
  head_dim: 128,
  hidden_size: 4096,
  sliding_window: 4096,
  layer_types: Array.from({ length: 32 }, (_, i) =>
    (i + 1) % 4 === 0 ? 'full_attention' : 'sliding_attention'
  ),
};

/**
 * https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/raw/main/config.json (`text_config`)
 *
 * MLA that is not DeepSeek's MLA: `kv_lora_rank` 256 rather than 512, and a query space narrower
 * than its value space. Seeded, and the check that the sparse-indexer guard hoisted out of the MLA
 * branch still lets ordinary MLA through.
 */
const MISTRAL_SMALL_4 = {
  num_hidden_layers: 36,
  num_attention_heads: 32,
  num_key_value_heads: 32,
  head_dim: 128,
  hidden_size: 4096,
  kv_lora_rank: 256,
  qk_rope_head_dim: 64,
  qk_nope_head_dim: 64,
  v_head_dim: 128,
  sliding_window: null,
};

// ---------------------------------------------------------------------------
// The fourth, fifth and sixth ways a stack can fail to be uniform
// ---------------------------------------------------------------------------

/**
 * https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/raw/main/config.json
 *
 * A per-block NAS export (`DeciLMForCausalLM`), which the issue that added it to the seed list
 * listed as an ordinary addition. It is not: `block_configs` describes each of the 80 blocks
 * separately, 31 of them with no attention at all, and `num_key_value_heads` is *explicitly null*
 * because the grouping is stated per block as `n_heads_in_group: 8`.
 *
 * The pattern below is the real one, read out of the config: `1` where the block attends.
 */
const NEMOTRON_SUPER_ATTENDS = [
  1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

const NEMOTRON_SUPER = {
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_key_value_heads: null,
  hidden_size: 8192,
  intermediate_size: null,
  block_configs: NEMOTRON_SUPER_ATTENDS.map((attends) => ({
    attention: attends
      ? { n_heads_in_group: 8, no_op: false, replace_with_linear: false }
      : { n_heads_in_group: null, no_op: true, replace_with_linear: false },
    ffn: { ffn_mult: 2.625, no_op: false, replace_with_linear: false },
  })),
};

/**
 * https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json
 *
 * The same sparse-attention indexer V3.2-Exp is refused for, on a config that does *not* take the
 * MLA branch: no `kv_lora_rank` at all, one KV head at `head_dim: 512`, and a bare
 * `qk_rope_head_dim`. So the guard that named this exact quantity never ran.
 */
const DEEPSEEK_V4_FLASH = {
  num_hidden_layers: 43,
  num_attention_heads: 64,
  num_key_value_heads: 1,
  head_dim: 512,
  hidden_size: 4096,
  qk_rope_head_dim: 64,
  index_n_heads: 64,
  index_head_dim: 128,
  index_topk: 512,
  sliding_window: 128,
  n_routed_experts: 256,
  num_experts_per_tok: 6,
  moe_intermediate_size: 2048,
};

/**
 * https://huggingface.co/google/gemma-4-31B-it/raw/main/config.json (`text_config`)
 *
 * The most-downloaded current model in the field, and an *ordinary* sliding-window GQA config by
 * every test above: `layer_types` entirely inside the closed vocabulary, `sliding_window` stated,
 * no hybrid key, no indexer, no `block_configs`. Three keys say the cache is not what the GQA branch
 * charges — `attention_k_eq_v`, and a second KV shape for the global layers.
 */
const GEMMA_4_31B = {
  num_hidden_layers: 60,
  num_attention_heads: 32,
  num_key_value_heads: 16,
  head_dim: 256,
  global_head_dim: 512,
  num_global_key_value_heads: 4,
  num_kv_shared_layers: 0,
  attention_k_eq_v: true,
  hidden_size: 5376,
  sliding_window: 1024,
  layer_types: Array.from({ length: 60 }, (_, i) =>
    (i + 1) % 6 === 0 ? 'full_attention' : 'sliding_attention'
  ),
};

/**
 * https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json (`text_config`)
 *
 * The other half of the same family: `attention_k_eq_v` is *false* here, and instead 18 of the 42
 * layers share an earlier layer's cache. Both axes ship in one generation, which is why neither key
 * is sufficient on its own.
 */
const GEMMA_4_E4B = {
  num_hidden_layers: 42,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  head_dim: 256,
  global_head_dim: 512,
  num_global_key_value_heads: null,
  num_kv_shared_layers: 18,
  attention_k_eq_v: false,
  hidden_size: 2560,
  sliding_window: 512,
  layer_types: Array.from({ length: 42 }, (_, i) =>
    (i + 1) % 6 === 0 ? 'full_attention' : 'sliding_attention'
  ),
};

describe('the attention shapes the shipped catalog is built from', () => {
  /**
   * First, because the vocabulary swap below is the kind of tightening that can quietly reject the
   * models already in the product. Every one of these was passing before the hybrid guards existed
   * and has to still pass after them.
   */
  it('reads gpt-oss as alternating sliding and full attention', () => {
    expect(deriveLayerWindows('openai/gpt-oss-20b', GPT_OSS_20B, 24)).toEqual(
      Array.from({ length: 24 }, (_, i) => (i % 2 === 0 ? 128 : null))
    );
    expect(deriveAttention('openai/gpt-oss-20b', GPT_OSS_20B, 24)).toEqual({
      core: { kind: 'gqa', kvHeads: 8, headDim: 64 },
      projectionWidth: 64 * 64,
    });
  });

  it("reads Gemma 3's pattern as sliding except every sixth layer, hybrid cache and all", () => {
    // And so proves `cache_implementation: "hybrid"` is not the chunked-attention signal: this seed
    // declares it and derives correctly.
    const windows = deriveLayerWindows('unsloth/gemma-3-12b-it', GEMMA_3_12B, 48);
    expect(windows).toEqual(
      Array.from({ length: 48 }, (_, i) => ((i + 1) % 6 === 0 ? null : 1024))
    );
  });

  it('reads a switched-off window as full attention everywhere', () => {
    expect(deriveLayerWindows('Qwen/Qwen3-32B', QWEN3_32B, 64)).toBeUndefined();
    expect(deriveAttention('Qwen/Qwen3-32B', QWEN3_32B, 64).core).toEqual({
      kind: 'gqa',
      kvHeads: 8,
      headDim: 128,
    });
  });

  it('reads a config with no window keys at all as full attention, head dim implied', () => {
    expect(
      deriveLayerWindows('NousResearch/Meta-Llama-3.1-8B-Instruct', LLAMA_31_8B, 32)
    ).toBeUndefined();
    expect(deriveAttention('NousResearch/Meta-Llama-3.1-8B-Instruct', LLAMA_31_8B, 32)).toEqual({
      core: { kind: 'gqa', kvHeads: 8, headDim: 128 },
      projectionWidth: 32 * 128,
    });
  });

  it('still reaches the MLA branch for DeepSeek V3', () => {
    expect(deriveAttention('deepseek-ai/DeepSeek-V3', DEEPSEEK_V3, 61).core).toEqual({
      kind: 'mla',
      kvLoraRank: 512,
      qkRopeHeadDim: 64,
    });
  });

  /**
   * The rows #77 added, checked against the guards #77 added.
   *
   * Half of a coverage change is not rejecting what it just admitted: three of the four guards below
   * are new, and a guard keyed one notch too wide takes the seed list with it. Gemma 3 is the sharp
   * one — same publisher, same window convention, one generation apart from a family that is refused.
   */
  it('reads Gemma 3 4B the way it reads its bigger siblings, Gemma 4 guards and all', () => {
    expect(deriveLayerWindows('unsloth/gemma-3-4b-it', GEMMA_3_4B, 34)).toEqual(
      Array.from({ length: 34 }, (_, i) => ((i + 1) % 6 === 0 ? null : 1024))
    );
    expect(deriveAttention('unsloth/gemma-3-4b-it', GEMMA_3_4B, 34)).toEqual({
      core: { kind: 'gqa', kvHeads: 4, headDim: 256 },
      projectionWidth: 8 * 256,
    });
  });

  it('reads Granite 4.1 as the plain GQA stack it is, head dim implied', () => {
    // The Granite name is on both sides of the linear-stack refusal: 4.0-h-small is 4 attending
    // layers of 40, and 4.1-8b is 40 of 40 with no `mamba_*` key anywhere in its config.
    expect(deriveLayerWindows('ibm-granite/granite-4.1-8b', GRANITE_4_1_8B, 40)).toBeUndefined();
    expect(deriveAttention('ibm-granite/granite-4.1-8b', GRANITE_4_1_8B, 40)).toEqual({
      core: { kind: 'gqa', kvHeads: 8, headDim: 128 },
      projectionWidth: 32 * 128,
    });
  });

  it('reads Command A+ as full attention every fourth layer', () => {
    expect(
      deriveLayerWindows('CohereLabs/command-a-plus-05-2026-bf16', COMMAND_A_PLUS, 32)
    ).toEqual(Array.from({ length: 32 }, (_, i) => ((i + 1) % 4 === 0 ? null : 4096)));
    expect(
      deriveAttention('CohereLabs/command-a-plus-05-2026-bf16', COMMAND_A_PLUS, 32).core
    ).toEqual({ kind: 'gqa', kvHeads: 8, headDim: 128 });
  });

  it('reads MLA that is not DeepSeek-shaped, and averages its two widths', () => {
    const attention = deriveAttention('mistralai/Mistral-Small-4-119B-2603', MISTRAL_SMALL_4, 36);
    expect(attention.core).toEqual({ kind: 'mla', kvLoraRank: 256, qkRopeHeadDim: 64 });
    // Query space 32 x (64 + 64) against value space 32 x 128 — equal here, which is the case that
    // proves the average is being taken over the right two quantities rather than over hidden size.
    expect(attention.projectionWidth).toBe(4096);
    expect(attention.projectionWidth / MISTRAL_SMALL_4.hidden_size).toBe(1);
  });
});

/**
 * A per-block architecture, which is the fourth way a stack turns out not to be uniform — and the
 * one an issue asking for wider coverage named as an ordinary seed.
 *
 * NVIDIA's Puzzle pipeline searches over per-layer variants and writes the result to
 * `block_configs`, so the top-level fields describe a block rather than the stack. Two errors
 * compound: blocks with no attention are charged a cache, and `num_key_value_heads: null` — a config
 * declining to answer, because the answer is per block — reads as full multi-head attention.
 */
describe('per-block NAS stacks are refused, not read off the top-level fields', () => {
  it('refuses the 13.1x it would otherwise hand the engine for Nemotron Super', () => {
    // What the row would have carried, read out of the GQA branch rather than written down: the same
    // config with `block_configs` and the null KV heads taken out, which is a clean hit with nothing
    // in it that looks wrong.
    const asPlainGqa = {
      num_hidden_layers: NEMOTRON_SUPER.num_hidden_layers,
      num_attention_heads: NEMOTRON_SUPER.num_attention_heads,
      hidden_size: NEMOTRON_SUPER.hidden_size,
    };
    const core = gqaOf(
      deriveAttention('nvidia/Llama-3_3-Nemotron-Super-49B-v1_5', asPlainGqa, 80).core
    );
    // No `num_key_value_heads` at all means one KV head per query head, and no `head_dim` means
    // hidden / heads. Both come from product code, so this arithmetic cannot outlive the derivation.
    expect(core).toEqual({ kind: 'gqa', kvHeads: 64, headDim: 128 });

    const flattened = gqaKvBytesPerToken(80, core.kvHeads, core.headDim);
    // What the 49 attending blocks actually cache, at the grouping their own entries state.
    const actual = gqaKvBytesPerToken(
      NEMOTRON_SUPER_ATTENDS.filter((a) => a === 1).length,
      core.kvHeads / 8,
      core.headDim
    );
    expect(flattened / KIB).toBe(2560);
    expect(actual / KIB).toBe(196);
    expect(flattened / actual).toBeCloseTo(13.06, 2);
    // 320 GiB of imaginary cache at 128K context, on a machine that needs 24.5.
    expect((flattened * 131072) / GIB).toBe(320);
    expect((actual * 131072) / GIB).toBeCloseTo(24.5, 1);

    expect(() =>
      deriveAttention('nvidia/Llama-3_3-Nemotron-Super-49B-v1_5', NEMOTRON_SUPER, 80)
    ).toThrowError(/declares block_configs for 80 blocks against num_hidden_layers 80/);
  });

  it('counts the blocks that attend, so the refusal is evidence rather than a shrug', () => {
    expect(() =>
      deriveAttention('nvidia/Llama-3_3-Nemotron-Super-49B-v1_5', NEMOTRON_SUPER, 80)
    ).toThrowError(/49 of 80 blocks carry attention; the other 31 declare none/);
  });

  it('reads the other spelling of the same claim, which shares only the key', () => {
    // Nemotron-Labs-3-Puzzle states `{block_type: "mamba" | "moe" | "attention"}` — no `attention`
    // object, no `no_op`, nothing a guard written against the export above would match. In the real
    // config the linear-stack guard fires first on its `mamba_*` keys; this is the same shape without
    // them, which is what the next NAS export with no recurrence in it will look like.
    const puzzle = {
      num_attention_heads: 32,
      num_key_value_heads: 2,
      head_dim: 128,
      hidden_size: 4096,
      block_configs: Array.from({ length: 88 }, (_, i) =>
        i % 4 === 0 ? { block_type: 'attention' } : { block_type: 'moe' }
      ),
    };
    expect(() =>
      deriveAttention('nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16', puzzle, 88)
    ).toThrowError(/22 of 88 blocks carry attention; the other 66 declare none/);
  });

  it('refuses an explicit null KV head count on its own, wherever it appears', () => {
    // The more general half of the same defect, and the reason it is guarded twice: `?? heads` cannot
    // tell "unstated, so full multi-head" from "stated as null, because it is per layer", and only
    // the first is a fact about the model. The next export to decline the question may carry no
    // `block_configs` for the guard above to catch.
    const nullKvHeads = { num_attention_heads: 64, num_key_value_heads: null, hidden_size: 8192 };
    expect(() => deriveAttention('hypothetical/null-kv-heads', nullKvHeads, 80)).toThrowError(
      /states num_key_value_heads: null rather than omitting it/
    );

    // And an *absent* one still means full multi-head attention, which is what Llama 2-era configs
    // leave unsaid — the distinction would be worthless if it rejected both.
    expect(
      deriveAttention('hypothetical/no-kv-heads', { num_attention_heads: 8, hidden_size: 1024 }, 4)
        .core
    ).toEqual({ kind: 'gqa', kvHeads: 8, headDim: 128 });
  });
});

/**
 * The sparse-attention refusal, moved out of the MLA branch — and DeepSeek V4, the model that walked
 * past it while it was in there.
 */
describe('a sparse-attention indexer is refused whichever branch the model would take', () => {
  it('refuses DeepSeek V4, whose config never reaches the MLA branch', () => {
    // The row it would have carried. Both figures come out of product code: one KV head at 512 is a
    // latent in everything but name, and `sliding_window: 128` with no `layer_types` is read as a
    // trailing 128-token window on every layer — which prices a million-token context at 11 MB of
    // cache, constant in sequence length, on the model whose headline is that context.
    const asPlainGqa = {
      num_hidden_layers: DEEPSEEK_V4_FLASH.num_hidden_layers,
      num_attention_heads: DEEPSEEK_V4_FLASH.num_attention_heads,
      num_key_value_heads: DEEPSEEK_V4_FLASH.num_key_value_heads,
      head_dim: DEEPSEEK_V4_FLASH.head_dim,
      hidden_size: DEEPSEEK_V4_FLASH.hidden_size,
      sliding_window: DEEPSEEK_V4_FLASH.sliding_window,
    };
    const core = gqaOf(deriveAttention('deepseek-ai/DeepSeek-V4-Flash', asPlainGqa, 43).core);
    expect(core).toEqual({ kind: 'gqa', kvHeads: 1, headDim: 512 });
    expect(deriveLayerWindows('deepseek-ai/DeepSeek-V4-Flash', asPlainGqa, 43)).toEqual(
      Array.from({ length: 43 }, () => 128)
    );

    const perToken = gqaKvBytesPerToken(43, core.kvHeads, core.headDim);
    expect(perToken / KIB).toBe(86);
    // Every layer windowed at 128 tokens: the whole cache, at any context length.
    expect((perToken * 128) / (1024 * 1024)).toBeCloseTo(10.75, 2);

    expect(() =>
      deriveAttention('deepseek-ai/DeepSeek-V4-Flash', DEEPSEEK_V4_FLASH, 43)
    ).toThrowError(/index_n_heads, index_head_dim, index_topk/);
  });

  it('still refuses the MLA-shaped one, so the move did not trade one branch for the other', () => {
    const v32 = { ...DEEPSEEK_V3, index_n_heads: 64, index_head_dim: 128, index_topk: 2048 };
    expect(() => deriveAttention('deepseek-ai/DeepSeek-V3.2-Exp', v32, 61)).toThrowError(
      /index_n_heads, index_head_dim, index_topk/
    );
  });

  it('reads the nested spelling too, which no flat key lookup sees', () => {
    // MiniMax M3 puts its whole sparse-attention block inside `sparse_attention_config` — the same
    // shape that hid Kimi-Linear's linear block from a flat lookup, one quantity over.
    const m3 = {
      num_attention_heads: 64,
      num_key_value_heads: 4,
      head_dim: 128,
      hidden_size: 6144,
      sparse_attention_config: { top_k: 2048 },
    };
    expect(() => deriveAttention('MiniMaxAI/MiniMax-M3', m3, 60)).toThrowError(
      /declares sparse_attention_config/
    );
  });
});

/**
 * Gemma 4 — a stack where `2 * kvHeads * headDim` is the wrong term three different ways, on a config
 * that passes every other guard in this file.
 *
 * This is the refusal with the largest audience in the catalog's own charts, and the one that most
 * looks like an ordinary row: the 31B, the 26B-A4B MoE and the 12B all state `layer_types` inside the
 * closed vocabulary with a `sliding_window` beside it, exactly as Gemma 3 does.
 */
describe('a cache that is not two tensors of one shape per layer', () => {
  it('refuses the 2x that shared keys and values would have cost the 31B', () => {
    const asPlainGqa = {
      num_hidden_layers: GEMMA_4_31B.num_hidden_layers,
      num_attention_heads: GEMMA_4_31B.num_attention_heads,
      num_key_value_heads: GEMMA_4_31B.num_key_value_heads,
      head_dim: GEMMA_4_31B.head_dim,
      hidden_size: GEMMA_4_31B.hidden_size,
      sliding_window: GEMMA_4_31B.sliding_window,
      layer_types: GEMMA_4_31B.layer_types,
    };
    const core = gqaOf(deriveAttention('google/gemma-4-31B-it', asPlainGqa, 60).core);
    expect(core).toEqual({ kind: 'gqa', kvHeads: 16, headDim: 256 });

    // The `2 *` is keys and values. `attention_k_eq_v` says there is one tensor, so every layer of
    // this stack is charged exactly twice what it holds.
    const charged = gqaKvBytesPerToken(60, core.kvHeads, core.headDim);
    const shared = charged / 2;
    expect(charged / KIB).toBe(960);
    expect(shared / KIB).toBe(480);

    expect(() => deriveAttention('google/gemma-4-31B-it', GEMMA_4_31B, 60)).toThrowError(
      /declares attention_k_eq_v/
    );
    expect(() => deriveAttention('google/gemma-4-31B-it', GEMMA_4_31B, 60)).toThrowError(
      /exactly 2x over for every layer of this stack/
    );
  });

  it('refuses the second KV shape the global layers carry', () => {
    // With `attention_k_eq_v` taken out, the 31B still cannot be priced: its 10 full-attention layers
    // cache 4 heads x 512 where its 50 windowed ones cache 16 x 256, and `AttentionCore` holds one
    // shape. The products agree here by luck — the 12B's are 1 x 512 against 8 x 256, which do not —
    // so the guard is on the keys being stated, not on the arithmetic coming out uneven.
    const withoutKEqV = { ...GEMMA_4_31B, attention_k_eq_v: false };
    expect(() => deriveAttention('google/gemma-4-31B-it', withoutKEqV, 60)).toThrowError(
      /declares num_global_key_value_heads — the full-attention layers cache 4 x 512 where the windowed ones cache 16 x 256/
    );
  });

  it('refuses the 1.75x on E4B, whose layers share a cache instead of keeping one', () => {
    // The same family, the other axis: `attention_k_eq_v` is false here and 18 of 42 layers reuse an
    // earlier layer's cache, so 24 layers' worth is charged as 42.
    expect(() => deriveAttention('google/gemma-4-E4B-it', GEMMA_4_E4B, 42)).toThrowError(
      /declares num_kv_shared_layers 18 of 42/
    );
    expect(() => deriveAttention('google/gemma-4-E4B-it', GEMMA_4_E4B, 42)).toThrowError(
      /1\.75x over/
    );
  });

  it('does not print a ratio for a config that shares every layer', () => {
    // A malformed config, and the shape of failure this file keeps producing: a predicate and its
    // sentence are one claim, so `42 / 0` must not be printed as `Infinityx over` beside a count
    // saying no layer keeps a cache.
    const allShared = { ...GEMMA_4_E4B, num_kv_shared_layers: 42 };
    expect(() => deriveAttention('hypothetical/all-shared', allShared, 42)).toThrowError(
      /num_kv_shared_layers 42 of 42/
    );
    expect(() => deriveAttention('hypothetical/all-shared', allShared, 42)).not.toThrowError(
      /Infinity/
    );
  });

  it('leaves a stack alone when it states none of the three', () => {
    // The guard has to be satisfied by silence, or it takes the whole catalog with it: `head_dim`
    // differing from hidden / heads is ordinary and true of half these fixtures, and a
    // `sliding_window` beside `layer_types` is Gemma 3, gpt-oss and Command A+.
    expect(() => deriveAttention('openai/gpt-oss-20b', GPT_OSS_20B, 24)).not.toThrow();
    expect(() => deriveAttention('unsloth/gemma-3-12b-it', GEMMA_3_12B, 48)).not.toThrow();
    expect(() =>
      deriveAttention('CohereLabs/command-a-plus-05-2026-bf16', COMMAND_A_PLUS, 32)
    ).not.toThrow();
    // A zero is not a claim: Gemma 4's own 31B states `num_kv_shared_layers: 0`, and refusing on the
    // key's presence would have refused for a value that says nothing is shared.
    const noSharing = { ...GRANITE_4_1_8B, num_kv_shared_layers: 0 };
    expect(() => deriveAttention('hypothetical/no-sharing', noSharing, 40)).not.toThrow();
  });
});

/**
 * The defect: a layer stack that mixes full attention with linear or state-space layers falls
 * through to the GQA branch and is catalogued as if **every** layer cached keys and values.
 *
 * These read as a clean GQA hit — which is the whole problem, and why the assertion that matters in
 * each test below is the refusal rather than the arithmetic beside it. The arithmetic is there to
 * say how large the number being refused was.
 */
describe('hybrid attention stacks are refused, not flattened into GQA', () => {
  it('refuses the 4.0x it would otherwise hand the engine for Qwen3-Next', () => {
    // The row the catalog would have carried, read out of the GQA branch rather than written down:
    // the same fixture with the hybrid keys removed, which is a clean GQA hit with nothing in it
    // that looks wrong. `kvHeads` and `headDim` then come from product code, so deleting the
    // derivation this test is about cannot leave the arithmetic below passing.
    const asPlainGqa = {
      num_hidden_layers: QWEN3_NEXT_80B.num_hidden_layers,
      num_attention_heads: QWEN3_NEXT_80B.num_attention_heads,
      num_key_value_heads: QWEN3_NEXT_80B.num_key_value_heads,
      head_dim: QWEN3_NEXT_80B.head_dim,
      hidden_size: QWEN3_NEXT_80B.hidden_size,
    };

    const core = gqaOf(deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', asPlainGqa, 48).core);
    expect(core).toEqual({ kind: 'gqa', kvHeads: 2, headDim: 256 });

    const flattened = gqaKvBytesPerToken(48, core.kvHeads, core.headDim);
    expect(flattened / KIB).toBe(96);

    // What 12 attention layers actually cache. The other 36 are gated DeltaNet, whose recurrent
    // state is constant in sequence length.
    const actual = gqaKvBytesPerToken(48 / 4, core.kvHeads, core.headDim);
    expect(actual / KIB).toBe(24);
    expect(flattened / actual).toBe(4);

    // At 128K context that is the difference between "buy another GPU" and "you're fine" — the
    // failure the README leads with, pointed the other way.
    expect((flattened * 131072) / GIB).toBe(12);
    expect((actual * 131072) / GIB).toBe(3);

    // So the full config is refused rather than emitted at 96 KiB/token. Matched on the hybrid
    // guard's own wording, not on "could not|declares|refus" — a generic pattern like that is also
    // satisfied by `require()`'s "could not determine <field> from config.json", so it would stay
    // green if the guard were removed and the fixture lost a field the GQA branch needs.
    expect(() =>
      deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', QWEN3_NEXT_80B, 48)
    ).toThrowError(/state is constant in sequence length/);
  });

  it('names the linear layers, so the refusal is evidence rather than a shrug', () => {
    // A `DerivationError` that says only "unsupported" leaves the next person to re-derive the
    // split from scratch. This one states it, names the key it read it from, and names every key
    // that made it refuse — sorted, so the sentence does not depend on JSON insertion order.
    expect(() =>
      deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', QWEN3_NEXT_80B, 48)
    ).toThrowError(
      /full_attention_interval states the split: 12 of 48 layers attend and cache; the other 36/
    );
    expect(() =>
      deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', QWEN3_NEXT_80B, 48)
    ).toThrowError(/full_attention_interval, linear_conv_kernel_dim, linear_key_head_dim/);
  });

  it('does not claim a split when full_attention_interval says there is none', () => {
    // An interval of 1 is legal and means every layer is full attention. The count clause used to
    // fire regardless, producing "48 of 48 layers attend and cache; the other 0 hold a recurrent
    // state" — one sentence whose two clauses contradict each other, which is the exact failure
    // ROADMAP records as "a predicate and its sentence are one claim". Still refused, because the
    // `linear_*` block is still declared; just no longer refused with a lie in it.
    const everyLayerAttends = { ...QWEN3_NEXT_80B, full_attention_interval: 1 };
    expect(() =>
      deriveAttention('hypothetical/qwen3-next-interval-1', everyLayerAttends, 48)
    ).toThrowError(/state is constant in sequence length/);
    expect(() =>
      deriveAttention('hypothetical/qwen3-next-interval-1', everyLayerAttends, 48)
    ).not.toThrowError(/48 of 48 layers attend/);
    expect(() =>
      deriveAttention('hypothetical/qwen3-next-interval-1', everyLayerAttends, 48)
    ).not.toThrowError(/states the split/);
  });

  it('refuses the 10x on Granite 4, along the axis that only saw sliding windows', () => {
    const flattened = gqaKvBytesPerToken(40, 8, 128);
    const actual = gqaKvBytesPerToken(4, 8, 128);
    expect(flattened / KIB).toBe(160);
    expect(actual / KIB).toBe(16);
    expect(flattened / actual).toBe(10);

    // `deriveLayerWindows` already refused a `layer_types` array it could not trust — but only
    // along the sliding axis, so an all-`mamba` array matched nothing, returned `undefined`, and
    // read as full attention on all 40 layers.
    expect(() =>
      deriveLayerWindows('ibm-granite/granite-4.0-h-small', GRANITE_4_H_SMALL, 40)
    ).toThrowError(/36 of 40 layers as "mamba"/);
  });

  it('refuses Granite on the other axis too, so neither guard is the only one', () => {
    // The two axes are independent and a model can present on either. Granite declares the Mamba-2
    // block *and* the array; Qwen3-Next declares only the block. A fix in one function would have
    // left the other family reachable.
    expect(() =>
      deriveAttention('ibm-granite/granite-4.0-h-small', GRANITE_4_H_SMALL, 40)
    ).toThrowError(/mamba_d_state/);
  });

  it("refuses Nemotron-H's third spelling of the same split", () => {
    // A per-layer string rather than an array or an interval: `M` for Mamba-2, `*` for attention,
    // `-` for an FFN. Nothing in this script reads it, so without a guard the stack is invisible.
    expect(() =>
      deriveAttention(
        'nvidia/Nemotron-H-47B-Base-8K',
        {
          num_attention_heads: 40,
          num_key_value_heads: 8,
          hidden_size: 8192,
          head_dim: 128,
          hybrid_override_pattern: 'M-M-M*-',
        },
        4
      )
    ).toThrowError(/hybrid_override_pattern/);
  });

  it("refuses Nemotron-Nano on its mamba_* keys alone, in this export's own spelling", () => {
    // The test for matching prefixes rather than exact names. Take away the one key the first draft
    // caught this config by and it must still refuse — on `mamba_state_dim` / `mamba_head_dim` /
    // `mamba_num_heads`, a spelling that shares not one exact name with Granite's `mamba_d_state` /
    // `mamba_d_head` / `mamba_n_heads`. An enumerated list is a list of the configs its author
    // happened to open.
    const withoutThePattern = { ...NEMOTRON_NANO_9B, hybrid_override_pattern: undefined };
    expect(() =>
      deriveAttention('nvidia/NVIDIA-Nemotron-Nano-9B-v2', withoutThePattern, 56)
    ).toThrowError(/mamba_head_dim, mamba_hidden_act, mamba_num_groups/);
  });

  it("refuses Kimi-Linear, whose linear block is nested where a flat lookup can't see it", () => {
    // The MLA branch rather than the GQA one, and the reason a flat list of key names was not the
    // fix: everything about the Kimi-Delta block is inside one `linear_attn_config` object, so
    // `config['linear_num_key_heads']`-style lookups all miss and `kv_lora_rank` reads as a clean
    // MLA hit. Kimi's headline claim is a 75%-smaller KV cache; the row said the opposite.
    const asPlainMla = {
      num_hidden_layers: KIMI_LINEAR_48B.num_hidden_layers,
      num_attention_heads: KIMI_LINEAR_48B.num_attention_heads,
      hidden_size: KIMI_LINEAR_48B.hidden_size,
      kv_lora_rank: KIMI_LINEAR_48B.kv_lora_rank,
      qk_rope_head_dim: KIMI_LINEAR_48B.qk_rope_head_dim,
      qk_nope_head_dim: KIMI_LINEAR_48B.qk_nope_head_dim,
      v_head_dim: KIMI_LINEAR_48B.v_head_dim,
    };
    const core = mlaOf(
      deriveAttention('moonshotai/Kimi-Linear-48B-A3B-Instruct', asPlainMla, 27).core
    );
    expect(core).toEqual({ kind: 'mla', kvLoraRank: 512, qkRopeHeadDim: 64 });

    const flattened = mlaKvBytesPerToken(27, core.kvLoraRank, core.qkRopeHeadDim);
    const actual = mlaKvBytesPerToken(7, core.kvLoraRank, core.qkRopeHeadDim);
    expect(flattened / KIB).toBe(30.375);
    expect(actual / KIB).toBe(7.875);
    expect(flattened / actual).toBeCloseTo(3.86, 2);
    expect((flattened * 131072) / GIB).toBeCloseTo(3.797, 3);
    expect((actual * 131072) / GIB).toBeCloseTo(0.984, 3);

    expect(() =>
      deriveAttention('moonshotai/Kimi-Linear-48B-A3B-Instruct', KIMI_LINEAR_48B, 27)
    ).toThrowError(/declares linear_attn_config/);
    expect(() =>
      deriveAttention('moonshotai/Kimi-Linear-48B-A3B-Instruct', KIMI_LINEAR_48B, 27)
    ).toThrowError(
      /linear_attn_config\.full_attn_layers states the split: 7 of 27 layers attend and cache; the other 20/
    );
  });

  it('refuses LFM2 on full_attn_idxs, the spelling with no layer_types beside it', () => {
    // LFM2's own GQA shape, derived from its own fields with the hybrid keys taken out.
    const asPlainGqa = {
      num_hidden_layers: LFM2_1_2B.num_hidden_layers,
      num_attention_heads: LFM2_1_2B.num_attention_heads,
      num_key_value_heads: LFM2_1_2B.num_key_value_heads,
      hidden_size: LFM2_1_2B.hidden_size,
    };
    const core = gqaOf(deriveAttention('LiquidAI/LFM2-1.2B', asPlainGqa, 16).core);
    expect(core).toEqual({ kind: 'gqa', kvHeads: 8, headDim: 64 });

    const flattened = gqaKvBytesPerToken(16, core.kvHeads, core.headDim);
    const actual = gqaKvBytesPerToken(6, core.kvHeads, core.headDim);
    expect(flattened / KIB).toBe(32);
    expect(actual / KIB).toBe(12);
    expect(flattened / actual).toBeCloseTo(2.67, 2);
    expect((flattened * 131072) / GIB).toBe(4);
    expect((actual * 131072) / GIB).toBe(1.5);

    expect(() => deriveAttention('LiquidAI/LFM2-1.2B', LFM2_1_2B, 16)).toThrowError(
      /full_attn_idxs states the split: 6 of 16 layers attend and cache; the other 10/
    );
  });

  it('refuses the other LFM2 export shape, which states no indices at all', () => {
    // The 2.6B and 8B-A1B exports of the same architecture put the split in `layer_types: ["conv",
    // ...]` and carry no `full_attn_idxs`. The window axis already refused those; this pins that the
    // attention axis does too, on `conv_L_cache` alone. One export of an architecture refusing while
    // another silently mis-prices *is* the bug, not a partial fix for it.
    const lfm2LayerTypesShape = {
      num_hidden_layers: 30,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      hidden_size: 2048,
      conv_L_cache: 3,
    };
    expect(() => deriveAttention('LiquidAI/LFM2-2.6B', lfm2LayerTypesShape, 30)).toThrowError(
      /declares conv_L_cache/
    );
  });

  it('refuses a hybrid that states no split at all', () => {
    // Phi-4-mini-flash: `mb_per_layer: 2` and nothing else. Where the Mamba blocks land is in the
    // modelling code, so there is no count to state — and the refusal must not depend on there
    // being one, or the models it cannot count become the models it admits.
    expect(() =>
      deriveAttention('microsoft/Phi-4-mini-flash-reasoning', PHI_4_MINI_FLASH, 32)
    ).toThrowError(/declares mb_per_layer/);
    expect(() =>
      deriveAttention('microsoft/Phi-4-mini-flash-reasoning', PHI_4_MINI_FLASH, 32)
    ).not.toThrowError(/states the split/);

    // Worth pinning what it derived instead, because it looked entirely healthy: a 32-layer GQA
    // stack with a 512-token window on every layer, which is a *narrower* answer than the truth on
    // the window axis and a wider one on the layer axis.
    expect(
      deriveLayerWindows('microsoft/Phi-4-mini-flash-reasoning', PHI_4_MINI_FLASH, 32)
    ).toEqual(Array.from({ length: 32 }, () => 512));
  });

  it('refuses a layer type nobody has taught it, whatever the type is called', () => {
    // The general form, and the reason the vocabulary is closed rather than a growing list of
    // things to exclude: the *next* family is the one this has to catch, and it will arrive under
    // a name written after this test.
    expect(() =>
      deriveLayerWindows(
        'some-org/some-model',
        {
          num_hidden_layers: 4,
          layer_types: ['full_attention', 'chunked_attention', 'full_attention', 'full_attention'],
        },
        4
      )
    ).toThrowError(/1 of 4 layers as "chunked_attention"/);
  });

  it("refuses Llama 4's chunked attention, which never reaches the layer_types vocabulary", () => {
    // Leaving `chunked_attention` out of `LAYER_TYPES` does not refuse Llama 4, because Scout and
    // Maverick ship no `layer_types` at all. A vocabulary only fires for configs that use the key it
    // is a vocabulary for; this one needs its own guard on `attention_chunk_size`.
    const core = gqaOf(
      deriveAttention('unsloth/Llama-4-Scout-17B-16E-Instruct', LLAMA_4_SCOUT_TEXT, 48).core
    );
    expect(core).toEqual({ kind: 'gqa', kvHeads: 8, headDim: 128 });

    const perLayerPerToken = gqaKvBytesPerToken(1, core.kvHeads, core.headDim);
    const global = LLAMA_4_SCOUT_TEXT.no_rope_layers.filter((v) => v === 0).length;
    const chunked = 48 - global;
    expect([global, chunked]).toEqual([12, 36]);

    // All 48 layers read as full attention, against the real 12-global / 36-chunked-at-8192 split.
    const flattened = 48 * perLayerPerToken * 131072;
    const actual = global * perLayerPerToken * 131072 + chunked * perLayerPerToken * 8192;
    expect((48 * perLayerPerToken) / KIB).toBe(192);
    expect(flattened / GIB).toBe(24);
    expect(actual / GIB).toBe(7.125);
    expect(flattened / actual).toBeCloseTo(3.37, 2);

    expect(() =>
      deriveLayerWindows('unsloth/Llama-4-Scout-17B-16E-Instruct', LLAMA_4_SCOUT_TEXT, 48)
    ).toThrowError(/declares attention_chunk_size 8192/);
  });

  it('refuses an array that disagrees with the layer count in either direction', () => {
    const short = { layer_types: ['full_attention', 'full_attention'] };
    expect(() => deriveLayerWindows('short/stack', short, 4)).toThrowError(
      /2 entries for 4 layers/
    );

    // The other direction was silently sliced, which is the same defect wearing the opposite sign:
    // `num_hidden_layers` and `layer_types` disagree and the script picked one without saying so.
    const long = { layer_types: Array.from({ length: 6 }, () => 'full_attention') };
    expect(() => deriveLayerWindows('long/stack', long, 4)).toThrowError(/6 entries for 4 layers/);
  });
});

/**
 * `attn_type_list` is the one per-layer convention in this sweep that has to be *read* rather than
 * refused on sight: MiniMax-M2's is all `1`, so M2 genuinely is full attention throughout, and M1's
 * hybrid lightning attention did not carry forward. A guard keyed on the presence of the key would
 * have rejected the model that turned out not to be a hybrid.
 */
describe('MiniMax — the per-layer array that mostly says full attention', () => {
  const M2 = {
    num_attention_heads: 48,
    num_key_value_heads: 8,
    head_dim: 128,
    hidden_size: 3072,
    attn_type_list: Array.from({ length: 62 }, () => 1),
  };

  it('admits MiniMax-M2, whose list is all ones', () => {
    expect(deriveAttention('MiniMaxAI/MiniMax-M2', M2, 62).core).toEqual({
      kind: 'gqa',
      kvHeads: 8,
      headDim: 128,
    });
  });

  it('refuses the same model with lightning-attention layers in the list', () => {
    // M1's shape: one full-attention layer every eight, lightning attention on the rest — 7 of 62
    // here, so 55 layers that hold a fixed-size state and would have been charged a growing cache.
    const m1Shaped = {
      ...M2,
      attn_type_list: Array.from({ length: 62 }, (_, i) => (i % 8 === 7 ? 1 : 0)),
    };
    expect(() => deriveAttention('MiniMaxAI/MiniMax-M1', m1Shaped, 62)).toThrowError(
      /marks 55 of 62 layers as 0/
    );
  });

  it('refuses a list that does not cover the stack', () => {
    const truncated = { ...M2, attn_type_list: [1, 1, 1] };
    expect(() => deriveAttention('MiniMaxAI/MiniMax-M2', truncated, 62)).toThrowError(
      /3 entries for 62 layers/
    );
  });
});

/**
 * The same axis, a different quantity. DeepSeek V3.2-Exp derives its *capacity* correctly through
 * the existing MLA path — what is wrong is that its lightning indexer keeps a cache of its own that
 * nothing here counts, and that its main attention reads at most `index_topk` selected positions
 * rather than everything before it, so prefill charges a quadratic the model does not compute.
 *
 * Deriving both is separate work. Emitting a row that is right about the latent and silently short
 * by the indexer is not a smaller version of that work.
 */
describe('MLA with a sparse-attention indexer', () => {
  it('refuses DeepSeek V3.2-Exp rather than pricing half of it', () => {
    const v32 = { ...DEEPSEEK_V3, index_n_heads: 64, index_head_dim: 128, index_topk: 2048 };
    expect(() => deriveAttention('deepseek-ai/DeepSeek-V3.2-Exp', v32, 61)).toThrowError(
      /index_n_heads, index_head_dim, index_topk/
    );
  });

  it('leaves plain MLA alone, so the guard is about the indexer and not about MLA', () => {
    expect(deriveAttention('deepseek-ai/DeepSeek-V3', DEEPSEEK_V3, 61).core.kind).toBe('mla');
  });
});

/**
 * The expert count, and the one value of it that produced a number rather than a refusal.
 *
 * `deriveMoe` throws on a partial config and returns null for a dense one, which covers every shape
 * except the one a *shared* config class produces: a dense variant that states the MoE keys and zeroes
 * them. Both keys are present, so the partial guard is satisfied, and `(perToken / total) * expertParams`
 * is `(0 / 0) * 0` — NaN, which `JSON.stringify` writes into the committed catalog as `null` and
 * `toModel` does not check, since it validates `activeDenseParams` and the projection width.
 */
describe('the expert count', () => {
  /**
   * https://huggingface.co/ibm-granite/granite-4.0-micro/raw/main/config.json
   *
   * Trimmed to the keys `deriveMoe` reads. The full config also carries the `mamba_*` block of the
   * hybrid variants it shares a config class with — all 40 of its `layer_types` are `attention`, but
   * `deriveAttention` refuses it a step earlier on those keys, so this fixture is about the arithmetic
   * rather than about that row being seedable.
   */
  const GRANITE_4_MICRO = {
    num_hidden_layers: 40,
    hidden_size: 2560,
    intermediate_size: 8192,
    num_local_experts: 0,
    num_experts_per_tok: 0,
  };

  it('reads zeroed expert keys as the dense model they describe', () => {
    expect(deriveMoe('ibm-granite/granite-4.0-micro', GRANITE_4_MICRO, 40)).toBeNull();
  });

  it('would have emitted a NaN active count for it', () => {
    // The consequence, computed the way `buildModel` computes it, from a shape that says what the old
    // branch returned. Not arithmetic on literals: the point is that this expression is what the row
    // is built from, and that `0 / 0` reaches it.
    const asPartialMoe = { expertParams: 0, experts: { total: 0, perToken: 0 } };
    const activeParams =
      3.4e9 +
      (asPartialMoe.experts.perToken / asPartialMoe.experts.total) * asPartialMoe.expertParams;
    expect(Number.isNaN(activeParams)).toBe(true);
    expect(JSON.parse(JSON.stringify({ activeParams }))).toEqual({ activeParams: null });
  });

  it('still refuses a config that zeroes one key and not the other', () => {
    // Zero experts routed per token out of 128 is not a dense model, and 128 available with none
    // routed is not one either. Only both-zero is a statement; either alone is a contradiction.
    expect(() =>
      deriveMoe('hypothetical/none-routed', { ...GRANITE_4_MICRO, num_local_experts: 128 }, 40)
    ).toThrowError(/partial MoE config/);
    expect(() =>
      deriveMoe('hypothetical/none-available', { ...GRANITE_4_MICRO, num_experts_per_tok: 8 }, 40)
    ).toThrowError(/partial MoE config/);
  });

  it('still derives an ordinary MoE, and still refuses a half-stated one', () => {
    const qwen3Moe = {
      num_hidden_layers: 48,
      hidden_size: 2048,
      moe_intermediate_size: 768,
      num_experts: 128,
      num_experts_per_tok: 8,
      decoder_sparse_step: 1,
    };
    expect(deriveMoe('Qwen/Qwen3-30B-A3B', qwen3Moe, 48)).toEqual({
      expertParams: 48 * 128 * 3 * 2048 * 768,
      experts: { total: 128, perToken: 8 },
    });

    const halfStated = { ...qwen3Moe, num_experts_per_tok: undefined };
    expect(() => deriveMoe('hypothetical/half-stated', halfStated, 48)).toThrowError(
      /partial MoE config/
    );
  });
});

/**
 * The seed list as data, and the report that exists because nothing else notices absence.
 *
 * `.github/workflows/catalog-refresh.yml` re-derives every figure on every row weekly, so a publisher
 * editing a `config.json` is caught within days. A model that was never listed is invisible to it —
 * which is how this list came to be a year behind the field with every number in it seven days old.
 */
describe('the seed list knows what it is not carrying', () => {
  it('never lists a repo as both seeded and deliberately absent', () => {
    // The two halves of one decision, and a repo in both is a contradiction that would also make the
    // candidate report lie in the quiet direction: suppressed as "written down", present as a row.
    const seeded = seededIds();
    for (const id of Object.keys(NOT_SEEDED)) {
      expect(seeded.has(id), `${id} is both seeded and in NOT_SEEDED`).toBe(false);
    }
  });

  it('states a reason for every absence, since the list is the written record', () => {
    for (const [id, reason] of Object.entries(NOT_SEEDED)) {
      expect(reason.length, `${id} has no reason`).toBeGreaterThan(10);
    }
  });

  it('seeds each repo once, and names each row once', () => {
    // A repeated id is silent in the product and nearly silent in the artifact: `MODELS` keeps both
    // rows, so the picker lists the model twice while `getModel` resolves it to whichever came last.
    expect(new Set(SEEDS.map((s) => s.id)).size).toBe(SEEDS.length);
    expect(new Set(SEEDS.map((s) => s.name)).size).toBe(SEEDS.length);
  });

  it('counts a mirror and the repo it borrows traffic from as the same model', () => {
    // Gemma and Llama are seeded through open mirrors, and the listing reports the *canonical* repo's
    // traffic — so a report keyed on seed ids alone would name `google/gemma-3-4b-it` every week as
    // something to add, while `unsloth/gemma-3-4b-it` is already a row under exactly that figure.
    const mirrored = SEEDS.filter((s) => s.popularityId);
    expect(mirrored.length).toBeGreaterThan(3);
    for (const seed of mirrored) {
      expect(seededIds().has(seed.popularityId!)).toBe(true);
    }
  });

  /**
   * The report's own filters, on rows taken from the live listing this runs against.
   *
   * Both wrong answers cost something specific. Naming forty derivative re-uploads trains people to
   * skip the report, which is the same as not having one; suppressing a real model means the list ages
   * silently, which is the defect this whole mechanism is for.
   */
  describe('the candidate report', () => {
    const since = new Date('2025-06-01T00:00:00Z');
    const options = {
      seeded: seededIds(),
      notSeeded: new Set(Object.keys(NOT_SEEDED)),
      minDownloads: 250_000,
      since,
    };

    it('names a model the field is downloading that nothing has decided about', () => {
      const live = [
        { id: 'someorg/Brand-New-42B-Instruct', downloads: 900_000, createdAt: '2026-07-01' },
      ];
      expect(unseededCandidates({ ...options, live }).map((m) => m.id)).toEqual([
        'someorg/Brand-New-42B-Instruct',
      ]);
    });

    it('says nothing about a row that is already seeded, mirror or canonical', () => {
      const live = [
        { id: 'Qwen/Qwen3-Coder-480B-A35B-Instruct', downloads: 900_000, createdAt: '2026-01-01' },
        { id: 'google/gemma-3-4b-it', downloads: 2_060_000, createdAt: '2025-07-01' },
        { id: 'unsloth/gemma-3-4b-it', downloads: 120_000, createdAt: '2025-07-01' },
      ];
      expect(unseededCandidates({ ...options, live })).toEqual([]);
    });

    it('says nothing about a refusal that has been written down', () => {
      // The nine refused families are the bulk of the current field's traffic. Reporting them weekly
      // would be a report whose every line is already answered in `NOT_SEEDED`.
      const live = [
        { id: 'Qwen/Qwen3.6-27B', downloads: 6_200_000, createdAt: '2026-04-21' },
        { id: 'google/gemma-4-31B-it', downloads: 12_400_000, createdAt: '2026-03-11' },
        { id: 'deepseek-ai/DeepSeek-V4-Flash', downloads: 3_100_000, createdAt: '2026-04-22' },
      ];
      expect(unseededCandidates({ ...options, live })).toEqual([]);
    });

    it('drops the derivative re-uploads that dominate the charts', () => {
      // Every one of these is an id from the live listing, and every one of them is the same
      // architecture as something already seeded or already refused.
      const live = [
        {
          id: 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF',
          downloads: 1_680_000,
          createdAt: '2026-01-01',
        },
        { id: 'Qwen/Qwen3-32B-AWQ', downloads: 2_000_000, createdAt: '2026-01-01' },
        { id: 'nvidia/GLM-5.2-NVFP4', downloads: 1_600_000, createdAt: '2026-06-22' },
        { id: 'Qwen/Qwen3-Coder-Next-FP8', downloads: 2_500_000, createdAt: '2026-02-01' },
        { id: 'google/gemma-4-31B-it-qat-w4a16-ct', downloads: 2_100_000, createdAt: '2026-06-04' },
        { id: 'Qwen/Qwen3.5-35B-A3B-Base', downloads: 800_000, createdAt: '2026-02-24' },
        {
          id: 'hmellor/tiny-random-LlamaForCausalLM',
          downloads: 5_500_000,
          createdAt: '2025-08-01',
        },
        {
          id: 'mistralai/Mistral-Medium-3.5-128B-EAGLE',
          downloads: 400_000,
          createdAt: '2026-04-27',
        },
        { id: 'nvidia/Kimi-K2.7-Code-DFlash', downloads: 500_000, createdAt: '2026-07-08' },
        {
          id: 'Bahushruth/Qwen3.6-35B-A3B-abliterated-v4',
          downloads: 630_000,
          createdAt: '2026-06-11',
        },
      ];
      expect(unseededCandidates({ ...options, live })).toEqual([]);
    });

    it('drops the decade of accumulated tutorial traffic', () => {
      // gpt2 and opt-125m outrank most of the current field and are not candidates for a hardware
      // calculator. The floor is a date rather than a size, because size needs a second fetch.
      const live = [
        { id: 'openai-community/gpt2', downloads: 13_900_000, createdAt: '2022-03-02' },
        { id: 'facebook/opt-125m', downloads: 17_100_000, createdAt: '2022-05-11' },
      ];
      expect(unseededCandidates({ ...options, live })).toEqual([]);
    });

    it('reports a row whose metadata is thin rather than swallowing it', () => {
      // The report exists for the thing nobody has looked at, so silence about a repo because its
      // `createdAt` was missing is the wrong default.
      const live = [{ id: 'someorg/Undated-30B', downloads: 800_000 }];
      expect(unseededCandidates({ ...options, live }).map((m) => m.id)).toEqual([
        'someorg/Undated-30B',
      ]);
    });

    it('stays quiet in a quiet week instead of scraping the barrel', () => {
      // A floor rather than a top-N: nothing to report is a legitimate answer, and a report that
      // always has five lines is one nobody can distinguish from a report that has news.
      const live = [{ id: 'someorg/Obscure-7B', downloads: 1_200, createdAt: '2026-07-01' }];
      expect(unseededCandidates({ ...options, live })).toEqual([]);
    });

    it('ranks by downloads, since the first line is the one that gets read', () => {
      const live = [
        { id: 'someorg/Quiet-9B', downloads: 300_000, createdAt: '2026-07-01' },
        { id: 'someorg/Loud-70B', downloads: 4_000_000, createdAt: '2026-07-01' },
      ];
      expect(unseededCandidates({ ...options, live }).map((m) => m.id)).toEqual([
        'someorg/Loud-70B',
        'someorg/Quiet-9B',
      ]);
    });
  });
});
