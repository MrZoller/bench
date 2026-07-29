import { describe, expect, it } from 'vitest';
import { deriveAttention, deriveLayerWindows } from './build-catalog';

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
