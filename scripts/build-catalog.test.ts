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

/** https://huggingface.co/unsloth/gemma-3-12b-it/raw/main/config.json — pattern, not an array. */
const GEMMA_3_12B = {
  num_hidden_layers: 48,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  head_dim: 256,
  hidden_size: 3840,
  sliding_window: 1024,
  sliding_window_pattern: 6,
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

  it("reads Gemma 3's pattern as sliding except every sixth layer", () => {
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
    // What the GQA branch emits, applied to all 48 layers, from the config's own fields.
    const flattened = gqaKvBytesPerToken(48, 2, 256);
    expect(flattened / KIB).toBe(96);

    // What 12 attention layers actually cache. The other 36 are gated DeltaNet, whose recurrent
    // state is constant in sequence length.
    const actual = gqaKvBytesPerToken(48 / 4, 2, 256);
    expect(actual / KIB).toBe(24);
    expect(flattened / actual).toBe(4);

    // At 128K context that is the difference between "buy another GPU" and "you're fine" — the
    // failure the README leads with, pointed the other way.
    expect((flattened * 131072) / GIB).toBe(12);
    expect((actual * 131072) / GIB).toBe(3);

    // So the row is refused rather than emitted at 96 KiB/token.
    expect(() =>
      deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', QWEN3_NEXT_80B, 48)
    ).toThrowError(/could not|declares|refus/i);
  });

  it('names the linear layers, so the refusal is evidence rather than a shrug', () => {
    // A `DerivationError` that says only "unsupported" leaves the next person to re-derive the
    // split from scratch. This one states it, and names the keys it read it from.
    expect(() =>
      deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', QWEN3_NEXT_80B, 48)
    ).toThrowError(/12 of 48 layers attend and cache; the other 36/);
    expect(() =>
      deriveAttention('Qwen/Qwen3-Next-80B-A3B-Instruct', QWEN3_NEXT_80B, 48)
    ).toThrowError(/full_attention_interval, linear_num_key_heads/);
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
