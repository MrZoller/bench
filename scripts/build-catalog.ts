import { appendFile } from 'node:fs/promises';
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

export interface Seed {
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
    /**
     * The vendor's published active-parameter count. **Checked, never applied.**
     *
     * A `totalParams` override is a published figure standing in for a derived one, and every
     * per-token figure on the row is then computed by subtracting an *exact* analytic expert count
     * from it. The residual is small — 19B of 355B for GLM 4.7, 17B of 671B for DeepSeek V3 — so
     * whatever the published total rounds away lands entirely in it, amplified by `total / dense`:
     * 18.6x and 39.3x respectively. That is not a hypothetical. GLM 4.7's published 355B sits 2.2B
     * above the sum of its own architecture and shipped a per-token basis 13.7% heavy.
     *
     * The notes on these rows used to assert the reconciliation in prose — "which also reproduces
     * the stated 12B active exactly" — which is a claim checked once by whoever typed it and never
     * again. Stating the figure makes the generator check it on every refresh, and the band is the
     * same 8% `src/data/catalog.test.ts` holds every other MoE row to.
     *
     * Omitted where the vendor publishes no figure to check against. A name is not a figure:
     * GLM 4.7 Flash is *called* 30B-A3B and its architecture puts 3.6B on the per-token path.
     */
    publishedActiveParams?: number;
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
 * embedding models and one-off GGUF re-uploads.
 *
 * **What earns a row.** Either the model answers a hardware question no other row answers — a size
 * class, an attention family, an active-parameter ratio — or it is the current head of a family the
 * catalog already carries, because a user who finds the older sibling and not the newer one
 * reasonably concludes the older one is current. A model that is shape-identical to a row already
 * here earns nothing: Devstral Small 2 is 24B dense at 40 layers of 5120, which is Mistral Small's
 * row with a different name on it, and the answer to "will it run" is the same pixel.
 *
 * **What the weekly refresh does and does not do.** `.github/workflows/catalog-refresh.yml` keeps
 * every figure on every row current, and cannot notice a model that was never listed — which is how
 * this list came to be a year behind the field while every number in it was seven days old. So the
 * generator now ends every run by asking Hugging Face what the field is downloading and printing
 * whatever is neither seeded nor written down in {@link NOT_SEEDED}. Absence is the failure mode
 * that no amount of refreshing figures reaches; the report is what makes it visible weekly instead
 * of whenever someone happens to look.
 *
 * Last re-probed against the live API on 2026-07-29, which is also when {@link NOT_SEEDED} was
 * written: twelve families were checked and nine of them refuse.
 */
export const SEEDS: Seed[] = [
  // --- Dense, small enough to run anywhere ---
  // The bottom of the range is most of the audience: an 8 GB card is the machine a "will it run"
  // question usually comes from, and for a long time Qwen3-4B was the only row it could select.
  {
    id: 'unsloth/Llama-3.2-3B-Instruct',
    name: 'Llama 3.2 3B Instruct',
    org: 'Meta',
    popularityId: 'meta-llama/Llama-3.2-3B-Instruct',
  },
  { id: 'mistralai/Ministral-3-3B-Instruct-2512', name: 'Ministral 3 3B', org: 'Mistral' },
  { id: 'Qwen/Qwen3-4B', name: 'Qwen3 4B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-4B-Instruct-2507', name: 'Qwen3 4B Instruct 2507', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-8B', name: 'Qwen3 8B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-14B', name: 'Qwen3 14B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-32B', name: 'Qwen3 32B', org: 'Alibaba' },
  { id: 'microsoft/phi-4', name: 'Phi-4', org: 'Microsoft' },
  { id: 'ibm-granite/granite-4.1-8b', name: 'Granite 4.1 8B', org: 'IBM' },
  { id: 'ByteDance-Seed/Seed-OSS-36B-Instruct', name: 'Seed-OSS 36B', org: 'ByteDance' },
  // Gemma is gated on google/*, so these point at open mirrors of the same weights.
  {
    id: 'unsloth/gemma-3-4b-it',
    name: 'Gemma 3 4B',
    org: 'Google',
    popularityId: 'google/gemma-3-4b-it',
  },
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
  // 3.3 is the newest Llama this script can price: Llama 4's chunked attention is refused in
  // `deriveLayerWindows`, and nothing has shipped from Meta since.
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
  {
    id: 'unsloth/Llama-3.3-70B-Instruct',
    name: 'Llama 3.3 70B Instruct',
    org: 'Meta',
    popularityId: 'meta-llama/Llama-3.3-70B-Instruct',
  },

  // --- MoE: the interesting cases for unified-memory hardware ---
  { id: 'openai/gpt-oss-20b', name: 'gpt-oss 20B', org: 'OpenAI' },
  { id: 'openai/gpt-oss-120b', name: 'gpt-oss 120B', org: 'OpenAI' },
  { id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3 30B-A3B', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-30B-A3B-Instruct-2507', name: 'Qwen3 30B-A3B Instruct 2507', org: 'Alibaba' },
  { id: 'Qwen/Qwen3-235B-A22B', name: 'Qwen3 235B-A22B', org: 'Alibaba' },
  {
    id: 'Qwen/Qwen3-235B-A22B-Instruct-2507',
    name: 'Qwen3 235B-A22B Instruct 2507',
    org: 'Alibaba',
  },
  // The largest open coding model, and the row people price a 512 GB Mac against. Nothing else in
  // the catalog sits between 235B and 671B.
  { id: 'Qwen/Qwen3-Coder-480B-A35B-Instruct', name: 'Qwen3 Coder 480B-A35B', org: 'Alibaba' },
  { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', name: 'Mixtral 8x7B', org: 'Mistral' },
  { id: 'MiniMaxAI/MiniMax-M2.7', name: 'MiniMax M2.7', org: 'MiniMax' },
  { id: 'CohereLabs/command-a-plus-05-2026-bf16', name: 'Command A+ (05-2026)', org: 'Cohere' },

  // --- MLA: the family the naive KV formula gets most wrong ---
  // Now at four scales rather than one, which is the point of the family: the 30B and 119B rows put
  // the compressed-latent cache on hardware someone owns, where the 671B and 1T ones only argue
  // about it. GLM 4.7 Flash at 30B-A3B caches 4.5 KiB/token against Qwen3-30B-A3B's 24.0.
  {
    id: 'zai-org/GLM-4.7-Flash',
    name: 'GLM 4.7 Flash',
    org: 'Z.ai',
    overrides: {
      totalParams: 30e9,
      // No `publishedActiveParams`: "A3B" is the model's name, not a measurement Z.ai states, and
      // the arithmetic below is what the row would be checked against.
      reason:
        "HF's safetensors index reports 31.2B including the MTP module, which inference does " +
        "not load; the card's 30B is the loaded figure and sits 0.2% above the sum of the " +
        'architecture itself. The "A3B" in the name is a round number rather than a stated ' +
        'count: 4 of 64 experts is 1.74B per token, and the part every token pays for anyway — ' +
        '47 layers of MLA, the shared expert, the output table — is another 1.90B, so this row ' +
        'derives 3.6B active.',
    },
  },
  { id: 'mistralai/Mistral-Small-4-119B-2603', name: 'Mistral Small 4 119B', org: 'Mistral' },
  {
    id: 'deepseek-ai/DeepSeek-V3',
    name: 'DeepSeek V3',
    org: 'DeepSeek',
    overrides: {
      totalParams: 671e9,
      publishedActiveParams: 37e9,
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
      publishedActiveParams: 37e9,
      reason: 'Same MTP module as DeepSeek V3; 671B is the published figure.',
    },
  },
  {
    id: 'deepseek-ai/DeepSeek-V3.1',
    name: 'DeepSeek V3.1',
    org: 'DeepSeek',
    overrides: {
      totalParams: 671e9,
      publishedActiveParams: 37e9,
      reason:
        'Same MTP module as DeepSeek V3, and the same published 671B / 37B active. V3.2 and V4 ' +
        'are refused for their sparse-attention indexer, so this is the newest DeepSeek this ' +
        'script can price.',
    },
  },
  // The top of the open-weight range. K2 is also where the gap between MLA and the naive formula is
  // largest: 61 layers of 576-wide latent against 61 layers of 64 KV heads.
  { id: 'moonshotai/Kimi-K2-Instruct', name: 'Kimi K2 Instruct', org: 'Moonshot AI' },

  // --- Glm4Moe: the 355B flagship class, and the Air variant people actually run ---
  {
    id: 'zai-org/GLM-4.5-Air',
    name: 'GLM 4.5 Air',
    org: 'Z.ai',
    overrides: {
      totalParams: 106e9,
      publishedActiveParams: 12e9,
      reason:
        "HF's safetensors index reports 110.5B including the MTP module. The published " +
        'figure is 106B, which also reproduces the stated 12B active exactly.',
    },
  },
  {
    id: 'zai-org/GLM-4.7',
    name: 'GLM 4.7',
    org: 'Z.ai',
    overrides: {
      /**
       * The measured non-MTP total, not the published 355B, and this is the one override in the
       * list that departs from a vendor's figure — so the arithmetic is here rather than asserted.
       *
       * The index reports 358.338B. Its MTP module is one `Glm4Moe` block plus its own copies of
       * the vocabulary tables, every term of which is in `config.json`: 160 experts x 3 x 5120 x
       * 1536 = 3.775B routed, a shared expert of 23.6M, one attention block of 136.3M (96 q heads
       * x 128 against 8 kv heads, with biases), an `eh_proj` at [5120, 10240] = 52.4M, and an
       * embedding and head of 151552 x 5120 = 775.9M each. 5.540B, leaving 352.798B — which is
       * also what summing the 92-layer stack term by term gives, to within 10 KiB of norms.
       *
       * Z.ai's 355B is 2.2B above that, which is 0.6% of the total and would be immaterial if the
       * total were where it stopped. It is not: `denseParams` is `totalParams - expertParams`, and
       * the routed experts are 335.964B of it, so the whole 2.2B lands in a 16.8B residual and
       * every per-token figure on the row inherits it. 19.036B dense against 16.834B, 13.1%; the
       * decode basis 13.7% heavy; 35.1B active against a stated 32B. Rounding a total is cheap and
       * rounding its residual is not.
       */
      totalParams: 352.8e9,
      publishedActiveParams: 32e9,
      reason:
        "HF's safetensors index reports 358.3B including a 5.5B MTP module that inference does " +
        "not load. Z.ai's own model table states 355B-A32B for 4.5, 4.6 and 4.7 alike, which are " +
        'one architecture at three checkpoints — but 355B is 2.2B above the sum of that ' +
        "architecture's own tensors, and since the routed experts are subtracted from the total " +
        'exactly, all 2.2B would land in the 16.8B that is left. 352.8B is the measured figure ' +
        'and reproduces the stated 32B active to 2.8%. GLM 5.x is refused for its ' +
        'sparse-attention indexer.',
    },
  },
];

/**
 * Models deliberately absent, and the reason each one is.
 *
 * This exists because "why isn't X in the catalog?" has an answer for every X worth asking about,
 * and until now that answer lived in a session transcript. Two jobs:
 *
 *   - It is the written record. Every entry was checked against the live `config.json` on the date
 *     in the {@link SEEDS} comment, and most of them refuse in a guard below — so a reader who
 *     wonders why the most-downloaded model in the world is missing gets the reason rather than an
 *     apparent oversight.
 *   - It suppresses the candidate report. Without it, {@link reportSeedCandidates} would name the
 *     same nine refused families every week, which is exactly how a weekly report stops being read.
 *
 * A repo listed here is a *decision*, not a permanent exclusion: the entries that say "refused"
 * become seedable the day the engine grows the term they need, and the report is what will say so —
 * nothing here is checked for still being true.
 */
export const NOT_SEEDED: Readonly<Record<string, string>> = {
  // Hybrid linear attention: refused in `refuseLinearStack`. This is now the *mainstream* of the
  // field rather than an exotic corner — the entire current Qwen generation, at every size.
  'Qwen/Qwen3.6-27B': 'hybrid: 16 of 64 layers attend (full_attention_interval 4)',
  'Qwen/Qwen3.6-35B-A3B': 'hybrid: 10 of 40 layers attend',
  'Qwen/Qwen3.5-9B': 'hybrid: 8 of 32 layers attend',
  'Qwen/Qwen3.5-4B': 'hybrid: 8 of 32 layers attend',
  'Qwen/Qwen3.5-2B': 'hybrid linear attention, same stack as Qwen3.5-9B',
  'Qwen/Qwen3.5-0.8B': 'hybrid linear attention, same stack as Qwen3.5-9B',
  'Qwen/Qwen3.5-122B-A10B': 'hybrid: 12 of 48 layers attend',
  'Qwen/Qwen3.5-397B-A17B': 'hybrid: 15 of 60 layers attend',
  'Qwen/Qwen3-Next-80B-A3B-Instruct': 'hybrid: 12 of 48 layers attend',
  'moonshotai/Kimi-K3': 'hybrid: linear_attn_config, Kimi-Delta on most layers',
  'moonshotai/Kimi-Linear-48B-A3B-Instruct': 'hybrid: 7 of 27 layers attend',
  'ibm-granite/granite-4.0-h-small': 'hybrid: 4 of 40 layers attend, 36 Mamba-2',
  'Qwen/Qwen3-Coder-Next': 'hybrid: Qwen3Next linear attention, 12 of 48 layers attend',
  'LiquidAI/LFM2.5-1.2B-Instruct': 'hybrid: 6 of 16 layers attend, 10 short-convolution',
  // NVIDIA has no seedable row at all, which is worth stating rather than leaving as a gap: every
  // current Nemotron is either Mamba-2 hybrid or a per-block NAS export, and the two guards that
  // refuse them are different guards.
  'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16': 'hybrid: Mamba-2 (NemotronH)',
  'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16': 'hybrid: Mamba-2 (NemotronH)',
  'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16': 'hybrid: Mamba-2 (NemotronH)',
  'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4': 'hybrid: Mamba-2 per hybrid_override_pattern',
  'nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16': 'per-block NAS architecture (block_configs)',
  'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5':
    'per-block NAS architecture: 31 of 80 blocks have no attention at all, and the KV grouping ' +
    'is per block (13x KV read as uniform MHA)',
  // Sparse attention: a second cache and a bounded read, refused in `deriveAttention`.
  'deepseek-ai/DeepSeek-V4-Pro': 'sparse-attention indexer (index_topk 1024)',
  'deepseek-ai/DeepSeek-V4-Flash': 'sparse-attention indexer (index_topk 512)',
  'deepseek-ai/DeepSeek-V3.2': 'sparse-attention indexer',
  'deepseek-ai/DeepSeek-V3.2-Exp': 'sparse-attention indexer',
  'zai-org/GLM-5.2': 'sparse-attention indexer (GlmMoeDsa)',
  'zai-org/GLM-5.1': 'sparse-attention indexer (GlmMoeDsa)',
  'zai-org/GLM-5': 'sparse-attention indexer (GlmMoeDsa)',
  'MiniMaxAI/MiniMax-M3': 'sparse_attention_config, and a per-layer moe_layer_freq array',
  // A cache this script has no term for, one key at a time.
  'google/gemma-4-31B-it': 'attention_k_eq_v, and global layers with their own KV shape',
  'google/gemma-4-26B-A4B-it': 'attention_k_eq_v, global KV shape, and top_k_experts',
  'google/gemma-4-12B-it': 'attention_k_eq_v, and global layers with their own KV shape',
  'google/gemma-4-E4B-it': 'num_kv_shared_layers 18 of 42, so 18 layers cache nothing of their own',
  'meta-llama/Llama-4-Scout-17B-16E-Instruct': 'chunked attention (attention_chunk_size 8192)',
  'meta-llama/Llama-4-Maverick-17B-128E-Instruct': 'chunked attention',
  // Native int4 exports whose safetensors total counts the group scales as parameters — one BF16
  // scale per 32 weights, which is +31.7B on a 1T model. Seeded through their FP8 sibling instead,
  // which is the same architecture with a derivable count.
  'moonshotai/Kimi-K2.6':
    'int4 export: safetensors total counts group scales (see Kimi-K2-Instruct)',
  'moonshotai/Kimi-K2.5': 'int4 export: safetensors total counts group scales',
  'moonshotai/Kimi-K2-Thinking': 'int4 export: safetensors total counts group scales',
  // Shape-identical to a row already in the catalog, so it answers no new question.
  'mistralai/Devstral-Small-2-24B-Instruct-2512': "24B dense at 40x5120 — Mistral Small 24B's row",
  'mistralai/Mistral-Small-3.2-24B-Instruct-2506': 'superseded by Mistral Small 4; same shape',
  /**
   * The three repos [#77](https://github.com/MrZoller/bench/issues/77) named by id and this list did
   * not answer, which is a different failure from the ones above: each was checked, each was
   * declined, and none of them was written down — so the entry a reader would look for was missing
   * for the same reason the seed list went a year stale. All three are also invisible to
   * {@link reportSeedCandidates}: Nemo predates its 18-month window and the other two are under its
   * 250K download floor, so the weekly report was never going to raise them either. Those filters
   * are a floor on what the report *surfaces*, not a bound on what this table has to explain.
   */
  'mistralai/Mistral-Nemo-Instruct-2407':
    '12B dense at 40x5120 — the 12-14B tier Gemma 3 12B and Qwen3 14B already answer, and a ' +
    'July 2024 model rather than the head of its family (Ministral 3 3B and Mistral Small 4 are). ' +
    "The gap it fills is in Mistral's lineup, not in the hardware question.",
  'mistralai/Devstral-Small-2507':
    "24B dense at 40x5120, which is Mistral Small 24B's row; superseded by Devstral Small 2",
  'CohereLabs/c4ai-command-a-03-2025':
    'superseded by Command A+ (05-2026), which is seeded; 111B dense against its 219B MoE',
  'zai-org/GLM-4.5': "one checkpoint of GLM 4.7's architecture",
  'zai-org/GLM-4.6': "one checkpoint of GLM 4.7's architecture",
  'MiniMaxAI/MiniMax-M2': "one checkpoint of MiniMax M2.7's architecture",
  'MiniMaxAI/MiniMax-M2.5': "one checkpoint of MiniMax M2.7's architecture",
  'moonshotai/Kimi-K2-Instruct-0905': "one checkpoint of Kimi K2's architecture",
  'deepseek-ai/DeepSeek-V3-0324': "one checkpoint of DeepSeek V3's architecture",
  'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 'a Qwen3-8B distill: 36 x 4096, which is Qwen3 8B',
  'Qwen/Qwen3-Coder-30B-A3B-Instruct': 'same shape as Qwen3-30B-A3B: 48 x 2048, 128 experts of 768',
  'HuggingFaceTB/SmolLM3-3B': '3B dense, the tier Llama 3.2 3B and Ministral 3 3B already answer',
  /**
   * The sub-2B tier, which is absent on purpose and not by oversight — Qwen3-0.6B is the
   * most-downloaded text-generation repo on the hub and would still be a row whose every cell says
   * the same thing. At 2 GB of BF16 weights it fits, comfortably, on every device this catalog
   * lists including the 8 GiB ones, so the placement question the product exists to answer has no
   * content. The interesting small models are the 3-4B ones, where an 8 GiB card at long context
   * starts to matter, and there are five of those.
   */
  'Qwen/Qwen3-0.6B': 'sub-2B: fits comfortably on every catalogued device, so every cell agrees',
  'Qwen/Qwen3-1.7B': 'sub-2B: fits comfortably on every catalogued device',
  'google/gemma-3-1b-it': 'sub-2B: fits comfortably on every catalogued device',
  'google/gemma-3-270m': 'sub-2B: fits comfortably on every catalogued device',
  'google/gemma-3-270m-it': 'sub-2B: fits comfortably on every catalogued device',
  'openbmb/MiniCPM5-1B': 'sub-2B: fits comfortably on every catalogued device',
};

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

/**
 * Refuses a key that is **present and unreadable** where this script has a fallback for it being
 * absent.
 *
 * `num()` maps anything that is not a number to `undefined`, so `x: null` and no `x` at all are one
 * value by the time any derivation sees them. That is right for most keys and wrong for a specific
 * few, and the line between them is what this guard is:
 *
 *   - **Absence means "this feature is not here."** `sliding_window`, `num_kv_shared_layers`,
 *     `index_topk`, `num_global_key_value_heads` — a config with no key and a config with an explicit
 *     `null` are making the same statement, and writing the `null` is common: `sliding_window: null`
 *     is on Qwen3 and Command A+, `rope_scaling: null` on GLM 4.7. Reading null as absent is correct
 *     there, and a guard would refuse rows that are already right.
 *   - **Absence means something substantive, and there is a fallback to match.** An absent
 *     `num_key_value_heads` means full multi-head attention; an absent `head_dim` means
 *     `hidden_size / num_attention_heads`; an absent `first_k_dense_replace` selects Qwen's MoE
 *     layer phase over DeepSeek's. Here a stated `null` is a config *declining* to answer — usually
 *     because the answer is per layer — and applying the fallback answers on its behalf.
 *
 * `num_key_value_heads: null` is the live case (`Llama-3_3-Nemotron-Super-49B-v1_5`, an 8x
 * overstatement) and it has its own refusal below, with its own arithmetic. This is the same
 * statement for the rest of that second group, because the same publisher writes
 * `intermediate_size: null` on the same config and the next export to decline a question will not
 * necessarily decline that one.
 *
 * Verified against all 35 seeds: none carries any of these keys as a non-number, so this rejects
 * nothing already in the product.
 */
function refuseUnreadableFallback(
  id: string,
  config: HfConfig,
  key: string,
  absenceMeans: string
): void {
  if (!Object.hasOwn(config, key)) return;
  const value = config[key];
  if (typeof value === 'number' && Number.isFinite(value)) return;

  throw new DerivationError(
    `${id}: states ${key}: ${JSON.stringify(value)} rather than omitting it. An absent ${key} ` +
      `means ${absenceMeans}, and a stated one that is not a number is a config declining to ` +
      'answer — so reading the two as the same thing applies that meaning to a model whose own ' +
      'config says it does not hold. Refusing to pick one.'
  );
}

// ---------------------------------------------------------------------------
// Architecture derivation
// ---------------------------------------------------------------------------

type AttentionCore =
  | { kind: 'gqa'; kvHeads: number; headDim: number }
  | { kind: 'mla'; kvLoraRank: number; qkRopeHeadDim: number };

/**
 * Config key *prefixes* that name a non-attention block's own dimensions.
 *
 * A prefix rather than a list of exact names, because an enumerated list is a list of the configs
 * its author happened to open. Thirteen exact names read off four configs already missed three
 * shipping models: Granite declares `mamba_chunk_size`, `mamba_conv_bias` and `mamba_proj_bias`
 * beside the six that got listed, Nemotron-Nano spells the same block `mamba_state_dim` /
 * `mamba_head_dim` / `mamba_num_heads` and shares *no* exact name with Granite's spelling, and
 * Kimi-Linear puts its entire linear-attention block inside a single nested `linear_attn_config`
 * object, so a flat name lookup saw nothing at all and the model derived as 27-layer MLA at 3.86x
 * over. The suffixes are per-architecture and move; the prefix is the part that holds. A block
 * configured under `linear_*` or `mamba_*` is a recurrence whose state is **constant in sequence
 * length**, whatever comes after the underscore.
 *
 * Verified against all 35 seeds — re-probed when the list grew from 17, since a prefix is exactly the
 * kind of guard that starts rejecting the product as the product moves: none of them carries a key
 * matching either prefix, and a match would refuse rather than mis-price, so the committed catalog
 * having 35 rows and no failures is the check rather than a claim about it.
 */
const LINEAR_STACK_PREFIXES = [/^linear_/, /^mamba_/];

/**
 * The same statement, in the spellings whose key names carry no prefix worth generalising.
 *
 * There are only two attention families in this script's vocabulary, GQA and MLA, and both charge
 * every layer a growing cache; a stack that mixes attention with a recurrence reads as a clean hit
 * and is billed for a cache most of its layers never allocate. Qwen3-Next is 12 attention layers of
 * 48 (4.0x over: 96.0 KiB/token derived against 24.0 actual, 12.0 GiB against 3.0 at 128K context),
 * Granite 4.0-h-small 4 of 40 (10x), Kimi-Linear 7 of 27 (3.86x), LFM2-1.2B 6 of 16 (2.67x).
 *
 * Listed by config key rather than by architecture, because `architectures` is a transformers class
 * name that moves and these keys are what the block itself is configured from:
 *
 *   - `full_attention_interval` — Qwen3-Next: one attention layer every N, gated DeltaNet between.
 *   - `full_attn_idxs` — LFM2: the attending layers by index, short convolutions on the rest.
 *     Present in the 1.2B and 350M exports, which carry no `layer_types` at all, while the 2.6B and
 *     8B-A1B exports of the *same architecture* spell it `layer_types: ["conv", ...]`. One export
 *     refusing and another silently mis-pricing is why both axes are guarded rather than one.
 *   - `conv_L_cache` — LFM2 again: the short-conv state length, which is a state and not a cache.
 *   - `hybrid_override_pattern` — Nemotron-H's per-layer `M`/`*`/`-` string.
 *   - `mb_per_layer` — Phi-4-mini-flash-reasoning (SambaY): Mamba blocks interleaved by index, in a
 *     32-layer stack that otherwise derives as clean GQA with `sliding_window: 512` throughout. The
 *     exact split is in the modelling code rather than the config, which is a reason to refuse.
 *
 * A key here that a future config uses to mean something else costs one loud refusal and a one-line
 * fix, which is the direction this file has chosen to fail in everywhere else too.
 */
const LINEAR_STACK_KEYS = [
  'full_attention_interval',
  'full_attn_idxs',
  'conv_L_cache',
  'hybrid_override_pattern',
  'mb_per_layer',
];

/**
 * Every key in this config that says the stack is not attention all the way down, sorted.
 *
 * Sorted rather than in config order so the refusal reads the same whoever exported the JSON —
 * `Object.keys` follows insertion order, and the message is something tests and humans both quote.
 */
function linearStackKeys(config: HfConfig): string[] {
  return Object.keys(config)
    .filter((key) => config[key] !== undefined && config[key] !== null)
    .filter(
      (key) => LINEAR_STACK_KEYS.includes(key) || LINEAR_STACK_PREFIXES.some((p) => p.test(key))
    )
    .sort();
}

/**
 * How many layers actually attend, where the config states it outright.
 *
 * Half the value of refusing is that the next person does not have to re-derive the split, so where
 * it is in `config.json` the refusal counts the layers rather than merely naming a key — the
 * difference between "this looks unusual" and "36 of these 48 layers were about to be charged for a
 * cache they do not have". Three spellings state it; the rest only imply a hybrid, and an implied
 * one gets a refusal without a count rather than a count that was guessed.
 */
function statedAttendingLayers(
  config: HfConfig,
  layers: number
): { count: number; from: string } | undefined {
  const interval = num(config, 'full_attention_interval');
  if (interval !== undefined && interval > 0) {
    return { count: Math.floor(layers / interval), from: 'full_attention_interval' };
  }

  const idxs = config.full_attn_idxs;
  if (Array.isArray(idxs)) return { count: idxs.length, from: 'full_attn_idxs' };

  // Kimi-Linear: the same array, one level down inside the block's own config object.
  const linearAttn = config.linear_attn_config;
  if (linearAttn !== null && typeof linearAttn === 'object') {
    const full = (linearAttn as HfConfig).full_attn_layers;
    if (Array.isArray(full)) {
      return { count: full.length, from: 'linear_attn_config.full_attn_layers' };
    }
  }

  return undefined;
}

/**
 * Refuses a model whose layers do not all cache keys and values.
 *
 * The rule this file already applies to an MTP module, a short `layer_types` and a ratio-guard
 * miss, applied to the third attention family: refuse what cannot be derived. Pricing one properly
 * needs a third `AttentionCore` kind carrying the per-layer split *and* the constant state term,
 * and only the first half is in `config.json` — the state's shape is specific to the block
 * (DeltaNet's `num_v_heads * head_k_dim * head_v_dim` plus its conv window; Mamba-2's
 * `n_heads * d_head * d_state` plus its own; Kimi-Delta's `num_heads * head_dim` plus a
 * `short_conv_kernel_size` window) and its width is set by the runtime rather than by `torch_dtype`.
 * Deriving the split and inventing the term would put a made-up number inside the fix for a made-up
 * number.
 *
 * So the error carries the evidence instead: how many layers cache, how many do not, and which key
 * said so. A catalog with a gap and a loud reason is worth more than one with a confident 4x error.
 */
function refuseLinearStack(id: string, config: HfConfig, layers: number): void {
  const declared = linearStackKeys(config);

  if (declared.length > 0) {
    // Counted only when the config states it *and* the count is a genuine split. An interval of 1 is
    // legal and means every layer attends, which would otherwise produce "48 of 48 layers attend
    // and cache; the other 0 hold a recurrent state" — a sentence whose two clauses disagree, which
    // is the failure this file's own rule about predicates and their prose exists to prevent.
    const stated = statedAttendingLayers(config, layers);
    const split =
      stated !== undefined && stated.count > 0 && stated.count < layers
        ? ` ${stated.from} states the split: ${stated.count} of ${layers} layers attend and cache; ` +
          `the other ${layers - stated.count} hold a recurrent state that does not grow with context.`
        : '';

    throw new DerivationError(
      `${id}: declares ${declared.join(', ')} — a block whose state is constant in sequence ` +
        `length, so this script cannot assume all ${layers} layers cache keys and values.${split} ` +
        'The GQA and MLA branches charge every layer a growing cache: 4.0x over for Qwen3-Next ' +
        "at 12 of 48, 10x for Granite 4.0-h-small at 4 of 40, 3.86x for Kimi-Linear's 7 of 27. " +
        'Refusing rather than deriving: this needs a third attention family carrying the per-layer ' +
        "split and the block's constant state term, and the state term is not in config.json."
    );
  }

  /**
   * MiniMax's spelling of the same split, and the one entry in this guard that has to *admit*
   * something. `attn_type_list` is a per-layer array in which `1` is full attention, and MiniMax-M2
   * is all ones — M2 really does attend fully on every layer, and M1's hybrid lightning attention
   * did not carry forward. So the test is per entry rather than on the key being present at all,
   * or the fix for the hybrids would reject the model that turned out not to be one.
   */
  const attnTypes = config.attn_type_list;
  if (Array.isArray(attnTypes)) {
    if (attnTypes.length !== layers) {
      throw new DerivationError(
        `${id}: attn_type_list lists ${attnTypes.length} entries for ${layers} layers. ` +
          'Refusing to decide which of the two is the stack.'
      );
    }
    const other = attnTypes.filter((t) => t !== 1);
    if (other.length > 0) {
      const names = [...new Set(other.map((t) => JSON.stringify(t)))].join('/');
      throw new DerivationError(
        `${id}: attn_type_list marks ${other.length} of ${layers} layers as ${names} rather than ` +
          '1 (full attention). Whatever those layers are, they are not the growing cache the GQA ' +
          'branch would charge them for.'
      );
    }
  }
}

/**
 * Refuses a stack whose layers are described one at a time in `block_configs`, rather than by the
 * top-level fields every other branch here reads.
 *
 * A neural-architecture-search export — NVIDIA's Puzzle pipeline, `DeciLMForCausalLM` and
 * `NemotronHPuzzleForCausalLM` — states a *per block* attention and FFN, and the top-level
 * `num_attention_heads` / `num_key_value_heads` describe the widest block rather than the stack. Two
 * things go wrong together, which is why the arithmetic is so far off:
 *
 *   - **Blocks with no attention at all.** Llama-3_3-Nemotron-Super-49B-v1_5 marks 31 of its 80
 *     blocks `attention.no_op: true`. Those layers cache nothing; the GQA branch charges all 80.
 *   - **`num_key_value_heads: null`.** The grouping is per block (`n_heads_in_group: 8`), so the
 *     top-level key is *explicitly* null — and `?? heads` then reads it as full multi-head
 *     attention, 64 KV heads where the attending blocks have 8.
 *
 * Together: 80 × 2 × 64 × 128 × 2 = 2560 KiB/token derived against 49 × 2 × 8 × 128 × 2 = 196, a
 * factor of 13.06. At 128K context that is 320.0 GiB of cache against 24.5 — the direction that
 * tells someone to buy another eight GPUs.
 *
 * Keyed on `block_configs` itself rather than on anything inside it, deliberately: the two live
 * spellings disagree about the contents (`{attention: {no_op}, ffn: {ffn_mult}}` for Nemotron Super,
 * `{block_type: "mamba" | "moe" | ...}` for Nemotron-Labs-3-Puzzle) and share only the key. A guard
 * written against one shape would have admitted the other. What they have in common is the claim
 * this script cannot represent: the layers are not all alike.
 */
function refusePerBlockStack(id: string, config: HfConfig, layers: number): void {
  const blocks = config.block_configs;
  if (!Array.isArray(blocks)) return;

  /**
   * Counted where the block says so, for the same reason the linear-stack refusal counts: the next
   * person should not have to re-derive the split. Both spellings are read, and a block that states
   * neither is left out of the count rather than assumed to attend — an undercount here understates
   * the error, which is the safe direction for a sentence in an error message.
   */
  const noAttention = blocks.filter((block) => {
    if (block === null || typeof block !== 'object') return false;
    const entry = block as Record<string, unknown>;
    const attention = entry.attention as Record<string, unknown> | undefined;
    if (attention && typeof attention === 'object') return attention.no_op === true;
    return typeof entry.block_type === 'string' && entry.block_type !== 'attention';
  }).length;

  const split =
    noAttention > 0 && noAttention < blocks.length
      ? ` ${blocks.length - noAttention} of ${blocks.length} blocks carry attention; the other ` +
        `${noAttention} declare none, so they cache nothing that grows with context.`
      : '';

  throw new DerivationError(
    `${id}: declares block_configs for ${blocks.length} blocks against num_hidden_layers ` +
      `${layers} — a per-block architecture whose layers are not alike, so the top-level fields ` +
      `do not describe the stack.${split} ` +
      'Llama-3_3-Nemotron-Super-49B-v1_5 is 31 of 80 blocks without attention *and* states ' +
      'num_key_value_heads: null because the grouping is per block, which together read as ' +
      '80 uniform MHA layers: 2560 KiB/token against 196, 13.1x, 320.0 GiB against 24.5 at 128K. ' +
      "Refusing rather than deriving: this needs a per-layer attention shape, not a stack's."
  );
}

/**
 * Sparse attention — a second cache, and a bound on what the first one reads.
 *
 * Hoisted out of the MLA branch, which is where it started and where it could only ever catch half
 * of what it names. DeepSeek V3.2-Exp declares `kv_lora_rank` beside its indexer, so the guard fired;
 * V4 does not declare `kv_lora_rank` at all — 1 KV head at `head_dim: 512` and a bare
 * `qk_rope_head_dim` — so it lands in the *GQA* branch, where the indexer keys were never read. The
 * row that produced was wrong twice over: no indexer cache, and `sliding_window: 128` applied to
 * every layer as though V4's DSA window were a Mistral-style trailing window, which prices a
 * million-token context at 43 layers × 128 tokens.
 *
 * A guard that names a quantity has to run wherever that quantity can appear. Whether the main
 * attention is latent or grouped is a different question from whether there is a second cache
 * beside it.
 *
 * Two spellings, and neither implies the other: the flat `index_*` keys (DeepSeek, GLM's
 * `GlmMoeDsa`) and a nested `sparse_attention_config` object (MiniMax M3, whose block is configured
 * entirely inside it — the same shape that hid Kimi-Linear's linear block from a flat lookup).
 */
function refuseSparseAttention(id: string, config: HfConfig): void {
  const flat = ['index_n_heads', 'index_head_dim', 'index_topk'].filter(
    (key) => num(config, key) !== undefined
  );
  const nested =
    config.sparse_attention_config !== null && typeof config.sparse_attention_config === 'object'
      ? ['sparse_attention_config']
      : [];
  const declared = [...flat, ...nested];
  if (declared.length === 0) return;

  throw new DerivationError(
    `${id}: declares ${declared.join(', ')} — a sparse-attention indexer, which keeps a cache of ` +
      'its own and bounds what the main attention reads. Neither is modelled, so this row would ' +
      'understate KV and overstate time-to-first-token together.'
  );
}

/**
 * Refuses a stack where what a layer caches is not `2 × kvHeads × headDim`, whatever the top-level
 * KV fields say.
 *
 * The whole GQA branch rests on one unstated assumption: every attending layer keeps a key tensor
 * and a value tensor, of the same shape, sized by the same two numbers. Gemma 4 breaks that
 * assumption three separate ways, and every one of them reads as an ordinary sliding-window GQA
 * config — `layer_types` in the closed vocabulary, `sliding_window` stated, nothing a guard above
 * matches. It is the most-downloaded current family in the catalog's own charts, so this is the
 * refusal with the largest audience:
 *
 *   - **`attention_k_eq_v: true`** (31B, 12B, 26B-A4B) — keys and values are one tensor, so the
 *     cache is half of what the `2 ×` charges. A clean 2x overstatement on every layer.
 *   - **`num_global_key_value_heads` / `global_head_dim`** — the full-attention layers have their
 *     own KV shape. On the 31B that is 4 heads × 512 against the local 16 × 256: the same product
 *     by luck, and 1 × 512 against 8 × 256 on the 12B, which is not. One `kvHeads`/`headDim` pair
 *     cannot describe both, and `AttentionCore` holds exactly one.
 *   - **`num_kv_shared_layers`** — E4B shares the cache of 18 of its 42 layers with an earlier
 *     layer, so 24 layers' worth of cache is charged as 42. 1.75x.
 *
 * Each is a term this script could carry and does not, which is the same doctrine as the linear
 * stacks: refuse with the evidence rather than ship a confident factor of two. Note what is
 * deliberately not the signal — `head_dim` differing from `hidden_size / num_attention_heads` is
 * ordinary and true of half the catalog, and `sliding_window` beside `layer_types` is Gemma 3, which
 * derives correctly and stays.
 */
function refuseNonUniformCache(id: string, config: HfConfig, layers: number): void {
  if (config.attention_k_eq_v === true) {
    throw new DerivationError(
      `${id}: declares attention_k_eq_v — keys and values are one tensor, so a layer caches ` +
        'half of what this script charges it for. The GQA term is 2 * kvHeads * headDim per ' +
        'layer, which is exactly 2x over for every layer of this stack.'
    );
  }

  const shared = num(config, 'num_kv_shared_layers');
  if (shared !== undefined && shared > 0) {
    // The ratio only exists while some layer still keeps a cache. A config sharing all of them — or
    // more than it has — is malformed, and printing `Infinityx over` beside it would be a sentence
    // contradicting the count in front of it.
    const keeping = layers - shared;
    throw new DerivationError(
      `${id}: declares num_kv_shared_layers ${shared} of ${layers} — those layers reuse an ` +
        `earlier layer's cache rather than keeping one, so the stack holds ${Math.max(0, keeping)} ` +
        `layers' worth of KV and this script would charge ${layers}. ` +
        (keeping > 0
          ? `${(layers / keeping).toFixed(2)}x over.`
          : 'Refusing before working out what that even means.')
    );
  }

  // Read as a pair: either key alone is enough to say the global layers have their own shape, and
  // whether the products happen to match is not the point — `AttentionCore` carries one shape, and
  // a stack with two is outside what it can say. `null` is how Gemma 4's E-series states "absent".
  const globalHeads = num(config, 'num_global_key_value_heads');
  const globalDim = num(config, 'global_head_dim');
  if (globalHeads !== undefined || globalDim !== undefined) {
    const local = `${num(config, 'num_key_value_heads')} x ${num(config, 'head_dim')}`;
    throw new DerivationError(
      `${id}: declares ${globalHeads !== undefined ? 'num_global_key_value_heads' : 'global_head_dim'}` +
        ` — the full-attention layers cache ${globalHeads ?? '?'} x ${globalDim ?? '?'} where the ` +
        `windowed ones cache ${local}. This script derives one KV shape for the whole stack, so ` +
        "one of the two layer types would be priced with the other one's cache."
    );
  }
}

/**
 * Multi-head latent attention caches one compressed latent per token per layer; grouped-query
 * caches keys and values per KV head. Detected by the presence of `kv_lora_rank`, which is
 * what DeepSeek's config uses and no GQA model defines.
 *
 * Two families, and a model outside both is refused rather than flattened into the nearer one —
 * see {@link refuseLinearStack}. That refusal comes first deliberately: Qwen3-Next carries
 * `num_attention_heads`, `num_key_value_heads` and `head_dim` exactly where the GQA branch expects
 * them, so the branch below reads as a clean hit and there is no signal in it that 36 of its 48
 * layers were just charged for a cache they never allocate.
 *
 * Also returns the **attention projection width**, which is what QK^T and AV actually scale by
 * and is emphatically *not* `hidden_size`. A model is free to project to a wider or narrower
 * query space than its residual stream, and most current ones do: GLM-4.5-Air is 3x its hidden
 * size, Qwen3's MoEs 2x, while Gemma 3 27B and Mistral Small are *narrower*. Using hidden size
 * mis-scaled long-prompt TTFT in both directions.
 *
 * The four refusals run first and in front of *both* branches, which is the shape the sparse-indexer
 * guard had to be moved into: a claim about what a layer caches has to be checked wherever that
 * layer can appear, and asking it inside the MLA branch meant DeepSeek V4 — the same indexer, no
 * `kv_lora_rank` — walked past it into the GQA one.
 */
export function deriveAttention(
  id: string,
  config: HfConfig,
  layers: number
): { core: AttentionCore; projectionWidth: number } {
  refuseLinearStack(id, config, layers);
  refusePerBlockStack(id, config, layers);
  refuseSparseAttention(id, config);
  refuseNonUniformCache(id, config, layers);

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
  // The GQA branch's other fallback, guarded for the same reason as `num_key_value_heads` below and
  // in front of the line that takes it rather than after — a refusal that runs second reads the
  // derived value as evidence, and here the derived value is the thing that must not be computed.
  refuseUnreadableFallback(id, config, 'head_dim', 'hidden_size / num_attention_heads');
  // Most configs state head_dim; older ones imply it from hidden_size / num_attention_heads.
  const headDim = require(num(config, 'head_dim') ?? hidden / heads, id, 'head_dim');

  /**
   * An *absent* `num_key_value_heads` means full multi-head attention — one KV head per query head,
   * which is what Llama 2-era configs leave unstated. A key that is present and `null` means
   * something else entirely: the config is declining to answer, because the answer is per layer.
   *
   * The two were the same expression until `?? heads` turned Nemotron Super's explicit null into 64
   * KV heads on a model whose attending blocks have 8 — an 8x overstatement stacked on top of the
   * per-block one, from a config that says out loud that it is not going to tell you. Distinguished
   * here as well as guarded in {@link refusePerBlockStack}, because the null is the more general
   * statement of the two: the next export to make it may not carry `block_configs` at all.
   */
  /**
   * Every *present but unreadable* value, not only the literal null (found in review).
   *
   * The guard tested `=== null` because that is the form Nemotron Super ships. Any other non-number
   * — a per-layer array, an object, a string — walked straight past it: `num()` returns `undefined`
   * for those, `?? heads` then reads it as full multi-head attention, and the overstatement this
   * exists to refuse comes back for whichever exporter next represents per-layer grouping as a list.
   * `Number.isFinite` rather than `=== undefined`, because `num()` passes a NaN straight through —
   * so the `undefined` form of this fix still let `kvHeads: NaN` reach the catalog, which is worse
   * than the overstatement it was replacing. The distinction that matters is present-and-unusable
   * versus absent, and unusable includes every value arithmetic cannot survive.
   */
  if (
    Object.hasOwn(config, 'num_key_value_heads') &&
    !Number.isFinite(num(config, 'num_key_value_heads'))
  ) {
    throw new DerivationError(
      `${id}: states num_key_value_heads as ${JSON.stringify(config.num_key_value_heads)} rather ` +
        'than a number or omitting it, which is a config saying the grouping is not a property of ' +
        'the stack. Reading that as full multi-head attention charges ' +
        `${heads} KV heads per layer where a grouped layer has a fraction of that — 8x for ` +
        'Nemotron Super, whose attending blocks state n_heads_in_group: 8. Refusing to pick one.'
    );
  }

  return {
    core: {
      kind: 'gqa',
      kvHeads: num(config, 'num_key_value_heads') ?? heads,
      headDim,
    },
    projectionWidth: heads * headDim,
  };
}

/**
 * `layer_types` entries this script can price, and what each one caches.
 *
 * A closed vocabulary rather than a substring test, and that swap is half the fix for the hybrid
 * stacks. The filter here used to be `t.includes('sliding')`, which asks only whether a layer is
 * *windowed* attention and reads every other answer as full attention — so Granite 4.0-h-small's
 * array of 36 `mamba` to 4 `attention` matched nothing, `sliding.length === 0` returned `undefined`,
 * and all 40 layers were charged a growing cache that 36 of them never allocate. 10x, silently, in
 * the direction that tells someone to buy another GPU. An unrecognised layer type is the same defect
 * as a missing one, one axis over.
 *
 * `full_attention` and `sliding_attention` are transformers' own names; `attention` is what the
 * hybrid exports call the attention layers of a stack that has others.
 *
 * `chunked_attention` is deliberately absent, so a config that names it refuses — but note that is
 * not how Llama 4 is caught. Scout and Maverick ship no `layer_types` at all; their guard is
 * `attention_chunk_size` in {@link deriveLayerWindows}. A vocabulary only fires for configs that
 * use the key it is a vocabulary for.
 */
const LAYER_TYPES: Record<string, 'full' | 'sliding'> = {
  full_attention: 'full',
  attention: 'full',
  sliding_attention: 'sliding',
};

/**
 * Per-layer attention window, or undefined when every layer attends over the full context.
 *
 * Four conventions in the wild, in order of precedence:
 *   - `attention_chunk_size` — Llama 4's chunked attention, which is refused rather than priced
 *   - `layer_types` — an explicit per-layer array (gpt-oss, recent transformers exports)
 *   - `sliding_window_pattern` — Gemma 3's "every Nth layer is full attention"
 *   - a bare `sliding_window` — applies to every layer (Mistral-style), unless switched off
 */
export function deriveLayerWindows(
  id: string,
  config: HfConfig,
  layers: number
): (number | null)[] | undefined {
  const window = num(config, 'sliding_window');
  const layerTypes = config.layer_types;

  /**
   * Chunked (block-local) attention, which is a fourth window convention and the one this script
   * has no term for at all.
   *
   * Llama 4 states it as `attention_chunk_size` and nothing else: Scout's `text_config` carries no
   * `layer_types`, so every guard below misses and all 48 layers read as full attention — 192.0
   * KiB/token, 24.0 GiB at 128K, against 7.125 GiB for its real 12-global / 36-chunked-at-8192
   * stack.
   * 3.4x, in the direction that tells someone to buy another GPU. Maverick is the same shape.
   *
   * Refused rather than derived, and the reason is narrower than for the linear stacks: the split
   * *is* derivable — `no_rope_layers` is 48 entries of 1/0, one global layer every fourth — but how
   * many tokens a chunked layer's cache holds is not. Chunked attention is not a sliding window
   * re-spelled: the mask is block-diagonal rather than trailing, and residency is set by the
   * runtime's chunked-cache implementation rather than by this key. Reusing `sliding_window`'s term
   * for it would be a guess wearing a derivation's clothes.
   *
   * Note what is deliberately *not* the signal: `cache_implementation: "hybrid"` sits on
   * unsloth/gemma-3-12b-it and -27b-it, two shipped seeds whose windows derive correctly from
   * `sliding_window_pattern`. Guarding on it would have refused two rows that are already right.
   */
  const chunk = num(config, 'attention_chunk_size');
  if (chunk !== undefined) {
    throw new DerivationError(
      `${id}: declares attention_chunk_size ${chunk} — chunked attention on some layers and full ` +
        `attention on others, with no layer_types to say which. Reading all ${layers} layers as ` +
        'full attention overstates KV by the whole ratio of the split: Llama 4 Scout derives ' +
        '192.0 KiB/token, 24.0 GiB at 128K context, against 7.125 GiB for its real 12-global / ' +
        '36-chunked-at-8192 stack. The split is in no_rope_layers, but how much a chunked layer ' +
        'keeps is not in config.json, and it is not a sliding window of attention_chunk_size — ' +
        'the mask is block-diagonal, not trailing. Give it a term of its own.'
    );
  }

  if (Array.isArray(layerTypes)) {
    // All three of these silently read as full attention downstream — an absent array entry, an
    // unrecognised one and an absent array are indistinguishable to `layerWindows?.[i]` — which
    // overstates KV and prefill attention for exactly the models this handling exists to get right.
    //
    // Length is `!==` rather than `<` because a longer array and `num_hidden_layers` disagree about
    // the stack, and slicing picks one of them without saying so.
    if (layerTypes.length !== layers) {
      throw new DerivationError(
        `${id}: layer_types lists ${layerTypes.length} entries for ${layers} layers. ` +
          'Refusing to decide which of the two is the stack; the extras or the gaps would read ' +
          'as full attention.'
      );
    }

    const kindOf = (t: unknown) =>
      typeof t === 'string' && Object.hasOwn(LAYER_TYPES, t) ? LAYER_TYPES[t] : undefined;

    const unpriced = layerTypes.filter((t) => kindOf(t) === undefined);
    if (unpriced.length > 0) {
      const names = [...new Set(unpriced.map((t) => JSON.stringify(t)))].join('/');
      throw new DerivationError(
        `${id}: layer_types names ${unpriced.length} of ${layers} layers as ${names}, which this ` +
          'script has no cache term for. A layer that is not attention caches nothing that grows ' +
          'with context, so reading it as full attention overstates KV by the whole ratio of the ' +
          'split — 10x for a 36-of-40 Mamba stack, at 128K context tens of gigabytes. Give the ' +
          'type its own term rather than letting it fall through here.'
      );
    }

    const sliding = layerTypes.filter((t) => kindOf(t) === 'sliding');
    if (sliding.length > 0 && window === undefined) {
      throw new DerivationError(
        `${id}: layer_types names ${sliding.length} sliding layers but no sliding_window size. ` +
          'Refusing to treat them as full attention.'
      );
    }
    if (sliding.length === 0) return undefined;

    return layerTypes.map((t) => (kindOf(t) === 'sliding' ? window! : null));
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

export interface MoeDerivation {
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
export function deriveMoe(id: string, config: HfConfig, layers: number): MoeDerivation | null {
  const total = firstNum(config, 'num_local_experts', 'n_routed_experts', 'num_experts');
  const perToken = firstNum(config, 'num_experts_per_tok', 'experts_per_token');

  if (total === undefined && perToken === undefined) return null;

  /**
   * A dense model whose config carries the MoE keys zeroed, which is how a shared config class
   * states "this variant has no experts": `ibm-granite/granite-4.0-micro` declares
   * `num_local_experts: 0` and `num_experts_per_tok: 0` on a 40-layer stack that is dense
   * throughout. Both keys are present, so the partial guard above is satisfied, and the arithmetic
   * downstream is `(0 / 0) * 0` — **NaN**, which `JSON.stringify` writes to the catalog as `null`
   * and `toModel` does not check. A loud refusal would be acceptable here and a silent `null`
   * active-parameter count is not, but neither is necessary: zero experts is not an ambiguity, it
   * is a dense model saying so.
   */
  if (total === 0 && perToken === 0) return null;

  if (total === undefined || perToken === undefined || total === 0 || perToken === 0) {
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
   *
   * All four keys are read through a fallback, and all four fallbacks are substantive rather than
   * "feature absent" — which is why they are guarded first. An absent `first_k_dense_replace`
   * chooses Qwen's phase over DeepSeek's, and `?? 1` on either step counts *every* layer as MoE, so
   * a stated non-number would not fail: it would return a confident expert count off by however many
   * layers the config was declining to describe. MiniMax M3 states a per-layer `moe_layer_freq`
   * array — it refuses a step earlier for its sparse indexer, but the next model to do it may not.
   */
  refuseUnreadableFallback(id, config, 'first_k_dense_replace', "Qwen's MoE layer phase");
  refuseUnreadableFallback(id, config, 'moe_layer_freq', 'every layer from the first MoE one');
  refuseUnreadableFallback(id, config, 'decoder_sparse_step', 'every layer');
  if (Object.hasOwn(config, 'mlp_only_layers') && config.mlp_only_layers !== null) {
    if (!Array.isArray(config.mlp_only_layers)) {
      throw new DerivationError(
        `${id}: states mlp_only_layers: ${JSON.stringify(config.mlp_only_layers)}, which is not a ` +
          'list of layer indices. Read as an empty one it silently charges experts to the layers ' +
          'the config was excluding.'
      );
    }
  }

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

/**
 * The headline active-parameter count, in the convention vendors publish.
 *
 * A function rather than an expression inside `buildModel` so a test can reach it, and that is not
 * incidental: the arithmetic is `(perToken / total) * expertParams`, and the one shape that produces
 * a *number* rather than a refusal from {@link deriveMoe} — a dense model whose shared config class
 * zeroes the MoE keys — makes it `(0 / 0) * 0`, **NaN**, which `JSON.stringify` writes into the
 * committed catalog as `null` and `toModel` does not check. A test that retypes that expression as
 * literals proves nothing about the shipped path; one that calls this proves the consequence is
 * reachable from a value `deriveMoe` could return.
 *
 * `totalParams` for a dense model, not `activeDense`: a dense model's active count *is* its total,
 * which is what every vendor states. Subtracting the embedding here emitted Qwen3-32B as 31.98B
 * active against 32.76B total with no routed experts anywhere.
 */
export function publishedActiveParams(
  totalParams: number,
  activeDense: number,
  moe: MoeDerivation | null
): number {
  if (!moe) return totalParams;
  return activeDense + (moe.experts.perToken / moe.experts.total) * moe.expertParams;
}

/**
 * How far a derived active count may sit from the vendor's published one before the row is refused.
 *
 * 8%, which is the band `src/data/catalog.test.ts` holds every other MoE row to — tight enough that
 * adding the input embedding back in (the correction this pipeline exists to apply) fails every one
 * of the models that test names. Every row that carries a published figure lands inside 3%.
 */
const PUBLISHED_ACTIVE_TOLERANCE = 0.08;

export function reconcileActiveParams(id: string, derived: number, published: number): void {
  const drift = Math.abs(derived - published) / published;
  if (drift <= PUBLISHED_ACTIVE_TOLERANCE) return;

  throw new DerivationError(
    `${id}: derives ${(derived / 1e9).toFixed(2)}B active against the published ` +
      `${(published / 1e9).toFixed(2)}B — ${(drift * 100).toFixed(1)}% out, past the ` +
      `${(PUBLISHED_ACTIVE_TOLERANCE * 100).toFixed(0)}% this catalog holds every other MoE row ` +
      'to. The row would render both figures in the same control: the derived one in the model ' +
      "label, the published one in the note beside it. Since the seed's totalParams is a constant " +
      'and the routed experts are subtracted from it exactly, the usual cause is that constant — ' +
      'whatever it rounds away lands in the dense residual, at total/dense times its size.'
  );
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

/**
 * Tensor names in a repo, and which shard each lives in.
 *
 * Fetched through `/resolve/` rather than `/raw/`, which is the difference between reading the index
 * and reading a three-line pointer to it. Git LFS stores anything large, and a trillion-parameter
 * model's index is large: `moonshotai/Kimi-K2-Instruct` lists 105,000 tensors in 12 MB, so `/raw/`
 * answers 200 with `version https://git-lfs.github.com/spec/v1` and `JSON.parse` fails on the letter
 * `v`. The failure was indistinguishable from a corrupt repo and it applied to precisely the models
 * this catalog most wants — the ones big enough that where they fit is the whole question.
 *
 * `/resolve/` serves the content either way, which is why the shard reads below have always used it.
 */
async function fetchTensorMap(id: string, revision: string): Promise<Record<string, string>> {
  const url = `https://huggingface.co/${id}/resolve/${revision}/model.safetensors.index.json`;
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
  /**
   * Parameters in an output head that a tied model ships anyway, which the repository stores and
   * inference does not hold.
   *
   * `ibm-granite/granite-4.1-8b` is the live case: `tie_word_embeddings: true` beside a
   * `lm_head.weight` of [100352, 4096] identical in shape to `model.embed_tokens.weight`.
   * `from_pretrained` lists `lm_head.weight` in `_tied_weights_keys` and overwrites it with the
   * embedding at load, and llama.cpp's converter drops it for the same reason — so the resident model
   * holds one table where the safetensors index counts two, and taking the index at face value
   * overstates the model by 0.41B of 8.79B. Subtracted from the total, not from the per-token count:
   * decode reads that one table every step, which is what being tied means.
   */
  duplicatedOutputParams: number;
}

/** Names an untied output projection is stored under, across architectures. */
const OUTPUT_HEAD_SUFFIXES = [
  'lm_head.weight',
  'output_layer.weight',
  'output.weight',
  'embed_out.weight',
];

/**
 * How many parameters a tied model's duplicate output table holds, read from the shard header.
 *
 * Pure and exported for the reason the arithmetic is here at all: the figure is subtracted from a
 * published parameter count, and the failure mode is not a crash but a row that is 4.9% heavy while
 * still claiming to be tied.
 *
 * Two things a `reduce` over a possibly-absent shape gets wrong, and the second is why this is a
 * function rather than an expression. `(shape ?? []).reduce((a, b) => a * b, 1)` is **1** for a
 * tensor the header does not carry — a real divergence, since the index and the shard header are
 * written by different tools and need not agree on a name — so a `<= 0` guard cannot fire, and
 * `granite-4.1-8b` would ship 8.79B against a resident 8.38B with `tiedEmbeddings: true` beside it:
 * exactly the row the subtraction exists to correct, silently uncorrected. The identity element of
 * multiplication is the one value that makes "no data" indistinguishable from a plausible answer.
 *
 * And the element count is compared against the embedding table, because a tie is a claim that the
 * two tables *are* one: `[100352, 4096]` against a `[100352, 4096]` embedding. Compared as a product
 * rather than as a shape so a transposed export does not read as a disagreement. Note what this
 * cannot see, since the comment it replaces implied otherwise: a head of identical shape that is
 * genuinely independent — a fine-tune that untied the head without editing `config.json` — is
 * indistinguishable from a duplicate by measurement alone, and the config's own `true` is the only
 * evidence either way. That case is wrong in both directions and no shape check reaches it.
 */
export function duplicatedOutputParams(
  id: string,
  outputHead: string,
  header: Record<string, { dtype?: string; shape?: number[] }>,
  embeddingParams: number
): number {
  const shape = header[outputHead]?.shape;
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new DerivationError(
      `${id}: config ties the embeddings and ${outputHead} is in the index, but the shard header ` +
        `gives it ${JSON.stringify(shape)} — so whether the total counts one table or two cannot ` +
        'be settled. An unreadable shape is not a zero-sized tensor.'
    );
  }

  const params = shape.reduce((a, b) => a * b, 1);
  if (!Number.isFinite(params) || params <= 0) {
    throw new DerivationError(
      `${id}: ${outputHead} has shape ${JSON.stringify(shape)}, which is not a parameter count.`
    );
  }
  if (params !== embeddingParams) {
    throw new DerivationError(
      `${id}: config ties the embeddings, but ${outputHead} holds ${params} parameters against ` +
        `the embedding table's ${embeddingParams}. A tie is a claim that the two are one tensor, ` +
        'so a head of a different size is an untied projection whatever the config says. Refusing ' +
        'to subtract it.'
    );
  }
  return params;
}

async function deriveStackShape(
  id: string,
  revision: string,
  declaredTied: boolean | undefined,
  embeddingParams: number
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

  /**
   * `tie_word_embeddings` is not trustworthy enough to derive from — it is absent on both Gemma
   * 3 repos despite them being tied — but when a repo *does* state it, a disagreement means
   * this list of names is incomplete for that architecture. Better to stop than to guess which
   * side is right.
   */
  if (declaredTied === false && outputHead === undefined) {
    throw new DerivationError(
      `${id}: config says the embeddings are untied, but no output projection matched ` +
        `${OUTPUT_HEAD_SUFFIXES.join(', ')}. Add this architecture's output tensor name.`
    );
  }

  /**
   * The mirror disagreement, which was unguarded and is not a tie: `tie_word_embeddings: true`
   * *beside* an output head means the head is a duplicate the loader discards, not that the tensor
   * list is incomplete.
   *
   * Note this does not reinstate `tie_word_embeddings` as the source of truth — absence of the key
   * still says nothing, which is the Gemma 3 case the field's comment records. What a stated `true`
   * settles is the one question the tensor list cannot answer on its own: whether a head that *is*
   * there is loaded. `granite-4.1-8b` ships both tables at [100352, 4096]; transformers lists
   * `lm_head.weight` in `_tied_weights_keys` and overwrites it at load, so the resident model holds
   * 8.38B parameters against the index's 8.79B. Reading the head as an untied projection got that
   * row wrong twice: 4.7% too heavy, and its embedding subtracted from a per-token count that reads
   * it in full every step.
   */
  const tiedEmbeddings = outputHead === undefined || declaredTied === true;
  let duplicated = 0;
  if (outputHead !== undefined && declaredTied === true) {
    // Measured from the shard rather than assumed to be vocab x hidden: the whole point of the
    // subtraction is that it is a real tensor with a real size. `duplicatedOutputParams` checks it
    // against the embedding table and refuses the shapes that cannot settle the question.
    const header = await fetchSafetensorsHeader(id, revision, weightMap[outputHead]);
    duplicated = duplicatedOutputParams(id, outputHead, header, embeddingParams);
  }

  const otherShards = [
    ...new Set(names.filter((n) => classifyTensor(n) === 'other').map((n) => weightMap[n])),
  ];
  if (otherShards.length === 0) {
    return { tiedEmbeddings, nonLanguageParams: 0, duplicatedOutputParams: duplicated };
  }

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
      // The same `?? []` as the output head above, and it cannot go wrong the same way — the name
      // comes from this header rather than from the index, so the tensor is always there. An absent
      // `shape` on a tensor that *is* there is still a header this script cannot read, and summing
      // it as one parameter would understate a vision tower by however many it holds. An empty shape
      // is left alone: safetensors spells a scalar that way, and a scalar really is one element.
      if (tensor.shape === undefined) {
        throw new DerivationError(
          `${id}: non-language tensor ${name} carries no shape, so its parameter count is ` +
            'unreadable. Refusing to charge it as one.'
        );
      }
      nonLanguageParams += tensor.shape.reduce((a, b) => a * b, 1);
    }
  }

  return { tiedEmbeddings, nonLanguageParams, duplicatedOutputParams: duplicated };
}

async function buildModel(seed: Seed) {
  // Two calls on purpose. The first resolves `main` to a commit; the second asks for that
  // commit specifically, so the parameter totals describe the same revision as the architecture
  // and the tensor layout. Reading the totals from the unpinned endpoint left one field of the
  // row sourced from an asynchronously computed cache that is not guaranteed to match the sha
  // reported beside it — a stale total next to current expert and embedding counts, consistent
  // enough to look right.
  const resolved = await fetchJson<HfApiModel>(
    `https://huggingface.co/api/models/${seed.id}?expand[]=sha`,
    seed.id
  );
  if (!resolved.sha) {
    throw new DerivationError(
      `${seed.id}: API returned no commit sha, so the fetches cannot be pinned to one revision`
    );
  }
  const revision = resolved.sha;

  const api = await fetchJson<HfApiModel>(
    `https://huggingface.co/api/models/${seed.id}/revision/${revision}` +
      '?expand[]=safetensors&expand[]=downloads&expand[]=likes&expand[]=createdAt&expand[]=sha',
    seed.id
  );
  if (api.sha !== revision) {
    throw new DerivationError(
      `${seed.id}: asked for revision ${revision} but the API answered for ${api.sha}`
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
  // Read here rather than beside the count it corrects, because `deriveStackShape` needs it to check
  // a tied model's duplicate head against the table it is supposed to be a duplicate of.
  const embeddingParams = vocabSize * hiddenSize;

  /**
   * The attention stack is settled here, before anything that touches the network again, and the
   * order is load-bearing rather than tidy. These two read `config.json` alone; `deriveStackShape`
   * below fetches a safetensors index and one header per non-language shard.
   *
   * Qwen3-Next is why. It ships an MTP module under an `mtp.` prefix, so run in the other order the
   * seed is refused for 1,553 unclassified tensors — a true statement about a different problem,
   * which sends whoever reads it to `LANGUAGE_PREFIXES` rather than to the 36 of 48 layers that
   * hold a recurrent state. An architecture this script cannot price should say so from the config
   * that says so, and should not spend a dozen range requests first.
   */
  const layerWindows = deriveLayerWindows(seed.id, config, layers);
  const attention = deriveAttention(seed.id, config, layers);

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

  /**
   * The tensor layout, settled before the parameter count rather than after it.
   *
   * It used to run below, which was fine while it only answered "is this tied?" — a question nothing
   * upstream depends on. It now also answers "does the index count the output table twice?", and that
   * *is* the parameter count. Everything above this line still reads `config.json` alone, so the
   * property the ordering exists for holds: a model this script cannot price refuses from the config
   * that says so, without spending a dozen range requests first.
   *
   * Two corrections come out of it, and both were wrong in the direction of a slower machine:
   *
   *   - **Tied embeddings.** When a model reuses the embedding table as its output projection, that
   *     table is a full vocab matmul on every step. Subtracting it is right for untied models like
   *     gpt-oss and wrong for tied ones like Gemma 3 and Qwen3-4B.
   *   - **Non-text towers.** Gemma 3's vision encoder occupies memory but does not run for a text
   *     token, so it belongs in `totalParams` and not in the per-token count.
   */
  const stack = await deriveStackShape(
    seed.id,
    revision,
    typeof config.tie_word_embeddings === 'boolean' ? config.tie_word_embeddings : undefined,
    embeddingParams
  );

  /**
   * An override is the *published* figure, so it already describes the model as loaded — the
   * duplicate-table subtraction applies only to a count derived from the index. Applying both would
   * take 0.41B off a number a vendor stated.
   */
  const totalParams =
    seed.overrides?.totalParams ??
    deriveTotalParams(seed.id, api, expertParams) - stack.duplicatedOutputParams;

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
  const activeDense = Math.max(0, denseParams - embeddingParams);
  const activeParams = publishedActiveParams(totalParams, activeDense, moe);

  /**
   * And checked against the vendor's own figure, where the seed states one.
   *
   * Only override rows carry one, because only they need it: a row whose total is derived reconciles
   * or fails visibly, while a row whose total is a published constant hides the whole of that
   * constant's rounding in the dense residual. See {@link Seed.overrides.publishedActiveParams}.
   */
  if (seed.overrides?.publishedActiveParams !== undefined) {
    reconcileActiveParams(seed.id, activeParams, seed.overrides.publishedActiveParams);
  }

  /**
   * `activeParams` above is the *published* convention, and it is not what a decode step reads — the
   * two corrections in {@link deriveStackShape}'s comment are what separate them.
   *
   * Kept as its own field rather than folded into `activeParams`, because the published figure
   * is what the catalog tests check against vendors and what users recognise on a model card.
   */
  const activeDenseParams = Math.max(
    0,
    denseParams - stack.nonLanguageParams - (stack.tiedEmbeddings ? 0 : embeddingParams)
  );

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
// What the field is downloading that this list does not carry
// ---------------------------------------------------------------------------

/** One row of Hugging Face's model listing, in the fields the report reads. */
export interface LiveModel {
  id: string;
  downloads?: number;
  createdAt?: string;
}

/**
 * Repo-id shapes that are a *derivative* of a model rather than a model.
 *
 * The download charts are mostly these: a quantized re-export, a GGUF conversion, a speculative
 * decoding draft, an ONNX or MLX conversion. Every one of them is the same architecture as something
 * already listed or refused, so reporting them would bury the two or three rows that matter under
 * forty that do not — which is the failure mode of every automated report that gets ignored.
 *
 * Matched on the id rather than on tags, because tags are applied by uploaders and these suffixes are
 * a naming convention publishers follow consistently. A model that slips through costs one line in a
 * weekly report; a filter on the wrong axis costs the whole report's credibility.
 */
const DERIVATIVE_PATTERNS = [
  /gguf/i,
  /awq/i,
  /gptq/i,
  /\bfp8\b|-fp8/i,
  /nvfp4|mxfp4|mxfp8/i,
  /-int[48]\b/i,
  /w[48]a(16|8|4)/i,
  /\b(4|8)bit\b|-(4|8)bit/i,
  /-onnx/i,
  /-mlx/i,
  /-bnb-/i,
  /eagle/i,
  /dspark|dflash/i,
  /-base$/i,
  /-original$/i,
  /-unquantized/i,
  /-qat-/i,
  /tiny-random/i,
  /abliterated/i,
];

/**
 * Orgs that mostly re-publish other people's weights.
 *
 * The catalog *uses* several of these as mirrors for gated repos, which is exactly why they cannot
 * appear in the report: `unsloth/gemma-3-4b-it` is already a row, under Google's traffic, and naming
 * it as a candidate would ask a reader to add what is already there.
 */
const MIRROR_ORGS = new Set([
  'unsloth',
  'NousResearch',
  'RedHatAI',
  'bartowski',
  'TheBloke',
  'mradermacher',
  'lmstudio-community',
  'ModelCloud',
  'cognitivecomputations',
  'trl-internal-testing',
  'peft-internal-testing',
  'hmellor',
]);

/**
 * Models the field is downloading that this seed list neither carries nor has written down a reason
 * for, most-downloaded first.
 *
 * The gap the weekly refresh structurally cannot see. It re-derives every figure on every row, so a
 * publisher editing `config.json` is caught within a week — and a model that was never seeded is
 * invisible to it forever, which is how {@link SEEDS} came to be a year behind while every number in
 * it was seven days old. Absence needs its own check, and this is it.
 *
 * Pure, and separated from the fetch for the same reason `compare` in `catalog-diff.ts` is: what it
 * decides is which names land in front of a human every Monday, and both wrong answers are
 * expensive. Too many and the report is noise nobody reads; too few and it quietly stops working.
 *
 * `minDownloads` is a floor rather than a top-N so that a quiet week reports nothing at all instead
 * of scraping the barrel for something to say. `since` drops the models whose downloads are a decade
 * of accumulated tutorials — gpt2 and opt-125m outrank most of the current field and are not
 * candidates for a hardware calculator.
 */
export function unseededCandidates(options: {
  live: readonly LiveModel[];
  seeded: ReadonlySet<string>;
  notSeeded: ReadonlySet<string>;
  minDownloads: number;
  since: Date;
}): LiveModel[] {
  const { live, seeded, notSeeded, minDownloads, since } = options;

  return live
    .filter((model) => (model.downloads ?? 0) >= minDownloads)
    .filter((model) => !seeded.has(model.id) && !notSeeded.has(model.id))
    .filter((model) => !MIRROR_ORGS.has(model.id.split('/')[0]))
    .filter((model) => !DERIVATIVE_PATTERNS.some((pattern) => pattern.test(model.id)))
    .filter((model) => {
      // A missing date is reported rather than dropped: the point of the report is the thing nobody
      // has looked at, and silence about a row because its metadata was thin is the wrong default.
      if (!model.createdAt) return true;
      const created = Date.parse(model.createdAt);
      return !Number.isFinite(created) || created >= since.getTime();
    })
    .sort((a, b) => (b.downloads ?? 0) - (a.downloads ?? 0));
}

/** Every repo id the catalog already speaks for, mirrors and canonical repos alike. */
export function seededIds(seeds: readonly Seed[] = SEEDS): Set<string> {
  return new Set(
    seeds.flatMap((seed) => [seed.id, ...(seed.popularityId ? [seed.popularityId] : [])])
  );
}

/**
 * Prints the candidates at the end of a run, so the weekly job's log carries them.
 *
 * Wrapped in its own error handling and deliberately after the write: this is a report, and a
 * listing endpoint having a bad minute must not be able to fail a refresh that has already
 * successfully derived every row.
 *
 * **The two thresholds below are a floor on what this surfaces, not a bound on what
 * {@link NOT_SEEDED} has to explain.** A repo can be worth an entry and permanently invisible here:
 * `Mistral-Nemo-Instruct-2407` has 438K downloads and predates the 18-month window, and
 * `Devstral-Small-2507` and `c4ai-command-a-03-2025` are under the download floor — all three were
 * named by id in #77, and all three were absent from this table until a reviewer looked for them.
 * The report catches the *field moving*; a question somebody has already asked has to be written
 * down when it is answered.
 */
async function reportSeedCandidates(): Promise<void> {
  const MIN_DOWNLOADS = 250_000;
  const MONTHS = 18;
  const since = new Date();
  since.setMonth(since.getMonth() - MONTHS);

  /**
   * Every pipeline tag this catalog actually carries, paginated until the downloads fall below the
   * bar. Both halves were wrong in the first version (found in review on #77).
   *
   * **`text-generation` alone hid the multimodal families the catalog already counts.** Gemma 3,
   * Mistral Small 4 and Command A+ are published under image-text-to-text, and the generator
   * deliberately classifies their non-language towers rather than refusing them — so a newly popular
   * multimodal family could clear every threshold here and never appear. The tags are listed rather
   * than dropped, because the point of the filter is to exclude the things this tool cannot price at
   * all: embedders, rerankers, diffusion.
   *
   * **And one page of 200 is not the top of the chart, it is the top of the *unfiltered* chart.** The
   * report's own prose says derivative uploads dominate the download ranking — GGUF conversions,
   * AWQ requantisations, fine-tunes — and every one of those is discarded downstream by
   * `unseededCandidates`. So the first 200 rows can be almost entirely noise and a qualifying model
   * at position 201 is never examined. Paged until a page's lowest download count drops under
   * `MIN_DOWNLOADS`, which is the point past which nothing can qualify however many pages remain.
   */
  const PIPELINES = ['text-generation', 'image-text-to-text', 'image-to-text'];
  const PAGE = 200;
  const live: LiveModel[] = [];
  try {
    for (const pipeline of PIPELINES) {
      for (let page = 0; ; page++) {
        const batch = await fetchJson<LiveModel[]>(
          `https://huggingface.co/api/models?pipeline_tag=${pipeline}&sort=downloads` +
            `&direction=-1&limit=${PAGE}&skip=${page * PAGE}` +
            '&expand[]=downloads&expand[]=createdAt',
          `seed candidates (${pipeline}, page ${page + 1})`
        );
        live.push(...batch);
        // Sorted by downloads descending, so once a page ends below the bar every later one is too.
        const lowest = batch.at(-1)?.downloads ?? 0;
        if (batch.length < PAGE || lowest < MIN_DOWNLOADS) break;
      }
    }
  } catch (error) {
    const why = error instanceof Error ? error.message : String(error);
    console.warn(`\nCould not list candidate models: ${why}`);
    /**
     * And say so where the report would have been (found in review on #77).
     *
     * The workflow's closing notice points a maintainer at this run's summary unconditionally, so a
     * failed listing left no summary at all beside a sentence saying the candidates are in it —
     * silence that reads exactly like "no candidates". That is worse than the transient-log problem
     * the summary was added to fix, because it is indistinguishable from good news.
     */
    const target = process.env.GITHUB_STEP_SUMMARY;
    if (target) {
      await appendFile(
        target,
        `\n### Seed candidates — not checked\n\nThe Hugging Face listing failed, so the seed list ` +
          `was **not** compared against what the hub is downloading on this run: ${why}\n\n` +
          'This is not "no candidates". The next scheduled run re-asks; a second failure is worth ' +
          'looking at, since nothing else here can detect that the seed list has aged.\n'
      );
    }
    return;
  }

  const candidates = unseededCandidates({
    live,
    seeded: seededIds(),
    notSeeded: new Set(Object.keys(NOT_SEEDED)),
    minDownloads: MIN_DOWNLOADS,
    since,
  });

  const heading =
    `Seed candidates — released in the last ${MONTHS} months, over ` +
    `${(MIN_DOWNLOADS / 1000).toFixed(0)}K downloads, neither seeded nor listed in NOT_SEEDED`;

  console.log(`\n${heading}:`);
  for (const model of candidates.slice(0, 25)) {
    console.log(
      `  ${String(model.downloads ?? 0).padStart(10)}  ${(model.createdAt ?? '').slice(0, 10)}  ${model.id}`
    );
  }
  if (candidates.length === 0) {
    console.log('  (none — the seed list covers what the field is downloading)');
  } else {
    console.log(
      `\n  ${candidates.length} candidate(s). Each one is either a seed or a line in NOT_SEEDED ` +
        'saying why not — the list ages silently otherwise.'
    );
  }

  /**
   * And the same report somewhere that outlives the job log (found in review on #77).
   *
   * The mechanism's whole purpose is to tell a maintainer the seed list has aged, and on the common
   * weekly run — figures unchanged, `changed != 'true'`, no pull request opened — the only trace was
   * this `console.log` inside a step nobody has a reason to expand. The job ends on "Catalog is
   * current", which is exactly the sentence that makes someone not look.
   *
   * `$GITHUB_STEP_SUMMARY` renders on the run's own page, so the candidates are visible from the
   * Actions list without opening a log, on every path through the workflow. Guarded on the variable
   * so a local `npm run catalog` is unchanged, and appended rather than written so a later step can
   * add to it.
   */
  const summary = process.env.GITHUB_STEP_SUMMARY;
  if (!summary) return;
  const rows = candidates
    .slice(0, 25)
    .map(
      (m) =>
        `| \`${m.id}\` | ${(m.downloads ?? 0).toLocaleString()} | ${(m.createdAt ?? '').slice(0, 10)} |`
    )
    .join('\n');
  await appendFile(
    summary,
    candidates.length === 0
      ? `\n### ${heading}\n\nNone — the seed list covers what the field is downloading.\n`
      : `\n### ${heading}\n\n${candidates.length} candidate(s). Each is either a new seed or a line in ` +
          '`NOT_SEEDED` saying why not; the list ages silently otherwise.\n\n' +
          `| model | downloads | released |\n| --- | --- | --- |\n${rows}\n`
  );
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
   * error: on the tolerance it carried, ten of thirty-five seeds could 503 and the run would still
   * exit 0 having dropped 29% of the catalog.
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
    await reportSeedCandidates();
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
  await reportSeedCandidates();
}

/**
 * Guarded so the derivations above can be imported by a test without the script running itself —
 * the same guard `catalog-diff.ts` carries, and for a sharper reason here: importing this module
 * unguarded starts one round of network fetches per seed, which is now thirty-five of them.
 *
 * They went untested for exactly that long. `deriveAttention` flattening a hybrid stack into GQA
 * and `deriveLayerWindows` refusing only along the sliding axis were both reachable from a
 * three-line unit test the whole time; what was missing was the ability to call either of them.
 */
if (process.argv[1] && /build-catalog\.ts$/.test(process.argv[1])) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
