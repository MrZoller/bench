import { describe, expect, it } from 'vitest';
import { judgeWorkloads, WORKLOADS, type VerdictInputs } from './verdict';
import { evaluate } from './index';
import {
  DGX_SPARK,
  EPYC_9654,
  GPT_OSS_20B,
  DEEPSEEK_V3,
  LLAMA_31_8B,
  LLAMA_CPP,
  MLX,
  RTX_5090,
  MAC_STUDIO_M3_ULTRA_256,
} from './fixtures';
import { getQuant } from '@/data/quants';

/**
 * The verdict layer turns a number into a decision, so what these tests guard is the *shape* of
 * that decision — that a fast rig passes the latency-sensitive archetypes, that a slow one is
 * still useful for batch, and above all that nothing is graded as usable when it cannot run.
 *
 * Thresholds are judgement, not measurement, so the assertions are about ordering and about the
 * cases where the answer is not arguable, rather than about exact boundaries.
 */

function judge(model: Parameters<typeof evaluate>[0]['model'], quantId: string, rig: VerdictRig) {
  const usage = {
    contextTokens: rig.contextTokens ?? 8192,
    concurrency: rig.concurrency ?? 1,
    promptTokens: rig.promptTokens ?? 2048,
    kvPrecision: 'fp16' as const,
  };
  const evaluation = evaluate({
    model,
    quant: getQuant(quantId),
    usage,
    rig: { device: rig.device, count: rig.count ?? 1 },
    runtime: rig.runtime ?? LLAMA_CPP,
  });

  const inputs: VerdictInputs = {
    placement: evaluation.placement,
    decode: evaluation.decode,
    usage,
    maxContextTokens: evaluation.maxContextTokens,
    runnableContextTokens: evaluation.runnableContextTokens,
    // Each archetype is graded at its own scenario, decode included.
    evaluateAt: (promptTokens, contextTokens) => {
      const e = evaluate({
        model,
        quant: getQuant(quantId),
        usage: { ...usage, promptTokens, contextTokens },
        rig: { device: rig.device, count: rig.count ?? 1 },
        runtime: rig.runtime ?? LLAMA_CPP,
      });
      return { decode: e.decode, prefill: e.prefill };
    },
  };
  return new Map(judgeWorkloads(inputs).map((v) => [v.workload.id, v]));
}

interface VerdictRig {
  device: Parameters<typeof evaluate>[0]['rig']['device'];
  count?: number;
  runtime?: Parameters<typeof evaluate>[0]['runtime'];
  contextTokens?: number;
  concurrency?: number;
  promptTokens?: number;
}

describe('workload verdicts', () => {
  it('grades every archetype, every time', () => {
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090 });
    expect(verdicts.size).toBe(WORKLOADS.length);
    for (const verdict of verdicts.values()) {
      expect(verdict.reason).not.toBe('');
      expect(['good', 'tight', 'fail']).toContain(verdict.fitness);
    }
  });

  /**
   * The headline case: a small dense model on a fast card should be good at the things people
   * buy a fast card for.
   */
  it('passes interactive chat on an 8B model and a 5090', () => {
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090 });
    expect(verdicts.get('chat')?.fitness).toBe('good');
  });

  /**
   * Completion is graded on prompt length, not just hardware, and that is the point: the same
   * 8B model on the same 5090 is *tight* at a 2K prompt and clears the bar at 512, because a
   * 400ms budget is spent almost entirely on prefill. A verdict layer that ignored the prompt
   * would call both of them the same thing.
   */
  /**
   * Each archetype is graded at the prompt it would really send, not at whatever the slider
   * happens to say. That is what stops a machine failing chat on an 8K prompt while "passing"
   * coding agent, which does everything chat does over a far bigger one.
   */
  it('grades each archetype at its own prompt length, not the slider', () => {
    const rank = { good: 2, tight: 1, fail: 0 };

    // A Spark prefills slowly. Whatever the slider says, an agent can never outrank chat.
    for (const promptTokens of [512, 8192, 65536]) {
      const verdicts = judge(GPT_OSS_20B, 'mxfp4', { device: DGX_SPARK, promptTokens });
      expect(rank[verdicts.get('agent')!.fitness]).toBeLessThanOrEqual(
        rank[verdicts.get('chat')!.fitness]
      );
    }
  });

  it('does not let the prompt slider change a verdict it should not touch', () => {
    const short = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, promptTokens: 512 });
    const long = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, promptTokens: 65536 });

    // Inline completion always sends a small prompt, so its grading is a property of the
    // hardware and model — not of what the user last dragged.
    expect(short.get('completion')?.fitness).toBe(long.get('completion')?.fitness);
  });

  /**
   * The inverse, and the reason the tool exists: a 671B model on a CPU host is not "slow", it is
   * a different category of machine. Batch still works; nothing interactive does.
   */
  it('fails interactive work on an EPYC host but keeps batch alive', () => {
    const verdicts = judge(DEEPSEEK_V3, 'q8_0', { device: EPYC_9654 });
    expect(verdicts.get('completion')?.fitness).toBe('fail');
    expect(verdicts.get('chat')?.fitness).toBe('fail');
    expect(verdicts.get('batch')?.fitness).not.toBe('fail');
  });

  /**
   * Latency budgets are a ladder: inline completion is strictly harder than chat, which is
   * strictly harder than batch. A grading that ever inverts that is wrong regardless of the
   * thresholds chosen.
   */
  it.each([
    ['5090 + 8B', LLAMA_31_8B, 'q4_k_m', RTX_5090],
    ['Spark + gpt-oss-20b', GPT_OSS_20B, 'mxfp4', DGX_SPARK],
    ['Mac + gpt-oss-20b', GPT_OSS_20B, 'mxfp4', MAC_STUDIO_M3_ULTRA_256],
  ])('never grades completion above chat on %s', (_label, model, quant, device) => {
    const rank = { good: 2, tight: 1, fail: 0 };
    const verdicts = judge(model, quant, { device });

    const completion = rank[verdicts.get('completion')!.fitness];
    const chat = rank[verdicts.get('chat')!.fitness];
    expect(completion).toBeLessThanOrEqual(chat);
  });

  /**
   * The one that must never be soft. A configuration that cannot run has no workloads it is
   * good at, and saying otherwise is worse than saying nothing.
   */
  it('fails everything when the runtime cannot drive the hardware', () => {
    const verdicts = judge(GPT_OSS_20B, 'mxfp4', {
      device: RTX_5090,
      runtime: MLX, // Apple-only, on an NVIDIA card.
    });

    for (const verdict of verdicts.values()) {
      expect(verdict.fitness).toBe('fail');
      expect(verdict.reason).toMatch(/does not run/i);
    }
  });

  it('fails everything when the model cannot fit and cannot spill', () => {
    const verdicts = judge(DEEPSEEK_V3, 'q8_0', {
      device: MAC_STUDIO_M3_ULTRA_256, // 671B at 8.5bpw against 256 GB of unified memory.
      runtime: MLX,
    });

    for (const verdict of verdicts.values()) {
      expect(verdict.fitness).toBe('fail');
    }
    expect(verdicts.get('batch')?.reason).toMatch(/does not fit/i);
  });

  it('reports long-context against what actually fits, not what the model claims', () => {
    // Same model, same card; the only difference is how many caches share the device.
    const alone = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency: 1 });
    const crowded = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency: 32 });

    const rank = { good: 2, tight: 1, fail: 0 };
    expect(rank[crowded.get('long-context')!.fitness]).toBeLessThanOrEqual(
      rank[alone.get('long-context')!.fitness]
    );
  });

  it('asks for concurrency before judging multi-user serving', () => {
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency: 1 });
    expect(verdicts.get('serving')?.reason).toMatch(/concurrency above 1/i);
  });
});

/**
 * Two ways the context limit can mislead a verdict, both found in review.
 */
describe('context limits and workload fit', () => {
  it('refuses RAG when its own 32K prompt has nowhere to live', () => {
    // Built directly rather than through a scenario: the case is a rig that *runs* while having
    // room for only a small context, and a real configuration tight enough to produce it is
    // usually `impossible` outright, which the top-level gate catches first for a different
    // reason. Prefill here is deliberately fast, so only the fit can be what fails it.
    const base = evaluate({
      model: LLAMA_31_8B,
      quant: getQuant('q4_k_m'),
      usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
      rig: { device: RTX_5090, count: 1 },
      runtime: LLAMA_CPP,
    });

    const verdicts = new Map(
      judgeWorkloads({
        placement: base.placement,
        decode: base.decode,
        usage: { contextTokens: 4096, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 4096,
        runnableContextTokens: 4096, // Far short of the 32K a RAG query sends.
        evaluateAt: (promptTokens, contextTokens) => {
          const e = evaluate({
            model: LLAMA_31_8B,
            quant: getQuant('q4_k_m'),
            usage: { contextTokens, concurrency: 1, promptTokens, kvPrecision: 'fp16' },
            rig: { device: RTX_5090, count: 1 },
            runtime: LLAMA_CPP,
          });
          return { decode: e.decode, prefill: e.prefill };
        },
      }).map((v) => [v.workload.id, v])
    );

    expect(verdicts.get('rag')!.fitness).toBe('fail');
    expect(verdicts.get('rag')!.reason).toMatch(/not enough for the 32K/i);
  });

  /**
   * `maxContextThatFits` requires a fully resident placement, so it is zero for *any* offloaded
   * configuration — even one whose KV would comfortably hold 128K once the weights are on the
   * host. Grading long-context on that figure reported "caps out at 0" for a working rig.
   */
  it('grades long-context on what can run, not only on what stays resident', () => {
    const evaluation = evaluate({
      model: DEEPSEEK_V3,
      quant: getQuant('q4_k_m'),
      usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
      rig: { device: RTX_5090, count: 1 },
      runtime: LLAMA_CPP,
    });

    // Weights spill, so the resident limit collapses...
    expect(evaluation.placement.offloadFraction).toBeGreaterThan(0);
    expect(evaluation.maxContextTokens).toBe(0);
    // ...while the runnable one reflects the KV that genuinely fits.
    expect(evaluation.runnableContextTokens).toBeGreaterThan(0);
  });
});

/**
 * Context holds the prompt *and* the generation, and the reasons have to stay internally
 * consistent when a capacity figure lands between round numbers.
 */
describe('context accounting in the verdicts', () => {
  it('refuses a workload whose prompt would fill the window exactly', () => {
    // Mistral Small and Mixtral cap at 32,768 — the same length as the RAG archetype's prompt,
    // leaving no position to answer from.
    const verdicts = new Map(
      judgeWorkloads({
        placement: {
          ...evaluate({
            model: LLAMA_31_8B,
            quant: getQuant('q4_k_m'),
            usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
            rig: { device: RTX_5090, count: 1 },
            runtime: LLAMA_CPP,
          }).placement,
        },
        decode: {
          perUserTokensPerSec: 60,
          aggregateTokensPerSec: 60,
          weightReadBytes: 1,
          kvReadBytes: 1,
          kvBound: false,
        },
        usage: { contextTokens: 32768, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 32768,
        runnableContextTokens: 32768,
        evaluateAt: () => ({
          decode: {
            perUserTokensPerSec: 60,
            aggregateTokensPerSec: 60,
            weightReadBytes: 1,
            kvReadBytes: 1,
            kvBound: false,
          },
          prefill: {
            ttftSeconds: 1,
            prefillTokensPerSec: 5000,
            linearFlops: 1,
            attentionFlops: 1,
            attentionBound: false,
          },
        }),
      }).map((v) => [v.workload.id, v])
    );

    expect(verdicts.get('rag')!.fitness).toBe('fail');
  });

  it('never rounds a failing capacity up into looking sufficient', () => {
    const verdicts = new Map(
      judgeWorkloads({
        placement: evaluate({
          model: LLAMA_31_8B,
          quant: getQuant('q4_k_m'),
          usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
          rig: { device: RTX_5090, count: 1 },
          runtime: LLAMA_CPP,
        }).placement,
        decode: {
          perUserTokensPerSec: 60,
          aggregateTokensPerSec: 60,
          weightReadBytes: 1,
          kvReadBytes: 1,
          kvBound: false,
        },
        usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 32700,
        // Between 32,256 and 32,767: rounds to "32K" and would read "Only 32K ... not enough
        // for the 32K document".
        runnableContextTokens: 32700,
        evaluateAt: () => ({
          decode: {
            perUserTokensPerSec: 60,
            aggregateTokensPerSec: 60,
            weightReadBytes: 1,
            kvReadBytes: 1,
            kvBound: false,
          },
          prefill: {
            ttftSeconds: 1,
            prefillTokensPerSec: 5000,
            linearFlops: 1,
            attentionFlops: 1,
            attentionBound: false,
          },
        }),
      }).map((v) => [v.workload.id, v])
    );

    const reason = verdicts.get('rag')!.reason;
    expect(reason).toMatch(/not enough for the 32K/);
    expect(reason).not.toMatch(/Only 32K of context fits/);
  });
});

/**
 * The four ways this layer had let a verdict disagree with its own evidence. Each was found one
 * neighbour over from a fix, so these assert the *class* rather than the instance.
 */
describe('a verdict never contradicts the numbers behind it', () => {
  const stub = (perUserTokensPerSec: number, ttftSeconds: number) => ({
    decode: {
      perUserTokensPerSec,
      aggregateTokensPerSec: perUserTokensPerSec,
      weightReadBytes: 1,
      kvReadBytes: 1,
      kvBound: false,
    },
    prefill: {
      ttftSeconds,
      prefillTokensPerSec: 5000,
      linearFlops: 1,
      attentionFlops: 1,
      attentionBound: false,
    },
  });

  const judged = (runnableContextTokens: number, perUser = 60, ttft = 0.2) =>
    new Map(
      judgeWorkloads({
        placement: evaluate({
          model: LLAMA_31_8B,
          quant: getQuant('q4_k_m'),
          usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
          rig: { device: RTX_5090, count: 1 },
          runtime: LLAMA_CPP,
        }).placement,
        decode: stub(perUser, ttft).decode,
        usage: { contextTokens: 512, concurrency: 1, promptTokens: 512, kvPrecision: 'fp16' },
        maxContextTokens: runnableContextTokens,
        runnableContextTokens,
        evaluateAt: () => stub(perUser, ttft),
      }).map((v) => [v.workload.id, v])
    );

  it('fails every archetype whose own prompt cannot fit, however fast it is', () => {
    // 813 tokens of runnable context: not even one chat turn, at any speed.
    const verdicts = judged(813, 200, 0.05);

    for (const id of ['chat', 'completion', 'agent', 'rag', 'long-context']) {
      expect(verdicts.get(id)!.fitness).toBe('fail');
    }
  });

  it('never prints a failing measurement as the threshold it missed', () => {
    // 14.5 tok/s fails the agent's 15 minimum; rounding would show "15".
    const reason = judged(200_000, 14.506, 1).get('agent')!.reason;

    expect(judged(200_000, 14.506, 1).get('agent')!.fitness).not.toBe('good');
    expect(reason).not.toMatch(/\b15 tok\/s/);
    expect(reason).toMatch(/\b14/);
  });

  it('uses one boundary for a condition and for the reason that explains it', () => {
    // Just under the tight threshold of 65536 + allowance: must fail *and* say why.
    const verdicts = judged(65_948);
    expect(verdicts.get('long-context')!.fitness).toBe('fail');
    expect(verdicts.get('long-context')!.reason).toMatch(/short of the 128K/);
  });
});
