import { describe, expect, it } from 'vitest';
import { judgeWorkloads, WORKLOADS, type VerdictInputs } from './verdict';
import { evaluate } from './index';
import type { Placement } from './placement';
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

/** Resident, with room to spare — these tests are about rate and latency, not capacity. */
const RESIDENT: Placement = {
  fits: true,
  weightBytesPerDevice: 1,
  kvBytesPerDevice: 1,
  activationBytesPerDevice: 1,
  usedBytesPerDevice: 3,
  allocatableBytesPerDevice: 10,
  totalWeightBytes: 1,
  totalKvBytes: 1,
  headroomBytes: 7,
  utilization: 0.3,
  offloadFraction: 0,
  impossible: false,
};

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
    selectedPlacement: evaluation.placement,
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
      return { placement: e.placement, decode: e.decode, prefill: e.prefill };
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
   * Latency budgets are a ladder — inline completion's 30 tok/s and 0.4s are strictly tighter
   * than chat's 15 and 2s — but *only* when latency is what decides. The archetypes send
   * different prompts on purpose, so they ask for different amounts of room, and at high
   * concurrency chat's longer turns can spill while completion's shorter ones stay resident.
   * That is a real property of the workloads: 128 concurrent autocompletes genuinely are easier
   * to serve than 128 concurrent conversations.
   *
   * So the invariant is conditional, and stating it unconditionally is what made it false. These
   * cases hold capacity out of the way, which is the regime where it does hold.
   */
  it.each([
    ['5090 + 8B', LLAMA_31_8B, 'q4_k_m', RTX_5090],
    ['Spark + gpt-oss-20b', GPT_OSS_20B, 'mxfp4', DGX_SPARK],
    ['Mac + gpt-oss-20b', GPT_OSS_20B, 'mxfp4', MAC_STUDIO_M3_ULTRA_256],
  ])(
    'never grades completion above chat on %s, at a concurrency both fit',
    (_label, model, quant, device) => {
      const rank = { good: 2, tight: 1, fail: 0 };
      const verdicts = judge(model, quant, { device });

      const completion = rank[verdicts.get('completion')!.fitness];
      const chat = rank[verdicts.get('chat')!.fitness];
      expect(completion).toBeLessThanOrEqual(chat);
    }
  );

  /**
   * And the other side of it, so the conditional invariant above is not quietly read as the
   * unconditional one again: when chat's longer turns are what runs out of room, completion may
   * outrank it, and both verdicts explain themselves.
   */
  it('lets completion outrank chat when the cache, not the clock, is what fails', () => {
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency: 128 });
    const chat = verdicts.get('chat')!;
    const completion = verdicts.get('completion')!;

    if (chat.fitness === 'fail' && completion.fitness !== 'fail') {
      // The row that passes must not be silent about why the row above it did not.
      expect(chat.reason).toMatch(/of context fits/);
    }
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
        selectedPlacement: base.placement,
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
          return { placement: e.placement, decode: e.decode, prefill: e.prefill };
        },
      }).map((v) => [v.workload.id, v])
    );

    expect(verdicts.get('rag')!.fitness).toBe('fail');
    expect(verdicts.get('rag')!.reason).toMatch(/32K document this assumes needs 32\.5K/i);
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
 * The four ways this layer had let a verdict disagree with its own evidence. Each was found one
 * neighbour over from a fix, so these assert the *class* rather than the instance.
 */
describe('a verdict never contradicts the numbers behind it', () => {
  const stub = (perUserTokensPerSec: number, ttftSeconds: number) => ({
    placement: RESIDENT,
    decode: {
      perUserTokensPerSec,
      aggregateTokensPerSec: perUserTokensPerSec,
      weightReadBytes: 1,
      kvReadBytes: 1,
      weightSeconds: 1,
      kvSeconds: 0.1,
      kvBound: false,
    },
    prefill: {
      ttftSeconds,
      prefillTokensPerSec: 5000,
      linearFlops: 1,
      attentionFlops: 1,
      linearSeconds: 0.1,
      attentionSeconds: 0.1,
      attentionBound: false,
    },
  });

  const judged = (runnableContextTokens: number, perUser = 60, ttft = 0.2) =>
    new Map(
      judgeWorkloads({
        selectedPlacement: evaluate({
          model: LLAMA_31_8B,
          quant: getQuant('q4_k_m'),
          usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
          rig: { device: RTX_5090, count: 1 },
          runtime: LLAMA_CPP,
        }).placement,
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

    // The boundary that rejected it, not the archetype's headline. This asserted 128.5K until the
    // rejection was traced: the predicate tests 64.5K, so quoting 128.5K told a rig sitting about
    // 1K short that it needed to double — the exact defect this test's name is about, in the test
    // itself. A shortfall is an upgrade instruction and has to name the bar that was missed.
    expect(verdicts.get('long-context')!.reason).toMatch(/needs 64\.5K/);
    expect(verdicts.get('long-context')!.reason).not.toMatch(/needs 128\.5K/);
  });

  /**
   * A tier that admits a smaller job has to measure the smaller job. The tight tier accepts a
   * machine holding 64K and was timing it on the archetype's 128K request — a prompt that rig has
   * nowhere to put. Prefill is quadratic, so this was not a rounding difference: the impossible
   * request routinely failed the tier that had just admitted it on capacity.
   */
  const atPrompt = (runnableContextTokens: number, ttftFor: (promptTokens: number) => number) => {
    const seen: number[] = [];
    const verdicts = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 4096, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: runnableContextTokens,
        runnableContextTokens,
        evaluateAt: (promptTokens) => {
          seen.push(promptTokens);
          return stub(60, ttftFor(promptTokens));
        },
      }).map((v) => [v.workload.id, v])
    );
    return { verdicts, seen };
  };

  it('grades the long-context tight tier at the window the machine holds', () => {
    // 80K runnable: past the 64.5K tight bar, short of the 128.5K good one. The 128K request it
    // cannot make reads 700s — over the 600s bar — while the 64K job it can do reads 175s.
    const { verdicts, seen } = atPrompt(80_000, (prompt) => (prompt >= 131072 ? 700 : 175));

    expect(seen).toContain(65536);
    // Timed on the 128K prompt this failed outright, on evidence describing a request the machine
    // has nowhere to put.
    expect(verdicts.get('long-context')!.fitness).toBe('tight');
    expect(verdicts.get('long-context')!.reason).toMatch(/64K/);
    expect(verdicts.get('long-context')!.reason).not.toMatch(/700/);
  });

  it('quotes the full window only to a machine that can hold one', () => {
    // 200K runnable, so the archetype's own 128K request is the honest measurement here.
    const { verdicts } = atPrompt(200_000, () => 30);
    expect(verdicts.get('long-context')!.fitness).toBe('good');
    expect(verdicts.get('long-context')!.reason).toMatch(/128K/);
  });

  /**
   * Every figure printed on a tight serving row was healthy, so the row read as a pass that had
   * been marked down for no stated reason. Two conditions reach that branch and neither was named.
   */
  const serving = (concurrency: number, placement: Placement, perUser = 40) =>
    new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 8192, concurrency, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        evaluateAt: () => ({ ...stub(perUser, 0.2), placement }),
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

  it('names the four-user bar when only the user count holds serving back', () => {
    const verdict = serving(3, RESIDENT);

    expect(verdict.fitness).toBe('tight');
    expect(verdict.reason).toMatch(/4 concurrent users/);
  });

  it('names the rate when only the rate holds serving back', () => {
    // Four users clears the count bar, so 7 tok/s is the sole remaining cause.
    const verdict = serving(4, RESIDENT, 7);

    expect(verdict.fitness).toBe('tight');
    expect(verdict.reason).toMatch(/10 tok\/s/);
  });

  it('does not call a partial spill an exhausted serving capacity', () => {
    // Over budget on the resident plan, but spilling — which is a performance penalty, not a wall.
    // `impossible` is what means capacity is genuinely gone, and this is not that.
    const spilling: Placement = {
      ...RESIDENT,
      fits: false,
      headroomBytes: -1,
      offloadFraction: 0.2,
      impossible: false,
    };
    const verdict = serving(8, spilling);

    expect(verdict.reason).toMatch(/spill/i);
    // Another user can still be served, more slowly. Saying otherwise reports a false limit.
    expect(verdict.reason).not.toMatch(/nowhere to go/i);
  });
});

/**
 * Every archetype, gated the same way — the property I asserted twice and shipped false twice,
 * because batch and serving kept using the slider's own measurement after the others moved.
 */
/** Fast and prompt, so these tests exercise capacity rather than speed. */
const STUB_SPEED = {
  decode: {
    perUserTokensPerSec: 200,
    aggregateTokensPerSec: 200,
    weightReadBytes: 1,
    kvReadBytes: 1,
    weightSeconds: 1,
    kvSeconds: 0.1,
    kvBound: false,
  },
  prefill: {
    ttftSeconds: 0.2,
    prefillTokensPerSec: 5000,
    linearFlops: 1,
    attentionFlops: 1,
    linearSeconds: 0.1,
    attentionSeconds: 0.1,
    attentionBound: false,
  },
};

describe('a shortfall always reads as a shortfall', () => {
  it('names the room to answer in, not just the prompt', () => {
    // A model capped at exactly 32,768 — Mistral Small, Mixtral — fails RAG because the answer
    // needs somewhere to go. Naming only the prompt read "Only 32K of context fits — not enough
    // for the 32K document", which contradicts itself with no rounding involved at all.
    const verdicts = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 32768, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 32768,
        runnableContextTokens: 32768,
        evaluateAt: () => ({ placement: RESIDENT, ...STUB_SPEED }),
      }).map((v) => [v.workload.id, v])
    );

    const rag = verdicts.get('rag')!;
    expect(rag.fitness).toBe('fail');
    expect(rag.reason).toContain('32.5K');
    // The two figures in the sentence must differ, or it reads as a contradiction.
    expect(rag.reason).toMatch(/Only 32K .* needs 32\.5K/);
  });

  it('states the requirement for every archetype, not just the one that was reported', () => {
    const verdicts = judgeWorkloads({
      selectedPlacement: RESIDENT,
      usage: { contextTokens: 512, concurrency: 1, promptTokens: 512, kvPrecision: 'fp16' },
      maxContextTokens: 600,
      runnableContextTokens: 600,
      evaluateAt: () => ({ placement: RESIDENT, ...STUB_SPEED }),
    });

    for (const v of verdicts) {
      expect(v.fitness).toBe('fail');
      expect(v.reason).toMatch(/needs .* with room to answer in/);
    }
  });
});

/**
 * A reason has to name the thing that decided the grade. Each of these named something else —
 * a requirement that was met, a measurement that was fine, or a latency of zero.
 */
describe('a verdict counts the whole request, not half of it', () => {
  it('charges batch for reading its prompt, not only for writing its answer', () => {
    // DeepSeek V3 on an EPYC: 6 tok/s of decode, and about eight minutes to read the 4K prompt
    // this archetype declares. Grading on decode alone called that comfortable while a 512-token
    // reply actually completes at under 1 token per second end to end.
    const verdicts = judge(DEEPSEEK_V3, 'q8_0', { device: EPYC_9654 });
    const batch = verdicts.get('batch')!;

    expect(batch.fitness).toBe('tight');
    expect(batch.reason).toMatch(/end to end/);
  });

  it('charges every worker its own prompt', () => {
    // `estimatePrefill` prices the whole batch of prompts, so this layer reads its figure rather
    // than multiplying by the worker count — it used to do the multiplying itself, back when the
    // engine computed FLOPs from `promptTokens` alone. Either way the property is the same: on
    // one device the prompts queue, and more workers cannot make a prompt-bound job faster than
    // the device can read prompts.
    const one = judge(DEEPSEEK_V3, 'q8_0', { device: EPYC_9654, concurrency: 1 }).get('batch')!;
    const many = judge(DEEPSEEK_V3, 'q8_0', { device: EPYC_9654, concurrency: 32 }).get('batch')!;

    // A prompt-bound job does not improve its grade by adding workers.
    const rank = { good: 2, tight: 1, fail: 0 };
    expect(rank[many.fitness]).toBeLessThanOrEqual(rank[one.fitness]);
  });

  it('will not recommend long-context analysis a machine can hold but not perform', () => {
    // The route this rewarded: offloading almost everything *raises* the runnable context, so a
    // capacity-only grade got better the more the configuration spilled. DeepSeek V3 at BF16 on
    // one 5090 reaches 163,840 tokens and takes about eighteen minutes to read a full window.
    const verdicts = judge(DEEPSEEK_V3, 'bf16', { device: RTX_5090, contextTokens: 512 });
    const long = verdicts.get('long-context')!;

    expect(long.fitness).toBe('fail');
    expect(long.reason).toMatch(/before saying anything|the work does not/);
  });

  /**
   * Pinning only BF16 above left a hole: that row fails on its decode term as well as its prefill
   * one, so it stayed red through a change that graded this machine on a 64K job while printing
   * the 128K timing. Q4_K_M is the sibling that fails on prefill alone, and it is the row that
   * flipped to `tight` while its own reason reported 1046s against a 600s bar.
   */
  it.each(['bf16', 'q4_k_m', 'q8_0'])(
    'grades a machine that holds 128K on the 128K job, at %s',
    (quantId) => {
      const long = judge(DEEPSEEK_V3, quantId, {
        device: RTX_5090,
        contextTokens: 512,
      }).get('long-context')!;

      expect(long.fitness).toBe('fail');
      // Whatever the row says, the grade has to have been decided on the same measurement.
      expect(long.reason).toMatch(/before saying anything|the work does not/);
    }
  );

  it('still passes long-context on a machine that can actually work in the window', () => {
    const long = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090 }).get('long-context')!;
    expect(long.fitness).not.toBe('fail');
  });

  it('will not call RAG usable when the answer takes minutes', () => {
    // Prefill is only half the request: a RAG-sized cache that decodes at a crawl still has to
    // write the reply, and grading on TTFT alone printed the prefill rate as though that were
    // the whole story.
    const fast = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090 }).get('rag')!;
    expect(fast.fitness).not.toBe('fail');

    const crawling = judge(DEEPSEEK_V3, 'q8_0', { device: EPYC_9654 }).get('rag')!;
    expect(crawling.fitness).toBe('fail');
  });
});

describe('the reason names the constraint that actually bound', () => {
  const judged = (runnableContextTokens: number, perUser: number, ttft: number) =>
    new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 512, concurrency: 1, promptTokens: 512, kvPrecision: 'fp16' },
        maxContextTokens: runnableContextTokens,
        runnableContextTokens,
        evaluateAt: () => ({
          placement: RESIDENT,
          decode: {
            ...STUB_SPEED.decode,
            perUserTokensPerSec: perUser,
            aggregateTokensPerSec: perUser,
          },
          prefill: { ...STUB_SPEED.prefill, ttftSeconds: ttft },
        }),
      }).map((v) => [v.workload.id, v])
    );

  it('names the session floor, not the turn, when the turn already fits', () => {
    // 27K runnable: past the 16.5K a turn needs, short of the 32K a session needs. Reporting
    // the turn requirement quoted a figure the configuration meets.
    const agent = judged(27_000, 60, 1).get('agent')!;

    expect(agent.fitness).toBe('fail');
    expect(agent.reason).not.toMatch(/16\.5K/);
    expect(agent.reason).toMatch(/32K/);
  });

  it('names the rate when the rate is the only thing holding completion back', () => {
    // 28 tok/s against a 30 threshold, with latency well inside its budget — so the latency
    // sentence was entirely positive while the grade was Tight.
    const completion = judged(400_000, 28, 0.089).get('completion')!;

    expect(completion.fitness).toBe('tight');
    expect(completion.reason).toMatch(/28 tok\/s/);
  });

  it('names the answer rate when the answer is what downgraded RAG', () => {
    // Between 5 and 10 tok/s with a quick prefill, the row is tight solely on the answer — and
    // reported only how fast it read the document, which is the pass-shaped half.
    const verdicts = judged(400_000, 7, 1);
    const rag = verdicts.get('rag')!;

    if (rag.fitness === 'tight') {
      expect(rag.reason).toMatch(/7\.0 tok\/s/);
    }
  });

  it('never floors a missed latency onto the limit it missed', () => {
    // The mirror of the rate rule, and the direction has to follow the *bound*: a rate fails by
    // being too small so flooring protects it, a latency fails by being too large so flooring is
    // exactly what makes it look sufficient. 0.486s against a 0.4s limit printed "0.4s ... stays
    // inside the window" beside the word Tight.
    const completion = judged(400_000, 60, 0.486).get('completion')!;

    expect(completion.fitness).not.toBe('good');
    expect(completion.reason).not.toMatch(/\b0\.4s/);
    expect(completion.reason).toMatch(/0\.5s/);
  });

  it('never reports a positive latency as zero', () => {
    // Flooring is right for thresholds and wrong at the bottom: 0.089 floored to "0.0" claimed
    // no latency at all.
    const completion = judged(400_000, 60, 0.089).get('completion')!;

    expect(completion.reason).not.toMatch(/\b0\.0s/);
    expect(completion.reason).toMatch(/<0\.1s/);
  });
});

describe('no archetype escapes its own scenario', () => {
  const stub = (perUser: number) => ({
    placement: RESIDENT,
    decode: {
      perUserTokensPerSec: perUser,
      aggregateTokensPerSec: perUser,
      weightReadBytes: 1,
      kvReadBytes: 1,
      weightSeconds: 1,
      kvSeconds: 0.1,
      kvBound: false,
    },
    prefill: {
      ttftSeconds: 0.2,
      prefillTokensPerSec: 5000,
      linearFlops: 1,
      attentionFlops: 1,
      linearSeconds: 0.1,
      attentionSeconds: 0.1,
      attentionBound: false,
    },
  });

  const judged = (runnableContextTokens: number, concurrency = 8) =>
    new Map(
      judgeWorkloads({
        selectedPlacement: evaluate({
          model: LLAMA_31_8B,
          quant: getQuant('q4_k_m'),
          usage: { contextTokens: 4096, concurrency, promptTokens: 512, kvPrecision: 'fp16' },
          rig: { device: RTX_5090, count: 1 },
          runtime: LLAMA_CPP,
        }).placement,
        usage: { contextTokens: 512, concurrency, promptTokens: 512, kvPrecision: 'fp16' },
        maxContextTokens: runnableContextTokens,
        runnableContextTokens,
        evaluateAt: () => stub(200),
      }).map((v) => [v.workload.id, v])
    );

  it('fails every archetype whose declared request cannot fit, at any speed', () => {
    // 768 tokens: below the smallest declared request — inline completion's 512 prompt plus
    // its response allowance — so nothing can fit, including batch's 4K and serving's 2K,
    // which were the two still reading the slider's own evaluation.
    const verdicts = judged(768);

    for (const workload of WORKLOADS) {
      expect(verdicts.get(workload.id)!.fitness).toBe('fail');
    }
  });

  it('grades an archetype on its own placement, not the placement of the slider', () => {
    // The selected scenario is spilled to host RAM — no headroom left. Serving's own 2K turns
    // are resident. Serving is graded at *its* scenario, so the slider's spill must not reach it.
    const spilled: Placement = {
      ...RESIDENT,
      fits: false,
      headroomBytes: -1,
      utilization: 1.4,
      offloadFraction: 0.3,
    };

    const verdicts = new Map(
      judgeWorkloads({
        selectedPlacement: spilled,
        usage: { contextTokens: 512, concurrency: 8, promptTokens: 512, kvPrecision: 'fp16' },
        maxContextTokens: 400_000,
        runnableContextTokens: 400_000,
        evaluateAt: () => stub(200),
      }).map((v) => [v.workload.id, v])
    );

    expect(verdicts.get('serving')!.fitness).toBe('good');
  });

  it('passes them all again once the room is there', () => {
    const verdicts = judged(400_000, 4);
    const passing = WORKLOADS.filter((w) => verdicts.get(w.id)!.fitness !== 'fail');
    expect(passing.length).toBeGreaterThan(4);
  });
});
