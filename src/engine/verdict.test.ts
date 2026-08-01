import { describe, expect, it } from 'vitest';
import {
  judgeWorkloads,
  WORKLOADS,
  WORKLOAD_BARS,
  type Fitness,
  type VerdictInputs,
} from './verdict';
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
  RTX_4090,
  RTX_5090,
  VLLM,
  MAC_STUDIO_M3_ULTRA_256,
} from './fixtures';
import { getQuant } from '@/data/quants';
// The monotonicity sweep is about the *model*, not about a rig, so it runs the shipped catalog
// rather than fixtures — a claim checked only on the hardware it was derived from is not checked.
import { DEVICES, MODELS, getDevice } from '@/data/catalog';
import { RUNTIMES, runtimeDrives } from '@/data/runtimes';

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
  // Cache and activations, which is what `impossible` weighs against the ceiling. Well under 10.
  floorBytesPerDevice: 2,
  offloadFraction: 0,
  impossible: false,
};

/** The same rig, asked for more than it can hold: cache and activations alone are over the ceiling. */
const OVER: Placement = {
  ...RESIDENT,
  fits: false,
  headroomBytes: -1,
  utilization: 1.4,
  impossible: true,
};

/**
 * A stub engine that refuses what it says it cannot hold.
 *
 * `evaluateAt: () => ({ placement: RESIDENT, ...STUB_SPEED })` describes a machine that reports a
 * runnable ceiling and then returns a resident placement for any request above it — which no engine
 * does, and which mattered the moment a capacity check stopped reading `runnableContextTokens`
 * (#96). Serving's fit is now the placement at its own turn and its own tier's user count, so a stub
 * that always fits reports a rig holding four served turns while its inputs say it holds none.
 *
 * Given the same ceiling the caller declares, this refuses above it exactly as `planPlacement` would,
 * which makes the two halves of the fixture agree instead of merely coexisting.
 */
const stubEngine =
  (runnableContextTokens: number) => (_promptTokens: number, contextTokens: number) => ({
    placement: contextTokens > runnableContextTokens ? OVER : RESIDENT,
    ...STUB_SPEED,
  });

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
    evaluateAt: (promptTokens, contextTokens, cachedPrefixTokens, concurrency) => {
      const e = evaluate({
        model,
        quant: getQuant(quantId),
        usage: { ...usage, promptTokens, contextTokens, cachedPrefixTokens, concurrency },
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

/**
 * The three grades as an order.
 *
 * Four copies of `{ good: 2, tight: 1, fail: 0 }` are one function, and it throws on anything it
 * does not know rather than returning `undefined` — an `undefined` on either side of
 * `toBeLessThanOrEqual` passes, which is how a fourth grade would slip past every ordering
 * assertion in this file. That guard was written for the `unmeasured` arm and outlives it (#96):
 * `Record<Fitness, …>` makes the coverage a compile-time claim and this keeps the runtime one.
 */
const RANK: Record<Fitness, number> = { good: 2, tight: 1, fail: 0 };
const rankOf = (fitness: Fitness) => {
  const rank = RANK[fitness];
  if (rank === undefined) throw new Error(`no rank for the grade "${fitness}"`);
  return rank;
};

describe('workload verdicts', () => {
  it('grades every archetype, every time', () => {
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090 });
    expect(verdicts.size).toBe(WORKLOADS.length);
    for (const verdict of verdicts.values()) {
      expect(verdict.reason).not.toBe('');
      expect(['good', 'tight', 'fail', 'unmeasured']).toContain(verdict.fitness);
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
    // A Spark prefills slowly. Whatever the slider says, an agent can never outrank chat.
    for (const promptTokens of [512, 8192, 65536]) {
      const verdicts = judge(GPT_OSS_20B, 'mxfp4', { device: DGX_SPARK, promptTokens });
      expect(rankOf(verdicts.get('agent')!.fitness)).toBeLessThanOrEqual(
        rankOf(verdicts.get('chat')!.fitness)
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
      const verdicts = judge(model, quant, { device });

      const completion = rankOf(verdicts.get('completion')!.fitness);
      const chat = rankOf(verdicts.get('chat')!.fitness);
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

    expect(rankOf(crowded.get('long-context')!.fitness)).toBeLessThanOrEqual(
      rankOf(alone.get('long-context')!.fitness)
    );
  });

  it('does not ask the reader for concurrency before judging multi-user serving', () => {
    // It used to: at one user this row printed "set concurrency above 1 to see whether this holds
    // several", which is an instruction rather than a verdict and the only sentence in the file
    // naming no measurement. Since #96 the row is graded at its own four users, so the question is
    // always answered and the instruction has nothing to attach to.
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency: 1 });
    const serving = verdicts.get('serving')!;
    expect(serving.reason).not.toMatch(/concurrency/i);
    expect(serving.fitness).toBe('good');
  });
});

/**
 * Serving answers the archetype's own question, at the archetype's own concurrency
 * ([#96](https://github.com/MrZoller/bench/issues/96)).
 *
 * This is the third and last fix at one root. Serving was the only archetype taking its defining
 * parameter from the slider rather than declaring it, and that asymmetry produced a red `fail` for
 * an unconfigured slider (#75), an ungraded fourth state that then hid proved failures (#94), and a
 * capacity check taken at a concurrency the grade was not about. Grading each tier at its own user
 * count — four and two, exactly as `long-context` grades at a full window and at the reduced one its
 * tight tier admits — removes all three and the fourth state with them.
 *
 * The invariant that survives all of it, and the one worth keeping green: **a row is ungraded only
 * when nothing about it has been measured**, and since every archetype now declares its own
 * scenario there is no such row. `Fitness` has three arms.
 */
describe('serving is graded at its own concurrency, not the slider’s', () => {
  /**
   * The defect at the root, stated as an identity rather than as a grade.
   *
   * Nothing about the machine changes when the reader drags a slider, so nothing about "can this
   * machine serve several people" may change either. That one assertion covers #75 (which was this
   * row reading `fail` at one user and `good` at four), #94's ungraded state, and the capacity
   * symptom below — all three were the verdict moving when only the slider had.
   */
  it('returns the same verdict at every concurrency, because the machine is the same', () => {
    const at = (concurrency: number) =>
      judge(LLAMA_31_8B, 'q4_k_m', {
        device: RTX_5090,
        concurrency,
        // A turn-sized window, so the reader's own scenario still loads at 128 users and the
        // panel-wide refusal below is not what this is measuring.
        contextTokens: 2048,
      }).get('serving')!;
    const one = at(1);

    // The precondition: a rig this row has something positive to say about, so "identical" is not
    // being satisfied by seven copies of one refusal.
    expect(one.fitness).toBe('good');
    // Up to 32, which is where this rig stops loading the reader's own scenario at all. Past that
    // the panel refuses every row together and serving goes with them — the sibling test below.
    for (const concurrency of [2, 3, 4, 8, 32]) {
      const other = at(concurrency);
      expect(other.fitness, `serving changed grade at ${concurrency} users`).toBe(one.fitness);
      expect(other.reason, `serving changed its reason at ${concurrency} users`).toBe(one.reason);
    }
  });

  /**
   * The one thing that still moves it, named so the identity above is not read as absolute.
   *
   * `judgeWorkloads` refuses all seven rows when the *selected* placement cannot load — one sentence
   * above the list rather than seven identical ones — and that is a fact about the reader's own
   * scenario, so it does depend on the slider.
   *
   * Serving is refused with the rest, and since #96 that is a decision rather than a corollary: it is
   * graded at four users, so a scenario refused at 128 could well be servable. It joins the collapse
   * because the reader's own configuration not loading is the first thing they need to know, and
   * because a row with a reason of its own is what the collapse cannot survive. The argument is at
   * the refusal in `verdict.ts`; this is the assertion that keeps the behaviour with it.
   */
  it('is refused with every other row when the reader’s own scenario cannot load', () => {
    const crowded = judge(LLAMA_31_8B, 'q4_k_m', {
      device: RTX_5090,
      concurrency: 128,
      contextTokens: 131072,
    });

    for (const verdict of crowded.values()) expect(verdict.fitness).toBe('fail');
    expect(new Set([...crowded.values()].map((v) => v.reason)).size).toBe(1);
  });

  it('names its own four users in the sentence, and never the slider’s count', () => {
    const serving = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency: 1 }).get(
      'serving'
    )!;

    // The scenario the grade came from, named — this file's rule that a sentence naming a scenario
    // has to name the one the estimate was called with.
    expect(serving.reason).toMatch(new RegExp(`\\b${WORKLOAD_BARS.serving.good.users} users\\b`));
    // And the two sentences the old slider-driven version produced, neither of which can be true of
    // a row that is never graded at the reader's setting.
    expect(serving.reason).not.toMatch(/Not measured/i);
    expect(serving.reason).not.toMatch(/set concurrency/i);
  });

  it('grades every archetype, at every concurrency', () => {
    // The invariant #96 leaves behind. `Fitness` is three arms and `judge` cannot emit a fourth, so
    // this is the runtime half of a claim the compiler already makes — worth having because the way
    // it broke before was a *value*, not a type: a row carrying a grade nothing had measured.
    for (const concurrency of [1, 2, 4, 128]) {
      const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090, concurrency });
      expect(verdicts.size).toBe(7);
      for (const verdict of verdicts.values()) {
        expect(['good', 'tight', 'fail']).toContain(verdict.fitness);
        expect(verdict.reason.length).toBeGreaterThan(0);
      }
    }
  });

  /**
   * **The capacity symptom, which is what could not be patched onto #94** — its second finding, and
   * the narrowest of the three: `fits('serving')` read a runnable context computed at the *current*
   * concurrency, so a rig with room for one served turn and not two passed it, measured fast at one
   * user, and reported "Not measured" while two users deterministically fails on capacity.
   *
   * Two of 2,278 drivable combinations were affected and both are this machine: a 24 GiB Mac mini
   * runs Llama-3.1-8B at BF16 with 3,364 tokens of runnable context at one user and 1,680 at two,
   * against the 2,560 one served turn needs. Graded at the tight tier's own two users, the capacity
   * failure is the answer.
   */
  it('measures capacity at the tier’s users, not the reader’s', () => {
    const serving = judge(LLAMA_31_8B, 'bf16', {
      device: getDevice('mac-mini-m4-pro-24'),
      concurrency: 1,
      // The reader's own window is small enough to load, so the panel-wide refusal is not what
      // produces this verdict — the point is that one served turn fits here and two do not.
      contextTokens: 2048,
    }).get('serving')!;

    expect(serving.fitness).toBe('fail');
    /*
     * The whole sentence, not a substring, and that is the lesson from the draft this replaced: it
     * matched `/a turn each for 2 users/` and `/needs 2.5K/` and was green on a sentence that
     * disproved its own verdict — "Only 3.2K of context fits … needs 2.5K", where the 3.2K was the
     * one-user runnable context and 3.2K is more than 2.5K. Two true substrings, one false claim.
     * Both figures here come from the placement that actually refused, at two users.
     */
    expect(serving.reason).toBe(
      '2 users each holding 2.5K of context need 17 GiB per device against the 16 GiB each one ' +
        'can allocate, and this machine has no host tier to spill the weights to.'
    );
  });

  /**
   * And the two figures in that sentence have to make its verdict true, on every rig that reaches it
   * — which is the property the substring version could not see.
   */
  it('never states a serving shortfall whose own figures say it fits', () => {
    let reached = 0;
    for (const device of DEVICES) {
      // Two sizes, because an 8B at a 2.5K turn fits nearly everywhere and only one catalogued rig
      // reaches this sentence on it — a sweep that hits one case is a spot check wearing a loop.
      for (const [model, quantId] of [
        [LLAMA_31_8B, 'q4_k_m'],
        [LLAMA_31_8B, 'bf16'],
        [DEEPSEEK_V3, 'q4_k_m'],
      ] as const) {
        // A turn-sized window, so the panel-wide refusal is not what most of these rigs return —
        // that path grades all seven `fail` with one shared sentence and never reaches this one.
        const serving = judge(model, quantId, {
          device,
          concurrency: 1,
          contextTokens: 2048,
        }).get('serving')!;
        const stated = /need ([\d.]+) GiB[^,]*? against (?:the )?([\d.]+) GiB/.exec(serving.reason);
        if (!stated) continue;
        reached++;
        expect(
          Number(stated[1]),
          `${device.id}/${model.id}/${quantId} reports a shortfall of ${stated[1]} against ${stated[2]}`
        ).toBeGreaterThan(Number(stated[2]));
      }
    }
    /*
     * One is a real floor here, not a weak one, and the number is the issue's own measurement: the
     * band this sentence lives in is a rig whose *own* scenario loads while two served turns do not,
     * and #96 swept the catalog and found it on **2 of 2,278** drivable combinations. Anything
     * smaller than the band and the sweep is vacuous; anything larger and it would be describing a
     * different defect. What the loop buys over a single case is that it fails if the band moves.
     */
    expect(
      reached,
      'no rig reached the serving capacity sentence, so this proves nothing'
    ).toBeGreaterThan(0);
  });

  /**
   * And the good tier's shortfall may not claim a spill on a machine that cannot spill.
   *
   * `impossible` and negative headroom are different claims — this file says so about the predicate
   * and said the opposite in this sentence. When four served turns are over a unified-memory
   * machine's ceiling the placement is `impossible` *because* there is no host tier, and its headroom
   * is negative, so the clause fired and told the reader the weights were spilling to host RAM on the
   * one class of machine where nothing can. Found in review on #96.
   */
  it('never tells a machine with no host tier that its weights are spilling', () => {
    const overAtFour: Placement = { ...OVER, offloadFraction: 0 };
    const serving = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 2048, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        // Two served turns fit; four are over, and cannot offload — the unified-memory shape.
        evaluateAt: (_prompt, _context, _prefix, concurrency) => ({
          ...STUB_SPEED,
          placement: concurrency >= WORKLOAD_BARS.serving.good.users ? overAtFour : RESIDENT,
        }),
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

    expect(serving.fitness).toBe('tight');
    expect(serving.reason).toMatch(/holds a turn each for 2 users but not for the 4/);
    expect(serving.reason).not.toMatch(/spilling to host RAM/);
  });

  it('never tells a discrete GPU its weights are spilling when four users cannot run', () => {
    /*
     * The same defect one class of machine over, and the reason `offloadFraction > 0` was not the
     * fix (found in review, second round). `planPlacement` computes the spilled fraction *before*
     * deciding `impossible` from the non-offloadable floor, so a card whose cache and activations
     * alone are over its ceiling carries a positive fraction on a configuration that cannot run —
     * and the sentence contrasted spilling with "simply not fitting" for a workload that is, exactly,
     * not fitting.
     */
    const overOnFloor: Placement = { ...OVER, offloadFraction: 0.4 };
    const serving = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 2048, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        evaluateAt: (_prompt, _context, _prefix, concurrency) => ({
          ...STUB_SPEED,
          placement: concurrency >= WORKLOAD_BARS.serving.good.users ? overOnFloor : RESIDENT,
        }),
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

    expect(serving.fitness).toBe('tight');
    expect(serving.reason).toMatch(/holds a turn each for 2 users but not for the 4/);
    expect(serving.reason).not.toMatch(/spilling to host RAM/);
  });

  it('quotes no four-user figure at all when four users cannot run', () => {
    /*
     * The class the two gates above were a subset of (found in review, third round).
     * `estimateScenario` returns decode and prefill figures for an impossible placement — it prices
     * what was asked for, and `planPlacement` separately says it cannot be held — so the rate and
     * TTFT clauses quoted a speed the machine cannot deliver, in the same sentence as "it does not
     * hold four users".
     *
     * Driven with figures that miss *every* good-tier bar, so all four clauses would fire if none
     * were gated; only the capacity one may.
     */
    const overAtFour: Placement = { ...OVER, offloadFraction: 0.4 };
    const slow = {
      decode: { ...STUB_SPEED.decode, perUserTokensPerSec: 1, aggregateTokensPerSec: 1 },
      prefill: { ...STUB_SPEED.prefill, ttftSeconds: 90 },
    };
    const serving = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 2048, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        evaluateAt: (_prompt, _context, _prefix, concurrency) =>
          concurrency >= WORKLOAD_BARS.serving.good.users
            ? { ...slow, placement: overAtFour }
            : { ...STUB_SPEED, placement: RESIDENT },
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

    expect(serving.fitness).toBe('tight');
    expect(serving.reason).toMatch(/holds a turn each for 2 users but not for the 4/);
    // The three figures a placement that cannot exist has no business reporting.
    expect(serving.reason).not.toMatch(/tok\/s each at 4 users/);
    expect(serving.reason).not.toMatch(/to first token across 4 users/);
    expect(serving.reason).not.toMatch(/spilling to host RAM/);
  });

  it('says why a rig with no spare byte is only tight', () => {
    /*
     * `pass` requires `headroomBytes > 0`, so a resident placement using precisely its allocatable
     * memory is downgraded — deliberately, since a rig with nothing left cannot take the next user.
     * Every other good-tier bar passed and nothing spilled, so `shortOfGood` had no live item and
     * fell through to the *positive* fallback: a `tight` row printing three healthy figures and no
     * reason, which is the defect that builder exists to prevent, through the one branch it did not
     * cover (found in review).
     */
    const exactlyFull: Placement = { ...RESIDENT, headroomBytes: 0, utilization: 1 };
    const serving = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 2048, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        evaluateAt: () => ({ ...STUB_SPEED, placement: exactlyFull }),
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

    expect(serving.fitness).toBe('tight');
    expect(serving.reason).toMatch(/^Usable, but /);
    expect(serving.reason).toMatch(/every allocatable byte at four users/);
  });

  /**
   * The other half of the capacity sentence, which the catalog sweep above cannot reach.
   *
   * `impossible` fires two ways — the machine cannot offload at all, or the non-offloadable floor is
   * over the ceiling even after spilling every weight — and the honest figure differs by which. The
   * sweep finds only the first, because a discrete GPU whose *cache alone* will not hold two 2.5K
   * turns is not a machine in this catalog. Built rather than found, so both branches are pinned.
   */
  it('names the cache alone when that is what makes it impossible', () => {
    const floorOver: Placement = {
      ...OVER,
      // Cache and activations past the ceiling on their own, which is what `impossible` tests on a
      // rig that *can* offload — so the weights are not the reason and must not be the figure.
      floorBytesPerDevice: 12 * 1024 ** 3,
      usedBytesPerDevice: 40 * 1024 ** 3,
      allocatableBytesPerDevice: 10 * 1024 ** 3,
      offloadFraction: 1,
    };
    const serving = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 2048, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        evaluateAt: () => ({ ...STUB_SPEED, placement: floorOver }),
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

    expect(serving.fitness).toBe('fail');
    // The floor, not the 40 GiB `used` — which would attribute the refusal to weights the planner
    // would happily spill, and overstate the requirement fourfold.
    expect(serving.reason).toContain('12 GiB of cache and overhead per device');
    expect(serving.reason).not.toContain('40 GiB');
    expect(serving.reason).toContain('the 10 GiB each one can allocate');
  });

  /**
   * **A measured failure is a failure**, which is #94's first finding and the reason the ungraded
   * state could not simply be widened. 384 of 1,292 drivable combinations miss a tight bar at what
   * used to be "not measured", and no number of users recovers them — see the monotonicity sweep
   * below, which is the premise that makes the tight tier's two users enough.
   */
  it.each([
    {
      bar: 'rate',
      model: DEEPSEEK_V3,
      quant: 'bf16',
      device: RTX_5090,
      // 671B of BF16 weights against a 31 GiB ceiling: almost everything streams over the host bus.
      says: /tok\/s each at 2 users, under the 5 tok\/s/,
    },
    {
      bar: 'TTFT',
      model: LLAMA_31_8B,
      quant: 'bf16',
      device: EPYC_9654,
      // Fits and decodes fine; the 2K prompts on CPU are what take too long.
      says: /before either of 2 users sees a token, past the 30s bar/,
    },
  ])('fails on a $bar the tight tier misses at its own two users', (c) => {
    const serving = judge(c.model, c.quant, { device: c.device, concurrency: 1 }).get('serving')!;

    expect(serving.fitness).toBe('fail');
    expect(serving.reason).toMatch(c.says);
    expect(serving.reason).not.toMatch(/Not measured/);
  });

  /**
   * The tiers are graded at *different* user counts, and this is what pins the pair apart.
   *
   * Grading both at four would make `BARS.serving.tight.users` decoration; grading both at two would
   * let "Multi-user serving — Yes" be earned by a two-user measurement and delete the only
   * enforcement of the good tier's own bar. The recorder below reads the concurrency each evaluation
   * was actually asked for, so this cannot be satisfied by a sentence that merely mentions them.
   */
  it('asks the engine for four users and for two, and for nothing else on this row', () => {
    const asked: number[] = [];
    judgeWorkloads({
      selectedPlacement: RESIDENT,
      usage: { contextTokens: 4096, concurrency: 9, promptTokens: 512, kvPrecision: 'fp16' },
      maxContextTokens: 200_000,
      runnableContextTokens: 200_000,
      evaluateAt: (_prompt, _context, _prefix, concurrency) => {
        asked.push(concurrency);
        return { placement: RESIDENT, ...STUB_SPEED };
      },
    });

    // Serving's two, and the slider's for the six archetypes that inherit it. Nine appears because
    // those six still read the scenario the reader configured; four and two appear because this one
    // does not.
    expect(new Set(asked)).toEqual(
      new Set([9, WORKLOAD_BARS.serving.good.users, WORKLOAD_BARS.serving.tight.users])
    );
  });

  /**
   * The premise the case above rests on, asserted over the catalog rather than argued from the
   * roofline: **more users never improve a served user's rate, and never shorten the wait.**
   *
   * Decode is memory-bound, so the weights are read once per step however many sequences are in
   * flight and the per-user share can only fall; prefill is compute-bound and one long prompt already
   * saturates the units, so serving `n` of them is `n` times the arithmetic. If either direction ever
   * reversed, "the one-user measurement settles it" would stop being true and the grade above would
   * be wrong rather than merely early.
   *
   * Every drivable combination, not a sample, because the claim is about the model and not about one
   * rig — and because a monotonicity claim checked on the machine it was derived from is not checked.
   */
  it('never improves a served user’s rate or wait by adding users', () => {
    const reversals: string[] = [];
    let checked = 0;

    for (const device of DEVICES) {
      for (const runtime of RUNTIMES) {
        if (!runtimeDrives(runtime, device)) continue;
        for (const model of MODELS) {
          for (const quantId of ['q4_k_m', 'bf16']) {
            if (!runtime.weightFormats.includes(quantId)) continue;
            const series = [1, 2, 4, 8].map((concurrency) => {
              const e = evaluate({
                model,
                quant: getQuant(quantId),
                runtime,
                rig: { device, count: 1 },
                usage: {
                  contextTokens: 2560,
                  concurrency,
                  promptTokens: 2048,
                  kvPrecision: 'fp16',
                },
              });
              return {
                concurrency,
                rate: e.decode.perUserTokensPerSec,
                ttft: e.prefill.ttftSeconds,
                unsupported: e.placement.unsupported,
              };
            });
            if (series.some((s) => s.unsupported)) continue;
            checked++;

            const where = `${device.id}/${runtime.id}/${model.id}/${quantId}`;
            for (let i = 1; i < series.length; i++) {
              const [prev, next] = [series[i - 1], series[i]];
              // A thousandth of tolerance for float noise; the reversals this guards are not subtle.
              if (next.rate > prev.rate * 1.001)
                reversals.push(
                  `${where}: per-user rate rose ${prev.rate.toFixed(2)} -> ${next.rate.toFixed(2)} from ${prev.concurrency} to ${next.concurrency} users`
                );
              if (next.ttft < prev.ttft * 0.999)
                reversals.push(
                  `${where}: TTFT fell ${prev.ttft.toFixed(2)}s -> ${next.ttft.toFixed(2)}s from ${prev.concurrency} to ${next.concurrency} users`
                );
            }
          }
        }
      }
    }

    // The sweep has to have swept: an empty filter passes this vacuously, and the first draft of
    // this probe compared `quant.format` — a field `QuantSpec` does not have — so `includes()` was
    // false for every combination and it reported monotonicity over zero cases.
    expect(checked, 'the sweep matched no combinations, so it proves nothing').toBeGreaterThan(500);
    expect(reversals.slice(0, 8), `${reversals.length} reversals`).toEqual([]);
  });

  /**
   * And the collapse the panel does when nothing runs has to keep working, because it is what stops
   * seven copies of one sentence reading as seven separate problems. It fires only when every row
   * carries the same reason, so a row that is ungraded — with a reason all its own — must not appear
   * on that path at any concurrency.
   */
  it.each([1, 4])(
    'grades every row identically when nothing runs, at %s user(s)',
    (concurrency) => {
      const verdicts = judge(GPT_OSS_20B, 'mxfp4', {
        device: RTX_5090,
        runtime: MLX, // Apple-only, on an NVIDIA card.
        concurrency,
      });

      for (const verdict of verdicts.values()) {
        expect(verdict.fitness).toBe('fail');
      }
      expect(new Set([...verdicts.values()].map((v) => v.reason)).size).toBe(1);
    }
  );
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
        evaluateAt: (promptTokens, contextTokens, cachedPrefixTokens) => {
          const e = evaluate({
            model: LLAMA_31_8B,
            quant: getQuant('q4_k_m'),
            usage: {
              contextTokens,
              concurrency: 1,
              promptTokens,
              cachedPrefixTokens,
              kvPrecision: 'fp16',
            },
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

  /**
   * The good tier's user count can still hold a row back — but on **capacity**, not on the slider.
   *
   * Before #96 this test drove the row to three users and read the sentence about being measured
   * against four. That branch is gone with the slider dependency: the tiers are graded at their own
   * counts, so the only way the four-user bar bites now is a machine that holds two served turns and
   * not four. Which is a fact about the machine, and therefore a thing this row is allowed to say.
   */
  it('names the four-user bar when the rig holds two served turns and not four', () => {
    const verdict = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 8192, concurrency: 1, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: 200_000,
        runnableContextTokens: 200_000,
        // Holds the tight tier's two and refuses the good tier's four, which is exactly the shape
        // `runnableContextTokens` could not express: it is one number and this is two answers.
        evaluateAt: (_prompt, _context, _prefix, concurrency) => ({
          ...stub(40, 0.2),
          placement: concurrency >= WORKLOAD_BARS.serving.good.users ? OVER : RESIDENT,
        }),
      }).map((v) => [v.workload.id, v])
    ).get('serving')!;

    expect(verdict.fitness).toBe('tight');
    expect(verdict.reason).toMatch(/holds a turn each for 2 users but not for the 4/);
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
      evaluateAt: stubEngine(600),
    });

    for (const v of verdicts) {
      expect(v.fitness).toBe('fail');
      // Serving states its shortfall in bytes at its own tier's user count rather than in context
      // against `runnableContextTokens`, because that figure is taken at the slider's concurrency
      // and this row is not graded there (#96). Same claim, different quantity.
      const states =
        v.workload.id === 'serving'
          ? /need .* GiB.* against the .* GiB each one can allocate/
          : /needs .* with room to answer in/;
      expect(v.reason, `${v.workload.id} does not state what it needs`).toMatch(states);
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
    expect(rankOf(many.fitness)).toBeLessThanOrEqual(rankOf(one.fitness));
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
        // Refuses above the declared ceiling, like the engine: serving's capacity is now the
        // placement at its own turn rather than `runnableContextTokens`, so a stub that fits
        // everything would exempt the one archetype this sweep was extended to cover (#96).
        evaluateAt: (_prompt, contextTokens) =>
          contextTokens > runnableContextTokens ? { ...stub(200), placement: OVER } : stub(200),
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

/**
 * A `tight` row that prints only healthy figures is a verdict with no evidence — the reader sees
 * three good numbers and a downgrade, and cannot tell which predicate did it. The review named
 * completion, agent and long-context; the same hole was in chat and rag, so this covers the class
 * across every archetype that has a `good` tier to miss.
 */
describe('a tight verdict always names the bar it missed', () => {
  const stub = (perUser: number, ttft: number, placement: Placement = RESIDENT) => ({
    placement,
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
      ttftSeconds: ttft,
      prefillTokensPerSec: 5000,
      linearFlops: 1,
      attentionFlops: 1,
      linearSeconds: 0.1,
      attentionSeconds: 0.1,
      attentionBound: false,
    },
  });

  const graded = (runnableContextTokens: number, perUser: number, ttft: number) =>
    new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens: 4096, concurrency: 4, promptTokens: 2048, kvPrecision: 'fp16' },
        maxContextTokens: runnableContextTokens,
        runnableContextTokens,
        evaluateAt: () => stub(perUser, ttft),
      }).map((v) => [v.workload.id, v])
    );

  /** Every tier that carries a `good` bar, and the band that leaves each one tight on it alone. */
  it.each([
    // id,          runnable,  rate, ttft, what the sentence has to own up to
    ['chat', 400_000, 12, 0.2, /15 tok\/s/],
    ['chat', 400_000, 60, 3, /to first token/],
    ['completion', 400_000, 25, 0.2, /finishes a line slower/],
    ['completion', 400_000, 60, 0.6, /inline suggestion can absorb/],
    ['agent', 40_000, 60, 1, /64K/],
    ['agent', 400_000, 20, 1, /25 tok\/s/],
    ['agent', 400_000, 60, 15, /brisk step/],
    ['rag', 400_000, 7, 1, /7\.0 tok\/s/],
    ['rag', 400_000, 60, 12, /over the 5s bar/],
    // Holds the full window, decode is fine, prefill alone is between the 120s good bar and the
    // 600s tight one. Deleting this conjunct left the whole suite green while reintroducing the
    // bug for any rig in that band.
    ['long-context', 200_000, 60, 300, /300s to read 128K is a long wait/],
    ['long-context', 200_000, 3.2, 110, /3\.2 tok\/s/],
    // Serving joined this table when it gained a latency bar. Its four `good` bars were three
    // hand-written branches that could not have absorbed a fourth; these two are the bands that
    // leave it tight on the rate alone and on the queue alone.
    ['serving', 400_000, 7, 0.2, /10 tok\/s/],
    ['serving', 400_000, 60, 15, /to first token/],
  ] as const)('explains %s at rate %s and ttft %s', (id, runnable, rate, ttft, expected) => {
    const verdict = graded(runnable, rate, ttft).get(id)!;

    expect(verdict.fitness).toBe('tight');
    expect(verdict.reason).toMatch(expected);
    // The positive fallback is for rows that missed nothing, and must not appear on a downgrade.
    expect(verdict.reason).toMatch(/^Usable, but /);
  });

  it('says nothing extra when a row misses no bar at all', () => {
    // Comfortable everywhere: the reason stays the plain positive sentence.
    const chat = graded(400_000, 200, 0.05).get('chat')!;

    expect(chat.fitness).toBe('good');
    expect(chat.reason).not.toMatch(/^Usable, but /);
  });

  it('names the decode rate when decode alone downgrades long-context', () => {
    // The full window fits and prefill clears its bar, so decode is the only thing left — and the
    // row used to mention only the offload and the prompt pass. DeepSeek V3 NVFP4 on one 5090 is
    // the real shape of this: 160K reached, ~110s to prefill, 3.2 tok/s out.
    const long = graded(200_000, 3.2, 110).get('long-context')!;

    expect(long.fitness).toBe('tight');
    expect(long.reason).toMatch(/3\.2 tok\/s/);
    expect(long.reason).toMatch(/^Usable, but /);
  });

  it('does not tell a machine holding 128K that it is short of 128K', () => {
    // `holdsFullWindow` is the prompt *plus* room to answer, so between 131,072 and 131,583 the
    // window is short only by the response allowance. Quoting the bare prompt figure produced
    // "holds 128.2K - short of the 128K", which contradicts its own numbers.
    const long = graded(131_300, 60, 1).get('long-context')!;

    expect(long.fitness).toBe('tight');
    expect(long.reason).not.toMatch(/short of the 128K\b/);
    expect(long.reason).toMatch(/128\.5K/);
    expect(long.reason).toMatch(/^Usable, but /);
  });

  it('names every live bar when long-context misses more than one', () => {
    // Slow to read *and* slow to answer, with the window held: both have to be owned up to.
    const long = graded(200_000, 3.2, 300).get('long-context')!;

    expect(long.reason).toMatch(/300s to read/);
    expect(long.reason).toMatch(/3\.2 tok\/s/);
    expect(long.reason).toMatch(/ and /);
  });

  it('names both causes when both are live', () => {
    // Under the rate bar and over the latency one at the same time.
    const chat = graded(400_000, 12, 3).get('chat')!;

    expect(chat.reason).toMatch(/15 tok\/s/);
    expect(chat.reason).toMatch(/to first token/);
    expect(chat.reason).toMatch(/ and /);
  });
});

/**
 * One defect, found three times: a tier graded on a measurement other than the one it recommends.
 *
 * The long-context tight tier was the first instance and the only one fixed at the time — it was
 * admitting a 64K machine and timing it on a 128K prompt. The same shape was live in three more
 * places, which is the ordinary outcome here: a review names a subset of a class.
 *
 *   - serving graded capacity and decode and never read its own prefill, so a deployment where
 *     every user waits minutes for a first token was reported healthy;
 *   - both agent tiers read a rate measured at the 16K turn while recommending the machine for a
 *     32K or 64K session;
 *   - the RAG sentence printed a machine-wide prompt rate beside the time for one document.
 *
 * These assert the class rather than the three instances, at both ends: a stub that varies with the
 * scenario proves the *right* scenario is the one being read, and the real configurations from the
 * issues prove the arithmetic is worth reading.
 */
describe('a tier is graded on the measurement it recommends', () => {
  const stub = (perUser: number, ttft: number, prefillTokensPerSec = 5000) => ({
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
      ttftSeconds: ttft,
      prefillTokensPerSec,
      linearFlops: 1,
      attentionFlops: 1,
      linearSeconds: 0.1,
      attentionSeconds: 0.1,
      attentionBound: false,
    },
  });

  /** Judges with an `evaluateAt` free to answer differently per scenario, and records what was asked. */
  const judgedAt = (
    runnableContextTokens: number,
    concurrency: number,
    speed: (promptTokens: number, contextTokens: number) => ReturnType<typeof stub>,
    contextTokens = 512
  ) => {
    const seen: Array<[number, number]> = [];
    const verdicts = new Map(
      judgeWorkloads({
        selectedPlacement: RESIDENT,
        usage: { contextTokens, concurrency, promptTokens: 512, kvPrecision: 'fp16' },
        maxContextTokens: runnableContextTokens,
        runnableContextTokens,
        evaluateAt: (promptTokens, contextTokens) => {
          seen.push([promptTokens, contextTokens]);
          return speed(promptTokens, contextTokens);
        },
      }).map((v) => [v.workload.id, v])
    );
    return { verdicts, seen };
  };

  it('grades the agent at the session context its tier claims, not at one turn', () => {
    // Fast with a turn in the cache, slow once a session fills it — which is the real shape of a
    // rig whose weights spill as the cache grows, and the shape a 16K measurement cannot see.
    const { verdicts, seen } = judgedAt(200_000, 1, (_prompt, context) =>
      stub(context >= 65536 ? 8.6 : 49.7, 1)
    );
    const agent = verdicts.get('agent')!;

    expect(seen).toContainEqual([16384, 65536]);
    // 8.6 tok/s is under the tight tier's 15, so the endorsement is withdrawn entirely.
    expect(agent.fitness).toBe('fail');
    expect(agent.reason).toMatch(/8\.6 tok\/s/);
    expect(agent.reason).not.toMatch(/49/);
  });

  it('measures the reduced agent session when that is the one it admits', () => {
    // 40K runnable: the good tier's 64K is out of reach, so the tight tier's 32K session is both
    // what is recommended and what has to be timed. Reading the 64K figure here would fail a rig
    // for a session it was never offered.
    const { verdicts, seen } = judgedAt(40_000, 1, (_prompt, context) =>
      stub(context >= 65536 ? 8.6 : 40, 1)
    );
    const agent = verdicts.get('agent')!;

    expect(seen).toContainEqual([16384, 32768]);
    expect(agent.fitness).toBe('tight');
    expect(agent.reason).toMatch(/short of the 64K/);
    expect(agent.reason).not.toMatch(/8\.6/);
  });

  it('names the session it measured, not the tier bar, when the slider asks for more', () => {
    // Every archetype is floored at the configured context, not pinned to its own, because that
    // cache really is the size the user asked for. So the tier's bar and the evaluated session are
    // two different numbers above 64K, and the sentence has to quote the second: quoting the bar
    // failed a rig at "10 tok/s with a 64K session in the cache" on evidence from 128K, when the
    // 64K it named would have been tight.
    const { verdicts, seen } = judgedAt(
      200_000,
      1,
      (_prompt, context) => stub(context >= 131072 ? 9 : 60, 1),
      131072
    );
    const agent = verdicts.get('agent')!;

    expect(seen).toContainEqual([16384, 131072]);
    expect(seen).not.toContainEqual([16384, 65536]);
    expect(agent.fitness).toBe('fail');
    expect(agent.reason).toMatch(/9\.0 tok\/s/);
    expect(agent.reason).toMatch(/128K session/);
    expect(agent.reason).not.toMatch(/64K/);
  });

  it('says whose throughput the batch figure is', () => {
    // `batchAggregate` sums every worker, so at 32 of them it is 32x what one job finishes at —
    // the same unlabelled-aggregate defect as #11, on the other aggregate in the file.
    const alone = judgedAt(400_000, 1, () => stub(60, 1)).verdicts.get('batch')!;
    const crowded = judgedAt(400_000, 32, () => stub(60, 1)).verdicts.get('batch')!;

    expect(alone.reason).toMatch(/end to end —/);
    expect(alone.reason).not.toMatch(/across/);
    expect(crowded.reason).toMatch(/end to end across 32 workers/);
  });

  it('reads the prefill of the serving scenario, not only its capacity and decode', () => {
    // Everything the old predicates looked at is healthy: it fits, and 40 tok/s each is well over
    // the 10 a served user expects. Only the queue is broken.
    const serving = judgedAt(400_000, 8, () => stub(40, 165)).verdicts.get('serving')!;

    expect(serving.fitness).toBe('fail');
    expect(serving.reason).toMatch(/165s before either of 2 users sees a token/);
  });

  it('says a long wait in minutes, ceiled, like every panel beside it', () => {
    // 8068s is two and a quarter hours, and it printed as "8068s" two panels under a tile that
    // says "134 min" about the same kind of figure (#125). The sentence switches units at the
    // same cutoff seconds() uses — and ceils, because a failing latency never rounds down: 135,
    // not the 134 the rounding panels show. The bar it missed stays quoted in seconds.
    const long = judgedAt(400_000, 1, () => stub(60, 8068)).verdicts.get('long-context')!;
    expect(long.reason).toMatch(/135 min to read/);
    expect(long.reason).not.toMatch(/8068/);

    // The near side of the switch, at the ceiling's own granularity: 598.9 ceils to 599 and
    // stays in seconds; 599.3 ceils across the cutoff and is already minutes.
    const near = judgedAt(400_000, 1, () => stub(60, 598.9)).verdicts.get('long-context')!;
    expect(near.reason).toMatch(/599s to read/);
    const across = judgedAt(400_000, 1, () => stub(60, 599.3)).verdicts.get('long-context')!;
    expect(across.reason).toMatch(/10 min to read/);
  });

  it('keeps serving good when the queue is short as well as the rate', () => {
    // The mirror of the case above, so the new bar cannot be satisfied by failing everything.
    const serving = judgedAt(400_000, 8, () => stub(40, 2)).verdicts.get('serving')!;

    expect(serving.fitness).toBe('good');
    expect(serving.reason).toMatch(/to first token/);
  });

  it('prints a serving aggregate that is the product of the figures beside it', () => {
    // 14.63 tok/s each at four users. Flooring each figure independently rendered "14 tok/s
    // each, 58 aggregate" — a multiplication that does not multiply (#124). The aggregate is
    // derived from the per-user figure as printed, so the sentence's arithmetic holds, and the
    // derived figure stays at or under the engine's 4 × 14.63 = 58.5.
    const serving = judgedAt(400_000, 8, () => stub(14.63, 2)).verdicts.get('serving')!;

    expect(serving.fitness).toBe('good');
    const [, each, aggregate] = serving.reason.match(
      /4 users at ([\d.]+) tok\/s each, ([\d.]+) tok\/s aggregate/
    )!;
    expect(Number(aggregate)).toBe(4 * Number(each));
    expect(Number(aggregate)).toBeLessThanOrEqual(4 * 14.63);
  });

  it('prints a RAG rate that divides into the wait beside it', () => {
    // Eight users, a 4s wait, and a machine-wide rate of 32,768 * 8 / 4. The sentence is about one
    // document, so its two figures have to multiply back to that document's length; unqualified,
    // the rate was eight times too large for the wait printed next to it.
    const rag = judgedAt(400_000, 8, () => stub(60, 4, (32768 * 8) / 4)).verdicts.get('rag')!;

    expect(rag.fitness).toBe('good');
    const [, rate, seconds] = rag.reason.match(/([\d.]+) tok\/s.*?([\d.]+)s for a 32K one/)!;
    expect(Number(rate) * Number(seconds)).toBeCloseTo(32768, -2);
    // The machine-wide figure, which is what used to be printed here.
    expect(rag.reason).not.toMatch(/65536|65,536/);
  });

  /**
   * The configurations from the issues, so the thresholds are exercised against real arithmetic
   * rather than only against a stub that was told what to say.
   */
  it('withdraws serving from a machine whose users wait minutes for a first token', () => {
    // Llama 3.1 8B Q4_K_M on an EPYC 9654, four users: it fits and decodes ~40 tok/s each, and
    // the four 2K prompts take about 165 seconds before anyone sees a token.
    const verdicts = judge(LLAMA_31_8B, 'q4_k_m', { device: EPYC_9654, concurrency: 4 });
    const serving = verdicts.get('serving')!;

    expect(serving.fitness).toBe('fail');
    expect(serving.reason).toMatch(/before either of 2 users sees a token/);
  });

  it('withdraws the agent from a machine that only holds its session', () => {
    // Llama 3.1 8B at BF16 on one 4090 under vLLM: 49.7 tok/s at the 16K turn, and about 8.6 once
    // its own 64K session is resident, because the weights spill to make room for the cache.
    const agent = judge(LLAMA_31_8B, 'bf16', {
      device: RTX_4090,
      runtime: VLLM,
      contextTokens: 512,
    }).get('agent')!;

    expect(agent.fitness).toBe('fail');
    expect(agent.reason).toMatch(/64K session/);
    expect(agent.reason).not.toMatch(/49/);
  });
});

/**
 * The bars now live in one structure, which is what makes this assertable at all.
 *
 * Every `good` bar used to be written down twice — once in a predicate, once in the sentence that
 * explains missing it — and every `tight` bar sat forty lines from its `good` counterpart. Nothing
 * was wrong; nothing *made* it right either, and this file's history is a list of the times two
 * copies of one number stopped matching.
 */
describe('every numeric good bar is at least as strict as its own tight bar', () => {
  // Numeric, because serving has one `good` bar that is not a number and has no `tight` counterpart
  // by design — `headroomOf('serving') > 0`. Everything expressible as a threshold is in the
  // structure; that one is a predicate about the placement and stays in the tier.
  //
  // Latency is an upper bound, so `good` is the smaller number. Everything else is a lower bound,
  // so `good` is the larger. Getting the direction wrong on one axis is the failure this catches —
  // it would silently make a tier unreachable, or make `good` pass where `tight` fails. An axis
  // that is an upper bound and not named here fails loudly rather than silently, which is the right
  // way round for a default.
  const UPPER_BOUND = new Set(['ttft']);

  it.each(Object.keys(WORKLOAD_BARS))('holds for %s', (id) => {
    const bars = WORKLOAD_BARS[id as keyof typeof WORKLOAD_BARS];
    const good = bars.good as Record<string, number>;
    const tight = bars.tight as Record<string, number>;

    // Both tiers describe the same axes, or one of them is grading something the other ignores.
    expect(Object.keys(good).sort()).toEqual(Object.keys(tight).sort());

    for (const [axis, goodBar] of Object.entries(good)) {
      if (UPPER_BOUND.has(axis)) {
        expect(goodBar).toBeLessThanOrEqual(tight[axis]);
      } else {
        expect(goodBar).toBeGreaterThanOrEqual(tight[axis]);
      }
    }
  });
});

/**
 * #23's decision, asserted rather than left in a comment: a coding-agent turn attends against a
 * session already in the cache, and no other archetype does.
 */
describe('only the agent grades its prompt against a resident session', () => {
  /**
   * `RESPONSE_ALLOWANCE` is private to `verdict.ts`, deliberately — it is an internal convention,
   * not part of the contract. Restated here so this test fails if the two ever disagree, which is
   * the whole point of asserting the occupancy closes on the window.
   */
  const RESPONSE_TOKENS = 512;

  /** Every `(prompt, context, prefix)` the verdict layer asks for, in order. */
  const scenariosAsked = () => {
    const asked: { promptTokens: number; contextTokens: number; cachedPrefixTokens: number }[] = [];
    const usage = {
      contextTokens: 8192,
      concurrency: 4,
      promptTokens: 2048,
      kvPrecision: 'fp16' as const,
    };
    const rig = { device: RTX_5090, count: 1 };
    const evaluation = evaluate({
      model: LLAMA_31_8B,
      quant: getQuant('q4_k_m'),
      usage,
      rig,
      runtime: LLAMA_CPP,
    });

    judgeWorkloads({
      selectedPlacement: evaluation.placement,
      usage,
      maxContextTokens: evaluation.maxContextTokens,
      runnableContextTokens: evaluation.runnableContextTokens,
      evaluateAt: (promptTokens, contextTokens, cachedPrefixTokens) => {
        asked.push({ promptTokens, contextTokens, cachedPrefixTokens });
        const e = evaluate({
          model: LLAMA_31_8B,
          quant: getQuant('q4_k_m'),
          usage: { ...usage, promptTokens, contextTokens, cachedPrefixTokens },
          rig,
          runtime: LLAMA_CPP,
        });
        return { placement: e.placement, decode: e.decode, prefill: e.prefill };
      },
    });
    return asked;
  };

  it('asks for a prefix on the agent turn and nowhere else', () => {
    const asked = scenariosAsked();
    expect(asked.length).toBeGreaterThan(1);

    const agentTurn = WORKLOADS.find((w) => w.id === 'agent')!.typicalPromptTokens;
    for (const scenario of asked) {
      if (scenario.cachedPrefixTokens === 0) continue;
      // The only scenario carrying a prefix is the agent's, and the prefix is exactly the part of
      // the session the turn does not need — the turn *and its answer*, not the turn alone.
      // Subtracting only the prompt spent the whole window and left the reply nowhere.
      expect(scenario.promptTokens).toBe(agentTurn);
      expect(scenario.cachedPrefixTokens).toBe(
        scenario.contextTokens - agentTurn - RESPONSE_TOKENS
      );
      // The occupancy has to close exactly on the window, or one of these three is wrong.
      expect(scenario.cachedPrefixTokens + agentTurn + RESPONSE_TOKENS).toBe(
        scenario.contextTokens
      );
    }

    // And it really does ask for one, or the loop above is vacuous.
    expect(asked.some((s) => s.cachedPrefixTokens > 0)).toBe(true);
  });

  it('declares the reading on the archetype rather than assuming it', () => {
    // The flag is what makes the six single-prompt archetypes safe from this change, so it has to
    // be exactly one archetype and it has to be the agent.
    const declared = WORKLOADS.filter((w) => w.prefixIsCached);
    expect(declared.map((w) => w.id)).toEqual(['agent']);
  });

  it('grades the agent slower for it, on a rig where that decides the tier', () => {
    // 8B at Q4_K_M on one 5090: a 16K turn against the 47.5K already resident in a 64K session is
    // ~14s where the turn alone is ~6s, and the good tier's latency bar is 10s. The tier moves
    // because the estimate finally describes what an agent does, not because a threshold changed.
    const agent = judge(LLAMA_31_8B, 'q4_k_m', { device: RTX_5090 }).get('agent')!;

    expect(agent.fitness).toBe('tight');
    // 47.5K, not 64K and not 48K: the prefix is the session minus the turn *and its answer*, and
    // the sentence must name the figure the estimate was called with. The half-K is the room to
    // answer in — house style here, matching the 16.5K, 32.5K and 128.5K the other tiers print.
    expect(agent.reason).toMatch(/against the 47\.5K already in the cache/);
    expect(agent.reason).not.toMatch(/against the (64K|48K)/);
    // Never "re-read": not re-reading the session is the whole point of a cached prefix.
    expect(agent.reason).not.toMatch(/re-read/);
  });
});
