import type {
  DeviceSpec,
  KvPrecision,
  ModelSpec,
  QuantSpec,
  RuntimeSpec,
  UsageSpec,
} from './types';
import { estimateScenario, type ScenarioEstimate } from './index';
import { maxContextThatFits } from './placement';
import { gradedScenarios, judgeWorkloads, WORKLOADS, type Fitness, type Workload } from './verdict';

/**
 * The question people actually arrive with (#138).
 *
 * Nobody thinks "evaluate this model at Q4_K_M under llama.cpp on an M3 Max". They think *"I want
 * a local coding assistant — what is the best model I can run?"* The engine has been able to answer
 * that from the beginning; no surface asked it. The Matrix holds the answer as 1,470 cells to
 * interpret, and this returns the decision.
 *
 * ## The sweep's axes are the engine's, not the Matrix's
 *
 * The Matrix renders every cell under one globally selected runtime, KV precision and quant
 * substitution, at a hardcoded `deviceCount: 1` — its cells are a *slice* of the space rather than
 * the space. A model categorically refused under the current runtime can run under another, and a
 * different quant changes both fit and rank. So this sweeps **models × runtimes × applicable
 * quants**, and takes KV precision and device count as explicit inputs rather than assuming them.
 *
 * ## Every ranking rule is stated, because otherwise the shortlist is an opinion
 *
 * A ranked list is a recommendation, and a recommendation with an unstated basis is an opinion
 * wearing the chassis of a measurement — the failure this whole codebase is organised against. So
 * the two orderings below are exported as sentences (`RANKING_RULE`, `FALLBACK_RULE`) and rendered
 * beside the shortlist, and the quant policy is stated too (`QUANT_RULE`).
 *
 * **bench deliberately knows nothing about which model is *better***, only which runs. The honest
 * within-tier ordering is therefore a capability *proxy*, and parameter count is the defensible one:
 * it is already derived from each repo's safetensors index, where benchmark scores are not in the
 * catalog at all and importing them would be a new curation surface with a freshness problem. That
 * is a real limitation and the sentence says so rather than implying a quality judgement.
 */

/** The order the tiers rank in. Not a score — a total order over three named states. */
const TIER_RANK: Record<Fitness, number> = { good: 0, tight: 1, fail: 2 };

/**
 * How the shortlist is ordered, in words, for the surface to print.
 *
 * Every clause is a decision that could have gone another way, which is why it is a sentence and
 * not a comment: "then by decode rate" would be a defensible ranking too, and would put a fast 8B
 * above a 235B that also clears the bar. Within a tier the bar is already met, so speed is adequate
 * by construction and the interesting axis is what the model can do.
 */
export const RANKING_RULE =
  'Ranked by verdict first, then by parameter count, then by how little the weights are ' +
  'compressed, then by decode rate. Parameter count is a proxy for capability and not a ' +
  'measurement of it — bench knows what runs, not what is good.';

/**
 * And the other ordering, which is a different question and therefore a different rule.
 *
 * When nothing clears the bar, "biggest that loads" is the wrong answer: a 671B that decodes at 0.3
 * tok/s is not more useful than an 8B at 40 that merely missed a threshold. What the reader needs
 * there is what runs *fastest*, so the fallback pick is chosen by decode rate and says so.
 */
export const FALLBACK_RULE =
  'Nothing clears this bar on this machine, so the fallback is the fastest configuration that ' +
  'loads at all rather than the largest.';

/**
 * The quant policy, which the issue names as one of the three pieces of real work.
 *
 * **Verdict first, then width.** A wider format is less lossy, so taking the narrowest thing that
 * fits would recommend Q3 where Q8 runs — and taking a fixed default would repeat the Matrix's P1,
 * where a hardcoded fallback scored dense rows at a format vLLM cannot read. Every format offered
 * here goes through the caller's `quantsFor`, which is `quantApplies` *with the runtime*.
 *
 * **The wording is the second draft, and the first was false on 347 shipping configurations.** It
 * read "the widest format that clears the bar", which is a different rule: `bestQuant` prefers a
 * *narrower* `good` over a wider `tight`, because the tiers rank above width everywhere else in
 * this file — `Shortlist.best` counts `tight` as clearing too, so "clears the bar" was doing two
 * jobs in one module. One meaning now, stated as the comparator implements it.
 */
export const QUANT_RULE =
  'The format that grades best, and the widest of those — narrowing only as far as the verdict ' +
  'requires. Only formats the runtime can actually load.';

export interface Candidate {
  model: ModelSpec;
  quant: QuantSpec;
  runtime: RuntimeSpec;
  fitness: Fitness;
  /** The verdict layer's own sentence, naming the bar cleared or missed. Never rewritten here. */
  reason: string;
  /**
   * The scenario this candidate was actually graded at — the archetype's, never the reader's.
   *
   * The archetype's own request wherever the machine can plan it, and otherwise the largest of its
   * tiers' scenarios that it can: see `planGraded`. **Carried rather than re-derived**, because the
   * panel's deep link has to name it too and a second derivation of a tier ladder is what
   * `gradedScenarios` exists to stop — it loads the row into the Bench at this scenario, and
   * reconstructing the archetype's full request there would land a long-context recommendation
   * earned at half the window on a configuration the Bench cannot place, which is the defect #167
   * fixed on the other axis.
   */
  contextTokens: number;
  promptTokens: number;
  /** Per-user decode at that scenario. */
  tokensPerSec: number;
  /** And the wait for that scenario's own prompt, which is why the two travel together. */
  ttftSeconds: number;
  /**
   * Whether this configuration only runs by spilling weights to host RAM.
   *
   * Carried because the surface owes it the host-RAM qualifier: `planPlacement` sizes a spill with
   * no host-RAM input at all, so a shortlist row saying a spilled configuration runs is promising
   * something never checked. Same condition the Matrix, the Envelope and Telemetry all key on.
   */
  offloadFraction: number;
}

export interface Shortlist {
  workload: Workload;
  /** Every configuration that loads on this machine, best first. */
  ranked: readonly Candidate[];
  /** The top configuration that clears the archetype's bar, if any does. */
  best?: Candidate;
  /**
   * The fastest configuration that loads, present only when nothing clears the bar.
   *
   * "Nothing" is a wrong answer when a smaller model at Q4 runs fine, and this is what stops the
   * surface giving it.
   */
  fallback?: Candidate;
  /**
   * After the headline pick, the next two — so the answer is a choice rather than an oracle.
   *
   * **One entry per model**, which is a rule rather than a tidy-up. The sweep's runtime axis means
   * a strong model appears two or three times over: gpt-oss 120B ranks at Q5_K_M under llama.cpp
   * *and* at NVFP4 under vLLM, and a shortlist whose three rows are two spellings of one model has
   * not offered a choice. The best entry for each of the next two distinct models is the choice the
   * reader asked for; the runtime is part of each answer rather than a way to fill the list.
   */
  runnersUp: readonly Candidate[];
  /** How many model × runtime pairs were considered, so the surface can say what it looked at. */
  pairsConsidered: number;
}

export interface RecommendInputs {
  device: DeviceSpec;
  /** An explicit axis, not an assumption — the Matrix's hardcoded 1 is what this exists to escape. */
  deviceCount: number;
  /** Likewise explicit: a narrower cache changes what fits, and therefore what is recommended. */
  kvPrecision: KvPrecision;
  /**
   * Sequences sharing the machine — the reader's own setting, and a third axis for the same reason.
   *
   * Inherited by the six archetypes that do not declare their own, exactly as `Workloads.tsx` hands
   * it to `judgeWorkloads`. Hardcoding 1 here let the shortlist and the verdict strip grade the same
   * configuration's *batch* row differently on one page — `batchAggregate` reads `usage.concurrency`
   * — so clicking a row landed the reader on a contradicting grade. Serving is unaffected either
   * way: it declares its own user count per tier.
   */
  concurrency: number;
  workloadId: string;
  models: readonly ModelSpec[];
  runtimes: readonly RuntimeSpec[];
  /**
   * The formats worth offering for a model on this device under this runtime, in any order.
   *
   * A callback for the same reason `computeMatrix` takes one: `quantApplies` lives outside
   * `src/engine/`, which imports nothing beyond itself.
   *
   * **The width ordering is imposed here rather than asked for, and the first draft asked.** It
   * documented "widest first" as the caller's responsibility, and the only caller passed
   * `QUANTS.filter(...)` — which is grouped by *checkpoint family* and deliberately not
   * bpw-descending, as `quants.ts`'s own docblock states at length: `q8_0` at 8.5 bpw sits below
   * `nvfp4` at 4.5. So the walk met `mxfp4` (4.25) before `q6_k` (6.57), stopped at the first
   * `good`, and picked a narrower format than the printed rule promised on **347** shipping
   * configurations. A precondition a caller can silently violate is not a precondition; the policy
   * depends on the order, so the policy owns it.
   */
  quantsFor: (model: ModelSpec, runtime: RuntimeSpec) => readonly QuantSpec[];
}

/**
 * Sweep the catalog for one machine and one workload, and rank what runs.
 *
 * Pure, like everything else here, and cheap enough to call on a selection change — but not on a
 * slider frame, which is why the surface renders a shortlist rather than a grid. The #101 lesson is
 * that cells cost wall-clock at render and not at compute; this returns three rows.
 */
export function recommend(inputs: RecommendInputs): Shortlist {
  const workload = WORKLOADS.find((w) => w.id === inputs.workloadId);
  if (workload === undefined) throw new Error(`Unknown workload: ${inputs.workloadId}`);

  const candidates: Candidate[] = [];
  /**
   * Every loadable configuration, not one per pairing.
   *
   * `bestQuant` reduces each model × runtime pair to its widest best-grading format, which is right
   * for the *ranked* list and wrong for the fallback: `FALLBACK_RULE` promises the fastest thing
   * that loads, and the fastest thing is routinely a narrower quant of a pairing whose widest one
   * won the reduction. Searching the reduced list found the fastest survivor of a pruning that
   * ranked on width — a different claim from the one printed. Raised by Codex on #167.
   */
  let fastest: Candidate | undefined;
  let pairsConsidered = 0;

  for (const model of inputs.models) {
    for (const runtime of inputs.runtimes) {
      /**
       * Sorted here, never trusted from the caller — see `quantsFor`. A copy, so a caller handing
       * back a frozen catalog slice is not mutated.
       *
       * **The id clause is what makes it a total order**, and without it the sort was only as
       * deterministic as the caller's input: `fp8` and `int8` are both 8.0 bpw, so `Array.sort`'s
       * stability left whichever the caller listed first to win the walk — and `bestQuant` breaks
       * at the first `good`, so the pick flipped when the catalog was reversed. Alphabetical is
       * arbitrary and that is the point; it is here to be *stable*, exactly like the model-id
       * clause in `compare`.
       */
      const quants = [...inputs.quantsFor(model, runtime)].sort(
        (a, b) => b.bpw - a.bpw || a.id.localeCompare(b.id)
      );
      if (quants.length === 0) continue;
      pairsConsidered += 1;

      const chosen = bestQuant(model, runtime, quants, workload, inputs, (candidate) => {
        if (fastest === undefined || candidate.tokensPerSec > fastest.tokensPerSec) {
          fastest = candidate;
        }
      });
      if (chosen !== undefined) candidates.push(chosen);
    }
  }

  const ranked = candidates.sort(compare);
  const best = ranked.find((c) => c.fitness !== 'fail');
  /**
   * By decode rate, per `FALLBACK_RULE`, and only when nothing cleared. `ranked[0]` would be the
   * largest thing that loads, which is the wrong answer to the question this field asks.
   */
  const fallback = best === undefined ? fastest : undefined;

  const headline = best ?? fallback;
  const seen = new Set(headline === undefined ? [] : [headline.model.id]);
  const runnersUp: Candidate[] = [];
  for (const c of ranked) {
    if (runnersUp.length === 2) break;
    if (seen.has(c.model.id)) continue;
    seen.add(c.model.id);
    runnersUp.push(c);
  }

  return { workload, ranked, best, fallback, runnersUp, pairsConsidered };
}

/**
 * The best-grading format for a pairing, and the widest of those.
 *
 * Walks widest-first — an order `recommend` imposes rather than inherits — and **stops at the first
 * `good`**, which is an early-out rather than the policy: nothing narrower can outrank it, since
 * `compare` puts tier above width. It does *not* stop on a `tight` or a failure, because a wide
 * format grading poorly says nothing about a narrower one, which is the whole reason narrowing is a
 * strategy.
 *
 * `undefined` when no format loads at all: a model this machine cannot run under this runtime is
 * absent from the shortlist rather than present as a refusal, because the shortlist is a list of
 * answers and the Matrix is where the refusals are already legible.
 *
 * **The early break costs the fallback nothing**, because `onLoadable` sees every format that
 * loads before the walk stops — including the narrower, faster ones the reduction discards.
 */
function bestQuant(
  model: ModelSpec,
  runtime: RuntimeSpec,
  quants: readonly QuantSpec[],
  workload: Workload,
  inputs: RecommendInputs,
  /** Called for every format that loads, so the fallback can see what the reduction discards. */
  onLoadable: (candidate: Candidate) => void
): Candidate | undefined {
  let best: Candidate | undefined;

  for (const quant of quants) {
    const candidate = grade(model, quant, runtime, workload, inputs);
    if (candidate === undefined) continue;
    onLoadable(candidate);
    if (best === undefined || compare(candidate, best) < 0) best = candidate;
    if (candidate.fitness === 'good') break;
  }

  return best;
}

/**
 * A scenario whose prompt is stated rather than inferred.
 *
 * `UsageSpec.promptTokens` is optional — `effectivePromptTokens` defaults it to 90% of the window —
 * and every scenario the sweep plans names its own, so the row can carry it without a fallback that
 * would silently describe a different request.
 */
type GradedUsage = UsageSpec & { promptTokens: number };

/**
 * The largest scenario this archetype is graded at that this machine can actually plan.
 *
 * **The sweep asks the verdict layer about tiers rather than reconstructing them here**
 * ([#170](https://github.com/MrZoller/bench/issues/170)). It used to plan exactly one placement per
 * candidate, at the archetype's own `typicalPromptTokens + RESPONSE_ALLOWANCE`, and several
 * archetypes are *graded* at scenarios that one never names: long-context's `tight` tier is a
 * 65,536-token prompt against its 131,072-token job, and the agent's two tiers carry 64K and 32K
 * sessions against a ~16.5K turn. `gradedScenarios` is that structure, and this walks it.
 *
 * **Largest first, stopping at the first entry that plans**, which keeps `judgeWorkloads`' own
 * top-level refusal exactly where it is: it fires when *no* tier's scenario loads, rather than being
 * weakened into admitting candidates whose headline scenario is impossible.
 *
 * **Six of the seven plan the same window they planned before**, on any machine: their first entry
 * *is* the archetype's declared request — the five whose tiers state no working size have only that
 * entry, and long-context's `good` bar is that request rather than a second copy of it. The *prompt*
 * moves on some of those rows, and in one direction only: `gradedScenarios` clamps it to a window
 * the model has truncated, so a long-context row on a 40K model is no longer timed reading a 128K
 * prompt it has nowhere to put. That changes `ttftSeconds`, which no ordering here reads and no
 * surface prints, and nothing else. **The agent is the one whose window moves**, and it moves by
 * design: its first entry is the 64K session its tiers
 * endorse rather than its ~16.5K turn, so the placement now describes the session the verdict is
 * about. Its *grade* does not move with it — `judgeWorkloads` takes that archetype's capacity from
 * `runnableContextTokens` and its session from the tier bars, so it was already grading at 64K while
 * the row's own figures came from the turn. What changes is that the two now agree.
 *
 * The one scenario returned is then everything the candidate is described by — the refusal basis,
 * the `usage` `judgeWorkloads` grades against, the figures the row carries, and the placement the
 * spill caveat is read from. Still one scenario per candidate; the correction is *which* one.
 * Keeping two would reintroduce the defect the verdict layer's own history is a list of, where a
 * grade and the figures printed beside it come from different working sets.
 *
 * `undefined` when nothing loads at any of them: `unsupported` for a pairing the runtime cannot
 * open — which no scenario rescues, since it is a property of the model, the format and the runtime
 * — and `impossible` for one whose cache alone is over the ceiling at every tier. Both are absences
 * rather than low rankings, because a shortlist entry is a recommendation.
 */
function planGraded(
  model: ModelSpec,
  quant: QuantSpec,
  runtime: RuntimeSpec,
  workload: Workload,
  inputs: RecommendInputs
): { usage: GradedUsage; selected: ScenarioEstimate } | undefined {
  const rig = { device: inputs.device, count: inputs.deviceCount };

  // The model's ceiling goes *into* the question rather than onto the answer. Clamping the returned
  // windows here would grade an archetype at a size no tier states — see `gradedScenarios`.
  for (const tier of gradedScenarios(workload.id, model.maxContext)) {
    const usage = {
      contextTokens: tier.contextTokens,
      concurrency: inputs.concurrency,
      promptTokens: tier.promptTokens,
      kvPrecision: inputs.kvPrecision,
    };
    const selected = estimateScenario({ model, quant, usage, rig, runtime });

    if (selected.placement.unsupported !== undefined) return undefined;
    if (selected.placement.impossible) continue;
    return { usage, selected };
  }

  return undefined;
}

/**
 * One configuration, graded at the scenario `planGraded` settled on.
 *
 * The grade comes from `judgeWorkloads` rather than from a second set of thresholds here, which is
 * the same rule the Matrix and the Bench already follow: the bar a verdict names has to be the bar
 * it was tested against. Six of the seven archetypes are discarded, which is the cost of not owning
 * a copy of the grading logic — and the engine is closed-form arithmetic, so it is a cost worth
 * paying to keep one definition of `good`.
 */
function grade(
  model: ModelSpec,
  quant: QuantSpec,
  runtime: RuntimeSpec,
  workload: Workload,
  inputs: RecommendInputs
): Candidate | undefined {
  const rig = { device: inputs.device, count: inputs.deviceCount };
  const planned = planGraded(model, quant, runtime, workload, inputs);
  if (planned === undefined) return undefined;

  const { usage, selected } = planned;
  const scenario = { model, quant, usage, rig, runtime };

  const verdicts = judgeWorkloads({
    selectedPlacement: selected.placement,
    usage,
    maxContextTokens: maxContextThatFits(model, quant, usage, rig, runtime),
    runnableContextTokens: maxContextThatFits(model, quant, usage, rig, runtime, {
      allowOffload: true,
    }),
    evaluateAt: (promptTokens, contextTokens, cachedPrefixTokens, concurrency) =>
      estimateScenario({
        ...scenario,
        usage: { ...usage, promptTokens, contextTokens, cachedPrefixTokens, concurrency },
      }),
  });

  const verdict = verdicts.find((v) => v.workload.id === workload.id);
  if (verdict === undefined) return undefined;

  return {
    model,
    quant,
    runtime,
    fitness: verdict.fitness,
    reason: verdict.reason,
    contextTokens: usage.contextTokens,
    promptTokens: usage.promptTokens,
    tokensPerSec: selected.decode.perUserTokensPerSec,
    ttftSeconds: selected.prefill.ttftSeconds,
    /**
     * From the scenario this candidate is graded at, which is the whole point of `planGraded`
     * returning one.
     *
     * An agent whose ~16.5K turn keeps every weight resident can spill at the 64K session its tier
     * endorses, and a row taking the fraction from the turn omitted the host-RAM qualifier on a
     * configuration that only runs by spilling. That was patched on #167 by widening the reading to
     * the archetype's largest bar; the walk supersedes the widening, and is exact where it was
     * conservative — a candidate that reaches only the reduced tier is now described by the
     * placement of the tier it reached, rather than by one the machine cannot hold.
     */
    offloadFraction: selected.placement.offloadFraction,
  };
}

/**
 * `RANKING_RULE`, as a comparator.
 *
 * The last clause is the model id, and it is not part of the printed rule because it is not a
 * judgement — it is what makes the order total, so the same catalog always produces the same
 * shortlist and a test can pin one. Without it two configurations equal on every stated axis would
 * rank by whatever order the sweep happened to visit them in.
 */
function compare(a: Candidate, b: Candidate): number {
  return (
    TIER_RANK[a.fitness] - TIER_RANK[b.fitness] ||
    b.model.totalParams - a.model.totalParams ||
    b.quant.bpw - a.quant.bpw ||
    b.tokensPerSec - a.tokensPerSec ||
    a.model.id.localeCompare(b.model.id) ||
    // Two formats of identical width on one model — `fp8` and `int8` are both 8.0 bpw — leave every
    // stated axis tied. Arbitrary, and here only so the order is total.
    a.quant.id.localeCompare(b.quant.id)
  );
}
