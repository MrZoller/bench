import type { UsageSpec } from './types';
import type { DecodeEstimate, PrefillEstimate } from './speed';
import type { Placement } from './placement';

/**
 * What the setup can actually be used for.
 *
 * "3.2 tok/s" is not an answer. "Unusable for a coding agent, fine for overnight batch" is. This
 * file is the translation, and it is the part of the tool that most calculators skip entirely —
 * they stop at a number and leave the reader to know what a good one looks like.
 *
 * **On the thresholds.** They are reading speed and patience, not benchmarks, and they are
 * judgement rather than measurement. Roughly: below 10 tok/s you are watching a cursor, around
 * 15 a chat keeps up with reading, past 30 it outruns most people. Time-to-first-token has a
 * much tighter budget for a completion popup than for a document you asked a question about.
 * They are stated as constants here so they can be argued with, rather than buried in prose.
 *
 * Every verdict carries a written reason. A colour alone would not survive contact with someone
 * deciding whether to spend money.
 */

/**
 * Three grades and one non-grade.
 *
 * `good`, `tight` and `fail` are judgements — this setup does the thing well, marginally, or not at
 * all — and they are the vocabulary every surface renders with a reserved status colour. `unmeasured`
 * is the *absence* of a judgement: the configuration on screen does not exercise the archetype, so
 * nothing was measured and nothing is being claimed either way.
 *
 * It exists because `fail` was doing both jobs, and the strongest negative in a vocabulary cannot
 * mean two things (#75). Multi-user serving at one concurrent user read `○ No` in critical red — the
 * same token as "Will not run" — on a Spark that would serve several users perfectly well, purely
 * because nobody had touched the slider. The row above it, `RAG / document Q&A — No — 31s to read a
 * 32K document`, is a real failure: measured, attributable, and fixable by changing hardware. The two
 * rendered identically.
 *
 * The consequence a consumer has to decide deliberately: an ungraded row belongs in *neither* side
 * of a "N of M workloads" count, because it is not evidence in either direction. See `Workloads.tsx`.
 */
export type Fitness = 'good' | 'tight' | 'fail' | 'unmeasured';

/**
 * Room a workload needs for its answer, on top of its prompt.
 *
 * `UsageSpec.contextTokens` covers prompt *and* generation, so a 32,768-token document on a
 * model capped at 32,768 leaves nowhere to reply from. Small, because these archetypes ask
 * short questions of long inputs — but not zero, which is what the fit checks assumed.
 */
const RESPONSE_ALLOWANCE = 512;

/**
 * The session sizes the two agent tiers recommend a machine for — and therefore have to grade it
 * at.
 *
 * Both tiers read a decode rate measured at the archetype's 16K *turn* while their capacity bars
 * endorsed the rig for a 32K or 64K *session*, which is the long-context defect one archetype
 * over: a grade taken at a smaller working set than the one it is recommending. Llama 3.1 8B at
 * BF16 on one 4090 under vLLM was `good` at "49 tok/s"; at the 64K session that tier claims, its
 * weights spill and it decodes at about 8.6 — under even the tight tier's 15.
 *
 * Both figures are the *context*, not a prompt, so no response allowance is added: `UsageSpec`
 * counts the whole window and these are the capacity bars the predicates already tested.
 */
const AGENT_SESSION_CONTEXT = 65536;
const AGENT_TIGHT_SESSION_CONTEXT = 32768;

export interface Workload {
  id: string;
  label: string;
  /** What the archetype actually is, for a reader who does not recognise the label. */
  description: string;
  /**
   * Prompt length this archetype actually sends, in tokens.
   *
   * Judging all seven against one slider conflates them, and the result is incoherent: at an 8K
   * prompt a Spark fails interactive chat on a 6.4s wait while passing "coding agent", which
   * does everything chat does and more. An agent is not easier than chat; it was simply being
   * graded without a latency term. Each archetype is now measured at the prompt it would really
   * send — a completion popup sees a few hundred tokens, a RAG query tens of thousands.
   */
  typicalPromptTokens: number;
  /**
   * Whether `typicalPromptTokens` is *new* tokens arriving into a session already in the cache.
   *
   * The modelling decision behind #23, declared per archetype rather than assumed for all of them.
   * A coding agent's turn sends ~16K new tokens and re-reads nothing: prefix caching is on by
   * default in vLLM and available in llama.cpp, and a client that re-sent its whole history every
   * turn would be paying quadratically for nothing. A RAG query is a fresh document and a
   * completion popup is the current buffer, so neither has a prefix to speak of.
   *
   * Chat is the honest second candidate and is deliberately not declared: it is back-and-forth by
   * its own description, and under the same prefix caching it re-reads nothing either. Deferred
   * rather than dismissed, because it is not free — at an 8K context a 1K turn against 7K resident
   * is ~15x the pairs, judged against a 2s bar, so it would regrade chat on slow rigs and wants its
   * own evidence rather than a change of one flag.
   *
   * Opt-in, so the six that say nothing are evaluated exactly as before. That is not tidiness: the
   * calibration anchors are single-prompt measurements, and a flag that quietly applied to all
   * seven would have moved them.
   */
  prefixIsCached?: true;
}

export interface WorkloadVerdict {
  workload: Workload;
  fitness: Fitness;
  /** One line, in words, saying what decides it. Never omitted. */
  reason: string;
}

export const WORKLOADS: readonly Workload[] = [
  {
    id: 'chat',
    label: 'Interactive chat',
    description: 'One person, back and forth, reading as it types.',
    typicalPromptTokens: 1024,
  },
  {
    id: 'completion',
    label: 'Inline code completion',
    description: 'Suggestions as you type. The tightest latency budget here.',
    typicalPromptTokens: 512,
  },
  {
    id: 'agent',
    label: 'Coding agent',
    description: 'Long multi-turn sessions over a large codebase.',
    typicalPromptTokens: 16384,
    // The only archetype whose prompt is an increment rather than a whole request. See
    // `prefixIsCached` — this is the declaration #23 says has to come before the arithmetic.
    prefixIsCached: true,
  },
  {
    id: 'rag',
    label: 'RAG / document Q&A',
    description: 'Big retrieved prompt, short answer. Prefill dominates.',
    typicalPromptTokens: 32768,
  },
  {
    id: 'long-context',
    label: 'Long-context analysis',
    description: 'Whole documents or repositories held in one window.',
    typicalPromptTokens: 131072,
  },
  {
    id: 'batch',
    label: 'Batch / offline',
    description: 'Overnight jobs where only total throughput matters.',
    typicalPromptTokens: 4096,
  },
  {
    id: 'serving',
    label: 'Multi-user serving',
    description: 'Several people at once, each with their own cache.',
    typicalPromptTokens: 2048,
  },
] as const;

const BY_ID = new Map(WORKLOADS.map((w) => [w.id, w]));

function workload(id: string): Workload {
  const found = BY_ID.get(id);
  if (!found) throw new Error(`Unknown workload: ${id}`);
  return found;
}

/**
 * Every bar, named once.
 *
 * The `good` tiers each stated theirs twice — once in the predicate, once in the sentence that
 * explains missing it — so `15`, `30`, `25`, `65536` and `5` were all written down in two places
 * that had to agree by hand. That is precisely the drift `needs()` was introduced to remove one
 * level down, still live one level up. Nothing was wrong: the tests pin the agreement. What was
 * wrong is that nothing *made* them agree, and this file's own history is a list of the times two
 * copies of one number stopped matching.
 *
 * The `tight` bars are here too, and for a different reason — most of them appear only in a
 * predicate, and a number written once cannot drift. They are here because a tier means nothing
 * except relative to the one above it, and the two were forty lines apart. Side by side, a `good`
 * bar looser than its own `tight` bar is visible rather than merely something a test would catch.
 * `verdict.test.ts` asserts that ordering on every axis.
 *
 * Latency bars are upper bounds and rate, capacity and user-count bars are lower bounds, which is
 * why `good` is the *smaller* number on the first and the *larger* on the rest. Serving has one
 * further `good` bar that is not a number and so is not here: `headroomOf('serving') > 0`.
 *
 * Below `WORKLOADS` rather than above it so `long-context.good.prompt` can *be* the archetype's
 * declared request instead of a second copy of 131,072. That pair is on exactly the split this file
 * has been burned by: when the full window fits, the measurement comes from `typicalPromptTokens`
 * while the sentence names this — two literals, one of which a maintainer would edit.
 */
export const WORKLOAD_BARS = {
  chat: {
    good: { rate: 15, ttft: 2 },
    tight: { rate: 10, ttft: 5 },
  },
  completion: {
    good: { rate: 30, ttft: 0.4 },
    tight: { rate: 20, ttft: 1 },
  },
  agent: {
    good: { rate: 25, ttft: 10, session: AGENT_SESSION_CONTEXT },
    tight: { rate: 15, ttft: 30, session: AGENT_TIGHT_SESSION_CONTEXT },
  },
  rag: {
    good: { rate: 10, ttft: 5 },
    tight: { rate: 5, ttft: 20 },
  },
  'long-context': {
    good: { rate: 5, ttft: 120, prompt: workload('long-context').typicalPromptTokens },
    /**
     * The tight tier is graded at a smaller job, not at the same job leniently.
     *
     * The archetype sends 131,072 tokens and this tier accepts a machine that holds half that.
     * Those cannot be the same measurement: a rig with 80K of runnable context was accepted on its
     * 64K capacity and then timed reading a 128K prompt it has nowhere to put, and prefill is
     * quadratic, so the impossible request routinely failed the tier that had just admitted it —
     * and the reason quoted that impossible timing as the time to read the window it does hold.
     */
    tight: { rate: 2, ttft: 600, prompt: 65536 },
  },
  batch: {
    good: { aggregate: 5 },
    tight: { aggregate: 0.5 },
  },
  serving: {
    good: { rate: 10, ttft: 10, users: 4 },
    tight: { rate: 5, ttft: 30, users: 2 },
  },
} as const;

const BARS = WORKLOAD_BARS;

/**
 * What the long-context tight tier needs held, prompt plus room to answer.
 *
 * Derived from the bar rather than stated beside it, so the capacity the predicate tests and the
 * prompt it is then measured on cannot describe different jobs.
 */
const LONG_CONTEXT_TIGHT_NEEDS = BARS['long-context'].tight.prompt + RESPONSE_ALLOWANCE;

export interface VerdictInputs {
  /**
   * The selected configuration's placement, used for one thing only: deciding whether anything
   * can load at all.
   *
   * Deliberately *not* the placement any archetype is graded against — see `evaluateAt`, which
   * returns a placement planned at the archetype's own context. This one is destructured under a
   * different name inside so that a condition reaching for `placement` does not silently get the
   * slider's.
   */
  selectedPlacement: Placement;
  usage: UsageSpec;
  /** Largest context the rig can hold with every weight resident. */
  maxContextTokens: number;
  /**
   * Largest context that can actually be run, allowing offload. Used for the archetypes that
   * only need the configuration to work, as opposed to work comfortably — the resident figure
   * is zero for any offloaded rig, which would fail long-context on a card that would hold it.
   */
  runnableContextTokens: number;
  /**
   * The whole scenario re-evaluated at an archetype's own prompt.
   *
   * This is the *only* source of a measurement here — the slider's own decode estimate is
   * deliberately not an input, because batch and serving quietly went on using it after every
   * other archetype had moved across, and an unused field cannot be reached for by mistake.
   *
   * All three parts, not just prefill. Each was added after the previous omission produced a
   * verdict that disagreed with its own evidence:
   *
   *   - prefill alone left decode describing the *selected* cache while the latency described a
   *     16K agent turn, so an agent could be graded on 26.8 tok/s measured at a 512-token
   *     context when its own context decodes at 12.4;
   *   - decode and prefill left `placement` describing the slider's context, so serving was
   *     forced from `good` to `tight` by spill belonging to a scenario it is not graded at.
   *
   * A callback because each archetype needs its own; the engine is pure arithmetic, so seven
   * extra evaluations cost nothing worth optimising.
   */
  evaluateAt: (
    promptTokens: number,
    contextTokens: number,
    /**
     * Tokens already resident that `promptTokens` attends against rather than re-reads. Zero for
     * every archetype that sends a standalone prompt, which is all of them but the agent — see
     * `Workload.prefixIsCached`.
     */
    cachedPrefixTokens: number
  ) => {
    placement: Placement;
    decode: DecodeEstimate;
    prefill: PrefillEstimate;
  };
}

/**
 * Grade every archetype against one configuration.
 *
 * Ordered by how tight their latency budget is. That ordering is real but it is *only* about
 * latency, and it does not make the list a ladder — a claim this file used to make and that is
 * false wherever capacity rather than speed decides.
 *
 * Each archetype is graded at the prompt it really sends, which was itself a correction: judging
 * all seven against one slider conflated them. But it also means they ask for different amounts
 * of room. Inline completion sends 512 tokens and chat sends 1,024, so at 128 concurrent users
 * on a small card the chat cache can spill while the completion cache stays resident, and
 * completion is genuinely the easier workload there. Serving 128 autocomplete sessions really is
 * easier than serving 128 conversations.
 *
 * Capping completion at chat's grade would restore the appearance of a ladder by reporting a
 * failure that is not happening. The ordering claim is the thing that was wrong.
 */
export function judgeWorkloads(inputs: VerdictInputs): WorkloadVerdict[] {
  const { selectedPlacement, usage, maxContextTokens, runnableContextTokens, evaluateAt } = inputs;

  // Nothing else is meaningful if it cannot load. Said once, rather than seven times. Every
  // archetype evaluates at a context no smaller than the selected one, so nothing that fails
  // here could succeed at its own scenario.
  //
  // `fail` for all seven, including serving at one user: a model that does not load has been
  // measured against every archetype, and this is the one path that gives all seven the same
  // sentence — which is what lets the panel say it once above the list instead of seven times. An
  // ungraded row here would break that collapse and claim the question is still open when it is not.
  if (selectedPlacement.unsupported || selectedPlacement.impossible) {
    const reason =
      selectedPlacement.unsupported ?? 'The model does not fit and cannot spill to host RAM.';
    return WORKLOADS.map((w) => ({ workload: w, fitness: 'fail' as const, reason }));
  }

  /**
   * End-to-end batch throughput: prompts read *and* answers written.
   *
   * Decode alone was the whole grade, and for a job with a substantial prompt it is not the
   * total this archetype claims to measure. DeepSeek V3 Q4 on an EPYC 9654 decodes 10.4 tok/s
   * and passed, while reading its declared 4K prompt takes about 522 seconds — so a 512-token
   * answer completes at under 1 generated token per second end to end. The prompt pass is the
   * job for that configuration, and it was not being counted at all.
   *
   * Modelled as generated tokens per second over the whole request: `RESPONSE_ALLOWANCE` tokens
   * out, after the prefill wait, across every concurrent worker.
   */
  const batchAggregate = () => {
    const { decode, prefill } = at('batch');
    const workers = Math.max(1, usage.concurrency);

    // Every worker brings its own prompt, and `estimatePrefill` now prices the whole batch of
    // them in one figure. This used to multiply by the worker count to compensate for an estimate
    // that read `promptTokens` and never `concurrency`; doing that on top of the engine's own
    // scaling would charge every prompt `workers` times over. The compensation moved to where the
    // arithmetic is, and this reads the result rather than re-deriving it.
    const prefillSeconds = prefill.ttftSeconds;
    const decodeSeconds =
      (RESPONSE_ALLOWANCE * workers) / Math.max(decode.aggregateTokensPerSec, 1e-9);
    const seconds = prefillSeconds + decodeSeconds;
    return seconds > 0 ? (RESPONSE_ALLOWANCE * workers) / seconds : 0;
  };

  /** Says whose throughput `batchAggregate` is, whenever it is more than one worker's. */
  const batchWorkers = () => (usage.concurrency > 1 ? ` across ${usage.concurrency} workers` : '');

  /**
   * What an archetype needs to hold: its prompt plus room to answer. Every fit check reads this
   * one function, so a boundary cannot be stated differently in a condition and in its reason —
   * which is exactly how long-context ended up testing 65,536 in one place and 66,048 in the
   * other.
   */
  const needs = (id: string) => workload(id).typicalPromptTokens + RESPONSE_ALLOWANCE;
  const fits = (id: string) => runnableContextTokens >= needs(id);

  /**
   * Why a request does not fit, said the same way in all seven places.
   *
   * Each of these used to name only what the archetype *sends*, so at a runnable context of
   * exactly 32,768 the RAG row read "Only 32K of context fits — not enough for the 32K
   * document", which contradicts itself. The missing 512 tokens are the room to answer in, and
   * no rounding was involved — an earlier fix to the formatter could not have caught it. Stating
   * the requirement makes the shortfall visible as one whatever the numbers happen to be.
   */
  const shortfall = (id: string, sends: string) =>
    `Only ${ctx(runnableContextTokens)} of context fits — ${sends} needs ${ctx(needs(id))} with room to answer in.`;

  /**
   * The `good` bars a `tight` row misses, named — or nothing, when it misses none.
   *
   * Every archetype had the same hole and it was found one tier at a time. Once the fail-level
   * branches were exhausted the fallback printed healthy figures for a row that had been marked
   * down, with nothing saying which predicate did it: an agent reading "139 tok/s over 40K of
   * context, 4.0s per turn" is being shown three good numbers and no reason. A review named
   * completion, agent and long-context; the same gap was also in chat and rag.
   *
   * A shared builder rather than five more nested ternaries, because five hand-written copies of
   * one sentence is exactly how the serving and long-context versions came to disagree. Returns
   * `undefined` when every bar is met, so a genuinely `good` row keeps its positive sentence and
   * this cannot fire on one.
   */
  const shortOfGood = (...bars: (string | false)[]) => {
    const live = bars.filter((b): b is string => typeof b === 'string');
    if (live.length === 0) return undefined;
    const list =
      live.length === 1 ? live[0] : `${live.slice(0, -1).join(', ')} and ${live[live.length - 1]}`;
    return `Usable, but ${list}.`;
  };

  /**
   * Every archetype measured at its own scenario, decode included. Memoised on the scenario
   * rather than on the archetype id, because a tier can be graded at a working size its archetype
   * does not name — see the long-context tight tier, which is a smaller job and not the same job
   * judged leniently.
   */
  const cache = new Map<string, ReturnType<typeof evaluateAt>>();
  const evaluateOnce = (promptTokens: number, contextTokens: number, cachedPrefixTokens = 0) => {
    // The prefix is part of the key: the same prompt at the same context is a different pass
    // depending on whether it arrives into an empty cache or a full one.
    const key = `${promptTokens}:${contextTokens}:${cachedPrefixTokens}`;
    const existing = cache.get(key);
    if (existing) return existing;
    const fresh = evaluateAt(promptTokens, contextTokens, cachedPrefixTokens);
    cache.set(key, fresh);
    return fresh;
  };

  /**
   * What is already in the cache when this archetype's prompt arrives.
   *
   * The window minus what the turn itself needs, for an archetype that declares its prefix cached —
   * and nothing at all for one that does not, which is the standalone-prompt reading the calibration
   * anchors are measured at. Both readings are expressible; which one applies is the archetype's
   * declaration rather than a default this file picks.
   *
   * Through `needs(id)`, which is the turn *and its answer*, not `typicalPromptTokens` alone.
   * `contextTokens` is prompt plus generation, so subtracting only the prompt spent the whole
   * window on prefix and turn and left the reply nowhere: at a 64K session the occupancy came to
   * 66,048 in a 65,536 window. The tell is at the boundary — `cachedPrefix(id, needs(id))` returned
   * 512, claiming the room to answer as cached history, for a scenario that by definition has no
   * history at all.
   *
   * `needs` because this file already made that mistake with a hand-written copy of the same
   * boundary and wrote a comment about it: a limit stated twice is a limit that will disagree with
   * itself. Raised by Codex on PR #31.
   */
  const cachedPrefix = (id: string, contextTokens: number) =>
    workload(id).prefixIsCached ? Math.max(0, contextTokens - needs(id)) : 0;

  const at = (id: string) => {
    const contextTokens = Math.max(usage.contextTokens, needs(id));
    return evaluateOnce(
      workload(id).typicalPromptTokens,
      contextTokens,
      cachedPrefix(id, contextTokens)
    );
  };

  const rateOf = (id: string) => at(id).decode.perUserTokensPerSec;
  const ttftOf = (id: string) => at(id).prefill.ttftSeconds;
  /** Headroom as planned for *this* archetype's context, never the slider's. */
  const headroomOf = (id: string) => at(id).placement.headroomBytes;

  const chatTtft = ttftOf('chat');
  const completionTtft = ttftOf('completion');
  /**
   * Read through `ttftOf` at each use rather than bound here, unlike its two neighbours.
   *
   * Every serving predicate and every serving sentence is reached only after the user-count
   * conjunct, so at the default concurrency of 1 the serving scenario is never evaluated at all —
   * as it was not before this archetype gained a latency term. `evaluateOnce` memoises, so the
   * repeated calls cost nothing; binding it here would add a whole `evaluateAt` to every render of
   * the common case, to grade a bar that case never sees.
   *
   * That used to be two binary searches on top of a placement and both speed estimates. Since #17
   * the callback runs only the placement and the estimates, so the saving is smaller — but an
   * estimate is not free, and the reason to keep this lazy is unchanged.
   */
  const servingTtft = () => ttftOf('serving');

  /**
   * The one archetype whose defining parameter comes from the slider rather than from itself — and
   * the scenario where that parameter says the question was never asked.
   *
   * Every other archetype declares the prompt it really sends, so it is graded at its own scenario
   * whatever the sliders say. Concurrency is not declarable that way: it is the whole content of
   * "multi-user", and one user is a scenario in which multi-user serving simply has not been tried.
   * Graded, it read `○ No` in critical red with a sentence telling the reader to move a slider —
   * the strongest negative in the vocabulary standing in for "you have not configured this yet",
   * beside rows where it means "this machine cannot do it" (#75).
   *
   * The other three conjuncts are what keep the ungraded state from swallowing a real answer, and
   * each is the same boundary a sentence below leads with. One expression, read by the tier and by
   * its reason, because this file's history is a list of the times a condition and the sentence
   * explaining it were written down twice.
   *
   * **`fits` is the capacity half**: a rig that cannot hold one 2K turn cannot hold two of them
   * either, so that failure is measured at any concurrency and stays `fail`.
   *
   * **The rate and TTFT conjuncts are the other half, and the first version of this omitted them**
   * (found in review on #94). At one user the serving scenario *is* measured — `at('serving')` reads
   * `usage.concurrency`, so a 2K turn at a single user is a real evaluation — and if that measurement
   * already misses the tight tier, more users cannot rescue it: per-user decode is non-increasing in
   * concurrency because the weights are read once per step however many are waiting, and prefill work
   * grows with it because one long prompt already saturates the units. Both directions are asserted
   * over the catalog rather than argued: 1,292 drivable model/device/runtime/quant combinations, zero
   * cases where per-user rate rose or TTFT fell as concurrency went 1 → 2 → 4 → 8.
   *
   * So the omission was not a corner: **384 of those 1,292 miss a tight bar at one user** — a 5090
   * running Llama-3.1-70B at BF16 decodes 0.58 tok/s against a 5 tok/s bar, and 0.57 at two users —
   * and every one of them read "Not measured" for a failure the engine had already proved. Hiding a
   * definitive failure behind "you have not configured this yet" is the same class of defect as #75
   * itself, pointing the other way.
   *
   * The cost is that serving is now evaluated at the default concurrency of 1, which the note on
   * `servingTtft` above was written to avoid. That saving is not available any more: whether the
   * question was asked cannot be answered without taking the measurement. `evaluateOnce` memoises,
   * so it is one placement and two estimates added to a render that already runs six archetypes.
   */
  const servingNotTried =
    fits('serving') &&
    usage.concurrency < BARS.serving.tight.users &&
    rateOf('serving') >= BARS.serving.tight.rate &&
    servingTtft() <= BARS.serving.tight.ttft;

  const ragPrefill = at('rag').prefill;
  const ragFits = fits('rag');

  /**
   * How fast a document is read, as opposed to how fast the machine reads documents.
   *
   * `prefillTokensPerSec` is `(promptTokens * batch) / ttft` — deliberately machine-wide, so it
   * holds steady as concurrency rises while `ttftSeconds` grows with it. Printed unqualified beside
   * "31s for a 32K document" it was eight times the rate that document is read at, and the two
   * numbers in one sentence did not divide into each other. Dividing the batch back out makes them
   * reconcile exactly: the printed length over the printed rate is the printed wait. The Telemetry
   * tile took the other route and labelled its figure "across all N prompts in flight"; that works
   * for a tile reporting the machine, not for a sentence whose subject is one document.
   */
  const ragPerDocumentTokensPerSec =
    ragPrefill.prefillTokensPerSec / Math.max(1, usage.concurrency);

  /**
   * Agent, measured once at the session context its tiers recommend rather than at its 16K turn.
   *
   * `agentSession` is whichever session the rig can hold, and every agent predicate and every
   * agent sentence reads this one evaluation — the `longMeasured` pattern, for the same reason.
   * The consequence is deliberate: a rig that holds 64K has its *tight* tier timed at 64K too,
   * though that tier would admit a 32K session. The reduced figure is for machines that cannot
   * hold more, not a lenient reading for machines that can, and splitting it — good at 64K, tight
   * at 32K — puts the grade and the sentence back on different measurements the moment `good`
   * fails and `tight` holds. With one value in both, `pass` implies `tight` by construction.
   */
  const holdsFullSession = runnableContextTokens >= AGENT_SESSION_CONTEXT;
  /**
   * The session that is actually evaluated — and therefore the only session the sentences may
   * name.
   *
   * The `Math.max` is the same floor every other archetype takes through `at()`: a configured
   * context larger than the tier's is the scenario the user is asking about, and the cache really
   * is that size. But the tier's bar and the evaluated size are then two different numbers, and
   * printing the bar beside a figure measured at the slider is this diff's own defect one
   * archetype further on: at a 128K slider the row said "10 tok/s with a 64K session in the
   * cache", failing a rig on evidence from a session twice the size of the one it named — and the
   * 64K it did claim would have been `tight`. The bar keeps its own sentence (`holdsFullSession`,
   * below); everything quoting a measurement quotes this.
   */
  const agentSession = Math.max(
    usage.contextTokens,
    holdsFullSession ? AGENT_SESSION_CONTEXT : AGENT_TIGHT_SESSION_CONTEXT
  );
  /**
   * The prefix the agent's turn is actually measured against — the session minus the turn *and the
   * room to answer it*, since `agentSession` is the whole window and both the 16K arriving into it
   * and the reply leaving it are part of that window, not on top of it.
   *
   * Bound rather than recomputed at each use, because the sentences below have to name this and not
   * `agentSession`. The first version of them said "against 64K already in the cache" for a prefix
   * of 48K — claiming a cache holding the whole window *before* the turn arrives, which is 80K of
   * working set in a window the placement sized at 64K. That is this file's own most-repeated
   * defect, reintroduced by the fix that was supposed to be about honesty.
   */
  const agentPrefix = cachedPrefix('agent', agentSession);
  const agentMeasured = evaluateOnce(
    workload('agent').typicalPromptTokens,
    agentSession,
    agentPrefix
  );
  const agentRate = agentMeasured.decode.perUserTokensPerSec;
  const agentTtft = agentMeasured.prefill.ttftSeconds;

  /**
   * Long-context, measured twice: once at the archetype's full request and once at the reduced
   * window its tight tier admits.
   *
   * `longMeasured` is whichever of those this machine can actually hold, and every long-context
   * *reason* reads it. Quoting the 128K timing to a rig that tops out at 80K described a request
   * it cannot make — and the figure was not merely unattainable but wrong for the job it does do,
   * since prefill is quadratic and the two differ by about fourfold rather than twofold.
   */
  const longTight = evaluateOnce(
    BARS['long-context'].tight.prompt,
    Math.max(usage.contextTokens, LONG_CONTEXT_TIGHT_NEEDS)
  );
  const holdsFullWindow =
    runnableContextTokens >= BARS['long-context'].good.prompt + RESPONSE_ALLOWANCE;
  const longMeasured = holdsFullWindow ? at('long-context') : longTight;
  const longPrompt = holdsFullWindow
    ? BARS['long-context'].good.prompt
    : BARS['long-context'].tight.prompt;

  return [
    judge('chat', {
      // Even a short conversation needs its own turn to fit — at 128 users on a small card the
      // runnable context can fall below 1K, and no amount of speed rescues that.
      pass:
        fits('chat') && rateOf('chat') >= BARS.chat.good.rate && chatTtft <= BARS.chat.good.ttft,
      tight:
        fits('chat') && rateOf('chat') >= BARS.chat.tight.rate && chatTtft <= BARS.chat.tight.ttft,
      why: () =>
        !fits('chat')
          ? shortfall('chat', 'one short exchange')
          : rateOf('chat') < BARS.chat.tight.rate
            ? `${fmt(rateOf('chat'))} tok/s reads slower than most people do.`
            : chatTtft > BARS.chat.tight.ttft
              ? `A ${secs(chatTtft)}s wait on a short message breaks the back-and-forth.`
              : (shortOfGood(
                  rateOf('chat') < BARS.chat.good.rate &&
                    `${fmt(rateOf('chat'))} tok/s is under the ${BARS.chat.good.rate} tok/s that keeps pace with reading`,
                  chatTtft > BARS.chat.good.ttft &&
                    `${secs(chatTtft)}s to first token is over the ${BARS.chat.good.ttft}s a conversation wants`
                ) ??
                `${fmt(rateOf('chat'))} tok/s, ${secs(chatTtft)}s to first token on a short message.`),
    }),
    judge('completion', {
      // A suggestion that arrives after you have typed the next line is worse than none.
      pass:
        fits('completion') &&
        rateOf('completion') >= BARS.completion.good.rate &&
        completionTtft <= BARS.completion.good.ttft,
      tight:
        fits('completion') &&
        rateOf('completion') >= BARS.completion.tight.rate &&
        completionTtft <= BARS.completion.tight.ttft,
      why: () =>
        !fits('completion')
          ? shortfall('completion', 'one suggestion')
          : completionTtft > BARS.completion.tight.ttft
            ? `${secs(completionTtft)}s to first token — the suggestion arrives after you have moved on.`
            : rateOf('completion') < BARS.completion.tight.rate
              ? `${fmt(rateOf('completion'))} tok/s is too slow to finish a line while you pause.`
              : // Both good-tier bars, through one builder. The throughput case had its own
                // branch and the latency case had none, so a 159 tok/s suggestion arriving in
                // 0.6s — correctly `tight` — was told its latency "stays inside the window where
                // a suggestion helps", which is the sentence for a row that passed.
                (shortOfGood(
                  rateOf('completion') < BARS.completion.good.rate &&
                    `${fmt(rateOf('completion'))} tok/s finishes a line slower than you type it`,
                  // This one bar is described rather than printed, and is the reason `BARS` is read
                  // for the *condition* here while the sentence names no number. `secs` ceils, so a
                  // measurement can no longer render as its own limit — but 0.4 is the sub-second
                  // case, where the limit and a near miss are one tenth apart and read as the same
                  // order of magnitude beside each other. The neighbouring test forbids the literal
                  // for that reason. Chat's 2s and rag's 5s are far enough from their measurements
                  // to state plainly, and do.
                  completionTtft > BARS.completion.good.ttft &&
                    `${secs(completionTtft)}s to first token is past what an inline suggestion can absorb`
                ) ??
                `${secs(completionTtft)}s to first token stays inside the window where a suggestion helps.`),
    }),
    judge('agent', {
      // Agents need all three: speed, headroom, and a prompt pass that does not stall each turn.
      // Omitting the latency term is what let a machine fail chat while "passing" this, which is
      // backwards — an agent does everything chat does, over a far larger prompt.
      //
      // Rate and latency both come from `agentMeasured`: one evaluation, planned with the session
      // each tier endorses in the cache rather than only the turn the archetype names. What that
      // buys is almost all on the decode side, and that is where the defect was — 8B BF16 on one
      // 4090 goes 49.7 -> 8.6 tok/s between the two, because a 64K cache spills the weights.
      //
      // It buys on the prefill side too, now that `estimatePrefill` can express the scenario. It
      // could not before #23: the linear and attention work both came from `promptTokens` alone,
      // so the session reached prefill only through the placement — on that same 4090 the turn and
      // the session differed by 1.5%, the streaming term, and on a resident rig they were identical
      // to the digit. The latency bars were being graded on a 16K prompt attending over *itself*.
      //
      // The agent declares `prefixIsCached`, so the turn is now 16K new tokens attending against
      // the resident session: `cachedPrefix('agent', agentSession)`. That is the reading the
      // archetype's `typicalPromptTokens: 16384` always implied and the estimator could not carry.
      // It makes this term *slower*, not faster: the prefix is the session minus what the turn
      // needs — 47.5K inside a 64K window, the half-K being the room to answer — and 16K attending
      // against 47.5K is about seven times the query-key pairs of 16K attending against itself. On
      // an 8B at Q4_K_M on one 5090 that takes the turn from 6.0s to 14s, which is the difference
      // between clearing the 10s bar and not.
      //
      // No other archetype declares it, so every single-prompt scenario, and every calibration
      // anchor, is evaluated exactly as before.
      pass:
        fits('agent') &&
        agentRate >= BARS.agent.good.rate &&
        runnableContextTokens >= BARS.agent.good.session &&
        agentTtft <= BARS.agent.good.ttft,
      tight:
        fits('agent') &&
        agentRate >= BARS.agent.tight.rate &&
        runnableContextTokens >= BARS.agent.tight.session &&
        agentTtft <= BARS.agent.tight.ttft,
      why: () =>
        !fits('agent')
          ? shortfall('agent', 'an agent turn')
          : // Between the 16.5K a turn needs and the 32K a session needs, the turn requirement
            // is satisfied — reporting it as the reason named a figure the configuration meets.
            runnableContextTokens < BARS.agent.tight.session
            ? `${ctx(runnableContextTokens)} of context holds a turn but not a session — an agent needs ${ctx(BARS.agent.tight.session)} to keep its history across steps.`
            : agentTtft > BARS.agent.tight.ttft
              ? `${secs(agentTtft)}s to read a 16K turn against the ${ctx(agentPrefix)} already in the cache makes every step a wait.`
              : agentRate < BARS.agent.tight.rate
                ? `${fmt(agentRate)} tok/s once a ${ctx(agentSession)} session is in the cache makes a multi-step run take minutes per step.`
                : // All three good-tier bars. The capacity one is the easiest to hit and was the
                  // most misleading: a 40K-capped model reading "139 tok/s over 40K of context,
                  // 4.0s per turn" is three healthy figures explaining nothing.
                  (shortOfGood(
                    !holdsFullSession &&
                      `${ctx(runnableContextTokens)} of context is short of the ${ctx(BARS.agent.good.session)} a comfortable session keeps`,
                    agentRate < BARS.agent.good.rate &&
                      `${fmt(agentRate)} tok/s with a ${ctx(agentSession)} session in the cache is under the ${BARS.agent.good.rate} tok/s that keeps a step brisk`,
                    agentTtft > BARS.agent.good.ttft &&
                      `${secs(agentTtft)}s to read a 16K turn against the ${ctx(agentPrefix)} already in the cache is longer than a brisk step allows`
                  ) ??
                  // Capacity and rate are stated as separate clauses on purpose. "49 tok/s over
                  // 128K of context" reads as a rate that holds at 128K, which is the claim this
                  // tier was making and could not support; the rate belongs to the session it was
                  // measured at, and that session is named next to it.
                  `Holds ${ctx(runnableContextTokens)}; ${fmt(agentRate)} tok/s and ${secs(agentTtft)}s per turn with a ${ctx(agentSession)} session in the cache.`),
    }),
    judge('rag', {
      // The answer is short; the prompt is not. This lives or dies on prefill — but speed is
      // moot if the 32K prompt has nowhere to live: prefill is estimated at the archetype's own
      // prompt length, which deliberately ignores the configured context, so the fit has to be
      // checked separately or a fast machine could be graded good for a prompt it cannot hold.
      // Prefill dominates, but it is not the whole request: the answer still has to be written.
      // A configuration whose RAG-sized cache decodes at 1.4 tok/s takes six minutes over a
      // 512-token reply, and grading on TTFT alone called that usable while printing only the
      // prefill rate — a positive number standing in for the thing that was wrong.
      pass:
        ragFits &&
        ragPrefill.ttftSeconds <= BARS.rag.good.ttft &&
        rateOf('rag') >= BARS.rag.good.rate,
      tight:
        ragFits &&
        ragPrefill.ttftSeconds <= BARS.rag.tight.ttft &&
        rateOf('rag') >= BARS.rag.tight.rate,
      why: () =>
        !ragFits
          ? shortfall('rag', 'the 32K document this assumes')
          : ragPrefill.ttftSeconds > BARS.rag.tight.ttft
            ? `${secs(ragPrefill.ttftSeconds)}s to read a 32K document before answering.`
            : rateOf('rag') < BARS.rag.tight.rate
              ? `Reads the document in ${secs(ragPrefill.ttftSeconds)}s, then answers at ${fmt(rateOf('rag'))} tok/s — minutes for a short reply.`
              : // Tight on the answer alone, or on the read alone. The first had its own branch —
                // the prompt figure by itself reads as a pass — and the second had none, so a
                // twelve-second read of a 32K document was reported as though it were quick.
                (shortOfGood(
                  ragPrefill.ttftSeconds > BARS.rag.good.ttft &&
                    `${secs(ragPrefill.ttftSeconds)}s to read the document is over the ${BARS.rag.good.ttft}s bar`,
                  rateOf('rag') < BARS.rag.good.rate &&
                    `it answers at only ${fmt(rateOf('rag'))} tok/s`
                ) ??
                // `fmt`, not `Math.round`: a rate fails by being too small, so rounding one up in
                // a sentence about how fast this reads is the same flattery the formatters exist
                // to prevent — 999.6 tok/s printed as "1000" beside its own unrounded wait.
                `${fmt(ragPerDocumentTokensPerSec)} tok/s through a document — ${secs(ragPrefill.ttftSeconds)}s for a 32K one.`),
    }),
    judge('long-context', {
      // Offload-aware: the resident figure is zero for any spilled configuration, which would
      // fail a card that holds 128K of KV perfectly well once its weights are on the host.
      //
      // Capacity was the whole grade, and it was the last archetype without a speed term —
      // which is exactly the configuration that route rewards. DeepSeek V3 at BF16 on one 5090
      // offloads 98% of its weights and therefore *reaches* 163,840 tokens, so it passed while
      // taking eighteen minutes to first token and answering at 0.87 tok/s. Holding a window is
      // not the same as being able to work in it.
      //
      // The budgets are wide on purpose. Reading a whole repository is not an interactive act
      // and nobody expects it to be, so two minutes to first token is comfortable here where it
      // would be catastrophic for chat. Ten minutes and 2 tok/s is the edge of a thing you would
      // still start and walk away from.
      //
      // Each tier is timed at the job it admits. The tight tier accepts a machine holding 64K, so
      // it is measured on a 64K prompt; timing it on the archetype's full 128K request charged it
      // for a prompt it cannot hold, and quadratic prefill made that gap large rather than
      // academic.
      pass:
        runnableContextTokens >= BARS['long-context'].good.prompt + RESPONSE_ALLOWANCE &&
        ttftOf('long-context') <= BARS['long-context'].good.ttft &&
        rateOf('long-context') >= BARS['long-context'].good.rate,
      //
      // `longMeasured`, not `longTight`: the grade has to come from the measurement the reason
      // quotes. Reading `longTight` here graded a rig that holds 160K on its 64K job while the
      // sentence beside it reported 1046s against a 600s bar — the same contradiction this whole
      // file exists to prevent, reintroduced by a fix aimed at rigs that cannot hold 128K at all.
      // With one measurement in both, `pass` implies `tight` by construction rather than by the
      // incidental monotonicity of prefill in prompt length.
      tight:
        runnableContextTokens >= LONG_CONTEXT_TIGHT_NEEDS &&
        longMeasured.prefill.ttftSeconds <= BARS['long-context'].tight.ttft &&
        longMeasured.decode.perUserTokensPerSec >= BARS['long-context'].tight.rate,
      why: () =>
        // Reports the bar that actually rejected it. `shortfall` would name the archetype's
        // 131,584, so a rig holding 65K was told it needed 128.5K when roughly another 1K would
        // have made it `tight` — an upgrade requirement twice the size of the real one.
        runnableContextTokens < LONG_CONTEXT_TIGHT_NEEDS
          ? `Only ${ctx(runnableContextTokens)} of context fits — these jobs assume a 128K window, and even the reduced bar needs ${ctx(LONG_CONTEXT_TIGHT_NEEDS)} with room to answer in.`
          : longMeasured.prefill.ttftSeconds > BARS['long-context'].tight.ttft
            ? `Holds ${ctx(runnableContextTokens)}, but takes ${secs(longMeasured.prefill.ttftSeconds)}s to read ${ctx(longPrompt)} of it before saying anything.`
            : longMeasured.decode.perUserTokensPerSec < BARS['long-context'].tight.rate
              ? `Holds ${ctx(runnableContextTokens)} and answers at ${fmt(longMeasured.decode.perUserTokensPerSec)} tok/s — the window fits, the work does not.`
              : // Every good-tier bar this misses, through the same builder as the other tiers.
                //
                // The capacity bar states the allowance-adjusted figure. Written as a bare 128K it
                // contradicted itself in the 512-token band above 131,072: a rig holding 131,300
                // read "holds 128.2K — short of the 128K", because `holdsFullWindow` is about the
                // prompt *plus room to answer* and the sentence quoted only the prompt.
                //
                // Prefill and decode are here because either can be the sole cause once the full
                // window fits: DeepSeek V3 NVFP4 on one 5090 reaches 160K and prefills in 110s
                // but decodes at 3.2 tok/s, and the row used to mention only the offload and the
                // prompt pass — hiding the predicate that actually downgraded it.
                (shortOfGood(
                  // No adjective on the read: reaching this bar guarantees only the 600s tight
                  // limit, not the 120s comfortable one, so calling a 590s read "comfortable"
                  // would be the same positive-language-on-a-downgrade this whole helper exists
                  // to remove. The measurement is printed instead, which is not an opinion.
                  !holdsFullWindow &&
                    `it holds ${ctx(runnableContextTokens)}, short of the ${ctx(BARS['long-context'].good.prompt + RESPONSE_ALLOWANCE)} a full window needs with room to answer in — it reads the ${ctx(longPrompt)} it can hold in ${secs(longMeasured.prefill.ttftSeconds)}s`,
                  holdsFullWindow &&
                    longMeasured.prefill.ttftSeconds > BARS['long-context'].good.ttft &&
                    `${secs(longMeasured.prefill.ttftSeconds)}s to read ${ctx(longPrompt)} is a long wait before it says anything`,
                  holdsFullWindow &&
                    longMeasured.decode.perUserTokensPerSec < BARS['long-context'].good.rate &&
                    `it answers at ${fmt(longMeasured.decode.perUserTokensPerSec)} tok/s`
                ) ??
                (runnableContextTokens > maxContextTokens
                  ? `Reaches ${ctx(runnableContextTokens)}, though only with weights offloaded — ${secs(longMeasured.prefill.ttftSeconds)}s to read ${ctx(longPrompt)}.`
                  : `Holds ${ctx(runnableContextTokens)} at this concurrency, ${secs(longMeasured.prefill.ttftSeconds)}s to read ${ctx(longPrompt)}.`)),
    }),
    judge('batch', {
      // No latency budget at all — but the request still has to fit, and the throughput has to
      // be measured at the batch scenario rather than at whatever the slider says.
      // Rescaled with the metric. These were 5 and 1 against decode-only throughput; end-to-end
      // is strictly smaller for any job with a prompt, so keeping the same numbers would have
      // silently tightened the grade rather than corrected it. An overnight run is about eight
      // hours: 0.5 tok/s is ~28 replies of 512 tokens, which is a real batch job and not a
      // comfortable one; 5 tok/s is ~280.
      pass: fits('batch') && batchAggregate() >= BARS.batch.good.aggregate,
      tight: fits('batch') && batchAggregate() >= BARS.batch.tight.aggregate,
      why: () =>
        !fits('batch')
          ? shortfall('batch', 'a batch job')
          : // The same qualifier the RAG sentence needed, on the other aggregate in this file:
            // this figure sums every worker, so at 32 of them it is 32x what one job finishes at.
            // Nothing here divides into it — no per-job time is printed beside it — but "end to
            // end" describes the prompt-plus-answer span, not the worker count, and read alone it
            // invites the per-job reading. Telemetry's wording, for the same reason.
            batchAggregate() >= BARS.batch.good.aggregate
            ? `${fmt(batchAggregate())} tok/s end to end${batchWorkers()} — latency does not matter here, only the total.`
            : `${fmt(batchAggregate())} tok/s end to end${batchWorkers()}, prompts included, makes even an overnight run small.`,
    }),
    judge('serving', {
      // Every concurrent user brings their own cache, which is what actually runs out.
      //
      // And every concurrent user brings their own prompt, which is what they wait on. Capacity
      // and decode were the whole grade, so this was the one archetype with no latency term at
      // all — Llama 3.1 8B Q4_K_M on an EPYC 9654 at four users fits and decodes 39.8 tok/s
      // apiece, and was reported healthy while the four 2K prompts take about 165 seconds before
      // anyone sees a token. That gap grew rather than shrank when `estimatePrefill` learned about
      // concurrency: the estimate became right and nothing on this path read it.
      //
      // The budgets are looser than chat's 2s and 5s on the same reasoning that makes them tight
      // there. A shared deployment queues by design — `estimatePrefill` prices one batched pass,
      // so this is the wait every admitted prompt sees, not the first-served one's — and a few
      // seconds of queue is what a served user is actually buying. Ten seconds is the edge of
      // comfortable, thirty is the edge of tolerable, and minutes is a broken deployment however
      // fast it decodes once it starts.
      //
      // And at one user none of that is measured, which is a different statement from failing it.
      // See `servingNotTried`.
      unmeasured: servingNotTried,
      pass:
        fits('serving') &&
        usage.concurrency >= BARS.serving.good.users &&
        rateOf('serving') >= BARS.serving.good.rate &&
        headroomOf('serving') > 0 &&
        servingTtft() <= BARS.serving.good.ttft,
      tight:
        fits('serving') &&
        usage.concurrency >= BARS.serving.tight.users &&
        rateOf('serving') >= BARS.serving.tight.rate &&
        servingTtft() <= BARS.serving.tight.ttft,
      why: () =>
        !fits('serving')
          ? // `user`/`users` rather than a bare plural. This was "the one shortfall sentence
            // reachable at a single user" until #94 — the two below are reachable there now, and
            // they have a branch of their own for it, because the plural phrasing that suits four
            // users misdescribes one. It read "a turn each for 1 users".
            shortfall(
              'serving',
              `a turn each for ${usage.concurrency} ${usage.concurrency === 1 ? 'user' : 'users'}`
            )
          : servingNotTried
            ? // States the absence before the instruction. The row's status word already says "Not
              // measured"; a reason that opened on "Set concurrency…" was the only sentence in this
              // file that named no measurement at all, which is exactly right for a row that has
              // none — but it has to say so, because read on its own it is indistinguishable from
              // advice attached to a failure.
              `Not measured at ${usage.concurrency} ${usage.concurrency === 1 ? 'user' : 'users'} — set concurrency above ${BARS.serving.tight.users - 1} to see whether this holds several.`
            : usage.concurrency < BARS.serving.tight.users
              ? // Below the tier's own user count and *not* ungraded, which since #94 means one
                // thing: the single-user measurement already missed a tight bar, and no number of
                // users recovers it. So the sentence says which bar, at what the machine actually
                // did, and why adding users is the wrong direction — rather than borrowing the
                // plural copy below, which would read "0.6 tok/s each once 1 users share the
                // device" and describe a shared deployment that is not what was measured.
                rateOf('serving') < BARS.serving.tight.rate
                ? `${fmt(rateOf('serving'))} tok/s for a single served turn, under the ${BARS.serving.tight.rate} tok/s a shared deployment needs — every user added divides it further.`
                : `${secs(servingTtft())}s to a first token for one prompt, past the ${BARS.serving.tight.ttft}s bar — and each prompt added is read on top of it.`
              : rateOf('serving') < BARS.serving.tight.rate
                ? `${fmt(rateOf('serving'))} tok/s each once ${usage.concurrency} users share the device.`
                : servingTtft() > BARS.serving.tight.ttft
                  ? `${secs(servingTtft())}s before anyone sees a token — ${usage.concurrency} prompts have to be read first, and prefill does not batch the way decode does.`
                  : // Four good-tier bars now, through the same builder as the other tiers rather
                    // than the nested ternaries this used to hand-write. Those covered two causes in
                    // three branches and could not have absorbed a fourth: spill had a branch of its
                    // own above, which is why a spilling row never mentioned the user count, and the
                    // pair that did name both had to re-state the rate in each. This is the file's
                    // own lesson about hand-written copies of one sentence, applied to the tier that
                    // still had them.
                    (shortOfGood(
                      // No internal "and", and no internal comma: `shortOfGood` joins its items with
                      // commas and a final "and", so a bar carrying either of its own leaves the
                      // reader unable to see where one item stops. Three of these can be live at
                      // once.
                      usage.concurrency < BARS.serving.good.users &&
                        `it is measuring ${usage.concurrency} users against the ${BARS.serving.good.users} concurrent users a serving deployment is graded from`,
                      rateOf('serving') < BARS.serving.good.rate &&
                        `${fmt(rateOf('serving'))} tok/s each is under the ${BARS.serving.good.rate} tok/s a served user expects`,
                      // Spill can hold serving back while every printed figure looks healthy: the
                      // rate is fine, the fit is fine, and the reason said so.
                      headroomOf('serving') <= 0 &&
                        'the weights are spilling to host RAM so every additional user makes that worse rather than simply not fitting',
                      servingTtft() > BARS.serving.good.ttft &&
                        `${secs(servingTtft())}s to first token across ${usage.concurrency} queued prompts is longer than a served user waits`
                    ) ??
                    `${usage.concurrency} users at ${fmt(rateOf('serving'))} tok/s each, ${fmt(at('serving').decode.aggregateTokensPerSec)} aggregate, ${secs(servingTtft())}s to first token.`),
    }),
  ];
}

function judge(
  id: string,
  {
    pass,
    tight,
    unmeasured,
    why,
  }: {
    pass: boolean;
    tight: boolean;
    /**
     * The scenario does not exercise this archetype, so it is not graded at all. Optional, and
     * declared by exactly one tier — see `serving`.
     */
    unmeasured?: boolean;
    why: () => string;
  }
): WorkloadVerdict {
  return {
    workload: workload(id),
    /**
     * Ungraded wins over graded, and the order is the claim rather than a convenience: a scenario
     * that was never tried cannot be a pass. For the one tier that declares it the two are disjoint
     * by construction — serving is unmeasured below two users and cannot reach `tight` below two —
     * so this only decides what happens if a later tier makes them overlap.
     *
     * What a tier must *not* do is declare `unmeasured` for a scenario that already answers the
     * question. Serving guards its flag with `fits('serving')` for that reason: a rig with nowhere
     * to put one 2K turn has nowhere to put four of them, and that failure is measured.
     */
    fitness: unmeasured ? 'unmeasured' : pass ? 'good' : tight ? 'tight' : 'fail',
    reason: why(),
  };
}

/**
 * A measurement, never rounded *up*.
 *
 * These appear inside sentences explaining why a threshold was missed, so rounding 14.506 to
 * "15" produced "15 tok/s makes a multi-step session take minutes per step" against a minimum of
 * 15 — a number that appears to satisfy the condition it just failed. Same rule as `ctx`.
 *
 * Flooring alone is not enough at the bottom of the range, though: every positive value under
 * 0.1 floors to "0.0", so a measured 0.089s TTFT read "0.0s to first token" — a claim of zero
 * latency, which is a different kind of wrong from rounding up but wrong in the same direction.
 * Anything below a tenth is reported as an upper bound instead, which is both true and still
 * never flattering.
 */
function fmt(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value >= 10) return Math.floor(value).toString();
  if (value > 0 && value < 0.1) return '<0.1';
  return (Math.floor(value * 10) / 10).toFixed(1);
}

/**
 * A latency, never rounded *down*.
 *
 * The mirror of `fmt`, and it took a second finding to see that the direction has to follow the
 * *bound*, not the quantity. A rate fails by being too small, so flooring is what keeps it from
 * looking sufficient. A latency fails by being too large, so flooring is exactly what makes it
 * look sufficient: 0.486s against a 0.4s limit printed "0.4s to first token stays inside the
 * window", which is the same self-contradiction one sign the other way.
 */
function secs(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value >= 10) return Math.ceil(value).toString();
  if (value > 0 && value < 0.1) return '<0.1';
  return (Math.ceil(value * 10) / 10).toFixed(1);
}

/**
 * A context length, never rounded *up*.
 *
 * These figures come from a token-by-token binary search, so they land anywhere — and rounding
 * 32,700 to "32K" produced the self-contradiction "Only 32K of context fits — not enough for
 * the 32K document". Exact multiples read as whole units; anything else keeps a decimal, floored,
 * so a shortfall always looks like one.
 */
function ctx(tokens: number): string {
  const unit = tokens >= 1e6 ? 1e6 : 1024;
  const suffix = unit === 1e6 ? 'M' : 'K';
  if (tokens < 1024) return String(tokens);

  const scaled = tokens / unit;
  if (Number.isInteger(scaled)) return `${scaled}${suffix}`;
  return `${(Math.floor(scaled * 10) / 10).toFixed(1)}${suffix}`;
}
