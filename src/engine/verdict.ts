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

export type Fitness = 'good' | 'tight' | 'fail';

/**
 * Room a workload needs for its answer, on top of its prompt.
 *
 * `UsageSpec.contextTokens` covers prompt *and* generation, so a 32,768-token document on a
 * model capped at 32,768 leaves nowhere to reply from. Small, because these archetypes ask
 * short questions of long inputs — but not zero, which is what the fit checks assumed.
 */
const RESPONSE_ALLOWANCE = 512;

/**
 * The working size the long-context *tight* tier is graded at.
 *
 * The archetype sends 131,072 tokens, and its tight tier accepts a machine that holds only half
 * that. Those cannot be the same measurement: a rig with 80K of runnable context was accepted on
 * its 64K capacity and then timed reading a 128K prompt it has nowhere to put, and prefill is
 * quadratic, so the impossible request routinely failed the tier that had just admitted it. The
 * reason then quoted that impossible timing as the time to read the window the machine holds.
 *
 * A tier that admits a smaller job has to measure the smaller job.
 */
const LONG_CONTEXT_TIGHT_PROMPT = 65536;
const LONG_CONTEXT_TIGHT_NEEDS = LONG_CONTEXT_TIGHT_PROMPT + RESPONSE_ALLOWANCE;

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
    contextTokens: number
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
  const evaluateOnce = (promptTokens: number, contextTokens: number) => {
    const key = `${promptTokens}:${contextTokens}`;
    const existing = cache.get(key);
    if (existing) return existing;
    const fresh = evaluateAt(promptTokens, contextTokens);
    cache.set(key, fresh);
    return fresh;
  };
  const at = (id: string) =>
    evaluateOnce(workload(id).typicalPromptTokens, Math.max(usage.contextTokens, needs(id)));

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
   * repeated calls cost nothing; binding it here would add a whole `evaluateAt` (two binary
   * searches, per #17) to every render of the common case, to grade a bar that case never sees.
   */
  const servingTtft = () => ttftOf('serving');
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
  const agentMeasured = evaluateOnce(workload('agent').typicalPromptTokens, agentSession);
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
    LONG_CONTEXT_TIGHT_PROMPT,
    Math.max(usage.contextTokens, LONG_CONTEXT_TIGHT_NEEDS)
  );
  const holdsFullWindow = runnableContextTokens >= 131072 + RESPONSE_ALLOWANCE;
  const longMeasured = holdsFullWindow ? at('long-context') : longTight;
  const longPrompt = holdsFullWindow ? 131072 : LONG_CONTEXT_TIGHT_PROMPT;

  return [
    judge('chat', {
      // Even a short conversation needs its own turn to fit — at 128 users on a small card the
      // runnable context can fall below 1K, and no amount of speed rescues that.
      pass: fits('chat') && rateOf('chat') >= 15 && chatTtft <= 2,
      tight: fits('chat') && rateOf('chat') >= 10 && chatTtft <= 5,
      why: () =>
        !fits('chat')
          ? shortfall('chat', 'one short exchange')
          : rateOf('chat') < 10
            ? `${fmt(rateOf('chat'))} tok/s reads slower than most people do.`
            : chatTtft > 5
              ? `A ${secs(chatTtft)}s wait on a short message breaks the back-and-forth.`
              : (shortOfGood(
                  rateOf('chat') < 15 &&
                    `${fmt(rateOf('chat'))} tok/s is under the 15 tok/s that keeps pace with reading`,
                  chatTtft > 2 &&
                    `${secs(chatTtft)}s to first token is over the 2s a conversation wants`
                ) ??
                `${fmt(rateOf('chat'))} tok/s, ${secs(chatTtft)}s to first token on a short message.`),
    }),
    judge('completion', {
      // A suggestion that arrives after you have typed the next line is worse than none.
      pass: fits('completion') && rateOf('completion') >= 30 && completionTtft <= 0.4,
      tight: fits('completion') && rateOf('completion') >= 20 && completionTtft <= 1,
      why: () =>
        !fits('completion')
          ? shortfall('completion', 'one suggestion')
          : completionTtft > 1
            ? `${secs(completionTtft)}s to first token — the suggestion arrives after you have moved on.`
            : rateOf('completion') < 20
              ? `${fmt(rateOf('completion'))} tok/s is too slow to finish a line while you pause.`
              : // Both good-tier bars, through one builder. The throughput case had its own
                // branch and the latency case had none, so a 159 tok/s suggestion arriving in
                // 0.6s — correctly `tight` — was told its latency "stays inside the window where
                // a suggestion helps", which is the sentence for a row that passed.
                (shortOfGood(
                  rateOf('completion') < 30 &&
                    `${fmt(rateOf('completion'))} tok/s finishes a line slower than you type it`,
                  // This one bar is described rather than printed. `secs` ceils, so a measurement
                  // can no longer render as its own limit — but 0.4 is the sub-second case, where
                  // the limit and a near miss are one tenth apart and read as the same order of
                  // magnitude beside each other. The neighbouring test forbids the literal for
                  // that reason. Chat's 2s and rag's 5s are far enough from their measurements to
                  // state plainly, and do.
                  completionTtft > 0.4 &&
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
      // It buys much less on the prefill side, and an earlier version of this comment claimed
      // otherwise. `estimatePrefill` takes its linear and attention work from `promptTokens`
      // alone, so the context reaches it only through the placement: on that same 4090 the turn
      // and the session differ by 1.5%, the streaming term, and on a resident rig they are
      // identical. So this term is the turn's prompt pass priced on the session's placement —
      // which is what the sentences below say, and is the right figure for an agent whose prefix
      // is cached. Charging a re-read of the session instead needs an estimator that can attend
      // new tokens against a cached prefix, which is #23. Passing the whole session as the prompt
      // is the option available today and is wrong in the other direction: it models an agent
      // with no prefix cache at all.
      pass:
        fits('agent') &&
        agentRate >= 25 &&
        runnableContextTokens >= AGENT_SESSION_CONTEXT &&
        agentTtft <= 10,
      tight:
        fits('agent') &&
        agentRate >= 15 &&
        runnableContextTokens >= AGENT_TIGHT_SESSION_CONTEXT &&
        agentTtft <= 30,
      why: () =>
        !fits('agent')
          ? shortfall('agent', 'an agent turn')
          : // Between the 16.5K a turn needs and the 32K a session needs, the turn requirement
            // is satisfied — reporting it as the reason named a figure the configuration meets.
            runnableContextTokens < AGENT_TIGHT_SESSION_CONTEXT
            ? `${ctx(runnableContextTokens)} of context holds a turn but not a session — an agent needs ${ctx(AGENT_TIGHT_SESSION_CONTEXT)} to keep its history across steps.`
            : agentTtft > 30
              ? `${secs(agentTtft)}s to re-read a 16K prompt with a ${ctx(agentSession)} session behind it makes every step a wait.`
              : agentRate < 15
                ? `${fmt(agentRate)} tok/s once a ${ctx(agentSession)} session is in the cache makes a multi-step run take minutes per step.`
                : // All three good-tier bars. The capacity one is the easiest to hit and was the
                  // most misleading: a 40K-capped model reading "139 tok/s over 40K of context,
                  // 4.0s per turn" is three healthy figures explaining nothing.
                  (shortOfGood(
                    !holdsFullSession &&
                      `${ctx(runnableContextTokens)} of context is short of the ${ctx(AGENT_SESSION_CONTEXT)} a comfortable session keeps`,
                    agentRate < 25 &&
                      `${fmt(agentRate)} tok/s with a ${ctx(agentSession)} session in the cache is under the 25 tok/s that keeps a step brisk`,
                    agentTtft > 10 &&
                      `${secs(agentTtft)}s to re-read a 16K prompt with ${ctx(agentSession)} in the cache is longer than a brisk step allows`
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
      pass: ragFits && ragPrefill.ttftSeconds <= 5 && rateOf('rag') >= 10,
      tight: ragFits && ragPrefill.ttftSeconds <= 20 && rateOf('rag') >= 5,
      why: () =>
        !ragFits
          ? shortfall('rag', 'the 32K document this assumes')
          : ragPrefill.ttftSeconds > 20
            ? `${secs(ragPrefill.ttftSeconds)}s to read a 32K document before answering.`
            : rateOf('rag') < 5
              ? `Reads the document in ${secs(ragPrefill.ttftSeconds)}s, then answers at ${fmt(rateOf('rag'))} tok/s — minutes for a short reply.`
              : // Tight on the answer alone, or on the read alone. The first had its own branch —
                // the prompt figure by itself reads as a pass — and the second had none, so a
                // twelve-second read of a 32K document was reported as though it were quick.
                (shortOfGood(
                  ragPrefill.ttftSeconds > 5 &&
                    `${secs(ragPrefill.ttftSeconds)}s to read the document is over the 5s bar`,
                  rateOf('rag') < 10 && `it answers at only ${fmt(rateOf('rag'))} tok/s`
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
        runnableContextTokens >= 131072 + RESPONSE_ALLOWANCE &&
        ttftOf('long-context') <= 120 &&
        rateOf('long-context') >= 5,
      //
      // `longMeasured`, not `longTight`: the grade has to come from the measurement the reason
      // quotes. Reading `longTight` here graded a rig that holds 160K on its 64K job while the
      // sentence beside it reported 1046s against a 600s bar — the same contradiction this whole
      // file exists to prevent, reintroduced by a fix aimed at rigs that cannot hold 128K at all.
      // With one measurement in both, `pass` implies `tight` by construction rather than by the
      // incidental monotonicity of prefill in prompt length.
      tight:
        runnableContextTokens >= LONG_CONTEXT_TIGHT_NEEDS &&
        longMeasured.prefill.ttftSeconds <= 600 &&
        longMeasured.decode.perUserTokensPerSec >= 2,
      why: () =>
        // Reports the bar that actually rejected it. `shortfall` would name the archetype's
        // 131,584, so a rig holding 65K was told it needed 128.5K when roughly another 1K would
        // have made it `tight` — an upgrade requirement twice the size of the real one.
        runnableContextTokens < LONG_CONTEXT_TIGHT_NEEDS
          ? `Only ${ctx(runnableContextTokens)} of context fits — these jobs assume a 128K window, and even the reduced bar needs ${ctx(LONG_CONTEXT_TIGHT_NEEDS)} with room to answer in.`
          : longMeasured.prefill.ttftSeconds > 600
            ? `Holds ${ctx(runnableContextTokens)}, but takes ${secs(longMeasured.prefill.ttftSeconds)}s to read ${ctx(longPrompt)} of it before saying anything.`
            : longMeasured.decode.perUserTokensPerSec < 2
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
                    `it holds ${ctx(runnableContextTokens)}, short of the ${ctx(131072 + RESPONSE_ALLOWANCE)} a full window needs with room to answer in — it reads the ${ctx(longPrompt)} it can hold in ${secs(longMeasured.prefill.ttftSeconds)}s`,
                  holdsFullWindow &&
                    longMeasured.prefill.ttftSeconds > 120 &&
                    `${secs(longMeasured.prefill.ttftSeconds)}s to read ${ctx(longPrompt)} is a long wait before it says anything`,
                  holdsFullWindow &&
                    longMeasured.decode.perUserTokensPerSec < 5 &&
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
      pass: fits('batch') && batchAggregate() >= 5,
      tight: fits('batch') && batchAggregate() >= 0.5,
      why: () =>
        !fits('batch')
          ? shortfall('batch', 'a batch job')
          : // The same qualifier the RAG sentence needed, on the other aggregate in this file:
            // this figure sums every worker, so at 32 of them it is 32x what one job finishes at.
            // Nothing here divides into it — no per-job time is printed beside it — but "end to
            // end" describes the prompt-plus-answer span, not the worker count, and read alone it
            // invites the per-job reading. Telemetry's wording, for the same reason.
            batchAggregate() >= 5
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
      pass:
        fits('serving') &&
        usage.concurrency >= 4 &&
        rateOf('serving') >= 10 &&
        headroomOf('serving') > 0 &&
        servingTtft() <= 10,
      tight:
        fits('serving') && usage.concurrency >= 2 && rateOf('serving') >= 5 && servingTtft() <= 30,
      why: () =>
        !fits('serving')
          ? shortfall('serving', `a turn each for ${usage.concurrency} users`)
          : usage.concurrency < 2
            ? 'Set concurrency above 1 to see whether this holds several users.'
            : rateOf('serving') < 5
              ? `${fmt(rateOf('serving'))} tok/s each once ${usage.concurrency} users share the device.`
              : servingTtft() > 30
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
                    usage.concurrency < 4 &&
                      `it is measuring ${usage.concurrency} users against the 4 concurrent users a serving deployment is graded from`,
                    rateOf('serving') < 10 &&
                      `${fmt(rateOf('serving'))} tok/s each is under the 10 tok/s a served user expects`,
                    // Spill can hold serving back while every printed figure looks healthy: the
                    // rate is fine, the fit is fine, and the reason said so.
                    headroomOf('serving') <= 0 &&
                      'the weights are spilling to host RAM so every additional user makes that worse rather than simply not fitting',
                    servingTtft() > 10 &&
                      `${secs(servingTtft())}s to first token across ${usage.concurrency} queued prompts is longer than a served user waits`
                  ) ??
                  `${usage.concurrency} users at ${fmt(rateOf('serving'))} tok/s each, ${fmt(at('serving').decode.aggregateTokensPerSec)} aggregate, ${secs(servingTtft())}s to first token.`),
    }),
  ];
}

function judge(
  id: string,
  { pass, tight, why }: { pass: boolean; tight: boolean; why: () => string }
): WorkloadVerdict {
  return {
    workload: workload(id),
    fitness: pass ? 'good' : tight ? 'tight' : 'fail',
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
