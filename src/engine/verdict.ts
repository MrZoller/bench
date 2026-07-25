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
    const perWorker = decode.aggregateTokensPerSec / Math.max(1, usage.concurrency);
    const seconds = prefill.ttftSeconds + RESPONSE_ALLOWANCE / Math.max(perWorker, 1e-9);
    return seconds > 0 ? (RESPONSE_ALLOWANCE * usage.concurrency) / seconds : 0;
  };

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
   * Every archetype measured at its own scenario, decode included. Memoised per id because the
   * conditions and their reasons each read it.
   */
  const cache = new Map<string, ReturnType<typeof evaluateAt>>();
  const at = (id: string) => {
    const existing = cache.get(id);
    if (existing) return existing;
    const w = workload(id);
    const fresh = evaluateAt(w.typicalPromptTokens, Math.max(usage.contextTokens, needs(id)));
    cache.set(id, fresh);
    return fresh;
  };

  const rateOf = (id: string) => at(id).decode.perUserTokensPerSec;
  const ttftOf = (id: string) => at(id).prefill.ttftSeconds;
  /** Headroom as planned for *this* archetype's context, never the slider's. */
  const headroomOf = (id: string) => at(id).placement.headroomBytes;

  const chatTtft = ttftOf('chat');
  const completionTtft = ttftOf('completion');
  const agentTtft = ttftOf('agent');
  const ragPrefill = at('rag').prefill;
  const ragFits = fits('rag');

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
              : `${fmt(rateOf('chat'))} tok/s, ${secs(chatTtft)}s to first token on a short message.`,
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
              : rateOf('completion') < 30
                ? // Tight on throughput alone: the latency sentence below is entirely positive,
                  // and printing it here explained why this passed rather than why it did not.
                  `${secs(completionTtft)}s to first token is quick, but ${fmt(rateOf('completion'))} tok/s finishes a line slower than you type it.`
                : `${secs(completionTtft)}s to first token stays inside the window where a suggestion helps.`,
    }),
    judge('agent', {
      // Agents need all three: speed, headroom, and a prompt pass that does not stall each turn.
      // Omitting the latency term is what let a machine fail chat while "passing" this, which is
      // backwards — an agent does everything chat does, over a far larger prompt.
      pass:
        fits('agent') && rateOf('agent') >= 25 && runnableContextTokens >= 65536 && agentTtft <= 10,
      tight:
        fits('agent') && rateOf('agent') >= 15 && runnableContextTokens >= 32768 && agentTtft <= 30,
      why: () =>
        !fits('agent')
          ? shortfall('agent', 'an agent turn')
          : // Between the 16.5K a turn needs and the 32K a session needs, the turn requirement
            // is satisfied — reporting it as the reason named a figure the configuration meets.
            runnableContextTokens < 32768
            ? `${ctx(runnableContextTokens)} of context holds a turn but not a session — an agent needs ${ctx(32768)} to keep its history across steps.`
            : agentTtft > 30
              ? `${secs(agentTtft)}s to re-read a 16K prompt makes every step a wait.`
              : rateOf('agent') < 15
                ? `${fmt(rateOf('agent'))} tok/s makes a multi-step session take minutes per step.`
                : `${fmt(rateOf('agent'))} tok/s over ${ctx(runnableContextTokens)} of context, ${secs(agentTtft)}s per turn.`,
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
              : `${Math.round(ragPrefill.prefillTokensPerSec)} tok/s prompt processing — ${secs(ragPrefill.ttftSeconds)}s for a 32K document.`,
    }),
    judge('long-context', {
      // Offload-aware: the resident figure is zero for any spilled configuration, which would
      // fail a card that holds 128K of KV perfectly well once its weights are on the host.
      pass: runnableContextTokens >= 131072 + RESPONSE_ALLOWANCE,
      tight: runnableContextTokens >= 65536 + RESPONSE_ALLOWANCE,
      why: () =>
        runnableContextTokens < 65536 + RESPONSE_ALLOWANCE
          ? shortfall('long-context', 'the 128K window these jobs assume')
          : runnableContextTokens > maxContextTokens
            ? `Reaches ${ctx(runnableContextTokens)}, though only with weights offloaded.`
            : `Holds ${ctx(runnableContextTokens)} at this concurrency.`,
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
          : batchAggregate() >= 5
            ? `${fmt(batchAggregate())} tok/s end to end — latency does not matter here, only the total.`
            : `${fmt(batchAggregate())} tok/s end to end, prompts included, makes even an overnight run small.`,
    }),
    judge('serving', {
      // Every concurrent user brings their own cache, which is what actually runs out.
      pass:
        fits('serving') &&
        usage.concurrency >= 4 &&
        rateOf('serving') >= 10 &&
        headroomOf('serving') > 0,
      tight: fits('serving') && usage.concurrency >= 2 && rateOf('serving') >= 5,
      why: () =>
        !fits('serving')
          ? shortfall('serving', `a turn each for ${usage.concurrency} users`)
          : usage.concurrency < 2
            ? 'Set concurrency above 1 to see whether this holds several users.'
            : rateOf('serving') < 5
              ? `${fmt(rateOf('serving'))} tok/s each once ${usage.concurrency} users share the device.`
              : // Spill is the one thing here that can hold serving back while every printed
                // figure looks healthy — the rate is fine, the fit is fine, and the reason said
                // so, leaving the actual constraint invisible.
                headroomOf('serving') <= 0
                ? `${usage.concurrency} users at ${fmt(rateOf('serving'))} tok/s each, but no headroom left — weights spill to host RAM, and one more user has nowhere to go.`
                : `${usage.concurrency} users at ${fmt(rateOf('serving'))} tok/s each, ${fmt(at('serving').decode.aggregateTokensPerSec)} aggregate.`,
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
