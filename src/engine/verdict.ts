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
  placement: Placement;
  decode: DecodeEstimate;
  usage: UsageSpec;
  /** Largest context the rig can hold at this concurrency. */
  maxContextTokens: number;
  /**
   * Prefill re-estimated at an arbitrary prompt length.
   *
   * A callback rather than a single estimate, because each archetype is graded at its own
   * characteristic prompt — see `Workload.typicalPromptTokens`. The engine is pure arithmetic,
   * so seven extra evaluations cost nothing worth optimising.
   */
  prefillAt: (promptTokens: number) => PrefillEstimate;
}

/**
 * Grade every archetype against one configuration.
 *
 * Ordered by how tight their latency budget is, so the list reads as a ladder: if inline
 * completion passes, everything below it does too.
 */
export function judgeWorkloads(inputs: VerdictInputs): WorkloadVerdict[] {
  const { placement, decode, usage, maxContextTokens, prefillAt } = inputs;

  // Nothing else is meaningful if it cannot load. Said once, rather than seven times.
  if (placement.unsupported || placement.impossible) {
    const reason = placement.unsupported ?? 'The model does not fit and cannot spill to host RAM.';
    return WORKLOADS.map((w) => ({ workload: w, fitness: 'fail' as const, reason }));
  }

  const perUser = decode.perUserTokensPerSec;
  const aggregate = decode.aggregateTokensPerSec;

  /** Time to first token for the prompt this archetype would actually send. */
  const ttftFor = (id: string) => prefillAt(workload(id).typicalPromptTokens).ttftSeconds;

  const chatTtft = ttftFor('chat');
  const completionTtft = ttftFor('completion');
  const agentTtft = ttftFor('agent');
  const ragPrefill = prefillAt(workload('rag').typicalPromptTokens);

  return [
    judge('chat', {
      pass: perUser >= 15 && chatTtft <= 2,
      tight: perUser >= 10 && chatTtft <= 5,
      why: () =>
        perUser < 10
          ? `${fmt(perUser)} tok/s reads slower than most people do.`
          : chatTtft > 5
            ? `A ${fmt(chatTtft)}s wait on a short message breaks the back-and-forth.`
            : `${fmt(perUser)} tok/s, ${fmt(chatTtft)}s to first token on a short message.`,
    }),
    judge('completion', {
      // A suggestion that arrives after you have typed the next line is worse than none.
      pass: perUser >= 30 && completionTtft <= 0.4,
      tight: perUser >= 20 && completionTtft <= 1,
      why: () =>
        completionTtft > 1
          ? `${fmt(completionTtft)}s to first token — the suggestion arrives after you have moved on.`
          : perUser < 20
            ? `${fmt(perUser)} tok/s is too slow to finish a line while you pause.`
            : `${fmt(completionTtft)}s to first token stays inside the window where a suggestion helps.`,
    }),
    judge('agent', {
      // Agents need all three: speed, headroom, and a prompt pass that does not stall each turn.
      // Omitting the latency term is what let a machine fail chat while "passing" this, which is
      // backwards — an agent does everything chat does, over a far larger prompt.
      pass: perUser >= 25 && maxContextTokens >= 65536 && agentTtft <= 10,
      tight: perUser >= 15 && maxContextTokens >= 32768 && agentTtft <= 30,
      why: () =>
        maxContextTokens < 32768
          ? `Only ${ctx(maxContextTokens)} of context fits — an agent fills that within a few turns.`
          : agentTtft > 30
            ? `${fmt(agentTtft)}s to re-read a 16K prompt makes every step a wait.`
            : perUser < 15
              ? `${fmt(perUser)} tok/s makes a multi-step session take minutes per step.`
              : `${fmt(perUser)} tok/s over ${ctx(maxContextTokens)} of context, ${fmt(agentTtft)}s per turn.`,
    }),
    judge('rag', {
      // The answer is short; the prompt is not. This lives or dies on prefill.
      pass: ragPrefill.ttftSeconds <= 5,
      tight: ragPrefill.ttftSeconds <= 20,
      why: () =>
        ragPrefill.ttftSeconds > 20
          ? `${fmt(ragPrefill.ttftSeconds)}s to read a 32K document before answering.`
          : `${Math.round(ragPrefill.prefillTokensPerSec)} tok/s prompt processing — ${fmt(ragPrefill.ttftSeconds)}s for a 32K document.`,
    }),
    judge('long-context', {
      pass: maxContextTokens >= 131072,
      tight: maxContextTokens >= 65536,
      why: () =>
        maxContextTokens >= 131072
          ? `Holds ${ctx(maxContextTokens)} at this concurrency.`
          : `Caps out at ${ctx(maxContextTokens)} — short of the 128K these jobs assume.`,
    }),
    judge('batch', {
      // No latency budget at all. The only question is whether it runs and finishes.
      pass: aggregate >= 5,
      tight: aggregate >= 1,
      why: () =>
        aggregate >= 5
          ? `${fmt(aggregate)} tok/s aggregate — latency does not matter here, only the total.`
          : `${fmt(aggregate)} tok/s aggregate makes even an overnight run small.`,
    }),
    judge('serving', {
      // Every concurrent user brings their own cache, which is what actually runs out.
      pass: usage.concurrency >= 4 && perUser >= 10 && placement.headroomBytes > 0,
      tight: usage.concurrency >= 2 && perUser >= 5,
      why: () =>
        usage.concurrency < 2
          ? 'Set concurrency above 1 to see whether this holds several users.'
          : perUser < 5
            ? `${fmt(perUser)} tok/s each once ${usage.concurrency} users share the device.`
            : `${usage.concurrency} users at ${fmt(perUser)} tok/s each, ${fmt(aggregate)} aggregate.`,
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

function fmt(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return value >= 10 ? Math.round(value).toString() : value.toFixed(1);
}

function ctx(tokens: number): string {
  if (tokens >= 1e6) return `${(tokens / 1e6).toFixed(1)}M`;
  if (tokens >= 1024) return `${Math.round(tokens / 1024)}K`;
  return String(tokens);
}
