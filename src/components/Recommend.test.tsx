import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { Recommend } from './Recommend';
import { FALLBACK_RULE, QUANT_RULE, RANKING_RULE } from '@/engine/recommend';
import { DEFAULT_CONFIG, useConfig } from '@/store/config';
import { getQuant } from '@/data/quants';

/**
 * The recommendation panel's own rules (#138).
 *
 * `recommend.test.ts` owns the ranking; what is left for here is the two things a shortlist can get
 * wrong on the way to the screen — **whether the basis is printed**, and **whether clicking a row
 * loads the configuration the row names**. Both are places where a panel can look right and be
 * wrong: an unstated tie-break reads as a measurement, and a deep link that lands one coercion away
 * from what it promised shows the reader a different model than the one they picked.
 */

const panel = () => screen.getByRole('region', { name: /what this machine should run/i });

beforeEach(() => {
  useConfig.getState().replace(DEFAULT_CONFIG);
});

describe('the shortlist', () => {
  it('offers a headline and two runners-up, each a different model', () => {
    render(<Recommend />);
    const rows = within(panel()).getAllByRole('button', { name: /load this into the bench/i });

    expect(rows).toHaveLength(3);
    const names = rows.map((r) => r.textContent ?? '');
    expect(new Set(names).size, 'the shortlist repeats a row').toBe(3);
  });

  it('prints the rules it ranked by, rather than implying them', () => {
    // A ranked list with an unstated basis is an opinion wearing the chassis of a measurement, and
    // this is the panel that most looks like the latter. Verbatim from the engine, so a paraphrase
    // here cannot come apart from the comparator there.
    render(<Recommend />);
    expect(within(panel()).getByText(RANKING_RULE)).toBeInTheDocument();
    expect(within(panel()).getByText(QUANT_RULE)).toBeInTheDocument();
  });

  it('names the two axes the sweep fixed, since both are the reader’s current settings', () => {
    // Device count and cache precision are inputs, not universal facts — a shortlist read after
    // changing either is otherwise a claim about a configuration nobody is looking at.
    render(<Recommend />);
    expect(
      within(panel()).getByText(/Swept \d+ model and runtime pairings on/)
    ).toBeInTheDocument();
    // All three axes the sweep fixes, named: the machine, the cache and the user count. The last
    // is why this list has three assertions and not two — hardcoding concurrency let this panel and
    // the verdict strip grade the same configuration's batch row differently on one page.
    // Runtime-neutral wording: the sweep crosses runtimes and they name the same precision
    // differently — vLLM's one-byte cache is FP8, not integer Q8.
    expect(within(panel()).getByText(/16-bit cache at one user/)).toBeInTheDocument();
    expect(
      within(panel()).getByText(/each graded at the prompt its own workload sends/)
    ).toBeInTheDocument();
  });

  it('names the user counts the archetype declares, where it declares its own', async () => {
    /**
     * Serving's tiers are graded at four users and two whatever the slider says, so a footer
     * printing the reader's own count over that list named a number no grade in it used — on the
     * one archetype whose whole subject is user count (#172).
     *
     * The setting is still named, because it has not stopped mattering: the sweep plans every
     * placement at it, so it decides which rows load at all. What changed is that the sentence says
     * which of the two questions it answers.
     */
    const user = userEvent.setup();
    useConfig.getState().replace({ ...DEFAULT_CONFIG, concurrency: 12 });
    render(<Recommend />);
    await user.selectOptions(within(panel()).getByLabelText(/ranked for/i), 'serving');

    expect(
      within(panel()).getByText(/at the 4 and 2 users multi-user serving declares for itself/)
    ).toBeInTheDocument();
    expect(
      within(panel()).getByText(/user count above decides what loads here, not what grades/)
    ).toBeInTheDocument();
    // And not the reader's, in the position that used to claim the grades came from it.
    expect(within(panel()).queryByText(/cache at 12 users/)).not.toBeInTheDocument();
  });

  it('names the cache width per precision, not as a two-way split', () => {
    // The first version's ternary mapped every non-FP16 precision to "8-bit", so a Q4 selection
    // made the footer misstate an axis the ranking used.
    useConfig.getState().replace({ ...DEFAULT_CONFIG, runtimeId: 'llama.cpp', kvPrecision: 'q4' });
    render(<Recommend />);
    expect(within(panel()).getByText(/4-bit cache at/)).toBeInTheDocument();
  });

  it('states the fallback rule only when the fallback fired', () => {
    // Two headline rules beside one headline row is two claims about one thing. The default device
    // clears the bar, so this sentence must be absent here — `recommend.test.ts` owns the case
    // where it fires.
    render(<Recommend />);
    expect(within(panel()).queryByText(FALLBACK_RULE)).not.toBeInTheDocument();
  });

  it('grades every row without relying on colour', () => {
    // Same rule as the workload strip: a grade carried by hue alone is not a grade for everyone.
    render(<Recommend />);
    const rows = within(panel()).getAllByRole('button', { name: /load this into the bench/i });
    for (const row of rows) {
      // The icon is `aria-hidden` but still in `textContent`, so the word follows it.
      expect(row.textContent, row.textContent ?? '').toMatch(/^[●◐○](Yes|Tight|No):/);
    }
  });
});

describe('a row loads the configuration it names', () => {
  it('sets model, quant and runtime together', async () => {
    const user = userEvent.setup();
    render(<Recommend />);

    const first = within(panel()).getAllByRole('button', { name: /load this into the bench/i })[0];
    const label = first.textContent ?? '';
    await user.click(first);

    const config = useConfig.getState();
    /**
     * The property that makes this worth a test rather than a glance: `coerce` runs on every store
     * write, so setting the model before the runtime can bounce the quant off `quantApplies` in
     * between and land on the fallback format — the row would name one thing and the Bench would
     * load another. One `replace` is one coercion, and this asserts the result rather than the
     * mechanism.
     */
    // Against the label the row printed, for every field. The first draft wrote
    // `toContain(config.quantId === 'bf16' ? 'BF16' : '')`, and `toContain('')` is always true —
    // so the assertion only existed when the coerced quant happened to be bf16.
    expect(label).toContain(getQuant(config.quantId).label);
    expect(label.toLowerCase()).toContain(config.runtimeId.toLowerCase().slice(0, 4));
    expect(config.modelId).not.toBe('');
  });

  it('survives the store’s own coercion, so the loaded quant is the one offered', async () => {
    const user = userEvent.setup();
    render(<Recommend />);

    for (const row of within(panel()).getAllByRole('button', {
      name: /load this into the bench/i,
    })) {
      useConfig.getState().replace(DEFAULT_CONFIG);
      const label = row.textContent ?? '';
      await user.click(row);

      const { quantId, runtimeId } = useConfig.getState();
      // The store keeps what it was given rather than replacing it, which is the whole claim: a
      // recommendation the store rejects is a recommendation the reader cannot act on.
      // Against the *label*, which is what the row prints — the id is `awq_4bit` and the row
      // says "AWQ 4-bit".
      expect(label, `quant ${quantId} was coerced away`).toContain(getQuant(quantId).label);
      expect(['llama.cpp', 'vllm', 'mlx']).toContain(runtimeId);
    }
  });
});

describe('a row loads the scenario it was graded at', () => {
  it('carries the workload’s own context and prompt, not the sliders’', async () => {
    /**
     * The row's caption promises "its own numbers", and every candidate is graded at the scenario
     * its workload really sends. Spreading `config` kept whatever the sliders held, so clicking a
     * row scrolled to a budget bar describing a different job — and where the preserved context
     * made the candidate impossible, the workload the reader had just chosen from read `No`.
     */
    const user = userEvent.setup();
    useConfig.getState().replace({ ...DEFAULT_CONFIG, contextTokens: 131072, promptTokens: 65536 });
    render(<Recommend />);

    await user.click(
      within(panel()).getAllByRole('button', { name: /load this into the bench/i })[0]
    );

    const { contextTokens, promptTokens } = useConfig.getState();
    // The default archetype is chat, which sends 1,024 tokens.
    expect(promptTokens).toBe(1024);
    expect(contextTokens).toBeLessThan(131072);
    expect(contextTokens).toBeGreaterThanOrEqual(promptTokens);
  });
});

describe('the workload picker changes what is ranked', () => {
  it('re-sweeps for the archetype chosen', async () => {
    const user = userEvent.setup();
    render(<Recommend />);

    const before = within(panel())
      .getAllByRole('button', { name: /load this into the bench/i })
      .map((r) => r.textContent);

    await user.selectOptions(within(panel()).getByLabelText(/ranked for/i), 'batch');

    const after = within(panel())
      .getAllByRole('button', { name: /load this into the bench/i })
      .map((r) => r.textContent);

    // Overnight batch and interactive chat want different machines, so a picker that changed
    // nothing would mean the archetype is not reaching the sweep.
    expect(after).not.toEqual(before);
  });
});
