import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG } from '@/store/config';
import { configToShareSearch } from '@/store/url';
import { MODELS, getModel } from '@/data/catalog';
import { params, tokens } from '@/lib/format';
import { SETTING_LABELS } from '@/lib/stops';
import { HOST_RAM_UNCHECKED } from '@/lib/verdicts';
import { DETAIL_ANCHOR_ID } from '@/components/Matrix';
import { judgeWorkloads } from '@/engine/verdict';
import { kvSubstitutionFor } from '@/data/runtimes';
import { effectiveActiveParams } from '@/engine/weights';
import { marks } from '@/design/tokens';

/**
 * Wrapped rather than replaced, so every other test in this file still exercises the real verdict
 * layer. The spy exists only so the memoisation guard below can see whether grading ran at all.
 */
vi.mock('@/engine/verdict', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/engine/verdict')>();
  return { ...actual, judgeWorkloads: vi.fn(actual.judgeWorkloads) };
});

/**
 * Same treatment for `kvSubstitutionFor`, and for a reason worth stating.
 *
 * Every cache precision in the shipped catalog now has an established width (#38), so the KV
 * marker has no trigger — and an unreachable branch is one nobody notices breaking. The mechanism
 * is not dead: it fires for the next precision added without a width, which is exactly the case it
 * exists for and exactly the case no fixture can reach. Wrapping the real function lets one test
 * force that future state and check the surfaces still render it, while every other test in this
 * file goes on exercising the real catalog.
 */
vi.mock('@/data/runtimes', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/runtimes')>();
  return { ...actual, kvSubstitutionFor: vi.fn(actual.kvSubstitutionFor) };
});

import { atFullGrid, boundGridByDefault } from '@/test/grid';

/**
 * The Matrix's extent is bounded by default and the real grid opted into — the whole design,
 * and the fixture itself, live in `src/test/grid.ts` (#101, #115). The mock is declared here
 * because `vi.mock` is hoisted per test file and cannot ride an import; the fixture's own
 * preconditions are held in `src/test/grid.test.ts`.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

boundGridByDefault();

afterEach(() => {
  cleanup();
  useConfig.setState(DEFAULT_CONFIG);
});

/**
 * Surface-level tests. The arithmetic is pinned in the engine's own suite; what these guard is
 * that the Bench renders it truthfully — in particular that it never shows a confident number
 * for a configuration that cannot run.
 */
describe('the Bench', () => {
  it('renders the three verdicts as separate answers', () => {
    render(<App />);

    // Capacity, decode and TTFT are independent axes. Collapsing them into one score is the
    // thing this layout exists to refuse.
    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).getByText('Capacity')).toBeInTheDocument();
    expect(within(verdicts).getByText('Decode')).toBeInTheDocument();
    expect(within(verdicts).getByText('Time to first token')).toBeInTheDocument();
  });

  it('describes the budget bar for a screen reader rather than leaving it as bare divs', () => {
    render(<App />);
    const bar = screen.getByRole('img', { name: /allocatable used/i });
    expect(bar).toHaveAccessibleName(/Weights/);
    expect(bar).toHaveAccessibleName(/KV cache/);
  });

  it('offers the same figures as a table, for anyone who cannot use the bar', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /figures as a table/i }));
    // Named explicitly: the Envelope and the Matrix have tables of their own now.
    const table = screen.getByRole('table', { name: /Memory budget breakdown/i });
    expect(
      within(table).getByRole('rowheader', { name: 'Allocatable ceiling' })
    ).toBeInTheDocument();
  });

  /**
   * The one that matters. vLLM cannot drive a Mac, and the engine will still happily return
   * arithmetic for the pair — it has no opinion about which software exists. Showing that
   * arithmetic would be a plausible number for something that cannot happen.
   */
  it('shows no throughput at all when the runtime cannot drive the hardware', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).getAllByText('Unsupported')).toHaveLength(3);
    expect(within(verdicts).queryByText(/tok\/s per user/)).not.toBeInTheDocument();
  });

  /**
   * The subtler sibling of the unsupported case, and the more dangerous one: over the ceiling
   * with nowhere to spill. On unified memory `offloadFraction` is 0, so decode computes as
   * though every weight were resident at full bandwidth — a green "Fast" beside a red "Will not
   * run". The optimism is what makes it worth a test.
   */
  it('shows no throughput when the model cannot fit and cannot spill', async () => {
    const user = userEvent.setup();
    render(<App />);

    // DeepSeek-V3 rather than the default model, since #121: gpt-oss-120b at Q5_K_M is 78.8 GiB
    // against the 96 GB Mac's 72 GiB *default* — past a setting, not past the machine, so the
    // strip now says "Past the default allocation" there rather than the flat refusal this test
    // pins. 445.6 GiB is past any ceiling this machine can be tuned to.
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-96');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q5_k_m');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).queryByText(/tok\/s per user/)).not.toBeInTheDocument();
    expect(within(verdicts).queryByText('Fast')).not.toBeInTheDocument();
    expect(within(verdicts).getAllByText('Will not run').length).toBeGreaterThan(0);
  });

  it('hides the multi-device control on hardware that cannot shard', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    expect(screen.getByLabelText('Device count')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    expect(screen.queryByLabelText('Device count')).not.toBeInTheDocument();
    // Any split needs a link, not only a tensor-parallel one: `canShard` is
    // `interconnect !== undefined` and asks nothing about the runtime.
    expect(screen.getByText(/needs a transport between them/i)).toBeInTheDocument();
  });

  it('does not claim a model fits when the budget says otherwise', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    expect(screen.getByText(/Over the ceiling by/)).toBeInTheDocument();
    expect(screen.queryByText(/Why this fits/)).not.toBeInTheDocument();
  });

  it('labels pre-release hardware so a rumoured spec is never presented as fact', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m5-ultra-512');
    expect(screen.getByText(/specs may change/i)).toBeInTheDocument();
  });
});

/**
 * Guards for the review round on the Bench. Every one of these is the same failure in a
 * different costume: something confident asserted for a state where it is not true.
 */
describe('the Bench does not overclaim', () => {
  it('refuses MLX on unified memory that is not Apple', async () => {
    const user = userEvent.setup();
    render(<App />);

    // `unified-soc` covers the Spark, Strix Halo and Apple silicon; MLX drives only the last.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'dgx-spark');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).getAllByText('Unsupported').length).toBeGreaterThan(0);
    expect(within(verdicts).queryByText(/tok\/s per user/)).not.toBeInTheDocument();
  });

  it('does not call an offloaded configuration fast', async () => {
    const user = userEvent.setup();
    render(<App />);

    // 671B on a 32 GB card: runnable via offload, and slow because of it.
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    expect(screen.queryByText(/runs fast/i)).not.toBeInTheDocument();
    // Either explanation is honest; what must never appear is a claim of speed. Which one shows
    // depends on whether the engine's resident estimate would itself have been fast.
    const explained =
      screen.queryAllByText(/crossing the host bus/i).length +
      screen.queryAllByText(/Even resident it would be slow/i).length;
    expect(explained).toBeGreaterThan(0);
  });

  /**
   * `impossible` asks every device whether its cache and activations alone are over the ceiling,
   * because under a layer split the busiest card by *combined* load is not necessarily the one
   * holding the most cache. The sentence explaining the refusal has to quote the device that caused
   * it — rebuilt from this bar's own per-device figures it named the card being drawn, which is a
   * different card, and printed a figure comfortably *under* the ceiling it was citing as the
   * reason. A predicate and its sentence are one claim.
   */
  it('quotes the card that made it impossible, and says it is not the one being drawn', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Gemma 3 12B over three 4090s at 128K and 8 users: two cards take three full-attention layers
    // each and the third takes the remaining 42, so the card with the most cache is the one with
    // the fewest layers.
    await user.selectOptions(screen.getByLabelText('Model'), 'unsloth/gemma-3-12b-it');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-4090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    act(() => {
      useConfig.getState().set('contextTokens', 131072);
      useConfig.getState().set('concurrency', 8);
      useConfig.getState().set('deviceCount', 3);
    });

    // The refusal names the card it belongs to rather than implying the one above it — "the card
    // holding the most cache", not "the busiest card", because the engine's busiest device is
    // busiest by combined load and in this split is the one being drawn. Its quantity is named in
    // full: `floorBytesPerDevice` is cache plus activations, the term the segments label
    // Overhead, with a tail whose subject is that singular figure. The shared tail it replaced
    // counted a pair ("and neither can be offloaded") this branch never names (#128).
    expect(
      screen.getByText(/the card holding the most cache needs .* of cache and overhead/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/which cannot be offloaded/i)).toBeInTheDocument();
    expect(screen.queryByText(/neither can be offloaded/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/the cache and overhead alone need/i)).not.toBeInTheDocument();
  });

  it('explains a full card as a full card, not as a Mac', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    // The shared-memory explanation must not appear for a discrete GPU.
    expect(screen.queryByText(/no faster tier/i)).not.toBeInTheDocument();
  });

  it('shows why a hand-entered parameter count differs from the index', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    expect(screen.getByText(/Multi-Token Prediction/i)).toBeInTheDocument();
  });

  it('says when the catalog was generated', () => {
    render(<App />);
    expect(screen.getByText(/Model catalog generated/i)).toBeInTheDocument();
  });
});

/**
 * The budget bar past the point its own shape stops arguing (#73).
 *
 * The panel's claim is that it says how far over you are *structurally*, with an overflow region
 * beyond the ceiling rule, "because it turned red does not tell you by how much". That holds while
 * the overshoot is small and inverts past about 3x: `scale` follows the stack, so the budget becomes
 * the sliver, the segments fill the bar, and 448 GiB against a 31 GiB ceiling draws as "nearly
 * full". The multiple is the figure the picture stopped conveying, and it was stated nowhere.
 *
 * Split deliberately. Whether the rule stays *legible* where it lands is geometry — jsdom reports
 * every width here as 0 — and lives in `e2e/budget-overshoot.spec.ts`. The clause and the legend key
 * are DOM, so they belong here, where they run in a second.
 */
describe('the budget bar states an overshoot its shape cannot show', () => {
  /** The overflow line, addressed by the part of it that does not move. */
  const overflow = () => screen.getByText(/Over the ceiling by/);
  const budget = () => screen.getByRole('region', { name: /memory budget/i });

  /**
   * The issue's own URL, reached through the controls: DeepSeek V3 at Q4_K_M on one 5090 at 128K
   * and 8 users. 448 GiB used against a 31 GiB ceiling — 14.5x, with the rule 6.9% from the left.
   */
  const fourteenTimesOver = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    act(() => {
      useConfig.getState().set('contextTokens', 131072);
      useConfig.getState().set('concurrency', 8);
    });
  };

  it('says how many times over the ceiling the stack is', async () => {
    const user = userEvent.setup();
    render(<App />);
    await fourteenTimesOver(user);

    // Both figures, because they are different quantities and the absolute one was never the
    // problem: 417 GiB is the overflow, 14x is the stack. The second is the one the bar lost.
    expect(overflow()).toHaveTextContent(/Over the ceiling by 417 GiB/);
    expect(overflow()).toHaveTextContent(/The stack is 14x the ceiling/);
  });

  /**
   * And does not, where the shape still carries it. At 1.1x the ceiling rule sits 91% along, the
   * gap is plainly visible, and "over by 2.2 GiB" and "1.1x the ceiling" are one fact said twice —
   * a clause on every overflow is a clause people stop reading, including at 14x.
   */
  it('does not restate a small overshoot as a multiple', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Gemma 3 12B over three 4090s at 128K and 8 users: 25.2 GiB against a 23 GiB ceiling.
    await user.selectOptions(screen.getByLabelText('Model'), 'unsloth/gemma-3-12b-it');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-4090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    act(() => {
      useConfig.getState().set('contextTokens', 131072);
      useConfig.getState().set('concurrency', 8);
      useConfig.getState().set('deviceCount', 3);
    });

    expect(overflow()).toHaveTextContent(/Over the ceiling by/);
    expect(overflow()).not.toHaveTextContent(/The stack is/);
    expect(overflow()).not.toHaveTextContent(/x the ceiling/);
  });

  /**
   * The legend is the dependable identity channel — which is exactly why the mark that is hardest
   * to see is the one that must be in it. Every fill had a key and the rule did not, so a reader
   * met a dashed red line inside a blue fill with nothing on the page naming it.
   */
  it('keys the ceiling rule in the legend, and only while the rule is drawn', async () => {
    const user = userEvent.setup();
    render(<App />);

    // The opening scenario fits, so no rule is drawn and there is nothing to key.
    expect(within(budget()).queryByText('Ceiling')).not.toBeInTheDocument();

    await fourteenTimesOver(user);
    const key = within(budget()).getByText('Ceiling').closest('li');
    expect(key).not.toBeNull();
    // The ceiling's own figure, for the same reason every other row carries its bytes.
    expect(key).toHaveTextContent('31 GiB');
  });

  /**
   * The table says it the same way, because the table is the channel with no shape at all.
   *
   * It exists "for anyone who cannot use the bar", and the reader who cannot watch the bar invert
   * was the one still handed "1222%" for a component twelve times the size of the whole budget —
   * the exact form `multiple` was added to replace, in the one place there is nothing else to read.
   *
   * The KV row is the control, and it is why this is a per-row rule rather than a column-wide one:
   * 2.2x is still inside the neighbourhood of 1 where a percentage is the better form, and the 0.7
   * GiB overhead row would print as "0x" if the whole column switched.
   */
  it('states a large share as a multiple in the table, not as a four-digit percentage', async () => {
    const user = userEvent.setup();
    render(<App />);
    await fourteenTimesOver(user);

    await user.click(within(budget()).getByRole('button', { name: /figures as a table/i }));
    const table = within(budget()).getByRole('table', { name: /Memory budget breakdown/i });
    const row = (name: string) => within(table).getByRole('rowheader', { name }).closest('tr');

    expect(row('Weights')).toHaveTextContent('12x');
    expect(row('Weights')).not.toHaveTextContent('1222%');
    expect(row('KV cache')).toHaveTextContent('221%');
  });

  /**
   * One line weight, not two — the placement invariant, checked where it can be checked cheaply.
   *
   * The halo is `lineWidth + 2·gap` wide and centres the rule inside itself, so the ink lands on the
   * true ceiling position only while the two widths are the same number. Written as `border-l-2`
   * beside a halo sized from `marks.lineWidth`, a token bumped to 3 would distribute 2.5px per side
   * and move the rule half a pixel off the ceiling — the sentence and the line would stop being two
   * readings of one expression, and the e2e position check's 3px tolerance would not notice.
   *
   * Geometry is e2e's job and this asserts none: jsdom lays nothing out, and it cannot see a Tailwind
   * class either. What it can see is that the token reaches the element at all, which is the whole
   * repair — reinstating the literal empties this style and fails here.
   */
  it('draws the ceiling rule from the same line-weight token its halo is sized from', async () => {
    const user = userEvent.setup();
    render(<App />);
    await fourteenTimesOver(user);

    const bar = within(budget()).getByRole('img', { name: /allocatable used/i });
    // Two children: the segment row, then the rule in its halo.
    const halo = bar.children[bar.children.length - 1] as HTMLElement;
    const line = halo.children[0] as HTMLElement;

    expect(line.style.borderLeftWidth).toBe(`${marks.lineWidth}px`);
    expect(halo.style.width).toBe(`${marks.lineWidth + marks.gap * 2}px`);
    // And the halo starts one gap early, which is what leaves the centred ink on the ceiling.
    expect(halo.style.left).toContain(`- ${marks.gap}px`);
  });
});

/**
 * A mark drawn on top of another mark is named where a reader can find it (#73's class).
 *
 * The budget bar's ceiling rule was the case the issue named, and the audit of the rest found the
 * same shape twice more: three overlay marks, all of them keyed only in an `aria-label` or in a hue,
 * none of them in a legend. A legend is the channel that does not depend on the mark being legible,
 * which is exactly the property an overlay cannot promise — it is drawn on whatever is beneath it.
 *
 * Whether each mark is *distinguishable* where it lands is geometry and pixels, and lives in
 * `e2e/budget-overshoot.spec.ts`. That it is identified at all is DOM, so it is here.
 */
describe('every mark drawn over another is named in a legend', () => {
  const envelope = () => screen.getByRole('region', { name: /how much room is left/i });
  const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });

  /**
   * The Envelope's ring. Its `aria-label` already said "Currently at 32K context and 1 user", so a
   * screen-reader user was told what the ring was and a sighted reader was not — the one inversion
   * of the usual gap, and no less a gap for it.
   */
  it('keys the ring that marks your scenario on the feasibility grid', () => {
    render(<App />);

    const key = within(envelope()).getByText('You are here').closest('li');
    expect(key).not.toBeNull();
    /*
     * In the legend itself, beside the other keys, rather than as a caption of its own somewhere.
     * Anchored on the ramp's key rather than on a state's since #65: the field is coloured by a
     * magnitude now, so the ramp is what the neighbouring *keys* key, and "Comfortable" is a line of
     * prose in the same list rather than a swatch.
     *
     * On the ramp key's own clause rather than on either end label, because the ends are per-measure
     * ("less room", "slower", "quicker to start") and this test is about where a key sits, not about
     * which measure is in force.
     */
    expect(key?.parentElement).toBe(
      within(envelope())
        .getByText(/graded against the others on this grid/)
        .closest('ul')
    );
  });

  /**
   * The Matrix's selection ring, keyed only when the grid actually holds the marked cell — which is
   * the pairing that matters, since `isCurrent` is false for every cell on a linked rig. A key to a
   * mark that appears nowhere is the failure this file's neighbour comment names.
   */
  it('keys the marked cell in the comparison grid, and only while a cell is marked', () => {
    render(<App />);

    expect(matrix().querySelectorAll('[aria-current="true"]')).toHaveLength(1);
    expect(within(matrix()).getByText(/the cell the Bench above is set to/)).toBeInTheDocument();

    // And the sample is the mark (#130): the swatch wears the marked cell's own inset-frame
    // utilities, not the retired offset ring — one constant, read by both, so the legend cannot
    // drift from the grid again. The mark utilities are exactly those the cell adds when marked.
    const marked = matrix().querySelector('[aria-current="true"]')!;
    const swatch = within(matrix())
      .getByText(/the cell the Bench above is set to/)
      .querySelector('span[aria-hidden="true"]')!;
    const markUtilities = (marked.getAttribute('class') ?? '')
      .split(/\s+/)
      .filter((u) => !u.includes('focus') && /^(inset-ring|shadow)/.test(u));
    expect(markUtilities.length).toBeGreaterThanOrEqual(2);
    for (const utility of markUtilities) {
      expect(swatch.getAttribute('class')).toContain(utility);
    }
    expect(swatch.getAttribute('class')).not.toMatch(/ring-offset/);

    // Every cell here is scored at one device, so a two-card rig marks nothing.
    act(() => {
      useConfig.getState().set('deviceCount', 2);
    });
    expect(matrix().querySelectorAll('[aria-current="true"]')).toHaveLength(0);
    expect(
      within(matrix()).queryByText(/the cell the Bench above is set to/)
    ).not.toBeInTheDocument();
  });
});

/**
 * The masthead survives having no canvas.
 *
 * Its backdrop is painted on a 2D context, and jsdom has none — `getContext` returns null here, and
 * a real browser can refuse one under memory pressure. Everything the masthead actually *says* is
 * DOM, so the draw failing must cost the decoration and nothing else. That the backdrop paints at
 * all is e2e's question, in `e2e/canvases.spec.ts`; this is the other half, and the half a headless
 * environment can answer.
 */
describe('the masthead', () => {
  it('renders the wordmark and tagline with no 2D context available', () => {
    render(<App />);

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('headroom');
    expect(screen.getByText(/What runs on your hardware/i)).toBeInTheDocument();
  });

  /**
   * And that its <header> sits outside <main>, which is the whole reason it is a `banner` landmark.
   * Nesting it back inside is a one-line change that removes the role silently: the assertion above
   * would still pass, and a screen-reader user's route to the top of the page would be gone with
   * nothing to say so.
   *
   * Asserted positionally rather than with `getByRole('banner')`, because jsdom's role mapping
   * reports *every* <header> as a banner — including the four nested inside <section> panels, which
   * are `generic` in any browser that implements the scoping. The query finds five elements here
   * and proves nothing. `e2e/canvases.spec.ts` makes the role claim where it means something.
   */
  it('sits outside <main>, which is what makes it a banner landmark', () => {
    const { container } = render(<App />);

    const header = screen.getByRole('heading', { level: 1 }).closest('header');
    expect(header).not.toBeNull();
    expect(container.querySelector('main')).not.toBeNull();
    expect(container.querySelector('main')!.contains(header!)).toBe(false);

    // The share control belongs up here too — it describes the whole scenario, not any one panel.
    expect(within(header!).getByRole('button', { name: /copy link/i })).toBeInTheDocument();
  });
});

/**
 * The verdict strip is now memoised on the scenario, so what needs guarding is the failure a memo
 * introduces: grades that keep describing the configuration they were computed for. It had no
 * coverage at this level at all before — the arithmetic is pinned in the engine's suite, but
 * nothing checked that this surface re-renders it.
 */
describe('the workload strip keeps up with the scenario', () => {
  const strip = () => screen.getByRole('region', { name: /what you could do with it/i });
  const rows = () =>
    within(strip())
      .getAllByRole('listitem')
      .map((li) => li.textContent ?? '');

  it('re-grades when the hardware changes under it', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    const onCard = rows();

    // A 671B model on the same card: every archetype has to move.
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    expect(rows()).not.toEqual(onCard);
  });

  it('re-grades when only a usage slider moves', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    const atOneUser = rows();

    // Concurrency is not part of any archetype's prompt, but it is part of every placement.
    act(() => useConfig.getState().set('concurrency', 128));
    expect(rows()).not.toEqual(atOneUser);
  });

  /**
   * And the toggle it exists to make free changes nothing but the prose. Each row keeps its grade
   * and its reason, with the description prepended to the reason.
   */
  it('adds descriptions without disturbing a single grade', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Each row is [grade, label, reason] as three element children, so the parts can be compared
    // separately — the row's own textContent interleaves the icon and would hide a moved grade.
    const parts = () =>
      within(strip())
        .getAllByRole('listitem')
        .map((li) => [...li.children].map((child) => (child.textContent ?? '').trim()));

    const before = parts();
    await user.click(screen.getByRole('button', { name: /what each workload means/i }));
    const after = parts();

    expect(after).toHaveLength(before.length);
    for (const [i, [grade, label, reason]] of before.entries()) {
      expect(after[i][0]).toBe(grade);
      expect(after[i][1]).toBe(label);
      // Grown at the front by the description, unchanged at the end.
      expect(after[i][2].endsWith(reason)).toBe(true);
      expect(after[i][2].length).toBeGreaterThan(reason.length);
    }
  });

  /**
   * And that the memo actually memoises, which nothing else here can tell.
   *
   * The saving rests entirely on `config` being reference-stable between renders that do not change
   * the scenario. Narrowing the Bench's bare `useConfig()` to a selector returning a fresh object is
   * the ordinary way to trim a zustand subscription, and it would silently turn this `useMemo` into
   * a no-op with every assertion above still passing.
   */
  it('does not re-grade to render a description', async () => {
    const user = userEvent.setup();
    render(<App />);

    const graded = vi.mocked(judgeWorkloads);
    graded.mockClear();
    await user.click(screen.getByRole('button', { name: /what each workload means/i }));
    expect(graded).not.toHaveBeenCalled();

    // And the spy is wired to something that does fire, so the assertion above is not vacuous.
    act(() => useConfig.getState().set('concurrency', 4));
    expect(graded).toHaveBeenCalled();
  });

  /**
   * One set of column tracks for the whole list, which is the invariant behind #70.
   *
   * The measurement belongs to `e2e/workload-columns.spec.ts` and cannot be made here: jsdom has no
   * layout engine, so every one of those offsets reads back as 0 and an equality assertion over them
   * is a tautology. What jsdom *can* see is where the tracks are declared, and that is the thing a
   * later edit would undo — putting the three tracks back on the row makes each `<li>` its own grid
   * container, so the middle `auto` is sized from that row's own label and the reason column starts
   * at a different x on all seven rows.
   *
   * So this asserts the mechanism rather than its effect, deliberately, in the suite that runs in a
   * second. If the mechanism is ever changed on purpose — `display: contents` and `subgrid` are the
   * same idea in three forms — this assertion and that spec both want editing, and that is the point
   * of it failing.
   *
   * Three things, then, not one: that the list is a *grid* (tracks on a flex box are inert, and a
   * subgrid whose parent is not a grid computes as `none` — both leave the rows stacked and satisfy
   * a test that only looks at where the track string sits), that the tracks are on the list, and
   * that the row's own template stays scoped below `sm` so it cannot outlive the subgrid it backs.
   */
  it('declares its column tracks once, on the list rather than on each row', () => {
    render(<App />);

    const list = within(strip()).getAllByRole('listitem')[0].parentElement;
    expect(list).not.toBeNull();
    // A grid container first: `grid-template-columns` is inert on a flex box and `subgrid` on an item
    // whose parent is not a grid computes as `none`, so tracks on a non-grid list satisfy every
    // assertion below while every cell in every row stacks at x=0.
    expect(list!.className, 'the list is not a grid container, so its tracks are inert').toMatch(
      /(^|\s)grid(\s|$)/
    );
    expect(list!.className, 'the list does not own the three tracks').toMatch(
      /sm:grid-cols-\[9rem_auto_1fr\]/
    );

    for (const row of within(strip()).getAllByRole('listitem')) {
      // Anchored, so the `max-sm:` template below is not read as a row declaring its own tracks
      // past `sm` — and so a bare `sm:`-prefixed template still is.
      expect(row.className, 'a row declares column tracks of its own past sm').not.toMatch(
        /(^|\s)sm:grid-cols-\[/
      );
      // And takes the list's instead, spanning all three of them.
      expect(row.className).toMatch(/sm:grid-cols-subgrid/);
      expect(row.className).toMatch(/sm:col-span-3/);
      // Below `sm` the row keeps its own two-column grid, because the stacked layout is built from
      // `order` and a spanning third cell — both relationships among one row's own children. Scoped
      // to `max-sm:` rather than left bare, because a browser without subgrid drops the declaration
      // above and keeps whatever the row declares unconditionally: a bare template would put the
      // status word before the label there, which is neither layout this component supports.
      expect(row.className, 'the row template leaks past sm').toMatch(
        /max-sm:grid-cols-\[auto_1fr\]/
      );
    }
  });

  // The count's noun is its own text node, so it is what a text query can reach — the numerals sit
  // in the nested `whitespace-nowrap` span `PanelCount` wraps them in. "workloads" plural also
  // distinguishes it from the disclosure button's "what each workload means".
  const headline = () =>
    within(strip())
      .getByText(/workloads/)
      .textContent!.replace(/\s+/g, ' ')
      .trim();

  /**
   * The headline counts what was graded, on both sides of the fraction — and since #96 that is
   * every row.
   *
   * This is the number the panel is read by, and at the setting every visitor arrives on it was
   * wrong twice in a row. First it read "5 of 7 workloads" on a Spark that would serve several
   * users perfectly well, because `usable` subtracted an *ungraded* serving row exactly as it
   * subtracted a failing one (#75). Then, with the row out of both sides, it read "5 of 6" beside
   * seven visible rows — coverage claimed through the denominator instead of the numerator (#94).
   *
   * Grading serving at its own four users removed the state both fixes were working around, so the
   * denominator is the list again. Asserted here rather than in the engine suite because the count
   * is this component's arithmetic: `judgeWorkloads` returns seven verdicts either way.
   */
  it('counts every row, on both sides of the headline', () => {
    render(<App />);

    // The default scenario — gpt-oss-120b on a Spark at one user — grades all seven now. The row
    // that used to be ungraded is the assertion: "Not measured" was a status word this panel could
    // render, and there is no longer a state that produces it.
    expect(rows().filter((text) => text.includes('Not measured'))).toHaveLength(0);
    expect(headline()).toMatch(/of 7 workloads$/);
    // And the qualifier goes with the shortfall: once the denominator is the whole list there is
    // nothing to disclose, and "of 7 measured workloads" would imply some other total exists.
    expect(headline()).not.toMatch(/measured/);
  });

  /**
   * The row that used to carry the ungraded state, checked for what it says now.
   *
   * `--color-critical` is this panel's "No" and Telemetry's "Will not run", and #75 was that word
   * appearing for "you have not configured this yet". The fix then was a fourth, recessive state;
   * the fix now is that the question is always answered, so the row wears a real grade at the
   * setting every visitor arrives on — and the neighbouring row it was confused with, RAG at 31s to
   * read a 32K document, still means the thing `fail` means.
   */
  it('gives multi-user serving a real grade at the setting readers arrive on', () => {
    render(<App />);

    const serving = within(strip())
      .getAllByRole('listitem')
      .find((li) => li.textContent?.includes('Multi-user serving'))!;
    const status = serving.children[0] as HTMLElement;

    expect(status.textContent).toMatch(/Yes|Tight|No/);
    expect(status.textContent).not.toContain('Not measured');
    // A status hue, because the row is on the scale — the recessive ink was for a row that was not.
    expect(status.style.color).not.toBe('var(--color-text-faint)');
    // And the sentence names the archetype's own users rather than the slider's one.
    expect(serving.textContent).toMatch(/4 users at /);
    expect(serving.textContent).not.toMatch(/set concurrency/i);
  });

  /**
   * The serving row does not move with the slider, on the rendered surface.
   *
   * The engine suite asserts the verdict; this asserts what a reader sees, because the defect was
   * always a rendered one — a row that changed from `○ No` to `● Yes` when nobody had touched the
   * hardware. Deliberately *not* the headline: six archetypes are still graded at the reader's
   * concurrency and legitimately move with it, batch most obviously, since its aggregate is summed
   * across workers. A test asserting the whole panel is slider-independent would be asserting
   * something false.
   */
  it('leaves the serving row alone when the reader moves the concurrency slider', () => {
    render(<App />);
    const servingRow = () =>
      within(strip())
        .getAllByRole('listitem')
        .find((li) => li.textContent?.includes('Multi-user serving'))!.textContent;

    const before = servingRow();
    expect(before).toMatch(/4 users at /);

    for (const concurrency of [2, 4, 8]) {
      act(() => useConfig.getState().set('concurrency', concurrency));
      expect(servingRow(), `the serving row changed at ${concurrency} users`).toBe(before);
    }
  });

  /**
   * And the collapse for a configuration that cannot run has to keep working.
   *
   * The strip says one shared reason above the list and blanks the rows' own, which is right only
   * when all seven genuinely say the same thing. That is the refusal path, which grades all seven
   * `fail`; nothing else in the panel may reach it.
   */
  it('still collapses seven identical reasons into one when nothing runs', async () => {
    const user = userEvent.setup();
    render(<App />);

    // MLX is Apple-only; on an NVIDIA card nothing loads at all.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    expect(rows().filter((text) => text.includes('Not measured'))).toHaveLength(0);
    expect(within(strip()).getByText(/does not run/i)).toBeInTheDocument();
    // Every row keeps its status and gives up its reason to the sentence above the list.
    for (const row of within(strip()).getAllByRole('listitem')) {
      expect(row.children[2].textContent).toBe('');
    }
  });
});

describe('the Bench keeps the controls and the engine in step', () => {
  /**
   * The slider must offer the values the engine will actually be given. `coerce` clamps context
   * to the model's maximum, so a fixed stop list showed 32K while the store held 40,960 — with
   * the budget bar and throughput computed for neither.
   */
  it('caps the context slider at the model, not at a fixed list', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-32B');
    const max = getModel('Qwen/Qwen3-32B').maxContext;

    const slider = screen.getByLabelText('Context per sequence');
    await user.click(slider);
    fireEvent.change(slider, { target: { value: '99' } }); // Past the end; clamps to the last stop.

    expect(useConfig.getState().contextTokens).toBe(max);
    // The displayed value and the stored one must agree. Scoped to the control's own output:
    // the Envelope's axis legitimately prints the same figure.
    expect(screen.getByLabelText('Context per sequence')).toHaveAttribute(
      'aria-valuetext',
      tokens(max)
    );
  });

  it('does not call a resident but slow configuration fast', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Fits an EPYC host with nothing offloaded, and decodes at ~10 tok/s.
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'epyc-9654');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    // Resident — nothing spilled — and still too slow to claim speed for.
    expect(useConfig.getState().deviceId).toBe('epyc-9654');
    expect(screen.queryByText(/runs fast/i)).not.toBeInTheDocument();
    expect(screen.getByText(/How DeepSeek V3 is put together/i)).toBeInTheDocument();
  });

  it('says a raiseable ceiling is raiseable instead of just refusing', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-512');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q5_k_m');

    expect(screen.getByText(/raise it/i)).toBeInTheDocument();
  });
});

describe('the Bench keeps its claims consistent with its own numbers', () => {
  /**
   * Two places make speed claims and they must not drift. gpt-oss-20b BF16 on a Spark lands in
   * the 15-30 band, where the tile says "Usable" — so the aside must not say "runs fast".
   */
  it('reserves the fast claim for the verdict that says Fast', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'openai/gpt-oss-20b');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'bf16');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    const saysFast = within(verdicts).queryByText('Fast') !== null;
    const claimsFast = screen.queryByText(/runs fast/i) !== null;
    expect(claimsFast).toBe(saysFast);
  });

  it('shows no memory budget for a runtime that cannot drive the hardware', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'epyc-9654');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    // The ceiling and overhead are vLLM's own numbers; drawing them here would be an assumption
    // about software that never loads.
    expect(screen.queryByRole('img', { name: /allocatable used/i })).not.toBeInTheDocument();
    expect(screen.getByText(/No budget to show/i)).toBeInTheDocument();
  });

  it('offers multi-device on a Spark, which has a real link between units', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'dgx-spark');
    expect(screen.getByLabelText('Device count')).toBeInTheDocument();

    // A Mac has no transport between chassis, so it stays single.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    expect(screen.queryByLabelText('Device count')).not.toBeInTheDocument();
  });

  it('keeps a curated note alongside the tunable-ceiling warning', async () => {
    const user = userEvent.setup();
    render(<App />);

    // A Mac's 75% is a default and `iogpu.wired_limit_mb` goes as far as memory allows, so this
    // one really is raiseable — and the curated note still has to survive beside the warning.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    expect(screen.getByText(/raiseable to/i)).toBeInTheDocument();
    // "Beside" is now "behind a disclosure under": the derivation is 96 words and was the control's
    // accessible description (#68). Still on the page, still one interaction away, no longer read
    // out before the reader can choose.
    await user.click(screen.getByRole('button', { name: /show the full hardware note/i }));
    expect(screen.getByText(/what the sysctl parses/i)).toBeInTheDocument();
  });

  it('does not promise a ceiling the platform will not raise', async () => {
    const user = userEvent.setup();
    render(<App />);

    // The Ryzen's 96 GiB is Variable Graphics Memory's *maximum*, not a default — it is already
    // at its ceiling, so telling the user to raise it is advice they cannot take.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'ryzen-ai-max-395');
    expect(screen.queryByText(/raiseable/i)).not.toBeInTheDocument();
    // The curated bandwidth note is a separate claim and must still be there — in the disclosure,
    // which is where the provenance for a row a reader has already picked now lives.
    await user.click(screen.getByRole('button', { name: /show the full hardware note/i }));
    expect(screen.getByText(/213/)).toBeInTheDocument();
  });
});

describe('the Bench refuses impossible combinations', () => {
  it('does not offer NVFP4 on hardware that cannot run it', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    // NVFP4 is a safetensors format; llama.cpp reads GGUF and cannot open it at all, so the
    // vendor rule under test only becomes visible under a runtime that could load it.
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    expect(screen.getByRole('option', { name: /NVFP4/i })).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mi355x');
    expect(screen.queryByRole('option', { name: /NVFP4/i })).not.toBeInTheDocument();
  });

  /**
   * `rate()` rounds, so classifying the raw estimate could print "15 tok/s · Slow" against a
   * threshold of 15. The verdict is read off the displayed number instead.
   */
  it('never labels a displayed rate against a threshold it has already crossed', () => {
    render(<App />);
    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    const shown = Number(
      within(verdicts).getByText(/tok\/s per user/).previousSibling?.textContent
    );

    if (!Number.isFinite(shown)) return;
    const word = ['Fast', 'Usable', 'Slow'].find((w) => within(verdicts).queryByText(w) !== null);
    if (shown >= 30) expect(word).toBe('Fast');
    else if (shown >= 15) expect(word).toBe('Usable');
    else expect(word).toBe('Slow');
  });
});

describe('the slider never displays a value the engine is not using', () => {
  it('keeps the stored context selectable after switching to a larger model', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Qwen caps at 40,960 — not one of the fixed stops.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-32B');
    const slider = screen.getByLabelText('Context per sequence');
    fireEvent.change(slider, { target: { value: '99' } });

    const capped = useConfig.getState().contextTokens;
    expect(capped).toBe(getModel('Qwen/Qwen3-32B').maxContext);

    // Switching to a roomier model preserves that value, so it must remain displayable.
    await user.selectOptions(screen.getByLabelText('Model'), 'openai/gpt-oss-120b');
    expect(useConfig.getState().contextTokens).toBe(capped);
    expect(screen.getByLabelText('Context per sequence')).toHaveAttribute(
      'aria-valuetext',
      tokens(capped)
    );
  });

  /**
   * The other direction of the same invariant (#134): the stop lists fold the stored value in so
   * an off-stop URL value displays truthfully, but the fold is keyed on the stored value — so
   * the injected stop vanished the moment a drag moved off it, and the input's `max` and
   * index-to-value mapping changed under the held pointer. Opened at `?u=3`, the first notch of
   * a drag snapped the thumb back and remapped the next pixel of movement against the new scale.
   */
  it('keeps one scale under the pointer while an off-stop value is dragged away from', () => {
    render(<App />);

    // A URL-borne off-stop value: 3 is not among the fixed concurrency stops.
    act(() => {
      useConfig.getState().set('concurrency', 3);
    });
    const slider = screen.getByLabelText('Concurrent users');
    // Nine stops while 3 is selected — the eight fixed ones plus the injection.
    expect(slider).toHaveAttribute('max', '8');
    expect(slider).toHaveValue('2');

    // A drag starts on the injected stop and moves one notch right.
    fireEvent.pointerDown(slider);
    fireEvent.change(slider, { target: { value: '3' } });
    expect(useConfig.getState().concurrency).toBe(4);

    // Mid-drag the scale must not collapse: same max, thumb exactly where the pointer put it.
    expect(slider).toHaveAttribute('max', '8');
    expect(slider).toHaveValue('3');

    // Released, the injection expires while nothing is being dragged against it.
    fireEvent.pointerUp(slider);
    expect(slider).toHaveAttribute('max', '7');
    expect(slider).toHaveValue('2');
    expect(useConfig.getState().concurrency).toBe(4);
  });

  it('does not offer NVFP4 on NVIDIA cards without FP4 tensor cores', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Blackwell has them — under a runtime that can load the format.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    expect(screen.getByRole('option', { name: /NVFP4/i })).toBeInTheDocument();

    // Ada and Hopper are NVIDIA and have none, so the vendor check alone was not enough.
    for (const id of ['rtx-4090', 'h100-sxm', 'rtx-3090']) {
      await user.selectOptions(screen.getByLabelText('Hardware'), id);
      expect(screen.queryByRole('option', { name: /NVFP4/i })).not.toBeInTheDocument();
    }
  });
});

describe('the Bench offers only what the runtime can do', () => {
  it('drops a 4-bit KV cache when the runtime has no such flag', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    // A radio now, not a toggle button: these are mutually exclusive alternatives.
    expect(screen.getByRole('radio', { name: 'Q4' })).toBeInTheDocument();

    // vLLM's --kv-cache-dtype has no 4-bit option; offering one charges 0.5 bytes per element
    // for something it cannot allocate, turning a long-context OOM into a reported fit.
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    expect(screen.queryByRole('radio', { name: 'Q4' })).not.toBeInTheDocument();
    expect(useConfig.getState().kvPrecision).not.toBe('q4');
  });

  it('runs vLLM on a Spark, which is a CUDA target', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'dgx-spark');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).queryByText('Unsupported')).not.toBeInTheDocument();

    // Apple unified memory is still refused — the class alone was never the rule.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    expect(within(verdicts).getAllByText('Unsupported').length).toBeGreaterThan(0);
  });
});

/**
 * The Envelope sits on screen beside the verdict tiles, so the two must agree about the same
 * configuration. Before the axes included the selected values, the "you are here" ring snapped
 * to the nearest cell — putting a green marker under three "Will not run" tiles at 128 users.
 */
describe('the Envelope agrees with the verdicts beside it', () => {
  /** The table cell carrying the "you are here" marker, as text. */
  const currentCell = () => {
    const table = screen.getByRole('table', { name: /Feasibility by context/i });
    // The marker is its own span, so read the cell it sits in rather than the span itself.
    return within(table).getByText(/▸/).closest('td')?.textContent ?? '';
  };

  it('marks a cell that matches the tiles when the configuration will not run', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(
      screen.getByLabelText('Model'),
      'NousResearch/Meta-Llama-3.1-8B-Instruct'
    );
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    fireEvent.change(screen.getByLabelText('Concurrent users'), { target: { value: '99' } });

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    const willNotRun = within(verdicts).queryAllByText('Will not run').length > 0;

    await user.click(screen.getByRole('button', { name: /region as a table/i }));
    if (willNotRun) expect(currentCell()).toMatch(/Will not run/);
  });

  /**
   * The other half of the agreement, on the one machine class the test above cannot reach: the
   * rtx-5090 has no tunable ceiling, so its "Will not run" premise is never violated there. On a
   * Mac past the default allocation but inside the raiseable ceiling, the capacity tile used to
   * say "Will not run" over its own detail explaining a setting would fix it, while the Envelope
   * cell one panel down said "Past the default allocation" about the same placement (#121).
   */
  it('says a raiseable ceiling is a setting, in the same words as the table', async () => {
    const user = userEvent.setup();
    render(<App />);

    // DeepSeek-V3 at Q5_K_M needs ~446 GiB: past the 512 GB Mac Studio's 384 GiB default
    // allocation, inside the ceiling macOS lets the user raise.
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-512');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q5_k_m');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    // The capacity tile's word stops contradicting its own detail — and no tile in the strip
    // asserts a flat refusal for a placement one setting would admit.
    expect(within(verdicts).getByText('Past the default allocation')).toBeInTheDocument();
    expect(within(verdicts).queryByText('Will not run')).not.toBeInTheDocument();

    // And the Envelope's marked cell describes the same placement in the same words.
    await user.click(screen.getByRole('button', { name: /region as a table/i }));
    expect(currentCell()).toMatch(/Past the default allocation/);
  });

  it('locates the current scenario for a screen reader, not only as a ring', () => {
    render(<App />);
    const field = screen.getByRole('img', { name: /Currently at/i });
    expect(field).toHaveAccessibleName(/Currently at .* context and 1 user/i);
  });

  it('closes the whole region and blames the runtime, not the memory', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    // MLX cannot drive an NVIDIA card at any size, so telling the user their hardware is too
    // small is both wrong and unactionable — no amount of VRAM fixes it.
    expect(
      screen.getByRole('img', { name: /runtime cannot drive this hardware/i })
    ).toBeInTheDocument();
    expect(screen.queryByText(/Past what this hardware can hold/i)).not.toBeInTheDocument();
  });
});

describe('the Bench and its tiles cannot disagree', () => {
  it('makes the aside and the decode tile use the same classification', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Sweep a range of configurations; wherever the tile says "Fast", the aside must agree, and
    // wherever it does not, the aside must not claim speed. Sharing the thresholds was not
    // enough — the tile classified its rounded figure and the aside the raw one.
    for (const device of ['rtx-5090', 'rtx-5080', 'dgx-spark', 'mac-studio-m3-ultra-256']) {
      await user.selectOptions(screen.getByLabelText('Hardware'), device);

      const verdicts = screen.getByRole('region', { name: 'Verdicts' });
      const tileSaysFast = within(verdicts).queryByText('Fast') !== null;
      const asideClaimsFast = screen.queryByText(/runs fast/i) !== null;
      expect(asideClaimsFast).toBe(tileSaysFast);
    }
  });

  it('exposes mutually exclusive choices as radios, not independent toggles', () => {
    render(<App />);
    const group = screen.getByRole('group', { name: /KV precision/i });
    const radios = within(group).getAllByRole('radio');

    expect(radios.length).toBeGreaterThan(1);
    expect(radios.filter((r) => (r as HTMLInputElement).checked)).toHaveLength(1);
  });
});

/**
 * The share button is the distribution mechanism, so its failure modes matter more than most.
 * Both of these were silent: no clipboard meant the button did nothing while looking like it
 * had worked, and an unthrottled history write can throw on a dragged slider.
 */
describe('sharing a scenario degrades honestly', () => {
  const clipboard = navigator.clipboard;

  afterEach(() => {
    Object.defineProperty(navigator, 'clipboard', { value: clipboard, configurable: true });
  });

  it('offers the link for manual copying when there is no clipboard API', async () => {
    const user = userEvent.setup();
    // After `setup`, which installs its own clipboard stub. Undefined is what a non-secure
    // origin or an embedded browser actually gives you.
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    render(<App />);
    await user.click(screen.getByRole('button', { name: /Copy link to this scenario/i }));

    const field = screen.getByLabelText('Link to this scenario') as HTMLInputElement;
    expect(field.value).toMatch(/\?m=/);
    expect(screen.queryByText('Link copied')).not.toBeInTheDocument();
  });

  it('keeps the fallback link current when the scenario changes underneath it', async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    render(<App />);
    await user.click(screen.getByRole('button', { name: /Copy link to this scenario/i }));

    const field = () => screen.getByLabelText('Link to this scenario') as HTMLInputElement;
    expect(field().value).not.toContain('rtx-5080');

    // The field stays on screen; the scenario moves. Holding the link in state left it offering
    // whatever was selected at the click, so a manual copy shared the wrong configuration.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5080');
    expect(field().value).toContain('rtx-5080');
  });

  it('does not steal focus back on every later change', async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    render(<App />);
    await user.click(screen.getByRole('button', { name: /Copy link to this scenario/i }));

    // The field is shown and selected. From here the user goes back to the controls — and a
    // callback ref recreated each render pulled focus straight back, so a keyboard user could
    // press an arrow key once and then lose the control they were operating.
    const users = screen.getByLabelText('Concurrent users');
    users.focus();
    fireEvent.change(users, { target: { value: '4' } });

    expect(document.activeElement).toBe(users);
  });

  it('says so rather than silently failing when the write is refused', async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: () => Promise.reject(new Error('denied')) },
      configurable: true,
    });
    render(<App />);
    await user.click(screen.getByRole('button', { name: /Copy link to this scenario/i }));

    expect(await screen.findByLabelText('Link to this scenario')).toBeInTheDocument();
  });

  it('survives a browser that refuses a history write, and stops retrying it', async () => {
    const replaceState = window.history.replaceState;
    let attempts = 0;
    window.history.replaceState = () => {
      attempts += 1;
      throw new DOMException('throttled', 'SecurityError');
    };
    try {
      const user = userEvent.setup();
      // Rendering alone writes the URL, and a throw there would take the app down.
      const view = render(<App />);
      await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
      expect(screen.getByRole('region', { name: 'Verdicts' })).toBeInTheDocument();

      // A catch that reschedules itself is a timer that never stops while the browser keeps
      // refusing — and the early-return path used to leave that chain running past unmount.
      view.unmount();
      const afterUnmount = attempts;
      // Long enough for several retry intervals. A timer that escapes cleanup shows up here as
      // a further attempt — and in CI showed up as `window is not defined` after teardown, from
      // a suite where every test passed.
      await new Promise((r) => setTimeout(r, 1500));
      expect(attempts).toBe(afterUnmount);
    } finally {
      window.history.replaceState = replaceState;
    }
  });
});

/**
 * A link that was sent is a claim; the address bar must not retract it. Opening a fully-encoded
 * link to the default scenario used to erase it on the first render, so the recipient's bookmark
 * of that address resolved against whatever defaults shipped later — the exact failure the full
 * encoding exists to prevent, reintroduced by the synchroniser.
 */
describe('an explicitly shared scenario survives being opened', () => {
  const original = window.location.search;

  afterEach(() => {
    window.history.replaceState(null, '', `${window.location.pathname}${original}`);
  });

  it('keeps the querystring when the page was opened with one', async () => {
    const shared = configToShareSearch(DEFAULT_CONFIG);
    window.history.replaceState(null, '', `${window.location.pathname}${shared}`);

    render(<App />);
    // The write is throttled, so wait for the address bar to settle rather than reading it now.
    await waitFor(() => {
      expect(window.location.search).not.toBe('');
    });
    expect(new URLSearchParams(window.location.search).get('m')).toBe(DEFAULT_CONFIG.modelId);
  });

  it('leaves a bare address bare, because it claimed nothing', async () => {
    window.history.replaceState(null, '', window.location.pathname);
    render(<App />);
    await waitFor(() => {
      expect(window.location.search).toBe('');
    });
  });

  /**
   * The synchroniser owns the query and was rebuilding the whole URL, so the first configuration
   * change dropped whatever fragment the page was opened with — and with it the anchor a bookmark
   * or a shared section link was pointing at. `DETAIL_ANCHOR_ID` makes that a real id on this page
   * rather than a hypothetical one.
   */
  it('keeps a fragment the page was opened with', async () => {
    window.history.replaceState(null, '', `${window.location.pathname}#${DETAIL_ANCHOR_ID}`);
    render(<App />);

    // Any configuration change triggers the rewrite that used to lose it.
    act(() => useConfig.getState().set('concurrency', 4));

    await waitFor(() => {
      expect(window.location.search).not.toBe('');
    });
    expect(window.location.hash).toBe(`#${DETAIL_ANCHOR_ID}`);
  });
});

/**
 * A grid can hold both kinds of closed cell at once, and the legend used to pick one explanation
 * from whether *any* cell was raiseable — telling the reader that cells past the machine itself
 * could be fixed with a setting.
 */
describe('the Envelope legend covers every reason its cells are closed', () => {
  it('names both causes when both are on screen', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-512');
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const region = screen.getByRole('region', { name: /how much room/i });
    const legend = within(region).queryByText(/past the ceiling it hands out by default/i);

    // Whenever the legend offers the raiseable explanation, it must not offer it alone if any
    // cell is genuinely past the hardware.
    if (legend) {
      const table = within(region).queryByText(/Some of these are past what this machine holds/i);
      const onlyRaiseable = within(region).queryByText(/^Within the memory this machine has/i);
      expect(table !== null || onlyRaiseable !== null).toBe(true);
    }
  });
});

/**
 * The canvas summary is the only form the picture takes for a screen reader, so any distinction
 * the legend draws and it does not is one that reader never receives.
 */
describe('the spoken summary says everything the legend says', () => {
  it('mentions the raiseable ceiling, not just "will not run"', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-512');
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const region = screen.getByRole('region', { name: /how much room/i });
    const plot = within(region).getByRole('img');
    const spoken = plot.getAttribute('aria-label') ?? '';

    // Whenever the visible legend offers the raiseable explanation, the spoken one must too.
    const legendSaysRaiseable =
      within(region).queryByText(/which you can raise/i) !== null ||
      within(region).queryByText(/past the ceiling it hands out by default/i) !== null;
    if (legendSaysRaiseable) {
      expect(spoken).toMatch(/allocation ceiling, which you can raise/i);
    }
  });
});

/**
 * The Matrix is the "what are my options" surface, so its job is to stay informative at every
 * configuration — and to keep the three measures independent, since their disagreement is the
 * argument the whole surface makes.
 */
/**
 * A spilled "runs" carries the qualifier `HOST_RAM_UNCHECKED` exists to enforce (#127).
 *
 * `planPlacement` sizes a spill with no host-RAM input at all, so any surface saying a spilled
 * configuration runs is promising something the engine never checked. The Envelope's legend and
 * Telemetry's tile carried the sentence; the Matrix — the surface read as a shortlist — said
 * "runs only by spilling 99% of its weights to host RAM" on every channel with no qualifier,
 * counted those cells in "N of M combinations run", and the budget bar's spill line was a third
 * unqualified copy. One constant, verbatim on each, since a near-copy per panel is the drift the
 * constant's own docblock records.
 */
describe('a spilled "runs" says what the engine did not check', () => {
  it('qualifies the count on both channels, and every spilled cell sentence', () => {
    render(<App />);

    // The bounded grid holds runnable spilled cells (DeepSeek-R1 on the 5090 among them), so
    // the keyed legend line and the caption clause are both live on the default page.
    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const copies = within(matrix).getAllByText(/Loads only if the host has RAM for the spilled/);
    // Legend line and sr-only caption — the visible and the spoken channel.
    expect(copies.length).toBeGreaterThanOrEqual(2);
    expect(copies.some((el) => el.closest('caption') !== null)).toBe(true);
    expect(copies.some((el) => el.closest('caption') === null)).toBe(true);

    const spilled = within(matrix)
      .getAllByRole('button')
      .filter((b) => /runs only by spilling/.test(b.getAttribute('aria-label') ?? ''));
    expect(spilled.length).toBeGreaterThan(0);
    for (const cell of spilled) {
      expect(cell.getAttribute('aria-label')).toContain(HOST_RAM_UNCHECKED);
    }
  });

  it('qualifies the budget bar’s spill line', async () => {
    const user = userEvent.setup();
    render(<App />);

    // 671B on a 32 GB card: runnable via offload, so the overshoot banner carries the spill
    // clause — and now the sentence saying what that clause does not promise.
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    const budget = screen.getByRole('region', { name: /memory budget/i });
    expect(
      within(budget).getByText(/Loads only if the host has RAM for the spilled/)
    ).toBeInTheDocument();
  });
});

/**
 * A control that names a flag the runtime does not accept is wrong even when the arithmetic
 * behind it is right. vLLM's one-byte cache is `fp8_e4m3`; there is no integer option at all.
 */
describe('the KV control names something the runtime accepts', () => {
  it('calls the one-byte cache FP8 under vLLM', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const group = screen.getByRole('group', { name: /KV precision/i });
    expect(within(group).getByText('FP8')).toBeInTheDocument();
    expect(within(group).queryByText('Q8')).not.toBeInTheDocument();
  });

  it('still calls it Q8 under llama.cpp, which really does mean q8_0', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');

    const group = screen.getByRole('group', { name: /KV precision/i });
    expect(within(group).getByText('Q8')).toBeInTheDocument();
  });
});

/**
 * `fast` was gated on `runnable` and the sentences underneath it were not, so an unsupported
 * configuration still blamed host-bus spill or pointed at a decode tile reading "Unsupported".
 */
describe('the teaching aside makes no speed claim about a configuration that cannot run', () => {
  it('says so plainly instead of explaining a number that means nothing', async () => {
    const user = userEvent.setup();
    render(<App />);

    // An MoE model, so the aside renders at all — then MLX on an NVIDIA card, which cannot run.
    await user.selectOptions(screen.getByLabelText('Model'), 'openai/gpt-oss-120b');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    expect(screen.getByText(/does not run as selected/i)).toBeInTheDocument();
    expect(screen.queryByText(/crossing the host bus every token/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Even resident it would be slow/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/the decode figure above measures/i)).not.toBeInTheDocument();
  });
});

/**
 * Headroom is only room to *grow* while there is somewhere to grow to. At a model's own ceiling
 * the spare memory is real and the invitation is not.
 */
describe('the capacity tile does not promise context a model cannot take', () => {
  it('says how far the model goes instead of offering more', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Qwen3 4B tops out at 40,960 and leaves a 5090 mostly empty.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-4B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    const slider = screen.getByLabelText('Context per sequence') as HTMLInputElement;
    fireEvent.change(slider, { target: { value: String(Number(slider.max)) } });

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).getByText(/as far as this model goes/i)).toBeInTheDocument();
    expect(within(verdicts).queryByText(/Room to grow/i)).not.toBeInTheDocument();
  });

  it('still offers the room when there is some', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-4B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    expect(within(verdicts).getByText(/Room to grow/i)).toBeInTheDocument();
  });
});

/**
 * The decode tile attributes the step to the term that costs the most time, not to whichever
 * term exists (#122). The KV axis has made this comparison since the 0.08%-offload finding; the
 * spill axis kept the existence test, so any configuration a hair past the ceiling was told the
 * bus "sets the pace" — on PCIe 4.0 that claim only becomes true past roughly a 4% spill, and
 * the band under it is exactly where a reader is deciding whether clearing the spill is worth it.
 */
describe('the decode tile blames the term that sets the pace', () => {
  it('does not blame the host bus for a spill the resident reads outweigh', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Qwen3-32B at Q4_K_M on a 4090 at 16K: spilled by 0.7%, so the bus is a sliver of the
    // step and VRAM bandwidth still sets the pace.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-32B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-4090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    act(() => {
      useConfig.getState().set('contextTokens', 16384);
    });

    expect(screen.queryByText(/host bus set the pace/i)).not.toBeInTheDocument();
    expect(
      screen.getByText(/resident reads still cost more per step than the 1% of weights/i)
    ).toBeInTheDocument();
  });

  it('does not blame the bus while the cache is the largest cost in the step', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Raised in review on #145: `kvBound` compares KV against the weight terms' *sum*, so at
    // ~7.8ms KV, ~4.6ms bus and ~3.3ms resident reads it is false — and a pairwise bus test
    // then named the bus while KV was the largest single term. The strict three-way max names
    // the cache.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-30B-A3B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-4090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    act(() => {
      useConfig.getState().set('contextTokens', 32768);
      useConfig.getState().set('concurrency', 2);
    });

    expect(screen.queryByText(/host bus set the pace/i)).not.toBeInTheDocument();
    expect(screen.getByText(/KV traffic is the largest cost in the step/i)).toBeInTheDocument();
  });

  it('still blames the bus once its time outweighs the resident reads', async () => {
    const user = userEvent.setup();
    render(<App />);

    // The same shape past the crossover: 4.4% spilled, and the bus term is the larger half.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-30B-A3B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q8_0');
    act(() => {
      useConfig.getState().set('contextTokens', 16384);
    });

    expect(screen.getByText(/host bus set the pace — 4% of them spill/i)).toBeInTheDocument();
  });
});

/**
 * "Comfortable" promises the answer starts promptly, and the tile beside it calls anything past
 * two seconds "Noticeable" in amber. A ten-second threshold here left the two disagreeing.
 */
describe('the region and the latency tile agree about promptness', () => {
  it('does not paint a cell green while the tile warns about its wait', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'epyc-9654');

    const verdicts = screen.getByRole('region', { name: 'Verdicts' });
    const warned =
      within(verdicts).queryByText('Noticeable') !== null ||
      within(verdicts).queryByText('Slow start') !== null;

    if (warned) {
      const region = screen.getByRole('region', { name: /how much room/i });
      await user.click(within(region).getByRole('button', { name: /region as a table/i }));
      const marked = within(region).getByText(/▸/).closest('td')?.textContent ?? '';
      expect(marked).not.toMatch(/^\s*▸?\s*Comfortable/);
    }
  });
});

/**
 * Copying a link makes a claim — "this is what I was looking at" — so a confirmation that belongs
 * to a superseded attempt is worse than no confirmation. Clearing the reset timer cancels the
 * previous attempt's timer and nothing else: `writeText` is not abortable, so an earlier promise
 * is still in flight and still holds its callbacks.
 */
describe('the share link never reports a result a later click has superseded', () => {
  // Restored for the same reason the block above restores it: this stub's promises are never
  // settled, so leaving it in place hangs any later test that clicks the button and clobbers the
  // one `userEvent.setup()` installs for `user.copy()`. Vitest isolates per file, so the blast
  // radius is this file — but "nothing runs after it today" is a property of the file's ordering,
  // not of the test.
  const clipboard = navigator.clipboard;

  afterEach(() => {
    Object.defineProperty(navigator, 'clipboard', { value: clipboard, configurable: true });
  });

  const stubClipboard = () => {
    const settlers: { resolve: () => void; reject: () => void }[] = [];
    const writeText = vi.fn(
      () =>
        new Promise<void>((resolve, reject) =>
          settlers.push({ resolve, reject: () => reject(new Error('denied')) })
        )
    );
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
      writable: true,
    });
    return settlers;
  };

  it('ignores a late success from an attempt the user has already replaced', async () => {
    const user = userEvent.setup();
    const settlers = stubClipboard();
    render(<App />);

    const button = screen.getByRole('button', { name: /copy link to this scenario/i });
    await user.click(button);
    await user.click(button);
    expect(settlers).toHaveLength(2);

    // The second attempt is refused, so the manual-copy field appears.
    await act(async () => settlers[1].reject());
    expect(screen.getByLabelText('Link to this scenario')).toBeInTheDocument();

    // The first now resolves, late. Before the attempt counter this hid the field and announced a
    // success for a link the user had already moved past.
    await act(async () => settlers[0].resolve());

    expect(screen.getByLabelText('Link to this scenario')).toBeInTheDocument();
    expect(screen.queryByText(/link copied/i)).not.toBeInTheDocument();
  });

  /**
   * The half of the race `clearTimeout` provably cannot reach, and the one with the worse symptom.
   *
   * When the *earlier* attempt succeeds, its reset timer is scheduled after the second click has
   * already cleared `resetTimer` — so there is nothing left to cancel it. Unfixed, a genuine
   * refusal shows the fallback field and then a stale timer silently erases it two seconds later,
   * leaving no trace that anything failed.
   */
  it('does not let a superseded success erase a real failure two seconds later', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const settlers = stubClipboard();
      render(<App />);

      const button = screen.getByRole('button', { name: /copy link to this scenario/i });
      await user.click(button);
      await user.click(button);

      // The superseded attempt succeeds first, then the live one is refused.
      await act(async () => settlers[0].resolve());
      await act(async () => settlers[1].reject());
      expect(screen.getByLabelText('Link to this scenario')).toBeInTheDocument();

      // Past the 2s confirmation window: the failure notice has to survive it.
      await act(async () => {
        vi.advanceTimersByTime(2500);
      });
      expect(screen.getByLabelText('Link to this scenario')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('still confirms the attempt that did win', async () => {
    const user = userEvent.setup();
    const settlers = stubClipboard();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /copy link to this scenario/i }));
    await act(async () => settlers[0].resolve());

    expect(screen.getByText(/link copied/i)).toBeInTheDocument();
  });

  /**
   * And clears itself when the scenario *doesn't* move, which is the case the derived comparison
   * cannot cover — nothing about `copiedHref === href` ever becomes false on its own. The timer was
   * reachable by no test at all: the only one that advanced the clock got there superseded, so the
   * success handler early-returned before scheduling anything. Deleting the line left the suite
   * green and "Link copied" on screen indefinitely.
   */
  it('returns to its resting label two seconds later', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const settlers = stubClipboard();
      render(<App />);

      await user.click(screen.getByRole('button', { name: /copy link to this scenario/i }));
      await act(async () => settlers[0].resolve());
      expect(screen.getByText(/link copied/i)).toBeInTheDocument();

      await act(async () => {
        vi.advanceTimersByTime(2500);
      });
      expect(screen.queryByText(/link copied/i)).not.toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /copy link to this scenario/i })
      ).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  /**
   * The same race a click causes, arrived at from the other direction. The counter only advanced on
   * a *click*, so a scenario change left the earlier write's callbacks live: the button beside the
   * new scenario announced a success for a link the clipboard no longer holds.
   */
  it('does not confirm a write the user has moved the scenario out from under', async () => {
    const user = userEvent.setup();
    const settlers = stubClipboard();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /copy link to this scenario/i }));
    expect(settlers).toHaveLength(1);

    // The scenario moves while the write is still in flight.
    act(() => useConfig.getState().set('concurrency', 4));

    await act(async () => settlers[0].resolve());
    expect(screen.queryByText(/link copied/i)).not.toBeInTheDocument();
  });

  /**
   * A confirmation already on screen is stale for the same reason: it describes what the clipboard
   * holds, and what the clipboard holds has stopped matching what is on screen.
   */
  it('withdraws a confirmation once the scenario it described has changed', async () => {
    const user = userEvent.setup();
    const settlers = stubClipboard();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /copy link to this scenario/i }));
    await act(async () => settlers[0].resolve());
    expect(screen.getByText(/link copied/i)).toBeInTheDocument();

    act(() => useConfig.getState().set('concurrency', 4));
    expect(screen.queryByText(/link copied/i)).not.toBeInTheDocument();
  });

  /**
   * But the manual-copy fallback is not a stale claim — the clipboard is still unavailable, and the
   * field renders `href`, so it is already offering the new link. Clearing it on every slider frame
   * would snatch the fallback away mid-copy, which is a worse failure than the one above.
   */
  it('keeps the manual-copy field through a scenario change, and updates it', async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    render(<App />);

    await user.click(screen.getByRole('button', { name: /copy link to this scenario/i }));
    const before = (screen.getByLabelText('Link to this scenario') as HTMLInputElement).value;

    act(() => useConfig.getState().set('concurrency', 4));

    const after = screen.getByLabelText('Link to this scenario') as HTMLInputElement;
    expect(after).toBeInTheDocument();
    expect(after.value).not.toBe(before);
  });
});

/**
 * Neither Envelope axis said what it measured (#81).
 *
 * The left gutter ran 1…128 and the strip under the plot ran 2K…128K — powers of two in
 * overlapping ranges, with a single `K` at the smallest and faintest type on the page carrying the
 * entire distinction between "128 users" and "128K tokens". Every *other* representation of the
 * same grid named both quantities: the hidden table has a caption and a column header, and the
 * canvas's `aria-label` opens by naming both. The picture — the default representation — was the
 * one that did not say, and its y axis runs bottom-up, which was stated only in a source comment.
 *
 * Every assertion here reads `SETTING_LABELS` rather than a string literal, and that is the point of
 * the test rather than a stylistic choice: what is being guarded is that an axis title and the
 * control that drives it cannot come apart. A test holding its own copy of the wording keeps
 * passing while the two surfaces drift, which is exactly the failure `kvLabel` was written for.
 */
describe('the Envelope names both of its axes', () => {
  const region = () => screen.getByRole('region', { name: /how much room is left/i });

  it('titles each axis with the words its own control uses', () => {
    render(<App />);

    // The controls first, so the constant is anchored to something a user can actually operate
    // rather than asserted against itself.
    expect(screen.getByLabelText(SETTING_LABELS.contextTokens)).toHaveAttribute('type', 'range');
    expect(screen.getByLabelText(SETTING_LABELS.concurrency)).toHaveAttribute('type', 'range');

    // Matched as the whole text of an element, so the title is the element that says exactly this
    // and nothing else. The subhead a few pixels above it names the pair as prose, deliberately —
    // it is a sentence, so it keeps its own English rather than reading the constant.
    expect(within(region()).getByText(SETTING_LABELS.contextTokens)).toBeInTheDocument();

    // The y title carries the direction as well as the name, so it is found by prefix and the cue
    // asserted separately. Rows are drawn bottom-up, and a reader who assumes top-to-bottom reads
    // the default field as "128 users at 2K is the comfortable one" when it is 1 user at 2K.
    const upward = within(region()).getByText(new RegExp(`^${SETTING_LABELS.concurrency}\\b`));
    expect(upward).toHaveTextContent('↑');
  });

  it('keeps the titles out of the accessible tree, which names the axes already', () => {
    render(<App />);

    /*
     * The canvas `aria-label` is this picture's only textual equivalent and it already names both
     * quantities, which is why both tick strips are `aria-hidden`. Visible titles that joined the
     * accessible tree would have a screen reader hear the axes named twice — so they are hidden
     * the same way, and this is the assertion that says so.
     */
    const titles = [
      within(region()).getByText(SETTING_LABELS.contextTokens),
      within(region()).getByText(new RegExp(`^${SETTING_LABELS.concurrency}\\b`)),
    ];
    for (const title of titles) {
      expect(title.closest('[aria-hidden="true"]')).not.toBeNull();
    }
  });

  /**
   * The same two settings, named the same way on the surface that is the picture's equivalent.
   *
   * This is the part the issue did not name and the grep found: the caption said "context length"
   * and the row-header column said "Users" while the sliders said "Context per sequence" and
   * "Concurrent users" — two settings under four spellings inside one panel, which is how a reader
   * comparing the table against the field has to work out that they are the same axis.
   */
  it('names them the same way in the table, not in two more spellings', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(within(region()).getByRole('button', { name: /region as a table/i }));

    const table = within(region()).getByRole('table', {
      name: `Feasibility by ${SETTING_LABELS.contextTokens} and ${SETTING_LABELS.concurrency}`,
    });
    expect(
      within(table).getByRole('columnheader', { name: SETTING_LABELS.concurrency })
    ).toBeInTheDocument();
  });
});

/**
 * The Envelope's canvas has exactly one textual equivalent and its table is hidden by default, so
 * whatever that sentence omits is simply not available to a screen-reader user. Both branches
 * omitted something, in opposite directions.
 */
describe('the Envelope says what a region does, not only what it fails', () => {
  const altText = () => {
    const region = screen.getByRole('region', { name: /how much room/i });
    return within(region).getByRole('img').getAttribute('aria-label') ?? '';
  };

  it('names the runnable states when nothing in range is closed', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Qwen3 4B on an EPYC 9755: every cell runs, none comfortably, and none closed. The fixture
    // matters — DeepSeek on the 9654 leaves 20 of 64 cells over the ceiling, so `whyClosed` fires
    // there and the guard under test is never reached.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-4B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'epyc-9755');

    const alt = altText();
    expect(alt).toMatch(/No comfortable configuration/i);
    expect(alt).toMatch(/run but sit near a limit/i);
    // The finding itself: "0 of N combinations will not run at all" for a region where all of
    // them do.
    expect(alt).not.toMatch(/will not run at all/i);
  });

  it('says how many cells are spilling, not only how many are comfortable', async () => {
    const user = userEvent.setup();
    render(<App />);

    // A grid with both comfortable and offloaded cells — the branch that mentioned neither the
    // spill nor the closed count, one over from the one the review named.
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-4B');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5080');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'bf16');

    const alt = altText();
    expect(alt).toMatch(/are comfortable/i);
    expect(alt).toMatch(/spilling weights to host RAM/i);
  });

  it('does not promise an offloaded cell loads, when host RAM is never checked', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    const region = screen.getByRole('region', { name: /how much room/i });
    // The legend's own words. `planPlacement` sizes the spill and has no host-RAM input at all,
    // so the caveat Telemetry already carries has to be here too.
    expect(within(region).getByText(/not checked here/i)).toBeInTheDocument();
  });
});

/**
 * MLX quantizes with its own affine scheme, the catalog has no measured entry for it, and other
 * catalogued formats stand in *by width*. The engine cannot tell the difference — a roofline consumes bits
 * per weight, and a stand-in of the right width produces plausible arithmetic — so every figure for
 * an Apple-silicon configuration derived from a format MLX does not read, with nothing on screen
 * saying which figures those were. The same rule `devices.json` already follows for pre-release
 * specs: an approximation that is documented is a modelling choice; one that is invisible is
 * invented data.
 */
describe('a figure derived from a stand-in format says so', () => {
  const marker = () => screen.queryByText(/derived from a format .* cannot load/i);

  /**
   * The width named has to be the width the figures beside it were computed at.
   *
   * MLX substitutes seven formats — six GGUF plus INT8 — from Q3_K_M's 3.91 bpw to Q8_0's 8.5, and
   * the note on the runtime is one static string. So a sentence naming a particular quant was true
   * of exactly one of them and off by up to a factor of two on the rest, while claiming "the
   * arithmetic is sound for that width". Both cases are asserted because the Q4_K_M one passes
   * either way; only Q8_0 distinguishes a composed width from a hardcoded one.
   */
  it.each([
    ['q4_k_m', /4\.85 bpw/],
    ['q8_0', /8\.5 bpw/],
  ])('names the width the figures were actually computed at, for %s', async (quantId, width) => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), quantId);

    expect(marker()).toBeInTheDocument();
    // And says what the substitution actually is, rather than only that there is one. Both halves:
    // the runtime's own note, and the width composed from the selected quant. Without the first,
    // deleting `{substitution}` from the banner leaves every other assertion here passing.
    expect(marker()).toHaveTextContent(/affine scheme/i);
    expect(marker()).toHaveTextContent(width);
  });

  it('stays silent on the formats MLX genuinely loads', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    // BF16 is a real MLX format — no groups, no scales, no biases, so 16 bpw is exact — and a
    // marker here would be crying wolf on the majority case and train people to ignore it where it
    // matters. It is also the landing state: switching the runtime to MLX coerces to BF16.
    await user.selectOptions(screen.getByLabelText('Quantization'), 'bf16');
    expect(marker()).not.toBeInTheDocument();
  });

  /**
   * INT8 is a stand-in under MLX, and the catalog said otherwise until PR #32.
   *
   * MLX's 8-bit is affine at 8 bits just as its 4-bit is, while the catalogued `int8` row is
   * LLM.int8() — per-channel, no group metadata, 8.0 bpw exactly, cited to arXiv 2208.07339 and
   * offered to vLLM. Listing it as native inverted the two 8-bit stand-ins against each other: on
   * a 235B, the *marked* Q8_0 at 8.5 bpw reported 13.7 GiB heavier than the unmarked INT8, so the
   * lighter and more optimistic of the two was the one carrying no provenance at all.
   *
   * Pinned because nothing asserted MLX + INT8 in either direction, which is how a modelling call
   * gets reversed by a one-word catalog edit and nobody notices.
   */
  it('marks INT8 under MLX, which quantizes 8-bit its own way too', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'int8');

    expect(marker()).toBeInTheDocument();
    // At the row's own width, not Q4_K_M's — the composed clause has to follow the selection here
    // as it does for every other stand-in.
    expect(marker()).toHaveTextContent(/8 bpw/);
  });

  /**
   * The banner promises "the memory and speed figures below", so it has to go quiet when there are
   * none. Reachable because the runtime picker deliberately permits a pairing it cannot drive and
   * `coerce` never reconciles the device against the runtime: on an RTX under MLX, BudgetBar,
   * Telemetry, Workloads and the Envelope all render a refusal — while this asserted their
   * arithmetic was sound for a width nothing used.
   */
  it('stays silent when the runtime cannot drive the device at all', async () => {
    const user = userEvent.setup();
    render(<App />);

    // Same runtime and same format throughout — only the device moves, so the gate is the one
    // thing that can account for the marker going away.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    expect(marker()).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    // Asserted, so this cannot go vacuous if the pairing ever stops being reachable — the point is
    // that there are no figures, not merely that the marker is gone.
    expect(screen.getByText(/no budget to show/i)).toBeInTheDocument();
    expect(marker()).not.toBeInTheDocument();
  });

  /**
   * The other side of that gate, and the one that makes it `wasEvaluated` rather than "does it
   * run". A configuration measured and found far over still took every figure on screen from the
   * stand-in's width, so it stays marked — dropping it here is the polarity error the Matrix
   * legend had, one surface over.
   */
  it('keeps marking a configuration that was measured and did not fit', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    // Over the machine, but MLX does drive a Mac — so the bytes were counted, at Q4_K_M's width.
    expect(screen.getByText(/over$/i)).toBeInTheDocument();
    expect(marker()).toBeInTheDocument();
  });

  it('stays silent on runtimes that load what they are given', async () => {
    const user = userEvent.setup();
    render(<App />);

    // llama.cpp reads GGUF natively — the same Q4_K_M, no substitution.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    expect(marker()).not.toBeInTheDocument();
  });

  it('tags the format picker that caused it, without repeating the whole derivation', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    // The control's own description says it is a stand-in; the panel says what the stand-in is.
    // Printing the same forty words in both taught people to skip both.
    //
    // The sentinel has to be a phrase that survives in the runtime's note, or this stops being an
    // assertion. It was `4.5 bpw`, which was the note's distinctive tail until the note stopped
    // naming a width — leaving a test that could not fail, guarding the thing it was written for.
    const picker = screen.getByLabelText('Quantization');
    expect(picker).toHaveAccessibleDescription(/stand-in for a format/i);
    expect(picker).not.toHaveAccessibleDescription(/affine scheme/i);
  });

  it('marks the Matrix when any row on it was scored at a stand-in', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);

    const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });
    const legend = () => within(matrix()).queryByText(/stand-in format .* cannot load/i);
    // Reachable today only because the *selection* is a stand-in — the `SUBSTITUTE_QUANT_IDS`
    // fallback cannot land on one with this catalog. The per-cell scan is defence for that route,
    // not something this can drive.

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    expect(legend()).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');
    expect(legend()).toBeInTheDocument();
  });

  /**
   * The all-blocked grid — the state that gating the legend on `runs` hid it in — is pinned in
   * `src/components/Matrix.test.tsx` rather than here.
   *
   * It was an App-level test driving the controls to the longest context and the most users, where
   * every Apple cell under MLX failed placement. Its precondition was asserted rather than assumed,
   * with a comment saying that a catalog change leaving one cell running would make it vacuous, and
   * #77 is that change: `unsloth/gemma-3-4b-it` keeps a 1024-token window on 29 of its 34 layers, so
   * it fits 128 users at 131K on the 512 GiB Mac Studio with room to spare. One running cell is
   * enough for a `runs`-gated legend to render too, and no setting blocks it — context, concurrency
   * and KV precision are already at their heaviest stops.
   *
   * So the scenario moved to where it can be *built*: one 671B row, mocked in, and no dependence on
   * what the catalog happens to contain. The test above still covers the app-level wiring of the
   * same marker.
   */
});

/**
 * The cache axis — #33, and then #38, which changed the answer.
 *
 * `kvElementBytes` falls back to a precision's nominal figure when a runtime declares no
 * `kvBytesPerElement`, and MLX's 8-bit cache used to have none: it was charged one byte per
 * element on no authority, and marked on screen because of it.
 *
 * That width is now derived from `mlx-lm`'s own source — `QuantizedKVCache(group_size=64, bits=8)`
 * with an fp16 scale *and* bias per group, so 8.5 bits — which means **the marker correctly no
 * longer fires anywhere in the shipped catalog.** These tests pin both halves: that it is silent
 * where a width is established, and that it still renders where one is not.
 */
describe('the cache-width marker', () => {
  const cacheMarker = () => screen.queryByText(/cache is charged .* at its nominal width/i);
  const weightMarker = () => screen.queryByText(/derived from a format .* cannot load/i);

  const mlxAt = async (user: ReturnType<typeof userEvent.setup>, quantId: string) => {
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), quantId);
  };

  /**
   * The configuration this whole marker was built for, now correctly unmarked.
   *
   * Native BF16 weights with an 8-bit cache was the case that carried no warning at all before
   * #33 and a warning after it. Both are now wrong answers: the width is established, so there is
   * nothing to caveat, and a marker here would be warning about a figure nobody is guessing at.
   */
  it('is silent on MLX at 8-bit, whose width is derived rather than assumed', async () => {
    const user = userEvent.setup();
    render(<App />);

    await mlxAt(user, 'bf16');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    expect(cacheMarker()).not.toBeInTheDocument();
    expect(weightMarker()).not.toBeInTheDocument();
  });

  /**
   * And the *weight* marker is untouched by that, which is the point of the two being separate
   * values. MLX still has no catalogued native quantization (#18), so a stand-in format is still
   * a stand-in — only the cache stopped being one.
   */
  it('leaves the weight marker alone, which is still a real substitution', async () => {
    const user = userEvent.setup();
    render(<App />);

    await mlxAt(user, 'q4_k_m');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    expect(weightMarker()).toBeInTheDocument();
    expect(cacheMarker()).not.toBeInTheDocument();
  });

  /**
   * The surfaces still render a cache substitution when there is one to render.
   *
   * Forced, because no shipped precision can reach this state and an unreachable branch is one
   * nobody notices breaking. This is the case the mechanism exists for — a precision added later
   * with no established width — and it has to keep working across the two surfaces that show it.
   */
  describe('when a precision has no established width', () => {
    beforeEach(() => {
      vi.mocked(kvSubstitutionFor).mockReturnValue(
        'A stand-in width, forced by the test so the surfaces can be checked.'
      );
    });

    afterEach(() => {
      vi.mocked(kvSubstitutionFor).mockReset();
    });

    it('says so on the Bench, beside the figures it applies to', async () => {
      const user = userEvent.setup();
      render(<App />);

      await mlxAt(user, 'bf16');
      act(() => useConfig.getState().set('kvPrecision', 'q8'));

      expect(cacheMarker()).toBeInTheDocument();
      expect(cacheMarker()).toHaveTextContent(/forced by the test/i);
      // The weight axis stays independent even here: BF16 is native, so only one marker shows.
      expect(weightMarker()).not.toBeInTheDocument();
    });

    /**
     * And both at once, which neither single-marker test covers.
     *
     * The panel holds two paragraphs and either may appear without the other, so "each fires
     * alone" does not establish that both fire together — a conditional rendering one *or* the
     * other would satisfy every other test here. Worth pinning because the defect that started
     * this was the mirror image: one axis marked, the other silent, on a page where both applied.
     */
    it('shows both markers when the weights are standing in too', async () => {
      const user = userEvent.setup();
      render(<App />);

      await mlxAt(user, 'q4_k_m');
      act(() => useConfig.getState().set('kvPrecision', 'q8'));

      expect(weightMarker()).toBeInTheDocument();
      expect(cacheMarker()).toBeInTheDocument();
    });

    it('says so on the Matrix, for every scored cell', async () => {
      atFullGrid();
      const user = userEvent.setup();
      render(<App />);

      const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });
      await mlxAt(user, 'bf16');
      act(() => useConfig.getState().set('kvPrecision', 'q8'));

      const legend = within(matrix()).queryByText(/cache charged at .* nominal width/i);
      expect(legend).toBeInTheDocument();
      // Neither "some rows" nor "every cell": under MLX the grid still carries every shipping
      // device while only the Apple columns are scored at all.
      expect(legend).toHaveTextContent(/every scored cell/i);
      expect(legend).not.toHaveTextContent(/every cell\u2019s/i);
    });

    /**
     * And it goes quiet when there are no figures to caveat, exactly as the weight marker does.
     * The sentence promises something about the readouts below it, and on a runtime that cannot
     * drive the device there are none.
     */
    it('stays silent when the runtime cannot drive the device at all', async () => {
      const user = userEvent.setup();
      render(<App />);

      await mlxAt(user, 'bf16');
      act(() => useConfig.getState().set('kvPrecision', 'q8'));
      expect(cacheMarker()).toBeInTheDocument();

      await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
      expect(screen.getByText(/no budget to show/i)).toBeInTheDocument();
      expect(cacheMarker()).not.toBeInTheDocument();
    });
  });

  /**
   * The precondition behind the Matrix's quantifier, asserted so it cannot go vacuous: if MLX ever
   * drove every device in the catalog, "every scored cell" and "every cell" would be the same
   * claim and the assertion above would stop distinguishing them.
   */
  it('has unscored cells under MLX, which is why the qualifier is there', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);

    await mlxAt(user, 'bf16');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const cells = within(matrix).getAllByRole('button', { name: /:/ });
    const unscored = cells.filter((c) =>
      /does not (run|support)|cannot drive|no estimate/i.test(c.getAttribute('aria-label') ?? '')
    );

    expect(cells.length, 'the grid rendered nothing').toBeGreaterThan(0);
    expect(unscored.length, 'every cell was scored, so the qualifier is vacuous').toBeGreaterThan(
      0
    );
    expect(unscored.length, 'the filter matched every cell').toBeLessThan(cells.length);
  });
});

/**
 * The per-token figure in the Bench's aside, which claims to be what sets the speed (#77 review).
 *
 * Three quantities are in play and they differ by enough to matter on the models this catalog exists
 * for. `activeParams` is the *published* convention: `publishedActiveParams` returns `totalParams`
 * outright on a dense model and only on an MoE rebuilds an embedding-subtracted dense residual with
 * the routed share added back. It disagrees with the physical count wherever the two **exclude
 * different things**, which is three cases and not one: a non-language tower, an untied input
 * embedding on a dense row, and — the one two drafts of this missed — a **tied** input embedding on
 * an MoE, which the published figure subtracts unconditionally and `activeDenseParams` correctly
 * keeps, a tied table being the output projection. Command A+ is that case, and it is why its
 * published figure is 0.578B *low* where Mistral Small 4's is 7% high: the omitted 1.074B table
 * outweighs the included 0.495B tower. `activeDenseParams` is the always-active dense part and
 * excludes the routed experts. `effectiveActiveParams(model, 1)` is the physical count, and the one
 * this sentence has to print.
 *
 * `speed.ts` divides by neither, which is worth saying here because the aside sounds as though it
 * does: `estimateDecode` reads `activeWeightBytes`, which prices the dense and expert halves at
 * their own widths — about a factor of two on an expert-only scheme like MXFP4, where the dense
 * tensors stay BF16. `effectiveActiveParams` is the parameter count behind that byte figure.
 *
 * Both wrong answers shipped briefly during #77 and each was caught by review rather than by a test:
 * `activeParams` overstated Mistral Small 4 (6.524B against a 6.096B basis),
 * and the correction to `activeDenseParams` understated every MoE in the other direction, by a
 * ratio that spans the catalog rather than one factor (Kimi K2 at 10.6B where a token traverses
 * 31.75B, but 1.91x on GLM-4.7-Flash and 8.65x on Mixtral). So this pins the sentence to the
 * engine's own expression, and asserts the two near neighbours are *not* what it prints — a test that
 * only checked the value against `effectiveActiveParams` would have passed on a dense model either
 * way, since all three coincide there.
 */
describe('the aside prints the basis the speed is actually computed from', () => {
  /**
   * An MoE whose published and physical bases actually differ, selected on the gap itself.
   *
   * On an *untied text-only* MoE the two coincide exactly — gpt-oss-20b is 3.61B on both — so a test
   * written against one passes whichever the component prints and the overstatement half goes
   * uncovered. What opens the gap is either a non-language tower inside `activeParams` or a **tied**
   * embedding, which the MoE branch of `publishedActiveParams` subtracts and the physical basis
   * keeps.
   *
   * The rows that satisfy it today are the two multimodal MoEs, and this deliberately does not say
   * "find a multimodal MoE": a tied text-only one would discriminate just as well with no tower
   * involved, and a selector naming the *cause* would reject it. The gap is the rule; which rows have
   * it is this week's catalog.
   */
  const moe = MODELS.find(
    (m) => m.expertParams > 0 && Math.abs(effectiveActiveParams(m, 1) - m.activeParams) > 1e8
  )!;

  it('quotes the decode basis at one sequence, not the published or the dense figure', () => {
    expect(
      moe,
      'no MoE in the catalog whose published and physical bases differ, so this has no subject'
    ).toBeDefined();

    const basis = effectiveActiveParams(moe, 1);
    // The premise: on an MoE the three figures genuinely differ, or none of this discriminates.
    expect(basis).toBeGreaterThan(moe.activeDenseParams);
    expect(Math.abs(basis - moe.activeParams)).toBeGreaterThan(1e8);

    act(() => useConfig.getState().set('modelId', moe.id));
    render(<App />);

    const aside = screen.getByText(/routes each token through only/i).closest('p')!;
    expect(aside.textContent).toContain(params(basis));
    expect(aside.textContent, 'prints the dense part, dropping the routed experts').not.toContain(
      params(moe.activeDenseParams)
    );
    expect(aside.textContent, 'prints the published figure, not the physical one').not.toContain(
      params(moe.activeParams)
    );
  });
});
