import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG, estimateConfig } from '@/store/config';
import { configToShareSearch } from '@/store/url';
import { DEVICES, MODELS, comparisonGrid, getDevice, getModel } from '@/data/catalog';
import { params, tokens } from '@/lib/format';
import { DEVICE_CLASS_LABELS, SETTING_LABELS, SETTING_NOTES, deviceCountNote } from '@/lib/stops';
import { DETAIL_ANCHOR_ID } from '@/components/Matrix';
// The one component this file mounts on its own, and only to sweep a renderer over all 43 catalog
// rows — see "leaves none of the markup in any note the catalog carries".
import { Select } from '@/components/Controls';
import { judgeWorkloads } from '@/engine/verdict';
import { RUNTIMES, getRuntime, kvSubstitutionFor, runtimeDrives } from '@/data/runtimes';
import { effectiveActiveParams } from '@/engine/weights';
import { canShard, maxAllocatablePerDevice } from '@/engine/placement';
import { colors, magnitudeRamp, marks } from '@/design/tokens';

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

/**
 * And the same treatment for the Matrix's extent, which is what this file's wall clock is made of
 * ([#101](https://github.com/MrZoller/bench/issues/101)).
 *
 * The Matrix is models × devices and every test here renders the whole page, so every test pays for
 * the whole grid — 1,470 buttons, each with a full-sentence `aria-label` that `getByRole` computes
 * on every name-matched query. #78 and #77 grew both axes in one sweep and took this file from 42s
 * to about fourteen minutes on CI, for changes that touched no component; the per-test timeout was
 * raised twice on the way and must not be raised again.
 *
 * So the grid is bounded by default here and the real one is opted into. **The polarity is the whole
 * design.** Bounding it by default makes the wall clock a constant — {@link BOUNDED_CELLS} cells
 * however large the catalog grows, which is the property the issue asks for: a change that touches
 * no component cannot fail CI on grid size. Opting in makes every full-grid case a written decision
 * rather than an accident, which matters because the full grid is what caught #52's roving tab index
 * and #64's rotated header, and a shrunken grid would have passed both.
 *
 * The line is not "does this test mention the Matrix" but **"does this test assert something about
 * the grid"** — see {@link atFullGrid}. Everything else merely renders it.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

/** The real extent, kept so {@link atFullGrid} restores it rather than re-deriving it. */
const realComparisonGrid = (
  await vi.importActual<typeof import('@/data/catalog')>('@/data/catalog')
).comparisonGrid;

/**
 * The rows and columns the bounded grid renders, chosen so it is small and still not degenerate.
 *
 * Every one of these is here for a property some test in this file reads off the grid, and dropping
 * any of them makes an assertion vacuous rather than red:
 *
 * - `openai/gpt-oss-120b` on `dgx-spark` is {@link DEFAULT_CONFIG}, so the grid contains the marked
 *   cell — without it `isCurrent` is false everywhere and the selection ring, its legend key and
 *   `aria-current` all have nothing to test.
 * - Three device classes in file order, so the class bands are a run of three rather than one, and
 *   `separated` is non-empty.
 * - `epyc-9654` is a column vLLM cannot drive, so the struck-heading branch is reachable.
 * - `rtx-3060-12gb` at 12 GiB refuses most rows on counted bytes while `rtx-5090` holds them, so
 *   both the ran and the did-not-fit inks are painted.
 * - `Qwen/Qwen3-8B` is dense, so the MXFP4 default substitutes on it and the stand-in note fires.
 *
 * Ids rather than indices, and asserted to resolve below: a slice of the catalog is a fixture that
 * silently becomes a different fixture the next time the catalog moves.
 */
const BOUNDED_MODEL_IDS = ['openai/gpt-oss-120b', 'Qwen/Qwen3-8B', 'deepseek-ai/DeepSeek-R1'];
const BOUNDED_DEVICE_IDS = ['rtx-5090', 'rtx-3060-12gb', 'dgx-spark', 'epyc-9654'];
const BOUNDED_CELLS = BOUNDED_MODEL_IDS.length * BOUNDED_DEVICE_IDS.length;

function boundedGrid(): ReturnType<typeof realComparisonGrid> {
  const real = realComparisonGrid();
  return {
    // Filtered out of the real extent rather than looked up, so both lists keep the order the
    // component is entitled to assume — popularity for the rows, file order for the columns.
    models: real.models.filter((m) => BOUNDED_MODEL_IDS.includes(m.id)),
    devices: real.devices.filter((d) => BOUNDED_DEVICE_IDS.includes(d.id)),
  };
}

/**
 * Render this test's Matrix at the real catalog extent.
 *
 * Call it before `render`, in any test whose subject is the grid itself: its size, its column
 * headings, its bands, its tab sequence, or a sweep over the cells it paints. The rule of thumb is
 * whether the assertion would still mean something with twelve cells — if it would, leave it
 * bounded.
 */
function atFullGrid() {
  vi.mocked(comparisonGrid).mockImplementation(realComparisonGrid);
}

beforeEach(() => {
  vi.mocked(comparisonGrid).mockImplementation(boundedGrid);
});

afterEach(() => {
  cleanup();
  useConfig.setState(DEFAULT_CONFIG);
});

/**
 * The fixture's own preconditions, since a fixture that quietly stops matching the catalog is the
 * failure this file has already had twice in other forms — a sweep that filtered on a field the type
 * does not have, and an exemption list that matched nothing. Both reported compliance over zero
 * cases.
 */
describe('the bounded grid this file renders', () => {
  it('names rows and columns the catalog still has', () => {
    const { models, devices } = boundedGrid();
    expect(models.map((m) => m.id)).toEqual(expect.arrayContaining(BOUNDED_MODEL_IDS));
    expect(devices.map((d) => d.id)).toEqual(expect.arrayContaining(BOUNDED_DEVICE_IDS));
    expect(models).toHaveLength(BOUNDED_MODEL_IDS.length);
    expect(devices).toHaveLength(BOUNDED_DEVICE_IDS.length);
  });

  it('is a constant the catalog cannot grow', () => {
    // The property #101 asks for, stated as an assertion: this file's grid is 12 cells whatever the
    // catalog does next, so a change that touches no component cannot fail CI on grid size.
    expect(BOUNDED_CELLS).toBe(12);
    const real = realComparisonGrid();
    // And it really is a reduction — a fixture equal to the catalog would satisfy everything above
    // while restoring the whole cost.
    expect(real.models.length * real.devices.length).toBeGreaterThan(BOUNDED_CELLS * 10);
  });

  it('spans the three device classes, in the catalog’s own order', () => {
    const classes = boundedGrid().devices.map((d) => d.class);
    expect(classes).toEqual(['discrete-gpu', 'discrete-gpu', 'unified-soc', 'cpu-ram']);
  });

  it('contains the default scenario, so a cell is marked', () => {
    const { models, devices } = boundedGrid();
    expect(models.map((m) => m.id)).toContain(DEFAULT_CONFIG.modelId);
    expect(devices.map((d) => d.id)).toContain(DEFAULT_CONFIG.deviceId);
  });
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

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('bench');
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
describe('the Matrix stays informative', () => {
  const matrix = () => screen.getByRole('region', { name: /Every model on every machine/i });

  it('does not blank half the catalog when the format is expert-only', () => {
    // The claim is about the catalog, so it is measured against the catalog.
    atFullGrid();
    render(<App />);
    // The default quant is MXFP4, which applies to no dense model. Forcing it across the grid
    // reported "does not apply" for a majority of rows — a quantization fact standing in for a
    // hardware comparison, on the surface whose only job is comparing hardware.
    // The phrase appears in the header and again in the table caption; either will do.
    const [, ran, total] = within(matrix())
      .getAllByText(/combinations run/)[0]
      .textContent!.match(/(\d+)\D+(\d+)/)!;

    expect(Number(ran) / Number(total)).toBeGreaterThan(0.8);
    expect(within(matrix()).getByText(/where it does not apply/i)).toBeInTheDocument();
  });

  it('says what every cell means without relying on its colour', () => {
    atFullGrid();
    render(<App />);
    const cells = within(matrix()).getAllByRole('button', { name: / on / });
    expect(cells.length).toBeGreaterThan(100);
    // Each carries model, device and the measured figure.
    expect(cells[0]).toHaveAccessibleName(/ on .+:/);
  });

  it('loads a cell into the Bench when clicked', async () => {
    const user = userEvent.setup();
    render(<App />);

    const before = useConfig.getState().modelId;
    const cells = within(matrix()).getAllByRole('button', { name: / on / });
    await user.click(cells[cells.length - 1]);

    const after = useConfig.getState();
    expect(`${after.modelId}/${after.deviceId}`).not.toBe(`${before}/${DEFAULT_CONFIG.deviceId}`);
  });

  it('rearranges when the measure changes, which is the point of having three', async () => {
    const user = userEvent.setup();
    render(<App />);

    const fills = () =>
      within(matrix())
        .getAllByRole('button', { name: / on / })
        .map((b) => b.getAttribute('style'));

    const byFit = fills();
    await user.click(within(matrix()).getByRole('button', { name: 'How fast' }));
    expect(fills()).not.toEqual(byFit);
  });

  /**
   * The ramp has to be **spent** on this grid too, and it was the worse of the two surfaces
   * ([#97](https://github.com/MrZoller/bench/issues/97)).
   *
   * The Matrix has offered "How responsive" for longer than the Envelope has, and its domain was
   * floored at zero — which for a reciprocal reduces the placement to `t_fastest / t`, so a cell ten
   * times slower than the grid's fastest painted the bottom step on a grid spanning a desktop CPU to
   * a B200. Measured at the default scenario: **1,025 of 1,269 timed cells on one step of seven**,
   * 81%, against decode's healthy 262 at its busiest. After the fix, 323 at its busiest — 25%.
   *
   * Full grid deliberately. The claim is about the shape of the real field, and a dozen cells cannot
   * have one; a bounded grid would satisfy every threshold here for want of anything to distribute.
   */
  it('spends the ramp across the field, not on one step of it', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(matrix()).getByRole('button', { name: 'How responsive' }));

    /*
     * The painted cells, which are the ones on the scale — a pair that cannot run is `transparent`
     * by design and is not a low score. Reading the inline background is the same channel the ramp
     * arithmetic sweep below uses.
     */
    const painted = within(matrix())
      .getAllByRole('button', { name: / on / })
      .map((b) => (b as HTMLElement).style.background)
      .filter((fill) => fill && fill !== 'transparent');
    expect(painted.length, 'the grid painted nothing from the ramp').toBeGreaterThan(300);

    // jsdom serialises an inline colour as `rgb(r, g, b)`, so the ramp's hexes are put in the same
    // form rather than compared across notations — which silently matches nothing.
    const asRgb = (hex: string) =>
      `rgb(${[1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16)).join(', ')})`;
    const held = magnitudeRamp.map((step) => painted.filter((f) => f === asRgb(step)).length);
    expect(
      held.reduce((a, b) => a + b, 0),
      'no painted cell matched a step of the ramp, so the notations disagree'
    ).toBe(painted.length);
    /*
     * Two properties, and the second is the one the old code failed. "Both ends are reached" was
     * already true of the collapsed ramp — the fastest and slowest cells still landed at the ends —
     * so only the distribution tells the two apart.
     */
    expect(
      held.filter((n) => n === 0),
      'a step of the ramp went unused'
    ).toEqual([]);
    expect(
      Math.max(...held) / painted.length,
      `one step holds ${Math.max(...held)} of ${painted.length} painted cells`
    ).toBeLessThan(0.5);
  });
});

/**
 * Every figure on the grid used to be behind a native `title` (#71).
 *
 * A mouse, a second of dwell, one cell at a time, gone on the next move. That leaves the colour and
 * nothing else for three readers: a sighted keyboard user, who sees the ring move and never hears the
 * `aria-label`; a touch user, who has no hover at all and can only read a cell by committing to it,
 * which rewrites five store keys and scrolls several sections away; and anyone comparing two cells,
 * since a native tooltip shows one and dismisses it on the way to the second. `fill` log-scales onto
 * seven steps deliberately, so the colour is a rank and never a magnitude — which leaves no other
 * route to one.
 *
 * All of this is DOM: which string the line holds, when it fills, when it clears, and whether it is
 * announced. jsdom answers every one of those in a second. What it cannot answer is whether the
 * reservation actually holds a line of height, which is `e2e/matrix-readout.spec.ts` — the same split
 * the focus-indicator suite draws between the indicator a control *declares* and the one it paints.
 */
describe('the comparison grid puts a cell’s value where it can be read', () => {
  const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });

  /**
   * The readout, addressed structurally.
   *
   * It has no role of its own on purpose — see the `aria-hidden` test below — so there is no accessible
   * name to find it by. The section's only direct paragraph is unambiguous: the workload caveat sits
   * inside the `header` and the measure hint inside the `fieldset`. Deliberately *not* keyed on
   * `aria-hidden`, which is the subject of one of the tests below and would make that one circular —
   * and the assertions here all compare its text against a cell's own `aria-label`, so a locator that
   * found the wrong element could not pass for the wrong reason either way.
   */
  const line = () => matrix().querySelector<HTMLElement>(':scope > p')!;
  /**
   * The wide form, which is what every assertion below is about.
   *
   * The line holds two sentences since #102 — the full one and a preamble-less one for widths where
   * the reservation cannot hold the wrapping — and CSS shows one. jsdom resolves no media query, so
   * it renders both and `textContent` returns them concatenated; `data-readout` is the seam that
   * lets this ask for the one it means. The narrow form has its own tests below.
   */
  const readout = () => line().querySelector('[data-readout="full"]')!.textContent;
  const briefReadout = () => line().querySelector('[data-readout="brief"]')!.textContent;

  /**
   * A tap, with the focus the browser interposes — which is the whole of what makes this test real.
   *
   * Tapping a button focuses it, so `onFocus` fires *before* `click` and the readout is already this
   * cell's by the time the handler runs. Written as `pointerDown` then `click`, these tests passed
   * against a guard that compared the live readout target and therefore never fired — the first tap
   * committed, and so did a tap on a different cell. Raised in review on #102, and the shape is the
   * repo's own: an event sequence the real browser produces and the fixture did not.
   */
  const tap = (cell: HTMLElement) => {
    // `pointerenter` with a button already down, which is what a finger arriving *is* — and the
    // reason the readout's hover record does not pick it up. A mouse arrives with `buttons: 0`.
    fireEvent.pointerEnter(cell, { pointerType: 'touch', buttons: 1, pointerId: 7 });
    fireEvent.pointerDown(cell, { pointerType: 'touch', pointerId: 7 });
    // `act`, so React has *committed* the focus before the click — which is the half that makes this
    // bite. Without it the state update is still queued when `click` runs and the handler reads the
    // previous render's target, which is the value the fix stores deliberately: the test would pass
    // against the defect for the same reason the defect was invisible.
    act(() => {
      cell.focus();
    });
    // `detail: 1`, because a pointer-generated click carries a count and a keyboard one does not —
    // which is how the handler tells them apart. A bare `fireEvent.click` is `detail: 0` and would
    // be read as Enter.
    fireEvent.click(cell, { detail: 1 });
  };
  const cells = () => [...matrix().querySelectorAll<HTMLButtonElement>('td button')];
  const legend = () => [...matrix().querySelectorAll<HTMLElement>(':scope > div')].at(-1)!;

  /**
   * The narrow form, and the two halves of [#102](https://github.com/MrZoller/bench/issues/102) it
   * belongs to.
   *
   * The readout's reservation is in `rem`, so at a 32px root it doubles — and the glyphs double with
   * it, so the same sentence wraps into roughly twice as many lines each twice as tall. Reserved
   * height grows linearly and required height closer to quadratically, so the line escaped its box:
   * measured at 320px with the browser default at 32px, the longest sentence needs **280px against
   * 160px reserved**, and 240 against 160 at 390. Above `sm` it fits at both root sizes.
   *
   * What makes it long is the preamble, and at phone width the preamble is the part the reader
   * already has — the model is the row heading and the device the column heading. So the narrow form
   * states the figure alone. `e2e/reflow.spec.ts` measures that it now fits; these pin what it says.
   */
  it('states the figure alone at a width the preamble does not fit', async () => {
    const user = userEvent.setup();
    render(<App />);
    const cell = cells()[0];
    await user.hover(cell);

    const full = readout()!;
    const brief = briefReadout()!;

    /*
     * The model goes and the machine stays, which is "drop what is sticky, keep what scrolls": the
     * row heading is `sticky left-0` and on screen at every scroll position, while the column
     * headings sit at the top of a 35-row grid. Dropping both left a reader in a lower row with
     * `69% of the ceiling free` and nothing anywhere saying which machine (found in review).
     */
    expect(full).toMatch(/^Qwen3 8B on GeForce RTX 5090/);
    expect(brief).not.toContain('Qwen3 8B');
    expect(brief).toContain('RTX 5090');
    // The figure itself, which is the payload both forms carry.
    expect(brief).toContain('69% of the ceiling free');
    expect(full).toContain('69% of the ceiling free');
    /*
     * Materially shorter, which is the property the reservation depends on — the geometry itself is
     * `e2e/reflow.spec.ts`, since jsdom reports every height as 0. The margin is smaller than it was
     * when this dropped the device too, and deliberately: the sentence has to fit *and* say which
     * machine, and the browser check is what holds the first of those.
     */
    expect(brief.length).toBeLessThan(full.length * 0.8);
  });

  it('keeps the stand-in qualifier in the narrow form, since no axis carries it', async () => {
    const user = userEvent.setup();
    render(<App />);
    // MXFP4 is expert-only, so a dense row is scored at a substitute — the one part of the preamble
    // that is not on an axis, and the one a figure derived from it has to keep at every width.
    await user.hover(cells()[0]);

    expect(readout()).toMatch(/at Q4_K_M/);
    expect(briefReadout()).toMatch(/at Q4_K_M/);
  });

  it('gives a screen reader the full sentence at every width', () => {
    render(<App />);
    // The accessible name is the one channel with no axis headings beside it, so the preamble the
    // readout may drop is exactly what a screen-reader user has instead of them. Never abbreviated.
    const label = cells()[0].getAttribute('aria-label') ?? '';
    expect(label).toMatch(/^Qwen3 8B on GeForce RTX 5090/);
  });

  /**
   * Inspection separated from activation, for the reader who has only a tap.
   *
   * #71 left this open and named it: on a touch-only device the only gesture a cell offers is a tap,
   * and that tap *is* `onClick` — five store keys rewritten and a scroll several sections away. So
   * the readout either filled while navigation was already happening or never filled at all, and a
   * touch reader could not compare two cells. Unlike the Envelope and the budget bar, this panel has
   * no table behind a disclosure to fall back to: it *is* the table, and its cells show colour.
   *
   * **The rule is a state, not a gesture: you may commit to a cell whose figures you have already
   * been shown.** Keying on `pointerType === 'touch'` was the first draft and assumed `pen` hovers,
   * which a direct-contact stylus does not — so a pen tap fell straight through to activation exactly
   * as the unfixed touch path did. Reading the readout's target at `pointerdown` answers every input
   * from one comparison instead: a mouse has hovered, a hovering pen has hovered, a finger has not,
   * and the keyboard has no `pointerdown` at all.
   */
  it('fills the line on the first tap and loads the cell on the second', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    tap(cell);

    // Inspected, not committed.
    expect(readout()).toContain('RTX 3060');
    expect(useConfig.getState().deviceId).toBe(before);

    tap(cell);

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('loads a cell on a single click from a mouse, on the same markup', () => {
    render(<App />);
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    // A mouse arrives before it clicks, which is what makes one click enough — and the reason this
    // needs no pointer-type test: the hover *is* the inspection.
    fireEvent.pointerEnter(cell, { pointerType: 'mouse', buttons: 0, pointerId: 1 });
    fireEvent.mouseEnter(cell);
    fireEvent.pointerDown(cell, { pointerType: 'mouse', pointerId: 1 });
    fireEvent.click(cell, { detail: 1 });

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('asks a contact-only stylus for a second tap, since it never hovered', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    /*
     * A direct-contact stylus reports `pen` and cannot show the readout before it lands, so the
     * first draft's `pointerType === 'touch'` test let it through and it committed on contact — the
     * unfixed touch path, one pointer type over. Nothing here mentions `pen`: it inspects first
     * because it has not hovered, which is the same reason a finger does.
     */
    // The contact itself generates the enter, with the button already down — which is exactly why
    // it is not a hover, and why inferring provenance from the *reading* gesture's pointer type took
    // this for a mouse and committed on contact.
    fireEvent.pointerEnter(cell, { pointerType: 'pen', buttons: 1, pointerId: 3 });
    fireEvent.pointerDown(cell, { pointerType: 'pen', pointerId: 3 });
    act(() => {
      cell.focus();
    });
    fireEvent.click(cell, { detail: 1 });

    expect(readout()).toContain('RTX 3060');
    expect(useConfig.getState().deviceId).toBe(before);
  });

  it('keeps one click for a mouse that has not moved since the keyboard did', () => {
    render(<App />);
    const [under, elsewhere] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    /*
     * The mouse rests on one cell while the keyboard arrows to another — and a stationary mouse
     * emits no further `mouseenter`, so the record of where it is has to survive the focus move.
     * Expiring it there was the previous round's fix for a returning *tap* and it broke this: an
     * ordinary click needed two after any keyboard use. Provenance at `pointerdown` answers both,
     * which is why nothing expires here now (found in review).
     */
    fireEvent.pointerEnter(under, { pointerType: 'mouse', buttons: 0, pointerId: 1 });
    fireEvent.mouseEnter(under);
    // The premise: the hover really did register, or this measures a mouse that never arrived.
    expect(readout(), 'the mouse hover did not reach the readout').toContain('RTX 3060');
    act(() => {
      elsewhere.focus();
    });
    // And the keyboard really did take the line, which is the state the record has to survive.
    expect(readout(), 'the keyboard did not take the readout').toContain('DGX Spark');

    fireEvent.pointerDown(under, { pointerType: 'mouse', pointerId: 1 });
    fireEvent.click(under, { detail: 1 });

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('does not let a finger inherit a mouse’s hover on a hybrid', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const [hovered, focusedElsewhere] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    /*
     * A mouse hovers one cell, the keyboard moves the readout to another, and a finger then lands on
     * the hovered one. Without an identity on the record the finger inherits the mouse's hover and
     * commits on its first tap, while the line still describes the cell the keyboard is on — which is
     * the whole gesture broken on any laptop with a touchscreen (found in review).
     */
    fireEvent.pointerEnter(hovered, { pointerType: 'mouse', buttons: 0, pointerId: 1 });
    act(() => {
      focusedElsewhere.focus();
    });

    fireEvent.pointerEnter(hovered, { pointerType: 'touch', buttons: 1, pointerId: 9 });
    fireEvent.pointerDown(hovered, { pointerType: 'touch', pointerId: 9 });
    act(() => {
      hovered.focus();
    });
    fireEvent.click(hovered, { detail: 1 });

    expect(useConfig.getState().deviceId, 'the finger committed on its first tap').toBe(before);
    expect(readout()).toContain('RTX 3060');
  });

  it('names the runtime when a narrow readout reports a refusal', async () => {
    const user = userEvent.setup();
    render(<App />);
    // MLX is Apple-only, so most columns are struck and every cell in them is a refusal.
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const refused = cells().find((c) =>
      /does not (run|support)/i.test(c.getAttribute('aria-label') ?? '')
    )!;
    expect(refused, 'no refused cell under MLX, so this has no subject').toBeDefined();
    await user.hover(refused);

    /*
     * The machine comes from the column heading and the runtime does not — the Runtime picker is
     * above the grid and off screen for a lower row, so eliding it left "the runtime does not drive
     * this", which is the one fact the axes cannot supply (found in review).
     */
    expect(briefReadout()).toContain('MLX (Apple)');
    expect(briefReadout()).not.toMatch(/^the runtime/i);
  });

  it('loads a cell from the keyboard, with no pointer gesture in front of it', () => {
    render(<App />);
    const cell = cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!;

    // Enter on a focused cell fires `click` with no `pointerdown`, and focus has already filled the
    // line — so there is nothing the gesture could have meant other than "load this".
    act(() => {
      cell.focus();
    });
    fireEvent.click(cell);

    expect(useConfig.getState().deviceId).toBe('rtx-3060-12gb');
  });

  it('lets the keyboard activate after a tap, rather than inheriting it', () => {
    render(<App />);
    const [tapped, typed] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    /*
     * The snapshot is consumed on every click, and this is why. Held, it would still describe the
     * tap when a later keyboard `click` arrived with no `pointerdown` of its own — so Enter would be
     * read as that tap still in progress and refuse to activate, for ever, since nothing else would
     * clear it. Found in review.
     */
    tap(tapped);
    act(() => {
      typed.focus();
    });
    fireEvent.click(typed);

    expect(useConfig.getState().deviceId).toBe('dgx-spark');
  });

  it('moves the line to the next cell a finger taps rather than loading it', () => {
    render(<App />);
    const before = useConfig.getState().deviceId;
    const [first, second] = [
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('RTX 3060'))!,
      cells().find((c) => (c.getAttribute('aria-label') ?? '').includes('DGX Spark'))!,
    ];

    tap(first);
    tap(second);

    // Comparing two cells is the thing #71 said a touch reader could not do, and the second tap
    // going to a *different* cell is what makes it possible without committing to either.
    expect(readout()).toContain('DGX Spark');
    expect(useConfig.getState().deviceId).toBe(before);
  });

  it('reserves the line before anything is pointed at', () => {
    render(<App />);

    // Present and empty, rather than absent until it has something to say: a line that appears
    // reflows whatever is under it every time a reader moves between two cells. Whether the
    // reservation is worth a line of height is geometry, and geometry is e2e's half.
    expect(line()).toBeInTheDocument();
    expect(readout()).toBe('');
    expect(line().className).toMatch(/min-h-/);
  });

  it('fills on focus, which is the half that answers the keyboard', () => {
    render(<App />);
    const cell = cells()[5];

    act(() => cell.focus());

    // The same sentence the cell announces, not a second wording of it: one `tooltip()` call feeding
    // both channels is what keeps the line and the accessible name from drifting apart.
    expect(readout()).toBe(cell.getAttribute('aria-label'));
    expect(readout()).toMatch(/ on .+:/);

    act(() => cell.blur());
    expect(readout()).toBe('');
  });

  it('fills on hover, without the dwell the tooltip charged for', () => {
    render(<App />);
    const cell = cells()[7];

    fireEvent.mouseEnter(cell);
    expect(readout()).toBe(cell.getAttribute('aria-label'));

    fireEvent.mouseLeave(cell);
    expect(readout()).toBe('');
  });

  it('falls back to the focused cell when the pointer leaves, rather than blanking', () => {
    render(<App />);
    const [held, pointed] = [cells()[0], cells()[9]];

    act(() => held.focus());
    fireEvent.mouseEnter(pointed);
    // The pointer wins while there is one — it is the more recent intent.
    expect(readout()).toBe(pointed.getAttribute('aria-label'));

    fireEvent.mouseLeave(pointed);
    // And the focus ring is still drawn on the first cell, so its sentence is still the true one. A
    // single `hovered` flag blanks here, which is a visible mark with nothing on the page explaining
    // it — the disagreement `isCurrent` exists to prevent, pointed the other way.
    expect(document.activeElement).toBe(held);
    expect(readout()).toBe(held.getAttribute('aria-label'));
  });

  /**
   * And the other order, which is the one that shipped broken.
   *
   * `hovered ?? focused` is the fallback above with no way back out of it: a pointer resting anywhere
   * on the grid fires `mouseenter` and, because the mouse does not move, never fires `mouseleave`. A
   * reader who then tabs in and arrows across a row moves the ring while the line goes on printing the
   * cell the pointer happens to be sitting on — indefinitely, and with no `hover:` style on the cells,
   * the ring is the only mark on screen. That is a *wrong* figure where #71's sighted keyboard reader
   * previously got none, which is worse than the defect being fixed.
   *
   * No contrivance in this test but the missing `mousemove`, which is the whole point: the pointer is
   * where it was and the keyboard has moved twice.
   */
  it('lets the keyboard take the line back from a pointer that has stopped moving', () => {
    atFullGrid();
    render(<App />);
    const resting = cells()[0];

    fireEvent.mouseEnter(resting);
    expect(readout()).toBe(resting.getAttribute('aria-label'));

    const stepped = cells()[300];
    act(() => stepped.focus());
    fireEvent.keyDown(stepped, { key: 'ArrowRight' });
    const arrowed = document.activeElement as HTMLButtonElement;

    // The ring moved, so the line has to have moved with it.
    expect(arrowed).not.toBe(resting);
    expect(readout()).toBe(arrowed.getAttribute('aria-label'));
    expect(readout()).not.toBe(resting.getAttribute('aria-label'));

    // And the pointer is not disabled by having been outranked once: the next `mouseenter` is a new
    // move, and the rule is that the newer input wins.
    const pointed = cells()[7];
    fireEvent.mouseEnter(pointed);
    expect(readout()).toBe(pointed.getAttribute('aria-label'));
  });

  /**
   * And the way back that `mouseenter` alone cannot provide (found in review on #71).
   *
   * The test above proves the pointer can reclaim the line by entering a *different* cell. It cannot
   * prove it can reclaim the cell it is already in, and it could not: `mouseenter` needs a boundary
   * crossing, so once `onFocus` expired the hover claim, a reader whose mouse was resting on a cell
   * had to leave it and come back. Last-input-wins held in one direction only, and the direction it
   * failed in is the one a mixed keyboard-and-mouse reader hits first — the mouse is where they left
   * it, and the hand that moves is the one already on it.
   *
   * `mousemove` without a preceding `mouseenter` is exactly that state, and it is why the contrivance
   * is the *absence* of an enter rather than the presence of a move.
   */
  it('lets a pointer that never left reclaim the line by moving in place', () => {
    atFullGrid();
    render(<App />);
    const resting = cells()[3];

    fireEvent.mouseEnter(resting);
    expect(readout()).toBe(resting.getAttribute('aria-label'));

    // Keyboard takes it, which also expires the pointer's claim.
    const stepped = cells()[200];
    act(() => stepped.focus());
    expect(readout()).toBe(stepped.getAttribute('aria-label'));

    // The pointer has not moved between cells and so fires no `mouseenter` — only a move inside the
    // one it is already in.
    fireEvent.mouseMove(resting);
    expect(
      readout(),
      'a pointer resting on a cell cannot get the line back without leaving it'
    ).toBe(resting.getAttribute('aria-label'));

    // The focus ring has not moved, so the fallback is still the stepped cell once the pointer goes.
    expect(document.activeElement).toBe(stepped);
    fireEvent.mouseLeave(resting);
    expect(readout()).toBe(stepped.getAttribute('aria-label'));
  });

  /**
   * The line is derived from where the reader is, not stored when they get there.
   *
   * Reachable without contriving anything: a pointer resting on a cell while the keyboard drives one
   * of the Usage sliders — which is what this does, since that is exactly what a slider does to the
   * store. Nothing moves the pointer, so no `mouseleave` fires, and every figure on the grid changes
   * underneath it. A stored string would keep quoting the old scenario over the new grid.
   */
  it('tracks the scenario instead of freezing the sentence it arrived with', () => {
    render(<App />);
    // A cell that runs, so it has a figure to report at all.
    const cell = cells().find((c) => /: \d/.test(c.getAttribute('aria-label') ?? ''))!;

    fireEvent.mouseEnter(cell);
    const before = readout();
    expect(before).toBe(cell.getAttribute('aria-label'));

    act(() => useConfig.getState().set('contextTokens', 131072));

    expect(readout()).not.toBe(before);
    expect(readout()).toBe(cell.getAttribute('aria-label'));
  });

  /**
   * The one place this deliberately departs from `BudgetBar`'s hint, which is `aria-live="polite"`.
   *
   * There the sentence exists nowhere else — a legend item is named "Weights 14 GiB" and the
   * explanation is not part of that name. Here the line is a verbatim copy of the focused cell's
   * accessible name, so a live region would announce every cell twice: once as the name, once as the
   * update, on every arrow key across a 42-column row. The channel #71 is missing is the visual one.
   */
  it('does not read the cell’s own sentence out a second time', () => {
    render(<App />);
    const cell = cells()[11];

    act(() => cell.focus());

    expect(line().getAttribute('aria-hidden')).toBe('true');
    expect(line().getAttribute('aria-live')).toBeNull();
    // And the spoken channel is untouched: the cell still carries the whole sentence.
    expect(cell.getAttribute('aria-label')).toBe(readout());
  });

  /**
   * The sweep. #71 names the cells, and the same defect was live on both headings.
   *
   * `headerColumns` shortens every device name — the vendor line goes, and the bracketed qualifier
   * with it wherever the stem is already unique — and the full name lived only in a `title`. The row
   * heading truncates at 9rem and put the model name *and* its parameter count in one. Three
   * hover-only strings, one line to put them in.
   */
  it('names the machine a shortened column heading stands for', () => {
    render(<App />);
    const heading = [...matrix().querySelectorAll('thead th')].at(-1)!;
    const label = heading.querySelector('span[title]')!;

    fireEvent.mouseEnter(heading);

    // The full catalog name, which is longer than what the column can print.
    expect(readout()).toBe(label.getAttribute('title'));
    expect(readout()!.length).toBeGreaterThan(label.textContent!.length);
  });

  it('carries the runtime refusal a struck heading only says in colour and ink', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const struck = [...matrix().querySelectorAll('thead th')].find((th) =>
      th.querySelector('span[title]')?.className.includes('line-through')
    )!;
    fireEvent.mouseEnter(struck);

    // Same string as the heading's own `aria-label`, from the same derivation — the strike is the
    // sighted channel and it says nothing about why on its own.
    expect(readout()).toBe(struck.getAttribute('aria-label'));
    expect(readout()).toMatch(/does not support this hardware, at any size/i);
  });

  it('punctuates a refusal once, now that the sentence is printed rather than hovered', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    // vLLM drives no Mac, no Strix Halo and no CPU host, so this fills the grid with the refusals
    // `planPlacement` writes — each of which already ends in a full stop.
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const blocked = cells().filter((c) =>
      /does not run on/i.test(c.getAttribute('aria-label') ?? '')
    );
    expect(blocked.length).toBeGreaterThan(50);

    fireEvent.mouseEnter(blocked[0]);
    expect(readout()).toMatch(/does not run on .+[^.]\.$/);
    for (const cell of blocked) expect(cell.getAttribute('aria-label')).not.toContain('..');
  });

  it('spells out a truncated model row, parameter count and all', () => {
    atFullGrid();
    render(<App />);
    const heading = matrix().querySelectorAll('tbody th')[3];

    fireEvent.mouseEnter(heading);

    expect(readout()).toBe(heading.getAttribute('title'));
    expect(readout()).toContain(heading.textContent);
    // The parameter count, which had no channel at all besides that `title`.
    expect(readout()).toMatch(/\d+(\.\d+)?B$/);
  });

  /**
   * And the parameter count needs a spoken channel of its own, which the line cannot be.
   *
   * The readout is `aria-hidden` for the reason above — it copies a cell's accessible name, so a live
   * region says every cell twice — and these headings take no focus, so hover was the *only* route to
   * the count: a cell's sentence names the model but never its size. An `aria-label` on the row header
   * is the same answer the column header already uses for its runtime refusal, from the same one
   * derivation, and a grid announces a row header once per row rather than once per cell.
   */
  it('tells a screen reader the row’s parameter count, which no cell states', () => {
    atFullGrid();
    render(<App />);
    const headings = [...matrix().querySelectorAll('tbody th')];

    expect(headings.length).toBeGreaterThan(10);
    for (const heading of headings) {
      // One string in all three channels — the visible name, the tooltip and the accessible name.
      expect(heading.getAttribute('aria-label')).toBe(heading.getAttribute('title'));
      expect(heading.getAttribute('aria-label')).toMatch(/\d+(\.\d+)?B$/);
      expect(heading.getAttribute('aria-label')).toContain(heading.textContent);
    }
  });

  /**
   * And the ramp is a scale now rather than an ordering.
   *
   * `worse [gradient] better` says which way to read the colour and nothing about what it spans, so a
   * mid-blue under "How fast" could be 20 tok/s or 200. The endpoints are asserted against the
   * *cells'* own sentences rather than recomputed here: whatever the engine says, the figure at each
   * end of the legend has to be the figure some cell reports, and the extreme one.
   */
  const spoken = (cells: HTMLButtonElement[], pattern: RegExp) =>
    cells.flatMap((cell) => {
      const found = pattern.exec(cell.getAttribute('aria-label') ?? '');
      return found ? [found[1]] : [];
    });

  /**
   * The extreme is chosen by number and asserted as the *string the cell printed*.
   *
   * `Math.min(...rates.map(Number))` and `toContain` looks equivalent and is not: `rate()` keeps a
   * decimal below 10 tok/s, so the day the slowest running cell lands on an exact tenth-free value the
   * legend correctly renders "worse 3.0 tok/s" while the round trip through `Number` asks for "worse
   * 3 tok/s" — a red test with nothing wrong in the code. It passes today only because the minimum is
   * 0.5. Same reason the TTFT test below compares by `asSeconds` and asserts the figure verbatim.
   */
  const extremeBy = (figures: string[], value: (figure: string) => number, want: 'min' | 'max') =>
    figures.reduce((a, b) =>
      want === 'min' ? (value(b) < value(a) ? b : a) : value(b) > value(a) ? b : a
    );

  it('anchors the ramp with the throughput its own cells report', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(matrix()).getByRole('button', { name: 'How fast' }));

    const rates = spoken(cells(), /: ([\d.]+) tok\/s per user\.$/);
    expect(rates.length).toBeGreaterThan(100);

    const text = legend().textContent!;
    expect(text).toContain(`worse ${extremeBy(rates, Number, 'min')} tok/s`);
    expect(text).toContain(`${extremeBy(rates, Number, 'max')} tok/s better`);
  });

  it('puts the longest wait at the worse end, which the stored value inverts', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(matrix()).getByRole('button', { name: 'How responsive' }));

    /** "188 ms", "2.7 s", "84 min" — one axis, three units, as `seconds()` prints them. */
    const UNIT: Record<string, number> = { ms: 0.001, s: 1, min: 60 };
    const asSeconds = (figure: string) => {
      const [value, unit] = figure.split(' ');
      return Number(value) * UNIT[unit];
    };

    const waits = spoken(cells(), /: (.+) to first token\.$/);
    expect(waits.length).toBeGreaterThan(100);
    const longest = extremeBy(waits, asSeconds, 'max');
    const shortest = extremeBy(waits, asSeconds, 'min');

    // `measureValue` inverts TTFT so that larger is better, so the ramp's worst end is the *slowest*
    // machine. An endpoint pair ordered by the number it prints puts the fastest one under "worse"
    // and leaves every colour on the grid correct, which is what makes it invisible.
    const text = legend().textContent!;
    expect(text).toContain(`worse ${longest}`);
    expect(text).toContain(`${shortest} better`);
    expect(asSeconds(longest)).toBeGreaterThan(asSeconds(shortest));
  });

  it('anchors the fit ramp with the most headroom any cell has', () => {
    atFullGrid();
    render(<App />);

    const free = spoken(cells(), /: (\d+)% of the ceiling free\.$/);
    expect(free.length).toBeGreaterThan(100);

    const text = legend().textContent!;
    expect(text).toContain(`${extremeBy(free, Number, 'max')}% free better`);
  });

  /**
   * And the low end of the *fit* ramp is the one endpoint that is not a cell's own figure.
   *
   * `measureValue('fit')` collapses every offloaded cell to zero headroom deliberately, so the dark
   * end is a population: a pair that just fits and a pair spilling most of its weights paint the same
   * square. "0% free" is the one statement true of all of them, which is why it is what the label
   * says — and the consequence is that the figure appears on no tooltip, since a spilled cell's own
   * sentence quotes the spill instead. Pinned rather than left to `/worse \d+% free/`, which any digit
   * satisfies: printing the tied cell's own 66% spill would describe every other square wrongly, and
   * `measureRange` hands back whichever tied cell came first in row-major order, so a label reading
   * any other field off it would be reading an arbitrary cell.
   */
  it('says the fit ramp’s dark end has no headroom, not what the worst cell spills', () => {
    render(<App />);

    const spilling = cells().filter((c) =>
      /spilling \d+% of its weights/.test(c.getAttribute('aria-label') ?? '')
    );
    expect(spilling.length, 'no cell spills, so the low end is a resident cell').toBeGreaterThan(0);

    const text = legend().textContent!;
    expect(text).toContain('worse 0% free');
    expect(text).not.toMatch(/worse \d+% (spilled|of its weights)/);
  });
});

/**
 * "Will not run" and "this runtime cannot drive it" were the same empty cell (#72).
 *
 * Select vLLM and every Mac, every Strix Halo and every CPU host empties out completely — 10 of the
 * 24 shipping columns as the catalog stands at this commit — every cell drawn `transparent` behind
 * the same dashed border as a pair that was measured and did not fit, under the same one-line
 * legend. A uniformly empty column is the pattern that reads as a confident finding, so the picture
 * said "this hardware cannot hold the model" — quantitatively backwards, since a 256 GB Mac Studio
 * holds Qwen3 8B many times over, and the fix a reader would derive from it (buy more memory) is not
 * the fix (change runtime). Every other surface already split them: the Envelope has an
 * `unsupported` state with its own sentence, Telemetry says `Unsupported` rather than `Will not
 * run`, and BudgetBar draws no stack at all.
 *
 * All of it is DOM, so all of it is here. The one thing jsdom cannot answer is whether
 * `line-through` and a dropped border actually *paint* — Tailwind classes are strings in this
 * environment — which is `e2e/matrix-undrivable.spec.ts`.
 */
describe('the Matrix tells a runtime refusal from a memory one', () => {
  const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });

  /**
   * Every device column, paired with its own cells.
   *
   * Read out of the DOM in column order rather than zipped against the catalog, for the reason the
   * header suite gives about its own pairing: the association between a heading and the cells under
   * it is part of what is being tested, and an assertion that assumes it cannot catch it going
   * wrong.
   */
  const columns = () => {
    const rows = [...matrix().querySelectorAll('tbody tr')];
    return [...matrix().querySelectorAll('thead th')].slice(1).map((th, i) => {
      const label = th.querySelector<HTMLElement>('span[title]')!;
      return {
        head: th,
        device: label.getAttribute('title') ?? '',
        struck: label.className.includes('line-through'),
        spoken: th.getAttribute('aria-label'),
        cells: rows.map((row) => row.querySelectorAll<HTMLButtonElement>('td button')[i]),
      };
    });
  };

  const legendKey = () =>
    within(matrix()).queryByText(/does not support this hardware, at any size/i);

  const caption = () => matrix().querySelector('caption')!.textContent ?? '';

  it('strikes the columns the runtime cannot drive, and only those', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);

    // llama.cpp drives every class of hardware in the catalog, so nothing is struck and nothing is
    // keyed — the precondition that keeps the vLLM half below from passing for a trivial reason.
    expect(columns().every((c) => !c.struck)).toBe(true);
    expect(legendKey()).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const struck = columns().filter((c) => c.struck);
    // Both sides populated: vLLM drives NVIDIA and AMD cards and drives no Mac, no Strix Halo and
    // no CPU host. A grid struck everywhere, or nowhere, would make every claim below vacuous.
    expect(struck.length).toBeGreaterThan(1);
    expect(struck.length).toBeLessThan(columns().length);
    expect(struck.some((c) => /Mac Studio/.test(c.device))).toBe(true);
    expect(columns().some((c) => !c.struck && /RTX 5090/.test(c.device))).toBe(true);

    /**
     * And the strike is the engine's own verdict rather than a second opinion about it.
     *
     * The component decides from `runtimeDrives` while every cell's refusal comes from
     * `planPlacement`'s own copy of that check, so this is the assertion that keeps the two from
     * drifting: a struck column's cells must *all* carry the runtime-level reason, and no cell
     * anywhere else may carry it.
     *
     * Marking a column that merely came up empty would be the same misattribution pointed the other
     * way. At #72's own URL the two sets happen to coincide — the DGX Spark still runs a good share
     * of its rows there, so the only empty columns are the undrivable ones — which is exactly why
     * deriving from emptiness looks safe. Take that grid to 32 concurrent users and the RTX 3090,
     * 4090 and 5080 columns empty out too, on counted bytes, under a runtime that drives all three.
     */
    for (const column of struck) {
      for (const cell of column.cells) {
        expect(cell).toHaveAccessibleName(/vLLM does not run on/i);
      }
      expect(column.spoken).toMatch(/vLLM does not support this hardware, at any size/i);
      // The device name stays in it: the visible label is deliberately shortened, so a name that
      // said only the runtime would trade one missing fact for another.
      expect(column.spoken).toContain(column.device);
      // And the sentence really is the column's accessible name rather than an attribute nothing
      // reads — this is the whole channel a reader who cannot see the strike has.
      expect(column.head).toHaveAccessibleName(column.spoken!);
    }
    for (const column of columns().filter((c) => !c.struck)) {
      for (const cell of column.cells) {
        expect(cell).not.toHaveAccessibleName(/does not run on/i);
      }
      // No name at all, so the heading keeps announcing the device it names.
      expect(column.spoken).toBeNull();
    }
  });

  it('keeps the dashed swatch for the cells that were actually measured', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const dashed = (cell: HTMLButtonElement) => cell.className.includes('border-dashed');

    // The swatch keys "measured, and over the ceiling". A column the runtime cannot open at all was
    // never measured, so it wears no ink — which is what stops the two states being byte-identical.
    for (const column of columns().filter((c) => c.struck)) {
      expect(column.cells.some(dashed)).toBe(false);
    }

    /**
     * And the capacity refusal still has its swatch, on a column the runtime does drive.
     *
     * Asserted rather than assumed: DeepSeek V3 does not fit a 3090 under any runtime that can load
     * it, so this is reachable — but if it ever stopped being, the assertion above would be the only
     * one left and "no cell has a dashed border" is a state this fix must not produce.
     */
    const measured = columns()
      .filter((c) => !c.struck)
      .flatMap((c) => c.cells)
      .filter(dashed);
    expect(measured.length).toBeGreaterThan(0);
    for (const cell of measured) {
      expect(cell).toHaveAccessibleName(/does not fit|past the default allocation/i);
    }
  });

  /**
   * Every hole on the grid, and nothing else, sits under a struck heading.
   *
   * The exhaustive version of the assertion above, and the one that keeps the two predicates from
   * coming apart in the direction nothing else watches. The heading is struck from `runtimeDrives`;
   * the cell's ink is dropped on `evaluated`, which `planPlacement` clears on **five** categorical
   * grounds, of which "this runtime does not drive this device" is one. The other four are filtered
   * out upstream today — the store coerces `kvPrecision` into `runtime.kvPrecisions`, `quantFor`
   * only ever returns a format the runtime lists, and this grid hardcodes one device per cell — so
   * the two sets coincide, and an assertion that only checked the `!drives` wording would keep
   * passing on the day one of the other four became reachable. That day the grid grows a column of
   * identical unexplained holes, which is #72 restated with a different ground.
   *
   * A hole is read off what the grid paints rather than out of the engine: `transparent` and no
   * dashed border is exactly "not judged on its numbers", since `fill` returns the panel surface for
   * anything that does not run and the border is what separates counted bytes from a categorical
   * refusal. Asserted as an equality in both directions, so it fails for a stray hole *and* for a
   * struck column whose cells kept their ink.
   */
  it('leaves no hole on the grid that a struck heading does not explain', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const hole = (cell: HTMLButtonElement) =>
      (cell.getAttribute('style') ?? '').includes('transparent') &&
      !cell.className.includes('border-dashed');

    const holes = columns()
      .filter((c) => !c.struck)
      .flatMap((c) => c.cells.filter(hole).map((cell) => `${c.device}: ${cell.ariaLabel}`));
    expect(holes, 'cells refused before the arithmetic with no struck heading saying why').toEqual(
      []
    );

    const closed = columns().filter((c) => c.struck);
    expect(closed.length).toBeGreaterThan(1);
    for (const column of closed) {
      expect(column.cells.every(hole)).toBe(true);
      // And the proxy really is reading refusals rather than figures, so "every cell is a hole"
      // cannot be satisfied by a grid that stopped measuring.
      for (const cell of column.cells) {
        expect(cell).not.toHaveAccessibleName(
          /of the ceiling free|tok\/s|to first token|spilling/i
        );
      }
    }
  });

  /**
   * A square with no ink is not a control.
   *
   * The other half of narrowing the border: these cells now have nothing drawn in them at all, and
   * they were still enabled buttons in the arrow-key sequence whose click set five config keys and
   * smooth-scrolled three sections up to a Bench that can only blank. `tokens.ts` puts the rule as
   * "a control's boundary is what identifies it as interactive, so it needs the 3:1 non-text minimum
   * *before* it is focused", and records `--color-border` at 1.18:1 — so no hairline was going to
   * make these look interactive either. They are inert instead.
   *
   * Still focusable and still named, which is why `aria-disabled` rather than `disabled`: a disabled
   * button takes no focus, so the arrows would stop dead at the first struck column and the per-cell
   * sentence — the only channel that says which machine and which runtime — would go with it.
   */
  it('makes a closed column inert without taking it out of the grid', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    const closed = columns().filter((c) => c.struck);
    const open = columns().filter((c) => !c.struck);
    expect(closed.length).toBeGreaterThan(1);
    expect(open.length).toBeGreaterThan(1);

    for (const column of closed) {
      for (const cell of column.cells) {
        expect(cell).toHaveAttribute('aria-disabled', 'true');
        expect(cell.className).toContain('cursor-not-allowed');
        // Not `disabled`: it has to keep taking focus for the roving tab stop to cross the column.
        expect(cell.disabled).toBe(false);
      }
    }
    for (const column of open) {
      for (const cell of column.cells) {
        expect(cell).not.toHaveAttribute('aria-disabled');
        expect(cell.className).not.toContain('cursor-not-allowed');
      }
    }

    // And clicking one loads nothing. `aria-disabled` is advisory — the browser still fires the
    // click — so the handler has to refuse it, which is what this actually checks.
    const before = useConfig.getState();
    await user.click(closed[0].cells[0]);
    const after = useConfig.getState();
    expect(`${after.modelId}/${after.deviceId}/${after.quantId}`).toBe(
      `${before.modelId}/${before.deviceId}/${before.quantId}`
    );

    // While a column the runtime does drive still adopts its cell, so the refusal above is the
    // narrow one and not a click handler that stopped working.
    await user.click(open[0].cells[0]);
    expect(useConfig.getState().deviceId).not.toBe(before.deviceId);
  });

  it('keys the strike in the legend, and only while the grid holds one', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    // The Envelope's reviewed sentence, with the runtime named — one wording for one refusal.
    expect(legendKey()).toBeInTheDocument();
    expect(legendKey()).toHaveTextContent(/vLLM does not support this hardware, at any size/i);
    // The sample is the mark: struck text, not a swatch beside it.
    expect(legendKey()!.querySelector('.line-through')).not.toBeNull();

    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    // Still keyed, and now naming MLX — the sentence follows the runtime rather than being frozen
    // at whichever one first rendered it.
    expect(legendKey()).toHaveTextContent(/MLX \(Apple\) does not support this hardware/i);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    expect(legendKey()).not.toBeInTheDocument();
  });

  it('states the closed columns in the caption, which is the channel with no strike to see', async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(caption()).not.toMatch(/does not support/i);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');

    /**
     * The count read back out of the sentence and checked against the grid, rather than written
     * here as a literal.
     *
     * A hard-coded "10 of 24" would pass a caption that had stopped counting and would fail every
     * time the catalog gains a device. What matters is that the three channels agree: the number of
     * struck headings, the number of columns, and the sentence a screen-reader user hears instead of
     * seeing either.
     */
    const stated = caption().match(
      /(\d+) of the (\d+) device columns are hardware vLLM does not support at any size/i
    );
    expect(stated, `the caption does not state the closed columns: "${caption()}"`).not.toBeNull();
    expect(Number(stated![1])).toBe(columns().filter((c) => c.struck).length);
    expect(Number(stated![2])).toBe(columns().length);
    // And it says which way to read the empty column, since that is the whole misreading.
    expect(caption()).toMatch(/not for want of memory/i);
  });
});

/**
 * The device headers are rotated 45 degrees, which costs height *and* width — one length, spent
 * twice. #64 is what happened when the component derived it once: 246px of header band reserved at
 * every viewport, from a 40-character label, while the same label's sideways extent went unreserved
 * and leaned 142px out of a scroll container the grid otherwise fitted exactly. The app paid a phone
 * screen of vertical space for four names it then cut off.
 *
 * Split across the two suites the way the quantity itself splits. The *derivation* is a string —
 * label text, an inline height, an inline `min-width`, a rotation class — and jsdom reads all four in
 * milliseconds. What those lengths buy in a laid-out browser is `e2e/matrix-header.spec.ts`, because
 * jsdom reports every width on this surface as 0, which is exactly why the sideways half went
 * unnoticed.
 */
describe('the Matrix header reserves the rotation once', () => {
  // #64 is arithmetic over the labels the catalog actually renders — a 40-character name at 45
  // degrees — and the qualifier rule only has something to do where two rows share a name stem.
  // A four-column fixture reproduces neither, so this whole describe takes the real header.
  beforeEach(atFullGrid);
  const matrix = () => screen.getByRole('region', { name: /Every model on every machine/i });

  /**
   * The rendered labels, each paired with the device it names.
   *
   * Read off the `title`, which carries the full catalog name, rather than by zipping the header
   * against `DEVICES` in column order — the pairing is the thing under test, and an assertion that
   * assumes it cannot catch it going wrong.
   */
  const headerLabels = () =>
    [...matrix().querySelectorAll('thead th span[title]')].map((span) => ({
      name: span.getAttribute('title') ?? '',
      label: span.textContent ?? '',
    }));

  /** The part of a name that is not the vendor line and not the trailing parenthetical. */
  const stem = (name: string) =>
    name.replace(/^(GeForce|Instinct|Radeon)\s+/, '').replace(/\s*\([^)]*\)\s*$/, '');

  it('drops the qualifier from every column that can be identified without it', () => {
    render(<App />);
    const labels = headerLabels();
    // The header is the whole shipping catalog; a locator that found four of them would make
    // everything below it pass for the wrong reason.
    expect(labels.length).toBe(DEVICES.filter((d) => d.status === 'shipping').length);

    for (const { label } of labels) {
      expect(label, `"${label}" still spends characters on brackets`).not.toMatch(/[()]/);
    }

    /**
     * Minimal, stated as a rule rather than against a list of names: a label may only be longer
     * than its own stem where another column answers to that same stem.
     *
     * This is the half of the fix that is easy to get wrong in the other direction. Stripping the
     * parenthetical unconditionally is the obvious reading of the issue and it reintroduces the
     * defect the rotation exists to prevent — the three Mac Studio M3 Ultra rows differ *only* in
     * capacity, so they would collapse into one string three columns wide, and a header that
     * cannot distinguish its own columns is worse than none.
     */
    const stems = labels.map((l) => stem(l.name));
    for (const { name, label } of labels) {
      if (label === stem(name)) continue;
      expect(
        stems.filter((s) => s === stem(name)).length,
        `"${label}" is longer than "${stem(name)}", which no other column answers to`
      ).toBeGreaterThan(1);
      expect(label.startsWith(stem(name))).toBe(true);
    }

    // And the rule bites on the shipped catalog rather than being vacuously true of it.
    expect(labels.filter(({ name, label }) => label !== stem(name)).length).toBeGreaterThan(1);
    expect(new Set(labels.map((l) => l.label)).size).toBe(labels.length);
  });

  it('spends one derived length on the band and on the column it leans over, not two', () => {
    render(<App />);

    const band = matrix().querySelector<HTMLElement>('thead th:nth-child(2)');
    const reservation = matrix().querySelector<HTMLElement>('thead th:first-child');
    expect(band, 'the header row reserves no height at all').not.toBeNull();

    const bandRem = parseFloat(band!.style.height);
    const leanRem = parseFloat(reservation!.style.minWidth);
    expect(
      leanRem,
      'the model column reserves no room for the labels leaning over it'
    ).toBeGreaterThan(0);
    // sin(45) and cos(45) are the same number: the band is the lean plus the row's own padding, so
    // the two axes cannot drift apart without this failing.
    expect(bandRem - leanRem).toBeCloseTo(1.25, 6);
    // Both in `rem`, because both are lengths measured from text — a px reservation stops covering
    // its own labels the moment the root font size moves, which is #44 twice over.
    expect(band!.style.height.endsWith('rem')).toBe(true);
    expect(reservation!.style.minWidth.endsWith('rem')).toBe(true);

    /**
     * And the labels lean the way the reservation faces.
     *
     * The reservation is on the model column, to the *left* of every label, which is only the right
     * place if the labels lean left: anchored bottom-right and turned clockwise. The first fix for
     * #64 kept them leaning right and reserved a trailing lane out of whatever free space the
     * viewport had going spare, which is not a quantity — it ran out between 948px and 1009px, and
     * the grid scrolled 50px for header text there while both browser assertions sat above the
     * window. jsdom cannot see a pixel of that, but it can see that the two halves still agree.
     */
    const label = matrix().querySelector('thead th span[title]')!;
    expect(label.className).toContain('origin-bottom-right');
    expect(label.className).toMatch(/(?:^|\s)rotate-45(?:\s|$)/);
  });

  it('reserves a band for the labels it renders, not for the ones it used to', () => {
    render(<App />);

    const bandRem = parseFloat(
      matrix().querySelector<HTMLElement>('thead th:nth-child(2)')!.style.height
    );
    const longest = Math.max(...headerLabels().map((l) => l.label.length));

    // Still long enough for the longest label at the same 0.5rem-per-character estimate the
    // rotation has always been sized by — the band may only shrink because the labels did, never
    // because someone capped it. Clipping the names is the failure the rotation exists to prevent.
    expect(bandRem).toBeGreaterThanOrEqual(longest * 0.5 * Math.SQRT1_2);

    /**
     * And the 246px in the issue has actually moved.
     *
     * 15.39rem was the band when the reservation carried `(12-ch DDR5-4800)` and its neighbours:
     * 40 characters, 246px at the default root, 16% of the Matrix panel on a phone, and unchanged
     * between a 320px screen and a 1440px one. 11rem is 176px — comfortably above the 10.09rem the
     * shipped catalog now asks for, and far enough below 15.39 that restoring the parentheticals
     * fails here rather than in review.
     */
    expect(bandRem).toBeLessThan(11);
  });
});

describe('clicking a Matrix cell loads what that cell was scored under', () => {
  it('carries the quantization the cell was evaluated at, not the one selected', async () => {
    atFullGrid();
    const user = userEvent.setup();
    render(<App />);

    // Default is MXFP4, which the Matrix substitutes for dense rows — so the grid and the Bench
    // would otherwise disagree about the square that was just clicked.
    expect(useConfig.getState().quantId).toBe('mxfp4');

    const matrix = screen.getByRole('region', { name: /Every model on every machine/i });
    const dense = within(matrix)
      .getAllByRole('button', { name: /Qwen3 32B on / })
      .at(0)!;
    await user.click(dense);

    const after = useConfig.getState();
    expect(after.modelId).toBe('Qwen/Qwen3-32B');
    expect(after.quantId).not.toBe('mxfp4');
    expect(after.deviceCount).toBe(1);
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
 * The ring has no visual equivalent for a screen reader, so its sentence has to carry everything
 * the table carries about that one cell — the same wording and the same disambiguated label.
 */
describe('the spoken marker describes the same cell the table does', () => {
  it('borrows the table wording rather than re-deriving it', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-512');
    await user.selectOptions(screen.getByLabelText('Model'), 'deepseek-ai/DeepSeek-V3');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');

    const region = screen.getByRole('region', { name: /how much room/i });
    const spoken = within(region).getByRole('img').getAttribute('aria-label') ?? '';

    // Open the table and read the marked cell, which is the same scenario the ring sits on.
    await user.click(within(region).getByRole('button', { name: /region as a table/i }));
    const table = within(region).getByRole('table');
    const marked = within(table).getByText(/▸/).closest('td')?.textContent ?? '';

    // Whatever the table says about that cell, the ring's sentence must say too.
    const said = marked.replace('▸', '').trim().toLowerCase();
    expect(spoken.toLowerCase()).toContain(said);
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
 * ARIA reference integrity, as a sweep rather than per instance (#131).
 *
 * Two disclosures mounted their region only while expanded, so collapsed they pointed
 * `aria-controls` at an id that was not in the DOM — a reference a screen reader's "jump to
 * controlled region" cannot resolve, and an axe `aria-valid-attr-value` failure. The contract on
 * `DisclosureToggle.controls` now requires the region in both states; this sweeps every
 * `aria-controls` on the default page, which renders both offending disclosures collapsed, so a
 * new call site that unmounts its region fails here rather than in an audit.
 */
describe('every aria-controls points at a node that exists', () => {
  it('resolves each reference on the default page, collapsed states included', () => {
    render(<App />);

    const referencing = Array.from(document.querySelectorAll('[aria-controls]'));
    // The sweep must be sweeping something: both #131 instances ship collapsed by default.
    expect(referencing.length).toBeGreaterThanOrEqual(2);
    for (const el of referencing) {
      const id = el.getAttribute('aria-controls')!;
      expect(
        document.getElementById(id),
        `aria-controls="${id}" resolves to nothing`
      ).not.toBeNull();
    }
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
 * The Matrix substitutes a format wherever the selected one does not apply, and that substitution
 * used to bypass the runtime check entirely — it asked `quantApplies` without the runtime and then
 * returned a hardcoded Q4_K_M. Under vLLM, which reads no GGUF K-quant, every dense row was
 * therefore sized, coloured and ranked at a checkpoint that cannot be loaded, and clicking one
 * landed in a Bench that coerced it to something else and showed different figures.
 */
describe('the Matrix only ever scores a format the runtime can load', () => {
  it('substitutes something vLLM can read, not a GGUF K-quant', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    // MXFP4 is expert-only, so every dense row needs a stand-in.
    await user.selectOptions(screen.getByLabelText('Quantization'), 'mxfp4');

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });

    // The heading states what is standing in, and it cannot be a format vLLM does not load.
    expect(within(matrix).queryByText(/Q4_K_M where it does not apply/i)).not.toBeInTheDocument();

    // No cell may be blocked by the tool's own substitution being unloadable. That string comes
    // from `planPlacement`, which now refuses these pairs — so before the substitute learned about
    // the runtime, this is exactly what the grid filled up with.
    const unloadable = within(matrix)
      .getAllByRole('button')
      .filter((b) => /cannot load/i.test(b.getAttribute('aria-label') ?? ''));
    expect(unloadable).toHaveLength(0);
  });

  it('still prefers the 4-bit stand-in where the runtime does read it', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'mxfp4');

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    // llama.cpp loads GGUF, so the honest local trade is still the substitute — the fix must not
    // have demoted every grid to BF16 on its way to being correct for vLLM.
    expect(within(matrix).getByText(/Q4_K_M where it does not apply/i)).toBeInTheDocument();
  });
});

/**
 * A cell that already matches the selection changes nothing the Matrix renders, so before the
 * selected square was marked, clicking one was indistinguishable from the click not registering.
 * The scroll that accompanies it cannot be tested here — jsdom has no `scrollIntoView` at all,
 * which is how an anchor that generates no box passed for a working one.
 */
describe('the Matrix acknowledges the cell you clicked', () => {
  it('marks the selected square, and moves the mark when the selection moves', async () => {
    const user = userEvent.setup();
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const marked = () =>
      within(matrix)
        .getAllByRole('button')
        .filter((b) => b.getAttribute('aria-current') === 'true');

    // Exactly one square is current at any time — the one the Bench above is showing.
    expect(marked()).toHaveLength(1);
    const before = marked()[0].getAttribute('aria-label');

    // Change the selection from outside the grid; the mark has to follow the store, not the click.
    // Deliberately not the Spark, which is the default rig — selecting it changes nothing, and
    // the assertion below would then pass against a mark that never moved.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');

    expect(marked()).toHaveLength(1);
    expect(marked()[0].getAttribute('aria-label')).not.toBe(before);
  });

  /**
   * Every cell is scored with `deviceCount: 1`. On a linked rig the mark therefore pointed at a
   * square whose capacity and speed describe a different machine from the one the Bench is
   * showing — and clicking it, the one square that ought to be a no-op, silently reset the
   * configuration to a single device.
   */
  it('marks nothing when the Bench is on a rig this grid does not score', async () => {
    const user = userEvent.setup();
    render(<App />);

    const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });
    const marked = () =>
      within(matrix())
        .getAllByRole('button')
        .filter((b) => b.getAttribute('aria-current') === 'true');

    // A device with an interconnect, so the count is offered rather than clamped back to 1.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    expect(marked()).toHaveLength(1);

    act(() => useConfig.getState().set('deviceCount', 4));
    expect(marked()).toHaveLength(0);

    // And the grid says why, rather than leaving the reader to notice the mark has gone.
    expect(within(matrix()).getByRole('heading', { level: 2 })).toHaveTextContent(
      /one device per cell/i
    );

    // Back to a rig the grid does score, and the mark returns.
    act(() => useConfig.getState().set('deviceCount', 1));
    expect(marked()).toHaveLength(1);
  });
});

/**
 * `kvPrecision` is an internal width, not a name anyone types. vLLM has no integer-Q8 cache — the
 * catalog maps that value to FP8 for exactly that reason — so upper-casing it in the heading
 * described a setting that does not exist, in the panel most likely to be screenshotted.
 */
describe('the Matrix names the cache the runtime actually has', () => {
  const heading = () =>
    within(screen.getByRole('region', { name: /every model on every machine/i })).getByRole(
      'heading',
      { level: 2 }
    );

  it('calls the one-byte cache FP8 under vLLM, as the Bench control does', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    expect(heading()).toHaveTextContent(/FP8 KV/);
    expect(heading()).not.toHaveTextContent(/Q8 KV/);
  });

  // llama.cpp keeps the table's own name, which is the fallback path — worth its own case so the
  // fix cannot be mistaken for "always print FP8". (Its real flag is `q8_0`; naming the width
  // rather than the flag is a milder version of the same gap, and is filed separately rather than
  // grown into this change.)
  it('leaves llama.cpp’s Q8 alone, so the fallback path is exercised too', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'rtx-5090');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'llama.cpp');
    act(() => useConfig.getState().set('kvPrecision', 'q8'));

    expect(heading()).toHaveTextContent(/Q8 KV/);
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
 * Tabbable, not merely focusable — the distinction #52's whole fix turns on, since `tabindex="-1"`
 * is reachable by script and never by Tab. Shared by the two suites below because they are two
 * readings of one sequence: which stops exist, and what order a reader meets them in.
 */
const TABBABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]',
]
  .map((selector) => `${selector}:not([tabindex="-1"])`)
  .join(', ');

/**
 * The page leads with the controls that drive it, which #52 explicitly left open.
 *
 * The five Usage sliders set the scenario every figure here is computed at — the memory bar, the three
 * verdict tiles, the workload strip, the Envelope, and a Matrix heading that prints the very numbers
 * they hold ("32K context, 8K prompt, 1 user, FP16 KV"). They used to render after all five: 2,260px
 * below the memory bar at 1440x900 when the issue measured it, 2,402px on `main` by the time this
 * landed — two and a half viewport heights either way — and past four screens of grid on an iPhone 14,
 * where the slider and the bar it fills were never on screen together at any scroll position. #52 took the keyboard cost from 422 Tab presses to 15 and closed by naming this as the
 * open question — "whether the Usage controls should sit above the two large grids in DOM order. They
 * are the primary input of the tool and are currently last." #66 is that question answered.
 *
 * **DOM order is the half jsdom can answer, and it is not the lesser half.** Whether the slider and
 * the bar land in one viewport is geometry, so it is `e2e/usage-placement.spec.ts` — every rect here
 * reads 0. But DOM order *is* reading order: it is the sequence a screen-reader user is handed, one
 * panel at a time, and no amount of CSS `order` changes it. Six panels of output before the first
 * input was the same defect in the channel that has no viewport at all. So the sequence is pinned
 * here, where it costs a second, and where a panel slipped in between later fails a test.
 */
describe('the controls come before the figures they drive', () => {
  /** True when `a` is announced before `b` — document order, which is reading order. */
  const precedes = (a: Element, b: Element) =>
    Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

  /**
   * Every landmark the Bench renders, in the order a reader must meet them.
   *
   * Named rather than indexed, so the assertion survives a panel being added and fails on one being
   * moved — which is the direction that matters. The two inputs first, then the four figures they
   * drive, then the two grids as the terminal panels.
   */
  const ORDER: readonly (string | RegExp)[] = [
    // "Setup" since #74: the two control panels take their names from `sr-only` headings rather than
    // from `aria-label` strings, so they are in the document outline as well as in this list.
    'Setup',
    'Usage',
    /memory budget/i,
    'Verdicts',
    /what you could do with it/i,
    /how much room is left/i,
    /every model on every machine/i,
  ];

  it('announces the two control panels before any of the six they drive', () => {
    render(<App />);
    const landmarks = ORDER.map((name) => screen.getByRole('region', { name }));

    for (let i = 1; i < landmarks.length; i++) {
      expect(
        precedes(landmarks[i - 1], landmarks[i]),
        `${String(ORDER[i])} is announced before ${String(ORDER[i - 1])}`
      ).toBe(true);
    }
  });

  /**
   * The same claim in the channel a keyboard reader travels, and a stronger form of it: the controls
   * are not merely early, they are a *prefix* of the page's tab sequence. Nothing that reports a
   * figure is reachable before every control that sets one.
   *
   * Scoped to `<main>` so the masthead's share button — a real stop, and legitimately first — is not
   * counted as a figure sitting ahead of the sliders.
   */
  it('offers every control before any figure in the tab sequence', () => {
    const { container } = render(<App />);
    const main = container.querySelector<HTMLElement>('main')!;
    // By accessible name rather than by attribute: both panels are named by an `sr-only` heading
    // since #74, so there is no `aria-label` left to select on.
    const setup = screen.getByRole('region', { name: 'Setup' });
    const usage = screen.getByRole('region', { name: 'Usage' });

    const stops = [...main.querySelectorAll<HTMLElement>(TABBABLE)];
    const controls = stops.filter((el) => setup.contains(el) || usage.contains(el));

    // Both panels are actually in the sweep — twelve elements today: four selects, the hardware
    // note's disclosure, four sliders and three KV options. A zero here would satisfy the prefix
    // trivially.
    //
    // Twelve *elements*, ten stops: a radio group offers Tab only its checked member, so the three
    // KV options are one press in a browser and three in this enumeration. That does not weaken the
    // prefix — a stop the browser skips cannot reorder the ones it does not — but it is why this
    // figure and `e2e/matrix-grid.spec.ts`'s differ by two on identical markup.
    expect(controls.length).toBeGreaterThan(8);
    expect(stops.slice(0, controls.length), 'a figure is reachable before a control').toEqual(
      controls
    );
  });

  /**
   * And the anchor a Matrix click scrolls back to still aims at the figures.
   *
   * It is the one thing in this section that had to *stay* where it was. A Matrix click rewrites the
   * model and device, so the detail it loads is the budget bar and the tiles; moving the anchor up
   * with the controls would scroll two panels of input into view and push the figures the click
   * actually changed back under the fold. #66 named this as the thing to check when moving the panel.
   */
  it('leaves the detail anchor between the controls and the figures', () => {
    const { container } = render(<App />);
    const anchor = container.querySelector(`#${DETAIL_ANCHOR_ID}`)!;
    const usage = screen.getByRole('region', { name: 'Usage' });
    const budget = screen.getByRole('region', { name: /memory budget/i });

    expect(anchor, 'the anchor is missing, so a Matrix click scrolls nowhere').not.toBeNull();
    expect(precedes(usage, anchor), 'the anchor scrolls the controls into view').toBe(true);
    expect(precedes(anchor, budget), 'the anchor sits past the detail it is meant to show').toBe(
      true
    );
  });

  /**
   * The same rule one level down, where it was already being followed and had nothing holding it.
   *
   * Both picture panels carry a measure switch that recolours the whole picture, and both put it
   * *above* the picture — which is the #66 property inside a panel rather than across the page. A
   * review of this change audited the class and found these two clean and two disclosures (the
   * workload glossary, the budget table) sitting under their output, which is the "show more under a
   * list" convention and stays: they reveal detail already in the same viewport rather than setting a
   * scenario something else is computed from. The two that recolour a whole figure are the ones worth
   * pinning, because moving one is a plausible tidy-up and nothing would have failed.
   */
  it('puts each measure switch above the picture it recolours', () => {
    const { container } = render(<App />);

    const gridSwitch = screen.getByRole('group', { name: /colour the grid by/i });
    const table = container.querySelector('table[role="grid"]')!;
    expect(precedes(gridSwitch, table), 'the Matrix recolours a grid drawn above its switch').toBe(
      true
    );

    const fieldSwitch = screen.getByRole('group', { name: /colour the field by/i });
    const canvas = screen
      .getByRole('region', { name: /how much room is left/i })
      .querySelector('canvas')!;
    expect(canvas, 'the Envelope drew no canvas, so this proves nothing').not.toBeNull();
    expect(
      precedes(fieldSwitch, canvas),
      'the Envelope recolours a field drawn above its switch'
    ).toBe(true);
  });
});

/**
 * The other way through a long page, and the one the controls were missing from entirely (#74).
 *
 * Reading order is what the suite above pins. **Heading navigation is the mechanism a screen-reader
 * user actually uses to skip around a page this tall** — 3,043px at 1440 and 4,887px on a phone — and
 * the full outline was `h1 bench`, `h2 Memory budget`, three `h3` tiles, and four more `h2`s. Both
 * control panels carried an `aria-label` and no heading at all, so all nine inputs the tool takes were
 * unreachable that way: a reader could jump to five panels of output and to none of the controls that
 * produce them. Landmark navigation did reach them, which is a second, less-used mechanism and puts
 * nothing in the outline.
 *
 * The mis-parenting is the half that is an active wrong claim rather than an absence. With the verdict
 * strip unheaded, the nearest `h2` above `h3 Capacity` was the memory budget's, so the outline said
 * capacity is a subsection of the budget — of the one panel whose whole design is that capacity, decode
 * and time-to-first-token are three independent axes that must not be collapsed. See `Telemetry.tsx`'s
 * docstring, which calls collapsing them "precisely the move that makes existing calculators give bad
 * advice".
 *
 * **Driven off every panel and every control rather than off the three sections the issue tabulates**,
 * which is what stops a panel added later reintroducing the gap — and is how the sweep found two more
 * instances: the memory panel's runtime-refusal branch, whose `<section>` had a heading and no
 * `aria-labelledby` and so was not a landmark at all, and the MoE aside, an unnamed `complementary`
 * with a perfectly good `h2` sitting inside it.
 *
 * Which of these are DOM and which are pixels splits the usual way. Three of the eight headings are
 * `sr-only`, and whether they are really invisible and really take no grid track is layout — every rect
 * here reads 0 — so it is `e2e/heading-outline.spec.ts`. Everything below is attributes and document
 * order, which jsdom answers exactly.
 */
describe('the heading outline reaches every control and mis-parents nothing', () => {
  /** Every heading the page renders, in document order, with its level and its text. */
  const outline = (container: HTMLElement) =>
    [...container.querySelectorAll<HTMLElement>('h1, h2, h3, h4, h5, h6')].map((el) => ({
      el,
      level: Number(el.tagName[1]),
      text: (el.textContent ?? '').replace(/\s+/g, ' ').trim(),
    }));

  /**
   * Every panel on the page: the `<section>`s and the MoE `<aside>`.
   *
   * Both element types are landmarks here — `region` for a named section, `complementary` for the
   * aside — and both are what a heading is supposed to be the outline entry for. Deliberately not a
   * list of the seven known panels: the point of the sweep is the eighth.
   */
  const panels = (container: HTMLElement) => [
    ...container.querySelectorAll<HTMLElement>('section, aside'),
  ];

  /** Enough of a panel to find it from a failure message. */
  const label = (panel: HTMLElement) =>
    panel.getAttribute('aria-label') ??
    panel.querySelector('h1, h2, h3, h4, h5, h6')?.textContent?.trim() ??
    `${(panel.textContent ?? '').replace(/\s+/g, ' ').trim().slice(0, 40)}…`;

  /**
   * The heading a panel's accessible name is computed from, where it has one.
   *
   * `aria-labelledby` is an IDREF *list*, so the value is split rather than handed to
   * `getElementById` whole — the same reason the `aria-describedby` resolver further down splits.
   * A panel written as `aria-labelledby="headingId subheadId"`, which is the obvious way to append a
   * subhead to a landmark name, would otherwise resolve to `null` and be reported by the sweep below
   * as having no heading at all: a red pointing at correctly-named markup instead of at the resolver.
   */
  const namingHeading = (panel: HTMLElement) =>
    (panel.getAttribute('aria-labelledby') ?? '')
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => panel.ownerDocument.getElementById(id))
      .find(
        (target): target is HTMLElement => target !== null && /^H[1-6]$/.test(target.tagName)
      ) ?? null;

  /**
   * The three ways a panel can be outside the outline, listed rather than counted.
   *
   * A per-element matcher can only report the first failure, and naming the instances nobody thought
   * of is the whole job of a sweep — the same reason the `aria-describedby` sweep below writes its
   * own resolver instead of reaching for `toHaveAccessibleDescription`.
   */
  const outsideTheOutline = (container: HTMLElement) => ({
    /** Named by a string, so the name exists for landmark navigation and nowhere else. */
    byString: panels(container)
      .filter((panel) => panel.hasAttribute('aria-label'))
      .map(label),
    /**
     * Named by something that is not a level-2 heading — which covers both a panel with no heading
     * of its own and one nested a level down. Every panel here is a sibling of every other, and that
     * is the general form of the mis-parenting: the verdict tiles read as part of the memory budget
     * precisely because their panel had no `h2` between them and the budget's.
     */
    outOfLevel: panels(container)
      .filter((panel) => namingHeading(panel)?.tagName !== 'H2')
      .map((panel) => `${label(panel)} (${namingHeading(panel)?.tagName ?? 'no heading'})`),
    /**
     * Named by a heading inside an `aria-hidden` subtree. `aria-labelledby` resolves into one, so a
     * heading marked hidden goes on naming the landmark perfectly while disappearing from the
     * outline — the half of this fix with no other symptom.
     */
    hidden: panels(container)
      .filter((panel) => namingHeading(panel)?.closest('[aria-hidden="true"]') != null)
      .map(label),
  });

  it('names every panel with a heading rather than with an aria-label string', () => {
    const { container } = render(<App />);

    // Eight today: Setup, Usage, the memory budget, the verdicts, the workloads, the Envelope, the
    // Matrix and the MoE aside. A lower bound, because the sweep must not go green by matching
    // nothing — and an exact count would fail on the next panel added rather than on the property.
    expect(panels(container).length, 'the panel sweep matched nothing').toBeGreaterThanOrEqual(7);

    const { byString, outOfLevel, hidden } = outsideTheOutline(container);
    expect(byString, 'panels named by a string, so their name is in no outline').toEqual([]);
    expect(outOfLevel, 'panels not named by a level-2 heading').toEqual([]);
    expect(hidden, 'panels named by a heading a screen reader cannot navigate to').toEqual([]);
  });

  /**
   * And in the branch that draws no bar — the memory panel's own refusal, which is a different
   * `<section>` in the same component and had no `aria-labelledby` on it at all.
   *
   * It is the branch where the loss is worst: nothing is computed, so the refusal is the only thing
   * in the panel to read, and the panel was not a landmark to arrive at.
   */
  it('names the memory panel in the branch that refuses to draw a budget', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    // vLLM cannot drive a Mac, and the ceiling and overhead band are vLLM's own numbers.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'vllm');
    // The other branch really is on screen, or this sweeps the drawn one a second time.
    expect(screen.queryByRole('img', { name: /allocatable used/i })).not.toBeInTheDocument();
    expect(screen.getByText(/No budget to show/i)).toBeInTheDocument();

    const { byString, outOfLevel, hidden } = outsideTheOutline(container);
    expect(byString).toEqual([]);
    expect(outOfLevel).toEqual([]);
    expect(hidden).toEqual([]);

    // Under the same name as the branch that does draw one, since it is the same panel.
    expect(screen.getByRole('region', { name: /memory budget/i })).toHaveTextContent(
      /No budget to show/i
    );
  });

  /**
   * The mis-parenting, asserted the way a reader meets it: the `h2` they last passed on the way to
   * an `h3` is the section that `h3` belongs to.
   */
  it('parents each verdict tile under Verdicts rather than under the memory budget', () => {
    const { container } = render(<App />);
    const headings = outline(container);

    for (const tile of ['Capacity', 'Decode', 'Time to first token']) {
      const index = headings.findIndex((heading) => heading.text === tile);
      expect(index, `no heading reads “${tile}”`).toBeGreaterThan(-1);
      expect(headings[index].level, `“${tile}” is not a subsection of anything`).toBe(3);

      const parent = headings
        .slice(0, index)
        .reverse()
        .find((heading) => heading.level === 2);
      expect(parent?.text, `“${tile}” is announced as part of “${parent?.text}”`).toBe('Verdicts');
    }
  });

  /**
   * Every control, not the nine the issue counts — the two measure switches are `fieldset`s in the
   * picture panels and belong to the same class, and a tenth control added to either control panel
   * has to be covered without anyone remembering to add it here.
   *
   * The predicate is what heading navigation actually gives a reader: they land on a heading, and the
   * controls they then meet before the next one are the controls that heading covers. So a control is
   * reachable when the heading immediately before it in document order is inside its own panel. Before
   * the fix that heading was the masthead's `h1 bench` for all nine inputs, which is the failure
   * stated exactly — the outline's only entry above the memory budget was the page's title.
   */
  it('puts a heading in front of every control the tool takes', () => {
    const { container } = render(<App />);
    const headings = outline(container);
    const controls = [
      ...container.querySelectorAll<HTMLElement>(
        'main select, main input[type="range"], main fieldset'
      ),
    ];

    // Eleven today: the nine the issue counts — four selects, four sliders and the KV group — plus the
    // Envelope's and the Matrix's measure switches. A lower bound, so the sweep cannot go green by
    // matching nothing and does not go red on the next control added.
    expect(controls.length, 'the control sweep matched nothing').toBeGreaterThanOrEqual(9);

    /** `Prompt length`, enough to name the offender. */
    const named = (control: HTMLElement) =>
      (control as HTMLInputElement).labels?.[0]?.textContent?.trim() ??
      control.querySelector('legend')?.textContent?.trim() ??
      `<${control.tagName.toLowerCase()}>`;

    /** The heading a reader arrives from: the last one before this control in document order. */
    const arrivedFrom = (control: HTMLElement) =>
      [...headings]
        .reverse()
        .find((heading) =>
          Boolean(heading.el.compareDocumentPosition(control) & Node.DOCUMENT_POSITION_FOLLOWING)
        );

    const unreachable = controls
      .filter((control) => {
        const heading = arrivedFrom(control);
        return heading === undefined || !heading.el.closest('section, aside')?.contains(control);
      })
      .map(named);
    expect(unreachable, 'controls that no heading in the outline leads to').toEqual([]);
  });

  /**
   * No level skipped, which is the assertion in this suite that was **green before the fix** and is
   * here anyway — said plainly, because a test that cannot fail against the defect it is filed under
   * is worth less than the line it takes and this repo has shipped three of them.
   *
   * What it guards is the fix's own failure mode rather than the bug's. Three headings that nobody
   * can see are three headings whose level nothing on screen betrays: written as `h3`, "Setup" lands
   * under the masthead's `h1` with no `h2` between them, reads as a subsection of the page title, and
   * every other assertion in this file still passes. The outline is a tree or it is decoration.
   */
  it('skips no level, so no heading claims a parent that is not there', () => {
    const { container } = render(<App />);
    const headings = outline(container);

    expect(headings[0]?.level, 'the page does not open at h1').toBe(1);
    const skips = headings
      .filter((heading, i) => i > 0 && heading.level - headings[i - 1].level > 1)
      .map((heading) => `h${heading.level} “${heading.text}”`);
    expect(skips, 'headings that jump more than one level past their predecessor').toEqual([]);
  });
});

/**
 * The Matrix is 408 cells at the catalog #52 was measured against — 714 today — each a `<button>`
 * carrying a full-sentence `aria-label`, and it sat above the Usage controls in DOM order. Every one
 * of those cells in the tab sequence put 422 Tab presses between the top of the page and the context
 * slider that drives every figure on the page, and a screen-reader user heard 408 sentences on the
 * way. #66 has since moved those controls above the grid, which does not retire the pattern: the grid
 * is the page's last tab stop, so the 714 presses it used to cost are now the price of leaving the
 * document rather than of reaching the next panel. One press either way, and only with the roving
 * index.
 *
 * The counting lives here rather than in `e2e/` because the tab *sequence* is a DOM property —
 * `tabindex="-1"` is reachable by script and never by Tab — and jsdom can answer it in a second.
 * What jsdom cannot answer is whether pressing Tab actually lands where the sequence says, since
 * it implements no sequential focus navigation at all; that assertion is in `e2e/matrix-grid.spec.ts`.
 */
describe('the comparison grid is one tab stop, not four hundred', () => {
  // #52's defect is a property of the real grid: 408 cells was the measurement, and a roving
  // index over a dozen would satisfy every assertion below while the page went back to one tab
  // stop per cell on the shipped catalog. Whole-describe rather than per-test for the same
  // reason — Home/End and the five-row page step mean nothing on a grid three rows tall.
  beforeEach(atFullGrid);
  const grid = (container: HTMLElement) =>
    container.querySelector<HTMLTableElement>('table[role="grid"]')!;
  const cellsOf = (container: HTMLElement) => [
    ...grid(container).querySelectorAll<HTMLButtonElement>('td button'),
  ];

  it('offers exactly one of its cells to Tab', () => {
    const { container } = render(<App />);
    const cells = cellsOf(container);

    // The grid really is the size the issue describes, so a fix that emptied it would not pass.
    expect(cells.length).toBeGreaterThan(300);
    expect(cells.filter((c) => c.tabIndex === 0)).toHaveLength(1);
    expect(cells.filter((c) => c.tabIndex === -1)).toHaveLength(cells.length - 1);
  });

  /**
   * The whole page, not the walk to one panel — which is what this counted until #66.
   *
   * It measured the index of the first Usage control, because the Usage panel was the one *after* the
   * grid: 422 before the roving index, 19 after it. #66 moved those controls to the top of the page,
   * where their index is 6 whatever the grid does with its 714 cells, so the original assertion would
   * have passed against a grid that had never been fixed. The total is the property #52 actually
   * bought, and it is indifferent to where any panel sits.
   */
  it('keeps the whole page inside forty tab stops', () => {
    const { container } = render(<App />);
    const stops = [...container.querySelectorAll<HTMLElement>(TABBABLE)];

    // The grid has to be in this page, or the count is of a page without the problem on it.
    expect(cellsOf(container).length).toBeGreaterThan(300);
    /* 26 as it stands, one of which is the grid. 1,495 if every cell were in the sequence again —
       26 − 1 + 1,470 — which is what replacing the roving `tabIndex` with `tabIndex={0}` reports.
       The subtrahend is the *shipping* device count times the model count, which is what this grid
       renders; it read 714 until #77 doubled the model list, and a counterfactual quoting the wrong
       grid is a wrong expected value for whoever reinjects the defect.

       Forty, the same ceiling `e2e/matrix-grid.spec.ts` uses, because the two count one sequence and
       a bound that fires in one channel and not the other is a bug report about the wrong file. Loose
       on purpose: at 30 this sat four stops from red on a measured 26, and this sweep has been adding
       two to five stops a PR — so the next disclosure would have failed a test named after the grid
       while nothing about the grid had changed. What matters is the order of magnitude. */
    expect(stops.length).toBeLessThan(40);
  });

  it('moves between cells with the arrow keys', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);
    const columns = grid(container).querySelectorAll('tbody tr')[0].querySelectorAll('td').length;

    cells[0].focus();
    expect(document.activeElement).toBe(cells[0]);

    await user.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(cells[1]);

    await user.keyboard('{ArrowDown}');
    expect(document.activeElement).toBe(cells[columns + 1]);

    await user.keyboard('{ArrowLeft}');
    expect(document.activeElement).toBe(cells[columns]);

    await user.keyboard('{ArrowUp}');
    expect(document.activeElement).toBe(cells[0]);
  });

  it('carries the tab stop with the reader rather than resetting it', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{ArrowRight}{ArrowDown}');

    // Returning to the grid returns to where they were, which is the point of a roving index.
    const moved = document.activeElement as HTMLButtonElement;
    expect(moved.tabIndex).toBe(0);
    expect(cells.filter((c) => c.tabIndex === 0)).toEqual([moved]);
  });

  it('stops at the edges rather than wrapping into another row', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{ArrowLeft}{ArrowUp}');
    // A wrap here would move the reader to the far end of the grid for a keypress that should do
    // nothing at all — and the event must stay unhandled so the page can still scroll.
    expect(document.activeElement).toBe(cells[0]);
  });

  it('jumps to the ends of a row, and of the grid', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const rows = grid(container).querySelectorAll('tbody tr');
    const columns = rows[0].querySelectorAll('td').length;
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{End}');
    expect(document.activeElement).toBe(cells[columns - 1]);

    await user.keyboard('{Home}');
    expect(document.activeElement).toBe(cells[0]);

    await user.keyboard('{Control>}{End}{/Control}');
    expect(document.activeElement).toBe(cells[cells.length - 1]);

    await user.keyboard('{Control>}{Home}{/Control}');
    expect(document.activeElement).toBe(cells[0]);
  });

  it('still loads a cell into the Bench from the keyboard', async () => {
    // The navigation must not have cost the grid its actual purpose.
    const user = userEvent.setup();
    const { container } = render(<App />);
    const cells = cellsOf(container);

    cells[0].focus();
    await user.keyboard('{ArrowDown}{ArrowRight}');
    const target = document.activeElement as HTMLButtonElement;
    const label = target.getAttribute('aria-label')!;

    await user.keyboard('{Enter}');
    await waitFor(() => {
      const config = useConfig.getState();
      expect(label).toContain(getModel(config.modelId).name);
    });
  });

  it('tells a screen reader how to drive it', () => {
    const { container } = render(<App />);
    const caption = grid(container).querySelector('caption')!;

    expect(caption.textContent).toMatch(/single tab stop/i);
    expect(caption.textContent).toMatch(/arrow keys/i);
  });
});

/**
 * Which focus indicator each control declares.
 *
 * The four primary selects suppressed the outline and replaced it with a 1px border colour change
 * measuring **1.95:1 against the unfocused edge** — WCAG 2.2 SC 2.4.13 asks for 3:1 at a 2px
 * minimum thickness, and a colour-only change at that size is most of what a deuteranope loses
 * (#67). Two further instances came out of the sweep for the same shape, both listed in the issue as
 * already correct: the budget legend drew `focus:ring-1`, which is half the minimum thickness and is
 * the whole indicator once the outline is suppressed; and the Matrix marked its selected square with
 * the same channel, width and colour as its focus ring, so focusing the marked square changed
 * nothing at all — a 1:1 change contrast, and the square Tab lands on after a click.
 *
 * **The split across the two suites is deliberate, and it is the one #52 made.** Whether an
 * indicator paints 2px and clears 3:1 against what it sits on is a question about a real stylesheet
 * and a real focus ring: jsdom has no Tailwind cascade, no layout and no painted outline, so
 * `getComputedStyle` here says nothing about any of it. That half is `e2e/focus-indicators.spec.ts`.
 * What jsdom can answer is which indicator each control *declares*, which is a DOM property — and it
 * answers it for every focusable element on the page in a second, so a control added later is
 * covered by default rather than by someone remembering this file exists.
 */
describe('every control declares an indicator that says it has focus', () => {
  /** SC 2.4.13's minimum thickness, in px. */
  const MINIMUM_THICKNESS = 2;

  /** The channels an indicator can be drawn in. A border colour is deliberately not one — see below. */
  type Channel = 'outline' | 'ring' | 'inset-ring';

  /**
   * Split a utility into its variants and its base, bracket-aware.
   *
   * A naive split on `:` loses `[@media(pointer:coarse)]:h-11` — the variant contains a colon of its
   * own — and the Matrix cells carry exactly that, so the sweep would have quietly stopped reading
   * the class list of all 408 of them.
   */
  const parse = (utility: string) => {
    const parts: string[] = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < utility.length; i += 1) {
      const char = utility[i];
      if (char === '[' || char === '(') depth += 1;
      else if (char === ']' || char === ')') depth -= 1;
      else if (char === ':' && depth === 0) {
        parts.push(utility.slice(start, i));
        start = i + 1;
      }
    }
    return { variants: parts, base: utility.slice(start) };
  };

  /**
   * What one element declares.
   *
   * `restingChannels` is what is drawn while the element is *not* focused, which is what a focus
   * indicator has to be distinguishable from. A `border-*` swap is not counted in either direction:
   * treating it as an indicator is precisely the mistake #67 documents, and it stays on the select as
   * a redundant second cue.
   */
  const declared = (el: Element) => {
    const focusChannels = new Set<Channel>();
    const restingChannels = new Set<Channel>();
    const focusColours = new Set<string>();
    let thickness: number | null = null;
    let suppressesOutline = false;

    for (const utility of (el.getAttribute('class') ?? '').split(/\s+/).filter(Boolean)) {
      const { variants, base } = parse(utility);
      const match = /^(inset-ring|ring|outline)(?:-(.*))?$/.exec(base);
      if (!match) continue;

      const channel = match[1] as Channel;
      const rest = match[2] ?? '';
      const onFocus = variants.some((v) => v.startsWith('focus'));

      // `outline-none`/`outline-hidden` removes the indicator the browser supplies for free, which
      // is the state that puts an element under any obligation here at all.
      if (channel === 'outline' && (rest === 'none' || rest === 'hidden')) {
        suppressesOutline = true;
        continue;
      }
      // `ring-offset-*` and `outline-offset-*` place a mark; they never are one.
      if (rest.startsWith('offset-')) continue;

      // A bare `ring`/`outline` is 1px in Tailwind v4; anything else numeric is that many px.
      const width = rest === '' ? 1 : Number(rest);
      if (Number.isFinite(width)) {
        if (onFocus) thickness = Math.max(thickness ?? 0, width);
      } else if (onFocus) {
        focusColours.add(rest);
      }
      (onFocus ? focusChannels : restingChannels).add(channel);
    }

    return { focusChannels, restingChannels, focusColours, thickness, suppressesOutline };
  };

  /**
   * Everything with a focus state worth looking at: what Tab or a script can focus, plus anything
   * hosting a `focus-within:` indicator on behalf of a control inside it. The segmented controls are
   * the second kind — their radios are `sr-only`, so the mark belongs to the label around them, and a
   * sweep of focusable elements alone would have looked straight past it.
   */
  const controls = (container: HTMLElement) => [
    ...container.querySelectorAll<HTMLElement>(
      'a[href], button, input, select, textarea, [tabindex], [class*="focus-within:"]'
    ),
  ];

  /** `<button> "Qwen3 32B on ..."`, enough to find the offender from the failure message. */
  const name = (el: Element) =>
    `<${el.tagName.toLowerCase()}> "${
      (el.textContent ?? '').trim().slice(0, 40) ||
      el.getAttribute('aria-label')?.slice(0, 40) ||
      el.getAttribute('id') ||
      '(no text)'
    }"`;

  it('never declares an indicator thinner than the 2px minimum', () => {
    atFullGrid();
    const { container } = render(<App />);
    const summaries = controls(container).map((el) => ({ el, ...declared(el) }));
    const withIndicator = summaries.filter((s) => s.thickness !== null);

    // Vacuity guards. The grid alone declares one on every cell, and the page has to have several
    // outside it — a selector that stopped matching would otherwise report a clean sweep.
    expect(withIndicator.length, 'nothing on the page declares a focus indicator').toBeGreaterThan(
      300
    );
    expect(
      withIndicator.filter((s) => !s.el.closest('table')).length,
      'the sweep found no indicators outside the grid'
    ).toBeGreaterThan(5);

    const tooThin = withIndicator
      .filter((s) => s.thickness! < MINIMUM_THICKNESS)
      .map((s) => `${name(s.el)} declares ${s.thickness}px`);
    expect(tooThin, `thinner than ${MINIMUM_THICKNESS}px`).toEqual([]);
  });

  it('never takes the browser’s indicator away without replacing it', () => {
    const { container } = render(<App />);
    const summaries = controls(container).map((el) => ({ el, ...declared(el) }));
    const suppressing = summaries.filter((s) => s.suppressesOutline);

    expect(
      suppressing.length,
      'nothing suppresses the outline, so this asserts nothing'
    ).toBeGreaterThan(0);
    expect(
      suppressing.filter((s) => !s.el.closest('table')).length,
      'only grid cells suppress it, so the rule is narrower than it reads'
    ).toBeGreaterThan(0);

    const unreplaced = suppressing
      .filter((s) => (s.thickness ?? 0) < MINIMUM_THICKNESS)
      .map((s) => `${name(s.el)} suppresses the outline and declares ${s.thickness ?? 'nothing'}`);
    expect(unreplaced, 'outline removed with no compliant replacement').toEqual([]);
  });

  /**
   * The Matrix instance, stated as the general rule it is. An indicator that shares its channel with
   * a mark the element already wears cannot be a *change*, however thick it is — the selected square
   * wore an accent ring at rest and lit an identical accent ring on focus.
   */
  it('never draws focus in a channel a resting state already uses', () => {
    const { container } = render(<App />);
    const summaries = controls(container).map((el) => ({ el, ...declared(el) }));

    // At least one element has to wear a resting mark, or the rule is trivially satisfied. The
    // marked square is that element, and it is the one this rule was written for.
    expect(
      summaries.filter((s) => s.restingChannels.size > 0).length,
      'no element wears a resting mark, so this asserts nothing'
    ).toBeGreaterThan(0);

    const collisions = summaries
      .filter((s) => [...s.focusChannels].some((c) => s.restingChannels.has(c)))
      .map((s) => `${name(s.el)} draws focus and state both as ${[...s.focusChannels].join('/')}`);
    expect(collisions, 'focus and a resting state share a channel').toEqual([]);
  });

  /**
   * And the named instance, control by control.
   *
   * The channel is pinned as an outline rather than left free, because the reason for it is invisible
   * from Chromium: a ring is a `box-shadow`, and a native `menulist` select is painted by the
   * platform in WebKit, which does not reliably paint one. The browser spec would go on passing while
   * Safari showed nothing, so this is the assertion that holds that decision.
   */
  it('gives each of the four primary selects a 2px accent outline', () => {
    render(<App />);

    for (const label of ['Model', 'Hardware', 'Quantization', 'Runtime']) {
      const select = screen.getByLabelText(label);
      const summary = declared(select);

      expect(summary.suppressesOutline, `${label} suppresses its outline`).toBe(false);
      expect([...summary.focusChannels], `${label}'s indicator channel`).toEqual(['outline']);
      expect(summary.thickness, `${label}'s indicator thickness`).toBeGreaterThanOrEqual(
        MINIMUM_THICKNESS
      );
      expect([...summary.focusColours].join(), `${label}'s indicator colour`).toContain(
        '--color-accent'
      );
    }
  });
});

/**
 * A mark drawn *on* the heatmap, measured against the heatmap.
 *
 * The rule above — focus and a resting state never share a channel — moved the Matrix's
 * selected-square mark from an offset ring outside the cell to a frame inside it, and that moved it
 * off `--color-surface`, where `tokens.ts` validated the accent at 7.14:1, and onto the ramp, where
 * it was never validated at all. A single-tone accent frame measures **1.06:1 to 4.52:1** across
 * the seven steps of `sequential` — below the 3:1 non-text minimum on 304 of the grid's 408 squares,
 * the default selection among them. So the mark is two tones, and this is the arithmetic that says
 * so: for every fill the grid actually paints, at least one of the mark's tones has to clear 3:1
 * against it.
 *
 * **jsdom can answer this one, which is why it is here rather than in the browser suite.** The fill
 * is an inline style and the mark's tones are token names in the class list; the rest is the WCAG
 * contrast formula. What jsdom cannot answer — whether the two tones land in the geometry the class
 * list implies, 2px of accent with the separator inside it — is `e2e/focus-indicators.spec.ts`.
 */
describe('a mark drawn on the heatmap stays visible on every step of the ramp', () => {
  /** SC 1.4.11's floor for a non-text mark. */
  const MINIMUM_CONTRAST = 3;

  const parseColour = (value: string): [number, number, number] => {
    if (value.startsWith('#')) {
      const hex = value.slice(1);
      return [0, 2, 4].map((i) => Number.parseInt(hex.slice(i, i + 2), 16)) as [
        number,
        number,
        number,
      ];
    }
    const parts = value.match(/[\d.]+/g)?.map(Number) ?? [];
    return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
  };

  const luminance = (value: string) => {
    const [r, g, b] = parseColour(value)
      .map((c) => c / 255)
      .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const contrast = (a: string, b: string) => {
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high + 0.05) / (low + 0.05);
  };

  /** `--color-surface-raised` -> `surfaceRaised`, so a token name resolves against `colors`. */
  const token = (cssName: string) =>
    cssName.replace('--color-', '').replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());

  /**
   * The tones a resting mark is drawn in, read off the element that wears it.
   *
   * Only the utilities that paint inside the cell count: the inset ring and the inset shadow
   * separator beneath it. The `focus:`-gated ring is drawn *outside* the box, over the panel
   * surface, so it is measured in the browser suite against a different colour entirely.
   */
  const restingTones = (el: Element) =>
    (el.getAttribute('class') ?? '')
      .split(/\s+/)
      .filter((u) => !u.includes('focus') && /^(inset-ring|inset-shadow|shadow)-\[/.test(u))
      .flatMap((u) => [...u.matchAll(/--color-[a-z-]+/g)].map((m) => m[0]))
      .map(token)
      .filter((name): name is keyof typeof colors => name in colors)
      .map((name) => colors[name]);

  it('keeps one of the selected square’s tones 3:1 against every fill it can land on', () => {
    atFullGrid();
    render(<App />);
    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const cells = within(matrix)
      .getAllByRole('button')
      .filter((button) => button.closest('td'));

    const marked = cells.filter((cell) => cell.getAttribute('aria-current') === 'true');
    expect(marked, 'no square is marked, so there is no mark to measure').toHaveLength(1);
    const tones = restingTones(marked[0]);

    // Vacuity guards. A mark with no tones, or a grid the selector stopped matching, would
    // otherwise report a clean sweep over nothing.
    expect(
      tones.length,
      'the selected square declares no mark inside the cell — if selection moved back outside it, ' +
        'the channel-collision rule above is what has to hold instead of this one'
    ).toBeGreaterThan(0);
    expect(cells.length, 'the grid is not rendering').toBeGreaterThan(300);

    /** Every fill the grid paints, with how many squares wear it. `transparent` shows the panel. */
    const fills = new Map<string, number>();
    for (const cell of cells) {
      const background = (cell as HTMLElement).style.background;
      const behind = background === 'transparent' || !background ? colors.surface : background;
      fills.set(behind, (fills.get(behind) ?? 0) + 1);
    }
    expect(
      fills.size,
      'the grid paints one colour, so the ramp is not being exercised'
    ).toBeGreaterThan(4);

    const best = (fill: string) => Math.max(...tones.map((tone) => contrast(tone, fill)));
    const unreadable = [...fills.entries()]
      .filter(([fill]) => best(fill) < MINIMUM_CONTRAST)
      .map(([fill, count]) => `${fill} (${count} squares) at ${best(fill).toFixed(2)}:1`);
    expect(unreadable, `the mark below ${MINIMUM_CONTRAST}:1 on a fill the grid paints`).toEqual(
      []
    );

    /**
     * And that the second tone is load-bearing rather than belt-and-braces. If every step of the
     * ramp were readable under one tone, this test would pass on the single-tone frame that shipped
     * 304 unreadable squares — the ramp is what makes it fail, so the ramp has to still be there.
     */
    const defeatsOneTone = tones.filter(
      (tone) => ![...fills.keys()].every((fill) => contrast(tone, fill) >= MINIMUM_CONTRAST)
    );
    expect(
      defeatsOneTone.length,
      'every tone clears the bar alone, so this measures nothing the ramp can break'
    ).toBeGreaterThan(0);
  });
});

/**
 * What a control says about itself (#80).
 *
 * The five Usage controls drive every figure on the page, and the panel's entire text content at the
 * default scenario was the labels and the values: "Context per sequence 32K Concurrent users 1
 * Prompt length 8K KV precision FP16 Q8 Q4 Device count 1x". The argument those five make together —
 * context times users times bits per token is most of what the budget bar draws — was written in
 * `Envelope.tsx`'s docstring and nowhere a reader can see, and there was no mechanism to fix it per
 * call site: `StopSlider` and `Segmented` took no note at all, and `Select`'s `hint` was a dead
 * escape hatch no call site passed.
 *
 * **This is DOM, not layout, so it is here.** Whether a sentence is *reachable* — resolved through
 * `aria-describedby` rather than merely sitting nearby — is an attribute question jsdom answers
 * exactly. Whether five extra lines of prose change the panel's geometry is a browser question, and
 * `e2e/reflow.spec.ts` already sweeps this panel at 320px and at 200% text for a page that scrolls
 * sideways; the notes are wrapping paragraphs and add no min-content floor, so they need no new spec.
 */
describe('the controls that drive every figure explain what they are', () => {
  /**
   * The description a screen reader resolves for a control: the `aria-describedby` ids in order,
   * each one's text, joined.
   *
   * Written out rather than reaching for `toHaveAccessibleDescription`, because the sweep has to
   * *list* the controls that have none. A per-element matcher can only report the first failure, and
   * the point of a sweep is naming the instances nobody thought of.
   */
  const description = (el: Element) =>
    (el.getAttribute('aria-describedby') ?? '')
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => el.ownerDocument.getElementById(id)?.textContent?.trim() ?? '')
      .join(' ')
      .trim();

  const usage = () => screen.getByRole('region', { name: 'Usage' });

  /**
   * Every control in a panel: the sliders, the selects, and the `fieldset` a set of radios lives in
   * — deliberately not the radios themselves, whose description hangs off the group.
   */
  const controlsOf = (panel: HTMLElement) => [
    ...panel.querySelectorAll<HTMLElement>('input[type="range"], select, fieldset'),
  ];

  /** `Prompt length`, enough to find the offender from the failure message. */
  const named = (el: Element) =>
    (el as HTMLInputElement).labels?.[0]?.textContent?.trim() ??
    el.querySelector('legend')?.textContent?.trim() ??
    `<${el.tagName.toLowerCase()}>`;

  /** The element a setting's description hangs off — the group for the radios, the input otherwise. */
  const controlFor = (key: keyof typeof SETTING_NOTES) =>
    key === 'kvPrecision'
      ? within(usage()).getByRole('group', { name: SETTING_LABELS[key] })
      : within(usage()).getByLabelText(SETTING_LABELS[key]);

  it('gives every Usage control a description, not just a label and a value', () => {
    render(<App />);
    const controls = controlsOf(usage());

    // Vacuity guards, and lower bounds rather than exact counts: what has to hold is that the sweep
    // below ran over something — a selector that stopped matching, or every control moving out of the
    // panel, cannot report a clean sweep over nothing. An *exact* four went red on a sixth control
    // that was wired perfectly, before the property under test was evaluated at all, which is a
    // failure about the count and not about the thing this test is named after. A control leaving the
    // panel is caught by the per-setting test below, which looks each one up inside the region.
    expect(
      controls.filter((c) => c.tagName === 'INPUT').length,
      'the sliders the sweep ran over'
    ).toBeGreaterThanOrEqual(4);
    expect(
      controls.filter((c) => c.tagName === 'FIELDSET').length,
      'the KV group'
    ).toBeGreaterThanOrEqual(1);

    const silent = controls.filter((c) => description(c) === '').map(named);
    expect(silent, 'Usage controls with no accessible description').toEqual([]);
  });

  /**
   * And that each one is wired to its *own* sentence. Five near-identical call sites is where a
   * copy-paste puts the context's sentence under the prompt slider, which reads as plausibly as the
   * right answer and is worse than no note at all.
   */
  it('wires each control to its own sentence', () => {
    render(<App />);

    // Four of the five the issue names. If a setting loses its note the sweep above catches the
    // control; this catches a note that is present and attached to the wrong thing.
    expect(Object.keys(SETTING_NOTES)).toHaveLength(4);

    for (const key of Object.keys(SETTING_NOTES) as (keyof typeof SETTING_NOTES)[]) {
      expect(description(controlFor(key)), `${SETTING_LABELS[key]}’s description`).toBe(
        SETTING_NOTES[key]
      );
    }

    // The fifth is not in the table, because what an extra device buys depends on the runtime. Same
    // assertion, resolved through the same runtime the store is on.
    expect(
      description(within(usage()).getByLabelText(SETTING_LABELS.deviceCount)),
      'Device count’s description'
    ).toBe(
      deviceCountNote(
        getRuntime(DEFAULT_CONFIG.runtimeId),
        // Derived rather than `true`, so this tracks a change of default rather than asserting
        // against the wrong branch if the default pairing ever becomes an undrivable one.
        runtimeDrives(getRuntime(DEFAULT_CONFIG.runtimeId), getDevice(DEFAULT_CONFIG.deviceId))
      )
    );
  });

  it('describes the KV group once rather than once per radio', () => {
    render(<App />);
    const group = within(usage()).getByRole('group', { name: SETTING_LABELS.kvPrecision });
    expect(description(group)).toBe(SETTING_NOTES.kvPrecision);

    // A description on each radio is re-announced on every arrow key — three sentences to move
    // between three options — which is how a description earns a reputation for being noise.
    const radios = within(group).getAllByRole('radio');
    expect(radios.length, 'the group rendered no options').toBeGreaterThan(1);
    expect(
      radios.filter((r) => r.getAttribute('aria-describedby') !== null).length,
      'radios carrying their own copy of the group description'
    ).toBe(0);
  });

  /**
   * The device-count branch, both ways round.
   *
   * The default machine shards, so a test written only against the default would pass just as
   * happily if the sentence had been added to the `!shardable` paragraph instead — which is exactly
   * where the panel's one pre-existing explanation already lived, visible only when there is nothing
   * to configure. So the assertion is that the note is reachable *through the slider*, and that in
   * the branch with no slider it is not on the page at all.
   */
  it('describes the device-count slider in the branch that has one', async () => {
    const user = userEvent.setup();
    render(<App />);

    const note = deviceCountNote(
      getRuntime(DEFAULT_CONFIG.runtimeId),
      runtimeDrives(getRuntime(DEFAULT_CONFIG.runtimeId), getDevice('dgx-spark'))
    );
    await user.selectOptions(screen.getByLabelText('Hardware'), 'dgx-spark');
    const slider = screen.getByLabelText(SETTING_LABELS.deviceCount);
    expect(description(slider)).toBe(note);

    // A Mac has no transport between chassis: no control, so nothing to describe. The panel says
    // why the control is absent instead, which is a different sentence for a different reason.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    expect(screen.queryByLabelText(SETTING_LABELS.deviceCount)).not.toBeInTheDocument();
    expect(screen.getByText(/needs a transport between them/i)).toBeInTheDocument();
    expect(screen.queryByText(note)).not.toBeInTheDocument();
  });

  /**
   * The sentence under the slider and the arithmetic the slider drives are one claim (found in
   * review).
   *
   * The issue's suggested copy — "shard the model across, tensor-parallel. Adds memory and
   * bandwidth, minus what the interconnect costs" — is true of vLLM and of nothing else the app
   * offers, and the default runtime is one of the others. `achievedBandwidth` and the FLOPS closure
   * in `speed.ts` both return the per-device figure and short-circuit before `effectiveDeviceCount`
   * whenever `parallelism === 'layer'`, which is exactly the derivation `docs/ROADMAP.md` records as
   * wrong-first and silent when it breaks: a layer split buys capacity, not speed.
   *
   * So this asserts the copy against a measurement rather than against itself. The model has to
   * *fit on one device* for the comparison to isolate bandwidth: with a spill in play a layer split
   * really does get faster with more cards — 14.25 tok/s to 190.11 for this model at Q4_K_M on one
   * 4090 versus four — because the extra card stops it spilling. That is the capacity arriving as
   * speed, and it is why the sentence says "buys capacity" rather than "makes no difference".
   */
  it('does not promise the device-count slider a speed-up the runtime cannot deliver', () => {
    const device = getDevice('dgx-spark');
    expect(canShard(device), 'the slider does not render at all without a link').toBe(true);

    const measured = RUNTIMES.filter((r) => runtimeDrives(r, device)).map((runtime) => {
      const at = (deviceCount: number) =>
        estimateConfig({
          ...DEFAULT_CONFIG,
          deviceId: device.id,
          runtimeId: runtime.id,
          deviceCount,
        });
      const one = at(1);
      expect(
        one.placement.fits,
        `${runtime.id} spills on one ${device.name}, so a speed change would be capacity, not bandwidth`
      ).toBe(true);
      return {
        id: runtime.id,
        // `true` is not an assumption: the map above filters to `runtimeDrives`, and the
        // unsupported branch has its own test below.
        note: deviceCountNote(runtime, true),
        // 1% rather than exact equality: what is being distinguished is "held still to the last
        // decimal" from "half again as fast", and neither side needs a tighter threshold than that.
        aggregates: at(4).decode.perUserTokensPerSec > one.decode.perUserTokensPerSec * 1.01,
      };
    });

    // Both sides of the distinction are present, or the assertion below measures one branch twice.
    expect(measured.filter((m) => m.aggregates).map((m) => m.id)).toEqual(['vllm']);
    expect(measured.filter((m) => !m.aggregates).map((m) => m.id)).toEqual(['llama.cpp']);

    const lying = measured.filter(
      (m) => /bandwidth as well as memory/.test(m.note) !== m.aggregates
    );
    expect(
      lying.map((m) => `${m.id}: ${m.note}`),
      'runtimes whose device-count sentence disagrees with their own throughput'
    ).toEqual([]);
  });

  /**
   * And that the sentence follows the runtime on screen, which is the whole reason it is a function.
   */
  it('rewrites the device-count sentence when the runtime changes what a device buys', async () => {
    const user = userEvent.setup();
    render(<App />);

    const slider = () => screen.getByLabelText(SETTING_LABELS.deviceCount);
    expect(description(slider())).toMatch(/buys capacity, not speed/);

    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'vllm');
    expect(description(slider())).toMatch(/bandwidth as well as memory/);
  });

  /**
   * Both notes described something no evaluation reaches, in configurations two clicks from the
   * default. Codex found them on #80; each is a sentence that reads the wrong one of two inputs.
   *
   * **The device-count note read the hardware and not the runtime.** `canShard` is
   * `interconnect !== undefined`, so on a DGX Spark the slider renders under MLX — which cannot
   * drive that machine at all — and the note promised a layer split buying capacity directly below
   * the Runtime control's "Does not run on" warning.
   *
   * **The runtime note claimed every weight is dequantized.** BF16 is a real format here, and MLX
   * coerces to it, so there is nothing to dequantize in a configuration a reader reaches by picking
   * the one runtime this catalog exists to cover for Apple hardware.
   */
  it('does not describe a split for a runtime that cannot drive the machine', () => {
    const device = getDevice('dgx-spark');
    const mlx = getRuntime('mlx');
    expect(canShard(device), 'the slider renders, which is the whole problem').toBe(true);
    expect(runtimeDrives(mlx, device), 'this test needs an undrivable pairing').toBe(false);

    const note = deviceCountNote(mlx, runtimeDrives(mlx, device));
    expect(note).not.toMatch(/buys capacity, not speed/);
    expect(note).not.toMatch(/bandwidth as well as memory/);
    expect(note, 'the control still stores a value, so it has to say why nothing moves').toMatch(
      /does not run on this machine/
    );
  });

  it('says on screen that a device count buys nothing under an undrivable runtime', async () => {
    const user = userEvent.setup();
    render(<App />);

    const slider = () => screen.getByLabelText(SETTING_LABELS.deviceCount);
    expect(description(slider())).toMatch(/buys capacity, not speed/);

    // The default device is the DGX Spark, which MLX does not drive.
    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'mlx');
    expect(description(slider())).not.toMatch(/buys capacity|bandwidth as well as memory/);
    expect(description(slider())).toMatch(/does not run on this machine/);
  });

  it('does not tell a BF16 reader that every weight is dequantized', async () => {
    const user = userEvent.setup();
    render(<App />);
    const runtimeNote = () => description(screen.getByLabelText(SETTING_LABELS.runtimeId)) ?? '';

    /**
     * Driven through the DOM rather than read off the `<option>`s: `Controls.tsx` renders **only the
     * selected** option's note, so a sweep over `option.textContent` finds nothing and passes
     * whatever the copy says. That was the first version of this test.
     */
    const dequantizing = RUNTIMES.filter((r) => !r.nativeLowPrecision);
    expect(dequantizing.length, 'nothing dequantizes, so this test has no subject').toBeGreaterThan(
      0
    );

    // llama.cpp on the default machine, and MLX on hardware it actually drives — otherwise MLX's
    // note is the "Does not run on" warning and the sentence under test never renders.
    expect(runtimeNote()).toMatch(/quantized checkpoint/);
    expect(runtimeNote(), 'false for BF16, which is a format this app offers').not.toMatch(
      /every weight/
    );

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-96');
    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'mlx');
    expect(runtimeNote(), 'MLX coerces to BF16, so this is the easiest place to check it').toMatch(
      /quantized checkpoint/
    );
    expect(runtimeNote()).not.toMatch(/every weight/);
  });

  /**
   * The instance one panel up (found in review).
   *
   * `Select` renders only the selected option's note, and `runtimeOptions` produced one for exactly
   * two states — hardware the runtime cannot drive, and a runtime that preallocates. Both are false
   * for llama.cpp on any machine it drives, so at the default scenario the Runtime picker emitted no
   * `aria-describedby` at all: byte-for-byte the `aria-describedby: null` #80 tabulated for the
   * Usage sliders, in the panel #80 held up as the counterexample. Switching to vLLM produced a
   * description and switching back removed it, which is the appear-and-vanish behaviour that got
   * `Select`'s `hint` prop deleted rather than wired.
   *
   * Every option, not just the default one, because "the description exists at the scenario the test
   * happens to render" is the shape of the bug.
   */
  it('describes the Runtime picker at every choice, not only when a caveat applies', async () => {
    const user = userEvent.setup();
    render(<App />);

    const select = screen.getByLabelText(SETTING_LABELS.runtimeId) as HTMLSelectElement;
    const options = within(select).getAllByRole('option') as HTMLOptionElement[];
    // Vacuity guard: the picker offers every runtime, so a loop over nothing is a green test.
    expect(options.length, 'runtimes the picker offered').toBe(RUNTIMES.length);

    const silent: string[] = [];
    for (const option of options) {
      await user.selectOptions(select, option.value);
      if (description(select) === '') silent.push(option.value);
    }
    expect(silent, 'runtimes that leave the picker with no accessible description').toEqual([]);
  });

  /**
   * The instance the issue did not name, and the reason the sweep is worth running: the Matrix's
   * measure switch already had its sentence on screen and never attached it to anything, so a
   * screen-reader user entering the group heard "Colour the grid by, Does it fit, pressed" and
   * nothing about what the colour means.
   */
  it('describes the grid’s measure switch, which had the sentence but never attached it', async () => {
    const user = userEvent.setup();
    render(<App />);

    const group = screen.getByRole('group', { name: /colour the grid by/i });
    expect(description(group)).toMatch(/headroom left/i);

    /**
     * It tracks the selection, which is what makes it this group's description rather than a static
     * caption: each measure means something different by a bright cell.
     *
     * Scoped to the group since #65 gave the Envelope a measure control of its own, reading the same
     * `MEASURES`. A page-wide `getByRole('button', { name: 'How fast' })` then finds two and throws —
     * and the failure is a real one about the query rather than about either control, because "the
     * grid" in this test's name is the Matrix and the Envelope's switch answers for a different
     * picture. Both surfaces having the toggle is the point of sharing the vocabulary.
     */
    await user.click(within(group).getByRole('button', { name: 'How fast' }));
    expect(description(group)).toMatch(/tokens per second/i);
  });

  /**
   * The Hardware picker, whose note was doing two jobs and neither of them well (#68).
   *
   * `[statusWarning, ceilingClause, row.note].join(' ')` fused a derived claim onto 40-180 words of
   * catalog provenance with a bare space, and handed the whole thing to the control as its
   * `aria-describedby`. Two consequences, and the DOM is where both are visible:
   *
   *   - **The punctuation.** "raiseable to 240 GiB The allocation ceiling reserves 16 GiB for
   *     macOS" reads as a parse error on the most prominent control on the page, and on the M5
   *     Ultra the sentence that ran on was the warning that its specs are rumour-grade.
   *   - **The audience.** A screen-reader user heard the entire derivation — `iogpu.wired_limit_mb`,
   *     unwired allocations, what the sysctl parses — every time focus landed on the picker, before
   *     they could choose anything.
   *
   * `src/data/catalog.test.ts` sweeps the composition across all 43 rows. These assert the wiring:
   * that the short claim is what the control is described by, that the provenance is still reachable,
   * and that it is reachable somewhere other than the description.
   */
  describe('the Hardware picker', () => {
    const hardware = () => screen.getByLabelText(SETTING_LABELS.deviceId);
    const toggle = () => screen.getByRole('button', { name: /the full hardware note/i });

    /** The disclosure's region, found through the button that controls it. */
    const detail = () => {
      const id = toggle().getAttribute('aria-controls');
      return id === null ? null : document.getElementById(id);
    };

    /**
     * Every row where a derived clause used to be followed immediately by a curated note — the nine
     * seams, derived from the catalog rather than listed, so a row added later joins the sweep.
     * The issue named seven of them and one of those (`ryzen-ai-max-395`, already at its own
     * ceiling) composes a single fragment and never had a seam at all.
     */
    const seams = DEVICES.filter(
      (d) =>
        d.note !== undefined &&
        (d.status !== 'shipping' ||
          (d.allocatableTunable === true && maxAllocatablePerDevice(d) > d.allocatableBytes))
    );

    it('describes the rumoured Mac with closed sentences instead of one fused figure', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.selectOptions(hardware(), 'mac-studio-m5-ultra-512');

      // The only three-fragment row, and the one the issue calls out: the rumour warning was fused
      // to a capacity figure, which is the sentence on that row that most needs to stand alone.
      const note = description(hardware());
      expect(note).toMatch(/^Rumoured — specs may change\. /);
      expect(note).toMatch(/384 GiB allocatable by default, raiseable to 480 GiB\.$/);

      // Fourteen words, against 146 before — and none of the derivation.
      expect(note.split(/\s+/)).toHaveLength(14);
      expect(note).not.toMatch(/iogpu|window server|sysctl|per-core rate|rumour-grade/i);
    });

    it.each(seams.map((d) => [d.id, d] as const))(
      'gives %s a description that is a claim, not a derivation',
      async (_id, device) => {
        const user = userEvent.setup();
        render(<App />);
        await user.selectOptions(hardware(), device.id);

        const note = description(hardware());
        // Something is still said about every one of these rows — this is not a deletion.
        expect(note).not.toBe('');
        // The claim ends as a sentence, so nothing that follows it can look like part of it.
        expect(note).toMatch(/[.!?…]$/);
        // And the curated prose is not in it. First 40 characters rather than the whole string,
        // because a substring is what a bare join produces.
        expect(note).not.toContain(device.note!.slice(0, 40));
      }
    );

    it('keeps the curated note reachable, and out of the description while open', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.selectOptions(hardware(), 'mac-studio-m3-ultra-512');

      // Collapsed: the provenance is hidden, not unmounted — `hidden` is display: none, so it
      // still sets no height on a grid cell whose row also holds the Quantization and Runtime
      // pickers, while the toggle's `aria-controls` keeps resolving to a real node (#131).
      expect(screen.getByText(/what the sysctl parses/i)).not.toBeVisible();

      await user.click(toggle());
      expect(screen.getByText(/what the sysctl parses/i)).toBeVisible();
      // Open, and still not part of the control's accessible description. A disclosure that got
      // wired into `aria-describedby` when expanded would be the same defect with a click in front
      // of it.
      expect(description(hardware())).not.toMatch(/sysctl/i);
    });

    it('renders the catalog’s prose rather than printing its markup', async () => {
      const user = userEvent.setup();
      render(<App />);
      // Five rows write `**strong**`, two write `*emphasis*` and nine write backticked identifiers;
      // nothing rendered any of them, so the picker printed literal asterisks. Moving the prose to
      // its own region without this would have moved the glitch with it.
      await user.selectOptions(hardware(), 'mac-studio-m3-ultra-96');
      await user.click(toggle());

      const region = detail();
      expect(region).not.toBeNull();
      expect(region!.querySelector('strong')?.textContent).toMatch(/60-core GPU/);
      expect(region!.querySelector('code')?.textContent).toBe('iogpu.wired_limit_mb');
      // Verbatim apart from the marks: the note is provenance, and losing a clause of it in a
      // renderer would be worse than printing the asterisks.
      expect(region!.textContent).toBe(
        getDevice('mac-studio-m3-ultra-96').note!.replace(/\*\*|\*|`/g, '')
      );

      // The single-asterisk register, which the first version of this renderer did not read: two
      // rows write their contrast with one mark rather than two, and both printed the asterisks in
      // the region this change created for them.
      await user.selectOptions(hardware(), 'rx-9070-xt');
      expect(detail()!.querySelector('em')?.textContent).toBe('matrix');
    });

    /**
     * And over the whole catalog, because "five rows write `**strong**`" is a fact about the file on
     * the day it was read.
     *
     * The property is that nothing of the markup reaches the reader as text: the region's text is
     * the note with its marks removed, exactly, which fails both ways — an unrendered mark shows up
     * as a stray asterisk, and a renderer that ate a clause shows up as missing prose. A fourth
     * mark, or a stray `*` in a figure, fails here rather than printing itself at a reader.
     *
     * `Select` on its own rather than the whole Bench, which is the one place in this file that
     * mounts a component instead of the app: the wiring from the catalog through `devicePickerNote`
     * to this control is what the tests above assert, on the real picker. What is swept here is the
     * renderer against every note in the file, and mounting the Matrix's 408 cells 29 times to read
     * one paragraph cost 19 seconds of a suite that runs in two minutes. One mount and one click
     * either way, since the disclosure deliberately stays open across a change of selection.
     */
    it('leaves none of the markup in any note the catalog carries', async () => {
      const user = userEvent.setup();
      const noted = DEVICES.filter((d) => d.note !== undefined);
      expect(noted.length, 'no row carries a note, so this sweep proves nothing').toBeGreaterThan(
        20
      );

      const picker = (value: string) => (
        <Select
          label={SETTING_LABELS.deviceId}
          value={value}
          onChange={() => {}}
          options={noted.map((d) => ({ value: d.id, label: d.name, detail: d.note }))}
        />
      );

      const { rerender } = render(picker(noted[0].id));
      await user.click(toggle());

      const wrong: string[] = [];
      for (const device of noted) {
        rerender(picker(device.id));
        if (detail()?.textContent !== device.note!.replace(/\*\*|\*|`/g, '')) wrong.push(device.id);
      }
      expect(wrong, 'notes whose markup reached the reader as text').toEqual([]);
    });

    /**
     * The other 34 rows, where the split leaves the control with no accessible description at all.
     *
     * That is what #68 asks for — the derivation was never a description of the control — but a
     * description that is deliberately absent and one that vanished by accident are the same DOM,
     * and nothing else in the suite looks at this panel's descriptions. So the sanctioned state is
     * pinned: no description, and the prose one click away in the disclosure. This row is the one
     * whose note was dropped from the picker entirely once before.
     */
    it('describes the control only where it has derived a claim', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.selectOptions(hardware(), 'rtx-3090');

      const device = getDevice('rtx-3090');
      expect(device.status).toBe('shipping');
      expect(device.allocatableTunable).toBeUndefined();
      expect(device.note).toBeDefined();

      expect(hardware()).not.toHaveAttribute('aria-describedby');
      expect(description(hardware())).toBe('');

      await user.click(toggle());
      expect(detail()!.textContent).toMatch(/NVLink/);
    });

    it('offers no disclosure for a row the catalog says nothing extra about', async () => {
      const user = userEvent.setup();
      render(<App />);

      // The 5090 carries no curated note, so there is nothing to disclose — and an empty
      // disclosure is a control that promises something and does nothing.
      await user.selectOptions(hardware(), 'rtx-5090');
      expect(getDevice('rtx-5090').note).toBeUndefined();
      expect(
        screen.queryByRole('button', { name: /the full hardware note/i })
      ).not.toBeInTheDocument();
    });
  });
});

/**
 * What a picker says *before* the choice, which is not the same string as what it says after (#69).
 *
 * `Controls.tsx` renders every option's label and only the **selected** option's note. So the
 * caveats that decide whether a row is worth choosing lived in the one string a `<select>` will not
 * show until the choice has been made: "Mac Studio M5 Ultra (512 GB) — 512 GiB" scrolled past as an
 * equal of the 512 GB M3 Ultra one line above it, which is real hardware with measured bandwidth,
 * and its `Rumoured — specs may change.` appeared only afterwards. CLAUDE.md states that one as a
 * requirement: pre-release specs must stay visibly labelled in the UI.
 *
 * **Swept over both pickers that share the component**, because the mechanism is the component's and
 * not the catalog's. The Runtime picker had the same shape and a harder consequence — on a Mac Studio
 * it offered llama.cpp, vLLM and MLX as three equals and produced "Does not run on …" only once vLLM
 * had been selected and every figure on the page had been replaced by a refusal.
 *
 * These read `option.textContent`, which is the one place a sweep like this is not vacuous: the file
 * already records that reading notes off the `<option>`s finds nothing and passes whatever the copy
 * says. Here the text under test really is the option's own.
 */
describe('a picker states its caveats where the choice is made', () => {
  /** Every option's own text, which is all a closed `<select>` has to distinguish its rows by. */
  const optionsOf = (label: string) =>
    Array.from((screen.getByLabelText(label) as HTMLSelectElement).options).map((o) => ({
      value: o.value,
      text: (o.textContent ?? '').trim(),
    }));

  /** The marker as a reader sees it, not as the code spells it — `devices.json` says `rumored`. */
  const PRE_RELEASE = /\s·\s(rumoured|announced)$/;

  it('marks every row whose specs are not final, in the option text', () => {
    render(<App />);

    const options = optionsOf(SETTING_LABELS.deviceId);
    expect(options.length, 'the picker offered no hardware at all').toBe(DEVICES.length);

    // From `status`, so a row added to the catalog as announced or rumoured fails this rather than
    // slipping through it. The named instance is one device; the class is the field.
    const preRelease = new Set(DEVICES.filter((d) => d.status !== 'shipping').map((d) => d.id));
    expect(
      preRelease.size,
      'no catalogued row is rumoured or announced, so this sweep proves nothing'
    ).toBeGreaterThan(0);

    expect(
      options
        .filter((o) => preRelease.has(o.value) && !PRE_RELEASE.test(o.text))
        .map((o) => o.text),
      'pre-release hardware offered as though it were shipping'
    ).toEqual([]);
    // The other half of it: a marker on every row would satisfy the assertion above and mean nothing.
    expect(
      options
        .filter((o) => !preRelease.has(o.value) && PRE_RELEASE.test(o.text))
        .map((o) => o.text),
      'shipping hardware carrying a pre-release marker'
    ).toEqual([]);
  });

  it('marks the rumoured Mac without spending the figures the row is chosen on', () => {
    render(<App />);

    // The string the issue quotes, plus the marker it was missing. Pinned whole because the defect
    // was not "the marker is absent" but "the label is indistinguishable from a shipping row": a
    // marker that displaced the name or the capacity would satisfy a regex and lose the comparison.
    expect(
      optionsOf(SETTING_LABELS.deviceId).find((o) => o.value === 'mac-studio-m5-ultra-512')?.text
    ).toBe('Mac Studio M5 Ultra (512 GB) — 512 GiB · rumoured');
  });

  it('still gives the chosen row the fuller sentence, rather than moving it into the label', async () => {
    const user = userEvent.setup();
    render(<App />);

    // The marker is a tag on a row being scanned; this is the clause for the row that was picked, and
    // it is the control's accessible description. Adding the first must not cost the second.
    await user.selectOptions(
      screen.getByLabelText(SETTING_LABELS.deviceId),
      'mac-studio-m5-ultra-512'
    );
    expect(screen.getByLabelText(SETTING_LABELS.deviceId)).toHaveAccessibleDescription(
      /^Rumoured — specs may change\./
    );
  });

  it('marks a runtime that cannot drive the machine currently selected', async () => {
    const user = userEvent.setup();
    render(<App />);

    /**
     * Two machines, because the marked runtime differs between them: vLLM does not run on Apple
     * unified memory and MLX runs on nothing else. A marker hard-coded to either would pass on one
     * device and fail on the other, which is why the expectation is derived from `runtimeDrives`.
     */
    for (const deviceId of ['mac-studio-m3-ultra-256', 'rtx-5090']) {
      await user.selectOptions(screen.getByLabelText(SETTING_LABELS.deviceId), deviceId);
      const device = getDevice(deviceId);

      const options = optionsOf(SETTING_LABELS.runtimeId);
      expect(options.length, 'the picker offered no runtimes').toBe(RUNTIMES.length);
      expect(
        RUNTIMES.filter((r) => !runtimeDrives(r, device)).length,
        `every runtime drives ${deviceId}, so it marks nothing`
      ).toBeGreaterThan(0);

      const wrong = options.filter(
        (o) =>
          runtimeDrives(getRuntime(o.value), device) ===
          /does not run on this hardware/.test(o.text)
      );
      expect(
        wrong.map((o) => `${deviceId}: “${o.text}”`),
        'runtime options whose marker disagrees with whether the runtime drives this machine'
      ).toEqual([]);
    }

    // And the note still names the machine, which the marker deliberately does not — it is what a
    // screen-reader user hears on the control once the choice has been made.
    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'mlx');
    expect(screen.getByLabelText(SETTING_LABELS.runtimeId)).toHaveAccessibleDescription(
      /Does not run on GeForce RTX 5090/
    );
  });

  it('says what a runtime will not run on, rather than leaving the reader to supply it', async () => {
    const user = userEvent.setup();
    render(<App />);

    /**
     * Pinned whole, because the first version of this marker said "does not run here" and an option's
     * own text is *all* that is announced for a row nobody has selected — so "here" resolved to
     * nothing, and the string that names the machine is the selected option's note, which is exactly
     * the dependency the marker exists to remove (found in review). The referent has to be inside the
     * option: "this hardware" is the control one row up.
     */
    await user.selectOptions(
      screen.getByLabelText(SETTING_LABELS.deviceId),
      'mac-studio-m3-ultra-256'
    );
    expect(optionsOf(SETTING_LABELS.runtimeId).find((o) => o.value === 'vllm')?.text).toBe(
      'vLLM · does not run on this hardware'
    );
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

/**
 * The list order, on the two surfaces that render it (#79).
 *
 * `devices.json`'s row order *is* the display order — `catalog.ts` maps the file straight through and
 * neither surface sorts — and until this landed nothing anywhere said so, enforced it, or showed it.
 * The picker was a flat list of 43 options, so scrolling from `rtx-3090` to `rtx-pro-6000-blackwell`
 * to `h100-sxm` crossed two segment boundaries in silence; the Matrix ran the three classes together
 * as one 42-column strip and its caption explained the arrow keys while naming neither axis.
 *
 * `catalog.test.ts` owns the file's own structure — class runs, vendor runs, and the prose that states
 * them. What is asserted here is the other half of the issue: that the structure reaches a reader.
 * Every claim below is about rendered markup and fails against the flat version.
 */
describe('the catalog shows the order it is listed in', () => {
  const hardware = () => screen.getByLabelText(SETTING_LABELS.deviceId) as HTMLSelectElement;

  /**
   * The bands a given set of rows has, in file order, paired with the heading each expects.
   *
   * Parameterised because the two surfaces render different sets: the picker offers the whole catalog,
   * the Matrix only the shipping rows. They happen to produce the same three bands today, and
   * "happen to" is what this file keeps recording as the thing that stops being true.
   */
  const expectedBands = (rows: readonly (typeof DEVICES)[number][]) => {
    const bands: { label: string; ids: string[] }[] = [];
    for (const device of rows) {
      const label = DEVICE_CLASS_LABELS[device.class];
      const last = bands.at(-1);
      if (last && last.label === label) last.ids.push(device.id);
      else bands.push({ label, ids: [device.id] });
    }
    return bands;
  };

  it('gives the Hardware picker a heading per class band, over the rows the file already grouped', () => {
    render(<App />);

    const groups = [...hardware().querySelectorAll('optgroup')].map((group) => ({
      label: group.label,
      ids: [...group.querySelectorAll('option')].map((option) => option.value),
    }));

    // The premise, so a picker that grew one group and lost the rest cannot pass the comparison below
    // for the wrong reason.
    expect(groups.length, 'the picker renders no optgroups at all').toBe(3);
    // Whole thing at once — headings *and* membership *and* sequence. Grouping by filtering the
    // catalog three times would satisfy a check on the headings while quietly owning the order.
    expect(groups).toEqual(expectedBands(DEVICES));
  });

  /**
   * That the grouping cannot reorder the list, demonstrated on a list where reordering would show.
   *
   * Asserting the catalog's own order against the rendered options proves nothing here: `DEVICES` is
   * already class-grouped in the declared band order, so a `Select` that built its groups by
   * filtering the list three times would emit the same sequence and pass. This is the mechanism
   * instead — `optionRuns` splits on a *change* of group, so an interleaved list renders as two runs
   * with one heading rather than being tidied into one, and the sequence the call site passed survives
   * untouched. That distinction is why the Hardware picker can be grouped at all without taking
   * ownership of `devices.json`'s order.
   */
  it('groups a Select by adjacency, so no call site loses the order it passed', () => {
    const options = [
      { value: 'a', label: 'A', group: 'First' },
      { value: 'b', label: 'B', group: 'Second' },
      { value: 'c', label: 'C', group: 'First' },
      { value: 'd', label: 'D' },
    ];
    render(<Select label="Interleaved" value="a" onChange={() => {}} options={options} />);

    const select = screen.getByLabelText('Interleaved') as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(['a', 'b', 'c', 'd']);
    expect(
      [...select.querySelectorAll('optgroup')].map((g) => ({
        label: g.label,
        ids: [...g.querySelectorAll('option')].map((o) => o.value),
      }))
    ).toEqual([
      { label: 'First', ids: ['a'] },
      { label: 'Second', ids: ['b'] },
      { label: 'First', ids: ['c'] },
    ]);
    // And an option with no group is rendered outside every heading rather than swept into the last
    // one — the two forms compose, which is what lets the other three pickers stay ungrouped.
    expect(select.querySelector('option[value="d"]')!.closest('optgroup')).toBeNull();
  });

  it('marks every column that opens a class band on the Matrix, and only those', () => {
    atFullGrid();
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const shipping = DEVICES.filter((d) => d.status === 'shipping');
    // Adjacency, like the component: the first column opens the first band and needs no separator,
    // since there is nothing to its left to be separated from.
    const expected = shipping.filter((d, i) => i > 0 && d.class !== shipping[i - 1].class);
    expect(expected.length, 'the shipping catalog spans one class, so this proves nothing').toBe(2);

    /**
     * Every column's heading, paired with whether it carries the band gap.
     *
     * By `data-band-start` rather than by the utility class that draws the gap. The first version of
     * this read `classList.contains()` on the border utility, and that border is a
     * `calc(var(--spacing) * 2)` now — the same length in the unit the columns are measured in — so a
     * class-name assertion would have gone quietly false while the markup got *more* correct. The
     * attribute is what the component promises; the border is how it currently looks.
     */
    const separated = [...matrix.querySelectorAll('thead th')]
      .slice(1)
      .map((th, i) => ({ id: shipping[i].id, gap: th.hasAttribute('data-band-start') }));
    expect(separated.filter((c) => c.gap).map((c) => c.id)).toEqual(expected.map((d) => d.id));

    // And down the grid, not only across the header — the gap is a full-height channel or it is a
    // decoration on the labels. One body row is enough: the class is a property of the column.
    const firstRow = matrix.querySelectorAll('tbody tr')[0];
    const cells = [...firstRow.querySelectorAll('td')].map((td, i) => ({
      id: shipping[i].id,
      gap: td.hasAttribute('data-band-start'),
    }));
    expect(cells.filter((c) => c.gap).map((c) => c.id)).toEqual(expected.map((d) => d.id));
  });

  /**
   * The gap, keyed where a sighted reader can find it.
   *
   * The band gap shipped named only inside the `sr-only` caption: a screen-reader user was told the
   * columns are grouped, and a sighted reader met two channels of whitespace with nothing on the page
   * saying what divided them — while the legend beside it keys every other mark on the surface. That is
   * #73's asymmetry, on the same surface and in the same direction, and the caption assertion below
   * passes happily with it live, which is why this is a separate case.
   */
  it('keys the band gap on the page, not only in the caption', () => {
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const key = within(matrix).getByText(/a gap between columns/i);
    // Outside the caption, which is the whole claim: `sr-only` text would satisfy a text query and
    // leave the sighted channel exactly as unkeyed as it was.
    expect(key.closest('caption')).toBeNull();
    // And it answers the question the gap raises rather than only labelling it — the bands, in order,
    // in the words the picker's headings use.
    expect(key).toHaveTextContent(
      expectedBands(DEVICES.filter((d) => d.status === 'shipping'))
        .map((band) => band.label)
        .join(', ')
    );
  });

  it('names both of the Matrix’s axes in its caption, which its headings cannot', () => {
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const caption = matrix.querySelector('caption')!.textContent ?? '';

    // The bands, in order and by the same names the picker's headings use — a reader who met
    // "Discrete GPUs" in the Hardware control should hear the same words here. Off the *shipping*
    // rows, which is what this grid renders.
    expect(caption).toContain(
      expectedBands(DEVICES.filter((d) => d.status === 'shipping'))
        .map((band) => band.label)
        .join(', ')
    );
    // And the row axis, which is the one fact the Matrix cannot state anywhere else: its row headings
    // are name-only, so 35 rows appeared in an order with no stated basis.
    expect(caption).toMatch(/rows run most-downloaded first/i);
  });
});
