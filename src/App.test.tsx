import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG } from '@/store/config';
import { configToShareSearch } from '@/store/url';
import { getModel } from '@/data/catalog';
import { tokens } from '@/lib/format';
import { DETAIL_ANCHOR_ID } from '@/components/Matrix';
import { judgeWorkloads } from '@/engine/verdict';
import { kvSubstitutionFor } from '@/data/runtimes';

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
    expect(screen.getByText(/needs a transport between devices/i)).toBeInTheDocument();
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

    // The refusal names the card it belongs to rather than implying the one above it.
    expect(screen.getByText(/busiest card by cache needs/i)).toBeInTheDocument();
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
  });

  it('does not promise a ceiling the platform will not raise', async () => {
    const user = userEvent.setup();
    render(<App />);

    // The Ryzen's 96 GiB is Variable Graphics Memory's *maximum*, not a default — it is already
    // at its ceiling, so telling the user to raise it is advice they cannot take.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'ryzen-ai-max-395');
    expect(screen.queryByText(/raiseable/i)).not.toBeInTheDocument();
    // The curated bandwidth note is a separate claim and must still be there.
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
});

describe('clicking a Matrix cell loads what that cell was scored under', () => {
  it('carries the quantization the cell was evaluated at, not the one selected', async () => {
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
   * The all-blocked grid, which is the state gating the legend on `runs` hid it in.
   *
   * At the longest context and the most users, every Apple cell under MLX fails placement, so a
   * scan for a *running* substituted cell finds nothing — while the grid goes on publishing a
   * verdict for every cell and, on some of them, "past the default allocation, which this machine
   * lets you raise". Every one of those rests on Q4_K_M's 4.85 bpw standing in for MLX's ~4.5,
   * and since the stand-in is the heavier of the two, a borderline "past the default" is the
   * verdict most likely to flip. The mark is least dispensable exactly where it was dropped.
   */
  it('marks the Matrix when every cell was scored at a stand-in and none of them fit', async () => {
    const user = userEvent.setup();
    render(<App />);

    const matrix = () => screen.getByRole('region', { name: /every model on every machine/i });
    const legend = () => within(matrix()).queryByText(/stand-in format .* cannot load/i);

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    await user.selectOptions(screen.getByLabelText('Runtime'), 'mlx');
    await user.selectOptions(screen.getByLabelText('Quantization'), 'q4_k_m');

    const context = screen.getByLabelText('Context per sequence') as HTMLInputElement;
    fireEvent.change(context, { target: { value: String(Number(context.max)) } });
    const users = screen.getByLabelText('Concurrent users') as HTMLInputElement;
    fireEvent.change(users, { target: { value: String(Number(users.max)) } });

    // Nothing on the grid runs — the precondition, asserted rather than assumed, since a catalog
    // change that leaves one cell running would make the rest of this test vacuous.
    expect(
      within(matrix()).getByText(
        (_, el) =>
          el?.tagName === 'CAPTION' && /\b0 of \d+ combinations run/.test(el.textContent ?? '')
      )
    ).toBeInTheDocument();

    expect(legend()).toBeInTheDocument();
  });
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
 * The Matrix is 408 cells, each a `<button>` carrying a full-sentence `aria-label`, and it sits
 * above the Usage controls in DOM order. Every one of those cells in the tab sequence put 422 Tab
 * presses between the top of the page and the context slider that drives every figure on the page,
 * and a screen-reader user heard 408 sentences on the way.
 *
 * The counting lives here rather than in `e2e/` because the tab *sequence* is a DOM property —
 * `tabindex="-1"` is reachable by script and never by Tab — and jsdom can answer it in a second.
 * What jsdom cannot answer is whether pressing Tab actually lands where the sequence says, since
 * it implements no sequential focus navigation at all; that assertion is in `e2e/matrix-grid.spec.ts`.
 */
describe('the comparison grid is one tab stop, not four hundred', () => {
  /** Tabbable, not merely focusable — the distinction the whole fix turns on. */
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

  it('leaves the Usage controls a short walk from the top of the page', () => {
    const { container } = render(<App />);
    const stops = [...container.querySelectorAll<HTMLElement>(TABBABLE)];
    const usage = container.querySelector<HTMLElement>('section[aria-label="Usage"]')!;
    const firstControl = usage.querySelector<HTMLElement>(TABBABLE)!;

    // 422 before this. The bound is deliberately loose — what matters is the order of magnitude,
    // and pinning the exact figure would fail on any unrelated control being added.
    expect(stops.indexOf(firstControl)).toBeLessThan(30);
    expect(stops.indexOf(firstControl)).toBeGreaterThan(0);
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
