import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG } from '@/store/config';
import { configToShareSearch } from '@/store/url';
import { getModel } from '@/data/catalog';
import { tokens } from '@/lib/format';

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
