import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG } from '@/store/config';
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
    const table = screen.getByRole('table');
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
    expect(screen.getByText(/crossing the host bus/i)).toBeInTheDocument();
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
    // The displayed value and the stored one must agree.
    expect(screen.getByText(tokens(max))).toBeInTheDocument();
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
