import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { DEFAULT_CONFIG, useConfig } from '@/store/config';

/**
 * The all-blocked grid, on a catalog chosen rather than hoped for.
 *
 * This test lived in `App.test.tsx` and drove the state through the controls: MLX on a Mac at the
 * longest context and the most users, where every Apple cell fails placement — so a scan for a
 * *running* substituted cell finds nothing while the grid goes on publishing a verdict for every
 * cell, several of them recommending the user raise an allocation ceiling. That is the distinction
 * `substitutedCells` exists to make (`evaluated`, not `runs`, closing #32), and its precondition was
 * a property of the model catalog: that nothing in it could fit.
 *
 * #77 ended that. `unsloth/gemma-3-4b-it` keeps a 1024-token window on 29 of its 34 layers, so at
 * 131K over 128 users it needs ~360 GB of cache against the 512 GiB Mac Studio's 412 — it fits, and
 * one running cell out of 1,470 is enough to make the old assertion false *and* the old test
 * vacuous, since a `runs`-gated legend would render from that one cell too. There is no setting that
 * blocks it: context, concurrency and KV precision are all at their heaviest stop already.
 *
 * So the scenario is built here instead of found. Mocking the catalog down to one 671B model is not
 * a smaller version of the old test — it is the same claim with its precondition pinned, and it
 * cannot be quietly falsified by the next model somebody seeds.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  // One row, and a real one: whatever else moves in the catalog, a 671B MoE at 131K over 128 users
  // does not fit on any Apple machine that exists.
  const only = actual.MODELS.filter((m) => m.id === 'deepseek-ai/DeepSeek-V3.1');
  return { ...actual, MODELS: only };
});

const { Matrix } = await import('./Matrix');

afterEach(() => {
  cleanup();
  useConfig.setState(DEFAULT_CONFIG);
});

it('marks the Matrix when every cell was scored at a stand-in and none of them fit', () => {
  const config = {
    ...DEFAULT_CONFIG,
    modelId: 'deepseek-ai/DeepSeek-V3.1',
    deviceId: 'mac-studio-m3-ultra-512',
    runtimeId: 'mlx',
    quantId: 'q4_k_m',
    contextTokens: 131072,
    concurrency: 128,
  };
  useConfig.setState(config);
  render(<Matrix config={config} />);

  const matrix = screen.getByRole('region', { name: /every model on every machine/i });

  // The precondition, asserted rather than assumed — the whole point of the mock above is that this
  // line stays true as the catalog grows.
  expect(
    within(matrix).getByText(
      (_, el) =>
        el?.tagName === 'CAPTION' && /\b0 of \d+ combinations run/.test(el.textContent ?? '')
    )
  ).toBeInTheDocument();

  // Q4_K_M's 4.85 bpw standing in for MLX's ~4.5 is what every one of those verdicts was computed
  // from, including the "past the default allocation, which this machine lets you raise" ones — and
  // since the stand-in is the heavier of the two, a borderline verdict is the one most likely to
  // flip. The mark is least dispensable exactly where gating it on `runs` dropped it.
  expect(within(matrix).getByText(/stand-in format .* cannot load/i)).toBeInTheDocument();
});
