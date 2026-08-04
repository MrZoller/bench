import { describe, expect, it, vi } from 'vitest';
import { getDevice } from '@/data/catalog';

/**
 * The guard for the prerender's one silent failure mode: pages that all say the same thing.
 *
 * A prerendered page that contains only layout chrome is a failure, and so is one containing the
 * *wrong* figures — and the second is much harder to see, because every page is the right size,
 * the right shape, and individually plausible. This asserts the thing that distinguishes them:
 * two routes, two device names, two different sets of numbers, in one process and in that order.
 *
 * It is a Vitest test rather than an e2e one because nothing here needs a browser — it is
 * `renderToString` over the same module the build script loads, and it runs in a second.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

import { boundGridByDefault } from '@/test/grid';
import { prerenderRoutes, renderRoute } from './entry-server';

boundGridByDefault();

/** Whatever the page states as the memory budget, which is a per-device figure. */
function budget(html: string): string | undefined {
  return /aria-label="[0-9.]+ [GTM]iB of ([0-9.]+ [GTM]iB) allocatable/.exec(html)?.[1];
}

describe('renderRoute', () => {
  it('renders the scenario it is given rather than the store it was imported with', () => {
    const device = getDevice('rtx-5090');
    const html = renderRoute({ deviceId: device.id });

    // The budget, not the name: every page lists every device name in its picker and its grid, so
    // a name assertion passes on a page rendered for something else entirely.
    expect(budget(html)).toBe('31 GiB');
    expect(budget(html)).not.toBe(budget(renderRoute({})));
  });

  /**
   * The regression this file exists for.
   *
   * Zustand hands React `getInitialState()` as the server snapshot, and that closure is fixed at
   * import time — so setting the scenario and rendering produced the *default* device's figures on
   * every page while `getState()` reported the right one. Three files of nearly identical size,
   * every one of them a DGX Spark. Asserting on the figures rather than on the length is the only
   * check that sees it.
   */
  it('gives two devices two different sets of figures', () => {
    const first = renderRoute({ deviceId: 'rtx-5090' });
    const second = renderRoute({ deviceId: 'epyc-9654' });

    expect(budget(first)).not.toBe(budget(second));
    expect(budget(first)).toBe('31 GiB');
    expect(budget(second)).toBe('720 GiB');
  });

  /**
   * The other half, and the reason the Phase 2 slice is more than one route: the store is a
   * module-level singleton, so a route that does not replace the scenario renders whatever the
   * previous one left behind. Rendering the same route twice with a different one in between is
   * what makes a leak visible.
   */
  it('does not carry one route into the next', () => {
    const before = renderRoute({ deviceId: 'rtx-5090' });
    renderRoute({ deviceId: 'dgx-spark' });
    const after = renderRoute({ deviceId: 'rtx-5090' });

    expect(after).toBe(before);
  });

  it('renders real engine output, not layout chrome', () => {
    const html = renderRoute({ deviceId: 'dgx-spark' });
    // A fit verdict, a memory figure, a prefill number and a decode number — the four things #178
    // asks a prerendered page to carry.
    expect(html).toMatch(/Fits|Tight|Spills|Will not fit/);
    expect(html).toMatch(/[0-9.]+ GiB of [0-9.]+ GiB allocatable/);
    expect(html).toMatch(/[0-9,]+ tok\/s prompt processing/);
    expect(html).toMatch(/tok\/s per user/);
  });

  it('renders every route the build will write', () => {
    for (const route of prerenderRoutes()) {
      const html = renderRoute(route.config);
      expect(html).toMatch(/[0-9.]+ GiB of [0-9.]+ GiB allocatable/);
    }
  });
});
