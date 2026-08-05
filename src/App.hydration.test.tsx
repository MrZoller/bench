import { StrictMode } from 'react';
import { act } from '@testing-library/react';
import { hydrateRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';

/**
 * That React actually *keeps* the prerendered markup.
 *
 * This is the silent failure of the whole mechanism (#178). A hydration mismatch makes React
 * throw the server's HTML away and re-render the tree from scratch: the page looks perfect in a
 * browser, every test that queries the DOM still passes, and the site is empty again to precisely
 * the crawlers and fetch tools prerendering exists for. Nothing about the built output shows it —
 * the HTML is there, it is simply discarded a few milliseconds after it arrives.
 *
 * React 19 routes every mismatch through `onRecoverableError`, so promoting that to a failure is
 * the assertion. The errors are also collected rather than only thrown, because a guard that
 * depends on an exception escaping React's internals is one refactor away from passing vacuously.
 *
 * **This is one of a pair and does not stand alone.** It passes on a page that was an empty shell
 * to begin with — what it checks is that the client agrees with the server, not that the server
 * produced anything. `entry-server.test.tsx` and `src/prerender/page.test.ts` are the other half,
 * asserting real per-device figures in the markup, and `e2e/prerendered.spec.ts` is the third:
 * a browser with JavaScript disabled, which is the one thing jsdom structurally cannot answer.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

import { boundGridByDefault } from '@/test/grid';
import { getDevice, getModel } from '@/data/catalog';
import { DEFAULT_CONFIG, useConfig } from '@/store/config';
import { configToShareSearch, locationToConfig, shouldHydrate } from '@/store/url';
import { prerenderRoutes, renderRoute } from './entry-server';

boundGridByDefault();

const mounted: Root[] = [];

/**
 * One route per shape of tree, rather than all 199.
 *
 * **This swept every route while there were four, and the arithmetic stopped working at 199.** A
 * hydrate of the whole app costs ~65ms here and the CI runner is roughly 5x this machine, so the
 * full sweep is ~70s against a 30s per-test limit — and `vite.config.ts` argues at length that the
 * limit is not the thing to raise, because "a test approaching this is a test rendering the full
 * grid for a claim that does not need it". This is that case.
 *
 * **What it does not need is the product.** A mismatch is the client and the server disagreeing,
 * and both compute the tree from the same coerced `Config`; the one place they can diverge is the
 * path parse, and `routes.test.ts` round-trips all 199 of those in milliseconds. What is left for
 * this file is that each *shape* of tree hydrates — and shape is decided by the handful of fields
 * that switch a branch: the tier (which fields the route names at all), the device class (the
 * runtime filter, the quant fallback, the shard control), the device status (the pre-release
 * label), whether the ceiling is tunable (a second note and a different capacity verdict), and the
 * model's attention kind and MoE-ness (the quant applicability rules and the parameter breakdown).
 *
 * Derived rather than listed, so a catalog that grows a new combination gets covered without
 * anybody remembering to add it — which is the property a hand-picked sample would not have.
 */
function routeShapes(): readonly ReturnType<typeof prerenderRoutes>[number][] {
  const seen = new Set<string>();
  return prerenderRoutes().filter((route) => {
    const device = route.config.deviceId ? getDevice(route.config.deviceId) : undefined;
    const model = route.config.modelId ? getModel(route.config.modelId) : undefined;
    const key = [
      route.tier,
      device?.class,
      device?.status,
      device?.allocatableTunable ?? false,
      model?.attention.core.kind,
      model?.experts !== undefined,
    ].join('|');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

afterEach(() => {
  act(() => {
    for (const root of mounted.splice(0)) root.unmount();
  });
  document.body.innerHTML = '';
});

/**
 * Hydrate the built markup for a scenario, exactly as `main.tsx` does, and return whatever React
 * reported as recoverable while doing it.
 */
async function hydrate(config: Parameters<typeof renderRoute>[0]): Promise<unknown[]> {
  const html = renderRoute(config);
  const container = document.createElement('div');
  container.innerHTML = html;
  document.body.appendChild(container);

  const recoverable: unknown[] = [];
  await act(async () => {
    mounted.push(
      hydrateRoot(
        container,
        <StrictMode>
          <App />
        </StrictMode>,
        {
          onRecoverableError: (error) => {
            recoverable.push(error);
            throw error;
          },
        }
      )
    );
  });
  return recoverable;
}

describe('hydration', () => {
  it('keeps the markup prerendered for the default scenario', async () => {
    expect(await hydrate({})).toEqual([]);
  });

  it('keeps the markup prerendered for a device route', async () => {
    // A different class from the default, so the branches that differ by class — the runtime
    // filter, the quant fallback, the shard control — are all exercised across the hydrate.
    expect(await hydrate({ deviceId: 'epyc-9654' })).toEqual([]);
  });

  it('keeps the markup prerendered for every shape of route the build writes', async () => {
    for (const route of routeShapes()) {
      expect(await hydrate(route.config), `/${route.segments.join('/')}/`).toEqual([]);
      act(() => {
        for (const root of mounted.splice(0)) root.unmount();
      });
      document.body.innerHTML = '';
    }
  });

  /**
   * The negative control, because a guard that has never been seen to fail is a guard nobody
   * knows the polarity of.
   *
   * It forges the one mismatch that matters here — a client whose scenario disagrees with the
   * markup it was handed, which is exactly what a `readInitialConfig` that ignored the path would
   * produce on every device page — and asserts React notices. Reaching into the store's initial
   * state is how that is staged rather than something the app does; it is the same object
   * `entry-server.tsx` writes, for the same documented reason.
   */
  it('notices when the client disagrees with the markup', async () => {
    const html = renderRoute({ deviceId: 'rtx-5090' });
    const container = document.createElement('div');
    container.innerHTML = html;
    document.body.appendChild(container);
    Object.assign(useConfig.getInitialState(), { deviceId: 'epyc-9654' });

    const recoverable: unknown[] = [];
    await act(async () => {
      mounted.push(
        hydrateRoot(
          container,
          <StrictMode>
            <App />
          </StrictMode>,
          { onRecoverableError: (error) => void recoverable.push(error) }
        )
      );
    });

    expect(recoverable.length).toBeGreaterThan(0);
  });

  /**
   * The same mismatch as the control above, arrived at rather than staged — and it is the arrival
   * the site actually gets.
   *
   * `configToShareSearch` writes the root path with all nine fields, so every link the share
   * button and the calibration issue hand out lands on prerendered `/` — a DGX Spark page —
   * carrying a query that may name any device at all. The attribute `main.tsx` used to branch on
   * says markup is present and nothing about which scenario it holds, so React hydrated, mismatched
   * and discarded the whole tree: the visitor watched a fully-painted wrong-device page swap.
   *
   * This is the case `shouldHydrate` exists for, asserted from the URL end so the guard cannot be
   * deleted later on the grounds that nothing demonstrates the failure.
   */
  it('notices when a shared link addresses a scenario the markup was not rendered for', async () => {
    const arrival = locationToConfig(
      '/',
      configToShareSearch({ ...DEFAULT_CONFIG, deviceId: 'rtx-5090' }),
      '/'
    );
    expect(shouldHydrate('/', configToShareSearch(arrival), '/', true)).toBe(false);

    const html = renderRoute({});
    const container = document.createElement('div');
    container.innerHTML = html;
    document.body.appendChild(container);
    Object.assign(useConfig.getInitialState(), arrival);

    const recoverable: unknown[] = [];
    await act(async () => {
      mounted.push(
        hydrateRoot(
          container,
          <StrictMode>
            <App />
          </StrictMode>,
          { onRecoverableError: (error) => void recoverable.push(error) }
        )
      );
    });

    expect(recoverable.length).toBeGreaterThan(0);
  });

  it('leaves the prerendered figures in the DOM afterwards', async () => {
    // The mismatch this guards against does not remove content, it replaces it — so the check
    // that the page still says something is worth having beside the one that says React did not
    // complain.
    await hydrate({ deviceId: 'rtx-5090' });
    expect(document.body.textContent).toMatch(/tok\/s/);
  });
});
