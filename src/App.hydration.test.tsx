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
 * produced anything. `entry-server.test.tsx` is the other half, asserting real per-device figures
 * in the markup; Phase 3 adds a browser check with JavaScript disabled, which is the one thing
 * jsdom structurally cannot answer.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

import { boundGridByDefault } from '@/test/grid';
import { useConfig } from '@/store/config';
import { prerenderRoutes, renderRoute } from './entry-server';

boundGridByDefault();

const mounted: Root[] = [];

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

  it('keeps the markup prerendered for every route the build writes', async () => {
    for (const route of prerenderRoutes()) {
      expect(await hydrate(route.config)).toEqual([]);
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

  it('leaves the prerendered figures in the DOM afterwards', async () => {
    // The mismatch this guards against does not remove content, it replaces it — so the check
    // that the page still says something is worth having beside the one that says React did not
    // complain.
    await hydrate({ deviceId: 'rtx-5090' });
    expect(document.body.textContent).toMatch(/tok\/s/);
  });
});
