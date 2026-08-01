import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useDevicePixelRatio } from './useDevicePixelRatio';

/**
 * The mechanism, which is all jsdom can see (#129): there is no `dprchange` event, so the hook
 * builds a `matchMedia` resolution query around the *current* ratio and rebuilds it on every
 * fire — each query is one-shot by construction. Whether a real browser actually fires that
 * query when a window crosses displays, and whether the bitmap then re-sizes, is
 * `e2e/canvases.spec.ts`'s half.
 */
describe('useDevicePixelRatio', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  /** A matchMedia stub that records each query and hands back its listener. */
  function stubMatchMedia() {
    const queries: string[] = [];
    let listener: (() => void) | null = null;
    vi.stubGlobal('matchMedia', (query: string) => {
      queries.push(query);
      return {
        matches: true,
        media: query,
        addEventListener: (_: string, fn: () => void) => {
          listener = fn;
        },
        removeEventListener: () => {
          listener = null;
        },
      };
    });
    return { queries, fire: () => act(() => listener?.()) };
  }

  it('reports the ratio the display changed to, through a re-armed query', () => {
    vi.stubGlobal('devicePixelRatio', 2);
    const media = stubMatchMedia();

    const { result } = renderHook(() => useDevicePixelRatio());
    expect(result.current).toBe(2);
    expect(media.queries).toEqual(['(resolution: 2dppx)']);

    // The window lands on a 1x display: the old query stops matching and fires.
    vi.stubGlobal('devicePixelRatio', 1);
    media.fire();

    expect(result.current).toBe(1);
    // Re-armed around the new ratio — the old query can never fire again, so a hook that does
    // not rebuild it reports the first transition and goes deaf.
    expect(media.queries).toEqual(['(resolution: 2dppx)', '(resolution: 1dppx)']);
  });

  it('reconciles a ratio that changed between first render and subscription', () => {
    const media = stubMatchMedia();
    // A getter whose first read (the state initializer, during render) sees the old display and
    // every later read (the effect, arming and reconciling) sees the new one — the gap a window
    // moved during a delayed initial render falls into. The armed query already matches the new
    // ratio, so no event ever corrects state; only the explicit reconcile can.
    let reads = 0;
    Object.defineProperty(window, 'devicePixelRatio', {
      configurable: true,
      get: () => (++reads === 1 ? 1 : 2),
    });
    try {
      const { result } = renderHook(() => useDevicePixelRatio());
      expect(result.current).toBe(2);
      expect(media.queries).toEqual(['(resolution: 2dppx)']);
    } finally {
      Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 1 });
    }
  });

  it('survives jsdom, where matchMedia does not exist', () => {
    vi.stubGlobal('devicePixelRatio', 1.5);
    // No matchMedia stub: the guard leaves the initial reading standing.
    const { result } = renderHook(() => useDevicePixelRatio());
    expect(result.current).toBe(1.5);
  });
});
