import { useSyncExternalStore } from 'react';

/**
 * The display scale a canvas must draw at, as a subscription rather than a read (#129).
 *
 * Every canvas here sizes its bitmap `CSS box × devicePixelRatio` so cell edges stay crisp on a
 * retina display — but a repaint triggered only by a `ResizeObserver` never fires for a DPR-only
 * change: dragging the window from a 2× display to a 1× one, or an OS display-scale change,
 * alters no CSS box. The bitmap then stays at the old resolution against the unchanged box —
 * blurry in one direction, wastefully supersampled in the other — until something else forces a
 * redraw.
 *
 * There is no `dprchange` event; the standard listener is a `matchMedia` resolution query that
 * matches exactly the *current* ratio, re-armed on every fire because each query is one-shot by
 * construction — the moment it stops matching, a new query has to be built around the new ratio.
 *
 * `useSyncExternalStore` rather than state in an effect, and not only for the lint rule: its
 * post-subscribe snapshot check is precisely the mount-gap reconciliation this needs (raised in
 * review on #150). A ratio that changes between first render and subscription leaves the armed
 * query already matching — it will never fire — and React re-reads the snapshot after
 * subscribing and re-renders on the difference itself.
 *
 * jsdom has no `matchMedia`; the guard subscribes to nothing and the snapshot stands.
 *
 * **The browser half of this is deliberately untested, and the reason is measured, not
 * assumed.** Playwright can only change `deviceScaleFactor` after load through CDP's
 * `Emulation.setDeviceMetricsOverride`, and that override mutates query state without
 * dispatching events: after it, `devicePixelRatio` reads 2 and `(resolution: 2dppx)` matches,
 * but a `change` listener armed before it — on this query, on `min-resolution`, even on window
 * `resize` — never fires. A transition spec written against it fails against correct code, and
 * a spec that dispatches the event by hand only re-tests what the unit test already covers. The
 * platform behaviour this leans on — the resolution query firing when a window crosses displays
 * — is the documented standard pattern for exactly this. `e2e/canvases.spec.ts` still proves
 * the draw uses the ratio in force whenever it runs, at 1x and, via the `retina` project, at 2x.
 */
function subscribe(onStoreChange: () => void): () => void {
  if (typeof window.matchMedia !== 'function') return () => {};

  let media: MediaQueryList | null = null;
  let detach = () => {};
  const arm = () => {
    detach();
    media = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    const onChange = () => {
      onStoreChange();
      arm();
    };
    media.addEventListener('change', onChange);
    detach = () => media?.removeEventListener('change', onChange);
  };
  arm();
  return () => detach();
}

function getSnapshot(): number {
  return window.devicePixelRatio || 1;
}

export function useDevicePixelRatio(): number {
  return useSyncExternalStore(subscribe, getSnapshot);
}
