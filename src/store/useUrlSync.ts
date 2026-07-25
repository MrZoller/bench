import { useEffect, useRef } from 'react';
import { useConfig, type Config } from './config';
import { configToSearch, sameScenario, searchToConfig } from './url';

/**
 * Keeps the address bar and the store in step, in both directions.
 *
 * Deliberately `replaceState` rather than `pushState`: a slider emits a value per frame while
 * being dragged, and pushing each one would bury the previous page under hundreds of history
 * entries. The trade is that the back button leaves the page rather than stepping through
 * scenarios — which is the right default for a tool people arrive at by link.
 *
 * The reverse direction still matters despite that, because `popstate` also fires when someone
 * edits the URL or follows a second link into the same page.
 */

/**
 * Minimum gap between history writes, in milliseconds.
 *
 * Safari throttles `replaceState` to roughly 100 calls per 30 seconds and throws `SecurityError`
 * past that — and a dragged slider emits a value per frame, so about three seconds of dragging
 * spends the entire budget. The comment above already said sliders emit hundreds of updates;
 * it just did not draw the conclusion.
 *
 * 250ms stays well inside the limit while still feeling immediate: the URL lands within a
 * quarter second of letting go, and nobody copies a link mid-drag.
 */
const MIN_WRITE_INTERVAL_MS = 250;

export function useUrlSync(): void {
  const config = useConfig();
  const replace = useConfig((s) => s.replace);

  const lastWriteAt = useRef(0);
  const pending = useRef<number | undefined>(undefined);

  // Store -> URL.
  useEffect(() => {
    /**
     * Reads current state rather than closing over this render's `config`, so a coalesced burst
     * writes the newest scenario instead of whichever frame happened to schedule the timer.
     */
    const write = () => {
      const search = configToSearch(useConfig.getState() as Config);
      const next = `${window.location.pathname}${search}`;
      if (next === `${window.location.pathname}${window.location.search}`) return;
      try {
        window.history.replaceState(null, '', next);
        lastWriteAt.current = Date.now();
      } catch {
        // A refused history write must not take the app down with it. The URL is a convenience;
        // what is on screen is the scenario. The next write carries whatever this one missed,
        // because it reads state fresh rather than replaying a queued value.
      }
    };

    const since = Date.now() - lastWriteAt.current;
    if (since >= MIN_WRITE_INTERVAL_MS) {
      write();
      return;
    }

    // Trailing edge, so the URL still ends up matching the screen even though the frames in
    // between are dropped.
    window.clearTimeout(pending.current);
    pending.current = window.setTimeout(write, MIN_WRITE_INTERVAL_MS - since);
    return () => window.clearTimeout(pending.current);
  }, [config]);

  // URL -> store.
  useEffect(() => {
    const onPop = () => {
      const fromUrl = searchToConfig(window.location.search);
      if (!sameScenario(fromUrl, useConfig.getState() as Config)) replace(fromUrl);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [replace]);
}
