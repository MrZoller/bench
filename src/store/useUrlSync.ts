import { useEffect, useRef } from 'react';
import { useConfig, type Config } from './config';
import {
  configToSearch,
  configToShareSearch,
  locationToConfig,
  pathBaseline,
  sameScenario,
} from './url';

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
 * past that. 250ms was the first attempt and is not enough: it permits 120 writes in the same
 * window, so a sustained drag still trips the limit after about 25 seconds — a smaller bug than
 * no throttle at all, and the same bug.
 *
 * 400ms allows at most 75 in any 30-second window, which leaves real headroom under a limit that
 * is approximate and browser-specific. The cost is that the address bar lags a continuous drag by
 * up to four tenths of a second, which nobody is reading mid-drag.
 */
const MIN_WRITE_INTERVAL_MS = 400;

/**
 * How many times a refused write is retried before giving up until the next change.
 *
 * A self-rescheduling catch retries forever, which turns a browser that is refusing writes into
 * a timer that never stops — including after the component unmounts. Two attempts covers the
 * case this exists for, a drag that ends on a refused write, and any change afterwards starts a
 * fresh budget anyway.
 */
const MAX_RETRIES = 2;

export function useUrlSync(): void {
  const config = useConfig();
  const replace = useConfig((s) => s.replace);

  const lastWriteAt = useRef(0);
  const pending = useRef<number | undefined>(undefined);
  const retriesLeft = useRef(MAX_RETRIES);

  /**
   * Whether the page was *opened* with a scenario in the URL.
   *
   * If it was, the address bar must keep carrying the whole scenario even when it happens to
   * equal the current defaults — otherwise opening a shared link to the default configuration
   * immediately erased it, and the recipient's bookmark of that address would resolve against
   * whatever defaults ship later. The sender said something explicit; the reader's address bar
   * should not quietly retract it.
   *
   * A bare URL stays bare, because it claimed nothing to begin with.
   */
  const arrivedExplicit = useRef(
    typeof window !== 'undefined' && window.location.search.length > 1
  );

  // Store -> URL.
  useEffect(() => {
    /**
     * Set by this effect's cleanup, and checked before every write.
     *
     * `clearTimeout` alone is not enough: a retry is scheduled from inside `write`, which runs
     * from a timer rather than from the effect, so a refusal firing in the same tick as teardown
     * can queue work nothing will ever clear. In the test environment that surfaced as a timer
     * running after jsdom had gone — `window is not defined`, from a suite where all 326 tests
     * passed. In a browser it is a write against a page that has already navigated away.
     *
     * A flag closed over per effect run makes escape harmless rather than merely unlikely.
     */
    let cancelled = false;

    /**
     * Reads current state rather than closing over this render's `config`, so a coalesced burst
     * writes the newest scenario instead of whichever frame happened to schedule the timer.
     */
    const write = () => {
      if (cancelled || typeof window === 'undefined') return;
      const current = useConfig.getState() as Config;
      /**
       * What this address already says without a query, which is what a bare query is allowed to
       * mean (#178).
       *
       * Read from `location` on every write rather than captured once, so it cannot go stale
       * against a path that changed — and so nothing here has to assume the path never does.
       *
       * The consequence when the scenario diverges from the path is deliberate: picking another
       * device on `/rtx-5090/` writes `/rtx-5090/?d=dgx-spark&…` and *keeps the path*. The
       * alternative — rewriting the path to match — is history churn for a URL that is already
       * unambiguous, because the query wins over the path wherever both speak.
       */
      const baseline = pathBaseline(window.location.pathname, import.meta.env.BASE_URL);
      const search = arrivedExplicit.current
        ? configToShareSearch(current)
        : configToSearch(current, baseline);
      /**
       * The fragment is carried through, not rebuilt.
       *
       * This effect owns the *query*, and `replaceState` takes a whole URL — so building one from
       * the pathname and the scenario alone silently dropped `#anything` on the first slider move.
       * The person who loses it is whoever arrived by an anchor link: their address bar, and the
       * history entry behind it, quietly stop pointing at the section they came for.
       *
       * Not hypothetical since the Matrix's click-to-scroll landed — `DETAIL_ANCHOR_ID` is a real
       * id on this page that someone can link to.
       *
       * The share link deliberately does *not* do this. It is constructed rather than preserved,
       * and it claims a scenario rather than a position; carrying the sender's scroll anchor into
       * someone else's link would be a different overclaim.
       */
      // The guard compares the one part this effect produces. Interpolating the fragment into both
      // sides would cancel out and read as though it participated — and would stop guarding the
      // moment `next` took its fragment from anywhere but the current location.
      if (search === window.location.search) return;
      const next = `${window.location.pathname}${search}${window.location.hash}`;

      try {
        window.history.replaceState(null, '', next);
        lastWriteAt.current = Date.now();
        retriesLeft.current = MAX_RETRIES;
      } catch {
        // A refused write is usually the rate limit, so back off as though it had succeeded and
        // try again after the interval. Without advancing the clock every subsequent change
        // retries immediately and stays refused; without the retry, a drag that ends on a
        // refused write leaves the address bar permanently stale.
        //
        // Bounded, because a catch that reschedules itself is a timer that never stops if the
        // browser keeps refusing. The app is unaffected either way — the URL is a convenience,
        // and what is on screen is the scenario.
        lastWriteAt.current = Date.now();
        if (retriesLeft.current > 0) {
          retriesLeft.current -= 1;
          window.clearTimeout(pending.current);
          pending.current = window.setTimeout(write, MIN_WRITE_INTERVAL_MS);
        }
      }
    };

    // Every change gets a fresh retry budget: this is a new scenario, not the old one failing.
    retriesLeft.current = MAX_RETRIES;

    const since = Date.now() - lastWriteAt.current;
    if (since >= MIN_WRITE_INTERVAL_MS) {
      write();
    } else {
      // Trailing edge, so the URL still ends up matching the screen even though the frames in
      // between are dropped.
      window.clearTimeout(pending.current);
      pending.current = window.setTimeout(write, MIN_WRITE_INTERVAL_MS - since);
    }

    // Returned on both paths, not just the deferred one: an immediate write that *fails* also
    // schedules a retry, and the early return left that chain running past unmount.
    return () => {
      cancelled = true;
      window.clearTimeout(pending.current);
    };
  }, [config]);

  // URL -> store.
  useEffect(() => {
    const onPop = () => {
      // Following a second link into the same page, or editing the URL by hand, is a fresh
      // arrival — so it re-establishes whether this address carries an explicit scenario.
      arrivedExplicit.current = window.location.search.length > 1;
      // The whole address, because a second link into this page can carry a path route as well as
      // a query — and the store has to read a `popstate` the same way it read the first arrival.
      const fromUrl = locationToConfig(
        window.location.pathname,
        window.location.search,
        import.meta.env.BASE_URL
      );
      if (!sameScenario(fromUrl, useConfig.getState() as Config)) replace(fromUrl);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [replace]);
}
