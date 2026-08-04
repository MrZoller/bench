import { StrictMode } from 'react';
import { renderToString } from 'react-dom/server';
import App from '@/App';
import { useConfig, type Config } from '@/store/config';
import { prerenderRoutes, routePath, type PrerenderRoute } from '@/data/routes';

/**
 * What `scripts/prerender.ts` loads: the app, rendered to a string, one route at a time.
 *
 * An entry of its own rather than pointing `vite build --ssr` at `src/App.tsx`, because the
 * script needs two things out of the bundle and a component module can only give it one. It needs
 * the store — the injection seam below is the whole mechanism — and importing `@/store/config`
 * from the script instead would load a *second* copy of a module-level singleton through a
 * different resolver, set the scenario on that one, and render the default from the first. The
 * failure would look exactly like the bug prerendering exists to avoid: every page identical.
 *
 * It is also what keeps the script free of the build: `@/` aliases, JSON imports and
 * `import.meta.env.BASE_URL` are all resolved by Vite here, against the same config and the same
 * `BASE_PATH` the client bundle was built with. A script that imported the source directly would
 * have to reproduce all three, and `BASE_URL` would simply be undefined.
 *
 * Re-exports the route list for the same reason: one bundle, so the pages that get rendered and
 * the pages that get written cannot come from two different reads of the catalog.
 */
export { prerenderRoutes, routePath };
export type { PrerenderRoute };

/**
 * Point the store's *server* snapshot at the scenario this render was just given.
 *
 * **The plan for #178 was wrong about this, and wrong in the way that looks right.** It proposed
 * `useConfig.getState().replace(config)` as a clean injection seam needing no source change, and
 * verified it by printing `getState()` after each render — which does change. The rendered bytes
 * do not. Zustand v5's `useStore` is
 *
 *     useSyncExternalStore(api.subscribe, () => selector(api.getState()),
 *                                         () => selector(api.getInitialState()))
 *
 * and React calls the *third* argument on the server, because that is the snapshot a hydrating
 * client will have to agree with. `getInitialState` returns the object the store was built from —
 * `coerce(readInitialConfig())`, evaluated at import time, which on a build machine is always
 * `DEFAULT_CONFIG`. So `replace` moved `getState` while every page rendered the default device,
 * and the four files differed only in the metadata the script injects: three device pages within
 * 45 bytes of each other, one set of figures, every one of them a DGX Spark. Page size is exactly
 * the check that cannot see this — which is why the verification for this slice is three devices
 * in three classes and a diff of the *figures*.
 *
 * Assigning `useConfig.getInitialState` does not work and is worth recording, because it is the
 * obvious first attempt and it fails silently: `create()` returns `Object.assign(hook, api)`, so
 * the hook carries a *copy* of the property while `useStore` reads `api.getInitialState` off the
 * object it closed over. The state object itself is the one thing both halves share, so that is
 * what this updates. It is plain and unfrozen, and nothing else holds a reference to it after the
 * first `setState` — zustand's `setState` builds a new object rather than mutating this one.
 *
 * On a server there is no state that arrived before this render, so making the initial snapshot
 * the current one is not a trick; it is what "initial" means with no client to have hydrated
 * from. It lives in this entry, which exists only inside the SSR bundle: in the browser the store
 * is read through `subscribe` and `getState`, and after hydration nothing consults this at all.
 *
 * If a future zustand stops working this way, `entry-server.test.tsx` fails on two device pages
 * rendering identical markup — the exact failure this file exists to prevent, asserted rather
 * than eyeballed.
 */
function adoptCurrentStateAsServerSnapshot(): void {
  Object.assign(useConfig.getInitialState(), useConfig.getState());
}

export function renderRoute(config: Partial<Config>): string {
  /**
   * The seam, and it must fire on every route including the first.
   *
   * The store is a module-level singleton, so one process rendering several routes carries the
   * previous route's scenario into the next one unless this replaces it every time — route 2
   * would render route 1's leftovers, silently and only in the built output. `replace` runs the
   * same `coerce` a hand-edited URL goes through, so a route is validated exactly as an arrival
   * is: the quant fallback, the context clamp and the shard check all apply.
   */
  useConfig.getState().replace(config);
  adoptCurrentStateAsServerSnapshot();

  // `StrictMode` renders no markup of its own; it is here so the tree the server writes is
  // spelled the same as the tree `main.tsx` hydrates over it.
  return renderToString(
    <StrictMode>
      <App />
    </StrictMode>
  );
}
