/**
 * Absolute links to this site, built without reading `window` during a render.
 *
 * Two call sites want one — the masthead's share button and the calibration panel's pre-filled
 * issue — and both built it from `window.location.origin` in a render body. That is a hard
 * blocker for prerendering ([#178](https://github.com/MrZoller/headroom/issues/178)): `window`
 * does not exist under `renderToString`, so the render threw before it reached a single figure.
 * Neither call site could defer to an effect without losing the property its docblock argues for
 * — the link is *derived* on every render so it can never describe a scenario the user has since
 * moved off — so the fix is to stop the value depending on the browser at all.
 *
 * **The origin is a build-time constant, so the server and the browser agree by construction.**
 * `SITE_ORIGIN` is fed from the `PAGES_SITE_ORIGIN` repository variable through `vite.config.ts`,
 * the same route and the same class of setting as `PAGES_BASE_PATH`: it says where the site is
 * served, it cannot be inferred from the code, and it fails quietly when wrong. The same value
 * reaches `scripts/prerender.ts` from `process.env.SITE_ORIGIN`, so a page's canonical link and
 * the link its share button copies name the same origin.
 *
 * **Unset, a build still succeeds**, because a repository that has not set the variable — a fork,
 * a pull-request build, the CI `build` job — must not fail on a setting that only matters once
 * something is published. The client falls back to the origin it is actually being served from,
 * which is right for `npm run dev` and for any deploy nobody has configured yet; the prerenderer
 * has no such fallback and omits the absolute metadata rather than inventing an origin.
 *
 * That fallback is the one place the two halves can disagree, and it is safe for a stated reason
 * rather than by luck: **neither call site puts this string in the initial markup.** The share
 * button starts `idle` and only renders the URL into a field once the clipboard API has refused
 * it, and the calibration link appears only once a paste has produced a comparable row. If a
 * third call site ever renders one of these on first paint, it has to use the constant on both
 * sides — under an unset variable this value differs between the prerendered HTML and the
 * hydrating client, and React would discard the page.
 */

const BUILD_ORIGIN = __SITE_ORIGIN__;

/**
 * The origin this site's absolute links are written against.
 *
 * Resolved once at module load. `import.meta.env.SSR` is a build-time boolean, so the fallback
 * branch is not merely unreached in the SSR bundle — it is not in it.
 */
export const SITE_ORIGIN: string =
  BUILD_ORIGIN ||
  (import.meta.env.SSR || typeof window === 'undefined' ? '' : window.location.origin);

/**
 * A link to a scenario, from the querystring that encodes it.
 *
 * Written against the site's base rather than the current pathname. A path route is a lossy entry
 * point — `/rtx-5090/` names one of the nine fields — while the querystring names all nine, so
 * the base plus the full encoding is the shortest URL that reproduces exactly what the sender was
 * looking at. Carrying the pathname too would add a device id that the query then overrides.
 */
export function scenarioLink(search: string): string {
  return `${SITE_ORIGIN}${import.meta.env.BASE_URL}${search}`;
}
