/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';

// https://vite.dev/config/
export default defineConfig({
  /**
   * Where the built site will be served from.
   *
   * `/` is right for a custom domain, which is the plan — and wrong for the GitHub Pages project
   * URL the site lands on before one is attached, where everything sits under `/<repo>/`. Baking
   * either in makes the other deploy serve a blank page with four 404s in the console: a failure
   * that reads as a broken build and is not one.
   *
   * So it is an input, set by the deploy workflow from a repository variable, and `/` by default
   * because that is what `npm run dev`, `npm run preview` and the Playwright suite all assume.
   */
  base: process.env.BASE_PATH || '/',
  define: {
    /**
     * Where the built site is served *from*, as opposed to where in it the pages sit.
     *
     * The same shape of input as `base` above and set the same way — a repository variable read by
     * the deploy workflow — because it is the same class of setting: it describes the deployment
     * rather than the code, it cannot be inferred (`<owner>.github.io` is a guess that is wrong the
     * moment a custom domain is attached), and it fails quietly rather than loudly when it is
     * wrong. Prerendered pages need it for their canonical and `og:url` links, and the share button
     * needs it to write a URL without reading `window` during a render.
     *
     * Empty by default, and every consumer has to handle empty: `npm run dev`, `npm run preview`,
     * the Playwright suite and any fork all run without it, and a build that failed on an unset
     * publishing address would fail in all four.
     */
    __SITE_ORIGIN__: JSON.stringify(process.env.SITE_ORIGIN ?? ''),
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    /**
     * Raised from vitest's 5s default because the catalog got bigger, not because a test got slower.
     *
     * The Matrix is models x devices and `App.test.tsx` renders the whole app, so every one of its
     * 102 tests pays for the full grid. Going from 25 devices to 43 took that file from 42.8s to
     * 92.6s locally — the same tests, 2.2x the wall clock — and on the CI runner three of them
     * crossed 5s and failed the build on a change that touched no component.
     *
     * 20s rather than a smaller bump: the slowest test measured ~5s locally, and the runner is
     * slower than this machine by roughly the margin that broke it, so a 2x headroom over the
     * observed worst case is what keeps a green suite from depending on runner load. This is a
     * budget for an integration suite that renders ~700 buttons per test, and it is not a licence
     * to let a genuinely slow test sit — if a single test approaches this, the grid it renders is
     * the thing to question.
     *
     * **Raised a second time, on #77, and that is the signal rather than the fix.** The grid is
     * models x devices and both axes grew in one sweep: 408 cells before it, 714 after #78 added
     * eighteen devices, 1,470 after #77 doubled the model list — shipping rows, since the Matrix
     * never renders a rumoured one. The suite went 42s to 837s — 3.6x the cells for 20x the wall
     * clock, because `userEvent` slows superlinearly with tree size — and two PRs that touched no
     * component have now failed CI on this limit alone.
     *
     * **#101 removed the pressure on it, and deliberately did not lower it.** `App.test.tsx` renders
     * a bounded grid by default now and opts the two dozen tests that are genuinely *about* the grid
     * back into the real one: that file alone went 143s to 22s locally, the whole suite 155s to 24s,
     * and the slowest single test from ~5s to 2.2s. The tempting follow-up is to claw the limit back
     * down, and the arithmetic says not to. The runner is about 5.4x this machine — the same suite
     * that measures 155s here measured 837s there before this change — so a 2.2s test
     * is ~12s in CI and anything under about 25s is a coin toss on runner load. Lowering it would be
     * a fourth adjustment of a number whose three previous adjustments were all the wrong lever.
     *
     * The number to watch has never been this one: a fourteen-minute suite stops being run locally
     * long before any single test crosses a per-test limit. What #101 actually fixed is asserted
     * rather than hoped for — `App.test.tsx` renders a fixed twelve cells whatever the catalog does
     * next, so a change that touches no component cannot fail CI on grid size. This stays a safety
     * net. A test approaching it is a test rendering the full grid for a claim that does not need it.
     */
    testTimeout: 30_000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      // The engine is the part that has to be right, but the store and the formatters sit
      // between it and the screen: a wrong clamp or a mis-scaled unit reaches a user exactly as
      // a wrong roofline would, and neither was measured while this listed only the engine.
      include: ['src/engine/**', 'src/store/**', 'src/lib/**', 'src/data/**'],
    },
    /**
     * Playwright specs live in e2e/ and run via `npm run test:e2e`.
     *
     * `.claude/worktrees/**` is the other kind of exclusion: a full checkout of this repo per
     * background agent, each with its own copy of every spec. Without it a plain `vitest run` in the
     * parent collected 22 copies of `App.test.tsx` and reported 44 failures belonging to whatever
     * those agents were mid-way through writing — a red suite that says nothing about this checkout,
     * which is worse than a slow one.
     */
    exclude: ['**/node_modules/**', '**/dist/**', 'e2e/**', '**/.claude/worktrees/**'],
  },
});
