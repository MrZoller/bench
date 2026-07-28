import { defineConfig, devices } from '@playwright/test';

/**
 * Browser-level coverage, for the things jsdom structurally cannot see.
 *
 * The unit suite runs in jsdom, which has no layout engine and no `scrollIntoView` at all. That is
 * not a gap in what the tests happen to assert — it is a gap in what they *can* assert, and it has
 * already shipped one real bug: the Matrix's click-to-scroll was first anchored on a
 * `display: contents` element, which generates no principal box, so `scrollIntoView` returned early
 * and the scroll was a silent no-op in every real browser. jsdom's missing method meant the guarded
 * call passed every test. It was caught in review.
 *
 * So the rule for what belongs here is narrow: a spec earns its place only if jsdom cannot answer
 * the question. Geometry, scrolling, media queries that depend on a real pointer, and canvas
 * actually painting. Everything else stays in Vitest, where it runs in a second.
 *
 * Served from a production build rather than the dev server, because the question these ask is
 * whether the thing users get works — and because Tailwind's generated stylesheet is the subject of
 * half of them.
 */
export default defineConfig({
  testDir: './e2e',
  /**
   * Named explicitly rather than left to discovery. Playwright takes the nearest `tsconfig.json`,
   * and this repo's root file is a solution-style one — project references, no `compilerOptions`
   * — so the `@/*` mapping the specs import through lives in `tsconfig.e2e.json`.
   *
   * Resolution does work without this: Playwright follows project references, and the pinned
   * 1.62 does so. But the dependency range is `^1.60.0`, and relying on a transitive lookup that
   * a minor bump could tighten is a needless bet when naming the file costs one line.
   */
  tsconfig: './tsconfig.e2e.json',
  // Playwright's own default is 50% of cores in both environments; this only says that the specs
  // are independent, which they are — each drives a fresh page from a fresh store.
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : [['list']],

  use: {
    baseURL: 'http://127.0.0.1:4173',
    // On first retry only, so a green run stays fast and a flake arrives with evidence.
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'desktop',
      use: { ...devices['Desktop Chrome'] },
      // The touch specs assert the coarse-pointer branch, which a mouse run cannot reach. Without
      // this they ran here too and failed on the branch they are not about — `testMatch` on a
      // sibling project narrows what *it* takes, not what everyone else leaves alone.
      testIgnore: /(touch-targets|reflow)\.spec\.ts/,
    },
    {
      /**
       * Text-only zoom at 200%, performed rather than modelled.
       *
       * `--blink-settings=defaultFontSize=32` changes the browser's *default* font size, which is
       * what a reader actually changes in Firefox's font preference or Chrome's appearance
       * settings. That matters beyond the root size: `rem` inside a media query resolves against
       * the browser default and ignores an author-set `documentElement.style.fontSize`, so
       * simulating zoom by setting the root leaves Tailwind's rem-based breakpoints where they
       * were — and a spec written that way puts the page into layout states no reader can reach.
       * It did: 640px with a scaled root reported three columns crushed into 213px each, a state
       * that only exists because the breakpoint did not move with the text.
       *
       * With the switch, it does: `sm` (40rem) becomes 1280px and `lg` 2048px, both verified.
       * `reflow.spec.ts` asserts that before measuring anything, so if the switch ever stops
       * working the suite fails loudly instead of quietly re-testing at 100%.
       *
       * A project rather than a launch argument on `desktop`, because every other spec wants a
       * normal browser — and because the flag is a Blink-internal switch with no stability
       * promise, so its blast radius is worth confining to the one file that needs it.
       */
      name: 'reflow',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: { args: ['--blink-settings=defaultFontSize=32'] },
      },
      testMatch: /reflow\.spec\.ts/,
    },
    {
      /**
       * A real coarse pointer, which is the whole point: `@media (pointer: coarse)` cannot be
       * forced with a viewport size, and the 44px hit targets it gates are invisible to any test
       * that does not emulate a touch device. The specs assert the media query matches before
       * asserting anything about size, so a change in how Playwright emulates this fails loudly
       * rather than silently measuring the mouse branch.
       */
      name: 'touch',
      use: { ...devices['Pixel 5'] },
      testMatch: /touch-targets\.spec\.ts/,
    },
    {
      /**
       * The canvas specs again, at a device pixel ratio that is not 1.
       *
       * `Desktop Chrome` has a `deviceScaleFactor` of 1, and that makes the bitmap assertions
       * vacuous in the direction they most need to hold: `cssSize * dpr` is `cssSize * 1`, so
       * dropping the dpr multiplication from the sizing effect entirely would still satisfy
       * every one of them, and only a retina user would see the resulting half-resolution plot.
       * The stretch this guards is the same defect the height assertion was added for — it just
       * hides on this project rather than in the DOM.
       *
       * A separate project rather than raising `desktop`'s scale factor, so the sizing is checked
       * at both ratios and the other specs keep laying out at the size they were written for.
       */
      name: 'retina',
      use: { ...devices['Desktop Chrome'], deviceScaleFactor: 2 },
      testMatch: /canvases\.spec\.ts/,
    },
  ],

  webServer: {
    // `--host 127.0.0.1` rather than vite's default: `localhost` resolves to `::1` on a machine
    // with IPv6, so the server listens somewhere the `url` probe below never looks and the run
    // dies on a webServer timeout that says nothing about why.
    command: 'npm run build && npm run preview -- --port 4173 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:4173',
    /**
     * Never reused, not even locally. The usual `!process.env.CI` reuses whatever already answers
     * on 4173 — and because this command *builds* before it serves, reuse skips the build too. An
     * earlier `vite preview` left running over a stale `dist` then gets tested in place of the
     * current checkout, silently, which defeats the reason these specs are served from a
     * production build at all. It is not hypothetical: it happened during this PR.
     *
     * With reuse off, `--strictPort` turns that same occupied port into an immediate failure
     * naming the conflict, which is the outcome worth having. Cost is one build per local run.
     */
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
