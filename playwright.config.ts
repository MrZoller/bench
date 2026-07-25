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
      // this they ran here too and failed on the branch they are not about — `testMatch` on the
      // other project narrows what *it* takes, not what everyone else leaves alone.
      testIgnore: /touch-targets\.spec\.ts/,
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
  ],

  webServer: {
    // `--host 127.0.0.1` rather than vite's default: `localhost` resolves to `::1` on a machine
    // with IPv6, so the server listens somewhere the `url` probe below never looks and the run
    // dies on a webServer timeout that says nothing about why.
    command: 'npm run build && npm run preview -- --port 4173 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
