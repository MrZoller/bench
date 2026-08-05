import { describe, expect, it, vi } from 'vitest';

/**
 * That a built page carries the figures — the regression guard #178 asks for by name.
 *
 * **Without this, a future change reverts the site to a shell and nothing says so.** That is not a
 * hypothetical class of bug, it is the state this issue was opened about: 860 bytes of
 * `<div id="root"></div>` served to every crawler while the app looked perfect in a browser. The
 * ways back are all silent — a render that throws inside a boundary, a store seam that stops
 * injecting, a substitution that stops matching — and every one of them writes a file of the right
 * name and roughly the right shape.
 *
 * **It composes the page rather than reading `dist/`, and that is deliberate.** CI runs
 * `npm test` *before* `npm run build`, so a test that read the built output would find nothing on a
 * fresh checkout — and the two ways around that are worse than composing here: skipping when
 * `dist/` is absent is a guard that silently does not run in the one place it is meant to, and
 * asserting against whatever `dist/` happens to hold makes a green suite depend on a build that may
 * be hours old. What is composed is the real thing on both sides: the shell is `index.html` as
 * committed, and the body is `renderRoute` over the real catalog and the real engine. `vite build`
 * changes the shell only by rewriting the script and stylesheet tags, and a shell that changed
 * shape enough to break a substitution fails the build loudly rather than passing here — the last
 * case below is that assertion.
 *
 * The browser half is `e2e/prerendered.spec.ts`, which loads a genuinely built page with
 * JavaScript disabled. Neither covers the other: this one passes on a page whose markup a browser
 * would never paint, and that one passes on a page composed from a shell nobody checked.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

import { boundGridByDefault } from '@/test/grid';
import { prerenderRoutes, renderRoute } from '@/entry-server';
import { missingFigures, pageHtml } from './page';
import type { PrerenderRoute } from '@/data/routes';

boundGridByDefault();

/**
 * The shell as committed — the file `vite build` rewrites the asset tags of and nothing else.
 *
 * Imported through Vite's `?raw` rather than read from disk, so the path is resolved by the same
 * bundler the build uses instead of against whatever directory the test runner happened to start
 * in.
 */
import SHELL from '../../index.html?raw';

const ORIGIN = 'https://mrzoller.github.io';

function build(route: PrerenderRoute, origin = ORIGIN): string {
  const url = `/headroom/${route.segments.map((segment) => `${segment}/`).join('')}`;
  return pageHtml(SHELL, route, url, renderRoute(route.config), origin);
}

/**
 * A device page, which every catalogued device has and no ranking can take away.
 *
 * The exact figures below are pinned to one, deliberately rather than to a pair page: the pair
 * tier is the ten most-downloaded models, so a week in which a model slips to eleventh would break
 * a test about prerendering with an error about popularity. A device route is `devices.json` plus
 * `DEFAULT_CONFIG`, and both are edited by hand.
 */
function deviceRoute(deviceId: string): PrerenderRoute {
  const route = prerenderRoutes().find(
    (candidate) => candidate.tier === 1 && candidate.config.deviceId === deviceId
  );
  if (!route) throw new Error(`${deviceId} has no prerendered device page`);
  return route;
}

describe('a built page', () => {
  /**
   * The assertion the issue asks for, and the reason it names figures rather than headings.
   *
   * Every number here was computed by `src/engine/` for this exact scenario — an RTX 5090's 31 GiB
   * allocatable ceiling against the default model's 61 GiB of weights, the cache the default
   * 32K x 1 usage sizes, the overage that follows, and the two rates the roofline produces once
   * the spill is charged to the host bus. A heading assertion passes on a page with no engine in
   * it at all; these cannot.
   *
   * They are exact, and a catalog refresh that moves a parameter count will break them. That is
   * the intended direction: correct the expected value, and never widen the pattern until it stops
   * being able to fail.
   */
  it('carries the figures the engine computed for its scenario', () => {
    const html = build(deviceRoute('rtx-5090'));

    expect(html).toContain(
      'aria-label="63 GiB of 31 GiB allocatable used. ' +
        'Weights 61 GiB, KV cache 1.1 GiB, Overhead 0.6 GiB. Over budget."'
    );
    expect(html).toContain('3678 tok/s prompt processing');
    expect(html).toContain('>23<span class="ml-1 text-sm text-[var(--color-text-faint)]">tok/s');
    expect(html).toContain('>2.2 s<');
    expect(html).toContain(
      '>32 GiB<span class="ml-1 text-sm text-[var(--color-text-faint)]">offloaded'
    );
  });

  /**
   * The other half of the same claim: that the figures belong to *this* page.
   *
   * A prerenderer that injects the same scenario every time produces 199 individually plausible
   * files, all of them one machine's numbers — which is what the first working build of this
   * mechanism actually did. Two devices, two ceilings, in one process.
   */
  it('gives two devices two different sets of figures', () => {
    const ceiling = (html: string) => /([0-9.]+ [GTM]iB) allocatable used/.exec(html)?.[1];

    expect(ceiling(build(deviceRoute('rtx-5090')))).toBe('31 GiB');
    expect(ceiling(build(deviceRoute('epyc-9654')))).toBe('720 GiB');
  });

  /**
   * A pair page, taken from the tier rather than named, since which pairs exist is a question the
   * catalog answers weekly. What is asserted is what a pair page has to be: both halves in the
   * metadata, and this scenario's figures rather than either half's default.
   */
  it('carries both halves of a pair, and the figures for the two together', () => {
    const pair = prerenderRoutes().find((route) => route.tier === 3)!;
    const html = build(pair);

    expect(html).toContain(`<title>${pair.title}</title>`);
    expect(missingFigures(html)).toEqual([]);
    expect(html).not.toBe(build(deviceRoute(pair.config.deviceId!)));
  });

  it('takes its title, description and canonical from its route', () => {
    const route = prerenderRoutes().find((candidate) => candidate.tier === 3)!;
    const html = build(route);

    expect(html).toContain(`<title>${route.title}</title>`);
    expect(html).toContain(`<meta name="description" content="${route.description}" />`);
    expect(html).toContain(
      `<link rel="canonical" href="${ORIGIN}/headroom/${route.segments.join('/')}/" />`
    );
    expect(html).toContain(
      `<meta property="og:url" content="${ORIGIN}/headroom/${route.segments.join('/')}/" />`
    );
    // The marker `main.tsx` branches on. Without it a prerendered page renders from scratch and
    // every byte above is thrown away on arrival.
    expect(html).toContain('<div id="root" data-prerendered>');
  });

  it('writes a relative canonical and no og:url when the origin is unset', () => {
    // A fork, a pull-request build, or a repository where nobody has set `PAGES_SITE_ORIGIN`.
    // Inventing `<owner>.github.io` would be the guess that variable exists to stop.
    const html = build(prerenderRoutes()[0], '');
    expect(html).toContain('<link rel="canonical" href="/headroom/" />');
    expect(html).not.toContain('og:url');
  });

  it('escapes what it injects', () => {
    const html = build({
      ...prerenderRoutes()[0],
      title: 'A & B "C" <script>',
      description: 'x & y',
    });
    expect(html).toContain('<title>A &amp; B &quot;C&quot; &lt;script&gt;</title>');
    expect(html).toContain('content="x &amp; y"');
    expect(html).not.toContain('<script>A');
  });
});

describe('missingFigures', () => {
  /**
   * The polarity check, so this is not a guard nobody has seen fail.
   *
   * The shell *is* the regression: it is what the site served before #178 and what `404.html` is
   * on purpose. Every figure has to be reported missing from it, and none from a real page.
   */
  it('reports every figure missing from the bare shell', () => {
    expect(missingFigures(SHELL)).toEqual([
      'a memory budget',
      'a memory breakdown',
      'a prefill figure',
      'a decode figure',
    ]);
  });

  it('reports nothing missing from any page the build writes', () => {
    for (const route of prerenderRoutes()) {
      expect(missingFigures(renderRoute(route.config)), routePathOf(route)).toEqual([]);
    }
  });

  /**
   * The 19 pages where there is no speed to report, which is a real answer rather than a gap: the
   * model overflows the machine, so the decode tile prints why instead of a rate. The guard
   * accepts that sentence and nothing else — silence still fails.
   */
  it('accepts a stated reason in place of a rate, and not an absence', () => {
    const stated =
      'X GiB of Y GiB allocatable used. Weights 1 GiB, KV cache 1 GiB, Overhead 1 GiB.'.replace(
        /X|Y/g,
        '1'
      ) + ' No estimate — the model does not fit, so there is no speed to report.';
    expect(missingFigures(stated)).toEqual([]);
    expect(missingFigures(stated.replace('no speed to report', 'nothing'))).toEqual([
      'a prefill figure',
      'a decode figure',
    ]);
  });
});

describe('the shell', () => {
  /**
   * The failure mode of every string-injecting prerenderer: `index.html` gains a line, one
   * substitution stops matching, and every page ships with the root's title while looking entirely
   * correct in a browser. It has to be an exception rather than a no-op, and the message has to
   * name what it could not find.
   */
  it('is refused rather than silently half-injected when it changes shape', () => {
    const route = prerenderRoutes()[0];
    const body = renderRoute(route.config);

    expect(() => pageHtml(SHELL.replace('<title>', '<title >'), route, '/', body, '')).toThrow(
      /exactly one title/
    );
    expect(() =>
      pageHtml(
        SHELL.replace('<div id="root"></div>', '<div id="root"> </div>'),
        route,
        '/',
        body,
        ''
      )
    ).toThrow(/exactly one root container/);
    expect(() => pageHtml(`${SHELL}${SHELL}`, route, '/', body, '')).toThrow(/found 2/);
  });
});

function routePathOf(route: PrerenderRoute): string {
  return `/${route.segments.join('/')}/`;
}
