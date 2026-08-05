import type { PrerenderRoute } from '@/data/routes';

/**
 * A built page: Vite's shell, this route's metadata, and the rendered app inside `#root`.
 *
 * Separated from `scripts/prerender.ts` so it can be tested (#178). The script is I/O and caps —
 * read two build outputs, check two ceilings, write files — and none of that is the part that can
 * silently go wrong. **The part that can is this one**, and its failure mode is specific: a
 * substitution stops matching, every page ships with the root's title, and the site looks entirely
 * correct in a browser. A function that takes a shell and returns a document is checkable against
 * the real shell and the real render in a second, which a script that exits the process is not.
 *
 * It lives under `src/` rather than beside the script because that is where Vitest and the `@/`
 * alias already are, and it reaches the script through the SSR bundle `src/entry-server.tsx`
 * exports — the same bundle, so the pages that get composed and the pages that get rendered cannot
 * come from two different reads of the module graph.
 */

/** `&`, `<`, `>` and `"`, which is everything that can escape an attribute or a text node here. */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Substitute once, and refuse to carry on if the shell no longer looks the way this expects.
 *
 * A silent no-op here is the failure mode of every string-injecting prerenderer: `index.html`
 * gains a line, one replacement stops matching, and every page ships with the same title as the
 * root while looking entirely correct in a browser.
 *
 * Throws rather than exiting, so the build script can report it its own way and a test can watch
 * it happen. A `process.exit` inside the one function worth testing is a function nobody tests.
 */
function replaceOnce(html: string, pattern: RegExp, replacement: string, what: string): string {
  const matches = html.match(new RegExp(pattern, 'g'));
  if (matches?.length !== 1) {
    throw new Error(
      `expected exactly one ${what} in dist/index.html, found ${matches?.length ?? 0}. ` +
        'The shell changed shape; update the pattern in src/prerender/page.ts rather than ' +
        'shipping pages that all say the same thing.'
    );
  }
  return html.replace(pattern, () => replacement);
}

/**
 * The document written to `dist/<route>/index.html`.
 *
 * `origin` is the site's publishing origin and may be empty: unset, the canonical link is written
 * as a root-relative URL — valid, and resolved against the page's own address — while `og:url` is
 * omitted entirely, because Open Graph consumers want an absolute URL and a relative one tells
 * them nothing. Inventing `<owner>.github.io` instead would be the same quietly-wrong guess
 * `PAGES_BASE_PATH` exists to avoid.
 */
export function pageHtml(
  shell: string,
  route: PrerenderRoute,
  url: string,
  body: string,
  origin: string
): string {
  const canonical = origin ? `${origin}${url}` : url;
  const meta = [
    /**
     * A page that is not advertised says so itself, rather than relying on not being mentioned.
     *
     * `indexable` was written to keep pre-release hardware out of search results, and omitting a
     * URL from `sitemap.xml` does not do that — a sitemap is a discovery hint, not a directive, and
     * a search engine indexes what it finds by any route. One external link is enough. `noindex` is
     * the only thing that actually holds the line `routes.ts` claims to hold.
     *
     * `follow` rather than `nofollow`: these pages carry no internal route links at all, and their
     * outbound links are the catalog's own `source` URLs, which are worth following. And no
     * `robots.txt` `Disallow` to go with it — blocking the crawl stops the directive from ever
     * being read, which is the usual way this fix gets silently undone.
     *
     * The canonical below stays, self-referential. `noindex` applies to this page and there is no
     * consolidation target to leak it to; pointing the canonical elsewhere would both risk the
     * directive being attributed to that target and claim these figures are a duplicate of another
     * page's, which they are not. The Open Graph tags stay too — they drive link previews, not
     * search, and the decision here was that the page works for whoever lands on it.
     */
    ...(route.indexable ? [] : ['<meta name="robots" content="noindex, follow" />']),
    `<link rel="canonical" href="${escapeHtml(canonical)}" />`,
    '<meta property="og:type" content="website" />',
    `<meta property="og:title" content="${escapeHtml(route.title)}" />`,
    `<meta property="og:description" content="${escapeHtml(route.description)}" />`,
    ...(origin ? [`<meta property="og:url" content="${escapeHtml(canonical)}" />`] : []),
  ]
    .map((tag) => `    ${tag}`)
    .join('\n');

  let html = replaceOnce(
    shell,
    /<title>[\s\S]*?<\/title>/,
    `<title>${escapeHtml(route.title)}</title>`,
    'title'
  );
  html = replaceOnce(
    html,
    /<meta\s+name="description"[\s\S]*?\/>/,
    `<meta name="description" content="${escapeHtml(route.description)}" />`,
    'description meta'
  );
  html = replaceOnce(html, /<\/head>/, `${meta}\n  </head>`, 'closing head tag');
  /**
   * The marker `main.tsx` branches on, written here so it exists on exactly the pages that have
   * markup to hydrate. `404.html` keeps the bare shell and therefore keeps rendering from scratch.
   */
  return replaceOnce(
    html,
    /<div id="root"><\/div>/,
    `<div id="root" data-prerendered>${body}</div>`,
    'root container'
  );
}

/**
 * What a page has to contain to be worth publishing, and the words for what is missing.
 *
 * #178 exists because the deployed site was 860 bytes of empty shell, and the way that comes back
 * is not a deleted feature — it is a render that throws inside a boundary, a store seam that stops
 * injecting, or a substitution above that starts matching nothing. All three produce a file of the
 * right name and roughly the right shape, and none of them produces a figure. So every page is
 * checked before it is written, and a build that cannot produce figures fails rather than
 * publishing a shell that reads as a success.
 *
 * Regexes over the *rendered* text rather than over component names or tag structure, because the
 * question is what a reader of the bytes would find. They are deliberately loose about the numbers
 * and strict about the units: any figure will do, an absent one will not.
 *
 * **The verdict word is deliberately not among them, and that is a finding rather than an
 * omission.** The obvious check — `Fits|Tight|Will not run` — matches every page in the catalog,
 * and not because every page has a verdict: those words are also legend entries and workload
 * grades, so the alternation is satisfied by chrome. Taken one at a time none is universal either
 * (`Fits` 126 pages, `Tight` 182, `Will not run` 186), so no fixed spelling can be required. The
 * figures below cannot be produced without the engine having run, which is the property worth
 * asserting.
 *
 * **Two of the four accept "there is no speed to report", and that is not a loosening.** A model
 * that overflows the machine has no decode rate, and 19 of the 199 pages are exactly that case.
 * The page still has to *say so* — the sentence is rendered by the same tile that would otherwise
 * carry the number — so what is required is a statement either way, and silence is what fails.
 */
const FIGURES: readonly (readonly [string, RegExp])[] = [
  // "allocatable used" rather than "allocatable": the hardware note on a tunable-ceiling device
  // says "27 GiB allocatable by default", which passed the shorter pattern on a page whose whole
  // budget meter was missing.
  ['a memory budget', /[0-9.]+ [GTM]iB of [0-9.]+ [GTM]iB allocatable used/],
  [
    'a memory breakdown',
    /Weights [0-9.]+ [GTM]iB, KV cache [0-9.]+ [GTM]iB, Overhead [0-9.]+ [GTM]iB/,
  ],
  ['a prefill figure', /[0-9,]+ tok\/s prompt processing|no speed to report/],
  ['a decode figure', /tok\/s per user|no speed to report/],
];

/** The names of the figures a page does not contain — empty when it carries all four. */
export function missingFigures(html: string): readonly string[] {
  return FIGURES.filter(([, pattern]) => !pattern.test(html)).map(([name]) => name);
}
