import { expect, test, type Page } from '@playwright/test';

/**
 * Reflow at 200% text-only zoom — WCAG 1.4.4 (Resize Text). Issue #35.
 *
 * **This is a different test from 1.4.10 (Reflow)**, and conflating them is why the defect
 * survived a responsive pass. 1.4.10 asks whether a 320px viewport works at the *default* text
 * size, which this app already satisfied. 1.4.4 asks whether text can be scaled to 200% without
 * loss of content or function — and browser text-only zoom does that by growing the root font
 * size while leaving the viewport where it is. So every `rem`-derived width grows, every
 * `whitespace-nowrap` line grows, and nothing gives them more room. A layout can pass one and fail
 * the other badly; this one did, by 89px.
 *
 * Why it belongs here rather than in Vitest: jsdom has no layout engine, so `scrollWidth` and
 * `clientWidth` are both 0 and every assertion below is a tautology there.
 *
 * **The two floors a narrow viewport cannot negotiate with**, both of which this found:
 *
 *   - `whitespace-nowrap` on a whole sentence. The four panel headers each carried one, and the
 *     `flex-wrap` on the header above only decides which line the sentence lands on, not how wide
 *     it is. The Matrix's took the document to 409/320 on its own.
 *   - A non-wrapping flex row of options. Its min-content is the *sum* of its children, and
 *     because that row sets the width of its grid column, every `w-full` slider sharing the column
 *     inherited it — four KV options widening three sliders in a panel they are not even part of.
 *
 * Neither is visible at the default root size, which is why this scales the root rather than
 * shrinking the viewport.
 */

/** The narrowest width anything still ships at, matching `matrix-legend.spec.ts`. */
const NARROW = { width: 320, height: 900 };

/**
 * 32px, because the browser default is 16px and this is a 200% test.
 *
 * Stated as a doubling of the default rather than as an absolute, since that is the thing the
 * success criterion actually asks for. Past this the page is *not* clean — at a 40px root
 * ("250%") long single words like "Unsupported" and the slider labels start escaping. That is
 * filed rather than fixed: 1.4.4 stops at 200%, and the honest statement is that the bar is met
 * rather than that the layout is unbreakable.
 */
const ROOT_200 = 32;

/**
 * Scenarios that put materially different content on the page, because the offending line is
 * built from the numbers in it. The default page and MLX/Q5_K_M are the two the issue was
 * measured at; the rest vary count width, runtime label and the conditional panels.
 */
const SCENARIOS = [
  { name: 'the default page', url: '/', heading: /32K context, 8K prompt, 1 user, FP16 KV/ },
  {
    name: 'MLX at Q5_K_M, where the Matrix legend is fullest',
    url: '/?r=mlx&q=q5_k_m',
    heading: /at Q5_K_M.*MLX \(Apple\)/,
  },
  {
    name: 'vLLM at 128K over eight users',
    url: '/?r=vllm&q=fp8&ctx=131072&u=8',
    heading: /vLLM — 128K context, .*8 users/,
  },
  {
    name: 'llama.cpp at 128 users on eight devices',
    url: '/?r=llama_cpp&q=q4_k_m&ctx=131072&u=128&n=8',
    heading: /128K context, .*128 users.*one device per cell/,
  },
] as const;

/**
 * That the querystring actually reached the store.
 *
 * `url.ts` reads only the keys in its own map and ignores everything else in silence, so a wrong
 * key is not an error — it is a page that loads at the defaults while the test name goes on
 * claiming 128K. The first draft said `c=131072`, where the key is `ctx`, and both long-context
 * scenarios were quietly measuring the default page.
 *
 * **The obvious guard for that is itself vacuous**, which is worth recording because it was
 * written and it passed: reading `searchParams.get('ctx')` and skipping when absent means the
 * wrong key produces no expectation at all, so restoring the bug leaves every test green. That is
 * why the expected heading is spelled out per scenario above instead of derived from the URL — an
 * expectation that comes from the same string it is checking cannot contradict it.
 *
 * Asserted through the Matrix heading because it names every input that moves the grid, and it is
 * rendered from the same config the layout is.
 */
async function assertScenarioLoaded(page: Page, expected: RegExp) {
  await expect(
    page.getByRole('heading', { name: /every model on every machine/i }),
    'the querystring did not reach the page'
  ).toContainText(expected);
}

async function scaleRoot(page: Page, px: number) {
  await page.evaluate((v) => {
    document.documentElement.style.fontSize = `${v}px`;
  }, px);
  // The scaled root relayouts everything; wait for it to settle before measuring.
  await page.waitForFunction(
    (v) => getComputedStyle(document.documentElement).fontSize === `${v}px`,
    px
  );
}

/**
 * Every element whose right edge escapes the document, ignoring anything inside a deliberate
 * scroll container.
 *
 * The exclusion is the point: the Matrix grid and the Envelope canvas are *supposed* to scroll
 * horizontally, inside a box of their own. What 1.4.4 forbids is the overflow reaching the
 * document, so the walk stops descending at the first `overflow-x` that is not `visible` — which
 * is also what keeps a failure naming the element that actually needs fixing rather than the
 * thousand table cells inside a scroller that is behaving correctly.
 */
async function escapees(page: Page) {
  return page.evaluate(() => {
    const docWidth = document.documentElement.clientWidth;
    const out: string[] = [];
    const walk = (el: HTMLElement) => {
      if (getComputedStyle(el).overflowX !== 'visible') return;
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.right > docWidth + 1) {
        const cls = (el.className?.toString() ?? '').slice(0, 70);
        out.push(
          `${Math.round(rect.right)}px <${el.tagName.toLowerCase()}> ` +
            `"${(el.textContent ?? '').trim().slice(0, 45)}" .${cls}`
        );
      }
      for (const child of Array.from(el.children)) walk(child as HTMLElement);
    };
    walk(document.body);
    return out;
  });
}

test.describe('at 200% text size on a 320px viewport', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize(NARROW);
  });

  /**
   * The precondition, asserted before any width is.
   *
   * Same reasoning as the touch project asserting `(pointer: coarse)` matches before measuring a
   * hit target: if the root never scaled, every assertion below runs at the default text size,
   * where the defect does not exist and all of them pass. A test that cannot fail is worse than
   * no test, and this suite has already produced three.
   */
  test('the probe actually scales the root font size', async ({ page }) => {
    await page.goto('/');
    const before = await page.evaluate(() =>
      parseFloat(getComputedStyle(document.querySelector('h1')!).fontSize)
    );

    await scaleRoot(page, ROOT_200);

    const after = await page.evaluate(() =>
      parseFloat(getComputedStyle(document.querySelector('h1')!).fontSize)
    );
    expect(before, 'the page does not start at the 16px default root').toBeGreaterThan(0);
    expect(after / before, 'scaling the root did not scale rem-derived text').toBeCloseTo(2, 1);
  });

  for (const { name, url, heading } of SCENARIOS) {
    test(`the page does not scroll sideways on ${name}`, async ({ page }) => {
      await page.goto(url);
      await assertScenarioLoaded(page, heading);
      await scaleRoot(page, ROOT_200);

      // Named before the numeric assertion, so a failure says which element to fix rather than
      // only by how much the document is too wide.
      expect(await escapees(page), 'elements escaping the document').toEqual([]);

      const doc = await page.evaluate(() => ({
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      expect(doc.scrollWidth, 'the page scrolls sideways at 200% text').toBeLessThanOrEqual(
        doc.clientWidth + 1
      );
    });
  }

  /**
   * The segmented control's own fix, asserted directly.
   *
   * The document-width test above passes if this row is fixed *or* if something else absorbs it,
   * so it does not pin the mechanism. This does: at 200% the four KV options cannot share a line,
   * so a wrapping row puts them on more than one. Remove `flex-wrap` and the options return to a
   * single line whose min-content is their sum — which is the state that widened the grid column
   * and dragged three sliders out of the viewport with it.
   */
  test('the segmented options wrap rather than summing their widths', async ({ page }) => {
    await page.goto('/');
    await scaleRoot(page, ROOT_200);

    const kv = page.getByRole('group', { name: /kv precision/i });
    await expect(kv).toBeVisible();

    const lines = await kv.locator('label').evaluateAll((labels) => {
      const tops = labels.map((l) => Math.round(l.getBoundingClientRect().top));
      return { count: labels.length, rows: new Set(tops).size };
    });

    expect(lines.count, 'the KV control lost its options').toBeGreaterThan(2);
    expect(lines.rows, 'the options are all on one line, so the row still sums').toBeGreaterThan(1);
  });

  /**
   * And that wrapping did not fix the overflow by breaking the thing worth protecting.
   *
   * "12 of 425" split across a line reads as two unrelated numbers, so `PanelCount` keeps a
   * `whitespace-nowrap` on the numeral pair alone while letting the noun after it wrap.
   *
   * Asserted structurally rather than by geometry, and deliberately so: the pair is short enough
   * that it never breaks on its own at any root size from 32px to 64px — measured, not assumed —
   * so `getClientRects().length === 1` is true with the class and true without it. That assertion
   * would pass against markup with the protection deleted, which is the failure mode this suite
   * keeps producing. The computed style is the honest thing to check here.
   */
  test('the numeral pair keeps its own nowrap', async ({ page }) => {
    await page.goto('/');
    await scaleRoot(page, ROOT_200);

    const pair = page
      .getByRole('region', { name: /every model on every machine/i })
      .locator('p', { hasText: /combinations run/ })
      .locator('span.whitespace-nowrap')
      .first();

    await expect(pair).toHaveText(/^\d+ of \d+$/);
    await expect(pair).toHaveCSS('white-space', 'nowrap');
    // The sentence around it must *not* be protected, or the fix is undone one level up.
    const parent = page
      .getByRole('region', { name: /every model on every machine/i })
      .locator('p', { hasText: /combinations run/ });
    await expect(parent).not.toHaveCSS('white-space', 'nowrap');
  });
});
