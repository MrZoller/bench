import { expect, test } from '@playwright/test';
import { marks } from '@/design/tokens';

/**
 * The hit-target rules, as they exist in the stylesheet that actually ships — issue #43.
 *
 * `pointer: coarse` describes the **primary** pointing device, so on a touchscreen laptop, a
 * Surface, or an iPad with a keyboard case it reports `fine`. Every coarse-pointer rule in the app
 * was gated on it, which meant a user who can still put a thumb on the screen got the mouse
 * layout — and the "show this as a table" toggles were back to 16px, on the very affordance #29
 * enlarged them for. `any-pointer: coarse` asks what the rule means: is *any* available pointer
 * coarse.
 *
 * **This asserts the CSS rather than the layout, and that is a limitation rather than a
 * preference.** A true hybrid cannot be emulated here. Playwright's `hasTouch` makes Chromium
 * report a touch-*only* device — `any-pointer: coarse` and `pointer: coarse` both true,
 * `any-pointer: fine` absent — which is a phone, not a laptop with a touchscreen. Driving
 * `Emulation.setEmulatedMedia` over CDP with explicit `pointer`/`any-pointer` features was tried
 * and is silently ignored; Chromium derives both from touch emulation. Both were measured rather
 * than assumed.
 *
 * So the behavioural assertion is unavailable and the honest thing is to say which claim is being
 * pinned: that the shipped stylesheet contains the right query, keyed to the right selector, at
 * the right size. That is falsifiable — reverting `any-pointer` to `pointer` fails it — and it
 * covers the failure mode an arbitrary Tailwind variant really has, which is compiling to nothing
 * at all.
 *
 * Runs on the desktop project because it reads the CSSOM and needs no emulation. The layout
 * assertions for a phone stay in `touch-targets.spec.ts`, where they can actually be measured.
 */

/** Every media query in the shipped stylesheet, with the rule text under it. */
async function mediaRules(page: import('@playwright/test').Page) {
  return page.evaluate(() => {
    const out: { condition: string; cssText: string }[] = [];
    for (const sheet of Array.from(document.styleSheets)) {
      let rules: CSSRule[];
      try {
        rules = Array.from(sheet.cssRules);
      } catch {
        continue; // cross-origin; none of ours are
      }
      /**
       * Descends into every grouping rule, not just media ones. Tailwind v4 emits its utilities
       * inside `@layer`, so a walk that only recursed through `CSSMediaRule` never reached them —
       * it found nothing, and every assertion downstream became a tautology over an empty array.
       * Which is why the precondition below counts what came back.
       */
      const walk = (list: CSSRule[]) => {
        for (const rule of list) {
          if (rule instanceof CSSMediaRule) {
            out.push({ condition: rule.conditionText, cssText: rule.cssText });
          }
          if (rule instanceof CSSGroupingRule) walk(Array.from(rule.cssRules));
        }
      };
      walk(rules);
    }
    return out;
  });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

/**
 * Guards the probe before anything is read through it: a stylesheet list that came back empty —
 * because the CSS is inlined differently, or `cssRules` threw — would make every assertion below
 * a tautology over an empty array.
 */
test('the shipped stylesheet is readable and carries pointer rules', async ({ page }) => {
  const rules = await mediaRules(page);

  expect(rules.length, 'no media rules were readable at all').toBeGreaterThan(0);
  expect(
    rules.filter((r) => r.condition.includes('pointer')).length,
    'no pointer media queries in the stylesheet'
  ).toBeGreaterThan(0);
});

/**
 * The fix. The toggle's min-height must sit under `any-pointer`, not `pointer`.
 *
 * `min-h-11` compiles to `calc(var(--spacing) * 11)` rather than a literal, so the assertion is on
 * the declaration's presence and the class it is keyed to; the resolved 44px is measured in
 * `touch-targets.spec.ts`, where a real coarse pointer exists.
 */
test('the disclosure toggle sizes on any available coarse pointer', async ({ page }) => {
  const rules = await mediaRules(page);

  const anyPointer = rules.filter((r) => /any-pointer\s*:\s*coarse/.test(r.condition));
  expect(anyPointer.length, 'no any-pointer: coarse query in the stylesheet').toBeGreaterThan(0);

  const sizing = anyPointer.filter((r) => /min-height/.test(r.cssText));
  expect(sizing.length, 'any-pointer: coarse carries no min-height rule').toBeGreaterThan(0);

  /**
   * The *selector*, not the condition.
   *
   * The obvious assertion — that the rule's `cssText` contains "any-pointer" — cannot fail: a
   * `CSSMediaRule`'s text begins with its own condition, and these rules were selected by matching
   * that condition. It was written, and it passed, and it would have gone on passing if the toggle
   * lost its variant class while any other element gained a `min-h-*` under the same query.
   *
   * Tailwind escapes the arbitrary variant into the class name, so the compiled selector contains
   * `min-h-11` — which is the toggle's own declaration and nothing else's.
   */
  expect(
    sizing.some((r) => /min-h-11/.test(r.cssText)),
    'no any-pointer rule is keyed to the disclosure toggle’s own class'
  ).toBe(true);
});

/**
 * And the grid deliberately stays on the narrower query — the other half of the decision.
 *
 * Widening it would give 44px rows to every laptop that happens to have a touchscreen, multiplied
 * across hundreds of cells, on a device being driven by a mouse. The toggles cost 28px once per
 * panel and are the accessibility affordance, so the trade goes the other way for them and not for
 * the grid. Asserted rather than left implicit, because "the grid is compact on a hybrid" is a
 * decision someone will otherwise read as an oversight and quietly fix.
 */
test('the Matrix grid keeps the primary-pointer query, which is the trade', async ({ page }) => {
  const rules = await mediaRules(page);

  const primaryOnly = rules.filter(
    (r) => /(^|[^-])pointer\s*:\s*coarse/.test(r.condition) && !/any-pointer/.test(r.condition)
  );
  expect(primaryOnly.length, 'the primary-pointer query is gone entirely').toBeGreaterThan(0);

  // The grid's rules size cells and columns. Matched on the utility class names rather than on
  // the declarations, because `/\bheight/` also matches `min-height` — `-` is a non-word
  // character, so the boundary sits happily in the middle of the property — and an earlier draft
  // documented an exclusion its own regex did not perform.
  const grid = primaryOnly.filter((r) => /:(w|h|min-w)-\d/.test(r.cssText));
  expect(grid.length, 'nothing sizes the grid under pointer: coarse').toBeGreaterThan(0);
});

/**
 * The two queries must not have collapsed into one.
 *
 * If a later edit moved everything to `any-pointer`, every assertion above still passes — the
 * `any-pointer` ones trivially, and the grid one because `any-pointer: coarse` contains the
 * substring the regex excludes only by luck. This is the assertion that the *asymmetry* survived,
 * which is the actual content of #43's resolution.
 */
test('the two queries stayed distinct', async ({ page }) => {
  const conditions = (await mediaRules(page)).map((r) => r.condition);

  expect(conditions.some((c) => /any-pointer\s*:\s*coarse/.test(c))).toBe(true);
  expect(
    conditions.some((c) => /(^|[^-])pointer\s*:\s*coarse/.test(c) && !/any-pointer/.test(c)),
    'both rules now use the same query, so the asymmetry is gone'
  ).toBe(true);
  // Sanity: the token the toggle is sized against is still what the other spec measures.
  expect(marks.hitTarget).toBe(44);
});
